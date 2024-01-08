// Computes the induced block function and induced matching 
// This uses the permovec package

#include <sstream>
#include <fstream>

// GUDHI modules
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/Flag_complex_edge_collapser.h>
#include <gudhi/reader_utils.h>
#include <gudhi/Points_off_io.h>
#include <gudhi/distance_functions.h>
#include <gudhi/graph_simplicial_complex.h>
// PHAT modules needed
#include <phat/compute_persistence_pairs.h>
#include <phat/boundary_matrix.h>
#include <phat/representations/default_representations.h>
#include <phat/helpers/misc.h>
#include <phat/algorithms/standard_reduction.h>
#include <phat/algorithms/twist_reduction.h>
// BOOST necessary modules
#include <boost/program_options.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "permovec/permovec.h"

#include "sorting.hh"


// GUDHI types
using Simplex_tree_options = Gudhi::Simplex_tree_options_full_featured;
using Simplex_tree = Gudhi::Simplex_tree<Simplex_tree_options>;
using Simplex_key = Simplex_tree_options::Simplex_key;
using Filtration_value = Simplex_tree::Filtration_value;
using Vertex_handle = Simplex_tree::Vertex_handle;
using Simplex_handle = Simplex_tree::Simplex_handle;
using Proximity_graph = Gudhi::Proximity_graph<Simplex_tree>;
using Point = std::vector<double>;
using Points_off_reader = Gudhi::Points_off_reader<Point>;
// Some handy additional types
using Distance_matrix = std::vector<std::vector<Filtration_value>>;
using Filtered_edge = std::tuple<Vertex_handle, Vertex_handle, Filtration_value>;
using Vertex_pair = std::pair<Vertex_handle, Vertex_handle>;
// PHAT types
using Phat_boundary_matrix = phat::boundary_matrix<phat::vector_vector>;
using Phat_index = phat::index;
using Phat_column = std::vector<Phat_index>;
using Phat_idx_pair = std::pair<Phat_index, Phat_index>;
// Custom Relevant Types
using Interval = std::pair<Filtration_value, Filtration_value>;
using Barcodes_dim = std::unordered_map<int, std::vector<Interval>>;
using Matrix_dim = std::unordered_map<int, std::vector<std::vector<Phat_index>>>;
using Distance_matrix = std::vector<std::vector<Filtration_value>>;


void program_options(
	int argc, char* argv[], std::string& file_S_dist, std::string& file_X_dist, std::string& file_sample_indices,
	Filtration_value& threshold, int& dim_max, int& edge_collapse_iter_nb, bool& print_0_pm, bool& coordinate_input
);

// Function to compute merge distances of classes
void merge_distances(
	Phat_index bar_idx, std::vector<Phat_index>& related_intervals, Phat_index num_vertices,
	Barcodes_dim& dimBarcode,
	Reps_dim& dimReps,
	Filtration_value& threshold,
	std::vector<double>& merge_values
);

int main(int argc, char* argv[]) {
	std::cout << "WELCOME TO IBLOFUNMATCH!!!" << std::endl;
	#ifdef DEBUG_STRENGTHS
		std::cout << "Defined DEBUG_STRENGTHS-------------------------(!!!!)" << std::endl;
    #endif
	#ifdef DEBUG_MERGES
		std::cout << "Defined DEBUG_MERGES-------------------------(!!!!)" << std::endl;
	#endif
	// ------------------------------------------------------------------------
	// READ DATA distance matrices (or coordinates) and other options
	// ------------------------------------------------------------------------
	// Read arguments and distance matrix locations
	std::string file_S_dist, file_X_dist, file_sample_indices;
	Filtration_value threshold;
	int dim_max;
	int edge_collapse_iter_nb;
	bool print_0_pm, coordinate_input;

	program_options(
		argc, argv, file_S_dist, file_X_dist, file_sample_indices,
		threshold, dim_max, edge_collapse_iter_nb, print_0_pm, coordinate_input
	);
	// ------------------------------------------------------------------------
	// Read and sort subset indices
	// ------------------------------------------------------------------------
	// Read subset indices
	std::vector<size_t> sample_indices;
	std::ifstream subset_idx(file_sample_indices);
	size_t idx_S;
	while (subset_idx >> idx_S) {
		sample_indices.push_back(idx_S);
	}
	
	// SORT indices of subset and dist S 
	std::vector<size_t> order_sample;
	for (size_t idx = 0; idx < sample_indices.size(); idx++) {
		order_sample.push_back(idx);
	}
	std::sort(
		order_sample.begin(),
		order_sample.end(),
		[&sample_indices](size_t& i, size_t& j) {
			return sample_indices[i] < sample_indices[j];
		}
	);

	// Store new reordering 
	std::vector<size_t> sample_indices_sort;
	for (size_t idx : order_sample) {
		sample_indices_sort.push_back(sample_indices[idx]);
	}
	sample_indices = sample_indices_sort;
	std::cout << "sorted sample indices, total: " << sample_indices.size() << std::endl;

	// ------------------------------------------------------------------------
	// DISTANCE MATRIX INPUT
	// ------------------------------------------------------------------------
	Proximity_graph graph_S, graph_X;
	size_t size_S = sample_indices.size(); 
	size_t size_X;
	if (!coordinate_input) {
		// Read Distance Matrices 
		std::cout << "Reading distance Matrices." << std::endl;
		Distance_matrix dist_S = Gudhi::read_lower_triangular_matrix_from_csv_file<Filtration_value>(file_S_dist, ' ');
		Distance_matrix dist_X = Gudhi::read_lower_triangular_matrix_from_csv_file<Filtration_value>(file_X_dist, ' ');
		// Store size of X 
		size_X = dist_X.size();
		std::cout << "Finished reading matrices and sample indices." << std::endl;
		// -------------- Sort subset distance matrix "dist_S" according to new order---------------//
		Distance_matrix dist_S_sort;
		for (size_t row_idx = 0; row_idx < order_sample.size(); row_idx++) {
			size_t old_row_idx = order_sample[row_idx];
			std::vector<Filtration_value> row;
			for (size_t col_idx = 0; col_idx < row_idx; col_idx++) {
				size_t old_col_idx = order_sample[col_idx];
				if (old_row_idx > old_col_idx) {
					row.push_back(dist_S[old_row_idx][old_col_idx]);
				}
				else {
					row.push_back(dist_S[old_col_idx][old_row_idx]);
				}
			}
			dist_S_sort.push_back(row);
		}
		dist_S = dist_S_sort;
		std::cout << "sorted S according to new order" << std::endl;
		// -------------------- Check that input is valid (if necessary)----------------------------------//
		std::cout << "stored new sorted distance matrix" << std::endl;
		if (!coordinate_input) {
			// Check that distances from S are greater than those from X 
			for (size_t row_idx = 0; row_idx < sample_indices.size(); row_idx++) {
				for (size_t col_idx = 0; col_idx < row_idx; col_idx++) {
					assert(dist_S[row_idx][col_idx] >= dist_X[sample_indices[row_idx]][sample_indices[col_idx]]);
				}
			}
			std::cout << "Correctly checked inequality on dist_S and dist_X" << std::endl;
		}
		// -------------------- Compute proximity graphs  ---------------------------// 
		graph_S = Gudhi::compute_proximity_graph<Simplex_tree>(
			boost::irange((size_t)0, dist_S.size()),
			threshold,
			[&dist_S](size_t i, size_t j) {
				return dist_S[j][i];
			}
		);
		graph_X = Gudhi::compute_proximity_graph<Simplex_tree>(
			boost::irange((size_t)0, dist_X.size()),
			threshold,
			[&dist_X](size_t i, size_t j) {
				return dist_X[j][i];
			}
		);
	}
	// ------------------------------------------------------------------------
	// COORDINATES INPUT
	// ------------------------------------------------------------------------
	else {
		// Read coordinates
		// Extract the points from the file filepoints (subset)
		Points_off_reader off_reader_points(file_X_dist);
		if (!off_reader_points.is_valid()) {
			std::cout << "Off reader failed." << std::endl;
		}
		std::vector<Point> point_X = off_reader_points.get_point_cloud();
		// Store size of X 
		size_X = point_X.size();
		// Now read subset indices
		std::vector<size_t> sample_indices;
		std::ifstream subset_idx(file_sample_indices);
		size_t idx_S;
		while (subset_idx >> idx_S) {
			sample_indices.push_back(idx_S);
		}
		std::sort(sample_indices.begin(), sample_indices.end());
		std::vector<Point> point_S;
		for (int idx : sample_indices) {
			point_S.push_back(point_X.at(idx));
		}
		// --------------------- Compute proximity graphs ---------------------------
		// Compute the proximity graph of the points for set and subset
		graph_S = Gudhi::compute_proximity_graph<Simplex_tree>(point_S,
			threshold,
			Gudhi::Euclidean_distance());
		graph_X = Gudhi::compute_proximity_graph<Simplex_tree>(point_X,
			threshold,
			Gudhi::Euclidean_distance());
	}
	
	// ------------------------------------------------------------------------
	// Initialize edge lists and call PerMoVec
	// ------------------------------------------------------------------------
	// Compute graph edges 
	auto edges_graph_S = boost::adaptors::transform(edges(graph_S), [&](auto&& edge) {
		return std::make_tuple(source(edge, graph_S),
		target(edge, graph_S),
		get(Gudhi::edge_filtration_t(), graph_S, edge));
		});
	auto edges_graph_X = boost::adaptors::transform(edges(graph_X), [&](auto&& edge) {
		return std::make_tuple(source(edge, graph_X),
		target(edge, graph_X),
		get(Gudhi::edge_filtration_t(), graph_X, edge));
		});
	// Store graph edges into list
	std::vector<Filtered_edge> edges_list_S(edges_graph_S.begin(), edges_graph_S.end());
	std::vector<Filtered_edge> edges_list_X(edges_graph_X.begin(), edges_graph_X.end());
	// Initialize output variables
	Barcodes_dim S_barcode, X_barcode;
	Reps_dim S_reps, S_reps_im, X_reps;
	Matrix_dim pm_matrix;
	// Call PerMoVec to get persistent homologies and associated matrices
	pairs_and_matrix_VR(
		size_X, size_S, sample_indices,
		edges_list_X, edges_list_S,
		threshold, dim_max, edge_collapse_iter_nb,
		S_barcode, S_reps, S_reps_im,
		X_barcode, X_reps,
		pm_matrix
	);
	std::cout << "Returned from PerMoVEC" << std::endl;
	// ---------------------------------------------------------------------------
	// Sort permovec output according to startpoint and endpoitn order and store into files
	// ---------------------------------------------------------------------------
	// In dimensions 0 and 1
	// Sort S_barcode, S_reps, S_reps_im and columns from pm_matrix 
	// by following the "startpoint order" 
	std::cout << "Sorting startpoint 0" << std::endl;
	sort_startpoint(S_barcode[0], std::get<0>(S_reps), std::get<0>(S_reps_im), pm_matrix[0]);
	std::cout << "Sorting startpoint 1" << std::endl;
	sort_startpoint(S_barcode[1], std::get<1>(S_reps), std::get<1>(S_reps_im), pm_matrix[1]);
	// Do the same on X by following the endpoint order 
	std::cout << "Sorting endpoint 0" << std::endl;
	sort_endpoint(X_barcode[0], std::get<0>(X_reps), pm_matrix[0]);
	std::cout << "Sorting endpoint 1" << std::endl;
	sort_endpoint(X_barcode[1], std::get<1>(X_reps), pm_matrix[1]);

	// Save permovec output into files (for dimensions 0 and 1)
	store_permovec_output(S_barcode, S_reps, S_reps_im, X_barcode, X_reps, pm_matrix, sample_indices, print_0_pm);
	
	//----------------------------------------------------------------
	// COMPUTE INDUCED MATCHING 
	//----------------------------------------------------------------
	std::vector<std::vector<Phat_index>> induced_matching_dim;
	std::cout << "Ready to compute the induced matchings" << std::endl;
	for (int dim = 0; dim < 2; dim++) {
		// Prepare matrix to reduce 
		Phat_boundary_matrix red_pm_matrix;
		std::cout << "Filling red_pm_matrix, dim " << dim << std::endl;
		Phat_index start_index = X_barcode[dim].size();
		red_pm_matrix.set_num_cols(start_index + pm_matrix[dim].size());
		for (Phat_index col_idx = 0; col_idx < pm_matrix[dim].size(); col_idx++) {
			red_pm_matrix.set_col(start_index + col_idx, pm_matrix[dim][col_idx]);
		}
		#ifdef DEBUG_MATCHING
		    std::cout << "Filled, printing: " << std::endl;
		    for (Phat_index col_idx = start_index; col_idx < red_pm_matrix.get_num_cols(); col_idx++) {
		    	std::vector<Phat_index> column;
		    	red_pm_matrix.get_col(col_idx, column);
		    	std::cout << col_idx << " : ";
		    	for (Phat_index entry : column) {
		    		std::cout << entry << " ";
		    	}
		    	std::cout << std::endl;
		    }
		    std::cout << "Now going to reduce" << std::endl;
		#endif
		// Reduce matrix using PHAT
		phat::persistence_pairs _pairs;
		// compute persistent homology by means of the standard reduction
		phat::compute_persistence_pairs<phat::standard_reduction>(_pairs, red_pm_matrix);
		// Read column pivots and store into matching 
		std::cout << "Reduced" << std::endl;
		// Value of -1 means that there is no matching 
		std::vector<Phat_index> induced_matching(pm_matrix[dim].size(), -1);
		for (Phat_index col_idx = start_index; col_idx < red_pm_matrix.get_num_cols(); col_idx++) {
			std::vector<Phat_index> column;
			red_pm_matrix.get_col(col_idx, column);
			if (column.size() > 0) {
				induced_matching[col_idx - start_index] = column.back();
			}
		}
		// Store induced matching into variable
		induced_matching_dim.push_back(induced_matching);
		// Store induced matching into file 
		std::string ind_match_f = "output/induced_matching_" + std::to_string(dim) + ".out";
		std::ofstream out_ind_match(ind_match_f);
		for (Phat_index idx_match : induced_matching) {
			out_ind_match << idx_match << std::endl;
		}
		out_ind_match.close();
	} // compute induced matching on dimensions 0 and 1
	
	// ---------------------------------------------------------------------------------//
	// Compute Matching Strength 
	// ---------------------------------------------------------------------------------//
	// ------------------------------------ DIMENSION 0 ---------------------------------
	std::ofstream out_match_strength("output/matching_strengths_0.out");
	for (idx_S = 0; idx_S < induced_matching_dim[0].size(); idx_S++) {
		// First compute the matched image interval length
		size_t idx_match = induced_matching_dim[0][idx_S];
		if ((idx_match < 0) || (idx_match >= X_barcode[0].size())) { // check it is a proper matching
			out_match_strength << -1 << " ";
			continue;
		}
		double death = X_barcode[0][idx_match].second;
		// 0 intervals are all born at 0
		out_match_strength << death << " ";
	}
	out_match_strength << std::endl;
	out_match_strength.close();

	// ------------------------------------ DIMENSION 1 ---------------------------------
	std::vector<double> matching_strengths;
	std::cout << "Computing strenghts..." << std::endl;
	for (idx_S = 0; idx_S < induced_matching_dim[1].size(); idx_S++) {
		// First compute the matched image interval length
		size_t idx_match = induced_matching_dim[1][idx_S];
		if ((idx_match < 0)||(idx_match>=X_barcode[1].size())) { // check it is a proper matching
			std::cout << "idx_S: " << idx_S << " (not matched)" << std::endl << std::endl;
			matching_strengths.push_back(-1);
			continue;
		}
		double birth = S_barcode[1][idx_S].first;
		double death = X_barcode[1][idx_match].second;
		double im_len = death - birth;

		#ifdef DEBUG_STRENGTHS
			std::cout << "idx_S: " << idx_S << " ";
			std::cout << "idx_match: " << idx_match << " ";
			std::cout << "[" << birth << ", " << death;
			std::cout << "), im_len: " << im_len << std::endl;
		#endif
		// -------------------------------------------------------------------------------//
		// Compute Strength on S 
		// -------------------------------------------------------------------------------//
		std::vector<Phat_index> related_intervals = {};
		std::vector<double> S_compare = {};
		for (Phat_index i = 0; i < S_barcode[1].size(); i++) {
			if ( abs(S_barcode[1][i].first - birth) < im_len) {
				if (i != idx_S) {
					related_intervals.push_back(i);
					S_compare.push_back(std::max(
						abs(S_barcode[1][i].first - birth),
						death - S_barcode[1][i].second
					));
				}
			}
		}
		#ifdef DEBUG_STRENGTHS
			std::cout << "Computed related intervals on S, size: " << related_intervals.size() << std::endl;
        #endif
		// If there are no related intervals in S, take image overlap as min_comp_S
		double min_comp_S = 0;
		if (related_intervals.size()==0) {
			min_comp_S = im_len;
		} else {
			#ifdef DEBUG_STRENGTHS
				std::cout << "related int S  : ";
				for (Phat_index i : related_intervals) {
					printf("%5ld, ", i);
				}
				std::cout << std::endl;
            #endif
			// Compute Strength of merges
			std::vector<double> merge_values = {};
			merge_distances(
				idx_S, related_intervals, size_S, S_barcode, S_reps, threshold, merge_values
			);
			#ifdef DEBUG_STRENGTHS
				std::cout << "S_compare orig : ";
				for (int i = 0; i < S_compare.size(); i++) {
					printf("%.3f, ", S_compare[i]);
				}
				std::cout << std::endl;
				std::cout << "Merge values S : ";
				for (int i = 0; i < merge_values.size(); i++) {
					printf("%.3f, ", merge_values[i]);
				}
				std::cout << std::endl;
			#endif
			for (int i = 0; i < merge_values.size(); i++) {
				S_compare[i] = std::max(S_compare[i], merge_values[i]);
			}
            #ifdef DEBUG_STRENGTHS
				std::cout << "S_compare res  : ";
				for (int i = 0; i < S_compare.size(); i++) {
					printf("%.3f, ", S_compare[i]);
				}
				std::cout << std::endl;
            #endif
			// Take minimum from endpoint comparisons 
			min_comp_S = S_compare[0];
			for (double comp_val : S_compare) {
				min_comp_S = std::min(min_comp_S, comp_val);
			}
		}
		// -------------------------------------------------------------------------------//
		// Compute Strength on X 
		// -------------------------------------------------------------------------------//
		std::vector<Phat_index> related_intervals_X = {};
		std::vector<double> X_compare = {};
		for (Phat_index i = 0; i < X_barcode[1].size(); i++) {
			if (abs(X_barcode[1][i].second - death) < im_len) {
				if (i != idx_match) {
					related_intervals_X.push_back(i);
					X_compare.push_back(std::max(
						abs(X_barcode[1][i].second - death),
						X_barcode[1][i].first - birth
					));
				}
			}
		}
		double min_comp_X;
		// If there are no related intervals in X, take image overlap
		if (related_intervals_X.size() == 0) {
			min_comp_X = im_len;
		} else {
			#ifdef DEBUG_STRENGTHS
				std::cout << "related int X  : ";
				for (Phat_index i : related_intervals_X) {
					printf("%5ld, ", i);
				}
				std::cout << std::endl;
            #endif
			// Compute Strength of merges
			std::vector<double> merge_values = {};
			merge_distances(
				idx_match, related_intervals_X, size_X, X_barcode, X_reps, threshold, merge_values
			);
			#ifdef DEBUG_STRENGTHS
			    std::cout << "X_compare orig : ";
			    for (int i = 0; i < X_compare.size(); i++) {
			    	printf("%.3f, ", X_compare[i]);
			    }
			    std::cout << std::endl;
			    std::cout << "Merge values X : ";
			    for (int i = 0; i < merge_values.size(); i++) {
			    	printf("%.3f, ", merge_values[i]);
			    }
			    std::cout << std::endl;
            #endif
			for (int i = 0; i < merge_values.size(); i++) {
				X_compare[i] = std::max(X_compare[i], merge_values[i]);
			}
			#ifdef DEBUG_STRENGTHS
			    std::cout << "X_compare res  : ";
			    for (int i = 0; i < X_compare.size(); i++) {
			    	printf("%.3f, ", X_compare[i]);
			    }
			    std::cout << std::endl;
            #endif	
			// Take minimum from endpoint comparisons 
			min_comp_X = X_compare[0];
			for (double comp_val : S_compare) {
				min_comp_X = std::min(min_comp_X, comp_val);
			}
		}
		#ifdef DEBUG_STRENGTHS
			std::cout << " STRENGTH: " << std::min(im_len, std::min(min_comp_S, min_comp_X)) << std::endl;
			std::cout << std::endl;
        #endif
		matching_strengths.push_back(std::min(im_len, std::min(min_comp_S, min_comp_X)));
	} // Compute matching strenghts over each bar
	// Store matching strengths into file 
	out_match_strength.open("output/matching_strengths_1.out");
	for (double strength_m : matching_strengths) {
		out_match_strength << strength_m << " ";
	}
	out_match_strength << std::endl;
	out_match_strength.close();
	// Now compute other quantities 
	return 0;
} // End main


void merge_distances(
	Phat_index bar_idx, 
	std::vector<Phat_index>& related_intervals, // indices to intervals to compare
	Phat_index num_vertices,
	Barcodes_dim& dimBarcode,
	Reps_dim& dimReps,
	Filtration_value& threshold,
	std::vector<double>& merge_values)
{
	// ---------------------------------------------------------------------
	// Initialize merging PHAT matrix and column cell dimensions
	// ---------------------------------------------------------------------
	// Initialize a matrix to compute when two components merge after they die
	Phat_boundary_matrix vertex_pairs_matrix;
	// Columns correspond to vertices, 0-dim PH classes and possible related intervals
	vertex_pairs_matrix.set_num_cols(num_vertices + dimBarcode[0].size() + related_intervals.size());
	// Set first num_vertices columns to dimension 0 (needed for PHAT)
	Phat_index col_idx = 0;
	while (col_idx < num_vertices) {
		vertex_pairs_matrix.set_dim(col_idx, 0);
		col_idx++;
	}
	// Set first dimension of other columns 
	while (col_idx < num_vertices + dimBarcode[0].size() + related_intervals.size()) {
		vertex_pairs_matrix.set_dim(col_idx, 1);
		col_idx++;
	}
	// ------------------------ Fill boundary matrix with zero PH representatives from S ---------------------
	col_idx = num_vertices;
	for (Phat_column zero_rep : std::get<0>(dimReps)) {
		vertex_pairs_matrix.set_col(col_idx, zero_rep);
		col_idx++;
		#ifdef DEBUG_MERGES
		    std::cout << "zero rep: " << std::endl;
		    for (Phat_index entry : zero_rep) {
		    	std::cout << entry << " ";
		    }
		    std::cout << std::endl;
		#endif
	}
	// ------------------------ Fill boundary matrix with vertices from first PH representatives from S ---------------------
	Phat_index start_index = num_vertices + dimBarcode[0].size(); // Column index where to start from
	Phat_index bar_vertex = std::get<1>(dimReps)[bar_idx][0].first; // Just get one vertex from the representative of bar_idx
	for (Phat_index rint_count = 0; rint_count < related_intervals.size(); rint_count++) {
		Phat_index i = related_intervals[rint_count];
		if (i == bar_idx) {
			std::cout << "Same index interval error" << std::endl;
			exit(1); // This should not happen
		}
		Phat_index vertex_i = std::get<1>(dimReps)[i][0].first;
		if (vertex_i != bar_vertex) {
			vertex_pairs_matrix.set_col(start_index + rint_count, { std::min(vertex_i, bar_vertex), std::max(vertex_i, bar_vertex) });
		}
	}
	#ifdef DEBUG_MERGES
	    std::cout << "Going to PHAT reduce the matrix:" << std::endl;
	    for (Phat_index j = num_vertices; j < vertex_pairs_matrix.get_num_cols(); j++) {
	    	Phat_column column;
	    	vertex_pairs_matrix.get_col(j, column);
	    	for (Phat_index entry : column) {
	    		std::cout << entry << " ";
	    	}
	    	std::cout << std::endl;
	    }
	# endif
	// ------------------------ PHAT reduce the matrix ---------------------
	// The result leads to the coefficients R_ij
	phat::persistence_pairs _pairs;
	phat::compute_persistence_pairs<phat::standard_reduction>(_pairs, vertex_pairs_matrix);
	#ifdef DEBUG_MERGES
	    std::cout << "phat reduction done, Result" << std::endl;
	    for (Phat_index j = num_vertices; j < vertex_pairs_matrix.get_num_cols(); j++) {
	    	Phat_column column;
	    	vertex_pairs_matrix.get_col(j, column);
	    	for (Phat_index entry : column) {
	    		std::cout << entry << " ";
	    	}
	    	std::cout << std::endl;
	    }
		std::cout << "num_vertices : " << num_vertices << std::endl;
		std::cout << "dim 0 barcode: " << dimBarcode[0].size() << std::endl;
		std::cout << "related_intervals.size(): " << related_intervals.size() << std::endl;
	#endif
	// Obtain merging distances by using the PHAT result
	for (Phat_index rint_count = 0; rint_count < related_intervals.size(); rint_count++) {
		Phat_column reduced_col;
		vertex_pairs_matrix.get_col(start_index + rint_count, reduced_col);
		// If the column is not totally reduced, set up the threshold as merging distance 
		if (reduced_col.size() > 0) {
			merge_values.push_back(threshold);
			continue;
		}
		Phat_index j_bar = related_intervals[rint_count];
		Phat_column preim_col;
		vertex_pairs_matrix.get_preimage(start_index + rint_count, preim_col);
		// sort preimage 
		std::sort(preim_col.begin(), preim_col.end());
		#ifdef DEBUG_MERGES
		    std::cout << j_bar << " col: ";
		    for (Phat_index entry : reduced_col) {
		    	std::cout << entry << " ";
		    }
		    std::cout << ") p(";
		    for (int i = 0; i < preim_col.size(); i++) {
		    	std::cout << preim_col[i] - num_vertices << " ";
		    }
		    std::cout << ") " << std::endl;
		#endif
		// Get value where components from both cycles merge
		double merge_val = 0;
		// Last entry of preimage_col is the index of the column, which we skip
		for (int i = 0; i < preim_col.size()-1; i++) {
			// First num_vertices columns are empty, barcode 0 counter starts at num_vertices
			merge_val = std::max(merge_val, dimBarcode[0][preim_col[i] - num_vertices].second);
		}
		// Get maximum death of both cycles
		double death_max = std::max(dimBarcode[1][bar_idx].second, dimBarcode[1][j_bar].second);
		// Compute the ''Merge Distance''
		double merge_dist = std::max(0.0, merge_val - death_max);
		merge_values.push_back(merge_dist);
	}
} // end of merge_distances

void program_options(
	int argc, char* argv[], std::string& file_S_dist, std::string& file_X_dist, std::string& file_sample_indices,
	Filtration_value& threshold, int& dim_max, int& edge_collapse_iter_nb, bool& print_0_pm, bool& coordinate_input
) {
	namespace po = boost::program_options;
	po::options_description hidden("Hidden options");
	hidden.add_options()(
		"S_dist", po::value<std::string>(&file_S_dist),
		"Text file containing the distance matrix of S. If coordinate-input=true, this file is ignored.");
	hidden.add_options()(
		"X_dist", po::value<std::string>(&file_X_dist),
		"Text file containing the distance matrix of X. If coordinate-input=true, this file should contain the coordinates of X.");
	hidden.add_options()(
		"sample-indices", po::value<std::string>(&file_sample_indices),
		"Indices corresponding to S within X");

	po::options_description visible("Allowed options", 100);
	visible.add_options()("help,h", "produce help message")(
		"max-edge-length,r", 
		po::value<Filtration_value>(&threshold)->default_value(std::numeric_limits<Filtration_value>::infinity()),
		"Maximal length of an edge for the Rips complex construction."
		)(
		"cpx-dimension,d", po::value<int>(&dim_max)->default_value(1),
		"Maximal dimension of the Rips complex we want to compute."
		)(
		"edge-collapse-iterations,i", po::value<int>(&edge_collapse_iter_nb)->default_value(1),
		"Number of iterations edge collapse is performed."
		)(
		"save-0-pm-matrix,z", po::value<bool>(&print_0_pm)->default_value(false),
		"Print 0 persistence morphism matrix into a file (might be big)."
		)(
		"coordinate-input,c", po::value<bool>(&coordinate_input)->default_value(false),
		"Whether the input are coordinates of the dataset."
	);

	po::positional_options_description pos;
	pos.add("S_dist", 1);
	pos.add("X_dist", 1);
	pos.add("sample-indices", 1);

	po::options_description all;
	all.add(visible).add(hidden);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(all).positional(pos).run(), vm);
	po::notify(vm);

	if (vm.count("help") || !vm.count("S_dist") || !vm.count("X_dist") || !vm.count("sample-indices")) {
		std::cout << std::endl;
		std::cout << "Given metric spaces S and X together with an inclusion f\n";
		std::cout << "from S to X, indicated by a set of indices, which is \n";
		std::cout << "such that, for all a,b from X,  \n\n";

		std::cout << "         d(a,b) >= d_X(f(a), f(b)).\n\n";

		std::cout << "This induces a morphism of persistence modules\n\n";

		std::cout << "         PH_k(VR(S))-->PH_k(VR(X)), \n\n";

		std::cout << "for all k>=0, and with fixed field Z/2Z.\n";
		std::cout << "Usage: " << argv[0] << " [options] file_dist_S file_dist_X sample-indices, where:\n";
		std::cout << " [file_dist_S] is the file storing the distance matrix from S\n";
		std::cout << " [file_dist_X] is the file storing the distance matrix from X\n";
		std::cout << " [sample-indices] is a file with the indices of elements from S in X.\n" << std::endl;
		std::cout << visible << std::endl;
		exit(-1);
	}
}

	