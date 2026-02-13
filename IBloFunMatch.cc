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
	int argc, char* argv[], std::string& file_S_dist, std::string& file_X_dist, std::string& file_sample_indices, std::string& folder_io,
	Filtration_value& threshold, int& dim_max, int& edge_collapse_iter_nb, bool& print_0_pm, bool& coordinate_input
);


int main(int argc, char* argv[]) {
	#ifdef IBLOFUNMATCH_MIN
		std::cout << "WELCOME TO IBLOFUNMATCH!!!" << std::endl;
	#endif
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
	std::string file_S_dist, file_X_dist, file_sample_indices, folder_io;
	Filtration_value threshold;
	int dim_max;
	int edge_collapse_iter_nb;
	bool print_0_pm, coordinate_input;

	program_options(
		argc, argv, file_S_dist, file_X_dist, file_sample_indices, folder_io,
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
	#ifdef DEBUG
		std::cout << "sorted sample indices, total: " << sample_indices.size() << std::endl;
	#endif

	// ------------------------------------------------------------------------
	// DISTANCE MATRIX INPUT
	// ------------------------------------------------------------------------
	Proximity_graph graph_S, graph_X;
	size_t size_S = sample_indices.size(); 
	size_t size_X;
	if (!coordinate_input) {
		// Read Distance Matrices 
		#ifdef DEBUG
			std::cout << "Reading distance Matrices." << std::endl;
		#endif
		Distance_matrix dist_S = Gudhi::read_lower_triangular_matrix_from_csv_file<Filtration_value>(file_S_dist, ' ');
		Distance_matrix dist_X = Gudhi::read_lower_triangular_matrix_from_csv_file<Filtration_value>(file_X_dist, ' ');
		// Store size of X 
		size_X = dist_X.size();
		#ifdef DEBUG
			std::cout << "Finished reading matrices and sample indices." << std::endl;
		#endif
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
		#ifdef DEBUG
			std::cout << "sorted S according to new order" << std::endl;
		#endif
		// -------------------- Check that input is valid (if necessary)----------------------------------//
		#ifdef DEBUG
			std::cout << "stored new sorted distance matrix" << std::endl;
		#endif
		if (!coordinate_input) {
			// Check that distances from S are greater than those from X 
			for (size_t row_idx = 0; row_idx < sample_indices.size(); row_idx++) {
				for (size_t col_idx = 0; col_idx < row_idx; col_idx++) {
					assert(dist_S[row_idx][col_idx] >= dist_X[sample_indices[row_idx]][sample_indices[col_idx]]);
				}
			}
			#ifdef DEBUG
				std::cout << "Correctly checked inequality on dist_S and dist_X" << std::endl;
			#endif
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
	
	#ifdef DEBUG
		std::cout << "Finished with matrices, calling PerMoVec" << std::endl;
	#endif
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
	#ifdef DEBUG
		std::cout << "Returned from PerMoVEC" << std::endl;
	#endif
	// ---------------------------------------------------------------------------
	// Sort permovec output according to startpoint and endpoitn order and store into files
	// ---------------------------------------------------------------------------
	// In dimensions 0 and 1
	// Sort S_barcode, S_reps, S_reps_im and columns from pm_matrix 
	// by following the "startpoint order" 
	sort_startpoint(S_barcode[0], std::get<0>(S_reps), std::get<0>(S_reps_im), pm_matrix[0]);
	sort_startpoint(S_barcode[1], std::get<1>(S_reps), std::get<1>(S_reps_im), pm_matrix[1]);
	// Do the same on X by following the endpoint order 
	sort_endpoint(X_barcode[0], std::get<0>(X_reps), pm_matrix[0]);
	sort_endpoint(X_barcode[1], std::get<1>(X_reps), pm_matrix[1]);

	// Save permovec output into files (for dimensions 0 and 1)
	store_permovec_output(S_barcode, S_reps, S_reps_im, X_barcode, X_reps, pm_matrix, sample_indices, print_0_pm, folder_io);
	
	//----------------------------------------------------------------
	// COMPUTE INDUCED MATCHING 
	//----------------------------------------------------------------
	std::vector<std::vector<Phat_index>> induced_matching_dim;
	for (int dim = 0; dim < 2; dim++) {
		// Prepare matrix to reduce 
		Phat_boundary_matrix red_pm_matrix;
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
		std::string ind_match_f = folder_io + "/induced_matching_" + std::to_string(dim) + ".out";
		std::ofstream out_ind_match(ind_match_f);
		for (Phat_index idx_match : induced_matching) {
			out_ind_match << idx_match << std::endl;
		}
		out_ind_match.close();
	} // compute induced matching on dimensions 0 and 1
	
	//----------------------------------------------------------------
	// COMPUTE INDUCED BLOCK FUNCTION 
	//----------------------------------------------------------------
	#ifdef DEBUG
		std::cout << "Computing block function" << std::endl;
	#endif
	// Basically, reduce associated matrix for each column, containing only the columns 
	// that come just before
	std::vector<std::vector<Phat_index>> block_fun_dim;
	for (int dim = 0; dim < 2; dim++) {
		// Value of -1 means that there is no assignment 
		std::vector<Phat_index> block_fun(pm_matrix[dim].size(), -1);
		Phat_index start_index = X_barcode[dim].size();
		Phat_index num_columns = start_index + pm_matrix[dim].size();
		Phat_boundary_matrix red_pm_matrix_I2;
		red_pm_matrix_I2.set_num_cols(num_columns);
		for (Phat_index i = start_index; i < num_columns; i++) {
			Interval I2 = S_barcode[dim].at(i - start_index);
			// Fill columns up to the column i
			for (Phat_index j = start_index; j <= i; j++) {
				Interval I1 = S_barcode[dim].at(j - start_index);
				// Fill column if associated interval comes first (in both endpoints)
				if ((I1.first <= I2.first) && (I1.second <= I2.second)) {
					red_pm_matrix_I2.set_col(j, pm_matrix[dim][j - start_index]);
				}
				else { // otherwise clear the column
					red_pm_matrix_I2.clear(j); 
				}
			}
			// Reduce matrix using PHAT
			phat::persistence_pairs _pairs;
			// compute persistent homology by means of the standard reduction
			phat::compute_persistence_pairs<phat::standard_reduction>(_pairs, red_pm_matrix_I2);
			// Read pivot from column "i"
			std::vector<Phat_index> column;
			red_pm_matrix_I2.get_col(i, column);
			if (column.size() > 0) {
				block_fun[i - start_index] = column.back();
			}
		}
		// Store induced matching into variable
		block_fun_dim.push_back(block_fun);
		// Store induced matching into file 
		std::string block_fun_f = folder_io + "/block_function_" + std::to_string(dim) + ".out";
		std::ofstream out_block_fun(block_fun_f);
		for (Phat_index idx_match : block_fun) {
			out_block_fun << idx_match << std::endl;
		}
		out_block_fun.close();
	} // compute block function in dimensions 0 and 1

	return 0;
} // End main


void program_options(
	int argc, char* argv[], std::string& file_S_dist, std::string& file_X_dist, std::string& file_sample_indices, std::string& folder_io,
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
		)(
		"folder-io,o", po::value<std::string>(&folder_io)->default_value("output"),
		"Folder to read and write."
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

	
