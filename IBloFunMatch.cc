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
using Cycle_rep = std::vector<Vertex_pair>;
using Matrix_dim = std::unordered_map<int, std::vector<std::vector<Phat_index>>>;
using Distance_matrix = std::vector<std::vector<Filtration_value>>;

void program_options(
	int argc, char* argv[], std::string& file_S_dist, std::string& file_X_dist, std::string& file_sample_indices,
	Filtration_value& threshold, int& dim_max, int& edge_collapse_iter_nb
);

void sort_startpoint(
	std::vector<Interval>& barcode_1,
	std::vector<Cycle_rep_1>& cycles_1,
	std::vector<Cycle_rep_1>& cycles_1_im,
	std::vector<std::vector<Phat_index>>& pm_matrix_1
);

void sort_endpoint(
	std::vector<Interval>& barcode_1,
	std::vector<Cycle_rep_1>& cycles_1,
	std::vector<std::vector<Phat_index>>& pm_matrix_1
);

int main(int argc, char* argv[]) {
	// ------------------------------------------------------------------------
	// READ DATA and sort subset 
	// ------------------------------------------------------------------------
	// Read arguments and distance matrix locations
	std::string file_S_dist, file_X_dist, file_sample_indices;
	Filtration_value threshold;
	int dim_max;
	int edge_collapse_iter_nb;

	program_options(
		argc, argv, file_S_dist, file_X_dist, file_sample_indices,
		threshold, dim_max, edge_collapse_iter_nb
	);
	// Read subset indices
	std::vector<size_t> sample_indices;
	std::ifstream subset_idx(file_sample_indices);
	size_t idx_S;
	while (subset_idx >> idx_S) {
		sample_indices.push_back(idx_S);
	}
	// Read Distance Matrices 
	Distance_matrix dist_S = Gudhi::read_lower_triangular_matrix_from_csv_file<Filtration_value>(file_S_dist, ' ');
	Distance_matrix dist_X = Gudhi::read_lower_triangular_matrix_from_csv_file<Filtration_value>(file_X_dist, ' ');
	std::cout << "sample_indices (" << sample_indices.size() << "): ";
	for (size_t idx : sample_indices) {
		std::cout << idx << ", ";
	}
	std::cout << std::endl;
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
	// Sort dist_S according to new order
	Distance_matrix dist_S_sort;
	for (size_t row_idx = 0; row_idx < order_sample.size(); row_idx++) {
		size_t old_row_idx = order_sample[row_idx];
		std::vector<Filtration_value> row;
		for (size_t col_idx = 0; col_idx < row_idx; col_idx++) {
			size_t old_col_idx = order_sample[col_idx];
			if (old_row_idx > old_col_idx) {
				row.push_back(dist_S[old_row_idx][old_col_idx]);
			} else {
				row.push_back(dist_S[old_col_idx][old_row_idx]);
			}
		}
		dist_S_sort.push_back(row);
	}
	// Store new reordering 
	std::vector<size_t> sample_indices_sort;
	for (size_t idx : order_sample) {
		sample_indices_sort.push_back(sample_indices[idx]);
	}
	sample_indices = sample_indices_sort;
	dist_S = dist_S_sort;
	// Check that distances from S are greater than those from X 
	for (size_t row_idx = 0; row_idx < sample_indices.size(); row_idx++) {
		for (size_t col_idx = 0; col_idx < row_idx; col_idx++) {
			assert(dist_S[row_idx][col_idx] >= dist_X[sample_indices[row_idx]][sample_indices[col_idx]]);
			assert(abs(dist_S[row_idx][col_idx] - dist_X[sample_indices[row_idx]][sample_indices[col_idx]])<_tolerance);
		}
	}
	std::cout << "Correctly checked inequality on dist_S and dist_X" << std::endl;
	std::cout << "Sample indices (sorted)" << std::endl;
	for (size_t idx : sample_indices) {
		std::cout << idx << ", ";
	}
	std::cout << std::endl;
	// ------------------------------------------------------------------------
	// Call PerMoVEC and initialize variables
	// ------------------------------------------------------------------------
	// Compute Proximity Graphs 
	Proximity_graph graph_X = Gudhi::compute_proximity_graph<Simplex_tree>(
		boost::irange((size_t)0, dist_X.size()),
		threshold,
		[&dist_X](size_t i, size_t j) {
			return dist_X[j][i];
		}
	);
	Proximity_graph graph_S = Gudhi::compute_proximity_graph<Simplex_tree>(
		boost::irange((size_t)0, dist_S.size()),
		threshold,
		[&dist_S](size_t i, size_t j) {
			return dist_S[j][i];
		}
	);
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
	std::vector<Filtered_edge> edges_list_S(edges_graph_S.begin(), edges_graph_S.end());
	std::vector<Filtered_edge> edges_list_X(edges_graph_X.begin(), edges_graph_X.end());
	// Compute barcodes and matrix associated to persistence morphism
	Barcodes_dim S_barcode, X_barcode;
	Reps_dim S_reps, S_reps_im, X_reps;
	Matrix_dim pm_matrix;
	pairs_and_matrix_VR(
		dist_X.size(), dist_S.size(), sample_indices,
		edges_list_X, edges_list_S,
		threshold, dim_max, edge_collapse_iter_nb,
		S_barcode, S_reps, S_reps_im,
		X_barcode, X_reps,
		pm_matrix
	);
	// ---------------------------------------------------------------------------
	// Use the PerMoVEC output to compute IBloFunMatch 
	// ---------------------------------------------------------------------------
	std::cout << "Ready to compute block functions" << std::endl;
	// Sort S_barcode, S_reps, S_reps_im and columns from pm_matrix 
	// by following the "startpoint order" (dim 1 only)
	sort_startpoint(S_barcode[1], std::get<1>(S_reps), std::get<1>(S_reps_im), pm_matrix[1]);
	// Do the same on X by following the endpoint order 
	sort_endpoint(X_barcode[1], std::get<1>(X_reps), pm_matrix[1]);
	// Print out barcodes, cycle reps and matrix 
	// Save info into files
	std::ofstream out_X_bar("output/X_barcode.out");
	for (Interval bar : X_barcode[1]) {
		out_X_bar << bar.first << " " << bar.second << std::endl;
	}
	out_X_bar.close();
	std::ofstream out_X_reps("output/X_reps.out");
	for (Cycle_rep_1 cycle_rep : std::get<1>(X_reps)) {
		for (Vertex_pair edge : cycle_rep) {
			out_X_reps << edge.first << " " << edge.second << " ";
		}
		out_X_reps << std::endl;
	}
	out_X_reps.close();
	std::ofstream out_S_bar("output/S_barcode.out");
	for (Interval bar : S_barcode[1]) {
		out_S_bar << bar.first << " " << bar.second << std::endl;
	}
	out_S_bar.close();
	std::ofstream out_S_reps("output/S_reps.out");
	for (Cycle_rep_1 cycle_rep : std::get<1>(S_reps)) {
		for (Vertex_pair edge : cycle_rep) {
			out_S_reps << sample_indices[edge.first] << " " << sample_indices[edge.second] << " ";
		}
		out_S_reps << std::endl;
	}
	out_S_reps.close();
	std::ofstream out_S_reps_im("output/S_reps_im.out");
	for (Cycle_rep_1 cycle_rep : std::get<1>(S_reps_im)) {
		for (Vertex_pair edge : cycle_rep) {
			out_S_reps_im << edge.first << " " << edge.second << " ";
		}
		out_S_reps_im << std::endl;
	}
	out_S_reps_im.close();
	std::ofstream out_pm_matrix("output/pm_matrix.out");
	for (std::vector<Phat_index>& column : pm_matrix[1]) {
		for (Phat_index entry : column) {
			out_pm_matrix << entry << " ";
		}
		out_pm_matrix << std::endl;
	}
	out_pm_matrix.close();
	//----------------------------------------------------------------
	// COMPUTE INDUCED MATCHING 
	//----------------------------------------------------------------
	// Prepare matrix to reduce 
	Phat_boundary_matrix red_pm_matrix;
	std::cout << "Filling red_pm_matrix" << std::endl;
	Phat_index start_index = X_barcode[1].size();
	red_pm_matrix.set_num_cols(start_index + pm_matrix[1].size());
	for (Phat_index col_idx = 0; col_idx < pm_matrix[1].size(); col_idx++) {
		red_pm_matrix.set_col(start_index + col_idx, pm_matrix[1][col_idx]);
	}
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
	// Reduce matrix using PHAT
	phat::persistence_pairs irrelevant_pairs;
	// compute persistent homology by means of the standard reduction
	phat::compute_persistence_pairs<phat::standard_reduction>(irrelevant_pairs, red_pm_matrix);
	// Read column pivots and store into matching 
	std::cout << "Reduced" << std::endl;
	// Value of -1 means that there is no matching 
	std::vector<Phat_index> induced_matching(pm_matrix[1].size(), -1);
	for (Phat_index col_idx = start_index; col_idx < red_pm_matrix.get_num_cols(); col_idx++) {
		std::vector<Phat_index> column;
		red_pm_matrix.get_col(col_idx, column);
		if (column.size() > 0) {
			induced_matching[col_idx-start_index] = column.back();
		}
	}
	// Store induced matching into file 
	std::ofstream out_ind_match("output/induced_matching.out");
	for (Phat_index idx_match : induced_matching) {
		out_ind_match << idx_match << std::endl;
	}
	out_ind_match.close();
	return 0;
} // End main

void sort_startpoint(
	std::vector<Interval>& barcode_1,
	std::vector<Cycle_rep_1>& cycles_1,
	std::vector<Cycle_rep_1>& cycles_1_im,
	std::vector<std::vector<Phat_index>>& pm_matrix_1) 
{
	std::vector<size_t> sort_indices;
	for (size_t idx = 0; idx < barcode_1.size(); idx++) {
		sort_indices.push_back(idx);
	}
	std::sort(
		sort_indices.begin(), sort_indices.end(),
		[&barcode_1](size_t& i, size_t& j) {
			return (barcode_1[i].first < barcode_1[j].first) || (
				(barcode_1[i].first == barcode_1[j].first) && (barcode_1[i].second < barcode_1[j].second)
				);
		}
	);
	std::vector<Interval> sorted_barcode;
	std::vector<Cycle_rep_1> sorted_cycle;
	std::vector<Cycle_rep_1> sorted_cycle_im;
	std::vector<std::vector<Phat_index>> sorted_matrix;
	for (size_t idx : sort_indices) {
		sorted_barcode.push_back(barcode_1[idx]);
		sorted_cycle.push_back(cycles_1[idx]);
		sorted_cycle_im.push_back(cycles_1_im[idx]);
		sorted_matrix.push_back(pm_matrix_1[idx]);

	}
	barcode_1 = sorted_barcode;
	cycles_1 = sorted_cycle;
	cycles_1_im = sorted_cycle_im;
	pm_matrix_1 = sorted_matrix;
}

void sort_endpoint(
	std::vector<Interval>& barcode_1,
	std::vector<Cycle_rep_1>& cycles_1,
	std::vector<std::vector<Phat_index>>& pm_matrix_1)
{
	std::vector<size_t> sort_indices;
	for (size_t idx = 0; idx < barcode_1.size(); idx++) {
		sort_indices.push_back(idx);
	}
	std::sort(
		sort_indices.begin(), sort_indices.end(),
		[&barcode_1](size_t& i, size_t& j) {
			return (barcode_1[i].second < barcode_1[j].second) || (
				(barcode_1[i].second == barcode_1[j].second) && (barcode_1[i].first < barcode_1[j].first)
				);
		}
	);
	std::vector<Interval> sorted_barcode;
	std::vector<Cycle_rep_1> sorted_cycle;
	for (size_t idx : sort_indices) {
		sorted_barcode.push_back(barcode_1[idx]);
		sorted_cycle.push_back(cycles_1[idx]);
	}
	barcode_1 = sorted_barcode;
	cycles_1 = sorted_cycle;
	// Now proceed to sort rows of pm_matrix 
	std::vector<std::vector<Phat_index>> sorted_matrix;
	// First create a handy unordered map
	std::unordered_map<Phat_index, Phat_index> sort_indices_inv; 
	size_t index_count = 0;
	for (size_t index : sort_indices) {
		sort_indices_inv[index] = index_count;
		index_count++;
	}
	for (std::vector<Phat_index>& column : pm_matrix_1) {
		std::vector<Phat_index> new_col;
		for (Phat_index entry : column) {
			new_col.push_back(sort_indices_inv[entry]);
		}
		std::sort(new_col.begin(), new_col.end());
		sorted_matrix.push_back(new_col);
	}
	pm_matrix_1 = sorted_matrix;
}


void program_options(
	int argc, char* argv[], std::string& file_S_dist, std::string& file_X_dist, std::string& file_sample_indices,
	Filtration_value& threshold, int& dim_max, int& edge_collapse_iter_nb
) {
	namespace po = boost::program_options;
	po::options_description hidden("Hidden options");
	hidden.add_options()(
		"S_dist", po::value<std::string>(&file_S_dist),
		"Text file containing the distance matrix of S.");
	hidden.add_options()(
		"X_dist", po::value<std::string>(&file_X_dist),
		"Text file containing the distance matrix of X.");
	hidden.add_options()(
		"sample-indices", po::value<std::string>(&file_sample_indices),
		"Indices corresponding to S within X");

	po::options_description visible("Allowed options", 100);
	visible.add_options()("help,h", "produce help message")(
		"max-edge-length,r",
		po::value<Filtration_value>(&threshold)->default_value(std::numeric_limits<Filtration_value>::infinity()),
		"Maximal length of an edge for the Rips complex construction.")(
			"cpx-dimension,d", po::value<int>(&dim_max)->default_value(1),
			"Maximal dimension of the Rips complex we want to compute.")(
				"edge-collapse-iterations,i", po::value<int>(&edge_collapse_iter_nb)->default_value(1),
				"Number of iterations edge collapse is performed.");

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
	