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
#include <boost/lexical_cast.hpp>

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

int main(int argc, char* argv[]) {
	std::string threshold_in = argv[1];
	Filtration_value threshold = boost::lexical_cast<Filtration_value>(*threshold_in);
	// Read subset indices
	std::vector<int> sample_indices;
	std::ifstream subset_idx("output\\indices_sample.out");
	size_t idx_S;
	while (subset_idx >> idx_S) {
		sample_indices.push_back(idx_S);
	}
	std::sort(sample_indices.begin(), sample_indices.end());
	// Read Distance Matrices 
	Distance_matrix dist_X = Gudhi::read_lower_triangular_matrix_from_csv_file<Filtration_value>("output\\dist_X.out", ' ');
	// Distance_matrix dist_S = Gudhi::read_lower_triangular_matrix_from_csv_file<Filtration_value>("output\\dist_S.out", " ");
	// Restrict distance matrix to subset indices 
	Distance_matrix dist_S;
	for (int row_idx : sample_indices) {
		std::vector<double> row;
		for (int col_idx : sample_indices) {
			row.push_back(dist_X[col_idx][row_idx]); // triangular 
		}
		dist_S.push_back(row);
	}
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
	

	//auto edges_graph_S = boost::adaptors::transform(edges(graph_S), [&](auto&& edge) {
	//	return std::make_tuple(source(edge, graph_S),
	//	target(edge, graph_S),
	//	get(Gudhi::edge_filtration_t(), graph_S, edge));
	//	});
	//auto edges_graph_X = boost::adaptors::transform(edges(graph_X), [&](auto&& edge) {
	//	return std::make_tuple(source(edge, graph_X),
	//	target(edge, graph_X),
	//	get(Gudhi::edge_filtration_t(), graph_X, edge));
	//	});
	//std::vector<Filtered_edge> edges_list_S(edges_graph_S.begin(), edges_graph_S.end());
	//std::vector<Filtered_edge> edges_list_X(edges_graph_X.begin(), edges_graph_X.end());
	//// Compute barcodes and matrix associated to persistence morphism
	//Barcodes_dim S_barcode, X_barcode;
	//Reps_dim S_reps, S_reps_im, X_reps;
	//Matrix_dim pm_matrix;
	//pairs_and_matrix_VR(
	//	point_X.size(), point_S.size(),
	//	edges_list_X, edges_list_S,
	//	threshold, dim_max, edge_collapse_iter_nb,
	//	S_barcode, S_reps, S_reps_im,
	//	X_barcode, X_reps,
	//	pm_matrix
	//);
	return 0;
}
	