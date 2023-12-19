// Script for sorting barcodes, representatives and associated matrices 
// IN startpoint and endpont orders respectively

#include <vector>

// GUDHI modules
#include <gudhi/Simplex_tree.h>
// PHAT modules needed
#include <phat/boundary_matrix.h>
#include <phat/representations/default_representations.h>
#include <phat/helpers/misc.h>

// GUDHI types
using Simplex_tree_options = Gudhi::Simplex_tree_options_full_featured;
using Simplex_tree = Gudhi::Simplex_tree<Simplex_tree_options>;
using Simplex_key = Simplex_tree_options::Simplex_key;
using Filtration_value = Simplex_tree::Filtration_value;
// PHAT types
using Phat_boundary_matrix = phat::boundary_matrix<phat::vector_vector>;
using Phat_index = phat::index;
using Phat_column = std::vector<Phat_index>;
// Custom Relevant Types
using Interval = std::pair<Filtration_value, Filtration_value>;

template <typename CR>
void sort_startpoint(
	std::vector<Interval>& barcode,
	std::vector<CR>& cycles,
	std::vector<CR>& cycles_im,
	std::vector<std::vector<Phat_index>>& pm_matrix)
{
	// Create a vector to store indices to sort
	std::vector<size_t> sort_indices;
	for (size_t idx = 0; idx < barcode.size(); idx++) {
		sort_indices.push_back(idx);
	}
	std::cout << "sortstart: created indices" << std::endl;
	// Sort indices following the startpoint order
	std::sort(
		sort_indices.begin(), sort_indices.end(),
		[&barcode](size_t& i, size_t& j) {
			return (barcode[i].first < barcode[j].first) || (
				(barcode[i].first == barcode[j].first) && (barcode[i].second < barcode[j].second)
				);
		}
	);
	std::cout << "sortstart: sorted indices" << std::endl;
	// Sort the cycle representatives from the barcode and the image
	// Also, sort the columns from the persistence morphism matrix
	std::vector<Interval> sorted_barcode;
	std::vector<CR> sorted_cycle;
	std::vector<CR> sorted_cycle_im;
	std::vector<std::vector<Phat_index>> sorted_matrix;
	for (size_t idx : sort_indices) {
		std::cout << idx << " ";
		std::cout << barcode.size() << " " << cycles.size() << " " << cycles_im.size() << " " << pm_matrix.size() << std::endl;
		sorted_barcode.push_back(barcode.at(idx));
		sorted_cycle.push_back(cycles.at(idx));
		sorted_cycle_im.push_back(cycles_im.at(idx));
		sorted_matrix.push_back(pm_matrix.at(idx));

	}
	std::cout << "sortstart: sorted content" << std::endl;
	barcode = sorted_barcode;
	cycles = sorted_cycle;
	cycles_im = sorted_cycle_im;
	pm_matrix = sorted_matrix;
}

template <typename CR>
void sort_endpoint(
	std::vector<Interval>& barcode,
	std::vector<CR>& cycles,
	std::vector<std::vector<Phat_index>>& pm_matrix)
{
	// Create a vector to store indices to sort in endpoint order
	std::vector<size_t> sort_indices;
	for (size_t idx = 0; idx < barcode.size(); idx++) {
		sort_indices.push_back(idx);
	}
	// Sort indices following the endpoint order on barcodes
	std::sort(
		sort_indices.begin(), sort_indices.end(),
		[&barcode](size_t& i, size_t& j) {
			return (barcode[i].second < barcode[j].second) || (
				(barcode[i].second == barcode[j].second) && (barcode[i].first < barcode[j].first)
				);
		}
	);
	// Sort barcode and cycle representatives
	std::vector<Interval> sorted_barcode;
	std::vector<CR> sorted_cycle;
	for (size_t idx : sort_indices) {
		sorted_barcode.push_back(barcode[idx]);
		sorted_cycle.push_back(cycles[idx]);
	}
	barcode = sorted_barcode;
	cycles = sorted_cycle;
	// Now proceed to sort rows of pm_matrix 
	std::vector<std::vector<Phat_index>> sorted_matrix;
	// First create a handy unordered map
	std::unordered_map<Phat_index, Phat_index> sort_indices_inv;
	size_t index_count = 0;
	for (size_t index : sort_indices) {
		sort_indices_inv[index] = index_count;
		index_count++;
	}
	for (std::vector<Phat_index>& column : pm_matrix) {
		std::vector<Phat_index> new_col;
		for (Phat_index entry : column) {
			new_col.push_back(sort_indices_inv[entry]);
		}
		std::sort(new_col.begin(), new_col.end());
		sorted_matrix.push_back(new_col);
	}
	pm_matrix = sorted_matrix;
}

// Write instances of template functions
// Representatives in 0 dimension are stored using phat columns
template void sort_startpoint(
	std::vector<Interval>& barcode,
	std::vector<Phat_column>& cycles,
	std::vector<Phat_column>& cycles_im,
	std::vector<std::vector<Phat_index>>& pm_matrix
);

template void sort_endpoint(
	std::vector<Interval>& barcode,
	std::vector<Phat_column>& cycles,
	std::vector<std::vector<Phat_index>>& pm_matrix
);

// Representatives in 1 dimension are stored using a list of pairs of vertices
template void sort_startpoint(
	std::vector<Interval>& barcode,
	std::vector<Cycle_rep_1>& cycles,
	std::vector<Cycle_rep_1>& cycles_im,
	std::vector<std::vector<Phat_index>>& pm_matrix
);

template void sort_endpoint(
	std::vector<Interval>& barcode,
	std::vector<Cycle_rep_1>& cycles,
	std::vector<std::vector<Phat_index>>& pm_matrix
);