/* Vietoris-Rips persistence morphism matrix.
* Copyright (C) Alvaro Torras Casas (2023)
*
* Consider a point cloud X together with a sample S.
* This program computes the associated matrix to the
* persistence morphisms: PH_k(VR(S))-->PH_k(VR(X))
* for all dimensions 0 <= k <= max_dim.
* It also computes the barcodes of the domain and codomain, as well
* as their cycle representatives.
* This is done combining two TDA libraries: GUDHI and PHAT
* In particular, the GUDHI library header file "Flag_complex_edge_collapser.h" modified by the author.
* as well as a PHAT repository modified by the author.
* BOOST is also used for type conversion between PHAT and GUDHI variables.
*
* This file follows the types and conventions of the GUDHI library whenever possible.
*/
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
// BOOST handling types between PHAT and BOOST
#include <boost/numeric/conversion/cast.hpp>

// Ignore bars of length < _tolerance
const double _tolerance=1e-12;
// GUDHI types
using Simplex_tree_options = Gudhi::Simplex_tree_options_full_featured;
using Simplex_tree = Gudhi::Simplex_tree<Simplex_tree_options>;
using Simplex_key = Simplex_tree_options::Simplex_key;
using Filtration_value = Simplex_tree::Filtration_value;
using Vertex_handle = Simplex_tree::Vertex_handle;
using Simplex_handle =  Simplex_tree::Simplex_handle;
using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp>;
// Some handy additional types
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
using Cycle_rep_1 = std::vector<Vertex_pair>;
using Reps_dim = std::tuple<std::vector<Phat_column>, std::vector<Cycle_rep_1>>;
using Matrix_dim = std::unordered_map<int, std::vector<std::vector<Phat_index>>>;

// Avoids repeating initializaiton of Vietoris-Rips complex twice
void init_VR(
    int dim_max, size_t num_points,
    std::vector<Filtered_edge>& edges_list, Simplex_tree& stree
);
// Avoids repeating initializaiton of Vietoris-Rips complex twice
void init_phat_boundary(Simplex_tree& stree, Phat_boundary_matrix& diff_mat);
// Takes out trivial intervals and organizes them by dimension
void barcodes_and_reps_dim(
    Simplex_tree& stree,
    Phat_boundary_matrix& diff_mat,
    int dim_max,
    phat::persistence_pairs& ph_pairs_all,
    std::unordered_map<int, Phat_column>& cycle_cols_dim,
    Barcodes_dim& barcodes_d,
    Reps_dim& reps_d
);

void init_VR(int dim_max, size_t num_points, std::vector<Filtered_edge>& edges_list, Simplex_tree& stree){
    for (Vertex_handle vertex = 0; static_cast<std::size_t>(vertex) < num_points; vertex++) {
      // insert the vertex with a 0. filtration value just like a Rips
      stree.insert_simplex({vertex}, 0.);
    }
    for (auto filtered_edge : edges_list) {
      stree.insert_simplex({std::get<0>(filtered_edge), std::get<1>(filtered_edge)}, std::get<2>(filtered_edge));
    }
    // Add higher dimensional (>=2) simplices
    stree.expansion(dim_max);
    // Print out some info about collapse and dimension
    #ifdef PERMOVEC_SIZES
        std::cout << "The subcomplex contains " << stree.num_simplices() << " simplices  after collapse. \n";
        std::cout << "   and has dimension " << stree.dimension() << " \n";
    #endif
    // Sort the simplices in the order of the filtration
    stree.initialize_filtration();
    // Initialize simplex keys (needed for PHAT persistent homology computation)
    Simplex_key spx_key = 0;
    for (auto f_spx : stree.filtration_simplex_range()) {
        stree.assign_key(f_spx, spx_key++);
    }
}

void pairs_and_matrix_VR(
    size_t size_X, size_t size_S,
    std::vector<size_t> indices_S, 
    std::vector<Filtered_edge>& edges_list_X,
    std::vector<Filtered_edge>& edges_list_S,
    Filtration_value threshold,
    int dim_max,
    int edge_collapse_iter_nb, // number of iterations of edge collapser
    Barcodes_dim& S_barcode, Reps_dim& S_reps, Reps_dim& S_reps_im,
    Barcodes_dim& X_barcode, Reps_dim& X_reps,
    Matrix_dim& pm_matrix_dim
){
  #ifdef PERMOVEC_SIZES
    std::cout << "Welcome to PerMoVEC!" << std::endl;
  #endif
  //-----------------------------------------------------------------------------------------
  // Collapse subcomplex and compute VR
  //-----------------------------------------------------------------------------------------
  for (int iter = 0; iter < edge_collapse_iter_nb; iter++) {
    auto remaining_edges_S = Gudhi::collapse::flag_complex_collapse_edges(edges_list_S);
    edges_list_S = std::move(remaining_edges_S);
    remaining_edges_S.clear();
  }
  Simplex_tree stree_S;
  init_VR(dim_max, size_S, edges_list_S, stree_S);
  //-----------------------------------------------------------------------------------------
  // PH of subcomplex and cycle representatives using PHAT
  //-----------------------------------------------------------------------------------------
  Phat_boundary_matrix diff_mat_S;
  std::vector<Phat_column> cycle_reps_null; //ignore
  init_phat_boundary(stree_S, diff_mat_S);
  // Reduce differential matrix and obtain pairs from PHAT
  phat::persistence_pairs ph_pairs_all_S;
  // compute persistent homology by means of the standard reduction
  phat::compute_persistence_pairs<phat::standard_reduction>(ph_pairs_all_S, diff_mat_S);
  // Clear trivial pairs out and store into ph_pairs
  std::unordered_map<int, Phat_column> S_cycle_cols_dim; // not relevant for subcomplex
  barcodes_and_reps_dim(stree_S, diff_mat_S, dim_max, ph_pairs_all_S, S_cycle_cols_dim, S_barcode, S_reps);
  // Get vector of births of 1 cycles in subset
  std::vector<Filtration_value> births_1;
  std::vector<Interval>& barcode_1 = S_barcode[1];
  for (Interval bar : barcode_1){
      births_1.push_back(bar.first);
  }
  //-----------------------------------------------------------------------------------------
  // Store list of edges from 1 dimensional cycles to send to "Flag_complex_edge_collapser"
  //-----------------------------------------------------------------------------------------
  std::vector<std::pair<Filtration_value, Cycle_rep_1>> cycles_image_1;
  // These cycles should be sorted because of the order followed by PHAT pairs
  std::vector<Cycle_rep_1>& reps_1 = std::get<1>(S_reps);
  auto cycle_it = reps_1.begin();
  auto birth_it = births_1.begin();
  assert(reps_1.size()==births_1.size());
  while(cycle_it < reps_1.end()) {
      // Compute image of cycle using indices_S
      Cycle_rep_1 im_cycle;
      for (Vertex_pair& edge : *cycle_it) {
          im_cycle.push_back({ indices_S[edge.first], indices_S[edge.second] });
      }
      cycles_image_1.push_back({*birth_it, im_cycle});
      birth_it++;
      cycle_it++;
  }
  //-----------------------------------------------------------------------------------------
  // Proceed to collapse large complex, collapsing embedded cycle representatives from subcomplex
  //-----------------------------------------------------------------------------------------
  std::vector<Filtered_edge> remaining_edges_X;
  for (int iter = 0; iter < edge_collapse_iter_nb; iter++) {
    auto remaining_edges_X = Gudhi::collapse::flag_complex_collapse_edges(edges_list_X, cycles_image_1);
    edges_list_X = std::move(remaining_edges_X);
    remaining_edges_X.clear();
  }
  // Store cycle images collapsed
  std::vector<Cycle_rep_1> reps_1_im;
  for(std::pair<Filtration_value, Cycle_rep_1>& cycle : cycles_image_1) {
      reps_1_im.push_back(cycle.second);
  }
  // We get the image of the 0 representatives directly using indices_S
  std::vector<Phat_column> reps_0_dim;
  for (Phat_column& rep : std::get<0>(S_reps)) {
      Phat_column im_rep;
      for (Phat_index idx : rep) {
          im_rep.push_back(indices_S[idx]);
      }
      reps_0_dim.push_back(im_rep);
  }
  // Store image representatives into a tuple
  S_reps_im = {reps_0_dim, reps_1_im};
  // Compute large VR and barcode
  Simplex_tree stree_X;
  init_VR(dim_max, size_X, edges_list_X, stree_X);
  //-----------------------------------------------------------------------------------------
  // PH of large complex and cycle representatives using PHAT
  //-----------------------------------------------------------------------------------------
  Phat_boundary_matrix diff_mat_X;
  init_phat_boundary(stree_X, diff_mat_X);
  // Reduce differential matrix and obtain pairs from PHAT
  phat::persistence_pairs ph_pairs_all_X;
  // compute persistent homology by means of the standard reduction
  phat::compute_persistence_pairs<phat::standard_reduction>(ph_pairs_all_X, diff_mat_X);
  // Clear trivial pairs out and store into ph_pairs
  std::unordered_map<int, Phat_column> X_cycle_cols_dim;
  barcodes_and_reps_dim(stree_X, diff_mat_X, dim_max, ph_pairs_all_X, X_cycle_cols_dim, X_barcode, X_reps);
  //-----------------------------------------------------------------------------------------
  // Compute Persistence Morphism Matrices (pm_matrix) on dimensions 0 and 1
  //-----------------------------------------------------------------------------------------
  Phat_boundary_matrix cycle_reduction_mat;
  cycle_reduction_mat.set_num_cols(
      diff_mat_X.get_num_cols() + S_barcode[0].size() + S_barcode[1].size()
  );
  // Fill in diff_mat_X columns
  for (Phat_index col_idx=0; col_idx < diff_mat_X.get_num_cols(); col_idx++){
      Phat_column temp_col;
      diff_mat_X.get_col(col_idx, temp_col);
      cycle_reduction_mat.set_col(col_idx, temp_col);
  }
  #ifdef DEBUG
    std::cout << "Cycle columns image: " << std::endl;
  #endif
  // Append image cycles in dimension 0
  Phat_index col_count=diff_mat_X.get_num_cols();
  for (Phat_column im_rep : std::get<0>(S_reps_im)) {
      cycle_reduction_mat.set_col(col_count, im_rep);
      col_count++;
  }
  // Append image cycles in dimension 1
  for(Cycle_rep_1 edges_rep : reps_1_im) {
      // Read current edge into a phat column
      // Use sets to take out repeated entries
      std::set<Phat_index> cycle_column_set;
      for (Vertex_pair& edge : edges_rep) {
          Phat_index new_entry = stree_X.key(stree_X.find({edge.first, edge.second}));
          if (cycle_column_set.count(new_entry)>0){
              cycle_column_set.erase(new_entry);
          } else {
              cycle_column_set.insert(new_entry);
          }
      }
      Phat_column cycle_column(cycle_column_set.begin(), cycle_column_set.end());
      // Phat assumes columns are sorted
      std::sort(cycle_column.begin(), cycle_column.end());
      cycle_reduction_mat.set_col(col_count, cycle_column);
      col_count++;
  }
  phat::persistence_pairs _ignore_pairs;
  phat::compute_persistence_pairs<phat::standard_reduction>(_ignore_pairs, cycle_reduction_mat);
  // Check that all image cycles have been reduced to 0
  #ifdef DEBUG
    std::cout << "Checking zero columns:" << std::endl;
  #endif
  for(Phat_index col_idx = diff_mat_X.get_num_cols(); col_idx<cycle_reduction_mat.get_num_cols(); col_idx++){
      assert(cycle_reduction_mat.is_empty(col_idx));
  }
  // Use preimages to deduce the persistence morphism matrices in dimensions 0 and 1
  col_count = diff_mat_X.get_num_cols();
  for (int dim = 0; dim < 2; dim++) {
      #ifdef DEBUG
        std::cout << dim << " PM_matrix:" << std::endl;
      #endif
      std::vector<Phat_index> X_cycle_cols = X_cycle_cols_dim[dim];
      std::vector<std::vector<Phat_index>> pm_matrix;
      auto int_S = S_barcode[dim].begin();
      for (Phat_index col_idx = 0; col_idx < S_barcode[dim].size(); col_idx++) {
          std::vector<Phat_index> preimage_col;
          cycle_reduction_mat.get_preimage(col_count, preimage_col);
          std::vector<Phat_index> cycle_coord;
          for (Phat_index entry : preimage_col) {
              auto entry_it = std::find(X_cycle_cols.begin(), X_cycle_cols.end(), entry);
              if (entry_it != X_cycle_cols.end()) {
                  size_t cycle_idx = entry_it - X_cycle_cols.begin();
                  // Check condition on intervals and store if necessary
                  if (X_barcode[dim][cycle_idx].second > int_S->first) {
                      if (X_barcode[dim][cycle_idx].second <= int_S->second) {
                          cycle_coord.push_back(cycle_idx);
                      }
                      else {
                          std::cout << "ERROR: Interval condition not satisfied" << std::endl;
                          exit(1);
                      }
                  }
              }
          }
          #ifdef DEBUG
            for (Phat_index entry : cycle_coord) {
                std::cout << entry << ", ";
            }
            std::cout << std::endl;
          #endif
          pm_matrix.push_back(cycle_coord);
          int_S++;
          col_count++;
      }
      #ifdef DEBUG
        std::cout << "PM_matrix end" << std::endl;
      #endif
      pm_matrix_dim[dim] = pm_matrix;
  } // for dim=0,1
} // pairs_and_matrix_VR


void init_phat_boundary(Simplex_tree& stree, Phat_boundary_matrix& diff_mat) {
    // Initialize differential PHAT boundary matrix
    diff_mat.set_num_cols(stree.num_simplices());
    // Fill differential matrix
    Phat_index col_idx = 0;
    for (auto f_spx : stree.filtration_simplex_range()) {
  		phat::dimension curr_dim;
          curr_dim = boost::numeric_cast<phat::dimension>(stree.dimension(f_spx));
  		diff_mat.set_dim(col_idx, curr_dim);
  		// fill in boundary column
  		Phat_column temp_col;
  		for (auto b_spx : stree.boundary_simplex_range(f_spx)) {
  			Phat_index column_entry;
            column_entry = boost::numeric_cast<Phat_index>(stree.key(b_spx));
  			temp_col.push_back(column_entry);
  		}
  		std::sort(temp_col.begin(), temp_col.end());
  		diff_mat.set_col(col_idx, temp_col);
  		col_idx++;
  	}
}

void sort_startpoint(Simplex_tree& stree, std::vector<Phat_idx_pair>& barcode) {
    std::sort(
        barcode.begin(), barcode.end(),
        [&stree](Phat_idx_pair& a, Phat_idx_pair& b){
            return (stree.filtration(stree.simplex(a.first)) < stree.filtration(stree.simplex(b.first))||(
                (stree.filtration(stree.simplex(a.first))==stree.filtration(stree.simplex(b.first))) && stree.filtration(stree.simplex(a.second)) < stree.filtration(stree.simplex(b.second))
            ));
        }
    );
}

void barcodes_and_reps_dim(
    Simplex_tree& stree,
    Phat_boundary_matrix& diff_mat,
    int dim_max,
    phat::persistence_pairs& ph_pairs_all,
    std::unordered_map<int, Phat_column>& cycle_cols_dim,
    Barcodes_dim& barcodes_d,
    Reps_dim& reps_d
){
    std::unordered_map<int, std::vector<Phat_idx_pair>> ph_pairs_dim;
    // keep non-trivial bars and print barcode on output file
    for (phat::index i = 0; i < ph_pairs_all.get_num_pairs(); i++) {
        Phat_index pos_idx = ph_pairs_all.get_pair( i ).first;
        Phat_index neg_idx = ph_pairs_all.get_pair( i ).second;
        Simplex_handle pos_spx = stree.simplex(pos_idx);
        Simplex_handle neg_spx = stree.simplex(neg_idx);
        Filtration_value birth = stree.filtration(pos_spx);
        Filtration_value death = stree.filtration(neg_spx);
        int dim = stree.dimension(pos_spx);
        if (abs(death-birth)>_tolerance) {
            Phat_idx_pair new_bar{pos_idx, neg_idx};
            ph_pairs_dim[dim].push_back(new_bar);
            // store column idx containing cycle representative (by Phat_index expression)
            cycle_cols_dim[dim].push_back(neg_idx);
        }
    }
    // Store barcodes by dimension
    barcodes_d.clear();
    for (int dim=0; dim < dim_max; dim++) {
        if (ph_pairs_dim.find(dim)!=ph_pairs_dim.end()){
          std::vector<Interval> barcode;
          // sort_startpoint(stree, ph_pairs_dim[dim]);
          for (Phat_idx_pair bar_el : ph_pairs_dim[dim]){
              Filtration_value birth = stree.filtration(stree.simplex(bar_el.first));
              Filtration_value death = stree.filtration(stree.simplex(bar_el.second));
              barcode.push_back({birth, death});
          }
          barcodes_d[dim]=barcode;
        }
    }
    // Get cycle representatives for all bars by dimensions (dim 0 and 1)
    // Dim 0 should be vector with phat index entries
    std::vector<Phat_column> reps_0;
    for (Phat_idx_pair bar_el : ph_pairs_dim[0]) {
        Phat_column rep;
        diff_mat.get_col(bar_el.second, rep);
        reps_0.push_back(rep);
    }
    // Dim 1 should be vector of vectors of edges
    std::vector<Cycle_rep_1> reps_1;
    if((dim_max>0) && (ph_pairs_dim[1].size()>0)){
        for (Phat_idx_pair bar_el : ph_pairs_dim[1]) {
            // Read cycle representative column
            Phat_column rep;
            diff_mat.get_col(bar_el.second, rep);
            // Turn column entries into list of edges
            Cycle_rep_1 rep_edges;
            for (Phat_index idx_entry : rep) {
                int count=0;
                Vertex_pair edge_entry;
                for(Vertex_handle vert : stree.simplex_vertex_range(stree.simplex(idx_entry))){
                    if (count==0){
                        edge_entry.first = vert;
                    } else if (count==1) {
                        edge_entry.second = vert;
                    } else {
                        assert(false);
                    }
                    count++;
                }
                rep_edges.push_back(edge_entry);
            }
            assert(rep.size()==rep_edges.size());
            reps_1.push_back(rep_edges);
        }
    }
    // Store cycle representatives
    reps_d = {reps_0, reps_1};
}

void store_permovec_output(
    Barcodes_dim& S_barcode,
    Reps_dim& S_reps,
    Reps_dim& S_reps_im,
    Barcodes_dim& X_barcode,
    Reps_dim& X_reps,
    Matrix_dim& pm_matrix_dim,
    std::vector<size_t>& sample_indices,
    bool& print_0_pm,
    std::string& folder_io
) {
    for (int dim = 0; dim < 2; dim++) {
        // Save domain barcode
        std::string Sbarcode_f = folder_io + "/S_barcode_" + std::to_string(dim) + ".out";
        std::ofstream out_S_bar(Sbarcode_f);
        for (Interval bar : S_barcode[dim]) {
            out_S_bar << bar.first << " " << bar.second << std::endl;
        }
        out_S_bar.close();
        // Save codomain barcode
        #ifdef DEBUG
            std::cout << "Length X_barcode[" << dim << "]: " << X_barcode[dim].size() << std::endl;
        #endif
        std::string Xbarcode_f = folder_io + "/X_barcode_" + std::to_string(dim) + ".out";
        std::ofstream out_X_bar(Xbarcode_f);
        for (Interval bar : X_barcode[dim]) {
            out_X_bar << bar.first << " " << bar.second << std::endl;
        }
        out_X_bar.close();
        // In dimension 0, print persistence morphism matrix only if required
        if ((!print_0_pm) && (dim == 0)) {
            continue;
        }
        // Save persistence morphism matrix
        std::string pm_matrix_f = folder_io + "/pm_matrix_" + std::to_string(dim) + ".out";
        std::ofstream out_pm_matrix(pm_matrix_f);
        for (std::vector<Phat_index>& column : pm_matrix_dim[dim]) {
            for (Phat_index entry : column) {
                out_pm_matrix << entry << " ";
            }
            out_pm_matrix << std::endl;
        }
        out_pm_matrix.close();

    } // Printed barcodes and matrix
    // ------------------------------------------------------------------------
    // Print 0 dim reps
    // ------------------------------------------------------------------------
    std::string Xreps_f = folder_io + "/X_reps_0.out";
    std::ofstream out_X_reps(Xreps_f);
    for (auto& cycle_rep : std::get<0>(X_reps)) {
        for (Phat_index idx : cycle_rep) {
            out_X_reps << idx << " ";
        }
        out_X_reps << std::endl;
    }
    out_X_reps.close();
    // Save domain representatives
    std::string Sreps_f = folder_io + "/S_reps_0.out";
    std::ofstream out_S_reps(Sreps_f);
    for (auto& cycle_rep : std::get<0>(S_reps)) {
        for (Phat_index idx : cycle_rep) {
            out_S_reps << sample_indices[idx] << " ";
        }
        out_S_reps << std::endl;
    }
    out_S_reps.close();
    // Save image reps
    std::string Sreps_im_f = folder_io + "/S_reps_im_0.out";
    std::ofstream out_S_reps_im(Sreps_im_f);
    for (auto& cycle_rep : std::get<0>(S_reps_im)) {
        for (Phat_index idx : cycle_rep) {
            out_S_reps_im << idx << " ";
        }
        out_S_reps_im << std::endl;
    }
    out_S_reps_im.close();
    // ------------------------------------------------------------------------
    // Print 1 dim reps
    // ------------------------------------------------------------------------
    Xreps_f = folder_io + "/X_reps_1.out";
    out_X_reps.open(Xreps_f);
    for (auto& cycle_rep : std::get<1>(X_reps)) {
        for (Vertex_pair edge : cycle_rep) {
            out_X_reps << edge.first << " " << edge.second << " ";
        }
        out_X_reps << std::endl;
    }
    out_X_reps.close();
    // Save domain representatives
    Sreps_f = folder_io + "/S_reps_1.out";
    out_S_reps.open(Sreps_f);
    for (auto& cycle_rep : std::get<1>(S_reps)) {
        for (Vertex_pair edge : cycle_rep) {
            out_S_reps << sample_indices[edge.first] << " " << sample_indices[edge.second] << " ";
        }
        out_S_reps << std::endl;
    }
    out_S_reps.close();
    // Save image reps
    Sreps_im_f = folder_io + "/S_reps_im_1.out";
    out_S_reps_im.open(Sreps_im_f);
    for (auto& cycle_rep : std::get<1>(S_reps_im)) {
        for (Vertex_pair edge : cycle_rep) {
            out_S_reps_im << edge.first << " " << edge.second << " ";
        }
        out_S_reps_im << std::endl;
    }
    out_S_reps_im.close();
} // end print_permovec_output
