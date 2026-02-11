#ifndef IONIZATION_TABLE_H_
#define IONIZATION_TABLE_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include <map>


// The following array defines the ionization stages indices for each species, which are fix and defined in species.net
const std::vector<std::vector<int>> index_species = {
    {0, 1},                     // H
    {2, 3, 4, 5, 6, 7},         // N
    {8, 9, 10},                 // He
    {11, 12, 13, 14} // Ar
};

const int tot_num_spec = 14 * 2 ; // Total number of species (ionization stages) * 2 (step ionization)

const std::vector<std::string> ion_names = {"H", "N", "He", "Ar"};
const std::size_t num_atom_species = ion_names.size();  // runtime constant, safe
const std::vector<int> start_idx = {1, 3, 9, 12}; // starting index for each atomic species
/**
 * @brief Populates a vector with indices of ionization species based on problem configuration.
 *
 * This function parses the species string defined in `problem::species` and retrieves
 * the corresponding indices from the ionization table. The resulting indices are sorted
 * in ascending order.
 *
 * @param[out] idx_spec A reference to a vector of integers that will be cleared and filled
 *                       with the sorted indices of the species. The indices correspond to
 *                       positions in the ionization table for each requested species.
 *
 * @details
 * - The function expects species names to be whitespace-separated in `problem::species`
 * - Each species name is matched against entries in the `ion_names` vector
 * - When a match is found, all indices associated with that species (from `index_species`)
 *   are added to `idx_spec`
 * - The resulting vector is sorted in ascending order before return
 * - Species not found in `ion_names` are silently skipped
 *
 * @see ion_names
 * @see index_species
 * @see problem::species
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_idx_spec(std::vector<int>& idx_spec){
    idx_spec.clear();
    std::istringstream iss(problem::species); // in the input file, species are given as a string
    std::string sp;

    while (iss >> sp) {
        auto it = std::find(ion_names.begin(), ion_names.end(), sp);
        if (it != ion_names.end()) { // if the species is found
            int idx = std::distance(ion_names.begin(), it);
            idx_spec.insert(idx_spec.end(),
                            index_species[idx].begin(),
                            index_species[idx].end());
        }
    }
    std::sort(idx_spec.begin(), idx_spec.end()); // sort the indices
}

namespace Acoeff {

    // Data are taken from fit on (ADAS fit) data for H, He, N, Ar
    // See /theory/impact_ionization_data/

    static constexpr int n_step = 2; // number of ionization steps per species

    // ------------------------------------------------------------------
    // A coefficients 
    // ------------------------------------------------------------------
    // Units: converted from cm^3/s to m^3/s (× 1e-6)
    static constexpr amrex::Real A_I[tot_num_spec][n_step] = {
        // Hydrogen (H)
        {7.1e-9, 12.4e-9}, //H0
        {0.0, 0.0}, //H1

        // Nitrogen (N)
        {0.0, 0.0}, //N0
        {0.0, 0.0}, //N1
        {0.0, 0.0}, //N2
        {0.0, 0.0}, //N3
        {0.0, 0.0}, //N4
        {0.0, 0.0}, //N5
    
        // Helium (He)
        {0.0, 0.0}, //He0   
        {0.0, 0.0}, //He1
        {0.0, 0.0}, //He2

        // Argon (Ar)
        {0.0, 0.0}, //Ar0
        {0.0, 0.0}, //Ar1
        {0.0, 0.0}, //Ar2
        {0.0, 0.0}, //Ar3
        {0.0, 0.0}  //Ar4
    };

    // ------------------------------------------------------------------
    // q coefficients
    // ------------------------------------------------------------------
    static constexpr amrex::Real q_I[tot_num_spec][n_step] = {
        // Hydrogen (H)
        {0.4, 0.3}, //H0
        {0.0, 0.0}, //H1

        // Nitrogen (N)
        {0.0, 0.0}, //N0
        {0.0, 0.0}, //N1
        {0.0, 0.0}, //N2
        {0.0, 0.0}, //N3
        {0.0, 0.0}, //N4
        {0.0, 0.0}, //N5

        // Helium (He)
        {0.0, 0.0}, //He0   
        {0.0, 0.0}, //He1
        {0.0, 0.0}, //He2

        // Argon (Ar)
        {0.0, 0.0}, //Ar0
        {0.0, 0.0}, //Ar1
        {0.0, 0.0}, //Ar2
        {0.0, 0.0}, //Ar3
        {0.0, 0.0}  //Ar4
    };

    // ------------------------------------------------------------------
    // Ionization energies (eV)
    // ------------------------------------------------------------------
    static constexpr amrex::Real eps_I[tot_num_spec][n_step] = {
        // Hydrogen (H)
        {13.6, 10.6}, //H0
        {0.0, 0.0}, //H1

        // Nitrogen (N)
        {0.0, 0.0}, //N0
        {0.0, 0.0}, //N1
        {0.0, 0.0}, //N2
        {0.0, 0.0}, //N3
        {0.0, 0.0}, //N4
        {0.0, 0.0}, //N5

        // Helium (He)
        {0.0, 0.0}, //He0   
        {0.0, 0.0}, //He1
        {0.0, 0.0}, //He2

        // Argon (Ar)
        {0.0, 0.0}, //Ar0
        {0.0, 0.0}, //Ar1
        {0.0, 0.0}, //Ar2
        {0.0, 0.0}, //Ar3
        {0.0, 0.0}  //Ar4
    };
}


#endif // #ifndef IONIZATION_TABLE_H_