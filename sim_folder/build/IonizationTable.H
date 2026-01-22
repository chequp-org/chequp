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

const int dimS = 31 * 2 ; // Total number of species (ionization stages) * 2 (step ionization)

const std::vector<std::string> ion_names = {"H", "N", "He", "Ar"};

namespace Acoeff {

    // Data are taken from fit on (ADAS fit) data for H, He, N, Ar
    // See /theory/impact_ionization_data/

    static constexpr int n_step = 2; // number of ionization steps per species

    // ------------------------------------------------------------------
    // A coefficients 
    // ------------------------------------------------------------------
    // Units: converted from cm^3/s to m^3/s (× 1e-6)
    static constexpr amrex::Real A_I[dimS][n_step] = {
        // Hydrogen (H)
        {7.1e-9, 12.4e-9},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Nitrogen (N)
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Helium (He)
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Argon (Ar)
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0}
    };

    // ------------------------------------------------------------------
    // q coefficients
    // ------------------------------------------------------------------
    static constexpr amrex::Real q_I[dimS][n_step] = {
        // Hydrogen (H)
        {0.4, 0.3},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Nitrogen (N)
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Helium (He)
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Argon (Ar)
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0}
    };

    // ------------------------------------------------------------------
    // Ionization energies (eV)
    // ------------------------------------------------------------------
    static constexpr amrex::Real eps_I[dimS][n_step] = {
        // Hydrogen (H)
        {13.6, 10.6},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Nitrogen (N)
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Helium (He)
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0},

        // Argon (Ar)
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        // Additional dummy value for indexing issue
        {0.0, 0.0}
    };

} // namespace Acoeff


#endif // #ifndef IONIZATION_TABLE_H_