#ifndef IONIZATION_TABLE_H_
#define IONIZATION_TABLE_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include <map>



const int tot_num_spec = 14 * 2 ; // Total number of species (ionization stages) * 2 (step ionization)

constexpr std::size_t num_atom_species = 4;

constexpr std::array<int, num_atom_species+1> start_idx = {1, 3, 9, 11, 13}; // Starting index of each atomic species in the species list (with shifted index)

namespace Acoeff {

    // Data are taken from fit on (ADAS fit) data for H, He, N, Ar
    // See /theory/impact_ionization_data/

    static constexpr int n_step = 2; // number of ionization steps per species

    // ------------------------------------------------------------------
    // A coefficients 
    // ------------------------------------------------------------------
    // Units: converted from cm^3/s to m^3/s (× 1e-6)
    __host__ __device__ static constexpr amrex::Real A_I[tot_num_spec][n_step] = {
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
        {3.20e-14, 5.710e-15}, //Ar0
        {0.0, 0.0}, //Ar1
        {0.0, 0.0}, //Ar2
        {0.0, 0.0} //Ar3
    };

    // ------------------------------------------------------------------
    // q coefficients
    // ------------------------------------------------------------------
    __host__ __device__ static constexpr amrex::Real q_I[tot_num_spec][n_step] = {
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
        {0.404, 0.490}, //Ar0
        {0.0, 0.0}, //Ar1
        {0.0, 0.0}, //Ar2
        {0.0, 0.0} //Ar3
    };

    // ------------------------------------------------------------------
    // Ionization energies (eV)
    // ------------------------------------------------------------------
    __host__ __device__ static constexpr amrex::Real eps_I[tot_num_spec][n_step] = {
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
        {15.76, 11.55}, //Ar0
        {0.0, 0.0}, //Ar1
        {0.0, 0.0}, //Ar2
        {0.0, 0.0} //Ar3
    };
}


#endif // #ifndef IONIZATION_TABLE_H_