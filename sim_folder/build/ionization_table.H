#ifndef IONIZATION_TABLE_H_
#define IONIZATION_TABLE_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include <map>

const int tot_num_spec = 14 * 2 ; // Total number of species (ionization stages) * 2 (step ionization)

constexpr int num_atom_species = 4;

constexpr amrex::Real X_min = 1e-20; // Threshold for considering a species to be present

// Helper lambda to map species index to atomic species
AMREX_GPU_HOST_DEVICE
inline int get_atom_index(int mu) {
    if (mu <= 1 && mu >= 0) return 0;      // Hydrogen
    if (mu <= 7 && mu > 1) return 1;      // Nitrogen
    if (mu <= 10 && mu > 7) return 2;     // Helium
    if (mu <= 14 && mu > 10) return 3;    // Argon
    return 0;
};

static constexpr int n_step = 2; // number of ionization steps per species
// Units: converted from cm^3/s to m^3/s (× 1e-6)
// Fit from Broks et  al. 2023 (https://arxiv.org/pdf/2305.16779) for H0 and H1
// Arrhenius fit R(Te) ∝ A_I * Te **q_I * exp(-eps_I / Te)
// eps_I: ionization energy
AMREX_GPU_HOST_DEVICE inline amrex::Real A_I[tot_num_spec][n_step] = {
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
    {0.0, 0.0} //Ar3
};

AMREX_GPU_HOST_DEVICE inline amrex::Real q_I[tot_num_spec][n_step] = {
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
    {0.0, 0.0} //Ar3
};

AMREX_GPU_HOST_DEVICE inline amrex::Real eps_I[tot_num_spec][n_step] = {
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
    {0.0, 0.0} //Ar3
};

#endif // #ifndef IONIZATION_TABLE_H_
