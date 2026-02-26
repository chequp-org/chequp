#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <ionization_table.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;

    // Extract density and electron temperature
    Real rho = state(i,j,k,URHO);
    const amrex::Real T_e = state(i,j,k,UTEMP);

    // Initialize species mass fractions and identify active species
    // Initialize species mass fractions
    Real X_stage[NumSpec];
    
    // Create mask to identify active species
    // This helps reduce computational cost by only processing species with significant populations
    bool active_mask[NumSpec] = {false};
    int active_spec[NumSpec];
    int n_active = 0;

    // Extract mass fractions and mark active species
    for (int mu = 0; mu < NumSpec; mu++) {
        // Get mass fraction for species mu from conserved state variable
        X_stage[mu] = state(i,j,k,UFS+mu) / rho;
        
        // Set very small mass fractions to zero to avoid numerical issues
        if (X_stage[mu] < X_min) {
            X_stage[mu] = 0.0;
        }
        
        // Mark species and neighbors as active if mass fraction is above threshold
        // Include neighbors to account for ionization/recombination between adjacent species
        if (X_stage[mu] > X_min) {
            active_mask[mu] = true;
            // Also activate the previous species (neutral form) for ionization processes
            if (mu > 0) active_mask[mu-1] = true;
        }
        
        // Build list of active species indices for later iteration
        // This avoids looping over all species when only a few are significant
        if (active_mask[mu]) {
            active_spec[n_active++] = mu;
        }
    }

    // Extract energy and ion fraction
    Real f = state(i,j,k,UFX) / rho;
    Real e = state(i,j,k,UEINT) / rho;

    // Calculate electron fraction from active species
    Real X_e = 0.0;
    for (int i = 0; i < n_active; i++) {
        int mu = active_spec[i];
        X_e += zion[mu] * X_stage[mu];
    }

    // Calculate ionization source terms
    Real e_ion = 0.0;
    for (int alpha = 0; alpha < NumSpec; alpha++) {
        Real sum = 0.0;
        for (int beta = 0; beta < NumSpec; beta++) {
            Real val = 0.0;

            // Electron-impact ionization collision frequency
            Real coll_I = 0.0;
            for (int step = 0; step < n_step; step++) {
                coll_I += A_I[beta][step] *
                    std::pow(C::k_B*T_e / C::ev2erg, q_I[beta][step]) *
                    std::exp(-eps_I[beta][step] * C::ev2erg / (C::k_B*T_e)) *
                    X_stage[beta] * X_e;
            }

            if (alpha == beta) {
                // Species being ionized
                val = -coll_I;
                e_ion += val * (rho * rho / C::m_p) * (eps_I[beta][0]*C::ev2erg / C::m_p);
            }
            else if (alpha == beta + 1) {
                // Species created by ionization of beta
                val = coll_I;
            }
            sum += val * (rho * rho / C::m_p);
        }
        src(i,j,k,UFS+alpha) = sum;
    }

    // Update energy source terms
    src(i,j,k,UEDEN) = e_ion;
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // Calculate electron and neutral number densities (SI units)
    amrex::Real n_e = 0.0;
    amrex::Real n_n_species[num_atom_species] = {};

    for (int i = 0; i < n_active; i++) {
        int mu = active_spec[i];
        n_e += zion[mu] * X_stage[mu] * rho / C::m_u * 1.e6;
        if (zion[mu] == 0) {
            int idx_atom = get_atom_index(mu);
            n_n_species[idx_atom] += X_stage[mu] * rho / C::m_u * 1.e6;
        }
    }

    // Compute collision frequencies
    amrex::Real v_ei;
    amrex::Real v_en[NumSpec];
    const amrex::Real T_h = T_e;
    get_collision_frequencies(n_e, n_n_species, T_e, T_h, v_ei, v_en);

    // Calculate momentum transfer collision frequencies
    amrex::Real v_e_E[NumSpec] = {};
    for (int i = 0; i < n_active; i++) {
        int mu = active_spec[i];
        v_e_E[mu] = 2 * SI::m_e / mass_vec[get_atom_index(mu)] *
                    (std::pow(zion[mu], 4) * v_ei + v_en[mu]);
    }

    // Calculate energy transfer source term
    amrex::Real Sc = 0.0;
    for (int im = 0; im < n_active; im++) {
        int mu = active_spec[im];
        Sc += rho * v_e_E[mu] * (1 - (1 + X_stage[mu]) * f);
    }

    src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc;
}

#endif