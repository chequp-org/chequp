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

    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state(i,j,k,UTEMP);

    // Initialize species mass fractions and identify active species
    // Initialize species mass fractions
    Real X_stage[NumSpec];
    
    // Create mask to identify active species
    bool active_mask[NumSpec] = {false};
    int active_spec[NumSpec];
    int n_active = 0;

    // Extract mass fractions and mark active species
    for (int mu = 0; mu < NumSpec; mu++) {
        X_stage[mu] = state(i,j,k,UFS+mu) / rho;
        
        // Mark species and neighbors as active if mass fraction is above threshold
        if (X_stage[mu] > X_min) {
            active_mask[mu] = true;
            if (mu > 0) active_mask[mu-1] = true;
            if (mu < NumSpec-1) active_mask[mu+1] = true;
        }
        
        // Build list of active species indices
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

    // Initialize sources
    Real e_ion = 0.0;

    // Loop over ionizing species only once
    for (int n = 0; n < n_active; n++) {
        // Access active index
        int beta = active_spec[n];
        // Reaction rate
        Real coll_I = 0.0;
        for (int step = 0; step < n_step; step++) {
            coll_I += A_I[beta][step] *
                std::pow(C::k_B*T_e / C::ev2erg, q_I[beta][step]) *
                std::exp(-eps_I[beta][step] * C::ev2erg / (C::k_B*T_e)) *
                X_stage[beta] * X_e;
        }

        Real coeff = coll_I * (rho * rho / C::m_p);

        // Loss
        src(i,j,k,UFS+beta) -= coeff;

        // Gain
        int beta_next = beta + 1;
        if (beta_next < NumSpec) {
            src(i,j,k,UFS+beta_next) += coeff;
        }

        // Energy lose by ionization
        e_ion -= coll_I * (rho * rho / C::m_p) *
                (eps_I[beta][0] * C::ev2erg / C::m_p);
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