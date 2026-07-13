#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <collisional_ionization.H>
#include <viscosity.H>

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
    Real X_stage[NumSpec];
    bool active_mask[NumSpec] = {false};
    int active_spec[NumSpec];
    int n_active = 0;

    for (int mu = 0; mu < NumSpec; mu++) {
        X_stage[mu] = state(i,j,k,UFS+mu) / rho;
        if (X_stage[mu] > X_min) {
            active_mask[mu] = true;
            if (mu > 0 && get_atom_index(mu) == get_atom_index(mu-1)) active_mask[mu-1] = true;
            if (mu < NumSpec-1 && get_atom_index(mu) == get_atom_index(mu+1)) active_mask[mu+1] = true;
        }
        if (active_mask[mu]) {
            active_spec[n_active++] = mu;
        }
    }
    
    Real f = state(i,j,k,UFX) / rho;
    Real e = state(i,j,k,UEINT) / rho;

    Real X_e = 0.0;
    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        X_e += zion[mu] * X_stage[mu] * aion_inv[mu];
    }
    
    Real e_ion = 0.0;

    // --- Collisional Ionization ---
    for (int ispe = 0; ispe < n_active; ispe++) {
        int beta = active_spec[ispe];
        int beta_next = beta + 1;
        int atom_idx = get_atom_index(beta);

        if (beta_next < NumSpec && get_atom_index(beta_next) == atom_idx && zion[beta] < zatom[atom_idx]) {
            Real coll_I_d = X_stage[beta] * X_e * coll_ion::get_rate_direct_ion(beta, T_e);
            Real coll_I_e = X_stage[beta] * X_e * coll_ion::get_rate_ex_ion(beta, T_e);
            Real three_body = X_stage[beta+1] * X_e * X_e * coll_ion::get_rate_3body(beta, rho, T_e);
            
            Real const coeff = (coll_I_d + coll_I_e - three_body) * (rho * rho / C::m_p);
            
            src(i,j,k,UFS+beta)      -= coeff;
            src(i,j,k,UFS+beta_next) += coeff;
            
            int idx_Eion = get_energy_index(zatom[atom_idx]) + zion[beta];
            Real E_ion_erg = ionization_energy[idx_Eion] * C::ev2erg;
            e_ion -= coeff * (E_ion_erg / (aion[beta] * C::m_p));
        }
    }

    src(i,j,k,UEDEN) += e_ion;
    src(i,j,k,UEINT) += e_ion;

    // --- Generalized Number Densities for Viscosity ---
    amrex::Real n_e = 0.0;
    amrex::Real n_n_species[num_atom_species] = {};
    amrex::Real n_s[NumSpec] = {0.0}; 

    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        const amrex::Real n_mu = X_stage[mu] * rho * aion_inv[mu] / C::m_u * 1.e6;
        
        n_s[mu] = n_mu; // Store specifically for viscosity loop
        n_e += zion[mu] * n_mu;
        
        if (zion[mu] == 0) {
            int idx_atom = get_atom_index(mu);
            n_n_species[idx_atom] += n_mu;
        } 
    }

    // --- Classic Explicit Viscosity Scheme using Spitzer ---
    const amrex::Real T_h = T_e; // Using single-temperature assumption

    // Calculate the dynamic viscosity using the simplified Spitzer formulation
    Real mu_visc = transport::compute_spitzer_viscosity(T_h, n_active, active_spec, n_s);

    // Call the abstracted spatial gradient and source term builder
    Real S_mx = 0.0, S_my = 0.0, S_mz = 0.0, S_visc = 0.0;
    transport::compute_viscous_sources(i, j, k, state, mu_visc, geomdata, S_mx, S_my, S_mz, S_visc);

    // Momentum change due to viscosity
    src(i,j,k,UMX) += S_mx;
    src(i,j,k,UMY) += S_my;
    src(i,j,k,UMZ) += S_mz;

    // Viscous heating
    Real u_c = state(i,j,k,UMX) / rho;
    Real v_c = state(i,j,k,UMY) / rho;
    Real w_c = state(i,j,k,UMZ) / rho;
    src(i,j,k,UEINT) += (S_mx * u_c + S_my * v_c + S_mz * w_c) + S_visc;

    // --- Energy Transfer ---
    amrex::Real v_ei;
    amrex::Real v_en[NumSpec];
    get_collision_frequencies(n_e, n_n_species, T_e, T_h, v_ei, v_en);

    amrex::Real v_e_E[NumSpec] = {};
    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        v_e_E[mu] = 2 * SI::m_e / (aion[get_atom_index(mu)] * C::m_u * 1e-3) *
                    (zion[mu] * v_ei + v_en[mu]);
    }

    amrex::Real Sc = 0.0;
    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        Sc += rho * v_e_E[mu] * (1 - (1 + X_stage[mu]) * f);
    }

    const Real tiny_e = 1.0e-30;
    if (e > tiny_e) {
        src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc;
    } else {
        src(i,j,k,UFX) = Sc;
    }
}

#endif