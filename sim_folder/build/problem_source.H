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
    // n_s and n_n_species are in m^-3 (SI) for the transport routines.
    amrex::Real n_e = 0.0;
    amrex::Real n_heavy = 0.0;                       // all heavy particles [m^-3]
    amrex::Real n_n_species[num_atom_species] = {};
    amrex::Real n_s[NumSpec] = {0.0};

    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        const amrex::Real n_mu = X_stage[mu] * rho * aion_inv[mu] / C::m_u * 1.e6;

        n_s[mu] = n_mu;          // Store specifically for viscosity loop
        n_e     += zion[mu] * n_mu;
        n_heavy += n_mu;

        if (zion[mu] == 0) {
            int idx_atom = get_atom_index(mu);
            n_n_species[idx_atom] += n_mu;
        }
    }

    // -----------------------------------------------------------------------
    // Heavy-species temperature.
    //
    // UFX/rho = f is the ELECTRON fraction of the internal energy, so the
    // heavy internal energy is e_h = (1-f) * e and, for monatomic heavies with
    // no internal excitation,  (3/2) n_h k_B T_h = rho * e_h.
    //
    // VERIFY this against your EOS/closure before production runs -- but do
    // not leave T_h = T_e, which collapses the two-temperature model and makes
    // mu(T_h) wrong wherever the plasma is out of thermal equilibrium.
    // -----------------------------------------------------------------------
    Real f_clamped   = amrex::min(amrex::max(f, 0.0_rt), 1.0_rt);
    Real e_h         = (1.0_rt - f_clamped) * e;             // erg/g
    Real n_heavy_cgs = n_heavy * 1.0e-6_rt;                  // m^-3 -> cm^-3

    Real T_h = T_e;
    if (e_h > 0.0_rt && n_heavy_cgs > 0.0_rt) {
        T_h = (2.0_rt / 3.0_rt) * (rho * e_h) / (n_heavy_cgs * C::k_B);
    }
    T_h = amrex::max(T_h, 1.0_rt);

    // --- 3. Dynamic Viscosity & Viscous Dissipation (Mathis Mewes Model) ---
    if (problem::viscosity == 1) {

        // Kinetic dynamic viscosity mu_H(T_h, T_e).  The transport namespace
        // works in SI and returns Pa*s; the hydro state is CGS, so convert to
        // poise (1 Pa*s = 10 g/cm/s) before differentiating the stress.
        Real mu_visc = transport::compute_dynamic_viscosity(T_h, n_active, active_spec, n_s, T_e);

        Real S_mx = 0.0, S_my = 0.0, S_mz = 0.0, S_visc = 0.0;
        transport::compute_viscous_sources(i, j, k, state, mu_visc, geomdata,
                                           S_mx, S_my, S_mz, S_visc);

        // Momentum: d(rho u)/dt += div(tau)
        src(i,j,k,UMX) += S_mx;
        src(i,j,k,UMY) += S_my;
        src(i,j,k,UMZ) += S_mz;

        // Energy.  With  d(rho u)/dt = div(tau)  the exact companion terms are
        //   d(rho E)/dt   += div(tau . u) = u . div(tau) + tau : grad(u)
        //   d(rho e)/dt   += tau : grad(u)                (dissipation only)
        // The u . div(tau) piece is kinetic-energy work and must NOT appear in
        // the internal-energy source, or viscosity spuriously heats/cools the
        // gas by the kinetic work.
        Real u_c = state(i,j,k,UMX) / rho;
        Real v_c = state(i,j,k,UMY) / rho;
        Real w_c = state(i,j,k,UMZ) / rho;

        Real S_kinetic = S_mx * u_c + S_my * v_c + S_mz * w_c;

        src(i,j,k,UEDEN) += S_kinetic + S_visc;
        src(i,j,k,UEINT) += S_visc;
    }

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

    // f = e_e / e_int is diluted by changes in INTERNAL energy, not total
    // energy, so this must track src(UEINT).  (Identical to the old form when
    // ionization was the only source, since that added the same amount to
    // UEDEN and UEINT; it differs once viscous kinetic work enters UEDEN.)
    const Real tiny_e = 1.0e-30;
    if (e > tiny_e) {
        src(i,j,k,UFX) = -f/e * src(i,j,k,UEINT) + Sc;
    } else {
        src(i,j,k,UFX) = Sc;
    }
}

#endif