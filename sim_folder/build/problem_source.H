#ifndef PROBLEM_SOURCE_H
#define PROBLEM_SOURCE_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <collisional_ionization.H>
#include <thd_diffusion_operator.H>
#include <thd_jfnk.H>

using namespace amrex;

// `diff_e` and `diff_fx` carry the IMPLICIT diffusion sources from the 
// preceding JFNK solve. Combining `diff_e` into `e_src` allows the 
// `f_dot` relation to handle heat diffusion consistently without 
// driving `f` past 1.

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time,
                     const Real diff_e = 0.0_rt, const Real diff_fx = 0.0_rt)
{
    using namespace problem;
    amrex::ignore_unused(geomdata, dt, time);

    const Real rho = state(i,j,k,URHO);

    // Compute Te dynamically from EOS rather than relying on cached UTEMP
    const Real T_e = thd_diff::get_Te(i, j, k, state);

    Real X_stage[NumSpec];
    bool active_mask[NumSpec] = {false};
    int  active_spec[NumSpec];
    int  n_active = 0;

    for (int mu = 0; mu < NumSpec; ++mu) {
        X_stage[mu] = state(i,j,k,UFS+mu) / rho;

        if (X_stage[mu] > X_min) {
            active_mask[mu] = true;
            if (mu > 0           && get_atom_index(mu) == get_atom_index(mu-1)) { active_mask[mu-1] = true; }
            if (mu < NumSpec - 1 && get_atom_index(mu) == get_atom_index(mu+1)) { active_mask[mu+1] = true; }
        }
        if (active_mask[mu]) { active_spec[n_active++] = mu; }
    }

    const Real f = state(i,j,k,UFX)   / rho;
    const Real e = state(i,j,k,UEINT) / rho;

    Real X_e = 0.0_rt;
    for (int n = 0; n < n_active; ++n) {
        const int mu = active_spec[n];
        X_e += zion[mu] * X_stage[mu] * aion_inv[mu];
    }
    
    // Calculate neutral, electron, and Z^2-weighted ion number densities [m-3]
    amrex::Real n_e = 0.0;
    amrex::Real n_n_species[num_atom_species] = {};
    amrex::Real n_iZ2[num_atom_species] = {};

    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        const amrex::Real n_mu = X_stage[mu] * rho * aion_inv[mu] / C::m_u * 1.e6;
        const int idx_atom = get_atom_index(mu);
        n_e += zion[mu] * n_mu;
        if (zion[mu] == 0) {
            n_n_species[idx_atom] += n_mu;
        } else {
            n_iZ2[idx_atom] += zion[mu] * zion[mu] * n_mu;
        }
    }

    // Total heavy-particle density over all stages
    amrex::Real n_h = 0.0;
    for (int mu = 0; mu < NumSpec; mu++) {
        n_h += X_stage[mu] * rho * aion_inv[mu] / C::m_u * 1.e6;
    }

    // Initialize sources
    Real e_ion = 0.0;

    for (int ispe = 0; ispe < n_active; ispe++) {
        int beta = active_spec[ispe];
        int beta_next = beta + 1;
        int atom_idx = get_atom_index(beta);

        if (beta_next < NumSpec && get_atom_index(beta_next) == atom_idx && zion[beta] < zatom[atom_idx]) {
            // Calculate Rates
            Real coll_I_d = X_stage[beta] * X_e * coll_ion::get_rate_direct_ion(beta, T_e);
            Real coll_I_e = X_stage[beta] * X_e * coll_ion::get_rate_ex_ion(beta, T_e);
            Real three_body = X_stage[beta+1] * X_e * X_e * coll_ion::get_rate_3body(beta, n_e, rho, T_e);
            // Net Rate Coefficient
            Real const coeff = (coll_I_d + coll_I_e - three_body) * (rho * rho / C::m_p);
            // Update Mass Fraction Sources
            src(i,j,k,UFS+beta)      -= coeff;
            src(i,j,k,UFS+beta_next) += coeff;

            const int  idx_Eion  = get_energy_index(zatom[atom_idx]) + zion[beta];
            const Real E_ion_erg = ionization_energy[idx_Eion] * C::ev2erg;
            e_ion -= coeff * (E_ion_erg / (aion[beta] * C::m_p));
        }
    }

    // Combined energy source: ionization + implicit diffusion
    const Real e_src = e_ion + diff_e;
    src(i,j,k,UEDEN) = e_src;
    src(i,j,k,UEINT) = e_src;

    // Compute collision frequencies (ei: electron-ion, en: electron-neutral)
    amrex::Real v_ei;
    amrex::Real v_en_atom[num_atom_species] = {};
    const amrex::Real T_h = T_e;
    get_collision_frequencies(n_e, n_n_species, T_e, T_h, v_ei, v_en_atom);

    // Total energy-transfer collision frequency, accumulated once per atom
    amrex::Real nu_E = 0.0;
    for (int a = 0; a < num_atom_species; ++a) {
        // Mass of the atom
        const amrex::Real m_a = aion[atom_first_species[a]] * C::m_u * 1.e-3;
        // If no electrons, no collision frequency
        const amrex::Real nu_ei_a = (n_e > 0.0) ? v_ei * n_iZ2[a] / n_e : 0.0;
        nu_E += 2.0 * SI::m_e / m_a * (nu_ei_a + v_en_atom[a]);
    }

    // Energy transfer source term
    // If no heavy particles, no energy transfer
    const amrex::Real Sc = (n_h > 0.0) ? rho * nu_E * (1.0 - (1.0 + n_e / n_h) * f) : 0.0;

    if (!amrex::Math::isfinite(Sc)) { Sc = 0.0_rt; }

    // --- Auxiliary Fraction (f) Source -------------------------------
    // Apply -(f/e) linearization strictly to ionization (small perturbation).
    // diff_fx is exact from the solver and added directly.
    constexpr Real tiny_e = 1.0e-30_rt;

    Real fdot = Sc + diff_fx;
    if (std::abs(e) > tiny_e) {
        fdot += -f / e * e_ion;
    }

    // Safety bound to constrain f to [0,1]
    if (dt > 0.0_rt && rho > 0.0_rt) {
        constexpr Real aux_safety = 0.4_rt;
        const Real ufx_c = state(i,j,k,UFX);
        const Real lo = -aux_safety * amrex::max(ufx_c, 0.0_rt)       / dt;
        const Real hi =  aux_safety * amrex::max(rho - ufx_c, 0.0_rt) / dt;
        fdot = amrex::min(amrex::max(fdot, lo), hi);
    }

    src(i,j,k,UFX) = fdot;

    // Final NaN traps
    if (!amrex::Math::isfinite(src(i,j,k,UFX)))   { src(i,j,k,UFX)   = 0.0_rt; }
    if (!amrex::Math::isfinite(src(i,j,k,UEINT))) { src(i,j,k,UEINT) = 0.0_rt; }
    if (!amrex::Math::isfinite(src(i,j,k,UEDEN))) { src(i,j,k,UEDEN) = 0.0_rt; }
}

template <class... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ignore__unused (const Ts&...) {}


// Whole-MultiFab driver for explicit sources and implicit JFNK diffusion

inline void
fill_ext_source2 (const Real time, const Real dt,
                  const MultiFab& state_old, const MultiFab& state_new,
                  MultiFab& ext_src, const GeometryData& geomdata)
{
    ignore__unused(state_old);

    // Reconstruct Geometry object from GeometryData
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        is_periodic[d] = geomdata.isPeriodic(d);
    }
    Geometry geom;
    geom.define(geomdata.Domain(), geomdata.prob_domain,
                geomdata.Coord(), is_periodic);

    // Implicit electron thermal diffusion (Backward Euler)
    const int nGrow = state_new.nGrow();

    MultiFab dsrc_e (ext_src.boxArray(), ext_src.DistributionMap(), 1, 0);
    MultiFab dsrc_fx(ext_src.boxArray(), ext_src.DistributionMap(), 1, 0);

    const thd_jfnk::Params& params = thd_jfnk::runtime_params();

    const bool converged = thd_jfnk::solve(state_new, dsrc_e, dsrc_fx,
                                            geom, dt, nGrow, params, time);

    if (!converged) {
        amrex::Print() << "WARNING: implicit diffusion did not converge, t="
                       << time << " dt=" << dt << "\n";
    }

    // Evaluate explicit pointwise sources alongside implicit diffusion
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(ext_src, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const s   = state_new.const_array(mfi);
        Array4<Real const> const de  = dsrc_e.const_array(mfi);
        Array4<Real const> const dfx = dsrc_fx.const_array(mfi);
        Array4<Real>       const src = ext_src.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            problem_source(i, j, k, geomdata, s, src, dt, time,
                           de(i,j,k), dfx(i,j,k));
        });
    }
}


// Macro hijack for Castro_problem_source.cpp to route execution through 
// fill_ext_source2 and ensure proper FillPatch handling across AMR boundaries.

#define ignore_unused \
    Real x = 0; \
    ignore__unused(x); \
    { \
        amrex::MultiFab state_new_ghost(state_new.boxArray(), state_new.DistributionMap(), \
                                        state_new.nComp(), amrex::IntVect(1)); \
        expand_state(state_new_ghost, time, state_new_ghost.nGrow()); \
        fill_ext_source2(time, dt, state_old, state_new_ghost, ext_src, geom.data()); \
        return; \
    } \
    ignore__unused

#endif