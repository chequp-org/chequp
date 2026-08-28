#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <conductivity.H>
#include <collisional_ionization.H>
#include <jfnk_diffusion.H>

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

    // NOTE: diffusion contribution is NOT added here anymore.
    // It is handled implicitly (backward Euler + JFNK) in fill_ext_source2
    // below, which operates on the whole state MultiFab after this
    // pointwise kernel has been applied everywhere.
    src(i,j,k,UEDEN) = e_ion;
    src(i,j,k,UEINT) = e_ion;

    amrex::Real n_e = 0.0;
    amrex::Real n_n_species[num_atom_species] = {};

    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        const amrex::Real n_mu = X_stage[mu] * rho * aion_inv[mu] / C::m_u * 1.e6;
        n_e += zion[mu] * n_mu;
        if (zion[mu] == 0) {
            int idx_atom = get_atom_index(mu);
            n_n_species[idx_atom] += n_mu;
        }
    }

    amrex::Real v_ei;
    amrex::Real v_en[NumSpec];
    const amrex::Real T_h = T_e;
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
    if (std::abs(e) > tiny_e) {
        src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc;
    } else {
        src(i,j,k,UFX) = Sc;
    }
}

template <class... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ignore__unused (const Ts&...) {}

// ---------------------------------------------------------------------
// fill_ext_source2: whole-MultiFab hook (host-side driver, not per-cell).
//
// 1. Applies the pointwise problem_source kernel everywhere (ionization,
//    collisional energy exchange) exactly as before -- fully explicit.
// 2. THEN solves thermal diffusion implicitly (backward Euler + JFNK) on
//    a working copy of state_new that already includes the explicit
//    source contributions from step 1, and writes the converged T, UEINT,
//    UEDEN directly back into state_new.
// 3. Zeroes ext_src's energy components so Castro's own explicit
//    accumulation of ext_src into the state does not double-count the
//    diffusion (or the energy sources already folded into state_new here).
//
// This keeps operator-splitting explicit: [explicit ionization/collision]
// then [implicit diffusion], both within one Castro timestep, executed
// from this single non-invasive hook.
// ---------------------------------------------------------------------
inline void
fill_ext_source2 (const Real time, const Real dt,
                  const MultiFab& state_old, MultiFab& state_new,
                  MultiFab& ext_src, const GeometryData& geomdata)
{
    ignore__unused(state_old);

    // --- Step 1: explicit pointwise sources (ionization, collisions) ---
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(ext_src, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        Array4<Real const> const snew = state_new.const_array(mfi);
        Array4<Real> const src = ext_src.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            problem_source(i, j, k, geomdata, snew, src, dt, time);
        });
    }

    // Fold the explicit sources into a working copy of state_new so the
    // implicit diffusion solve starts from the post-explicit-source state.
    // (state_new here should already have >=1 ghost cell -- see the
    // ignore_unused macro below which guarantees this.)
    amrex::MultiFab::Saxpy(state_new, dt, ext_src, 0, 0, ext_src.nComp(), 0);

    // Re-derive UTEMP from the updated UEINT/species so the Newton
    // initial guess is consistent before diffusion.
    // (Uses same EOS call as sync_energy_from_temp but inverted: here we
    // instead directly reuse UTEMP as initial guess and let
    // sync_energy_from_temp inside the solver reconcile it on the first
    // residual evaluation -- acceptable since Newton will correct any
    // small inconsistency in the first iteration.)

    amrex::Geometry geom(geomdata.Domain(), geomdata.ProbDomain(), geomdata.Coord(),
                          geomdata.isPeriodic());

    // --- Step 2: implicit diffusion (backward Euler + JFNK) ---
    bool converged = jfnk_diffusion::solve_implicit_diffusion(state_new, geom, dt);
    if (!converged) {
        amrex::Print() << "WARNING: JFNK diffusion solve did not converge to tolerance "
                        << "within max_newton_iters at time " << time
                        << ", dt = " << dt << "\n";
    }

    // --- Step 3: zero ext_src energy components so Castro's own
    // explicit accumulation doesn't double-apply what we already folded
    // into state_new above (species/UFX sources were already applied too).
    ext_src.setVal(0.0);
}

// attack the line "amrex::ignore_unused(state_old);" in Castro_problem_source.cpp
// to call our version of fill_ext_source instead
#define ignore_unused \
    Real x = 0; \
    ignore__unused(x); \
    if (state_new.nGrowVect().allGE(amrex::IntVect(1))) { \
        fill_ext_source2(time, dt, state_old, state_new, ext_src, geom.data()); \
        return; \
    } else { \
        amrex::MultiFab state_new_ghost(state_new.boxArray(), state_new.DistributionMap(), \
                                        state_new.nComp(), amrex::IntVect(1)); \
        expand_state(state_new_ghost, time, state_new_ghost.nGrow()); \
        fill_ext_source2(time, dt, state_old, state_new_ghost, ext_src, geomdata.data()); \
        state_new.ParallelCopy(state_new_ghost, 0, 0, state_new.nComp(), 0, 0); \
        return; \
    } \
    ignore__unused

#endif