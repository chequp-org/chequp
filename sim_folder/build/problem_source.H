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
// IMPORTANT: state_new arrives here as `const MultiFab&` (this matches
// the real Castro::fill_ext_source signature -- confirmed from the
// build error, not assumed). This function NEVER writes into state_new.
// The only writable output is ext_src, exactly as in the original
// working explicit version.
//
// 1. Applies the pointwise problem_source kernel everywhere (ionization,
//    collisional energy exchange) exactly as before -- fully explicit,
//    written into ext_src.
// 2. Builds a local working copy of state_new with the explicit sources
//    already folded in (this copy is owned here, so it's fine to mutate).
// 3. Solves thermal diffusion implicitly (backward Euler + JFNK) on that
//    working copy, and gets back the EFFECTIVE volumetric source rate
//    that would reproduce the implicit result when Castro later does
//    its own state_new += dt*ext_src.
// 4. Adds that effective diffusion source into ext_src(UEINT/UEDEN), on
//    top of (not replacing) the explicit ionization/collision sources
//    from step 1.
//
// This keeps everything consistent with Castro's actual accumulation
// contract: fill_ext_source2 only ever writes ext_src.
// ---------------------------------------------------------------------
inline void
fill_ext_source2 (const Real time, const Real dt,
                  const MultiFab& state_old, const MultiFab& state_new,
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

    // --- Step 2: local working copy with explicit sources folded in ---
    // state_new has whatever ghost cells were provided by the caller
    // (see the ignore_unused macro below, which guarantees >=1).
    int nGrow = state_new.nGrow();
    amrex::MultiFab state_work(state_new.boxArray(), state_new.DistributionMap(),
                                state_new.nComp(), nGrow);
    amrex::MultiFab::Copy(state_work, state_new, 0, 0, state_new.nComp(), nGrow);
    amrex::MultiFab::Saxpy(state_work, dt, ext_src, 0, 0, ext_src.nComp(), 0);
    // Note: ext_src has 0 ghost cells, so only the valid region of
    // state_work is updated here; ghost cells of state_work still hold
    // the pre-source values. FillBoundary is called inside the JFNK
    // solver before any stencil evaluation, which corrects this for
    // periodic BCs. For physical (non-periodic) BCs this is a known gap
    // -- see note in the accompanying explanation.

    // GeometryData exposes prob_domain as a field (not ProbDomain()),
    // and isPeriodic(int) per-direction (not a no-arg isPeriodic()).
    // Confirmed AMReX pattern: default-construct then .define(...) --
    // there is no single Geometry constructor taking all four pieces.
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        is_periodic[d] = geomdata.isPeriodic(d);
    }
    amrex::Geometry geom;
    geom.define(geomdata.Domain(), geomdata.prob_domain,
                geomdata.Coord(), is_periodic);

    // --- Step 3: implicit diffusion (backward Euler + JFNK) ---
    amrex::MultiFab dsrc_mf(ext_src.boxArray(), ext_src.DistributionMap(), 1, 0);
    bool converged = jfnk_diffusion::solve_implicit_diffusion(state_work, dsrc_mf, geom, dt, nGrow);
    if (!converged) {
        amrex::Print() << "WARNING: JFNK diffusion solve did not converge to tolerance "
                        << "within max_newton_iters at time " << time
                        << ", dt = " << dt << "\n";
    }

    // --- Step 4: add the effective diffusion source into ext_src,
    // on top of the explicit ionization/collision sources already there.
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(ext_src, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const dsrc = dsrc_mf.const_array(mfi);
        Array4<Real> const src = ext_src.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            src(i,j,k,UEDEN) += dsrc(i,j,k);
            src(i,j,k,UEINT) += dsrc(i,j,k);
        });
    }
}

// attack the line "amrex::ignore_unused(state_old);" in Castro_problem_source.cpp
// to call our version of fill_ext_source instead.
// Note: state_new is const here, matching Castro's real signature --
// confirmed from the build error message rather than assumed.
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
        fill_ext_source2(time, dt, state_old, state_new_ghost, ext_src, geom.data()); \
        return; \
    } \
    ignore__unused

#endif