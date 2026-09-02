#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <collisional_ionization.H>
#include <sh_diffusion_operator.H>
#include <sh_jfnk.H>

using namespace amrex;

// =====================================================================
// Pointwise explicit physics: collisional ionization / recombination and
// electron-heavy energy exchange.
//
// `diff_e` is the IMPLICIT diffusion energy source rate at this cell,
// computed beforehand by the JFNK solve. It is added into the energy
// source here, BEFORE the f-equation source is formed, so that the
// existing  f_dot = -(f/e) * e_dot  relation covers the diffusion too.
// This mirrors the original explicit implementation, where the
// diffusion term was added to src(UEDEN) ahead of the UFX line.
//
// Deriving the aux source this way keeps it bounded. Exporting a finite
// jump in rho*f from the converged implicit solve instead does not:
// when a cell's electrons are largely drained, f saturates at 1 and the
// jump over dt exceeds rho/dt, which drives f past 1 and makes the
// electron energy E*(1-f) -- and therefore Te -- negative.
//
// diff_e defaults to 0 because Castro's own Castro_problem_source.cpp
// still contains an 8-argument call to this function. The macro below
// returns before that line runs, but it must still type-check, so the
// default has to stay.
// =====================================================================
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time,
                     const Real diff_e = 0.0_rt, const Real diff_fx = 0.0_rt,
                     const bool aux_only = false)
{
    using namespace problem;
    amrex::ignore_unused(geomdata, dt, time);

    const Real rho = state(i,j,k,URHO);

    // Electron temperature from the EOS (rho, e, aux=f), not from the
    // stored UTEMP field: UTEMP is a placeholder at initialization and
    // is only re-derived by Castro's computeTemp later in the step, so
    // reading it here would feed the rate coefficients a stale Te.
    const Real T_e = sh_diff::get_Te(i, j, k, state);

    Real X_stage[NumSpec];
    bool active_mask[NumSpec] = {false};
    int  active_spec[NumSpec];
    int  n_active = 0;

    for (int mu = 0; mu < NumSpec; ++mu) {
        X_stage[mu] = state(i,j,k,UFS+mu) / rho;

        if (X_stage[mu] > X_min) {
            active_mask[mu] = true;
            if (mu > 0            && get_atom_index(mu) == get_atom_index(mu-1)) { active_mask[mu-1] = true; }
            if (mu < NumSpec - 1  && get_atom_index(mu) == get_atom_index(mu+1)) { active_mask[mu+1] = true; }
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

    // --- ionization / three-body recombination -----------------------
    Real e_ion = 0.0_rt;

    for (int n = 0; n < n_active; ++n) {
        const int beta      = active_spec[n];
        const int beta_next = beta + 1;
        const int atom_idx  = get_atom_index(beta);

        if (beta_next < NumSpec &&
            get_atom_index(beta_next) == atom_idx &&
            zion[beta] < zatom[atom_idx])
        {
            const Real coll_I_d   = X_stage[beta]   * X_e       * coll_ion::get_rate_direct_ion(beta, T_e);
            const Real coll_I_e   = X_stage[beta]   * X_e       * coll_ion::get_rate_ex_ion(beta, T_e);
            const Real three_body = X_stage[beta+1] * X_e * X_e * coll_ion::get_rate_3body(beta, rho, T_e);

            const Real coeff = (coll_I_d + coll_I_e - three_body) * (rho * rho / C::m_p);

            src(i,j,k,UFS+beta)      -= coeff;
            src(i,j,k,UFS+beta_next) += coeff;

            const int  idx_Eion  = get_energy_index(zatom[atom_idx]) + zion[beta];
            const Real E_ion_erg = ionization_energy[idx_Eion] * C::ev2erg;
            e_ion -= coeff * (E_ion_erg / (aion[beta] * C::m_p));
        }
    }

    // Energy source = ionization + diffusion. The diffusion contribution
    // must be included HERE, before src(UFX) is formed below, so the
    // f-equation picks it up through -f/e * e_dot.
    //
    // In aux_only mode Castro is applying conduction to the energy
    // itself, so diff_e must be kept OUT of the energy source; it is
    // still used for the f-equation below.
    const Real e_src = aux_only ? e_ion : (e_ion + diff_e);
    src(i,j,k,UEDEN) = e_src;
    src(i,j,k,UEINT) = e_src;

    // --- electron-heavy energy exchange ------------------------------
    Real n_e = 0.0_rt;
    Real n_n_species[num_atom_species] = {};

    for (int n = 0; n < n_active; ++n) {
        const int  mu   = active_spec[n];
        const Real n_mu = X_stage[mu] * rho * aion_inv[mu] / C::m_u * 1.e6_rt;
        n_e += zion[mu] * n_mu;
        if (zion[mu] == 0) { n_n_species[get_atom_index(mu)] += n_mu; }
    }

    Real v_ei;
    Real v_en[NumSpec];
    const Real T_h = T_e;   // TODO: use the true heavy temperature here
    get_collision_frequencies(n_e, n_n_species, T_e, T_h, v_ei, v_en);

    Real v_e_E[NumSpec] = {};
    for (int n = 0; n < n_active; ++n) {
        const int mu = active_spec[n];
        v_e_E[mu] = 2.0_rt * SI::m_e /
                    (aion[atom_first_species[get_atom_index(mu)]] * C::m_u * 1e-3_rt) *
                    (zion[mu] * v_ei + v_en[mu]);
    }

    Real Sc = 0.0_rt;
    for (int n = 0; n < n_active; ++n) {
        const int mu = active_spec[n];
        Sc += rho * v_e_E[mu] * (1.0_rt - (1.0_rt + X_stage[mu]) * f);
    }

    // get_collision_frequencies takes sqrt() inside the Coulomb
    // logarithm, which returns NaN if n_e or T_e ever go non-positive
    // (advected species can do this). A NaN here would be written
    // straight into src(UFX) and from there into the state, so trap it.
    if (!amrex::Math::isfinite(Sc)) { Sc = 0.0_rt; }

    // f-equation source.
    //
    // The -(f/e) linearization is applied ONLY to the ionization energy
    // e_ion, which is a small perturbation where the approximation
    // holds. It must NOT be applied to the diffusion energy: the solver
    // already returns the exact aux change for that (diff_fx, computed
    // as rho*(E_h/E^{n+1} - E_h/E^n) with E_h frozen), so putting the
    // diffusion energy through -(f/e) as well would count it twice.
    //
    // The exact form is also the only one that survives this problem's
    // stiffness. |dE|/E routinely reaches ~50 here, where a
    // linearization in dE/E is meaningless; capping dE to keep it valid
    // was discarding ~90% of the physical conduction every step.
    const Real tiny_e = 1.0e-30_rt;

    // Which energy the -(f/e) linearisation is applied to.
    //
    // Normally only the ionisation term: the solver returns the EXACT
    // auxiliary change for conduction in diff_fx, and putting the
    // conduction energy through -(f/e) as well would count it twice.
    //
    // In aux_only mode the solver returns diff_fx = 0 and hands back
    // L in diff_e purely so the conduction term reaches the f-equation,
    // in the differential form of the governing document. It is excluded
    // from the energy source above, since Castro applies it there.
    const Real e_lin = aux_only ? (e_ion + diff_e) : e_ion;

    Real fdot = Sc + diff_fx;
    if (std::abs(e) > tiny_e) {
        fdot += -f / e * e_lin;
    }

    // Backstop on the TOTAL aux source. diff_fx is already bounded in
    // the solver, but Sc and the ionization term feed the same variable
    // and the incoming state may itself be out of range. Keep f inside
    // [0,1]: dt*fdot must not exceed the room available on either side.
    //
    // Bounding UFX per cell is safe -- it carries no energy, only the
    // electron/heavy split -- so unlike the energy source this cannot
    // break conservation.
    if (dt > 0.0_rt && rho > 0.0_rt) {
        constexpr Real aux_safety = 0.4_rt;
        const Real ufx_c = state(i,j,k,UFX);
        const Real lo = -aux_safety * amrex::max(ufx_c, 0.0_rt)        / dt;
        const Real hi =  aux_safety * amrex::max(rho - ufx_c, 0.0_rt)  / dt;
        fdot = amrex::min(amrex::max(fdot, lo), hi);
    }

    src(i,j,k,UFX) = fdot;

    if (!amrex::Math::isfinite(src(i,j,k,UFX)))   { src(i,j,k,UFX)   = 0.0_rt; }
    if (!amrex::Math::isfinite(src(i,j,k,UEINT))) { src(i,j,k,UEINT) = 0.0_rt; }
    if (!amrex::Math::isfinite(src(i,j,k,UEDEN))) { src(i,j,k,UEDEN) = 0.0_rt; }
}

template <class... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ignore__unused (const Ts&...) {}

// =====================================================================
// fill_ext_source2 -- whole-MultiFab driver.
//
// Castro's real signature is
//   fill_ext_source(Real, Real, const MultiFab& S_old,
//                   const MultiFab& S_new, MultiFab& ext_src)
// so `state_new` is CONST and `ext_src` is the only writable output.
// Everything below respects that: the implicit solve runs on a private
// copy and is returned to Castro as an equivalent source rate.
//
// Additive operator splitting: the explicit pointwise sources and the
// implicit diffusion source are both evaluated at `state_new` and summed
// into ext_src. Castro then applies state += dt*ext_src itself.
// =====================================================================
inline void
fill_ext_source2 (const Real time, const Real dt,
                  const MultiFab& state_old, const MultiFab& state_new,
                  MultiFab& ext_src, const GeometryData& geomdata)
{
    ignore__unused(state_old);

    // ---- 1. rebuild a Geometry from the GeometryData ----------------
    // GeometryData carries prob_domain as a member and isPeriodic(dir);
    // Geometry itself is default-constructed then defined.
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        is_periodic[d] = geomdata.isPeriodic(d);
    }
    Geometry geom;
    geom.define(geomdata.Domain(), geomdata.prob_domain,
                geomdata.Coord(), is_periodic);

    // ---- 2. implicit electron thermal diffusion (backward Euler) ----
    // Done FIRST so the pointwise kernel below can fold the resulting
    // energy source into src(UEDEN) before it forms src(UFX).
    const int nGrow = state_new.nGrow();

    MultiFab dsrc_e (ext_src.boxArray(), ext_src.DistributionMap(), 1, 0);
    MultiFab dsrc_fx(ext_src.boxArray(), ext_src.DistributionMap(), 1, 0);

    // Parameters come from the inputs file via sh_diffusion.*, so
    // switching schemes needs no rebuild:
    //
    //   implicit scheme    sh_diffusion.mode = implicit
    //                      castro.diffuse_temp = 0
    //
    //   explicit baseline  sh_diffusion.mode = off
    //                      castro.diffuse_temp = 1
    //                      sh_diffusion.aux_for_castro_diffusion = 1
    const sh_jfnk::Params& params = sh_jfnk::runtime_params();

    // Aux-only: Castro applies conduction to the energy itself, and the
    // solver returns L only so the f-equation can pick up its
    // -(f/e)*div(lambda_e grad Te) term. It must reach the auxiliary
    // source but NOT the energy source.
    const bool aux_only = (params.mode == sh_jfnk::MODE_OFF &&
                           params.aux_for_castro_diffusion);
    // `time` lets the solver recognise a repeated identical call:
    // Castro evaluates ext_src more than once per step from the same
    // old-time state.
    const bool converged = sh_jfnk::solve(state_new, dsrc_e, dsrc_fx,
                                          geom, dt, nGrow, params, time);

    if (!converged) {
        amrex::Print() << "WARNING: implicit diffusion did not converge, t="
                       << time << " dt=" << dt << "\n";
    }

    // ---- 3. pointwise sources, consuming the diffusion source -------
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
                           de(i,j,k), dfx(i,j,k), aux_only);
        });
    }
}

// ---------------------------------------------------------------------
// Hijack of "amrex::ignore_unused(state_old);" in
// Castro_problem_source.cpp so our whole-MultiFab driver runs instead.
//
// expand_state() is called UNCONDITIONALLY, even when state_new already
// carries ghost cells. Having a ghost cell is not the same as having a
// FILLED one: at a coarse-fine boundary only FillPatch, which
// expand_state performs, can interpolate from the coarse level. Taking
// the state as-is whenever it happened to have room for ghosts left
// those cells stale under AMR, and the diffusion stencil read them along
// every refinement interface.
//
// The cost is one FillPatch per call, which is negligible against the
// solve and is the price of correct coarse-fine boundary data.
// ---------------------------------------------------------------------
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