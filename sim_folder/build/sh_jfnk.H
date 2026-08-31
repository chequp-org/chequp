#ifndef SH_JFNK_H
#define SH_JFNK_H
//
// Jacobian-Free Newton-Krylov solve of backward-Euler electron thermal
// diffusion for a two-temperature Castro setup.
//
//   F(E) = E - E^n - dt * div( kappa(Te) grad Te )  =  0
//
// where E = UEINT is the TOTAL internal energy density and the heavies'
// energy density E_h is held FIXED across the diffusion substep, so all
// diffused energy goes to the electrons. At every Newton iterate the aux
// variable is refreshed as
//
//   UFX = rho * f,   f = E_h / E
//
// which is what makes the EOS return the correct electron temperature.
// This is the same physics as the `-f/e * src(UEDEN)` term in the
// original explicit implementation, just carried through the implicit
// solve consistently.
//
// Nothing here writes into caller-owned state: the solver works on its
// own copy and returns effective source rates.
//
#include <sh_diffusion_operator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <vector>
#include <memory>

using namespace amrex;

namespace sh_jfnk {

// ------------------------- solver controls ---------------------------
//
// `mode` exists to bisect where a problem lives. Rebuild with each in
// turn; the first one that misbehaves identifies the guilty layer.
//
//   MODE_OFF      zero source. Diffusion contributes nothing. If the
//                 run still misbehaves, the cause is NOT this code.
//
//   MODE_EXPLICIT dsrc = L(state), a single evaluation of the operator
//                 at the current state, no solve at all. Isolates the
//                 operator + conductivity from the Newton/Krylov solve.
//                 Needs a small dt to be stable -- that is expected and
//                 is not what is being tested; what matters is whether
//                 the PROFILE is smooth.
//
//   MODE_IMPLICIT full backward-Euler JFNK (the real scheme).
//
// If OFF is clean and EXPLICIT is clean but IMPLICIT is not, the solver
// or the way its result is handed to Castro is at fault. If EXPLICIT is
// already bad, the operator or the conductivity model is at fault and
// the solver is innocent.
enum DiffMode { MODE_OFF = 0, MODE_EXPLICIT = 1, MODE_IMPLICIT = 2,
                MODE_ADAPTIVE = 3 };

struct Params {
//   MODE_ADAPTIVE decides per step whether the implicit solve is
    //                 needed at all, and falls back to a single explicit
    //                 evaluation when it is not. See explicit_sigma.
    DiffMode mode         = MODE_ADAPTIVE;

    // Stiffness threshold for MODE_ADAPTIVE.
    //
    // sigma = dt*|L|/E_e is the fractional change in a cell's electron
    // energy over the step -- the explicit stability/accuracy parameter.
    // Backward and forward Euler are BOTH first order in dt and differ
    // by O(sigma), so below this value the implicit solve buys nothing
    // and costs ~100 residual evaluations instead of one.
    //
    // The test is free: the entry residual F(E^n) is exactly -dt*L(E^n),
    // so sigma per cell is |F|/E_e with no additional work, and the
    // explicit update is simply UEINT = E^n - F.
    //
    // 0.1 is conservative (<=10% change in any single cell). Raise
    // toward 0.25 for more speed, lower it if you want the implicit
    // path taken more often. Set to 0 to always solve implicitly.
    Real explicit_sigma   = 0.1_rt;

    // Reuse the previous result when a solve is an exact repeat.
    // Castro evaluates ext_src more than once per step from the same
    // old-time state; the logs show byte-identical output. Guarded by a
    // state checksum, so a mismatch simply falls through to a real solve.
    bool use_cache        = true;

    // Multiplies the returned source. Castro evaluates ext_src at both
    // old and new time and combines them; if it sums rather than
    // averages, the diffusion lands at roughly double strength, and
    // over-applying a change computed to reach equilibrium overshoots
    // past it. Try 0.5 to test that directly.
    Real source_scale     = 1.0_rt;

    int  max_newton       = 20;
    // Relative to the energy scale, not to F0.
    //
    // 1e-8 was needlessly tight: the diffusion changes the energy by
    // well under 1% per step, so resolving the residual to 1e-8 of the
    // TOTAL energy resolves it to ~1e-4 of the actual change. That
    // routinely costs a whole extra Newton iteration -- and each one is
    // ~30 Krylov iterations -- for no accuracy that survives the
    // operator splitting. Loosen further if Newton still takes more
    // iterations than you want.
    Real newton_rtol      = 1.0e-6_rt;
    // Maximum Krylov iterations.
    //
    // 0 means "choose automatically from the grid". With only a
    // diagonal preconditioner the iteration count for diffusion scales
    // with the number of cells ACROSS the domain, because one Krylov
    // iteration propagates information exactly one cell and implicit
    // diffusion is globally coupled. A fixed cap that works at 32^2
    // therefore fails at 128^2 -- GMRES stops early and hands Newton an
    // inexact direction.
    //
    // Cost grows as k^2 in orthogonalization (k dot products per
    // iteration, each a device sync plus an MPI Allreduce) and k
    // MultiFabs of memory, so this scaling is a stopgap. The real fix
    // is a multigrid preconditioner, whose iteration count does not
    // grow with resolution at all.
    int  max_gmres        = 0;
    Real gmres_rtol       = 1.0e-4_rt;   // inexact Newton: loose inner tolerance

    // Reorthogonalization trigger. If Arnoldi shrinks the vector by
    // more than this factor, cancellation was severe and orthogonality
    // is suspect, so a second Gram-Schmidt pass is run. 0 disables it.
    Real reorth_tol       = 0.7_rt;
    Real fd_eps           = 1.0e-7_rt;   // finite-difference step for J*v
    int  max_linesearch   = 8;
    bool verbose          = true;
};

// ---------------------------------------------------------------------
// Deterministic ghost fill.
//
// FillBoundary handles internal (grid-grid) and periodic boundaries.
// At PHYSICAL boundaries it leaves ghosts untouched, and those cells may
// never have been initialized -- a zero density there divides to NaN and
// poisons the whole stencil. So we explicitly extrapolate with zero
// gradient (copy the nearest valid cell), which is a well-defined
// no-flux / adiabatic thermal boundary rather than an accident.
//
// If your problem needs a real Dirichlet wall temperature instead,
// this is the single place to change it.
// ---------------------------------------------------------------------
inline void
fill_ghosts (MultiFab& state, const Geometry& geom, int scomp, int ncomp)
{
    state.FillBoundary(scomp, ncomp, geom.periodicity());

    const Box& domain = geom.Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);

    const int per[AMREX_SPACEDIM] = {
        AMREX_D_DECL(geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2))
    };

    for (MFIter mfi(state); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.fabbox();
        const Box& vbx = mfi.validbox();
        const auto vlo = amrex::lbound(vbx);
        const auto vhi = amrex::ubound(vbx);
        Array4<Real> const s = state.array(mfi);

        amrex::ParallelFor(gbx, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) noexcept
        {
            const int n = scomp + nn;
            int ii = i, jj = j, kk = k;
            bool outside = false;

            // Only extrapolate cells outside the DOMAIN (physical
            // boundary); interior ghosts were already filled by
            // FillBoundary and must be left alone.
            //
            // The source index is then clamped to THIS FAB's valid box,
            // not to the domain. Clamping to the domain is unsafe under
            // domain decomposition: a box that does not touch the domain
            // edge in some direction can produce a source index far
            // outside its own fab. On CPU that silently reads
            // neighbouring heap memory; on GPU it is an illegal access.
            // Clamping to the valid box also guarantees the source is a
            // cell this kernel never writes, so there is no race between
            // ghost layers when nGrow > 1.
            if (per[0] == 0) {
                if      (i < dlo.x) { outside = true; }
                else if (i > dhi.x) { outside = true; }
            }
#if AMREX_SPACEDIM >= 2
            if (per[1] == 0) {
                if      (j < dlo.y) { outside = true; }
                else if (j > dhi.y) { outside = true; }
            }
#endif
#if AMREX_SPACEDIM == 3
            if (per[2] == 0) {
                if      (k < dlo.z) { outside = true; }
                else if (k > dhi.z) { outside = true; }
            }
#endif
            if (outside) {
                ii = amrex::min(amrex::max(i, vlo.x), vhi.x);
#if AMREX_SPACEDIM >= 2
                jj = amrex::min(amrex::max(j, vlo.y), vhi.y);
#endif
#if AMREX_SPACEDIM == 3
                kk = amrex::min(amrex::max(k, vlo.z), vhi.z);
#endif
                s(i,j,k,n) = s(ii,jj,kk,n);
            }
        });
    }
}

// All-components convenience wrapper.
inline void
fill_ghosts (MultiFab& state, const Geometry& geom)
{
    fill_ghosts(state, geom, 0, state.nComp());
}

// ---------------------------------------------------------------------
// Refresh UFX (= rho*f = rho*E_h/E) over the grown box, so the EOS sees
// an f consistent with the current energy iterate. Eh must be valid over
// the same grown region.
// ---------------------------------------------------------------------
inline void
refresh_aux (MultiFab& state, const MultiFab& Eh)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        Array4<Real>       const s   = state.array(mfi);
        Array4<Real const> const eh  = Eh.const_array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real E   = s(i,j,k,UEINT);
            const Real rho = s(i,j,k,URHO);
            if (E > 0.0_rt && rho > 0.0_rt && amrex::Math::isfinite(eh(i,j,k))) {
                // Strictly inside (0,1): f = 1 means zero electron
                // energy and a degenerate T_e, which restarts the
                // kappa ~ T_e^(5/2) feedback loop.
                constexpr Real f_min = 1.0e-12_rt;
                constexpr Real f_max = 1.0_rt - 1.0e-8_rt;
                const Real f = amrex::min(amrex::max(eh(i,j,k) / E, f_min), f_max);
                s(i,j,k,UFX) = rho * f;
            }
        });
    }
}

// ---------------------------------------------------------------------
// F(E) = E - E^n - dt*L(E).  `state` must have filled ghosts and an aux
// already consistent with its UEINT (call refresh_aux + FillBoundary).
// ---------------------------------------------------------------------
inline void
residual (const MultiFab& state, const MultiFab& En, MultiFab& F,
          const Geometry& geom, Real dt, sh_diff::Workspace* ws = nullptr)
{
    MultiFab L(F.boxArray(), F.DistributionMap(), 1, 0);
    sh_diff::compute_L(state, L, geom, ws);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(F, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const s  = state.const_array(mfi);
        Array4<Real const> const en = En.const_array(mfi);
        Array4<Real const> const l  = L.const_array(mfi);
        Array4<Real>       const f  = F.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            f(i,j,k) = s(i,j,k,UEINT) - en(i,j,k) - dt * l(i,j,k);
        });
    }
}

// Set UEINT = base + lambda*delta on the valid region, then make the
// whole thing (ghosts + aux) self-consistent again.
inline void
apply_update (MultiFab& state, const MultiFab& base, const MultiFab& delta,
              Real lambda, const MultiFab& Eh, const Geometry& geom)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real>       const s = state.array(mfi);
        Array4<Real const> const b = base.const_array(mfi);
        Array4<Real const> const d = delta.const_array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // No clamping here. Clamping inside the iteration makes
            // Newton converge on a modified equation and distorts the
            // answer; positivity is enforced by rejecting the step in
            // the line search instead.
            s(i,j,k,UEINT) = b(i,j,k) + lambda * d(i,j,k);
        });
    }
    // Only UEINT ghosts need exchanging. refresh_aux then derives UFX
    // over the grown box from UEINT and Eh, so UFX ghosts come for free;
    // URHO, UFS and UTEMP are frozen for the whole solve and were filled
    // once at entry. Filling all NUM_STATE components here instead moved
    // an order of magnitude more data per Krylov iteration.
    fill_ghosts(state, geom, UEINT, 1);
    refresh_aux(state, Eh);
}

// ---------------------------------------------------------------------
// Matrix-free J*v ~= [F(E + eps*v) - F(E)] / eps
// ---------------------------------------------------------------------
inline void
jac_vec (const MultiFab& state, const MultiFab& F0, const MultiFab& v,
         MultiFab& Jv, const MultiFab& En, const MultiFab& Eh,
         const Geometry& geom, Real dt, const Params& p,
         MultiFab* pert_ws = nullptr, MultiFab* base_ws = nullptr,
         sh_diff::Workspace* ws = nullptr)
{
    // Standard JFNK step:  eps = sqrt(eps_mach) * (1 + ||E||) / ||v||,
    // with BOTH norms the same (L2). Using norm2 for the state but
    // norm0 for the direction makes eps too large by roughly sqrt(N),
    // which degrades the Jacobian approximation and slows Newton.
    // Standard JFNK step:  eps = sqrt(eps_mach) * (1 + ||E||) / ||v||,
    // with BOTH norms the same (L2). Using norm2 for the state but
    // norm0 for the direction makes eps too large by roughly sqrt(N),
    // which degrades the Jacobian approximation and slows Newton.
    const Real norm_E = state.norm2(UEINT);
    const Real norm_v = v.norm2(0);
    const Real eps = p.fd_eps * (norm_E + 1.0_rt) / (norm_v + 1.0e-300_rt);

    // The perturbed state and the base energy are supplied by the
    // caller and reused across Krylov iterations.
    //
    // Reallocating and full-copying the state here was costly: it moved
    // every component (NUM_STATE, with ghosts) once per Krylov
    // iteration, when only UEINT actually changes. URHO, UFS and UTEMP
    // are frozen for the whole solve, so a single full copy made once
    // per solve is enough -- apply_update then overwrites UEINT and
    // refreshes UFX, which is all the operator reads that varies.
    MultiFab  pert_local;
    MultiFab  base_local;
    MultiFab* pert = pert_ws;
    MultiFab* base = base_ws;

    if (pert == nullptr) {
        pert_local.define(state.boxArray(), state.DistributionMap(),
                          state.nComp(), state.nGrow());
        MultiFab::Copy(pert_local, state, 0, 0, state.nComp(), state.nGrow());
        pert = &pert_local;
    }
    if (base == nullptr) {
        base_local.define(state.boxArray(), state.DistributionMap(), 1, 0);
        MultiFab::Copy(base_local, state, UEINT, 0, 1, 0);
        base = &base_local;
    }

    apply_update(*pert, *base, v, eps, Eh, geom);

    MultiFab Feps(F0.boxArray(), F0.DistributionMap(), 1, 0);
    residual(*pert, En, Feps, geom, dt, ws);

    MultiFab::LinComb(Jv, 1.0_rt/eps, Feps, 0, -1.0_rt/eps, F0, 0, 0, 1, 0);
}

// ---------------------------------------------------------------------
// Preconditioner, split into BUILD (expensive) and APPLY (cheap).
//
//   diag( dF/dE ) ~= 1 + dt * (sum_dir k_face/dx^2) * dTe/dE
//
// For a gamma-law electron gas E_e = n_e k Te/(gamma-1), so dTe/dE =
// Te/E_e exactly, with E_e = E - E_h the electron energy.
//
// PERFORMANCE: the diagonal depends only on the current Newton iterate,
// so it is CONSTANT for an entire GMRES solve. It used to be rebuilt on
// every preconditioner apply -- twice per Krylov iteration -- with
// identical results each time. Each rebuild costs 2*DIM+1 EOS
// inversions per cell, so at ~11 Krylov iterations that was ~22
// redundant evaluations per Newton iteration, and it dominated the
// solver's runtime. Now built once per Newton iteration; the apply is
// a single divide per cell with no EOS calls.
//
// Only used to precondition GMRES -- it never enters the residual, so
// an imperfect diagonal costs iterations, not accuracy.
// ---------------------------------------------------------------------
inline void
build_precond (const MultiFab& state, const MultiFab& Eh, MultiFab& diag,
               const Geometry& geom, Real dt, sh_diff::Workspace* ws = nullptr)
{
    sh_diff::Workspace local;
    sh_diff::Workspace* w = ws;
    if (w == nullptr) {
        local.define(state.boxArray(), state.DistributionMap());
        w = &local;
    }
    sh_diff::compute_coeffs(state, w->coef);
    sh_diff::compute_face_coeffs(w->coef, w->fk, geom);

    // See the note in sh_diff::compute_L -- geom.CellSize() is a HOST
    // pointer and must not be captured into a device lambda.
    const amrex::GpuArray<Real,AMREX_SPACEDIM> problo = geom.ProbLoArray();
    const amrex::GpuArray<Real,AMREX_SPACEDIM> dx     = geom.CellSizeArray();
    const int coord = geom.Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(diag, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const s  = state.const_array(mfi);
        Array4<Real const> const cc = w->coef.const_array(mfi);
        Array4<Real const> const eh = Eh.const_array(mfi);
        AMREX_D_TERM(Array4<Real const> const fx = w->fk[0].const_array(mfi);,
                     Array4<Real const> const fy = w->fk[1].const_array(mfi);,
                     Array4<Real const> const fz = w->fk[2].const_array(mfi);)
        Array4<Real>       const d  = diag.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real Te_c = cc(i,j,k,sh_diff::CTE);

            // Exact diagonal of the discretized operator: the same face
            // coefficients and the same cell metric the stencil uses, so
            // the preconditioner cannot drift away from the operator.
            Real stiff = 0.0_rt;
            for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
                Real kl = 0.0_rt, kr = 0.0_rt;
                if (idir == 0) { kl = fx(i,j,k); kr = fx(i+1,j,k); }
#if AMREX_SPACEDIM >= 2
                if (idir == 1) { kl = fy(i,j,k); kr = fy(i,j+1,k); }
#endif
#if AMREX_SPACEDIM == 3
                if (idir == 2) { kl = fz(i,j,k); kr = fz(i,j,k+1); }
#endif
                stiff += sh_diff::cell_metric(i, j, k, idir,
                                              problo.data(), dx.data(), coord)
                         * (kl + kr);
            }

            const Real E    = amrex::max(s(i,j,k,UEINT), 1.0e-300_rt);
            const Real Ee   = amrex::max(E - eh(i,j,k), 1.0e-300_rt);
            const Real dTdE = Te_c / Ee;

            Real dg = 1.0_rt + dt * stiff * dTdE;
            if (!amrex::Math::isfinite(dg) || dg < 1.0_rt) { dg = 1.0_rt; }
            d(i,j,k) = dg;
        });
    }
}

// Cheap apply: one divide per cell, no EOS calls.
inline void
precond (const MultiFab& diag, const MultiFab& r, MultiFab& z)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(z, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const d  = diag.const_array(mfi);
        Array4<Real const> const rr = r.const_array(mfi);
        Array4<Real>       const zz = z.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            zz(i,j,k) = rr(i,j,k) / d(i,j,k);
        });
    }
}

// ---------------------------------------------------------------------
// Left-preconditioned GMRES for J dE = -F.
// ---------------------------------------------------------------------
inline int
gmres (const MultiFab& state, const MultiFab& F0, const MultiFab& En,
       const MultiFab& Eh, const MultiFab& diag, MultiFab& dE,
       const Geometry& geom, Real dt, const Params& p,
       sh_diff::Workspace* ws = nullptr, Real* final_resid = nullptr)
{
    const BoxArray&            ba = F0.boxArray();
    const DistributionMapping& dm = F0.DistributionMap();

    dE.setVal(0.0_rt);

    // Allocated ONCE per GMRES solve and reused by every jac_vec call.
    // pert holds the full state; only its UEINT/UFX are rewritten each
    // iteration, so the expensive full copy happens once here rather
    // than once per Krylov iteration.
    MultiFab pert(state.boxArray(), state.DistributionMap(),
                  state.nComp(), state.nGrow());
    MultiFab::Copy(pert, state, 0, 0, state.nComp(), state.nGrow());

    MultiFab base(ba, dm, 1, 0);
    MultiFab::Copy(base, state, UEINT, 0, 1, 0);

    MultiFab r(ba, dm, 1, 0);
    MultiFab::Copy(r, F0, 0, 0, 1, 0);
    r.mult(-1.0_rt);

    MultiFab z(ba, dm, 1, 0);
    precond(diag, r, z);

    const Real beta = z.norm2(0);
    if (final_resid != nullptr) { *final_resid = 0.0_rt; }
    if (!(beta > 1.0e-300_rt)) { return 0; }   // also catches NaN

    // Auto-size from the domain when max_gmres == 0.
    int m = p.max_gmres;
    if (m <= 0) {
        const Box& dom = geom.Domain();
        int nmax = dom.length(0);
        for (int d = 1; d < AMREX_SPACEDIM; ++d) {
            nmax = amrex::max(nmax, dom.length(d));
        }
        m = amrex::min(2 * nmax, 400);   // hard ceiling on memory/orthogonalization
        m = amrex::max(m, 40);
    }
    std::vector<std::unique_ptr<MultiFab>> V;
    V.reserve(m + 1);
    V.push_back(std::make_unique<MultiFab>(ba, dm, 1, 0));
    MultiFab::Copy(*V[0], z, 0, 0, 1, 0);
    V[0]->mult(1.0_rt / beta);

    std::vector<std::vector<Real>> H(m + 1, std::vector<Real>(m, 0.0_rt));
    std::vector<Real> g(m + 1, 0.0_rt), cs(m, 0.0_rt), sn(m, 0.0_rt);
    g[0] = beta;

    int kend = 0;
    for (int k = 0; k < m; ++k) {
        kend = k + 1;

        MultiFab w(ba, dm, 1, 0);
        jac_vec(state, F0, *V[k], w, En, Eh, geom, dt, p, &pert, &base, ws);

        MultiFab wpc(ba, dm, 1, 0);
        precond(diag, w, wpc);
        MultiFab::Copy(w, wpc, 0, 0, 1, 0);

        const Real wnorm_before = w.norm2(0);

        // Modified Gram-Schmidt: each dot product uses the partially
        // orthogonalized w, which is far more stable than the classical
        // variant.
        for (int i = 0; i <= k; ++i) {
            H[i][k] = MultiFab::Dot(w, 0, *V[i], 0, 1, 0);
            MultiFab::Saxpy(w, -H[i][k], *V[i], 0, 0, 1, 0);
        }
        H[k+1][k] = w.norm2(0);

        // Selective reorthogonalization.
        //
        // Even MGS loses orthogonality once the Krylov space is deep and
        // the operator ill-conditioned -- and a diagonal preconditioner
        // forces exactly that, since the iteration count grows with the
        // number of cells across the domain (order 150-250 at 128^2).
        // Lost orthogonality makes the computed step wrong even while
        // the Givens recurrence still reports convergence, which appears
        // as accuracy that DEGRADES with resolution.
        //
        // Classic criterion: if orthogonalization collapsed the norm,
        // cancellation was severe, so run a second pass. It triggers
        // rarely, so the added cost is small.
        if (p.reorth_tol > 0.0_rt &&
            H[k+1][k] < p.reorth_tol * wnorm_before)
        {
            for (int i = 0; i <= k; ++i) {
                const Real corr = MultiFab::Dot(w, 0, *V[i], 0, 1, 0);
                H[i][k] += corr;
                MultiFab::Saxpy(w, -corr, *V[i], 0, 0, 1, 0);
            }
            H[k+1][k] = w.norm2(0);
        }

        // Arnoldi breakdown test. Written as !(x > tol) so that a NaN
        // norm is also caught: NaN fails every comparison, and the old
        // form silently skipped the push_back and then dereferenced
        // V[k+1] on the next sweep -- an out-of-bounds abort. A genuine
        // lucky breakdown (H == 0, Krylov space exhausted) hits the same
        // path and must also stop rather than continue.
        const bool breakdown = !(H[k+1][k] > 1.0e-300_rt);

        if (!breakdown && k + 1 < m) {
            V.push_back(std::make_unique<MultiFab>(ba, dm, 1, 0));
            MultiFab::Copy(*V[k+1], w, 0, 0, 1, 0);
            V[k+1]->mult(1.0_rt / H[k+1][k]);
        }

        for (int i = 0; i < k; ++i) {
            const Real t = cs[i]*H[i][k] + sn[i]*H[i+1][k];
            H[i+1][k]    = -sn[i]*H[i][k] + cs[i]*H[i+1][k];
            H[i][k]      = t;
        }
        Real den = std::sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
        if (den < 1.0e-300_rt) { den = 1.0e-300_rt; }
        cs[k] = H[k][k]   / den;
        sn[k] = H[k+1][k] / den;
        H[k][k]   = cs[k]*H[k][k] + sn[k]*H[k+1][k];
        H[k+1][k] = 0.0_rt;

        const Real gt = cs[k]*g[k];
        g[k+1] = -sn[k]*g[k];
        g[k]   = gt;

        // Stop before the next sweep would touch a basis vector that
        // was never created.
        if (final_resid != nullptr) { *final_resid = std::abs(g[k+1]) / beta; }

        if (breakdown) { break; }
        if (static_cast<std::size_t>(k + 1) >= V.size()) { break; }
        if (std::abs(g[k+1]) / beta < p.gmres_rtol) { break; }
    }

    std::vector<Real> y(kend, 0.0_rt);
    for (int i = kend - 1; i >= 0; --i) {
        Real s = g[i];
        for (int j = i + 1; j < kend; ++j) { s -= H[i][j] * y[j]; }
        y[i] = (std::abs(H[i][i]) > 1.0e-300_rt) ? s / H[i][i] : 0.0_rt;
    }
    for (int i = 0; i < kend; ++i) {
        MultiFab::Saxpy(dE, y[i], *V[i], 0, 0, 1, 0);
    }

    return kend;
}

// ---------------------------------------------------------------------
// Result cache for repeated identical solves.
//
// Castro evaluates ext_src more than once per timestep, and the
// old-time evaluations are made from the SAME state -- the logs show
// byte-identical |F| sequences and byte-identical dsrc for consecutive
// calls. Recomputing a full Newton-Krylov solve for that is pure waste.
//
// The key is (time, dt) plus a checksum of the input energy field. The
// checksum is what makes this safe: if anything about the state differs
// the cache is rejected and the solve runs normally. Three global
// reductions cost microseconds against a solve costing ~100 residual
// evaluations.
// ---------------------------------------------------------------------
struct SolveCache {
    bool     valid = false;
    Real     time  = -1.0_rt;
    Real     dt    = -1.0_rt;
    Real     sum   = 0.0_rt;
    Real     lo    = 0.0_rt;
    Real     hi    = 0.0_rt;
    BoxArray ba;
    MultiFab dsrc_e;
    MultiFab dsrc_fx;
    bool     converged = false;
};

inline SolveCache& solve_cache ()
{
    static SolveCache c;
    return c;
}

// ---------------------------------------------------------------------
// Top-level driver.
//
//  state_in : full state, const, with >= 1 filled ghost cell. Never
//             written to.
//  dsrc_e   : OUT, (E^{n+1} - E^n)/dt  -> caller adds to UEINT and UEDEN.
//
//  dsrc_fx  : OUT, (rho f^{n+1} - rho f^n)/dt -> caller adds to UFX.
//
// The aux source is the EXACT change implied by holding E_h fixed,
// f^{n+1} = E_h/E^{n+1}, which is valid for arbitrarily large dE. The
// line search enforces E > E_h, so f^{n+1} is in [0,1] by construction
// -- that guarantee did not exist in an earlier revision, which is why
// exporting this source previously drove f past 1. Downstream,
// problem_source must NOT also apply -(f/e)*e_dot to the diffusion
// energy, or the aux change is counted twice.
//
// Returns true if Newton converged.
// ---------------------------------------------------------------------
inline bool
solve (const MultiFab& state_in, MultiFab& dsrc_e, MultiFab& dsrc_fx,
       const Geometry& geom, Real dt, int nGrow, const Params& p = Params(),
       Real time = -1.0_rt)
{
    const BoxArray&            ba = state_in.boxArray();
    const DistributionMapping& dm = state_in.DistributionMap();
    const int ncomp = state_in.nComp();

    // ---- MODE_OFF: contribute nothing -------------------------------
    if (p.mode == MODE_OFF) {
        dsrc_e.setVal(0.0_rt);
        dsrc_fx.setVal(0.0_rt);
        if (p.verbose) {
            amrex::Print() << "[SH-JFNK] mode=OFF, zero diffusion source\n";
        }
        return true;
    }

    // ---- cache lookup ----------------------------------------------
    // Reuse the result if this is a repeat of the immediately preceding
    // solve. Guarded by (time, dt), the BoxArray (rejects a regrid) and
    // a three-number checksum of the input energy.
    Real ck_sum = 0.0_rt, ck_lo = 0.0_rt, ck_hi = 0.0_rt;
    if (p.use_cache && time >= 0.0_rt) {
        ck_sum = state_in.norm2(UEINT);
        ck_lo  = state_in.min(UEINT, 0);
        ck_hi  = state_in.max(UEINT, 0);

        SolveCache& c = solve_cache();
        if (c.valid && c.time == time && c.dt == dt &&
            c.sum == ck_sum && c.lo == ck_lo && c.hi == ck_hi &&
            c.ba == state_in.boxArray())
        {
            MultiFab::Copy(dsrc_e,  c.dsrc_e,  0, 0, 1, 0);
            MultiFab::Copy(dsrc_fx, c.dsrc_fx, 0, 0, 1, 0);
            if (p.verbose) {
                amrex::Print() << "[SH-JFNK] identical to previous solve, reusing result\n";
            }
            return c.converged;
        }
    }

    // Working copy. We deliberately copy the VALID REGION ONLY (nghost
    // = 0) and then fill ghosts ourselves. Inheriting the caller's ghost
    // cells is unsafe: on the new-time source evaluation they may never
    // have been filled, and a zero density there divides to NaN inside
    // the EOS, which then propagates through the entire stencil.
    const int nG = amrex::max(nGrow, 1);
    MultiFab state(ba, dm, ncomp, nG);
    MultiFab::Copy(state, state_in, 0, 0, ncomp, 0);
    fill_ghosts(state, geom);

    // ---- MODE_EXPLICIT: one operator evaluation, no solve -----------
    // Isolates the discretization + conductivity from the solver.
    if (p.mode == MODE_EXPLICIT) {
        sh_diff::compute_L(state, dsrc_e, geom);
        dsrc_e.mult(p.source_scale);
        dsrc_fx.setVal(0.0_rt);
        if (p.verbose) {
            amrex::Print() << "[SH-JFNK] mode=EXPLICIT  L min/max = "
                           << dsrc_e.min(0) << " / " << dsrc_e.max(0) << "\n";
        }
        return true;
    }

    // Frozen heavy-species energy density, valid on the grown region.
    // E_h = E * f = UEINT * UFX / rho.  Constant for the whole substep:
    // electron conduction must not move energy into or out of the heavies.
    MultiFab Eh(ba, dm, 1, nG);
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(Eh, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        Array4<Real const> const s  = state.const_array(mfi);
        Array4<Real>       const eh = Eh.array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real rho = s(i,j,k,URHO);
            const Real E   = s(i,j,k,UEINT);
            if (rho > 0.0_rt && amrex::Math::isfinite(E)) {
                const Real f = amrex::min(amrex::max(s(i,j,k,UFX)/rho, 0.0_rt), 1.0_rt);
                eh(i,j,k) = E * f;
            } else {
                eh(i,j,k) = 0.0_rt;
            }
        });
    }

    // Backward-Euler anchor E^n.
    MultiFab En(ba, dm, 1, 0);
    MultiFab::Copy(En, state, UEINT, 0, 1, 0);

    MultiFab F(ba, dm, 1, 0);
    MultiFab dE(ba, dm, 1, 0);
    MultiFab base(ba, dm, 1, 0);
    MultiFab diag(ba, dm, 1, 0);   // preconditioner, rebuilt once per Newton iteration

    // One workspace (cell coefficients + face conductivities) reused by
    // every residual and preconditioner build in this solve, so nothing
    // is reallocated across the hundreds of evaluations.
    sh_diff::Workspace wsp;
    wsp.define(ba, dm);

    // Record the Te bounds BEFORE the solve. The maximum principle
    // constrains Te -- the quantity the operator actually diffuses --
    // not E. Since E_e = E - E_h = c*Te with c ~ n_e varying in space,
    // a cell with large n_e can gain E while its Te falls, so bounds on
    // Te do not transfer to E. Checking E was a false alarm.
    Real Te_lo_old = 0.0_rt, Te_hi_old = 0.0_rt;
    if (p.verbose) {
        sh_diff::compute_coeffs(state, wsp.coef);
        Te_lo_old = wsp.coef.min(sh_diff::CTE);
        Te_hi_old = wsp.coef.max(sh_diff::CTE);
    }

    residual(state, En, F, geom, dt, &wsp);

    Real Fnorm = F.norm2(0);

    // Fail fast and informatively rather than crashing several layers
    // down inside the Krylov solve.
    if (!amrex::Math::isfinite(Fnorm)) {
        amrex::Print() << "[SH-JFNK] ERROR: initial residual is not finite.\n"
                       << "          valid-region rho  min/max = "
                       << state.min(URHO,  0) << " / " << state.max(URHO,  0) << "\n"
                       << "          valid-region eint min/max = "
                       << state.min(UEINT, 0) << " / " << state.max(UEINT, 0) << "\n"
                       << "          ghost-incl.  rho  min/max = "
                       << state.min(URHO,  nG) << " / " << state.max(URHO,  nG) << "\n"
                       << "          ghost-incl.  eint min/max = "
                       << state.min(UEINT, nG) << " / " << state.max(UEINT, nG) << "\n"
                       << "          Skipping diffusion this step (zero source).\n";
        dsrc_e.setVal(0.0_rt);
        dsrc_fx.setVal(0.0_rt);
        return false;
    }

    // Tolerance is anchored to the ENERGY scale, not to F itself. A
    // relative-to-F0 test would declare success on iteration zero
    // whenever the diffusion term happens to be small.
    const Real escale = amrex::max(En.norm2(0), 1.0e-300_rt);
    const Real tol    = p.newton_rtol * escale;

    bool converged = (Fnorm <= tol);

    // ---- adaptive stiffness test -----------------------------------
    //
    // sigma = dt*|L|/E_e per cell = |F|/E_e at entry, since
    // F(E^n) = -dt*L(E^n). Take the worst cell: if no cell would change
    // its electron energy by more than explicit_sigma over this step,
    // the diffusion is not stiff and a single explicit evaluation is as
    // accurate as the full Newton solve (both are first order in dt and
    // differ by O(sigma)).
    //
    // The explicit update is then E^{n+1} = E^n + dt*L = E^n - F, which
    // costs one kernel instead of ~100 residual evaluations.
    bool took_explicit = false;
    Real sigma_max = 0.0_rt;

    if ((p.mode == MODE_ADAPTIVE || p.verbose) && !converged) {
        MultiFab sig(ba, dm, 1, 0);
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(sig, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<Real const> const ff = F.const_array(mfi);
            Array4<Real const> const en = En.const_array(mfi);
            Array4<Real const> const eh = Eh.const_array(mfi);
            Array4<Real>       const sg = sig.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const Real Ee = amrex::max(en(i,j,k) - eh(i,j,k), 1.0e-300_rt);
                const Real r  = std::abs(ff(i,j,k)) / Ee;
                sg(i,j,k) = amrex::Math::isfinite(r) ? r : 1.0e30_rt;
            });
        }
        sigma_max = sig.max(0);

        if (p.mode == MODE_ADAPTIVE && p.explicit_sigma > 0.0_rt &&
            sigma_max < p.explicit_sigma) {
            // E^{n+1} = E^n - F  (explicit / forward Euler)
#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi(F, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                Array4<Real const> const ff = F.const_array(mfi);
                Array4<Real const> const en = En.const_array(mfi);
                Array4<Real>       const st = state.array(mfi);

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    st(i,j,k,UEINT) = en(i,j,k) - ff(i,j,k);
                });
            }
            fill_ghosts(state, geom, UEINT, 1);
            refresh_aux(state, Eh);

            took_explicit = true;
            converged     = true;   // nothing further to solve
        }
    }

    if (p.verbose) {
        // |F(E^n)| = |dt*L(E^n)|, so this ratio is the explicit
        // stability parameter for the diffusion term. Above ~0.5 an
        // EXPLICIT treatment of conduction is outside its stability
        // limit and is not converging to the right answer.
        //
        // It grows as 1/dx under refinement: explicit stability needs
        // dt ~ dx^2/chi while the hydro CFL only gives dt ~ dx. So an
        // explicit reference solution gets worse at higher resolution
        // while this implicit solve does not -- worth keeping in mind
        // before treating an explicit run as ground truth.
        const Real stab = Fnorm / amrex::max(escale, 1.0e-300_rt);
        amrex::Print() << "[SH-JFNK] enter  |F|=" << Fnorm
                       << "  tol=" << tol
                       << "  Escale=" << escale
                       << "  |dt*L|/|E|=" << stab
                       << (stab > 0.5_rt ? "  (explicit would be UNSTABLE here)" : "")
                       << "\n";
        // Always report the worst-cell stiffness. sigma_max is a MAX over
        // cells, and a diffusion front always contains cold cells whose
        // electron energy E_e is tiny, so even a small absolute flux is a
        // huge relative change there. One such cell forces the whole
        // domain implicit. If this number is persistently well above
        // explicit_sigma, raising the threshold will not help -- the
        // problem is genuinely stiff and the implicit solve is required.
        amrex::Print() << "[SH-JFNK] worst-cell sigma=" << sigma_max
                       << " (threshold " << p.explicit_sigma << ")"
                       << (took_explicit ? "  -> EXPLICIT, Newton skipped"
                                         : "  -> implicit solve") << "\n";
    }

    for (int it = 0; it < p.max_newton && !converged && !took_explicit; ++it) {

        build_precond(state, Eh, diag, geom, dt, &wsp);
        Real kres = 0.0_rt;
        const int nkr = gmres(state, F, En, Eh, diag, dE, geom, dt, p, &wsp, &kres);

        // A silent GMRES failure is the worst kind: it returns a partial
        // step and Newton is fed a wrong direction with no complaint.
        //
        // In practice the Krylov count here is resolution INDEPENDENT
        // (~35-40 from 64^2 to 512^2), because the Jacobian is
        // I - dt*dL/dE and its condition number is ~1+sigma, not the
        // condition number of L itself. With sigma small the operator is
        // a mild perturbation of the identity. Only in a strongly stiff
        // phase (sigma >> 1) does the count climb.
        if (kres > p.gmres_rtol) {
            amrex::Print() << "[SH-JFNK] WARNING: GMRES stopped at " << nkr << " iterations"
                           << " with relative residual " << kres
                           << " > gmres_rtol=" << p.gmres_rtol << "\n"
                           << "          The Newton direction is inexact. If this"
                           << " persists, raise max_gmres\n"
                           << "          (0 = auto-size from the grid) or loosen"
                           << " gmres_rtol.\n";
        }

        MultiFab::Copy(base, state, UEINT, 0, 1, 0);

        MultiFab trial(ba, dm, ncomp, nG);
        MultiFab Ftrial(ba, dm, 1, 0);

        Real lambda = 1.0_rt;
        bool ok = false;
        Real Ftrial_norm = Fnorm;

        for (int ls = 0; ls < p.max_linesearch && !ok; ++ls) {
            MultiFab::Copy(trial, state, 0, 0, ncomp, nG);
            apply_update(trial, base, dE, lambda, Eh, geom);

            // PHYSICAL CONSTRAINT: electron conduction moves electron
            // energy only. The heavies' energy E_h is frozen for this
            // substep, so the total must satisfy
            //
            //     E >= E_h        <=>       E_e = E - E_h >= 0
            //
            // Violating this is what produced the runaway: with most of
            // the energy in the heavies (f -> 1) the electron budget E_e
            // is tiny, and a step that withdraws more than E_e drives
            // E_e negative. That gives either f > 1 (negative T_e) or,
            // through the linearized f-update, f < 0 and hence
            // E_e = E(1-f) > E -- an electron temperature larger than
            // the total energy supports. Because kappa ~ T_e^(5/2), a
            // too-large T_e immediately produces a larger source, which
            // pushes f further out of range: exponential blow-up.
            //
            // Rejecting the step here bounds |dE| by the available
            // electron energy automatically, with no artificial cap.
            MultiFab Ee(ba, dm, 1, 0);
            MultiFab::Copy(Ee, trial, UEINT, 0, 1, 0);
            MultiFab::Subtract(Ee, Eh, 0, 0, 1, 0);

            if (!(Ee.min(0) > 0.0_rt) || !(trial.min(UEINT, 0) > 0.0_rt)) {
                lambda *= 0.5_rt;
                continue;
            }

            residual(trial, En, Ftrial, geom, dt, &wsp);
            Ftrial_norm = Ftrial.norm2(0);

            if (Ftrial_norm < Fnorm || Ftrial_norm <= tol) {
                ok = true;
            } else {
                lambda *= 0.5_rt;
            }
        }

        if (!ok) {
            // Every backtrack failed. Committing the last trial anyway
            // would install a step known to INCREASE the residual, which
            // is how a converging solve turns into a diverging one.
            // Keep the current iterate and stop.
            if (p.verbose) {
                amrex::Print() << "[SH-JFNK]   it=" << it
                               << "  line search failed, keeping previous iterate\n";
            }
            break;
        }

        MultiFab::Copy(state, trial, 0, 0, ncomp, nG);
        MultiFab::Copy(F, Ftrial, 0, 0, 1, 0);
        Fnorm = Ftrial_norm;
        converged = (Fnorm <= tol);

        if (p.verbose) {
            amrex::Print() << "[SH-JFNK]   it=" << it
                           << "  krylov=" << nkr
                           << "  lambda=" << lambda
                           << "  |F|=" << Fnorm
                           << (converged ? "  CONVERGED" : "") << "\n";
        }
    }

    // Effective source rates for Castro to apply.
    //
    // Positivity is enforced by ONE GLOBAL scale factor, not by a
    // per-cell clamp. The implicit update is a flux divergence: what one
    // cell loses, its neighbour gains, and the sum telescopes to zero.
    // Clamping one cell without touching its neighbour breaks that
    // telescoping and creates energy from nothing. Multiplying the whole
    // field by a single constant leaves it a conservative flux
    // divergence, so total energy is still conserved exactly.
    // Effective source rates for Castro to apply.
    //
    // The aux source is the EXACT change implied by freezing E_h:
    //
    //     f^{n+1} = E_h / E^{n+1}
    //
    // valid for any dE, not just small ones. The line search enforces
    // E > E_h, so f^{n+1} lands in [0,1] by construction. This replaces
    // the linearization df = -f dE/E, which is only valid for
    // |dE| << E -- and the stiffness here routinely gives |dE|/E ~ 50,
    // where that approximation is meaningless. Clamping dE to keep the
    // approximation valid throttled the physics to protect the
    // approximation, which is the wrong way round.
    //
    // Consequently there is NO heating cap. The only remaining bound is
    // on cooling, where the electron energy E_e is a genuine physical
    // budget -- electron conduction cannot draw on the heavies. The
    // line search already enforces that; max_frac_drop keeps a little
    // margin because Castro applies the source to a state that hydro
    // has since advanced.
    constexpr Real max_frac_drop = 0.9_rt;

    const Real Ee_floor = 1.0e-10_rt * amrex::max(En.max(0), 1.0e-300_rt);

    MultiFab dEmf (ba, dm, 1, 0);
    MultiFab dFmf (ba, dm, 1, 0);
    MultiFab ratio(ba, dm, 1, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dEmf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const snew = state.const_array(mfi);
        Array4<Real const> const sold = state_in.const_array(mfi);
        Array4<Real const> const eh   = Eh.const_array(mfi);
        Array4<Real>       const de   = dEmf.array(mfi);
        Array4<Real>       const df   = dFmf.array(mfi);
        Array4<Real>       const rt   = ratio.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real Eold = sold(i,j,k,UEINT);
            const Real Ee   = Eold - eh(i,j,k);

            Real dE = snew(i,j,k,UEINT) - Eold;
            if (!amrex::Math::isfinite(dE)) { dE = 0.0_rt; }
            de(i,j,k) = dE;

            Real dF = snew(i,j,k,UFX) - sold(i,j,k,UFX);
            if (!amrex::Math::isfinite(dF)) { dF = 0.0_rt; }

            // Bound the aux change so f stays strictly inside [0,1].
            //
            // Inside the solve f = E_h/E is bounded by construction, but
            // the EXPORTED change is not safe as a source term: a cold
            // cell at the foot of the front sits at f ~ 1 and is heated
            // to f ~ 0, so one application moves f by nearly the entire
            // physical range. Castro applies that to a state hydro has
            // advanced, and combines the old-time and new-time
            // evaluations, so f can overshoot past 0. Then
            // E_e = E(1-f) > E and T_e exceeds what the total energy
            // supports -- a single-cell temperature spike with total
            // energy still perfectly conserved, which drives a pressure
            // spike and blows the cell open.
            //
            // Unlike the ENERGY source, clamping this per cell is safe:
            // UFX carries no energy, only the electron/heavy split, so
            // a per-cell bound cannot break the flux-divergence
            // telescoping and cannot create or destroy energy.
            //
            // aux_safety < 0.5 leaves headroom for the source being
            // applied more than once across the old/new combination.
            {
                constexpr Real aux_safety = 0.4_rt;
                const Real rho_c = sold(i,j,k,URHO);
                const Real ufx_c = sold(i,j,k,UFX);
                if (rho_c > 0.0_rt) {
                    const Real lo = -aux_safety * amrex::max(ufx_c, 0.0_rt);
                    const Real hi =  aux_safety * amrex::max(rho_c - ufx_c, 0.0_rt);
                    dF = amrex::min(amrex::max(dF, lo), hi);
                } else {
                    dF = 0.0_rt;
                }
            }
            df(i,j,k) = dF;

            // Cooling only. Heating is unconstrained: the exact aux
            // relation handles arbitrarily large dE > 0 without any
            // approximation breaking down.
            Real r = 1.0_rt;
            if (dE < 0.0_rt && Ee > Ee_floor) {
                r = amrex::min(1.0_rt, max_frac_drop * Ee / (-dE));
            }
            rt(i,j,k) = r;
        });
    }

    Real lam = ratio.min(0);
    if (!amrex::Math::isfinite(lam) || lam < 0.0_rt) { lam = 0.0_rt; }
    lam = amrex::min(lam, 1.0_rt);

    MultiFab::Copy(dsrc_e,  dEmf, 0, 0, 1, 0);
    MultiFab::Copy(dsrc_fx, dFmf, 0, 0, 1, 0);
    dsrc_e .mult(p.source_scale * lam / dt);
    dsrc_fx.mult(p.source_scale * lam / dt);

    if (p.verbose) {
        amrex::Print() << "[SH-JFNK] global scaling lambda=" << lam
                       << (lam < 0.99_rt ? "  (source damped -- conduction is being discarded)"
                                         : "  (full solve delivered)") << "\n";
    }

    // Maximum-principle check, on Te.
    //
    // Backward Euler with non-negative face coefficients is an
    // M-matrix, so the converged Te must lie within the range of the
    // starting Te. If it does not, the solver itself is at fault. If it
    // does, but the run still misbehaves, then what Castro applies
    // differs from what was computed here -- look at how the old-time
    // and new-time sources are combined, not at this solver.
    //
    // NOTE: this deliberately does NOT check E. E is not the diffused
    // quantity, and its bounds legitimately widen when n_e varies in
    // space.
    if (p.verbose) {
        sh_diff::compute_coeffs(state, wsp.coef);
        const Real Te_lo_new = wsp.coef.min(sh_diff::CTE);
        const Real Te_hi_new = wsp.coef.max(sh_diff::CTE);
        const Real slack = 1.0e-8_rt * amrex::max(std::abs(Te_hi_old), 1.0_rt);

        const bool violated = (Te_hi_new > Te_hi_old + slack) ||
                              (Te_lo_new < Te_lo_old - slack);

        amrex::Print() << "[SH-JFNK] max-principle(Te): [" << Te_lo_old << ", " << Te_hi_old
                       << "] -> [" << Te_lo_new << ", " << Te_hi_new << "]"
                       << (violated ? "   *** VIOLATED: solver at fault ***"
                                    : "   ok") << "\n";
    }

    if (p.verbose) {
        // Worst-case f after one application of the aux source. f must
        // stay inside [0,1]: outside it the electron energy E(1-f) is
        // either negative (f>1, Te<0) or larger than the total energy
        // (f<0, Te spikes), both while total energy stays conserved --
        // which is what makes it hard to spot without this line.
        MultiFab fpred(ba, dm, 1, 0);
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(fpred, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<Real const> const sold = state_in.const_array(mfi);
            Array4<Real const> const dfx  = dsrc_fx.const_array(mfi);
            Array4<Real>       const fp   = fpred.array(mfi);
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const Real rho_c = sold(i,j,k,URHO);
                fp(i,j,k) = (rho_c > 0.0_rt)
                          ? (sold(i,j,k,UFX) + dt * dfx(i,j,k)) / rho_c
                          : 0.5_rt;
            });
        }
        const Real fmin = fpred.min(0);
        const Real fmax = fpred.max(0);
        amrex::Print() << "[SH-JFNK] f after source in [" << fmin << ", " << fmax << "]"
                       << ((fmin < 0.0_rt || fmax > 1.0_rt)
                               ? "   *** OUT OF RANGE: Te will be wrong ***" : "   ok")
                       << "\n";

        amrex::Print() << "[SH-JFNK] exit   dsrc_e  min/max = "
                       << dsrc_e.min(0)  << " / " << dsrc_e.max(0)  << "\n"
                       << "[SH-JFNK]        dsrc_fx min/max = "
                       << dsrc_fx.min(0) << " / " << dsrc_fx.max(0) << "\n";
    }

    if (p.use_cache && time >= 0.0_rt) {
        SolveCache& c = solve_cache();
        c.dsrc_e .define(ba, dm, 1, 0);
        c.dsrc_fx.define(ba, dm, 1, 0);
        MultiFab::Copy(c.dsrc_e,  dsrc_e,  0, 0, 1, 0);
        MultiFab::Copy(c.dsrc_fx, dsrc_fx, 0, 0, 1, 0);
        c.time = time; c.dt = dt;
        c.sum = ck_sum; c.lo = ck_lo; c.hi = ck_hi;
        c.ba = state_in.boxArray();
        c.converged = converged;
        c.valid = true;
    }

    return converged;
}

} // namespace sh_jfnk

#endif