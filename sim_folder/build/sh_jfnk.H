#ifndef SH_JFNK_H
#define SH_JFNK_H
//
// Jacobian-Free Newton-Krylov solve of backward-Euler electron thermal
// diffusion for a two-temperature Castro setup.
//
//   F(E) = E - E^n - dt * div( kappa(Te) grad Te ) = 0
//
// E = UEINT is the TOTAL internal energy density. The heavy-species
// energy E_h is frozen across the sub-step -- electron conduction does
// not appear in the heavy energy equation -- so at every iterate
//
//   f = E_h / E,    UFX = rho * f,
//
// which is what makes the EOS return the electron temperature.
//
#include <sh_diffusion_operator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <vector>
#include <string>
#include <memory>

using namespace amrex;

namespace sh_jfnk {

//   MODE_OFF       contribute nothing. Use with castro.diffuse_temp = 1
//                  to run Castro's own explicit diffusion instead.
//   MODE_IMPLICIT  backward-Euler JFNK (the real scheme).

enum DiffMode { MODE_OFF = 0, MODE_IMPLICIT = 1 };

struct Params {
    DiffMode mode = MODE_IMPLICIT;

    // Castro evaluates ext_src more than once per step from the same
    // old-time state. Reuse the result when the input is identical;
    // guarded by a checksum, so a mismatch falls through to a real solve.
    bool use_cache = true;

    // Multiplies the returned source. Castro combines the old-time and
    // new-time evaluations; try 0.5 to test whether that over-applies.
    Real source_scale = 1.0_rt;

    // Newton. Tolerance is relative to the energy scale, not to the
    // initial residual, which would be self-referential.
    int  max_newton     = 20;
    Real newton_rtol    = 1.0e-6_rt;
    int  max_linesearch = 8;
    Real fd_eps         = 1.0e-7_rt;   // finite-difference step for J*v

    // Krylov. max_gmres = 0 auto-sizes from the grid. gmres_rtol is
    // deliberately loose: resolving the linear system beyond the
    // accuracy of the current Newton direction is wasted work.
    int  max_gmres   = 0;
    Real gmres_rtol  = 1.0e-4_rt;

    // Second Gram-Schmidt pass when orthogonalization collapses the
    // vector norm by more than this factor. 0 disables.
    Real reorth_tol = 0.7_rt;

    // Per-step diagnostics. Costs two extra coefficient evaluations.
    bool verbose = false;
};


// Runtime parameters, read once from the inputs file.
//
//   implicit scheme    sh_diffusion.mode   = implicit
//                      castro.diffuse_temp = 0
//
//   explicit baseline  sh_diffusion.mode   = off
//                      castro.diffuse_temp = 1
//
// Any Params field may be overridden, e.g. sh_diffusion.newton_rtol.

inline Params& runtime_params ()
{
    static Params p;
    static bool initialised = false;
    if (initialised) { return p; }
    initialised = true;

    amrex::ParmParse pp("sh_diffusion");

    std::string m;
    if (pp.query("mode", m)) {
        if      (m == "off")      { p.mode = MODE_OFF; }
        else if (m == "implicit") { p.mode = MODE_IMPLICIT; }
        else { amrex::Abort("sh_diffusion.mode must be 'off' or 'implicit'"); }
    }

    pp.query("source_scale", p.source_scale);
    pp.query("max_newton",   p.max_newton);
    pp.query("newton_rtol",  p.newton_rtol);
    pp.query("max_gmres",    p.max_gmres);
    pp.query("gmres_rtol",   p.gmres_rtol);
    pp.query("reorth_tol",   p.reorth_tol);
    pp.query("fd_eps",       p.fd_eps);

    int itmp = p.verbose ? 1 : 0;
    if (pp.query("verbose", itmp))   { p.verbose = itmp; }
    itmp = p.use_cache ? 1 : 0;
    if (pp.query("use_cache", itmp)) { p.use_cache = itmp; }

    // Castro's own temperature diffusion is EXPLICIT (it evaluates
    // div(kappa grad T) and adds it as a source, with estdt_temp_diffusion
    // limiting dt). Running it alongside this solver applies conduction
    // twice; running neither applies none. Both are silent, so catch them.
    int castro_diffuse_temp = 0;
    { amrex::ParmParse pc("castro"); pc.query("diffuse_temp", castro_diffuse_temp); }

    const bool we_diffuse = (p.mode != MODE_OFF);

    if (we_diffuse && castro_diffuse_temp == 1) {
        amrex::Abort("Conduction applied TWICE: sh_diffusion.mode != off and "
                     "castro.diffuse_temp = 1. Set one of them off.");
    }
    if (!we_diffuse && castro_diffuse_temp != 1) {
        amrex::Print() << "[SH-JFNK] WARNING: no thermal conduction anywhere "
                       << "(mode = off, castro.diffuse_temp != 1).\n";
    }

    if (p.verbose) {
        amrex::Print() << "[SH-JFNK] mode = "
                       << ((p.mode == MODE_OFF) ? "off" : "implicit")
                       << ", castro.diffuse_temp = " << castro_diffuse_temp << "\n";
    }
    return p;
}


// Ghost fill.
//
// FillBoundary covers same-level and periodic ghosts. Ghosts OUTSIDE the
// domain are then filled by zero-gradient extrapolation, a no-flux
// (adiabatic) thermal wall -- change here for a Dirichlet wall.
//
// Ghosts at a COARSE-FINE boundary are touched by neither, and are left
// at the FillPatch values the caller supplied.
//
// The extrapolation source is clamped to the fab's own valid box, not to
// the domain: under domain decomposition a box interior to the domain
// would otherwise index outside its own data.

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
        const auto vlo = amrex::lbound(mfi.validbox());
        const auto vhi = amrex::ubound(mfi.validbox());
        Array4<Real> const s = state.array(mfi);

        amrex::ParallelFor(gbx, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) noexcept
        {
            const int n = scomp + nn;
            bool outside = false;

            if (per[0] == 0 && (i < dlo.x || i > dhi.x)) { outside = true; }
#if AMREX_SPACEDIM >= 2
            if (per[1] == 0 && (j < dlo.y || j > dhi.y)) { outside = true; }
#endif
#if AMREX_SPACEDIM == 3
            if (per[2] == 0 && (k < dlo.z || k > dhi.z)) { outside = true; }
#endif
            if (outside) {
                const int ii = amrex::min(amrex::max(i, vlo.x), vhi.x);
#if AMREX_SPACEDIM >= 2
                const int jj = amrex::min(amrex::max(j, vlo.y), vhi.y);
#else
                const int jj = j;
#endif
#if AMREX_SPACEDIM == 3
                const int kk = amrex::min(amrex::max(k, vlo.z), vhi.z);
#else
                const int kk = k;
#endif
                s(i,j,k,n) = s(ii,jj,kk,n);
            }
        });
    }
}

inline void
fill_ghosts (MultiFab& state, const Geometry& geom)
{
    fill_ghosts(state, geom, 0, state.nComp());
}


// Refresh UFX = rho * E_h/E over valid + ghost cells, so the EOS sees an
// f consistent with the current energy iterate.
//
// f is bounded strictly inside (0,1): at f = 1 the electron energy is
// exactly zero and Te degenerate, which restarts the kappa ~ Te^(5/2)
// feedback loop.

inline void
refresh_aux (MultiFab& state, const MultiFab& Eh)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        Array4<Real>       const s  = state.array(mfi);
        Array4<Real const> const eh = Eh.const_array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real E   = s(i,j,k,UEINT);
            const Real rho = s(i,j,k,URHO);
            if (E > 0.0_rt && rho > 0.0_rt && amrex::Math::isfinite(eh(i,j,k))) {
                constexpr Real f_min = 1.0e-12_rt;
                constexpr Real f_max = 1.0_rt - 1.0e-8_rt;
                const Real f = amrex::min(amrex::max(eh(i,j,k) / E, f_min), f_max);
                s(i,j,k,UFX) = rho * f;
            }
        });
    }
}

// F(E) = E - E^n - dt*L(E). `state` needs filled ghosts and an aux
// consistent with its UEINT.
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

// UEINT = base + lambda*delta, then restore consistency.
//
// No clamping here: clamping inside the iteration makes Newton converge
// on a modified equation. Positivity is enforced by the line search.
//
// Only UEINT ghosts are exchanged. refresh_aux then derives UFX over the
// grown box, and rho/species/temperature are frozen for the whole solve.
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
            s(i,j,k,UEINT) = b(i,j,k) + lambda * d(i,j,k);
        });
    }
    fill_ghosts(state, geom, UEINT, 1);
    refresh_aux(state, Eh);
}


// Matrix-free J*v ~= [F(E + eps*v) - F(E)] / eps
//
// Both norms in eps must be the same norm; mixing L2 for the state with
// L-infinity for the direction inflates eps by ~sqrt(N).
//
// pert and base are supplied by the caller and reused: only UEINT
// changes per Krylov iteration, so the full state copy happens once per
// solve rather than once per iteration.

inline void
jac_vec (const MultiFab& state, const MultiFab& F0, const MultiFab& v,
         MultiFab& Jv, const MultiFab& En, const MultiFab& Eh,
         const Geometry& geom, Real dt, const Params& p,
         MultiFab* pert_ws = nullptr, MultiFab* base_ws = nullptr,
         sh_diff::Workspace* ws = nullptr)
{
    const Real norm_E = state.norm2(UEINT);
    const Real norm_v = v.norm2(0);
    const Real eps = p.fd_eps * (norm_E + 1.0_rt) / (norm_v + 1.0e-300_rt);

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


// Preconditioner: the exact diagonal of the discretized Jacobian,
//
//   d = 1 + dt * sum_dir m_dir * (k_lo + k_hi) * dTe/dE,
//
// using the same face coefficients and cell metric as the operator, so
// the two cannot drift apart. dTe/dE = Te/E_e is exact for a gamma-law
// electron gas.
//
// The diagonal depends only on the Newton iterate, so it is constant
// across a GMRES solve and is built once per Newton iteration. It never
// enters the residual: an imperfect diagonal costs iterations, not
// accuracy.

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

    // *Array() returns by value; geom.CellSize() would be a host pointer
    // captured into a device lambda.
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
        Array4<Real> const d = diag.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real Te_c = cc(i,j,k,sh_diff::CTE);

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

// One divide per cell, no EOS calls.
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


// Left-preconditioned GMRES for J dE = -F.
//
// Modified Gram-Schmidt, with a second pass when orthogonalization
// collapses the norm (deep Krylov spaces lose orthogonality, and the
// Givens recurrence would still report convergence). Breakdown is tested
// as !(h > tol) so a NaN norm is caught as well as a genuine one.

inline int
gmres (const MultiFab& state, const MultiFab& F0, const MultiFab& En,
       const MultiFab& Eh, const MultiFab& diag, MultiFab& dE,
       const Geometry& geom, Real dt, const Params& p,
       sh_diff::Workspace* ws = nullptr, Real* final_resid = nullptr)
{
    const BoxArray&            ba = F0.boxArray();
    const DistributionMapping& dm = F0.DistributionMap();

    dE.setVal(0.0_rt);

    // Allocated once per solve and reused by every jac_vec call.
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
    if (!(beta > 1.0e-300_rt)) { return 0; }

    int m = p.max_gmres;
    if (m <= 0) {
        const Box& dom = geom.Domain();
        int nmax = dom.length(0);
        for (int d = 1; d < AMREX_SPACEDIM; ++d) { nmax = amrex::max(nmax, dom.length(d)); }
        m = amrex::max(amrex::min(2 * nmax, 400), 40);
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

        for (int i = 0; i <= k; ++i) {
            H[i][k] = MultiFab::Dot(w, 0, *V[i], 0, 1, 0);
            MultiFab::Saxpy(w, -H[i][k], *V[i], 0, 0, 1, 0);
        }
        H[k+1][k] = w.norm2(0);

        if (p.reorth_tol > 0.0_rt && H[k+1][k] < p.reorth_tol * wnorm_before) {
            for (int i = 0; i <= k; ++i) {
                const Real corr = MultiFab::Dot(w, 0, *V[i], 0, 1, 0);
                H[i][k] += corr;
                MultiFab::Saxpy(w, -corr, *V[i], 0, 0, 1, 0);
            }
            H[k+1][k] = w.norm2(0);
        }

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


// Result cache. Castro evaluates ext_src more than once per step from
// the same old-time state; the key is (time, dt), the BoxArray and a
// three-number checksum of the input energy.

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


// Top-level driver.
//
//   state_in : full state, const, with >= 1 FillPatch'd ghost cell.
//   dsrc_e   : OUT, (E^{n+1} - E^n)/dt        -> caller adds to UEINT, UEDEN.
//   dsrc_fx  : OUT, (rho f^{n+1} - rho f^n)/dt -> caller adds to UFX.
//
// The aux source is the EXACT change implied by freezing E_h, valid for
// any dE. The line search enforces E > E_h, so f stays in [0,1] by
// construction. problem_source must NOT also put the conduction energy
// through -(f/e)*Edot, or the aux change is counted twice.
//
// Returns true if Newton converged.

inline bool
solve (const MultiFab& state_in, MultiFab& dsrc_e, MultiFab& dsrc_fx,
       const Geometry& geom, Real dt, int nGrow, const Params& p = Params(),
       Real time = -1.0_rt)
{
    const BoxArray&            ba = state_in.boxArray();
    const DistributionMapping& dm = state_in.DistributionMap();
    const int ncomp = state_in.nComp();

    // ---- MODE_OFF ---------------------------------------------------
    if (p.mode == MODE_OFF) {
        dsrc_e.setVal(0.0_rt);
        dsrc_fx.setVal(0.0_rt);
        return true;
    }

    // ---- cache lookup -----------------------------------------------
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
                amrex::Print() << "[SH-JFNK] repeat solve, reusing result\n";
            }
            return c.converged;
        }
    }

    // Working copy INCLUDING the caller's ghosts. Under AMR only
    // FillPatch (done by Castro's expand_state) can fill coarse-fine
    // ghosts; copying the valid region alone discarded them and the
    // stencil then read stale memory along every refinement interface.
    // Keeping them freezes the coarse-fine ghosts at their interpolated
    // values for the solve -- a Dirichlet condition from the coarse
    // level, lagged by a step but stable.
    const int nG = amrex::max(nGrow, 1);
    const int ncopy = amrex::min(nG, state_in.nGrow());
    MultiFab state(ba, dm, ncomp, nG);
    MultiFab::Copy(state, state_in, 0, 0, ncomp, ncopy);
    fill_ghosts(state, geom);

    if (!(state.min(URHO, nG) > 0.0_rt) ||
        !amrex::Math::isfinite(state.min(UEINT, nG)) ||
        !amrex::Math::isfinite(state.max(UEINT, nG)))
    {
        amrex::Print() << "[SH-JFNK] WARNING: ghost cells look invalid (rho min = "
                       << state.min(URHO, nG) << "). Caller must supply a "
                       << "FillPatch'd state with >= 1 ghost cell.\n";
    }

    // E_h = E*f, frozen for the whole sub-step: electron conduction does
    // not move energy into or out of the heavies.
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

    MultiFab En(ba, dm, 1, 0);
    MultiFab::Copy(En, state, UEINT, 0, 1, 0);

    MultiFab F(ba, dm, 1, 0), dE(ba, dm, 1, 0);
    MultiFab base(ba, dm, 1, 0), diag(ba, dm, 1, 0);

    // Cell coefficients + face conductivities, reused by every residual
    // and preconditioner build in this solve.
    sh_diff::Workspace wsp;
    wsp.define(ba, dm);

    // Te bounds before the solve, for the maximum-principle check below.
    Real Te_lo_old = 0.0_rt, Te_hi_old = 0.0_rt;
    if (p.verbose) {
        sh_diff::compute_coeffs(state, wsp.coef);
        Te_lo_old = wsp.coef.min(sh_diff::CTE);
        Te_hi_old = wsp.coef.max(sh_diff::CTE);
    }

    residual(state, En, F, geom, dt, &wsp);
    Real Fnorm = F.norm2(0);

    if (!amrex::Math::isfinite(Fnorm)) {
        amrex::Print() << "[SH-JFNK] ERROR: initial residual not finite.\n"
                       << "          rho  valid/ghost min = " << state.min(URHO,0)
                       << " / " << state.min(URHO,nG) << "\n"
                       << "          eint valid/ghost min = " << state.min(UEINT,0)
                       << " / " << state.min(UEINT,nG) << "\n"
                       << "          Skipping diffusion this step.\n";
        dsrc_e.setVal(0.0_rt);
        dsrc_fx.setVal(0.0_rt);
        return false;
    }

    const Real escale = amrex::max(En.norm2(0), 1.0e-300_rt);
    const Real tol    = p.newton_rtol * escale;
    bool converged    = (Fnorm <= tol);

    if (p.verbose) {
        // |F(E^n)| = |dt*L(E^n)|, so this ratio is the explicit stability
        // parameter. Above ~0.5 an explicit treatment would be unstable.
        const Real stab = Fnorm / escale;
        amrex::Print() << "[SH-JFNK] enter  |F|=" << Fnorm << "  tol=" << tol
                       << "  |dt*L|/|E|=" << stab
                       << (stab > 0.5_rt ? "  (explicit would be unstable)" : "")
                       << "\n";
    }

    for (int it = 0; it < p.max_newton && !converged; ++it) {

        build_precond(state, Eh, diag, geom, dt, &wsp);
        Real kres = 0.0_rt;
        const int nkr = gmres(state, F, En, Eh, diag, dE, geom, dt, p, &wsp, &kres);

        // A silent inner failure returns a partial vector and Newton then
        // follows a wrong direction with no indication.
        if (kres > p.gmres_rtol) {
            amrex::Print() << "[SH-JFNK] WARNING: GMRES stopped at " << nkr
                           << " iterations, relative residual " << kres
                           << " > " << p.gmres_rtol
                           << ". Newton direction is inexact.\n";
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

            // Electron conduction can only move electron energy, so
            // E > E_h. Enforcing it here bounds |dE| by the available
            // electron energy and keeps f in [0,1], with no artificial cap.
            MultiFab Ee(ba, dm, 1, 0);
            MultiFab::Copy(Ee, trial, UEINT, 0, 1, 0);
            MultiFab::Subtract(Ee, Eh, 0, 0, 1, 0);

            if (!(Ee.min(0) > 0.0_rt) || !(trial.min(UEINT, 0) > 0.0_rt)) {
                lambda *= 0.5_rt;
                continue;
            }

            residual(trial, En, Ftrial, geom, dt, &wsp);
            Ftrial_norm = Ftrial.norm2(0);

            if (Ftrial_norm < Fnorm || Ftrial_norm <= tol) { ok = true; }
            else                                           { lambda *= 0.5_rt; }
        }

        if (!ok) {
            // Committing a step known to increase the residual is how a
            // converging solve turns into a diverging one.
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
            amrex::Print() << "[SH-JFNK]   it=" << it << "  krylov=" << nkr
                           << "  lambda=" << lambda << "  |F|=" << Fnorm
                           << (converged ? "  CONVERGED" : "") << "\n";
        }
    }

    // ---- source back-out --------------------------------------------
    //
    // ENERGY: bounded by ONE GLOBAL factor. The update is a flux
    // divergence, so what one cell loses its neighbours gain and the sum
    // telescopes; clamping a single cell would create energy. Scaling the
    // whole field by a constant keeps it conservative. Only cooling is
    // bounded -- the electron energy is a real budget for it -- while
    // heating needs no cap, since the exact aux relation handles any dE.
    //
    // AUX: bounded PER CELL. UFX carries no energy, only the
    // electron/heavy split, so a per-cell bound cannot break the
    // telescoping. It is needed because a cold cell at the front moves f
    // across nearly the whole [0,1] range in one application, and Castro
    // applies the source to a state hydro has advanced.
    constexpr Real max_frac_drop = 0.9_rt;
    constexpr Real aux_safety    = 0.4_rt;   // < 0.5: headroom for old/new combination
    const Real Ee_floor = 1.0e-10_rt * amrex::max(En.max(0), 1.0e-300_rt);

    MultiFab dEmf(ba, dm, 1, 0), dFmf(ba, dm, 1, 0), ratio(ba, dm, 1, 0);

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

            Real dE_c = snew(i,j,k,UEINT) - Eold;
            if (!amrex::Math::isfinite(dE_c)) { dE_c = 0.0_rt; }
            de(i,j,k) = dE_c;

            Real dF = snew(i,j,k,UFX) - sold(i,j,k,UFX);
            if (!amrex::Math::isfinite(dF)) { dF = 0.0_rt; }

            const Real rho_c = sold(i,j,k,URHO);
            const Real ufx_c = sold(i,j,k,UFX);
            if (rho_c > 0.0_rt) {
                const Real lo = -aux_safety * amrex::max(ufx_c, 0.0_rt);
                const Real hi =  aux_safety * amrex::max(rho_c - ufx_c, 0.0_rt);
                dF = amrex::min(amrex::max(dF, lo), hi);
            } else {
                dF = 0.0_rt;
            }
            df(i,j,k) = dF;

            Real r = 1.0_rt;
            if (dE_c < 0.0_rt && Ee > Ee_floor) {
                r = amrex::min(1.0_rt, max_frac_drop * Ee / (-dE_c));
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

    // ---- diagnostics -------------------------------------------------
    if (p.verbose) {
        amrex::Print() << "[SH-JFNK] lambda=" << lam
                       << (lam < 0.99_rt ? "  (source damped)" : "") << "\n";

        // Maximum principle applies to Te, the quantity actually
        // diffused -- NOT to E, whose bounds legitimately widen when n_e
        // varies in space.
        sh_diff::compute_coeffs(state, wsp.coef);
        const Real Te_lo_new = wsp.coef.min(sh_diff::CTE);
        const Real Te_hi_new = wsp.coef.max(sh_diff::CTE);
        const Real slack = 1.0e-8_rt * amrex::max(std::abs(Te_hi_old), 1.0_rt);
        const bool violated = (Te_hi_new > Te_hi_old + slack) ||
                              (Te_lo_new < Te_lo_old - slack);
        amrex::Print() << "[SH-JFNK] Te [" << Te_lo_old << ", " << Te_hi_old
                       << "] -> [" << Te_lo_new << ", " << Te_hi_new << "]"
                       << (violated ? "  *** MAX-PRINCIPLE VIOLATED ***" : "  ok")
                       << "\n";

        // f outside [0,1] gives either negative Te or a Te the total
        // energy cannot support, while total energy still balances --
        // hard to spot without this.
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
        amrex::Print() << "[SH-JFNK] f in [" << fmin << ", " << fmax << "]"
                       << ((fmin < 0.0_rt || fmax > 1.0_rt)
                               ? "  *** OUT OF RANGE ***" : "  ok") << "\n"
                       << "[SH-JFNK] dsrc_e  " << dsrc_e.min(0)  << " / " << dsrc_e.max(0)
                       << "   dsrc_fx " << dsrc_fx.min(0) << " / " << dsrc_fx.max(0) << "\n";
    }

    if (p.use_cache && time >= 0.0_rt) {
        SolveCache& c = solve_cache();
        c.dsrc_e .define(ba, dm, 1, 0);
        c.dsrc_fx.define(ba, dm, 1, 0);
        MultiFab::Copy(c.dsrc_e,  dsrc_e,  0, 0, 1, 0);
        MultiFab::Copy(c.dsrc_fx, dsrc_fx, 0, 0, 1, 0);
        c.time = time;   c.dt = dt;
        c.sum  = ck_sum; c.lo = ck_lo; c.hi = ck_hi;
        c.ba   = state_in.boxArray();
        c.converged = converged;
        c.valid = true;
    }

    return converged;
}

} // namespace sh_jfnk

#endif