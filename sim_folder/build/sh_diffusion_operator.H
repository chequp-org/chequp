#ifndef SH_DIFFUSION_OPERATOR_H
#define SH_DIFFUSION_OPERATOR_H
//
// Nonlinear electron thermal diffusion operator for a two-temperature
// (gamma_law_2T) Castro setup, WITH a free-streaming flux limiter.
//
//   L = div( kappa_eff grad Te ),
//   kappa_eff = kappa_SH / (1 + |kappa_SH grad Te| / q_FS)
//
// The limiter matters here, it is not a refinement. Unlimited
// Spitzer-Harm kappa ~ T^(5/2) across a steep front predicts a heat flux
// far larger than the electrons can physically carry, which drains cold
// cells past zero energy in a single step. Negative energy then gives a
// negative Te, and the Coulomb-log expression in get_collision_frequencies
// takes sqrt() of a negative number -> NaN, which propagates into both
// the conductivity and the electron-heavy exchange term Sc.
//
// Conventions (match problem_initialize_state_data.H):
//   URHO  : total mass density
//   UEINT : TOTAL internal energy density (electrons + heavies)
//   UFX   : rho * f, f = E_heavies / E_total
//   UTEMP : electron temperature -- used only as an EOS seed, never trusted.
//
#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <conductivity.H>
#include <collisional_ionization.H>
#include <eos.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array.H>
#include <AMReX_IndexType.H>

using namespace amrex;

namespace sh_diff {

// Free-streaming flux limiter coefficient. 0.03-0.1 is the usual range
// in laser-plasma / ICF practice; smaller clamps harder.
//
// DEFAULT OFF, so the face averaging can be compared against the
// reference in isolation. Turning it on alongside a suppressive
// averaging choice stacks two independent reductions in conduction and
// makes the cause of a slow front impossible to attribute. The sharp
// front dropping vertically to zero is this limiter's signature.
constexpr bool use_flux_limiter = false;
constexpr Real flux_limiter_f   = 0.06_rt;

// Retained for reference. No longer used as a conductivity cutoff: see
// the note in get_Te_kappa_qfs on why zeroing kappa on an n_e threshold
// freezes the conduction front.
constexpr Real min_ne_cgs = 1.0e-30_rt;

// ---------------------------------------------------------------------
// Cell-centred electron temperature, conductivity, and free-streaming
// heat flux magnitude.
//
//   Te    [K]
//   kappa [erg / (cm s K)]
//   q_fs  [erg / (cm^2 s)] = f_lim * n_e * k_B * Te * v_th,e
//
// Every unphysical input is trapped here and turned into a cold,
// non-conducting cell, so that a single bad cell degrades locally
// instead of NaN-ing the whole run.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void get_Te_kappa_qfs (int i, int j, int k,
                       Array4<const Real> const& state,
                       Real& Te, Real& kappa, Real& q_fs)
{
    Te    = castro::small_temp;
    kappa = 0.0_rt;
    q_fs  = 0.0_rt;

    const Real rho  = state(i,j,k,URHO);
    const Real reint = state(i,j,k,UEINT);

    if (!(rho > 0.0_rt) || !amrex::Math::isfinite(rho) ||
        !amrex::Math::isfinite(reint)) {
        return;
    }

    const Real rhoinv = 1.0_rt / rho;

    eos_t eos_state;
    eos_state.rho = rho;
    eos_state.T   = amrex::max(state(i,j,k,UTEMP), castro::small_temp); // seed only
    eos_state.e   = reint * rhoinv;

    // Clamp species. Advection can produce small negative mass
    // fractions, and a negative n_e makes the Coulomb-log sqrt()
    // argument in get_collision_frequencies negative -> NaN.
    //
    // NOTE: this is the ONE input that actual_conductivity() sees
    // modified. It is a NaN guard, and it only alters cells where the
    // state is already unphysical (X outside [0,1]). Remove the clamp
    // if you need bit-identical inputs to the conductivity model.
    //
    // The n_e accumulated here is LOCAL to this function: it gates the
    // "no electrons, no conduction" check and sets the free-streaming
    // flux q_fs. It is NOT the n_e that actual_conductivity() computes
    // internally -- that one is built from its own convention. The two
    // agree on whether n_e is zero, which is all the gate needs.
    Real n_e = 0.0_rt;
    for (int n = 0; n < NumSpec; ++n) {
        const Real X = amrex::min(amrex::max(state(i,j,k,UFS+n) * rhoinv, 0.0_rt), 1.0_rt);
        eos_state.xn[n] = X;
        n_e += zion[n] * X * rho * aion_inv[n] / C::m_u;   // cm^-3
    }

#if NAUX_NET > 0
    // aux[0] = f = E_heavies / E_total.
    //
    // Bound strictly INSIDE (0,1), not just to [0,1]. At f = 1 the
    // electron energy E(1-f) is exactly zero and at f > 1 it is
    // negative; both give a degenerate or negative T_e, and since
    // kappa ~ T_e^(5/2) that feeds straight back into a larger source
    // and a worse f on the next step. Keeping a margin makes the
    // conductivity finite and the feedback loop unable to start.
    constexpr Real f_min = 1.0e-12_rt;
    constexpr Real f_max = 1.0_rt - 1.0e-8_rt;
    for (int n = 0; n < NumAux; ++n) {
        eos_state.aux[n] = amrex::min(amrex::max(state(i,j,k,UFX+n) * rhoinv, f_min), f_max);
    }
#endif

    if (eos_state.e < 0.0_rt) {
        eos_state.T = castro::small_temp;
        eos(eos_input_rt, eos_state);
    } else {
        eos(eos_input_re, eos_state);
    }

    // Trap a bad inversion before it reaches the conductivity model.
    if (!amrex::Math::isfinite(eos_state.T) || eos_state.T <= 0.0_rt) {
        Te = castro::small_temp;
        return;
    }
    Te = amrex::max(eos_state.T, castro::small_temp);
    eos_state.T = Te;

    // Note on n_e: this guard is deliberately loose. In the conductivity
    // model kappa ~ n_e T / nu, and nu_ei is itself proportional to n_e,
    // so n_e CANCELS in the ionized limit -- kappa does not vanish as
    // n_e falls. Zeroing kappa on a generous n_e threshold instead
    // freezes the front: a cell with kappa = 0 cannot receive heat, so
    // it never warms, never ionizes, and never gains conductivity.
    // Only a genuinely non-positive or non-finite n_e is trapped here,
    // because the Coulomb logarithm takes sqrt() of a quantity with n_e
    // in the denominator and would return NaN.
    if (!(n_e > 0.0_rt) || !amrex::Math::isfinite(n_e)) {
        return;
    }

    if (rho > castro::diffuse_cutoff_density) {
        conductivity(eos_state);
        if (rho < castro::diffuse_cutoff_density_hi) {
            const Real mult = (rho - castro::diffuse_cutoff_density) /
                              (castro::diffuse_cutoff_density_hi - castro::diffuse_cutoff_density);
            eos_state.conductivity *= mult;
        }
    } else {
        eos_state.conductivity = 0.0_rt;
    }

    kappa = castro::diffuse_cond_scale_fac * eos_state.conductivity;
    if (!amrex::Math::isfinite(kappa) || kappa < 0.0_rt) { kappa = 0.0_rt; }

    // Free-streaming limit: q_fs = f * n_e * k_B * Te * sqrt(k_B Te / m_e)
    const Real v_th = std::sqrt(C::k_B * Te / C::m_e);          // cm/s
    q_fs = flux_limiter_f * n_e * C::k_B * Te * v_th;           // erg/cm^2/s
    if (!amrex::Math::isfinite(q_fs) || q_fs < 0.0_rt) { q_fs = 0.0_rt; }
}

// ---------------------------------------------------------------------
// Face conductivity averaging.
//
// For nonlinear diffusion  dT/dt = div(kappa(T) grad T)  the exact flux
// across a face follows from the Kirchhoff integral Phi = int kappa dT:
//
//     kappa_face = (Phi(T_r) - Phi(T_l)) / (T_r - T_l)
//
// For kappa = k0 T^n that is k0 (T_r^{n+1} - T_l^{n+1}) / ((n+1)(T_r - T_l)),
// which is dominated by the HOT side.
//
// WHY NOT HARMONIC: harmonic averaging is the right choice for a
// DISCONTINUOUS material property -- two media in contact, where flux
// continuity across the interface fixes the face value. It is the wrong
// choice here, because kappa is not a material property but a smooth
// function of the solution itself. Harmonic collapses toward the COLD
// value while the true flux integral is dominated by the HOT one. For
// n = 5/2 it underestimates the face flux by ~2e-2 at a 10x temperature
// contrast and ~2e-7 at 1000x, which appears as a heat front that
// barely advances while energy piles up behind it.
//
//   FACE_ARITHMETIC  0.5*(k_l + k_r). Within ~1.75x of exact at every
//                    contrast for n = 5/2. The original scheme.
//   FACE_KIRCHHOFF   exact for a local power law, with the exponent
//                    recovered from the two cells as
//                    n = ln(k_r/k_l) / ln(T_r/T_l), so it adapts to
//                    whatever the conductivity model actually does
//                    instead of assuming T^(5/2).
enum FaceAvg { FACE_ARITHMETIC = 0, FACE_KIRCHHOFF = 1 };

// Default: arithmetic, matching the validated reference. Switch to
// FACE_KIRCHHOFF for the more accurate flux -- expect a slightly slower
// front, since arithmetic runs ~1.75x high at large contrast.
constexpr FaceAvg face_avg_mode = FACE_ARITHMETIC;

// Face kappa is floored at this fraction of max(k_a, k_b), so a cell
// whose kappa was zeroed by a cutoff (low n_e, or rho below
// diffuse_cutoff_density) cannot completely block conduction into
// itself, never warm, and lock the front in place. Set to 0 to disable.
constexpr Real kappa_floor_rel = 1.0e-3_rt;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real face_kappa (Real k_a, Real k_b, Real T_a = 0.0_rt, Real T_b = 0.0_rt)
{
    if (!amrex::Math::isfinite(k_a) || k_a < 0.0_rt) { k_a = 0.0_rt; }
    if (!amrex::Math::isfinite(k_b) || k_b < 0.0_rt) { k_b = 0.0_rt; }

    const Real k_max = amrex::max(k_a, k_b);
    if (!(k_max > 0.0_rt)) { return 0.0_rt; }   // both genuinely zero

    const Real arith = 0.5_rt * (k_a + k_b);
    Real kf = arith;

    if (face_avg_mode == FACE_KIRCHHOFF &&
        k_a > 0.0_rt && k_b > 0.0_rt &&
        T_a > 0.0_rt && T_b > 0.0_rt)
    {
        const Real lnTr = std::log(T_b / T_a);
        const Real dT   = T_b - T_a;
        if (std::abs(lnTr) > 1.0e-8_rt && std::abs(dT) > 1.0e-30_rt) {
            const Real n = std::log(k_b / k_a) / lnTr;
            if (amrex::Math::isfinite(n) && std::abs(n + 1.0_rt) > 1.0e-8_rt) {
                // k0 T^(n+1) = k*T, so Phi(T_r)-Phi(T_l) = (k_b T_b - k_a T_a)/(n+1)
                const Real cand = (k_b * T_b - k_a * T_a) / ((n + 1.0_rt) * dT);
                if (amrex::Math::isfinite(cand) && cand > 0.0_rt) { kf = cand; }
            }
        }
    }

    return amrex::max(kf, kappa_floor_rel * k_max);
}

// ---------------------------------------------------------------------
// Harmonic flux limiter applied at a face:
//     kappa_eff = kappa / (1 + kappa|dT/dx| / q_fs)
// so kappa_eff -> kappa in shallow gradients and the flux saturates at
// q_fs in steep ones.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real limit_kappa (Real kappa_face, Real dT, Real dx_len, Real qfs_face)
{
    if (!(kappa_face > 0.0_rt)) { return 0.0_rt; }

    if (!use_flux_limiter) { return kappa_face; }   // unlimited Spitzer-Harm

    if (!(qfs_face > 0.0_rt)) { return 0.0_rt; }

    const Real q_sh = kappa_face * std::abs(dT) / dx_len;
    return kappa_face / (1.0_rt + q_sh / qfs_face);
}

// ---------------------------------------------------------------------
// Electron temperature only.
//
// For callers that need Te but not the conductivity -- the ionisation
// and collisional-exchange rates in problem_source, for instance. It
// still runs the full evaluation, since Te comes out of the same EOS
// inversion, but it makes the intent explicit at the call site instead
// of passing a discarded kappa.
//
// Inside the solver, prefer the cached coefficients: compute_coeffs()
// evaluates this once per cell into a MultiFab and the stencil reads
// from it, rather than re-inverting the EOS per stencil point.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real get_Te (int i, int j, int k, Array4<const Real> const& state)
{
    Real Te, kappa, q_fs;
    get_Te_kappa_qfs(i, j, k, state, Te, kappa, q_fs);
    return Te;
}

// Backward-compatible two-output form.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void get_Te_and_kappa (int i, int j, int k,
                       Array4<const Real> const& state,
                       Real& Te, Real& kappa)
{
    Real q_fs;
    get_Te_kappa_qfs(i, j, k, state, Te, kappa, q_fs);
}

// ---------------------------------------------------------------------
// Cached coefficients.
//
// The stencil needs Te, kappa and q_fs at the centre AND at every
// neighbour. Evaluating them inside the stencil means each cell's EOS
// inversion is repeated 2*DIM+1 times per residual -- 5x in 2D, 7x in
// 3D -- and every repeat is a Newton root-find plus the
// collision-frequency model.
//
// Caching them into a MultiFab is pure memoization: the arithmetic is
// identical, so the answer is bit-for-bit unchanged. It also lifts
// eos_t (which carries xn[NumSpec]) out of the stencil kernel, which is
// the main source of register pressure and local-memory spilling on
// GPU.
// ---------------------------------------------------------------------
constexpr int NCOEF = 3;
constexpr int CTE   = 0;   // electron temperature
constexpr int CKAP  = 1;   // conductivity
constexpr int CQFS  = 2;   // free-streaming flux magnitude

// Fills coef over valid + ghost cells. `state` must have >= 1 filled
// ghost cell, and `coef` must have >= 1 ghost so the stencil can read
// neighbours of valid cells.
inline void
compute_coeffs (const MultiFab& state, MultiFab& coef)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(coef, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        Array4<Real const> const s = state.const_array(mfi);
        Array4<Real>       const c = coef.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real Te, kap, qfs;
            get_Te_kappa_qfs(i, j, k, s, Te, kap, qfs);
            c(i,j,k,CTE)  = Te;
            c(i,j,k,CKAP) = kap;
            c(i,j,k,CQFS) = qfs;
        });
    }
}

// ---------------------------------------------------------------------
// Geometric cell factor for direction idir.
//
// Returns the multiplier applied to the flux difference for this cell,
// i.e. 1/dx^2 combined with the curvilinear metric that belongs on the
// CELL (the face part of the metric is baked into the face coefficient
// by compute_face_coeffs). Shared by the operator and the
// preconditioner so the two cannot drift apart.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real cell_metric (int i, int j, int k, int idir,
                  const Real* problo, const Real* dx, int coord)
{
    amrex::ignore_unused(k);
    Real fac = 1.0_rt / (dx[idir] * dx[idir]);

#if AMREX_SPACEDIM < 3
    if ((coord != 0) && (idir == 0)) {
        const Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
        fac /= (coord == 1) ? rc : (rc * rc);
    } else if ((coord == 2) && (idir == 1)) {
        const Real rc     = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
        const Real thetac = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1];
        fac /= (rc * rc * std::sin(thetac));
    }
#else
    amrex::ignore_unused(i, j, problo, coord);
#endif
    return fac;
}

// ---------------------------------------------------------------------
// Face-centred conductivities.
//
// Each interior face is evaluated exactly ONCE. Computing them inside a
// cell loop evaluates every interior face twice -- as the right face of
// one cell and the left face of its neighbour -- which doubles the
// face_kappa cost, and in FACE_KIRCHHOFF that means twice as many
// std::log calls. Evaluating on the face grid also makes the two sides
// bit-identical, so the flux telescoping is exact rather than exact up
// to rounding.
//
// The face part of the curvilinear metric (r, r^2, sin(theta)) is baked
// in here; the cell part is applied by cell_metric() in the stencil.
// The free-streaming limiter is applied here too, since it depends on
// the temperature jump across this same face.
//
// fk[idir] is face-centred in direction idir with the usual convention
// that face index i separates cells i-1 and i.
// ---------------------------------------------------------------------
inline void
compute_face_coeffs (const MultiFab& coef,
                     Array<MultiFab,AMREX_SPACEDIM>& fk,
                     const Geometry& geom)
{
    const amrex::GpuArray<Real,AMREX_SPACEDIM> problo = geom.ProbLoArray();
    const amrex::GpuArray<Real,AMREX_SPACEDIM> dx     = geom.CellSizeArray();
    const int coord = geom.Coord();

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(fk[idir], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            Array4<Real const> const c = coef.const_array(mfi);
            Array4<Real>       const f = fk[idir].array(mfi);
            const Real h = dx[idir];

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Face (i,j,k) in direction idir separates the "minus"
                // cell and the "plus" cell.
                const int im = i - (idir == 0);
                const int jm = j - (idir == 1);
                const int km = k - (idir == 2);

                const Real Te_m  = c(im,jm,km,CTE);
                const Real kap_m = c(im,jm,km,CKAP);
                const Real qfs_m = c(im,jm,km,CQFS);
                const Real Te_p  = c(i ,j ,k ,CTE);
                const Real kap_p = c(i ,j ,k ,CKAP);
                const Real qfs_p = c(i ,j ,k ,CQFS);

                Real kf = face_kappa(kap_m, kap_p, Te_m, Te_p);
                kf = limit_kappa(kf, Te_p - Te_m, h, face_kappa(qfs_m, qfs_p));

#if AMREX_SPACEDIM < 3
                // Face part of the curvilinear metric.
                if ((coord != 0) && (idir == 0)) {
                    const Real rf = problo[0] + static_cast<Real>(i) * dx[0];
                    kf *= (coord == 1) ? rf : (rf * rf);
                } else if ((coord == 2) && (idir == 1)) {
                    const Real thetaf = problo[1] + static_cast<Real>(j) * dx[1];
                    kf *= std::sin(thetaf);
                }
#endif
                f(i,j,k) = kf;
            });
        }
    }
}

// ---------------------------------------------------------------------
// L = div( kappa_eff grad Te ) at (i,j,k), from PRECOMPUTED face
// coefficients.
//
// The face coefficients already carry the flux limiter and the face part
// of the curvilinear metric, so this is a pure 2*DIM+1 point
// read-and-multiply-add with no equation-of-state work, no logs, and no
// face averaging. Each interior face was evaluated once, on the face
// grid, rather than twice from the two adjoining cells.
//
// Because the two cells sharing a face read the SAME stored value, the
// flux difference telescopes exactly and the operator conserves energy
// to machine precision.
// ---------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real div_kappa_grad_Te (int i, int j, int k,
                        Array4<const Real> const& c,
                        AMREX_D_DECL(Array4<const Real> const& fx,
                                     Array4<const Real> const& fy,
                                     Array4<const Real> const& fz),
                        const Real* problo, const Real* dx, const int coord)
{
    const Real Te_c = c(i,j,k,CTE);

    Real L = 0.0_rt;

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

        int il = i, jl = j, kl = k;
        int ir = i, jr = j, kr = k;
        if (idir == 0) { il = i-1; ir = i+1; }
#if AMREX_SPACEDIM >= 2
        else if (idir == 1) { jl = j-1; jr = j+1; }
#endif
#if AMREX_SPACEDIM == 3
        else { kl = k-1; kr = k+1; }
#endif

        // Low face of this cell sits at the cell index; the high face is
        // one further along the face-centred direction.
        Real kth_l = 0.0_rt, kth_r = 0.0_rt;
        if (idir == 0) {
            kth_l = fx(i  ,j,k);
            kth_r = fx(i+1,j,k);
        }
#if AMREX_SPACEDIM >= 2
        if (idir == 1) {
            kth_l = fy(i,j  ,k);
            kth_r = fy(i,j+1,k);
        }
#endif
#if AMREX_SPACEDIM == 3
        if (idir == 2) {
            kth_l = fz(i,j,k  );
            kth_r = fz(i,j,k+1);
        }
#endif

        const Real fac = cell_metric(i, j, k, idir, problo, dx, coord);

        L += fac * ( kth_r * (c(ir,jr,kr,CTE) - Te_c)
                   - kth_l * (Te_c - c(il,jl,kl,CTE)) );
    }

    return amrex::Math::isfinite(L) ? L : 0.0_rt;
}

// ---------------------------------------------------------------------
// Whole-MultiFab evaluation of L. Ghosts must already be filled.
// ---------------------------------------------------------------------
// Workspace reused across the hundreds of residual evaluations in a
// solve: cell-centred coefficients plus the face-centred conductivities.
struct Workspace {
    MultiFab                        coef;   // CTE, CKAP, CQFS; >= 1 ghost
    Array<MultiFab,AMREX_SPACEDIM>  fk;     // face conductivities

    void define (const BoxArray& ba, const DistributionMapping& dm)
    {
        if (coef.ok()) { return; }
        coef.define(ba, dm, NCOEF, 1);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            fk[d].define(amrex::convert(ba, IntVect::TheDimensionVector(d)),
                         dm, 1, 0);
        }
    }
};

inline void
compute_L (const MultiFab& state, MultiFab& L_mf, const Geometry& geom,
           Workspace* ws = nullptr)
{
    Workspace local;
    Workspace* w = ws;
    if (w == nullptr) {
        local.define(state.boxArray(), state.DistributionMap());
        w = &local;
    }

    compute_coeffs(state, w->coef);
    compute_face_coeffs(w->coef, w->fk, geom);

    // CRITICAL: geom.ProbLo() and geom.CellSize() return pointers into
    // the HOST-side Geometry object. Capturing them with [=] copies the
    // POINTER, not the data, so a device lambda dereferences a host
    // address -> CUDA error 700 on the first launch. The *Array()
    // accessors return a GpuArray BY VALUE, so the data is copied into
    // the lambda closure.
    const amrex::GpuArray<Real,AMREX_SPACEDIM> problo = geom.ProbLoArray();
    const amrex::GpuArray<Real,AMREX_SPACEDIM> dx     = geom.CellSizeArray();
    const int coord = geom.Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(L_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const cc = w->coef.const_array(mfi);
        AMREX_D_TERM(Array4<Real const> const fx = w->fk[0].const_array(mfi);,
                     Array4<Real const> const fy = w->fk[1].const_array(mfi);,
                     Array4<Real const> const fz = w->fk[2].const_array(mfi);)
        Array4<Real> const L = L_mf.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            L(i,j,k) = div_kappa_grad_Te(i, j, k, cc,
                                         AMREX_D_DECL(fx, fy, fz),
                                         problo.data(), dx.data(), coord);
        });
    }
}

} // namespace sh_diff

#endif