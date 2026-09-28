#ifndef THD_DIFFUSION_OPERATOR_H
#define THD_DIFFUSION_OPERATOR_H

// Linear/Nonlinear un-limited electron thermal diffusion operator for two-temperature
// (gamma_law_2T) Castro setups:
//   L = div( kappa * grad(Te) )

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

namespace thd_diff {

// Retained for reference
constexpr Real min_ne_cgs = 1.0e-30_rt;

// Coefficient indices for cached MultiFab
constexpr int NCOEF = 2;
constexpr int CTE   = 0;   // Electron temperature [K]
constexpr int CKAP  = 1;   // Thermal conductivity [erg / (cm s K)]

enum FaceAvg { FACE_ARITHMETIC = 0, FACE_KIRCHHOFF = 1 };
constexpr FaceAvg face_avg_mode = FACE_ARITHMETIC;

// Relative floor for face conductivity to prevent frozen conduction fronts
constexpr Real kappa_floor_rel = 1.0e-3_rt;


// Computes cell-centered Te and conductivity (kappa).
// Handles non-physical states gracefully to prevent NaNs.

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void get_Te_kappa (int i, int j, int k,
                   Array4<const Real> const& state,
                   Real& Te, Real& kappa)
{
    Te    = castro::small_temp;
    kappa = 0.0_rt;

    const Real rho   = state(i,j,k,URHO);
    const Real reint = state(i,j,k,UEINT);

    if (!(rho > 0.0_rt) || !amrex::Math::isfinite(rho) ||
        !amrex::Math::isfinite(reint)) {
        return;
    }

    const Real rhoinv = 1.0_rt / rho;

    eos_t eos_state;
    eos_state.rho = rho;
    eos_state.T   = amrex::max(state(i,j,k,UTEMP), castro::small_temp);
    eos_state.e   = reint * rhoinv;

    // Clamp mass fractions to [0,1] to prevent negative electron density
    Real n_e = 0.0_rt;
    for (int n = 0; n < NumSpec; ++n) {
        const Real X = amrex::min(amrex::max(state(i,j,k,UFS+n) * rhoinv, 0.0_rt), 1.0_rt);
        eos_state.xn[n] = X;
        n_e += zion[n] * X * rho * aion_inv[n] / C::m_u;
    }

#if NAUX_NET > 0
    // aux[0] = f = E_heavies / E_total. Strictly bound inside (0,1) to avoid degenerate Te
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

    if (!amrex::Math::isfinite(eos_state.T) || eos_state.T <= 0.0_rt) {
        Te = castro::small_temp;
        return;
    }
    Te = amrex::max(eos_state.T, castro::small_temp);
    eos_state.T = Te;

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
}


// Face conductivity averaging

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real face_kappa (Real k_a, Real k_b, Real T_a = 0.0_rt, Real T_b = 0.0_rt)
{
    if (!amrex::Math::isfinite(k_a) || k_a < 0.0_rt) { k_a = 0.0_rt; }
    if (!amrex::Math::isfinite(k_b) || k_b < 0.0_rt) { k_b = 0.0_rt; }

    const Real k_max = amrex::max(k_a, k_b);
    if (!(k_max > 0.0_rt)) { return 0.0_rt; }

    Real kf = 0.5_rt * (k_a + k_b);

    if (face_avg_mode == FACE_KIRCHHOFF &&
        k_a > 0.0_rt && k_b > 0.0_rt &&
        T_a > 0.0_rt && T_b > 0.0_rt)
    {
        const Real lnTr = std::log(T_b / T_a);
        const Real dT   = T_b - T_a;
        if (std::abs(lnTr) > 1.0e-8_rt && std::abs(dT) > 1.0e-30_rt) {
            const Real n = std::log(k_b / k_a) / lnTr;
            if (amrex::Math::isfinite(n) && std::abs(n + 1.0_rt) > 1.0e-8_rt) {
                const Real cand = (k_b * T_b - k_a * T_a) / ((n + 1.0_rt) * dT);
                if (amrex::Math::isfinite(cand) && cand > 0.0_rt) { kf = cand; }
            }
        }
    }

    return amrex::max(kf, kappa_floor_rel * k_max);
}

// Helper evaluations for Te
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real get_Te (int i, int j, int k, Array4<const Real> const& state)
{
    Real Te, kappa;
    get_Te_kappa(i, j, k, state, Te, kappa);
    return Te;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void get_Te_and_kappa (int i, int j, int k,
                       Array4<const Real> const& state,
                       Real& Te, Real& kappa)
{
    get_Te_kappa(i, j, k, state, Te, kappa);
}

// Fills cell-centered coefficient cache over valid + ghost cells
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
            Real Te, kap;
            get_Te_kappa(i, j, k, s, Te, kap);
            c(i,j,k,CTE)  = Te;
            c(i,j,k,CKAP) = kap;
        });
    }
}


// Returns the geometric factor (1/dx^2 combined with cell metric)

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

// Precomputes face-centered conductivities and metrics
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

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const int im = i - (idir == 0);
                const int jm = j - (idir == 1);
                const int km = k - (idir == 2);

                const Real Te_m  = c(im,jm,km,CTE);
                const Real kap_m = c(im,jm,km,CKAP);
                const Real Te_p  = c(i ,j ,k ,CTE);
                const Real kap_p = c(i ,j ,k ,CKAP);

                Real kf = face_kappa(kap_m, kap_p, Te_m, Te_p);

#if AMREX_SPACEDIM < 3
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


// Computes L = div( kappa * grad(Te) ) at (i,j,k)

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

// Reusable workspace container for solver iterations
struct Workspace {
    MultiFab                        coef;
    Array<MultiFab,AMREX_SPACEDIM>  fk;

    void define (const BoxArray& ba, const DistributionMapping& dm)
    {
        if (coef.ok()) { return; }
        coef.define(ba, dm, NCOEF, 1);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            fk[d].define(amrex::convert(ba, IntVect::TheDimensionVector(d)), dm, 1, 0);
        }
    }
};

// Evaluates L across the full MultiFab
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

} // namespace thd_diff

#endif