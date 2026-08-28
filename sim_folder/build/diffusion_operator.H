#ifndef diffusion_operator_H
#define diffusion_operator_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <conductivity.H>
#include <collisional_ionization.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

using namespace amrex;

// -----------------------------------------------------------------------
// Returns the thermal conductivity coefficient k_th at point (i,j,k)
// Identical to your working version: reproduces fill_temp_cond from
// diffusion_util.cpp, calling the Microphysics conductivity() closure
// (your Spitzer-Harm-like actual_conductivity.H).
// -----------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real get_conductivity_coeff( int i, int j, int k, Array4<const Real> const& state )
{
    eos_t eos_state;
    eos_state.rho  = state(i,j,k,URHO);
    Real rhoinv = 1.0_rt/eos_state.rho;

    eos_state.T = state(i,j,k,UTEMP);   // initial guess
    eos_state.e = state(i,j,k,UEINT) * rhoinv;
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = state(i,j,k,UFS+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = state(i,j,k,UFX+n) * rhoinv;
    }
#endif

    if (eos_state.e < 0.0_rt) {
      eos_state.T = castro::small_temp;
      eos(eos_input_rt, eos_state);
    } else {
      eos(eos_input_re, eos_state);
    }

    if (eos_state.rho > castro::diffuse_cutoff_density) {
      conductivity(eos_state);

      if (eos_state.rho < castro::diffuse_cutoff_density_hi) {
        Real multiplier = (eos_state.rho - castro::diffuse_cutoff_density) /
          (castro::diffuse_cutoff_density_hi - castro::diffuse_cutoff_density);
        eos_state.conductivity = eos_state.conductivity * multiplier;
      }
    } else {
      eos_state.conductivity = 0.0_rt;
    }
    return castro::diffuse_cond_scale_fac * eos_state.conductivity;
}

// -----------------------------------------------------------------------
// Returns L(T) = div(k_th grad T) at point (i,j,k).
// Identical stencil/geometry handling to your working version.
// Requires state to have >=1 valid ghost cell (filled by caller).
// -----------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real compute_temp_diffusion_term(
    int i, int j, int k, Array4<const Real> const& state, const Real* problo, const Real* dx, const int coord )
{
    amrex::Real temp_diffusion_term = 0.0;
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        int il = i, jl = j, kl = k;
        int ir = i, jr = j, kr = k;

        Real dxinv2 = 1.0/(dx[idir]*dx[idir]);
        Real kth_r, kth_l;

        if (idir == 0) {
            il = i - 1; ir = i + 1;
        }
#if AMREX_SPACEDIM >= 2
        else if (idir == 1) {
            jl = j - 1; jr = j + 1;
        }
#endif
#if AMREX_SPACEDIM == 3
        else {
            kl = k - 1; kr = k + 1;
        }
#endif
        kth_r = 0.5*(get_conductivity_coeff(ir,jr,kr,state) + get_conductivity_coeff(i,j,k,state));
        kth_l = 0.5*(get_conductivity_coeff(il,jl,kl,state) + get_conductivity_coeff(i,j,k,state));

#if AMREX_SPACEDIM < 3
        if ((coord != 0) && (idir == 0)) {
            Real rr = problo[idir] + static_cast<Real>(ir) * dx[idir];
            Real rl = problo[idir] + static_cast<Real>(i) * dx[idir];
            Real rc = problo[idir] + (static_cast<Real>(i) + 0.5_rt) * dx[idir];

            if (coord == 1) {
                kth_r *= rr;
                kth_l *= rl;
                dxinv2 *= 1.0_rt / rc;
            } else {
                kth_r *= rr * rr;
                kth_l *= rl * rl;
                dxinv2 *= 1.0_rt / (rc * rc);
            }
        } else if ((coord == 2) && (idir == 1)) {
            Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
            Real thetar = problo[idir] + static_cast<Real>(jr) * dx[idir];
            Real thetal = problo[idir] + static_cast<Real>(j) * dx[idir];
            Real thetac = problo[idir] + (static_cast<Real>(j) + 0.5_rt) * dx[idir];

            kth_r *= std::sin(thetar);
            kth_l *= std::sin(thetal);
            dxinv2 *= 1.0_rt / (rc * rc * std::sin(thetac));
        }
#endif
        temp_diffusion_term += dxinv2 *
            (kth_r * (state(ir,jr,kr,UTEMP) - state(i ,j ,k ,UTEMP)) -
             kth_l * (state(i ,j ,k ,UTEMP) - state(il,jl,kl,UTEMP)));
    }

    return temp_diffusion_term;
}

// -----------------------------------------------------------------------
// Whole-MultiFab wrapper: fills L_mf(i,j,k) = div(k_th grad T) everywhere
// in bx, reading UTEMP (and EOS-relevant components) from state_mf.
// state_mf MUST have >=1 valid ghost cell filled (FillBoundary/FillPatch)
// before calling this.
// -----------------------------------------------------------------------
inline void
compute_diffusion_term_mf( const amrex::MultiFab& state_mf,
                            amrex::MultiFab& L_mf,
                            const amrex::Geometry& geom )
{
    const Real* problo = geom.ProbLo();
    const Real* dx = geom.CellSize();
    const int coord = geom.Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(L_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Array4<Real const> const state = state_mf.const_array(mfi);
        Array4<Real> const L = L_mf.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            L(i,j,k) = compute_temp_diffusion_term(i, j, k, state, problo, dx, coord);
        });
    }
}

#endif