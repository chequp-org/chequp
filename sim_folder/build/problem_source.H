#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real conductivity_coeffs( int i, int j, int k, Array4<const Real> const& state )
{
    // TODO: Reproduces the same code as `fill_temp_cond` in `diffusion_util.cpp`

    return 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
// Return the term ∇ ⋅(k_th ∇T) at point (i,j,k)
amrex::Real compute_temp_diffusion_term(
    int i, int j, int k, Array4<const Real> const& state, const Real* problo, const Real* dx, const int coord )
{
    // Reproduces the same code as `getTempDiffusionTerm` in `Castro_diffusion.cpp`

    amrex::Real temp_diffusion_term = 0.0;
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        int il = i;
        int jl = j;
        int kl = k;

        int ir = i;
        int jr = j;
        int kr = k;

        Real dxinv2 = 1.0/(dx[idir]*dx[idir]);
        Real kth_r;
        Real kth_l;

        if (idir == 0) {
            il = i - 1;
            ir = i + 1;
        }
#if AMREX_SPACEDIM >= 2
        else if (idir == 1) {
            jl = j - 1;
            jr = j + 1;
        }
#endif
#if AMREX_SPACEDIM == 3
        else {
            kl = k - 1;
            kr = k + 1;
        }
#endif
        // Conductivity at the cell edges: average of the conductivity from cell centers to the edges
        // TODO: This could be made more efficient by avoiding to recalculate
        // conductivity_coeffs(i,j,k,state) multiple times
        kth_r = 0.5*(conductivity_coeffs(ir,jr,kr,state) + conductivity_coeffs(i,j,k,state));
        kth_l = 0.5*(conductivity_coeffs(il,jl,kl,state) + conductivity_coeffs(i,j,k,state));

#if AMREX_SPACEDIM < 3
        // Apply geometric terms for curvilinear coordinates

        if ((coord != 0) && (idir == 0)) {
            // In curilinear radial direction

            Real rr = problo[idir] + static_cast<Real>(ir) * dx[idir];
            Real rl = problo[idir] + static_cast<Real>(i) * dx[idir];
            Real rc = problo[idir] + (static_cast<Real>(i) + 0.5_rt) * dx[idir];

            if (coord == 1) {
                // Cylindrical radial equation looks like: 1/r d(r kth dT/dr)/dr

                kth_r *= rr;
                kth_l *= rl;
                dxinv2 *= 1.0_rt / rc;
            } else {
                // Spherical radial equation looks like: 1/r^2 d(r^2 kth dT/dr)/dr

                kth_r *= rr * rr;
                kth_l *= rl * rl;
                dxinv2 *= 1.0_rt / (rc * rc) ;
            }
        } else if ((coord == 2) && (idir == 1)) {
            // In spherical theta direction
            // Spherical theta equation looks like: 1/r^2 sin(θ) d(sin(θ) kth dT/dθ)/dθ

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

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;

    auto coord = geomdata.Coord();
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();

    // Density and temperature
    Real rho = state(i,j,k,URHO);

    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state(i,j,k,UTEMP);

    // Abundance of neutral and ionized H
    Real X_H0 = state(i,j,k,UFS) / rho;
    Real X_H1 = state(i,j,k,UFS+1) / rho;

    // Electron and heavy temperatures
    Real f = state(i,j,k,UFX)/rho;  // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;  // internal energy per unit mass

    // Rate of ionization
    Real R_H1 = rho * rho / C::m_p * (
        problem::a_direct_ioniz *
        std::pow( C::k_B*T_e/C::ev2erg, problem::q_direct_ioniz ) *
        std::exp(-problem::e_direct_ioniz_eV*C::ev2erg/(C::k_B*T_e)) * X_H0 * X_H1
        +
        problem::a_twostep_ioniz *
        std::pow( C::k_B*T_e/C::ev2erg, problem::q_twostep_ioniz ) *
        std::exp(-problem::e_twostep_ioniz_eV*C::ev2erg/(C::k_B*T_e)) * X_H0 * X_H1
    );

    // Update state
    src(i,j,k,UFS) = -R_H1;
    src(i,j,k,UFS+1) = R_H1;
    src(i,j,k,UEDEN) = -R_H1 * (problem::e_loss_ioniz_eV*C::ev2erg / C::m_p);
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // In order to calculate the energy transfer between electrons and heavy particles,
    // compute the collision frequency between electrons and ions (using SI units)
    const amrex::Real n_e = rho * X_H1 / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    const amrex::Real n_n = rho * X_H0 / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    // Get collision frequencies
    amrex::Real v_ei, v_en;
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)
    get_collision_frequencies(n_e, n_n, T_e, T_h, v_ei, v_en);

    // Calculate the collision frequency for energy transfer between electrons and heavies
    const amrex::Real v_e_E = 2*SI::m_e/SI::m_H * (v_ei + v_en);

    // Add contribution for temperature diffusion
    if (diffuse_temp) {
        const amrex::Real temp_diffusion_term = compute_temp_diffusion_term(i,j,k,state,prob_lo,dx,coord);
        src(i,j,k,UEDEN) += temp_diffusion_term;
        src(i,j,k,UEINT) += temp_diffusion_term;
    }

    // Update fraction of internal energy stored in ions
    src(i,j,k,UFX) = - f/e * src(i,j,k,UEDEN) + rho * v_e_E * ( 1 - (1+X_H1)*f );
}

#endif
