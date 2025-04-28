#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k, Array4<Real> const& state, const GeometryData& geomdata)
{
    int coord_type = geomdata.Coord();

    const Real* dx = geomdata.CellSize();
    const Real* problo = geomdata.ProbLo();

    Real x = 0.0;
    Real y = 0.0;
    Real z = 0.0;

    x = problo[0] + dx[0] * (static_cast<Real>(i)+0.5);
#if AMREX_SPACEDIM >= 2
    y = problo[1] + dx[1] * (static_cast<Real>(j)+0.5);
#endif
#if AMREX_SPACEDIM == 3
    z = problo[2] + dx[2] * (static_cast<Real>(k)+0.5);
#endif


    Real sigma1 = 38e-4; // in cm
    Real ioniz_fraction = (1. - 1.e-3)*std::exp( -std::pow((x*x + y*y + z*z)/(2*sigma1*sigma1), 12) ) + 1.e-3;
    Real sigma2 = 35e-4; // in cm
    Real Te_max = 313322.; // 27 eV, in Kelvins
    Real Ta = 348.; // 0.03 eV, in Kelvins
    Real Te = (Te_max - Ta)*std::exp( -std::pow((x*x + y*y + z*z)/(2*sigma2*sigma2), 3) ) + Ta;

    Real eint = problem::dens_ambient * (ioniz_fraction * 3./2 * C::k_B * Te +  3./2 * C::k_B * Ta)/C::m_u;

    state(i,j,k,URHO) = problem::dens_ambient;
    state(i,j,k,UMX) = 0.e0_rt;
    state(i,j,k,UMY) = 0.e0_rt;
    state(i,j,k,UMZ) = 0.e0_rt;

    state(i,j,k,UEDEN) = eint +
                         0.5e0_rt * (state(i,j,k,UMX) * state(i,j,k,UMX) / state(i,j,k,URHO) +
                                     state(i,j,k,UMY) * state(i,j,k,UMY) / state(i,j,k,URHO) +
                                     state(i,j,k,UMZ) * state(i,j,k,UMZ) / state(i,j,k,URHO));

    state(i,j,k,UEINT) = eint;
    state(i,j,k,UFX) = eint; // internal energy of heavy ions

    // The temperature initialization will be done later
    // in the call to clean_state. We want to avoid
    // EOS calls in ca_initdata. So that there is valid
    // data for an initial guess for the EOS, we initialize
    // to the ambient temperature everywhere.

    state(i,j,k,UTEMP) = problem::temp_ambient;

    state(i,j,k,UFS) = state(i,j,k,URHO)*(1-ioniz_fraction);
    state(i,j,k,UFS+1) = state(i,j,k,URHO)*ioniz_fraction;
}

#endif
