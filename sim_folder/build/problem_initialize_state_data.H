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

#if AMREX_SPACEDIM == 1
    int nsubx = problem::nsub;
    int nsuby = 1;
    int nsubz = 1;
#elif AMREX_SPACEDIM == 2
    int nsubx = problem::nsub;
    int nsuby = problem::nsub;
    int nsubz = 1;
#else
    int nsubx = problem::nsub;
    int nsuby = problem::nsub;
    int nsubz = problem::nsub;
#endif

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

    Real ioniz_fraction = std::exp( -x*x/(problem::r_init*problem::r_init) );
    Real eint = problem::dens_ambient * ( ioniz_fraction * problem::e_exp + (1-ioniz_fraction) * problem::e_ambient );

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
