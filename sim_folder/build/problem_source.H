#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;

    const Real* problo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();

    AMREX_D_DECL(
    Real x = (Real(i)+0.5e0_rt)*dx[0] + problo[0],
    Real y = (Real(j)+0.5e0_rt)*dx[1] + problo[1],
    Real z = (Real(k)+0.5e0_rt)*dx[2] + problo[2]
    );

    Real A_direct_ionization = 7.1e-9; // m^3.s-1
    Real q_direct_ionization = 0.4;
    Real E_direct_ionization = 13.6 * C::ev2erg; // ergs

    // Density and temperature
    Real rho = state(i,j,k,URHO);
    Real T = state(i,j,k,UTEMP);
    // Abundance of neutral and ionized H
    Real X_Hn = state(i,j,k,UFS) / rho;
    Real X_Hp = state(i,j,k,UFS+1) / rho;

    // Rate of ionization
    Real R_Hp = rho * rho / C::m_p * (
        A_direct_ionization * ( C::k_B*T/C::ev2erg )**q_direct_ionization *
        std::exp(-E_direct_ionization/(C::k_B*T)) *
        X_Hn * X_Hp
    );

    // Update state
    src(i,j,k,UFS) = -R_Hp;
    src(i,j,k,UFS+1) = R_Hp;
    src(i,j,k,UEDEN) = 0;
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);
}

#endif
