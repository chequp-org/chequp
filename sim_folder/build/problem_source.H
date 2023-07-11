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

    // Density and temperature
    Real rho = state(i,j,k,URHO);
    Real T = state(i,j,k,UTEMP);
    // Abundance of neutral and ionized H
    Real X_Hn = state(i,j,k,UFS) / rho;
    Real X_Hp = state(i,j,k,UFS+1) / rho;

    // Rate of ionization
    Real R_Hp = rho * rho / C::m_p * (
        problem::a_direct_ioniz *
        std::pow( C::k_B*T/C::ev2erg, problem::q_direct_ioniz ) *
        std::exp(-problem::e_direct_ioniz_eV*C::ev2erg/(C::k_B*T)) *
        X_Hn * X_Hp
        +
        problem::a_twostep_ioniz *
        std::pow( C::k_B*T/C::ev2erg, problem::q_twostep_ioniz ) *
        std::exp(-problem::e_twostep_ioniz_eV*C::ev2erg/(C::k_B*T)) *
        X_Hn * X_Hp
    );

    // Update state
    src(i,j,k,UFS) = -R_Hp;
    src(i,j,k,UFS+1) = R_Hp;
    src(i,j,k,UEDEN) = -R_Hp * (problem::e_loss_ioniz_eV*C::ev2erg / C::m_p);
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // Update internal energy of heavy species
    // compute -p_h div{U}
    // (See Castro/Source/sources/Castro_thermo.cpp)
    Real divU = 0;
    // radius for non-Cartesian
    Real rp = prob_lo[0] + (static_cast<Real>(i) + 1.5_rt)*dx[0];
    Real rm = prob_lo[0] + (static_cast<Real>(i) - 0.5_rt)*dx[0];
    Real r = 0.5_rt*(rm + rp);
    // compute div{U}
    if (coord == 0) {
        divU = 0.5_rt*(U(i+1,j,k,UMX)/U(i+1,j,k,URHO)  -
                                U(i-1,j,k,UMX)/U(i-1,j,k,URHO))/dx[0];

    } else if (coord == 1) {
        // axisymmetric
        divU = 0.5_rt*(rp*U(i+1,j,k,UMX)/U(i+1,j,k,URHO) -
                                rm*U(i-1,j,k,UMX)/U(i-1,j,k,URHO))/(r*dx[0]);

    } else if (coord == 2) {
        // spherical
        divU = 0.5_rt*(rp*rp*U(i+1,j,k,UMX)/U(i+1,j,k,URHO) -
                                rm*rm*U(i-1,j,k,UMX)/U(i-1,j,k,URHO))/(r*r*dx[0]);
    }
#if AMREX_SPACEDIM >= 2
    divU += 0.5_rt*(U(i,j+1,k,UMY)/U(i,j+1,k,URHO) -
                                U(i,j-1,k,UMY)/U(i,j-1,k,URHO))/dx[1];
#endif
#if AMREX_SPACEDIM == 3
    divU += 0.5_rt*(U(i,j,k+1,UMZ)/U(i,j,k+1,URHO) -
                                U(i,j,k-1,UMZ)/U(i,j,k-1,URHO))/dx[2];
#endif
    // Compute pressure of heavy species
    Real e_h = state(i,j,k,UFX);
    Real p_h = 2./3 * rho*e_h;
    src(i,j,k,UFX) = -p_h * divU;
}

#endif
