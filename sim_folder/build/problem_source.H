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
    src(i,j,k,UEDEN) = 0;
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);
}

#endif
