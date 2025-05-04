#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>

using namespace amrex;

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

    const amrex::Real T_e = state(i,j,k,UTEMP); // Electron temperature
    // TODO: Extract correct ion temperature
    const amrex::Real T_h = T_e;

    // Abundance of neutral and ionized H
    Real X_Hn = state(i,j,k,UFS) / rho;
    Real X_Hp = state(i,j,k,UFS+1) / rho;

    // Electron and heavy temperatures
    Real f = state(i,j,k,UFX)/rho;  // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;  // internal energy per unit mass

    // Rate of ionization
    Real R_Hp = rho * rho / C::m_p * (
        problem::a_direct_ioniz *
        std::pow( C::k_B*T_e/C::ev2erg, problem::q_direct_ioniz ) *
        std::exp(-problem::e_direct_ioniz_eV*C::ev2erg/(C::k_B*T_e)) *
        X_Hn * X_Hp
        +
        problem::a_twostep_ioniz *
        std::pow( C::k_B*T_e/C::ev2erg, problem::q_twostep_ioniz ) *
        std::exp(-problem::e_twostep_ioniz_eV*C::ev2erg/(C::k_B*T_e)) *
        X_Hn * X_Hp
    );

    // Update state
    src(i,j,k,UFS) = -R_Hp;
    src(i,j,k,UFS+1) = R_Hp;
    src(i,j,k,UEDEN) = -R_Hp * (problem::e_loss_ioniz_eV*C::ev2erg / C::m_p);
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // In order to calculate the energy transfer between electrons and heavy particles,
    // compute the collision frequency between electrons and ions (using SI units)
    const amrex::Real n_e = rho * X_Hp / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    const amrex::Real n_n = rho * X_Hn / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    // Get collision frequencies
    amrex::Real v_ei, v_en;
    get_collision_frequencies(n_e, n_n, T_e, T_h, v_ei, v_en);

    // Calculate the collision frequency for energy transfer between electrons and heavies
    const amrex::Real v_e_E = 2*SI::m_e/SI::m_H * (v_ei + v_en);

    // Update fraction of internal energy stored in ions
    src(i,j,k,UFX) = - f/e * src(i,j,k,UEDEN) + rho*X_Hp*v_e_E*(T_e - T_h)/(T_h + X_Hp*T_e);
}

#endif
