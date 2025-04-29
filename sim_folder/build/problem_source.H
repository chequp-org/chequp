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

    auto coord = geomdata.Coord();
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();

    // Density and temperature
    Real rho = state(i,j,k,URHO);
    Real T = state(i,j,k,UTEMP);

    // Abundance of neutral and ionized H
    Real X_Hn = state(i,j,k,UFS) / rho;
    Real X_Hp = state(i,j,k,UFS+1) / rho;

    // Electron and heavy temperatures
    Real f = state(i,j,k,UFX)/rho;  // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;  // internal energy per unit mass
    Real T_h = 2 * C::m_u * e * f / (3 * C::k_B);
    Real T_e = 2 * C::m_u * e * (1 - f) / (3 * X_Hp * C::k_B);

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

    // In order to calculate the energy transfer between electrons and heavy particles,
    // compute the collision frequency between electrons and ions (using SI units)
    const double k_B = 1.380649e-23; // Boltzmann constant in J/K
    const double q_e = 1.602176634e-19; // Elementary charge in C
    const double epsilon0 = 8.854187817e-12; // Permittivity of free space in F/m
    const double m_e = 9.1093837015e-31; // Electron mass in kg
    const double m_H = 1.67262192369e-27; // Proton mass in kg
    // Electron density in SI units
    const amrex::Real n_e = rho * X_Hp / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    // Calculate Lambda_ei
    const amrex::Real rawLambdaEi = std::log(
        (3.0 / (2.0 * std::sqrt(M_PI))) * std::sqrt(
            std::pow(4.0 * M_PI * epsilon0 * k_B * T_e, 3) /
            (std::pow(q_e, 6) * n_e * (1.0 + T_e / T_h))
        )
    );
    // Apply lower limit
    const amrex::Real Lambda_ei = std::max(rawLambdaEi, 0.5 * std::log(2.0));
    // Calculate v_ei (or v_eH) using the corrected expression
    const amrex::Real v_ei = (4.0 / 3.0) * std::sqrt(2.0 * M_PI / m_e) *
        (n_e * std::pow(q_e, 4) * Lambda_ei) /
        (std::pow(4.0 * M_PI * epsilon0, 2) * std::pow(k_B * T_e, 1.5));
    // Calculate the collision frequency for energy transfer between electrons and ions
    const amrex::Real v_ei_E = 2*m_e/m_H * v_ei;

    // Update fraction of internal energy stored in ions
    src(i,j,k,UFX) = - f/e * src(i,j,k,UEDEN) + rho*X_Hp*v_ei_E*(T_e - T_h)/(T_h + X_Hp*T_e);
}

#endif
