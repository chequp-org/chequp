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
    // SI unit constants
    const double k_B = 1.380649e-23; // Boltzmann constant in J/K
    const double q_e = 1.602176634e-19; // Elementary charge in C
    const double epsilon0 = 8.854187817e-12; // Permittivity of free space in F/m
    const double m_e = 9.1093837015e-31; // Electron mass in kg
    const double m_H = 1.67262192369e-27; // Hydrogen mass in kg

    const amrex::Real n_e = rho * X_Hp / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    const amrex::Real n_n = rho * X_Hn / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula

    // Compute frequency of electron-ion collisions (Eq. A1 in https://arxiv.org/pdf/2305.16779)
    // - Calculate Lambda_ei
    const amrex::Real rawLambdaEi = std::log(
        (3.0 / (2.0 * std::sqrt(M_PI))) * std::sqrt(
            std::pow(4.0 * M_PI * epsilon0 * k_B * T_e, 3) /
            (std::pow(q_e, 6) * n_e * (1.0 + T_e / T_h))
        )
    );
    // - Apply lower limit
    const amrex::Real Lambda_ei = std::max(rawLambdaEi, 0.5 * std::log(2.0));
    // - Calculate v_ei using the corrected expression
    const amrex::Real v_ei = (4.0 / 3.0) * std::sqrt(2.0 * M_PI / m_e) *
        (n_e * std::pow(q_e, 4) * Lambda_ei) /
        (std::pow(4.0 * M_PI * epsilon0, 2) * std::pow(k_B * T_e, 1.5));

    // Compute frequency of electron-neutral collisions (Eq. A5 in https://arxiv.org/pdf/2305.16779)
    // - Find mean energy and reduced mass
    const amrex::Real E_bar = 4.0 / M_PI * k_B * T_e;
    const amrex::Real mu_en = m_e * m_H / (m_e + m_H);
    // - Calculate cross section for electron-neutral collisions, using polynomial fit of the
    //   Morgan database, www.lxcat.net (PROCESS: E + H -> E + H, Elastic)
    amrex::Real log_sigma_m2;
    if (E_bar < 0.1*q_e) {
        log_sigma_m2 = -4.28906183e+01;
    } else {
        const amrex::Real log_E_eV = std::log(E_bar / q_e);
        log_sigma_m2 = -4.28906183e+01
            -3.20501602e-01*log_E_eV
            -9.75994914e-02*std::pow(log_E_eV, 2)
            -1.72242991e-02*std::pow(log_E_eV, 3)
            + 2.59407262e-03*std::pow(log_E_eV, 4);
    }
    const amrex::Real sigma_m2 = std::exp(log_sigma_m2);
    // - Calculate frequency of electron-neutral collisions
    const amrex::Real v_en = n_n * 4.0/3.0 * std::sqrt(2*E_bar/mu_en) * sigma_m2;

    // Calculate the collision frequency for energy transfer between electrons and heavies
    const amrex::Real v_e_E = 2*m_e/m_H * (v_ei + v_en);

    // Update fraction of internal energy stored in ions
    src(i,j,k,UFX) = - f/e * src(i,j,k,UEDEN) + rho*X_Hp*v_e_E*(T_e - T_h)/(T_h + X_Hp*T_e);
}

#endif
