#ifndef COLLISION_FREQUENCIES_H
#define COLLISION_FREQUENCIES_H

#include <cmath>
#include <AMReX.H>
#include <AMReX_Math.H>

namespace SI {
    const double k_B = 1.380649e-23; // Boltzmann constant in J/K
    const double q_e = 1.602176634e-19; // Elementary charge in C
    const double epsilon0 = 8.854187817e-12; // Permittivity of free space in F/m
    const double m_e = 9.1093837015e-31; // Electron mass in kg
    const double m_H = 1.67262192369e-27; // Hydrogen mass in kg
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_collision_frequencies(amrex::Real n_e, amrex::Real n_n, amrex::Real T_e, amrex::Real T_h,
                              amrex::Real& v_ei, amrex::Real& v_en)
{
    // Compute frequency of electron-ion collisions (Eq. A1 in https://arxiv.org/pdf/2305.16779)
    // - Calculate Lambda_ei
    const amrex::Real rawLambdaEi = std::log(
        amrex::Real(3.0 / (2.0 * std::sqrt(M_PI))) * std::sqrt(
            amrex::Math::powi<3>(4.0 * M_PI * SI::epsilon0 * SI::k_B * T_e) /
            (namrex::Math::powi<6>(SI::q_e) * n_e * (1.0 + T_e / T_h))
        )
    );
    // - Apply lower limit
    const amrex::Real Lambda_ei = std::max(rawLambdaEi, 0.5 * std::log(2.0));
    // - Calculate v_ei using the corrected expression
    v_ei = (4.0 / 3.0) * std::sqrt(2.0 * M_PI / SI::m_e) *
        (n_e * amrex::Math::powi<4>(SI::q_e) * Lambda_ei) /
        (amrex::Math::powi<2>(4.0 * M_PI * SI::epsilon0) * (SI::k_B * T_e) * std::sqrt(SI::k_B * T_e));

    // Compute frequency of electron-neutral collisions (Eq. A5 in https://arxiv.org/pdf/2305.16779)
    // - Find mean energy and reduced mass
    const amrex::Real E_bar = 4.0 / M_PI * SI::k_B * T_e;
    const amrex::Real mu_en = SI::m_e * SI::m_H / (SI::m_e + SI::m_H);
    // - Calculate cross section for electron-neutral collisions, using polynomial fit of the
    //   Morgan database, www.lxcat.net (PROCESS: E + H -> E + H, Elastic)
    amrex::Real log_sigma_m2;
    if (E_bar < 0.1*SI::q_e) {
        log_sigma_m2 = -4.28906183e+01;
    } else {
        const amrex::Real log_E_eV = std::log(E_bar / SI::q_e);
        log_sigma_m2 = -4.28906183e+01
            -3.20501602e-01*log_E_eV
            -9.75994914e-02*amrex::Math::powi<2>(log_E_eV)
            -1.72242991e-02*amrex::Math::powi<3>(log_E_eV)
            + 2.59407262e-03*amrex::Math::powi<4>(log_E_eV);
    }
    const amrex::Real sigma_m2 = std::exp(log_sigma_m2);
    // - Calculate frequency of electron-neutral collisions
    v_en = n_n * 4.0/3.0 * std::sqrt(2*E_bar/mu_en) * sigma_m2;
}

#endif