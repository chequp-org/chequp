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
    const double m_N = 2.3258671e-26; // Nitrogen mass in kg
    const amrex::Real m_vec[NumSpec] = {SI::m_H, SI::m_H, SI::m_N, SI::m_N, SI::m_N, SI::m_N, SI::m_N, SI::m_N};
}

namespace Acoeff {
    const int n_step = 2; // Number of ionization steps
    const amrex::Real A_I[NumSpec][n_step] = {
        // Hydrogen
        {7.1e-15 * 1e6, 12.4e-15 * 1e6}, {0.0, 0.0},
        // Nitrogen
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };

    const amrex::Real q_I[NumSpec][n_step] = {
        // Hydrogen
        {0.4, 0.3}, {0.0, 0.0},
        // Nitrogen
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };

    const amrex::Real eps_I[NumSpec][n_step] = {
        // Hydrogen
        {13.6, 10.6}, {0.0, 0.0},
        // Nitrogen
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };
    const amrex::Real g_deg[NumSpec] = {
        // Hydrogen
        2.0, 1.0,
        // Nitrogen
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    };

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_collision_frequencies(amrex::Real n_e, amrex::Real n_n, amrex::Real T_e, amrex::Real T_h,
                               amrex::Real& v_ei, amrex::Real* v_en)
{   
    // ------ Constant variables ------
    // Compute frequency of electron-ion collisions (Eq. A1 in https://arxiv.org/pdf/2305.16779)
    // - Calculate Lambda_ei
    const amrex::Real rawLambdaEi = std::log(
        amrex::Real(3.0 / (2.0 * std::sqrt(M_PI))) * std::sqrt(
            amrex::Math::powi<3>(4.0 * M_PI * SI::epsilon0 * SI::k_B * T_e) /
            (amrex::Math::powi<6>(SI::q_e) * n_e * (1.0 + T_e / T_h))
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

    // ----- Following variables are now in vector form -------
    const amrex::Real m_vec[NumSpec] = {SI::m_H, SI::m_H, SI::m_N, SI::m_N, SI::m_N, SI::m_N, SI::m_N, SI::m_N};
    amrex::Real mu_en[NumSpec];
    for (int mu = 0; mu < NumSpec; mu++) {
        mu_en[mu] = SI::m_e * m_vec[mu] / (SI::m_e + m_vec[mu]);
    }
    // - Calculate cross section for electron-neutral collisions, using polynomial fit of the
    //   Morgan database, www.lxcat.net 
    
    // ---- (PROCESS: E + H -> E + H, Elastic) ---
    amrex::Real log_sigma_m2_H;
    if (E_bar < 0.1*SI::q_e) {
        log_sigma_m2_H = -4.28906183e+01;
    } else {
        const amrex::Real log_E_eV = std::log(E_bar / SI::q_e);
        log_sigma_m2_H = -4.28906183e+01
            -3.20501602e-01*log_E_eV
            -9.75994914e-02*amrex::Math::powi<2>(log_E_eV)
            -1.72242991e-02*amrex::Math::powi<3>(log_E_eV)
            + 2.59407262e-03*amrex::Math::powi<4>(log_E_eV);
    }
    const amrex::Real sigma_m2_H = std::exp(log_sigma_m2_H);

    // ---- (PROCESS: E + N -> E + N, Elastic) ---

    // TO BE DONE LATER FOR ALL SPECIES 

    // - Calculate frequency of electron-neutral collisions
    amrex::Real sigma_m2[NumSpec] = {0.0, sigma_m2_H, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // TO BE UPDATED FOR ALL SPECIES
    for (int mu = 0; mu < NumSpec; mu++) {
        v_en[mu] = n_n * 4.0/3.0 * std::sqrt(2*E_bar/mu_en[mu]) * sigma_m2[mu];
    }
}

#endif