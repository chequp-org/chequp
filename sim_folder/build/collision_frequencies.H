#ifndef COLLISION_FREQUENCIES_H
#define COLLISION_FREQUENCIES_H

#include <cmath>
#include <AMReX.H>
#include <AMReX_Math.H>
#include <ionization_table.H>

namespace SI {
    constexpr double k_B = 1.380649e-23; // Boltzmann constant in J/K
    constexpr double q_e = 1.602176634e-19; // Elementary charge in C
    constexpr double epsilon0 = 8.854187817e-12; // Permittivity of free space in F/m
    constexpr double m_e = 9.1093837015e-31; // Electron mass in kg
}
/**
 * @brief Computes electron-ion and electron-neutral collision frequencies.
 *
 * Calculates collision frequencies based on the matrix formalism for Hydrogen.
 * Uses Coulomb logarithm for electron-ion collisions and cross-section data from LXCat
 * for electron-neutral collisions.
 *
 * @param[in] n_e Electron density [m^-3]
 * @param[in] n_n Array of neutral densities [m^-3]
 * @param[in] T_e Electron temperature [K]
 * @param[in] T_h Ion temperature [K]
 * @param[out] v_ei Electron-ion collision frequency [Hz]
 * @param[out] v_en Array of electron-neutral collision frequencies [Hz]
 *
 * @see https://arxiv.org/pdf/2305.16779
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_collision_frequencies(amrex::Real n_e, amrex::Real* n_n, amrex::Real T_e, amrex::Real T_h,
                              amrex::Real& v_ei, amrex::Real* v_en)
{
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
    amrex::Real mu_en[num_atom_species] = {};
    for (int mu = 0; mu < num_atom_species; mu++) {
            mu_en[mu] = SI::m_e * aion[mu] * C::m_u * 1e-3 / (SI::m_e + aion[mu] * C::m_u * 1e-3);}
    // - Calculate cross section for electron-neutral collisions, using polynomial fit of the
    //   Morgan database, www.lxcat.net
    // Hydrogen
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
    amrex::Real sigma_m2_H = std::exp(log_sigma_m2_H);
    // Nitrogen 
    amrex::Real log_sigma_m2_N;
    if (E_bar < 0.1*SI::q_e) {
        log_sigma_m2_N = -4.34e+01;
    } else {
        const amrex::Real log_E_eV = std::log(E_bar / SI::q_e);
        log_sigma_m2_N = -4.34e+01
            -1.91e-1*log_E_eV
            +5.50e-3*amrex::Math::powi<2>(log_E_eV)
            -8.56e-3*amrex::Math::powi<3>(log_E_eV)
            +2.02e-2*amrex::Math::powi<4>(log_E_eV)
            -9.21e-3*amrex::Math::powi<5>(log_E_eV)
            +1.04e-3*amrex::Math::powi<6>(log_E_eV);
    }
    amrex::Real sigma_m2_N = std::exp(log_sigma_m2_N);
    // Helium
    amrex::Real log_sigma_m2_He;
    if (E_bar < 0.1*SI::q_e) {
        log_sigma_m2_He = -4.41e1;
    } else {
        const amrex::Real log_E_eV = std::log(E_bar / SI::q_e);
        log_sigma_m2_He = -4.41e1
            +1.22e-1*log_E_eV
            +2.40e-3*amrex::Math::powi<2>(log_E_eV)
            -5.15e-2*amrex::Math::powi<3>(log_E_eV)
            -7.32e-3*amrex::Math::powi<4>(log_E_eV)
            +3.27e-3*amrex::Math::powi<5>(log_E_eV)
            -2.39e-4*amrex::Math::powi<6>(log_E_eV);
    }
    amrex::Real sigma_m2_He = std::exp(log_sigma_m2_He);
    // Argon
    amrex::Real log_sigma_m2_Ar;
    if (E_bar < 0.1*SI::q_e) {
        log_sigma_m2_Ar = -4.59e-1;
    } else {
        const amrex::Real log_E_eV = std::log(E_bar / SI::q_e);
        log_sigma_m2_Ar = -4.59e-1
            +9.15e-1*log_E_eV
            -3.63e-1*amrex::Math::powi<2>(log_E_eV)
            +5.30e-1*amrex::Math::powi<3>(log_E_eV)
            +7.68e-2*amrex::Math::powi<4>(log_E_eV)
            -1.96e-1*amrex::Math::powi<5>(log_E_eV)
            +3.19e-2*amrex::Math::powi<6>(log_E_eV)
            +1.32e-2*amrex::Math::powi<7>(log_E_eV)
            -5.07e-3*amrex::Math::powi<8>(log_E_eV)
            +6.17e-4*amrex::Math::powi<9>(log_E_eV)
            -2.62e-5*amrex::Math::powi<10>(log_E_eV);
    }
    amrex::Real sigma_m2_Ar = std::exp(log_sigma_m2_Ar);

    // Calculate v_en for each species
    amrex::Real sigma_m2[num_atom_species] = {sigma_m2_H, sigma_m2_N, sigma_m2_He, sigma_m2_Ar};
    for (int mu = 0; mu < NumSpec; mu++) {
        v_en[mu] = 0.0; // default zero
        const int idx_spec = get_atom_index(mu);
        if (zion[mu] > 0){
            v_en[mu] = n_n[idx_spec] * 4.0/3.0 * std::sqrt(2.0*E_bar / mu_en[idx_spec]) * sigma_m2[idx_spec];
        }
    }
}

#endif
