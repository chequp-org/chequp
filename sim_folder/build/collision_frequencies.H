#ifndef COLLISION_FREQUENCIES_H
#define COLLISION_FREQUENCIES_H

#include <cmath>
#include <AMReX.H>
#include <AMReX_Math.H>
#include <ionization_table.H>

namespace SI {
    const double k_B = 1.380649e-23; // Boltzmann constant in J/K
    const double q_e = 1.602176634e-19; // Elementary charge in C
    const double epsilon0 = 8.854187817e-12; // Permittivity of free space in F/m
    const double m_e = 9.1093837015e-31; // Electron mass in kg
    const double m_H = 1.67262192369e-27; // Hydrogen mass in kg
    const double m_N = 2.3258671e-26; // Nitrogen mass in kg
    const double m_He = 6.6464764e-27; // Helium mass in kg 
    const double m_Ar = 6.6335209e-26; // Argon mass in kg
}
const std::vector<amrex::Real> mass_vec = {SI::m_H, SI::m_N, SI::m_He, SI::m_Ar};
/**
 * @brief Computes electron-ion and electron-neutral collision frequencies.
 *
 * This function calculates collision frequencies for electron interactions with ions and neutral atoms
 * based on kinetic theory. The electron-ion collision frequency is computed using the Coulomb logarithm,
 * while electron-neutral collision frequencies use cross-section data from the Morgan database.
 *
 * @param[in] n_e Electron number density [m^-3]
 * @param[in] n_n_spe Array of neutral species number densities [m^-3]
 * @param[in] T_e Electron temperature [K]
 * @param[in] T_h Heavy particle temperature (ion temperature) [K]
 * @param[out] v_ei Electron-ion collision frequency [s^-1]
 * @param[out] v_en Array of electron-neutral collision frequencies for each species [s^-1]
 * @param[in] idx_spec Vector of species indices to consider in the simulation
 *
 * @details
 * The electron-ion collision frequency is computed using Equation A1 from
 * https://arxiv.org/pdf/2305.16779, with a lower limit applied to the Coulomb logarithm (Lambda_ei >= 0.5*ln(2)).
 *
 * The electron-neutral collision frequencies are computed using Equation A5 from the same reference.
 * Cross-sections for electron-neutral collisions (E + H -> E + H, Elastic) are obtained from polynomial
 * fits of the Morgan database (www.lxcat.net). Cross-sections for other species are to be updated.
 *
 * @note This function is GPU-compatible and marked for inlining.
 * @note Cross-sections for nitrogen and other species are currently set to zero and require implementation.
 *
 * @see https://arxiv.org/pdf/2305.16779
 * @see www.lxcat.net
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_collision_frequencies(amrex::Real n_e, amrex::Real* n_n_spe, amrex::Real T_e, amrex::Real T_h,
                               amrex::Real& v_ei, amrex::Real* v_en, std::vector<int>& idx_spec)
{   
    // Constant variables
    // Compute frequency of electron-ion collisions (Eq. A1 in https://arxiv.org/pdf/2305.16779)
    // Calculate Lambda_ei
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

    // Following variables are now in vector form
    amrex::Real mu_en[num_atom_species] = {};
    for (std::size_t mu = 0; mu < num_atom_species; mu++) {
        mu_en[mu] = SI::m_e * mass_vec[mu] / (SI::m_e + mass_vec[mu]);
    }
    // - Calculate cross section for electron-neutral collisions, using polynomial fit of the
    //   Morgan database, www.lxcat.net 
    
    // PROCESS: E + H -> E + H, Elastic)
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
    const amrex::Real sigma_m2_H = std::exp(log_sigma_m2_H); // in m^2
    const amrex::Real sigma_m2_N = 0.0; // TO BE UPDATED

    // - Calculate v_en for each species

    amrex::Real sigma_m2[num_atom_species] = {sigma_m2_H, sigma_m2_N, 0.0, 0.0}; // TO BE UPDATED FOR ALL SPECIES
    int dim_sim_species = idx_spec.size(); // Total number of species and their ionized level considered
    for (int mu = 0; mu < dim_sim_species; mu++) {
        int k = idx_spec[mu]; // species index with the shifted index
        v_en[mu] = 0.0; // default zero
        for (std::size_t s = 0; s < num_atom_species; s++) {
            if (k >= start_idx[s] && k < start_idx[s+1] -1 ) { // species s found
                v_en[mu] = n_n_spe[s] * 4.0/3.0 * std::sqrt(2.0*E_bar / mu_en[s]) * sigma_m2[s]; // electron-neutral collision frequency
                break; // found species, exit loop
            }
        }
    }
}

#endif