#ifndef COLLISION_FREQUENCIES_H
#define COLLISION_FREQUENCIES_H

#include <cmath>
#include <AMReX.H>
#include <AMReX_Math.H>
#include <collisional_ionization.H>

namespace SI {
    constexpr double k_B = 1.380649e-23; // Boltzmann constant in J/K
    constexpr double q_e = 1.602176634e-19; // Elementary charge in C
    constexpr double epsilon0 = 8.854187817e-12; // Permittivity of free space in F/m
    constexpr double m_e = 9.1093837015e-31; // Electron mass in kg
}
// -------------------------------------------------------------------------
// Cross-Section Helper Functions
// -------------------------------------------------------------------------

// Helper for rational function approximations
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eval_sigma_rational(amrex::Real log_E_eV, amrex::Real offset, amrex::Real scale,
                                const amrex::Real* num, const amrex::Real* den) {
    const amrex::Real x = (log_E_eV - offset) / scale;
    const amrex::Real x2 = amrex::Math::powi<2>(x);
    const amrex::Real x3 = amrex::Math::powi<3>(x);
    const amrex::Real x4 = amrex::Math::powi<4>(x);
    const amrex::Real num_val = num[0] + num[1]*x + num[2]*x2 + num[3]*x3 + num[4]*x4;
    const amrex::Real den_val = 1.0_rt + den[0]*x + den[1]*x2 + den[2]*x3 + den[3]*x4;
    return std::exp(num_val / den_val);
}

// -------------------------------------------------------------------------
// Species Cross-Sections
// -------------------------------------------------------------------------

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real get_sigma_m2_H(amrex::Real log_E_eV) {
    const amrex::Real num[] = {-4.451560e+01_rt, -4.477831e+01_rt, -6.605789e+02_rt,
        9.020842e+01_rt, -1.798055e+02_rt};
    const amrex::Real den[] = {8.890529e-01_rt, 1.461417e+01_rt, -3.442456e+00_rt, 3.765710e+00_rt};
    return eval_sigma_rational(log_E_eV, 2.606962e+00_rt, 4.605170e+00_rt, num, den);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real get_sigma_m2_He(amrex::Real log_E_eV) {
    const amrex::Real num[] = {-4.538238e+01_rt, 9.075481e+01_rt, -3.875089e+02_rt,
        6.894316e+02_rt, -5.129732e+02_rt};
    const amrex::Real den[] = {-2.215854e+00_rt, 8.615551e+00_rt, -1.553593e+01_rt, 1.151640e+01_rt};
    return eval_sigma_rational(log_E_eV, 3.261721e+00_rt, 8.006685e+00_rt, num, den);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real get_sigma_m2_N(amrex::Real log_E_eV) {
    const amrex::Real num[] = {-4.333081e+01_rt, -1.969554e+02_rt, -2.120393e+02_rt,
        2.839629e+01_rt, -4.466647e+01_rt};
    const amrex::Real den[] = {4.530368e+00_rt, 4.856744e+00_rt, -8.234106e-01_rt, 7.219589e-01_rt};
    return eval_sigma_rational(log_E_eV, -3.991000e-01_rt, 5.756650e+00_rt, num, den);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real get_sigma_m2_Ar(amrex::Real log_E_eV) {
    const amrex::Real num[] = {-4.341181e+01_rt, 6.670171e+01_rt, -4.380806e+02_rt,
        -2.230848e+03_rt, -2.086728e+03_rt};
    const amrex::Real den[] = {-1.425189e+00_rt, 9.207079e+00_rt, 4.881694e+01_rt, 4.621947e+01_rt};
    return eval_sigma_rational(log_E_eV, 2.128350e+00_rt, 6.152029e+00_rt, num, den);
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
        amrex::Real(3.0_rt / (2.0_rt * std::sqrt(M_PI))) * std::sqrt(
            amrex::Math::powi<3>(4.0_rt * M_PI * SI::epsilon0 * SI::k_B * T_e) /
            (amrex::Math::powi<6>(SI::q_e) * n_e * (1.0_rt + T_e / T_h))
        )
    );
    // - Apply lower limit
    const amrex::Real Lambda_ei = std::max(rawLambdaEi, 0.5_rt * std::log(2.0_rt));
    // - Calculate v_ei using the corrected expression
    v_ei = (4.0_rt / 3.0_rt) * std::sqrt(2.0_rt * M_PI / SI::m_e) *
        (n_e * amrex::Math::powi<4>(SI::q_e) * Lambda_ei) /
        (amrex::Math::powi<2>(4.0_rt * M_PI * SI::epsilon0) 
        * (SI::k_B * T_e) * std::sqrt(SI::k_B * T_e));

    // Compute frequency of electron-neutral collisions (Eq. A5 in https://arxiv.org/pdf/2305.16779)
    // - Find mean energy and reduced mass
    const amrex::Real E_bar = 4.0_rt / M_PI * SI::k_B * T_e;
    const amrex::Real log_E_eV = std::log(E_bar / SI::q_e);
    amrex::Real mu_en[num_atom_species] = {};
    for (int mu = 0; mu < num_atom_species; mu++) {
            mu_en[mu] = SI::m_e * aion[mu] * C::m_u * 1e-3_rt / (SI::m_e + aion[mu] * C::m_u * 1e-3_rt);}
    // Populate cross sections using our helpers
    amrex::Real sigma_m2[num_atom_species] = {
        get_sigma_m2_H(log_E_eV),
        get_sigma_m2_N(log_E_eV),
        get_sigma_m2_He(log_E_eV),
        get_sigma_m2_Ar(log_E_eV)
    };

    // Calculate v_en for each species
    for (int mu = 0; mu < NumSpec; mu++) {
        v_en[mu] = 0.0_rt; // default zero
        const int idx_spec = get_atom_index(mu);
        if (zion[mu] > 0){
            v_en[mu] = n_n[idx_spec] * 4.0_rt / 3.0_rt 
                * std::sqrt(2.0_rt * E_bar / mu_en[idx_spec]) * sigma_m2[idx_spec];
        }
    }
}

#endif