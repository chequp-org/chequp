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
    constexpr double m_H = 1.67262192369e-27; // Hydrogen mass in kg
    constexpr double m_N = 2.3258671e-26; // Nitrogen mass in kg
    constexpr double m_He = 6.6464764e-27; // Helium mass in kg 
    constexpr double m_Ar = 6.6335209e-26; // Argon mass in kg
}
AMREX_GPU_HOST_DEVICE
constexpr std::array<amrex::Real, 4> mass_vec = {SI::m_H, SI::m_N, SI::m_He, SI::m_Ar};
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
            mu_en[mu] = SI::m_e * mass_vec[mu] / (SI::m_e + mass_vec[mu]);}
    // - Calculate cross section for electron-neutral collisions, using polynomial fit of the
    //   Morgan database, www.lxcat.net
    // Hydrogen
    // H - Rational function fit
    const amrex::Real log_E_norm_H = (std::log(E_bar / SI::q_e) - 2.606962e+00) / 4.605170e+00;
    amrex::Real numerator_H = -4.451560e+01 + -4.477831e+01*amrex::Math::powi<1>(log_E_norm_H) + -6.605789e+02*amrex::Math::powi<2>(log_E_norm_H) + 9.020842e+01*amrex::Math::powi<3>(log_E_norm_H) + -1.798055e+02*amrex::Math::powi<4>(log_E_norm_H);
    amrex::Real denominator_H = 1.0 + 8.890529e-01*amrex::Math::powi<1>(log_E_norm_H) + 1.461417e+01*amrex::Math::powi<2>(log_E_norm_H) + -3.442456e+00*amrex::Math::powi<3>(log_E_norm_H) + 3.765710e+00*amrex::Math::powi<4>(log_E_norm_H);
    amrex::Real sigma_m2_H = std::exp(numerator_H / denominator_H);


    // Nitrogen 
    // N - Lookup table with linear interpolation in log-log space
    // Total points: 8

    // Lookup table: log(E_eV) values
    static const amrex::Real log_E_table_N[] = {
        -2.1931927162e+00, -1.8124112570e+00, -6.0842007720e-01, 6.4433501615e-01, 1.1833332669e+00, 
        1.5887983750e+00, 2.3227601999e+00, 4.9130298154e+00
    };

    // Lookup table: log(sigma_m2) values
    static const amrex::Real log_sigma_table_N[] = {
        -4.1798399478e+01, -4.2169818779e+01, -4.3041034358e+01, -4.3320742142e+01, -4.3450829719e+01, 
        -4.3621938169e+01, -4.3706800093e+01, -4.3783406605e+01, -4.4693000603e+01, -4.5276580652e+01
    };

    static const int table_size_N = 8;

    // Calculate log(E_eV)
    const amrex::Real log_E_eV = std::log(E_bar / SI::q_e);
    amrex::Real log_sigma_m2_N;

    // Low energy extrapolation (linear in log-log)
    if (log_E_eV < log_E_table_N[0]) {
        const amrex::Real slope = -2.9262399064e+00;
        log_sigma_m2_N = -4.1798399478e+01 + 
            slope * (log_E_eV - log_E_table_N[0]);
    }
    // High energy extrapolation (linear in log-log)
    else if (log_E_eV > log_E_table_N[table_size_N-1]) {
        const amrex::Real slope = -6.7589108677e-01;
        log_sigma_m2_N = -4.5276580652e+01 + 
            slope * (log_E_eV - log_E_table_N[table_size_N-1]);
    }
    // Interior: linear interpolation in lookup table
    else {
        // Binary search for the interval
        int i_low = 0;
        int i_high = table_size_N - 1;
        while (i_high - i_low > 1) {
            int i_mid = (i_low + i_high) / 2;
            if (log_E_eV < log_E_table_N[i_mid]) {
                i_high = i_mid;
            } else {
                i_low = i_mid;
            }
        }
        
        // Linear interpolation
        const amrex::Real log_E_low = log_E_table_N[i_low];
        const amrex::Real log_E_high = log_E_table_N[i_high];
        const amrex::Real log_sigma_low = log_sigma_table_N[i_low];
        const amrex::Real log_sigma_high = log_sigma_table_N[i_high];
        
        const amrex::Real alpha = (log_E_eV - log_E_low) / (log_E_high - log_E_low);
        log_sigma_m2_N = log_sigma_low + alpha * (log_sigma_high - log_sigma_low);
    }

    amrex::Real sigma_m2_N = std::exp(log_sigma_m2_N);
    // Helium
    // He - Lookup table with linear interpolation in log-log space
    // Total points: 7

    // Lookup table: log(E_eV) values
    static const amrex::Real log_E_table_He[] = {
        -4.6051701860e+00, 0.0000000000e+00, 1.3862943611e+00, 2.3025850930e+00, 3.4011973817e+00, 
        4.6051701860e+00, 6.9077552790e+00
    };

    // Lookup table: log(sigma_m2) values
    static const amrex::Real log_sigma_table_He[] = {
        -4.4388434801e+01, -4.4351113483e+01, -4.4159240137e+01, -4.4068125055e+01, -4.4400551213e+01, 
        -4.5547157836e+01, -4.8255889239e+01, -5.0076072728e+01, -5.1406099577e+01
    };

    static const int table_size_He = 7;
    amrex::Real log_sigma_m2_He;

    // Low energy extrapolation (linear in log-log)
    if (log_E_eV < log_E_table_He[0]) {
        const amrex::Real slope = 2.4312663650e-02;
        log_sigma_m2_He = -4.4388434801e+01 + 
            slope * (log_E_eV - log_E_table_He[0]);
    }
    // High energy extrapolation (linear in log-log)
    else if (log_E_eV > log_E_table_He[table_size_He-1]) {
        const amrex::Real slope = -1.7328699641e+00;
        log_sigma_m2_He = -5.1406099577e+01 + 
            slope * (log_E_eV - log_E_table_He[table_size_He-1]);
    }
    // Interior: linear interpolation in lookup table
    else {
        // Binary search for the interval
        int i_low = 0;
        int i_high = table_size_He - 1;
        while (i_high - i_low > 1) {
            int i_mid = (i_low + i_high) / 2;
            if (log_E_eV < log_E_table_He[i_mid]) {
                i_high = i_mid;
            } else {
                i_low = i_mid;
            }
        }
        
        // Linear interpolation
        const amrex::Real log_E_low = log_E_table_He[i_low];
        const amrex::Real log_E_high = log_E_table_He[i_high];
        const amrex::Real log_sigma_low = log_sigma_table_He[i_low];
        const amrex::Real log_sigma_high = log_sigma_table_He[i_high];
        
        const amrex::Real alpha = (log_E_eV - log_E_low) / (log_E_high - log_E_low);
        log_sigma_m2_He = log_sigma_low + alpha * (log_sigma_high - log_sigma_low);
    }

    amrex::Real sigma_m2_He = std::exp(log_sigma_m2_He);
    // Argon
    // Ar - Rational function fit
    const amrex::Real log_E_norm_Ar = (std::log(E_bar / SI::q_e) - -2.964491e-01) / 3.656610e+00;
    amrex::Real numerator_Ar = -4.603229e+01 + -9.762298e+01*amrex::Math::powi<1>(log_E_norm_Ar) + 7.863646e+01*amrex::Math::powi<2>(log_E_norm_Ar) + 3.714900e+02*amrex::Math::powi<3>(log_E_norm_Ar) + -4.349251e+02*amrex::Math::powi<4>(log_E_norm_Ar);
    amrex::Real denominator_Ar = 1.0 + 2.227788e+00*amrex::Math::powi<1>(log_E_norm_Ar) + -1.683525e+00*amrex::Math::powi<2>(log_E_norm_Ar) + -8.217511e+00*amrex::Math::powi<3>(log_E_norm_Ar) + 9.590648e+00*amrex::Math::powi<4>(log_E_norm_Ar);
    amrex::Real sigma_m2_Ar = std::exp(numerator_Ar / denominator_Ar);

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