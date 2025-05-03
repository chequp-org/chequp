#ifndef ACTUAL_CONDUCTIVITY_H
#define ACTUAL_CONDUCTIVITY_H

#include <cmath>
#include <eos_type.H>
#include <network.H>
#include <extern_parameters.H>

// SI unit constants
const double k_B = 1.380649e-23; // Boltzmann constant in J/K
const double q_e = 1.602176634e-19; // Elementary charge in C
const double epsilon0 = 8.854187817e-12; // Permittivity of free space in F/m
const double m_e = 9.1093837015e-31; // Electron mass in kg
const double m_H = 1.67262192369e-27; // Hydrogen mass in kg

AMREX_FORCE_INLINE
void
actual_conductivity_init()
{
    // Initialization function if needed
}

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
actual_conductivity (T& state)
{
    // Extract relevant state parameters

    // Infer electron density by summing over species and assuming overall neutrality
    // See https://github.com/AMReX-Astro/Microphysics/blob/main/EOS/gamma_law/actual_eos.H#L62
    // for a similar calculation
    amrex::Real sum = 0;
    for (int n=0; n < NumSpec; n++) {
        sum += zion[n] * state.xn[n] * aion_inv[n];
    }
    const amrex::Real n_e_cgs = state.rho * sum / C::m_u;
    const amrex::Real n_n_cgs = state.rho * (1-sum) / C::m_u;
    const amrex::Real n_e = n_e_cgs * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    const amrex::Real n_n = n_n_cgs * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula

    const amrex::Real T_e = state.T; // Electron temperature
    // TODO: Extract correct ion temperature
    const amrex::Real T_h = T_e; // Assuming this parameter is available in your state object

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
    // - Calculate cross section for electron-neutral collisions, using polynomial of the
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

    // Calculate thermal conductivity using Eq. B1 in https://arxiv.org/pdf/2305.16779
    const amrex::Real gamma_e = 0; // TODO: Use proper value
    const amrex::Real conductivitySI = 15 * std::pow(k_B, 2) * n_e * T_e / (2 * M_PI * m_e * (v_en + (1+gamma_e)*v_ei));
    // SI to cgs conversion factor for thermal conductivity
    const amrex::Real si_to_cgs_conversion_factor = 1e5; // (10^7 erg/J) / (10^2 cm/m)

    // Convert state.conductivity from SI units to cgs units and store in a new variable
    state.conductivity = conductivitySI * si_to_cgs_conversion_factor;
}
#endif
