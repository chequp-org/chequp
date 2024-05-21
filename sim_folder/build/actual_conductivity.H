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
    const amrex::Real n_e = n_e_cgs * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula

    const amrex::Real T_e = state.T; // Electron temperature
    const amrex::Real T_h = T_e; // Assuming this parameter is available in your state object

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

    // Calculate thermal conductivity using the earlier formula
    const amrex::Real denominator = 1; // Eventually, this should be 1 + gamma_e
    const amrex::Real conductivitySI = (1.0 / denominator) *
        15 * std::pow(k_B, 2) * n_e * T_e / (2 * M_PI * m_e * v_ei);
    // SI to cgs conversion factor for thermal conductivity
    const amrex::Real si_to_cgs_conversion_factor = 1e5; // (10^7 erg/J) / (10^2 cm/m)

    // Convert state.conductivity from SI units to cgs units and store in a new variable
    state.conductivity = conductivitySI * si_to_cgs_conversion_factor;
}
#endif
