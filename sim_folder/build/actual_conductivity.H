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
const double m_p = 1.67262192369e-27; // Proton mass in kg
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
    const Real n_e = state.rho * state.xn[1] / m_p; // Electron number density
    const Real T_e = state.T; // Electron temperature
    const Real T_h = T_e; // Assuming this parameter is available in your state object

    // Calculate Lambda_ei
    const Real rawLambdaEi = std::log(
        (3.0 / (2.0 * std::sqrt(M_PI))) * 
        std::pow(4.0 * M_PI * epsilon0 * k_B * T_e, 3) / 
        (std::pow(q_e, 6) * n_e * (1.0 + T_e / T_h))
    );

    // Apply lower limit
    const Real Lambda_ei = std::max(rawLambdaEi, 0.5 * std::log(2.0));

        // Calculate v_ei (or v_eH) using the corrected expression
    const Real v_ei = (4.0 / 3.0) * std::sqrt(2.0 * M_PI / m_e) * 
        (n_e * std::pow(q_e, 4) * Lambda_ei) / 
        (std::pow(4.0 * M_PI * epsilon0, 2) * std::pow(k_B * T_e, 1.5));


    // Calculate thermal conductivity using the earlier formula
    const Real denominator = 1; // Update this as necessary for the correct SI units
    const Real numerator = 15 * std::pow(k_B, 2) * n_e * T_e;

    const Real conductivitySI = (1.0 / denominator) * (numerator / (2 * M_PI * m_e * v_ei));
    // SI to cgs conversion factor for thermal conductivity
    const double si_to_cgs_conversion_factor = 1e5; // (10^7 erg/J) / (10^2 cm/m)

    // Convert state.conductivity from SI units to cgs units and store in a new variable
    state.conductivity = conductivitySI * si_to_cgs_conversion_factor;
}
#endif
