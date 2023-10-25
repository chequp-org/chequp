#ifndef ACTUAL_CONDUCTIVITY_H
#define ACTUAL_CONDUCTIVITY_H

//calculate rho from ne, use the syntax from castro EOS page (eos_state)
// use xn, mass fractions of species
// species.net file for 

#include <cmath>
#include <eos_type.H>
#include <network.H>
#include <extern_parameters.H>
#include <fundamental_constants.H>

using namespace conductivity_rp;
const double epsilon0 = 1;

const std::string cond_name = "constant";



AMREX_FORCE_INLINE
void
actual_conductivity_init()
{
	//maybe define constants here if I can't pass them into T& state
}


template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
actual_conductivity (T& state)
{
// Extract relevant state parameters 
    const Real n_e = state.rho * state.xn[1]/C::m_p;
    const Real T_e = state.T;
    const Real T_h = T_e; // Assuming this parameter is available in your state object
    // Calculate Lambda_ei
    const Real Lambda_ei = std::log(
        (3.0 / (2.0 * std::sqrt(M_PI))) * 
        std::pow(4.0 * M_PI * 1 * C::k_B * T_e, 3) / 
        (std::pow(C::q_e, 6) * n_e * (1.0 + T_e / T_h))
    );

    // Calculate v_ei (or v_eH)
    const Real v_ei = (4.0 / 3.0) * std::sqrt(
        (2.0 * M_PI * n_e * std::pow(C::q_e, 4) * Lambda_ei) / 
        (C::m_e * std::pow(4.0 * M_PI * 1, 2) * std::pow(C::k_B * T_e, 1.5))
    );

    // Calculate thermal conductivity using the earlier formula
    // double gamma_e = state.gamma_e; 

    
    const Real denominator = 1;
    const Real numerator = 15 * std::pow(C::k_B, 2) * n_e * T_e;
    
    state.conductivity = (1.0 / denominator) * (numerator / (2 * M_PI * C::m_e * v_ei));
}
#endif
