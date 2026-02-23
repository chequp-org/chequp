#ifndef ACTUAL_CONDUCTIVITY_H
#define ACTUAL_CONDUCTIVITY_H

#include <cmath>
#include <eos_type.H>
#include <network.H>
#include <extern_parameters.H>
#include <collision_frequencies.H>
#include <AMReX_Math.H>


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
    // Infer electron density by summing over species and assuming overall neutrality
    // See https://github.com/AMReX-Astro/Microphysics/blob/main/EOS/gamma_law/actual_eos.H#L62
    // for a similar calculation
    amrex::Real sum = 0;
    for (int n=0; n < NumSpec; n++) {
        sum += zion[n] * state.xn[n] * aion_inv[n];
    }
    const amrex::Real n_e_cgs = state.rho * sum / C::m_u;
    const amrex::Real n_n_cgs = state.rho * state.xn[0] * aion_inv[0] / C::m_u;
    const amrex::Real n_e = n_e_cgs * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    const amrex::Real n_n = n_n_cgs * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula

    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state.T;

    // Get collision frequencies
    amrex::Real v_ei, v_en;
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)
    get_collision_frequencies(n_e, n_n, T_e, T_h, v_ei, v_en);

    // Calculate thermal conductivity using Eq. B1 in https://arxiv.org/pdf/2305.16779
    // The coefficient 3.16 comes from matching the Spitzer-Harm conductivity
    // in the fully ionized case (v_en = 0)
    const amrex::Real conductivitySI =  SI::k_B * SI::k_B * n_e * T_e
        / (SI::m_e * amrex::Real((2*M_PI/15) * v_en + amrex::Real(1/3.16) * v_ei));
    // SI to cgs conversion factor for thermal conductivity
    const amrex::Real si_to_cgs_conversion_factor = 1e5; // (10^7 erg/J) / (10^2 cm/m)

    // Convert state.conductivity from SI units to cgs units and store in a new variable
    state.conductivity = conductivitySI * si_to_cgs_conversion_factor;
}
#endif
