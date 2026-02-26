#ifndef ACTUAL_CONDUCTIVITY_H
#define ACTUAL_CONDUCTIVITY_H

#include <cmath>
#include <eos_type.H>
#include <network.H>
#include <extern_parameters.H>
#include <collision_frequencies.H>
#include <AMReX_Math.H>
#include <ionization_table.H>


AMREX_FORCE_INLINE
void
actual_conductivity_init()
{
    // Initialization function if needed
}

/**
 * @brief Calculate the thermal conductivity based on electron-ion and electron-neutral collision frequencies.
 *
 * @param[in,out] state The state object containing density (rho), temperature (T), species mass fractions (xn), and where conductivity will be stored.
 *
 * @details This function computes thermal conductivity using the matrix formalism approach, accounting for both
 * electron-ion (Coulomb) and electron-neutral collisions. The result is converted from SI to CGS units.
 * Assumes overall charge neutrality to infer electron density from species composition.
 */
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void actual_conductivity (T& state)
{
    // Infer electron density by summing over species and assuming overall neutrality
    // See https://github.com/AMReX-Astro/Microphysics/blob/main/EOS/gamma_law/actual_eos.H#L62
    amrex::Real n_n_species[num_atom_species] = {};
    amrex::Real n_e = 0.0;
    for (int mu=0; mu < NumSpec; mu++) {
        n_e += zion[mu] * state.xn[mu] * state.rho / C::m_u * 1.e6;
        if (zion[mu] == 0) {
            int idx_atom = get_atom_index(mu);
            n_n_species[idx_atom] = state.rho * state.xn[mu] * aion_inv[mu] / C::m_u * 1.e6; // number density of neutral species mu in m^-3
        }
    }
    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state.T;

    // Get collision frequencies
    amrex::Real v_ei;
    amrex::Real v_en[NumSpec]; // electron-neutral collision frequency for all species

    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)

    get_collision_frequencies(n_e, n_n_species, T_e, T_h, v_ei, v_en);
    amrex::Real v_en_tot = 0.0;
    amrex::Real v_ei_tot = 0.0;
    for (int mu = 0; mu < NumSpec; mu++) {
        if (state.xn[mu] > X_min) {
            v_en_tot += v_en[mu];
            v_ei_tot += v_ei * std::pow(zion[mu],4);
        }
    }
    // Calculate thermal conductivity using Eq. B1 in https://arxiv.org/pdf/2305.16779
    // The coefficient 3.16 comes from matching the Spitzer-Harm conductivity
    // in the fully ionized case (v_en = 0)
    const amrex::Real conductivitySI =  SI::k_B * SI::k_B * n_e * T_e
        / (SI::m_e * amrex::Real((2*M_PI/15) * v_en_tot + amrex::Real(1/3.16) * v_ei_tot));
    // SI to cgs conversion factor for thermal conductivity
    const amrex::Real si_to_cgs_conversion_factor = 1e5; // (10^7 erg/J) / (10^2 cm/m)

    // Convert state.conductivity from SI units to cgs units and store in a new variable
    state.conductivity = conductivitySI * si_to_cgs_conversion_factor;
}
#endif
