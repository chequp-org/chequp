#ifndef ACTUAL_CONDUCTIVITY_H
#define ACTUAL_CONDUCTIVITY_H

#include <cmath>
#include <eos_type.H>
#include <network.H>
#include <extern_parameters.H>
#include <collision_frequencies.H>
#include <AMReX_Math.H>
#include <collisional_ionization.H>


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
    amrex::Real n_e = 0.0;      // electron density [m^-3]
    amrex::Real n_ions = 0.0;   // total ion density [m^-3]
    amrex::Real sum_nZ2 = 0.0;  // sum_s n_s Z_s^2 [m^-3]
    for (int mu=0; mu < NumSpec; mu++) {
        const amrex::Real n_mu = state.rho * state.xn[mu] * aion_inv[mu] / C::m_u * 1.e6;
        n_e     += zion[mu] * n_mu;
        sum_nZ2 += zion[mu] * zion[mu] * n_mu;
        if (zion[mu] > 0) {
            n_ions += n_mu;
        } else {
            int idx_atom = get_atom_index(mu);
            n_n_species[idx_atom] = n_mu; // neutral density of this atom, m^-3
        }
    }
    if (n_e <= 0.0_rt) {
        state.conductivity = 0.0_rt;
        return;
    }
    const amrex::Real Zeff = sum_nZ2 / n_e;
    const amrex::Real Zbar = (n_ions > 0.0_rt) ? n_e / n_ions : 1.0_rt;
    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state.T;

    // Get collision frequencies
    amrex::Real v_ei;
    amrex::Real v_en[num_atom_species]; // electron-neutral frequency, one per ATOM

    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)

    get_collision_frequencies(n_e, n_n_species, T_e, T_h, Zbar, v_ei, v_en);

    amrex::Real v_en_tot = 0.0;
    for (int a = 0; a < num_atom_species; a++) {
        v_en_tot += v_en[a];
    }

    const amrex::Real v_ei_tot = Zeff * v_ei;
    // Calculate thermal conductivity using Eq. B1 in https://arxiv.org/pdf/2305.16779
    // With the correct Spitzer-Harm gamma_0 parameter based on Zbar
    // For H we have recover the case where Zbar = 1 (see https://arxiv.org/pdf/2305.16779.pdf Eq. A1)
    // But for other species we need to use the correct Zbar
    const amrex::Real conductivitySI = (2*M_PI/15) * v_en_tot + v_ei_tot / spitzer_gamma0(Zbar);

    if (conductivitySI <= 0.0_rt) {
        state.conductivity = 0.0_rt;
        return;
    }

    // SI to cgs conversion factor for thermal conductivity
    const amrex::Real si_to_cgs_conversion_factor = 1e5; // (10^7 erg/J) / (10^2 cm/m)

    // Convert state.conductivity from SI units to cgs units and store in a new variable
    state.conductivity = conductivitySI * si_to_cgs_conversion_factor;
}
#endif
