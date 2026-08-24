#ifndef ACTUAL_CONDUCTIVITY_H
#define ACTUAL_CONDUCTIVITY_H

#include <cmath>
#include <eos_type.H>
#include <network.H>
#include <extern_parameters.H>
#include <prob_parameters.H>
#include <collision_frequencies.H>
#include <AMReX_Math.H>
#include <collisional_ionization.H>


AMREX_FORCE_INLINE
void
actual_conductivity_init()
{
    // Initialization function if needed
}

// Weights of the effective collision frequency in Eq. B1 of https://arxiv.org/pdf/2305.16779.
// The 1/3.16 matches the Spitzer-Harm conductivity in the fully ionized case (v_en = 0).
namespace cond_coeff {
    constexpr double en = 2.0 * M_PI / 15.0;
    constexpr double ei = 1.0 / 3.16;
    // SI to cgs conversion for a conductivity: (10^7 erg/J) / (10^2 cm/m)
    constexpr double si_to_cgs = 1.0e5;
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
    amrex::Real n_e = 0.0_rt;
    amrex::Real z4_n_i = 0.0_rt; // sum of n_i * Z^4 over the ions
    for (int mu=0; mu < NumSpec; mu++) {
        // Clip the negative mass fractions the hydro solver can produce at low density
        const amrex::Real X_mu = std::max(state.xn[mu], 0.0_rt);
        const amrex::Real n_mu = X_mu * state.rho * aion_inv[mu] / C::m_u * 1.e6; // in m^-3
        if (zion[mu] > 0) {
            n_e += zion[mu] * n_mu;
            z4_n_i += n_mu * amrex::Math::powi<4>(zion[mu]);
        } else {
            n_n_species[get_atom_index(mu)] += n_mu;
        }
    }
    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state.T;

    // Get collision frequencies
    amrex::Real v_ei;
    amrex::Real v_en[NumSpec];               // per simulation species
    amrex::Real v_en_atom[num_atom_species]; // per atom species

    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)

    get_collision_frequencies(n_e, n_n_species, T_e, T_h, v_ei, v_en, v_en_atom);

    // v_en depends only on the neutral density of the parent atom, so it is counted once
    // per atom species: summing over ionization stages would scale it with the number of
    // populated stages, and jump whenever a trace stage crosses X_min.
    amrex::Real v_en_tot = 0.0_rt;
    for (int idx_atom = 0; idx_atom < num_atom_species; idx_atom++) {
        v_en_tot += v_en_atom[idx_atom];
    }
    // Z^4 averaged over the ion population, instead of applied in full to every stage
    // above X_min. Continuous in composition, and reduces to v_ei for a single charge state.
    amrex::Real v_ei_tot = 0.0_rt;
    if (n_e > 0.0_rt) {
        v_ei_tot = v_ei * z4_n_i / n_e;
    }

    // Effective collision frequency of Eq. B1
    amrex::Real v_eff = cond_coeff::en * v_en_tot + cond_coeff::ei * v_ei_tot;

    // Free-streaming limiter. v_ei scales with n_e, so the collisional conductivity is
    // density-independent once ionized (Spitzer) and the diffusivity K/(rho*c_v) setting
    // Castro's diffusion time step diverges as 1/rho. Adding v_th/lambda_max caps the
    // conductivity at the free-streaming value n_e*k_B*v_th*lambda_max. Additive rather
    // than a hard min() so the conductivity stays smooth.
    if (problem::conductivity_max_mfp > 0.0_rt) {
        const amrex::Real lambda_max = problem::conductivity_max_mfp * 1.e-2_rt; // cm -> m
        const amrex::Real T_e_s = std::max(T_e, amrex::Real(plasma_limits::T_floor));
        v_eff += std::sqrt(SI::k_B * T_e_s / SI::m_e) / lambda_max;
    }

    // Thermal conductivity, Eq. B1 in https://arxiv.org/pdf/2305.16779
    amrex::Real conductivity_cgs = 0.0_rt;
    if (v_eff > 0.0_rt) {
        conductivity_cgs = SI::k_B * SI::k_B * n_e * T_e / (SI::m_e * v_eff) * cond_coeff::si_to_cgs;
    }
    // A vacuum cell has no electrons and no collisions, so the ratio above is 0/0.
    // The test is also false for NaN, so a poisoned state cannot reach the diffusion solver.
    state.conductivity = (conductivity_cgs > 0.0_rt) ? conductivity_cgs : 0.0_rt;
}
#endif
