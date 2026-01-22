#ifndef ACTUAL_CONDUCTIVITY_H
#define ACTUAL_CONDUCTIVITY_H

#include <cmath>
#include <eos_type.H>
#include <network.H>
#include <extern_parameters.H>
#include <prob_parameters.H>
#include <collision_frequencies.H>
#include <AMReX_Math.H>
#include <IonizationTable.H>


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

    std::vector<int> idx_spec; // Indices of the species to consider
    std::istringstream iss(problem::species);
    std::string sp;

    while (iss >> sp) {
        // Search for the species in the ion_names array
        auto it = std::find(ion_names.begin(), ion_names.end(), sp);
        if (it != ion_names.end()) {
            int idx = std::distance(ion_names.begin(), it);  // get the index
            idx_spec.insert(idx_spec.end(), index_species[idx].begin(), index_species[idx].end()); // Append the indices for this species
        }
    }
    std::sort(idx_spec.begin(), idx_spec.end());
    const int dimS = idx_spec.size(); // Total number of species considered

    // Electron and neutral density
    amrex::Real n_e = 0.0;
    amrex::Real n_n_spe[ion_names.size()] = {0.0, 0.0, 0.0, 0.0};
    amrex::Real start_idx[ion_names.size()] = {1, 3, 9, 12};
    for (int mu = 0; mu < dimS; mu++) {
        int k = idx_spec[mu];
        n_e += zion[k] * state.rho * state.xn[mu] / (aion[k] * C::m_u) * 1.e6; // electron density
        if (zion[k] == 0) {
            // find species s corresponding to k
            for (int s = 0; s < ion_names.size(); s++) {
                if (k == start_idx[s] - 1) {
                    n_n_spe[s] += state.rho * state.xn[mu] / (aion[k] * C::m_u) * 1.e6; // neutral density for species s
                    break; // specie found, exit loop
                }
            }
        }
    }

    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state.T;

    // Get collision frequencies
   
    amrex::Real v_ei;
    amrex::Real v_en[dimS];
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)
    get_collision_frequencies(n_e, n_n_spe, T_e, T_h, v_ei, v_en, idx_spec);
    // Calculate thermal conductivity using Eq. B1 in https://arxiv.org/pdf/2305.16779
    // The coefficient 3.16 comes from matching the Spitzer-Harm conductivity
    // in the fully ionized case (v_en = 0)
    amrex::Real v_en_tot = 0.0; // total electron-neutral collision frequency
    amrex::Real v_ei_tot = 0.0; // total electron-ion collision frequency
    for (int mu = 0; mu < dimS; mu++) {
            if (state.xn[mu] > 1e-20) {
            v_en_tot += v_en[mu];
            v_ei_tot += v_ei * std::pow(zion[idx_spec[mu]],4); ;
        }
    }
    const amrex::Real conductivitySI =  SI::k_B * SI::k_B * n_e * T_e
        / (SI::m_e * amrex::Real((2*M_PI/15) * v_en_tot + amrex::Real(1/3.16) * v_ei_tot)); 
    // SI to cgs conversion factor for thermal conductivity
    const amrex::Real si_to_cgs_conversion_factor = 1e5; // (10^7 erg/J) / (10^2 cm/m)

    // Convert state.conductivity from SI units to cgs units and store in a new variable
    state.conductivity = conductivitySI * si_to_cgs_conversion_factor;
}
#endif
