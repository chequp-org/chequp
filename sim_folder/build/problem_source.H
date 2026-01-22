#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <IonizationTable.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source(
    int i, int j, int k,
    GeometryData const& geomdata,
    Array4<const Real> const& state,
    Array4<Real> const& src,
    const Real dt,
    const Real time)
{
    using namespace problem;
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
    // Density and electron temperature
    Real rho = state(i,j,k,URHO);
    Real T_e = state(i,j,k,UTEMP);

    // Step (1) : Get current mass fractions
    Real X_stage[dimS];
    for (int mu = 0; mu < dimS; mu++) {
        X_stage[mu] = state(i,j,k,UFS+idx_spec[mu]) / rho;
    }
    // Electron fraction
    Real X_e = 0.0;
    for (int mu = 0; mu < dimS; mu++) {
        X_e += zion[idx_spec[mu]] * X_stage[mu];
    }
    // Reaction matrix that have the dimension of the species considered
    Real S_r[dimS][dimS];
    // Loop over target state alpha (row)
    for (int alpha = 0; alpha < dimS; alpha++) {
        for (int beta = 0; beta < dimS; beta++) {
            Real val = 0.0;
            // --- Electron-impact ionization ---
            if (alpha == beta) { // Diagonal: loss
                for (int step = 0; step < Acoeff::n_step; step++) {
                    val -= Acoeff::A_I[idx_spec[beta]][step] *
                        std::pow(C::k_B*T_e / C::ev2erg, Acoeff::q_I[idx_spec[beta]][step]) *
                        std::exp(-Acoeff::eps_I[idx_spec[beta]][step] * C::ev2erg / (C::k_B*T_e)) * X_e;
                }
            }
            if (alpha == beta + 1) { // Lower diagonal: gain from previous ionization
                for (int step = 0; step < Acoeff::n_step; step++) {
                    val += Acoeff::A_I[idx_spec[beta]][step] *
                        std::pow(C::k_B*T_e / C::ev2erg, Acoeff::q_I[idx_spec[beta]][step]) *
                        std::exp(-Acoeff::eps_I[idx_spec[beta]][step] * C::ev2erg / (C::k_B*T_e)) * X_e;
                }
            }
            // Convert to source term per unit mass
            S_r[alpha][beta] = val * rho * rho / (aion[idx_spec[beta]] * C::m_p) * X_stage[beta];
        }
    }

    // Update sources
    for (int alpha = 0; alpha < dimS; alpha++) {
        Real sum = 0.0;
        for (int beta = 0; beta < dimS; beta++) {
            sum += S_r[alpha][beta];
        }
        src(i,j,k,UFS+idx_spec[alpha]) = sum;
    }

    // Energy source
    Real f = state(i,j,k,UFX)/rho;      // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;    // internal energy per unit mass
    Real e_ion = 0.0; // ionization energy per unit mass
    for (int alpha = 0; alpha < dimS; alpha++) {
        for (int beta = alpha; beta < dimS; beta++) {
            e_ion += S_r[alpha][beta] * Acoeff::eps_I[idx_spec[alpha]][0] * C::ev2erg / (aion[idx_spec[beta]] * C::m_p);
        }
    }
    src(i,j,k,UEDEN) = e_ion;
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // Electron and neutral density
    Real n_e = 0.0;
    Real n_n_spe[ion_names.size()] = {0.0, 0.0, 0.0, 0.0};
    Real start_idx[ion_names.size()] = {1, 3, 9, 12};
    for (int mu = 0; mu < dimS; mu++) {
        int k = idx_spec[mu];
        n_e += zion[k] * rho * X_stage[mu] / (aion[k] * C::m_u) * 1.e6;
        if (zion[k] == 0) {
            // find species s corresponding to k
            for (int s = 0; s < ion_names.size(); s++) {
                if (k == start_idx[s] - 1) {
                    n_n_spe[s] = rho * X_stage[mu] / (aion[k] * C::m_u) * 1.e6;
                    break;
                }
            }
        }
    }
    // Get collision frequencies
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature
    amrex::Real v_ei; // electron-ion collision frequency
    amrex::Real v_en[dimS]; // electron-neutral collision frequency for all species
    get_collision_frequencies(n_e, n_n_spe, T_e, T_h, v_ei, v_en, idx_spec);

    // Step (11) : Update fraction of internal energy stored in ions
    amrex::Real Sc = 0.0;
    amrex::Real v_e_E[dimS]; // momentum transfer collision frequency for each species
    for (int mu = 0; mu < dimS; mu++) {
        if (X_stage[mu] < 1e-20) {
            v_e_E[mu] = 0.0; // if the species is not present, set collision frequency to zero
        }
        else{
        v_e_E[mu] = 2 * SI::m_e / SI::m_vec[idx_spec[mu]] * (std::pow(zion[idx_spec[mu]],4) * v_ei + v_en[mu]) ; // momentum transfer collision frequency for each species
        }
    }
    for (int mu = 0; mu < dimS; mu++) {
        Sc += rho * v_e_E[mu] * (1 - (1 + X_stage[mu]) * f); // Compute collision source term for the energy equation
    }
    src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc; // Update source term for energy
}

#endif
