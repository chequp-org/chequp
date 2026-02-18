#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <ionization_table.H>

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
    // Density and electron temperature
    Real rho = state(i,j,k,URHO);
    Real T_e = state(i,j,k,UTEMP);

    // Step (1) : Get current mass fractions
    Real X_stage[NumSpec];
    constexpr Real X_min = 1e-20;
    bool active_mask[NumSpec] = {false};
    int active_spec[NumSpec];
    int n_active = 0;
    for (int mu = 0; mu < NumSpec; mu++) {
        X_stage[mu] = state(i,j,k,UFS+mu) / rho; // update the correct mass fraction with the shifted index
        if (X_stage[mu] > X_min) {
            active_mask[mu] = true;
            // Allow neighbors to be activated
            if (mu > 0)
                active_mask[mu-1] = true;
            if (mu < NumSpec-1)
                active_mask[mu+1] = true;
        }
        if (active_mask[mu]) {
            active_spec[n_active++] = mu;
        }
    }
    // Electron fraction
    Real X_e = 0.0;
    for (int im = 0; im < n_active; im++) {
        int mu = active_spec[im];
        X_e += zion[mu] * X_stage[mu]; // electron fraction
    }
    // Reaction matrix that have the dimension of the species considered
    Real e_ion = 0.0; // ionization energy per unit mass
    // Loop over target state alpha (row)
    for (int ia = 0; ia < n_active; ia++) {
        int alpha = active_spec[ia];
        Real sum = 0.0;
        for (int ib = 0; ib < n_active; ib++) {
            int beta = active_spec[ib];
            Real val = 0.0;
            // --- Electron-impact ionization ---
            if (alpha == beta) { // Diagonal: loss
                for (int step = 0; step < Acoeff::n_step; step++) {
                    val -= Acoeff::A_I[beta][step] *
                        std::pow(C::k_B*T_e / C::ev2erg, Acoeff::q_I[beta][step]) *
                        std::exp(-Acoeff::eps_I[beta][step] * C::ev2erg / (C::k_B*T_e)) * X_e;
                }
            }
            if (alpha == beta + 1) { // Lower diagonal: gain from previous ionization
                for (int step = 0; step < Acoeff::n_step; step++) {
                    val += Acoeff::A_I[beta][step] *
                        std::pow(C::k_B*T_e / C::ev2erg, Acoeff::q_I[beta][step]) *
                        std::exp(-Acoeff::eps_I[beta][step] * C::ev2erg / (C::k_B*T_e)) * X_e;
                }
            }
            sum += val * rho * rho / (aion[beta] * C::m_p) * X_stage[beta];;
            if (beta >= alpha){
                    e_ion += (val * rho * rho / (aion[beta] * C::m_p) * X_stage[beta]) * Acoeff::eps_I[alpha][0] * C::ev2erg / (aion[beta] * C::m_p);
            }
        }
        src(i,j,k,UFS+alpha) = sum; // Update the source term for each species
    }

    // Energy source
    Real f = state(i,j,k,UFX)/rho;      // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;    // internal energy per unit mass
    src(i,j,k,UEDEN) = e_ion; // Energy source due to ionization
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // Electron and neutral density
    Real n_e = 0.0;
    Real n_n_spe[num_atom_species] = {};
    
    // Helper lambda to map species index to atomic species
    auto get_atom_species = [](int mu) -> int {
        if (mu <= 1) return 0;      // Hydrogen
        if (mu <= 7 && mu > 1) return 1;      // Nitrogen
        if (mu <= 10 && mu > 7) return 2;     // Helium
        return 3;                    // Argon
    };
    for (int im = 0; im < n_active; im++) {
        int mu = active_spec[im];
        Real density_factor = rho * X_stage[mu] / (aion[mu] * C::m_u) * 1.e6;
        n_e += zion[mu] * density_factor;
        if (zion[mu] == 0) {
            n_n_spe[get_atom_species(mu)] = density_factor;
        }
    }
    // Get collision frequencies
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature
    amrex::Real v_ei; // electron-ion collision frequency
    amrex::Real v_en[NumSpec]; // electron-neutral collision frequency for all species
    get_collision_frequencies(n_e, n_n_spe, T_e, T_h, v_ei, v_en);

    // Update fraction of internal energy stored in ions
    amrex::Real Sc = 0.0;
    amrex::Real v_e_E[NumSpec] = {}; // momentum transfer collision frequency for each species
    for (int im = 0; im < n_active; im++) {
        int mu = active_spec[im];
        if (X_stage[mu] > 1e-20){
            int s = get_atom_species(mu);
            v_e_E[mu] = 2 * SI::m_e / mass_vec[s] * (std::pow(zion[mu], 4) * v_ei + v_en[s]);
        }
    }
    for (int im = 0; im < n_active; im++) {
        int mu = active_spec[im];
        Sc += rho * v_e_E[mu] * (1 - (1 + X_stage[mu]) * f); // Compute collision source term for the energy equation
    }
    src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc; // Update source term for energy
}

#endif
