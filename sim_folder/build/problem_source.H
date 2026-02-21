#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <ionization_table.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;

    auto coord = geomdata.Coord();
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();

    // Density and temperature
    Real rho = state(i,j,k,URHO);

    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state(i,j,k,UTEMP);

    Real X_stage[NumSpec];
    constexpr Real X_min = 1e-20;
    bool active_mask[NumSpec] = {false}; // Mask to track which species are active (above threshold) and their neighbors
    int active_spec[NumSpec]; // Array to store indices of active species
    int n_active = 0; // Counter for number of active species
    for (int mu = 0; mu < NumSpec; mu++) {
        X_stage[mu] = state(i,j,k,UFS+mu) / rho; // update the correct mass fraction with the shifted index
        if (X_stage[mu] < X_min) {
            X_stage[mu] = 0.0; // Set to zero if below threshold
        }
        if (X_stage[mu] > X_min) {
            active_mask[mu] = true;
            // Allow neighbors to be activated
            if (mu > 0)
                active_mask[mu-1] = true;
        }
        if (active_mask[mu]) {
            active_spec[n_active++] = mu; // Store index of active species
        }
    }

    // Electron and heavy temperatures
    Real f = state(i,j,k,UFX)/rho;  // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;  // internal energy per unit mass
    
    Real X_e = 0.0;
    for (int i = 0; i < n_active; i++) {
        int mu = active_spec[i]; // Get the actual species index from the active_spec array
        X_e += zion[mu] * X_stage[mu]; // electron fraction
    }
    Real e_ion = 0.0;
    for (int alpha = 0; alpha <NumSpec; alpha++) {
        Real sum = 0.0;
        for (int beta = 0; beta < NumSpec; beta++) {
            Real val = 0.0;
            // Electron-impact ionization
            Real coll_I = 0.0;
            for (int step = 0; step < Acoeff::n_step; step++) {
                    coll_I += Acoeff::A_I[beta][step] *
                        std::pow(C::k_B*T_e / C::ev2erg, Acoeff::q_I[beta][step]) *
                        std::exp(-Acoeff::eps_I[beta][step] * C::ev2erg / (C::k_B*T_e)) * X_stage[beta] * X_e;
                }
            if (alpha == beta) { 
                val = - coll_I;
                e_ion += val * (rho * rho / C::m_p) * (Acoeff::eps_I[beta][0]*C::ev2erg / C::m_p);
            }
            else if (alpha == beta + 1) { 
                val = coll_I;
            }
            sum += val * (rho * rho / C::m_p);
        }
    src(i,j,k,UFS+ alpha ) = sum;
    }
    // Update state
    src(i,j,k,UEDEN) = e_ion;
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // In order to calculate the energy transfer between electrons and heavy particles,
    // compute the collision frequency between electrons and ions (using SI units)
    amrex::Real n_e = 0.0; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    amrex::Real n_n_species[num_atom_species] = {}; // number density of each neutral species in m^-3
    for (int i = 0; i < n_active; i++) {
        int mu = active_spec[i];
        n_e += zion[mu] * X_stage[mu] * rho / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
        if (zion[mu] == 0) {
            int idx_spec = get_atom_index(mu);
            n_n_species[idx_spec] += X_stage[mu] * rho / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
        }
    }
    amrex::Real v_ei;
    amrex::Real v_en[NumSpec]; // electron-neutral collision frequency for all species
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)
    get_collision_frequencies(n_e, n_n_species, T_e, T_h, v_ei, v_en);
    // Update fraction of internal energy stored in ions
    amrex::Real Sc = 0.0;
    amrex::Real v_e_E[NumSpec] = {}; // momentum transfer collision frequency for each species 
    for (int i = 0; i < n_active; i++) {
        int mu = active_spec[i];
        v_e_E[mu] = 2 * SI::m_e / mass_vec[get_atom_index(mu)] * (std::pow(zion[mu], 4) * v_ei + v_en[mu]); 
    }
    // Update fraction of internal energy stored in ions
    for (int im = 0; im < n_active; im++) {
            int mu = active_spec[im]; // Get the actual species index from the active_spec array
            // Compute collision source term for the energy equation, using the momentum transfer collision frequency v_e_E[mu]
            Sc += rho * v_e_E[mu] * (1 - (1 + X_stage[mu]) * f);
        }
    src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc; // Update source term for energy
}

#endif