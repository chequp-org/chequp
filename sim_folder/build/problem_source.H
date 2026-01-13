#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>

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

    // Species fractions (H0 and H+)
    Real X_stage[NumSpec];
    for (int mu = 0; mu < NumSpec; mu++) {
        X_stage[mu] = state(i,j,k,UFS+mu) / rho;
    }

    // Electron fraction
    Real X_e = 0.0;
    for (int mu = 0; mu < NumSpec; mu++) {
        X_e += zion[mu] * X_stage[mu];
    }

    // Reaction matrix
    Real S_r[NumSpec][NumSpec];
    // Loop over target state alpha (row)
    for (int alpha = 0; alpha < NumSpec; alpha++) {
        for (int beta = 0; beta < NumSpec; beta++) {
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
            // Convert to source term per unit mass
            S_r[alpha][beta] = val * rho * rho / (aion[alpha] * C::m_p) * X_stage[beta];
        }
    }

    // Update sources
    for (int alpha = 0; alpha < NumSpec; alpha++) {
        Real sum = 0.0;
        for (int beta = 0; beta < NumSpec; beta++) {
            sum += S_r[alpha][beta];
        }
        src(i,j,k,UFS+alpha) = sum;
    }

    // Step (6) : Energy source
    Real f = state(i,j,k,UFX)/rho;      // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;    // internal energy per unit mass
    Real e_ion = 0.0;
    for (int alpha = 0; alpha < NumSpec; alpha++) {
        for (int beta = alpha; beta < NumSpec; beta++) {
            e_ion += S_r[alpha][beta] * Acoeff::eps_I[alpha][0] * C::ev2erg / (aion[alpha] * C::m_p);
        }
    }
    src(i,j,k,UEDEN) = e_ion;
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // Step (7) : Electron and neutral density
    Real n_e = 0.0;
    Real n_n = 0.0;
    for (int mu = 0; mu < NumSpec; mu++) {
        n_e += zion[mu] * rho * X_stage[mu] / (aion[mu] * C::m_u) * 1.e6;
        if (zion[mu] == 0) {
            n_n += rho * X_stage[mu] / (aion[mu] * C::m_u) * 1.e6;
        }
    }

    // Step (8) : Get collision frequencies
    amrex::Real v_ei, v_en;
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature
    get_collision_frequencies(n_e, n_n, T_e, T_h, v_ei, v_en);

    // Step (11) : Update fraction of internal energy stored in ions
    Real Sc = 0.0;
    Real v_e_E[NumSpec];
    for (int mu = 0; mu < NumSpec; mu++) {
        v_e_E[mu] = 0.0;
    }
    // Hydrogen 
    v_e_E[1] = 2 * SI::m_e / SI::m_H * (std::pow(zion[1],4) * v_ei + v_en);

    for (int mu = 0; mu < NumSpec; mu++) {
        Sc += rho * v_e_E[mu] * (1 - (1 + X_stage[mu]) * f);
    }

    // Step (12)
    src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc;
}

#endif
