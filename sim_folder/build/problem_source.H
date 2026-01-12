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

    // Ionization steps
    const int n_step = 2;

    // Step coefficients
    Real A_I[NumSpec][n_step];
    Real q_I[NumSpec][n_step];
    Real eps_I[NumSpec][n_step];
    // No recombination
    Real A_T[NumSpec][n_step];
    Real q_T[NumSpec][n_step];
    Real eps_T[NumSpec][n_step];

    // Degeneracy
    Real g[NumSpec];
    // Initialize to zero
    for (int mu = 0; mu < NumSpec; mu++) {
        for (int step = 0; step < n_step; step++) {
            A_I[mu][step] = 0.0;
            q_I[mu][step] = 0.0;
            eps_I[mu][step] = 0.0;
            A_T[mu][step] = 0.0;
            q_T[mu][step] = 0.0;
            eps_T[mu][step] = 0.0;
        }
        g[mu] = 0.0;
    }

    A_I[0][0] = problem::a_direct_ioniz;
    q_I[0][0] = problem::q_direct_ioniz;
    eps_I[0][0] = problem::e_direct_ioniz_eV;
    A_I[0][1] = problem::a_twostep_ioniz;
    q_I[0][1] = problem::q_twostep_ioniz;
    eps_I[0][1] = problem::e_twostep_ioniz_eV;

    Real X_e = 0.0;
    for (int mu = 0; mu < NumSpec; mu++) {
        X_e += zion[mu] * X_stage[mu];
    }

    // Reaction matrix
    Real S_r[NumSpec][NumSpec];
    for (int alpha = 0; alpha < NumSpec; alpha++) {
        for (int beta = 0; beta < NumSpec; beta++) {

            Real val = 0.0;

            // Loss Ionization
            if (alpha == beta) {
                for (int step = 0; step < n_step; step++) {
                    val -= A_I[beta][step] *
                        std::pow(C::k_B*T_e / C::ev2erg, q_I[beta][step]) *
                        std::exp(-eps_I[beta][step] * C::ev2erg / (C::k_B*T_e)) *
                        X_stage[beta] * X_e;
                }
            }

            // Gain Ionization
            if (alpha == beta + 1) {
                for (int step = 0; step < n_step; step++) {
                    val += A_I[beta][step] *
                        std::pow(C::k_B*T_e / C::ev2erg, q_I[beta][step]) *
                        std::exp(-eps_I[beta][step] * C::ev2erg / (C::k_B*T_e)) * X_stage[beta] * X_e;
                }
            }

            S_r[alpha][beta] = val * rho * rho / (aion[alpha] * C::m_p);
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
            e_ion += S_r[alpha][beta] * eps_I[alpha][0] * C::ev2erg / (aion[alpha] * C::m_p);
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

    v_e_E[1] = 2 * SI::m_e / SI::m_H * (std::pow(zion[1],4) * v_ei + v_en);

    for (int mu = 0; mu < NumSpec; mu++) {
        Sc += rho * v_e_E[mu] * (1 - (1 + X_stage[mu]) * f);
    }

    // Step (12)
    src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc;
}

#endif
