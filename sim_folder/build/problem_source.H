#ifndef problem_source_H 
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;

    // Step (1)
    Real rho = state(i,j,k,URHO);
    Real T_e = state(i,j,k,UTEMP);

    // Step (2)
    Real X_stage[NumSpec];
    for (int mu = 0; mu < NumSpec; mu++) {
        X_stage[mu] = state(i,j,k,UFS+mu)/rho;
    }

    // Step (3)
    // Ionization
    Real A_I[NumSpec];
    Real q_I[NumSpec];
    Real eps_I[NumSpec];

    // Three-body recombination
    Real A_T[NumSpec];
    Real q_T[NumSpec];
    Real eps_T[NumSpec];         // in eV

    // Degeneracy g (statistical weight)
    Real g[NumSpec];

    // Hardcoded Arhenius parameters for Hydrogen (how and where should these be stored?)
    A_I[0] = 1.27e-8; // convert to m^3/s
    q_I[0] = 0.5;
    eps_I[0] = 13.6;        // in eV

    A_T[1] = 2.0e-27;       // in cm^6/s

    g[0] = 2.0; // H^0
    g[1] = 1.0; // H^+

    Real lambda = std::sqrt(2 * M_PI * C::hbar * C::hbar / (C::m_p * C::k_B * T_e));
    Real X_e;

    for (int mu = 0; mu < NumSpec; mu++) {
        X_e += zion[mu] * X_stage[mu];
    }

    // Build reaction matrix S_r
    Real S_r[NumSpec][NumSpec];

    for (int alpha = 0; alpha < NumSpec; alpha++) {
        for (int beta = 0; beta < NumSpec; beta++) {
            Real val = 0.0;

            // Loss from alpha
            if (alpha == beta) {
                val -= A_I[alpha] *
                        std::pow(C::k_B*T_e/C::ev2erg, q_I[alpha]) *
                        std::exp(-eps_I[alpha]*C::ev2erg/(C::k_B*T_e)) * X_e;

                if (alpha>0) {
                    val -= A_T[alpha-1] *
                           std::pow(C::k_B*T_e/C::ev2erg, q_T[alpha-1]) *
                           std::exp(-eps_T[alpha-1]*C::ev2erg/(C::k_B*T_e)) *
                           rho * std::pow(lambda,3) / g[alpha-1] * X_e*X_e;
                }
            }

            // Gain from recombination
            if (beta == alpha+1 && alpha<NumSpec-1) {
                val += A_T[alpha] *
                        std::pow(C::k_B*T_e/C::ev2erg, q_T[alpha]) *
                        std::exp(-eps_T[alpha]*C::ev2erg/(C::k_B*T_e)) *
                        rho * std::pow(lambda,3) / g[alpha] * X_e*X_e;
            }

            // Gain from ionization
            if (beta == alpha-1 && alpha>0) {
                val += A_I[alpha-1] *
                        std::pow(C::k_B*T_e/C::ev2erg, q_I[alpha-1]) *
                        std::exp(-eps_I[alpha-1]*C::ev2erg/(C::k_B*T_e)) * X_e;
            }

            S_r[alpha][beta] = val * rho * rho / (aion[alpha] * C::m_u); // Include rho^2/m_p
        }
    }

    // Step (5)
    // Update the X vector using S_r
    for (int alpha=0; alpha<NumSpec; alpha++) {
        Real sum = 0.0;
        for (int beta=0; beta<NumSpec; beta++) {
            sum += S_r[alpha][beta] * X_stage[beta];
        }
        src(i,j,k,UFS+alpha) = sum;
    }

    // Step (6)
    Real f = state(i,j,k,UFX)/rho;  // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;  // internal energy per unit mass
    Real e_ion = 0.0;
    for (int alpha = 0; alpha < NumSpec; alpha++) {
        Real sum = 0.0;
        for (int beta = 0; beta < NumSpec; beta++) {
            sum += S_r[alpha][beta] / (rho * rho);   // (m^{-1})^\beta, here m_p
        }
        e_ion -= eps_I[alpha] * C::ev2erg * sum; // convert eV → erg
    }
    // Assign to source array
    src(i,j,k,UEDEN) = e_ion;
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // Step (7)
    // Electron and neutral density
    Real n_e = 0.0;
    Real n_n = 0.0;
    for (int mu = 0; mu < NumSpec; mu++) {
        n_e +=zion[mu] * rho * X_stage[mu] / (aion[mu] * C::m_u * 1.e6);
        if (zion[mu] == 0) {
            n_n += rho * X_stage[mu] / (aion[mu] * C::m_u * 1.e6);
        }
    }

    // Step (8)
    // Get collision frequencies
    amrex::Real v_ei, v_en;
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)
    get_collision_frequencies(n_e, n_n, T_e, T_h, v_ei, v_en);

    // Step (11)
    // Calculate the collision frequency for energy transfer between electrons and heavies
    const amrex::Real v_e_E = 2*SI::m_e/SI::m_H * (v_ei + v_en);
    
    // Update fraction of internal energy stored in ions
    Real Sc = 0.0;
    Real v_e[NumSpec] = {v_e_E, v_e_E, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int mu = 0; mu < NumSpec; mu++) {
        Sc += rho * v_e[mu] * ( 1 - (1+X_stage[mu])*f);
    }

    // Step (12)
    src(i,j,k,UFX) = - f/e * src(i,j,k,UEDEN) + Sc;
}

#endif
