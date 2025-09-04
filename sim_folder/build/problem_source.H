#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;


    Real rho = state(i,j,k,URHO);
    Real T = state(i,j,k,UTEMP);

    // Guard against unphysically low temperatures
    if (T < 100.0) { return; }

    // =========================================================================
    // Nitrogen ionization energies and rate coefficients benchmarked with the hydrogen value
    // Fitted to ADAS ADF07 data (szd96#n_n*.dat files)
    // Rate = a_ioniz * sqrt(Te_eV) * exp(-E_ioniz_eV/Te_eV)
    // Units: cm³/s
    // =========================================================================
    const Real e_ioniz_N0_to_N1_eV = 14.53;
    const Real a_ioniz_N0_to_N1 = 9.985e-09;
    
    const Real e_ioniz_N1_to_N2_eV = 29.60;
    const Real a_ioniz_N1_to_N2 = 3.012e-09;
    
    const Real e_ioniz_N2_to_N3_eV = 47.45;
    const Real a_ioniz_N2_to_N3 = 9.451e-10;
    
    const Real e_ioniz_N3_to_N4_eV = 77.47;
    const Real a_ioniz_N3_to_N4 = 2.596e-10;
    
    const Real e_ioniz_N4_to_N5_eV = 97.89;
    const Real a_ioniz_N4_to_N5 = 9.606e-1;

    // =========================================================================
    // Get mass fractions for ALL species defined in species.net
    // The order UFS, UFS+1, etc. MUST match species.net
    // =========================================================================
    Real X_Hn  = state(i,j,k,UFS)   / rho; // H_neutral
    Real X_Hp  = state(i,j,k,UFS+1) / rho; // H_ion
    Real X_Nn  = state(i,j,k,UFS+2) / rho; // N_neutral
    Real X_N1  = state(i,j,k,UFS+3) / rho; // N+
    Real X_N2  = state(i,j,k,UFS+4) / rho; // N2+
    Real X_N3  = state(i,j,k,UFS+5) / rho; // N3+
    Real X_N4  = state(i,j,k,UFS+6) / rho; // N4+

    // Electron mass fraction (primarily from ionized hydrogen and nitrogen)
    // For better accuracy, you could calculate this based on all ion contributions
    Real X_e = X_Hp; // Simplified - add nitrogen contributions if needed

    // Convert temperature to eV for rate calculations
    Real T_eV = C::k_B * T / C::ev2erg;

    // =========================================================================
    // Initialize all source terms to zero
    // =========================================================================
    for (int s=0; s < NumSpec; ++s) {
        src(i,j,k,UFS+s) = 0.0;
    }
    src(i,j,k,UEDEN) = 0.0;
    src(i,j,k,UEINT) = 0.0;

    // =========================================================================
    // --- Hydrogen Ionization (if still needed) ---
    // =========================================================================
    if (problem::a_direct_ioniz > 0.0) {
        Real R_H_to_Hp = rho * rho / C::m_p * (
            problem::a_direct_ioniz *
            std::pow(T_eV, problem::q_direct_ioniz) *
            std::exp(-problem::e_direct_ioniz_eV / T_eV) *
            X_Hn * X_e
        );

        src(i,j,k,UFS)   -= R_H_to_Hp;
        src(i,j,k,UFS+1) += R_H_to_Hp;
        src(i,j,k,UEDEN) -= R_H_to_Hp * (problem::e_loss_ioniz_eV * C::ev2erg / C::m_p);
    }

    // =========================================================================
    // --- Nitrogen Ionization Reactions ---
    // =========================================================================
    
    // N0 -> N+ (First ionization)
    Real R_N0_to_N1 = rho * rho / C::m_n * (
        a_ioniz_N0_to_N1 * std::sqrt(T_eV) * 
        std::exp(-e_ioniz_N0_to_N1_eV / T_eV) * X_Nn * X_e
    );
    
    src(i,j,k,UFS+2) -= R_N0_to_N1;  // Remove N0
    src(i,j,k,UFS+3) += R_N0_to_N1;  // Add N+
    src(i,j,k,UEDEN) -= R_N0_to_N1 * (e_ioniz_N0_to_N1_eV * C::ev2erg / C::m_n);

    // N+ -> N2+ (Second ionization)
    Real R_N1_to_N2 = rho * rho / C::m_n * (
        a_ioniz_N1_to_N2 * std::sqrt(T_eV) * 
        std::exp(-e_ioniz_N1_to_N2_eV / T_eV) * X_N1 * X_e
    );
    
    src(i,j,k,UFS+3) -= R_N1_to_N2;  // Remove N+
    src(i,j,k,UFS+4) += R_N1_to_N2;  // Add N2+
    src(i,j,k,UEDEN) -= R_N1_to_N2 * (e_ioniz_N1_to_N2_eV * C::ev2erg / C::m_n);

    // N2+ -> N3+ (Third ionization)
    Real R_N2_to_N3 = rho * rho / C::m_n * (
        a_ioniz_N2_to_N3 * std::sqrt(T_eV) * 
        std::exp(-e_ioniz_N2_to_N3_eV / T_eV) * X_N2 * X_e
    );
    
    src(i,j,k,UFS+4) -= R_N2_to_N3;  // Remove N2+
    src(i,j,k,UFS+5) += R_N2_to_N3;  // Add N3+
    src(i,j,k,UEDEN) -= R_N2_to_N3 * (e_ioniz_N2_to_N3_eV * C::ev2erg / C::m_n);

    // N3+ -> N4+ (Fourth ionization)
    Real R_N3_to_N4 = rho * rho / C::m_n * (
        a_ioniz_N3_to_N4 * std::sqrt(T_eV) * 
        std::exp(-e_ioniz_N3_to_N4_eV / T_eV) * X_N3 * X_e
    );
    
    src(i,j,k,UFS+5) -= R_N3_to_N4;  // Remove N3+
    src(i,j,k,UFS+6) += R_N3_to_N4;  // Add N4+
    src(i,j,k,UEDEN) -= R_N3_to_N4 * (e_ioniz_N3_to_N4_eV * C::ev2erg / C::m_n);

    // N4+ -> N5+ (Fifth ionization) - Optional, can be commented out if not needed
    // this reaction is probably negligible
    /*
    Real R_N4_to_N5 = rho * rho / C::m_n * (
        a_ioniz_N4_to_N5 * std::sqrt(T_eV) * 
        std::exp(-e_ioniz_N4_to_N5_eV / T_eV) * X_N4 * X_e
    );
    
    src(i,j,k,UFS+6) -= R_N4_to_N5;  // Remove N4+
    src(i,j,k,UFS+7) += R_N4_to_N5;  // Add N5+
    src(i,j,k,UEDEN) -= R_N4_to_N5 * (e_ioniz_N4_to_N5_eV * C::ev2erg / C::m_n);
    */

    // =========================================================================
    // --- Final update for internal energy ---
    // =========================================================================
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);
    
    // =========================================================================
    // --- Optional: Add diagnostics for debugging ---
    // =========================================================================
    #ifdef DEBUG_IONIZATION
    if (i == 0 && j == 0 && k == 0) {
        printf("T = %.2f K (%.2f eV), rho = %.3e g/cm³\n", T, T_eV, rho);
        printf("Mass fractions: X_Nn=%.3e, X_N1=%.3e, X_N2=%.3e, X_N3=%.3e, X_N4=%.3e\n",
               X_Nn, X_N1, X_N2, X_N3, X_N4);
        printf("Ionization rates (cm³/s): R01=%.3e, R12=%.3e, R23=%.3e, R34=%.3e\n",
               R_N0_to_N1*C::m_n/rho/rho, R_N1_to_N2*C::m_n/rho/rho, 
               R_N2_to_N3*C::m_n/rho/rho, R_N3_to_N4*C::m_n/rho/rho);
    }
    #endif
}

#endif
