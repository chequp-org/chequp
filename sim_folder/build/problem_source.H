#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;
    // Nitrogen ionization energies and rate coefficients
    // Fitted to ADAS ADF07 data
    const Real e_ioniz_N0_to_N1_eV = 14.53;
    const Real a_ioniz_N0_to_N1 = 9.985e-09;
    
    const Real e_ioniz_N1_to_N2_eV = 29.60;
    const Real a_ioniz_N1_to_N2 = 3.012e-09;
    
    const Real e_ioniz_N2_to_N3_eV = 47.45;
    const Real a_ioniz_N2_to_N3 = 9.451e-10;
    
    const Real e_ioniz_N3_to_N4_eV = 77.47;
    const Real a_ioniz_N3_to_N4 = 2.596e-10;
    
    const Real e_ioniz_N4_to_N5_eV = 97.89;
    const Real a_ioniz_N4_to_N5 = 9.606e-11;
    auto coord = geomdata.Coord();
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();

    // Density and temperature
    Real rho = state(i,j,k,URHO);

    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state(i,j,k,UTEMP);

    // Abundance of neutral and ionized H
    Real X_Hn = state(i,j,k,UFS) / rho;
    Real X_Hp = state(i,j,k,UFS+1) / rho;
    Real X_Nn = state(i,j,k,UFS+2) / rho; // N_neutral
    Real X_N1 = state(i,j,k,UFS+3) / rho; // N+
    Real X_N2 = state(i,j,k,UFS+4) / rho; // N2+
    Real X_N3 = state(i,j,k,UFS+5) / rho; // N3+
    Real X_N4 = state(i,j,k,UFS+6) / rho; // N4+
    // Electron and heavy temperatures
    Real f = state(i,j,k,UFX)/rho;  // fraction of energy in heavy ions
    Real e = state(i,j,k,UEINT)/rho;  // internal energy per unit mass

    // Rate of ionization
    Real R_Hp = rho * rho / C::m_p * (
        problem::a_direct_ioniz *
        std::pow( C::k_B*T_e/C::ev2erg, problem::q_direct_ioniz ) *
        std::exp(-problem::e_direct_ioniz_eV*C::ev2erg/(C::k_B*T_e)) * X_Hn * X_Hp
        +
        problem::a_twostep_ioniz *
        std::pow( C::k_B*T_e/C::ev2erg, problem::q_twostep_ioniz ) *
        std::exp(-problem::e_twostep_ioniz_eV*C::ev2erg/(C::k_B*T_e)) * X_Hn * X_Hp
    );

    // Update state
    src(i,j,k,UFS) = -R_Hp;
    src(i,j,k,UFS+1) = R_Hp;
    src(i,j,k,UEDEN) = -R_Hp * (problem::e_loss_ioniz_eV*C::ev2erg / C::m_p);
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // In order to calculate the energy transfer between electrons and heavy particles,
    // compute the collision frequency between electrons and ions (using SI units)
    const amrex::Real n_e = rho * (X_Hp + X_N1 + X_N2 + X_N3 + X_N4) / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    const amrex::Real n_n = rho * (X_Hn + X_Nn) / C::m_u * 1.e6; // Convert from cm^-3 to m^-3, since we will use an SI unit formula
    // Get collision frequencies
    amrex::Real v_ei, v_en;
    const amrex::Real T_h = T_e; // TODO: Extract correct ion temperature (this is only used for the Coulomb logarithm)
    get_collision_frequencies(n_e, n_n, T_e, T_h, v_ei, v_en);

    // Calculate the collision frequency for energy transfer between electrons and heavies
    const amrex::Real v_e_E = 2*SI::m_e/SI::m_H * (v_ei + v_en);

        // Initialize nitrogen species source terms
    src(i,j,k,UFS+2) = 0.0; // N_neutral
    src(i,j,k,UFS+3) = 0.0; // N+
    src(i,j,k,UFS+4) = 0.0; // N2+
    src(i,j,k,UFS+5) = 0.0; // N3+
    src(i,j,k,UFS+6) = 0.0; // N4+
    
    // Nitrogen ionization reactions
    // N0 -> N+
    Real R_N0_to_N1 = rho * rho / (14 * C::m_u) * (
        a_ioniz_N0_to_N1 * std::sqrt(C::k_B*T_e/C::ev2erg) * 
        std::exp(-e_ioniz_N0_to_N1_eV*C::ev2erg/(C::k_B*T_e)) * X_Nn * (X_Hp + X_N1 + 2.0*X_N2 + 3.0*X_N3 + 4.0*X_N4)
    );
    src(i,j,k,UFS+2) -= R_N0_to_N1;
    src(i,j,k,UFS+3) += R_N0_to_N1;
    src(i,j,k,UEDEN) -= R_N0_to_N1 * (e_ioniz_N0_to_N1_eV*C::ev2erg / (14 * C::m_u));
    src(i,j,k,UEINT) -= R_N0_to_N1 * (e_ioniz_N0_to_N1_eV*C::ev2erg / (14 * C::m_u));
    
    // N+ -> N2+
    Real R_N1_to_N2 = rho * rho / (14 * C::m_u) * (
        a_ioniz_N1_to_N2 * std::sqrt(C::k_B*T_e/C::ev2erg) * 
        std::exp(-e_ioniz_N1_to_N2_eV*C::ev2erg/(C::k_B*T_e)) * X_N1 * (X_Hp + X_N1 + 2.0*X_N2 + 3.0*X_N3 + 4.0*X_N4)
    );
    src(i,j,k,UFS+3) -= R_N1_to_N2;
    src(i,j,k,UFS+4) += R_N1_to_N2;
    src(i,j,k,UEDEN) -= R_N1_to_N2 * (e_ioniz_N1_to_N2_eV*C::ev2erg / (14 * C::m_u));
    src(i,j,k,UEINT) -= R_N1_to_N2 * (e_ioniz_N1_to_N2_eV*C::ev2erg / (14 * C::m_u));
    
    // N2+ -> N3+
    Real R_N2_to_N3 = rho * rho / (14 * C::m_u) * (
        a_ioniz_N2_to_N3 * std::sqrt(C::k_B*T_e/C::ev2erg) * 
        std::exp(-e_ioniz_N2_to_N3_eV*C::ev2erg/(C::k_B*T_e)) * X_N2 * (X_Hp + X_N1 + 2.0*X_N2 + 3.0*X_N3 + 4.0*X_N4)
    );
    src(i,j,k,UFS+4) -= R_N2_to_N3;
    src(i,j,k,UFS+5) += R_N2_to_N3;
    src(i,j,k,UEDEN) -= R_N2_to_N3 * (e_ioniz_N2_to_N3_eV*C::ev2erg / (14 * C::m_u));
    src(i,j,k,UEINT) -= R_N2_to_N3 * (e_ioniz_N2_to_N3_eV*C::ev2erg / (14 * C::m_u));
    
    // N3+ -> N4+
    Real R_N3_to_N4 = rho * rho / (14 * C::m_u) * (
        a_ioniz_N3_to_N4 * std::sqrt(C::k_B*T_e/C::ev2erg) * 
        std::exp(-e_ioniz_N3_to_N4_eV*C::ev2erg/(C::k_B*T_e)) * X_N3 * (X_Hp + X_N1 + 2.0*X_N2 + 3.0*X_N3 + 4.0*X_N4)
    );
    src(i,j,k,UFS+5) -= R_N3_to_N4;
    src(i,j,k,UFS+6) += R_N3_to_N4;
    src(i,j,k,UEDEN) -= R_N3_to_N4 * (e_ioniz_N3_to_N4_eV*C::ev2erg / (14 * C::m_u));
    src(i,j,k,UEINT) -= R_N3_to_N4 * (e_ioniz_N3_to_N4_eV*C::ev2erg / (14 * C::m_u));
    // Update fraction of internal energy stored in ions
    src(i,j,k,UFX) = - f/e * src(i,j,k,UEDEN) + rho * v_e_E * ( 1 - (1+X_Hp)*f );

}

#endif
