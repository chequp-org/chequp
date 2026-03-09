#ifndef COLLISIONAL_IONIZATION_H
#define COLLISIONAL_IONIZATION_H

#include <prob_parameters.H>
#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include <map>
#include <ionization_table.H>

// Reduced database for H, N, He, Ar only
// Atomic numbers: H=1, N=7, He=2, Ar=18

const int zatom[num_atom_species] = {1, 7, 2, 18}; // Atomic numbers for H, N, He, Ar

const double K_to_eV = 8.617333262145e-5; // Boltzmann constant in eV/K

// Ionization energies - reduced to H, N, He, Ar
// Total entries: 1 + 7 + 3 + 18 = 29 entries
AMREX_GPU_HOST_DEVICE inline amrex::Real ionizationEnergy[29] = {
    /* Z=1 (H) */ 13.598434005136,
    /* Z=7 (N) */ 14.53413, 29.60125, 47.4453, 77.4735, 97.89013, 552.06731, 667.04609,
    /* Z=2 (He) */ 24.587387936, 54.41776311,
    /* Z=18 (Ar) */ 15.7596112, 27.62967, 40.735, 59.58, 74.84, 91.290, 124.41, 143.457, 422.60,
    479.76, 540.4, 619.0, 685.47, 755.13, 855.47, 918.374, 4120.6655, 4426.2227
};

// Binding energies - reduced to H, N, He, Ar
AMREX_GPU_HOST_DEVICE inline amrex::Real bindingEnergy[29] = {
    /* Z=1 (H) */ 13.5981,
    /* Z=7 (N) */ 403.78, 403.78, 20.33, 20.33, 14.534, 14.534, 14.534,
    /* Z=2 (He) */ 24.588, 24.588,
    /* Z=18 (Ar) */ 3206.2, 3206.2, 324.2, 324.2, 247.74, 247.74, 247.74, 247.74, 247.74, 247.74,
    29.24, 29.24, 15.76, 15.76, 15.76, 15.76, 15.76, 15.76
};

// Average Gaunt factor fit coefficients for excitation ionization (H, N, He, Ar)
AMREX_GPU_HOST_DEVICE inline amrex::Real gaunt_fit[num_atom_species][7] = {
    {-7.61094672e-06,  1.42014919e-04, -5.26821513e-04, -2.27233338e-03, 2.38468040e-03,  5.36211118e-02,  2.56972926e-01},
    { 1.28897560e-05, -6.31681622e-05,  3.94454841e-04, -2.02240986e-03, -1.12428478e-02,  4.14377747e-02,  3.31875681e-01},
    {-4.17177174e-07,  5.84617203e-05, -4.65772440e-04, -1.06125512e-03, 4.16013477e-03,  5.01909786e-02,  2.23090873e-01},
    { 1.12925396e-05,  7.83928108e-05, -8.07569320e-04, -1.45478529e-03,4.14430633e-03,  5.21310441e-02,  2.49330993e-01}};

AMREX_GPU_HOST_DEVICE inline amrex::Real oscillator_strength[num_atom_species] = {0.416, 0.276, 0.26, 0.12}; // Oscillator strengths for H, N, He, Ar
AMREX_GPU_HOST_DEVICE inline amrex::Real ex_energy_atom[num_atom_species] = {10.6, 19.8, 11.6, 10.3}; // Excitation energies for H, N, He, Ar (eV)

// Index mapping for reduced arrays
// Returns the starting index in the energy arrays for a given Z
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int get_energy_index(int Z) {
    if (Z == 1) return 0;      // H: index 0
    if (Z == 7) return 1;      // N: index 1
    if (Z == 2) return 9;      // He: index 9
    if (Z == 18) return 11;    // Ar: index 11
    return 0;
}

namespace coll_ion
{
    using namespace amrex;

    // Temperature grid
    static constexpr int NT = 200;
    static constexpr Real T_min = 1e-3;      // eV
    static constexpr Real T_max = 3.0e2;    // eV

    // Energy grid for cross-section integration
    static constexpr int npoints = 200;
    static constexpr Real normalization = 2.81794e-15 / (2.*299792458.); // r_e / 2c

    // Table storage
    inline AMREX_GPU_MANAGED
    Real rate_table[NumSpec][NT];

    inline AMREX_GPU_MANAGED
    bool initialized = false;

    // Maxwellian weight
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real maxwell_weight(Real E, Real Te)
    {
        return std::sqrt(E) * std::exp(-E/Te);
    }

    // Cross section using RBEB model from Kim et al. (2000)
    /**
     * @brief Compute the electron impact ionization cross section.
     *
     * @param atomic_number The atomic number of the target element.
     * @param zstar The current ionization state (number of electrons removed).
     * @param ep The energy of the incident electron in eV.
     * @return The ionization cross section in atomic units.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real sigma_E(int atomic_number, int zstar, Real ep)
    {
        Real sigma = 0.0;
        int n = 1;
        int base_idx = get_energy_index(atomic_number);

        for (int k = 0; k < atomic_number - zstar; k++) {
            int idx_k = base_idx + k;
            int idx_k_p = base_idx + (k + 1);
            Real me_c2 = 510998;

            Real bp = (bindingEnergy[idx_k] / me_c2);
            if (k < atomic_number - zstar - 1 &&
                bp == (bindingEnergy[idx_k_p] / me_c2)) {
                n++;
                continue;
            }
            ep /= me_c2;  // Convert to mec^2 units
            Real e = ep / bp;
            if (e > 1.0) {
                Real betae2 = 1.0 - 1.0/((1.0+ep)*(1.0+ep));
                Real betab2 = 1.0 - 1.0/((1.0+bp)*(1.0+bp));
                Real betau2 = betab2;

                Real s0 = normalization * n /
                          (bp*(betae2 + betab2 + betau2));

                Real ep2 = 1.0/(1.0 + 0.5*ep);
                ep2 *= ep2;

                Real a1c = (1.0+2.0*ep)/(1.0+e)*ep2;
                Real a2c = (e-1.0)*bp*bp*0.5*ep2;
                Real a3 = std::log(betae2/(1.0-betae2)) - betae2 - std::log(2.0*bp);

                Real sk = s0 * (0.5*a3*(1.0-1.0/(e*e)) + 1.0-1.0/e + a2c - a1c*std::log(e));

                sigma += sk;
            }
            n = 1;
        }
        return sigma;
    };

    // Maxwellian rate for one temperature
    /**
     * @brief Computes the collisional ionization rate via Maxwell-averaged cross section.
     *
     * @param atomic_number Atomic number Z
     * @param zstar Current ionization state (number of electrons removed)
     * @param Te Electron temperature in eV
     * @return Rate in appropriate units (cross section × velocity averaged over Maxwell distribution)
     */
    inline AMREX_GPU_HOST_DEVICE
    Real compute_rate(int atomic_number, int zstar, Real Te)
    {
        Real sum = 0.0;
        Real norm = 0.0;
        Real E_min = 0;
        Real E_max = 1e3;
        Real dE = (E_max - E_min) / npoints;

        for (int i = 0; i < npoints; i++) {
            Real E = E_min + i * dE;
            Real w = maxwell_weight(E, Te);
            Real sig = sigma_E(atomic_number, zstar, E);
            Real v = std::sqrt(2.0 * E * C::ev2erg / C::m_e);

            sum  += sig * v * w;
            norm += w;
        }

        if (norm > 0.0) sum /= norm;

        return sum;
    };

    // Initialization (host only)
    /// @brief Initializes the collisional ionization rate table.
    /// @details Computes and caches ionization rates for all species across a temperature grid.
    /// Only executes once; subsequent calls return immediately.
    inline void initialize()
    {
        if (initialized) return;

        for (int ispe = 0; ispe < NumSpec; ispe++) {
            int Z = zatom[get_atom_index(ispe)];
            int zstar = zion[ispe];

            for (int it = 0; it < NT; it++) {
                Real logT = std::log(T_min)
                        + (std::log(T_max) - std::log(T_min)) * it / (NT - 1);
                Real Te = std::exp(logT);
                rate_table[ispe][it] = 0.1*compute_rate(Z, zstar, Te);
            }
        }
        initialized = true;
    };

    /**
     * @brief Retrieves the direct ionization rate for a given species at a specified electron temperature.
     * 
     * @param ispe Species index.
     * @param Te Electron temperature in Kelvin.
     * @return Direct ionization rate interpolated from the rate table.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real get_rate_direct_ion(int ispe, Real Te)
    {
        Real rate = 0.0;
        if(problem::use_source_DI == "Analytic") {
            Te *= K_to_eV;
            if (Te <= T_min) return rate_table[ispe][0];
            if (Te >= T_max) return rate_table[ispe][NT-1];
            Real x = (std::log(Te) - std::log(T_min)) / (std::log(T_max) - std::log(T_min));
            Real f = x * (NT - 1);
            int i = (int)f;
            Real w = f - i;
            rate = (1.0 - w) * rate_table[ispe][i] + w * rate_table[ispe][i + 1];
        }
        else if (problem::use_source_DI == "Fit") {
            rate = A_I[ispe][0] * std::pow(C::k_B*Te / C::ev2erg, q_I[ispe][0]) * std::exp(-eps_I[ispe][0] * C::ev2erg / (C::k_B*Te));
        }
        return rate * (1.0 + problem::deltarearate);
    };
    /**
     * @brief Computes the collisional excitation ionization rate for an atom.
     * @param ispe Species index
     * @param Te Electron temperature in Kelvin
     * @return Excitation ionization rate coefficient (cm³/s)
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real get_rate_ex_ion(int ispe, Real Te)
    {
        Real rate = 0.0;
        if(problem::use_source_EI == "Analytic") {
            if (zion[ispe] == 0){
                Te *= K_to_eV;
                int idx = get_atom_index(ispe);
                Real x = std::log(Te);
                Real poly = gaunt_fit[idx][0];
                for (int n = 0; n < 7; n++) {
                    poly += gaunt_fit[idx][n] * std::pow(x, 6 - n);
                }
                Real Gaunt_value = std::exp(poly);
                Real dE = ex_energy_atom[idx];
                Real f   = oscillator_strength[idx];
                Real prefactor = 1.57e-7;
                rate = prefactor * f * Gaunt_value * 1.0 / (dE * std::sqrt(Te)) * std::exp(-dE / Te);
            }
        }
        else if (problem::use_source_EI == "Fit") {
            rate = A_I[ispe][1] * std::pow(C::k_B*Te / C::ev2erg, q_I[ispe][1]) * std::exp(-eps_I[ispe][1] * C::ev2erg / (C::k_B*Te));
        }
        return rate * (1.0 + problem::deltarearate);
    };

    /**
     * @brief Computes the three-body recombination rate for a given species.
     *
     * @param[in] ispe Species index
     * @param[in] rho Mass density
     * @param[in] Te Electron temperature
     * @return Three-body recombination rate
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real get_rate_3body(int ispe, Real rho, Real Te)
    {
        Real rate = 0.0;
        if(problem::use_source_3b == "Analytic") {
            Real const lambda_DeBroglie = C::hbar * std::sqrt( 2*M_PI/(C::m_e*C::k_B*Te) );
            Real const reverse_rate_factor = rho * std::pow(lambda_DeBroglie, 3) / (aion[ispe] * C::m_p) * g_deg[ispe] / (2 * g_deg[ispe+1]);
            int idx_atom = get_atom_index(ispe);
            int idx_Eion = get_energy_index(zatom[idx_atom]) + zion[ispe];
            Real rate_forward = get_rate_direct_ion(ispe, Te) * std::exp(-ionizationEnergy[idx_Eion] / (Te * K_to_eV)) + get_rate_ex_ion(ispe, Te) * std::exp(-ex_energy_atom[idx_atom] / (Te * K_to_eV));
            rate = rate_forward * reverse_rate_factor;
        }
        else if (problem::use_source_3b == "Fit") {
            Real const lambda_DeBroglie = C::hbar * std::sqrt( 2*M_PI/(C::m_e*C::k_B*Te));
            Real const reverse_rate_factor = rho * std::pow(lambda_DeBroglie, 3) / (aion[ispe] * C::m_p) * g_deg[ispe] / (2 * g_deg[ispe+1]);
            for (int step = 0; step < 2; step++) {
                rate += A_I[ispe][step] * std::pow(C::k_B*Te / C::ev2erg, q_I[ispe][step]) * reverse_rate_factor;
            }
        }
        return rate * (1.0 + problem::deltarearate);
    }
    
};
#endif
