#ifndef COLLISIONAL_IONIZATION_H 
#define COLLISIONAL_IONIZATION_H 

#include <AMReX_AmrCore.H> 
#include <AMReX_REAL.H> 
#include <map> 

// Quantum degeneracy for each species (g = 2J + 1, ground-state term) 
AMREX_GPU_HOST_DEVICE 
inline amrex::Real g_deg[15] = { 
    // H (indices 0–1) 
    2.0, 1.0, 
    // N (indices 2–7) 
    4.0, 9.0, 6.0, 1.0, 2.0, 1.0, 
    // He (indices 8–10) 
    1.0, 2.0, 1.0, 
    // Ar (indices 11–14) 
    1.0, 4.0, 5.0, 4.0 
}; 

constexpr int num_atom_species = 4; 
constexpr amrex::Real X_min = 1e-20; // Threshold for considering a species to be present 

// Helper lambda to map species index to atomic species 
AMREX_GPU_HOST_DEVICE 
inline int get_atom_index(int mu) { 
    if (mu <= 1 && mu >= 0) return 0;   // Hydrogen 
    if (mu <= 7 && mu > 1) return 1;    // Nitrogen 
    if (mu <= 10 && mu > 7) return 2;   // Helium 
    if (mu <= 14 && mu > 10) return 3;  // Argon 
    return 0; 
}; 

AMREX_GPU_HOST_DEVICE 
const int zatom[num_atom_species] = {1, 7, 2, 18}; // Atomic numbers for H, N, He, Ar 

// Binding energies - reduced to H, N, He, Ar 
AMREX_GPU_HOST_DEVICE 
inline amrex::Real binding_energy[12] = { 
    /* Z=1 (H)  —  1 electron  */ 13.5981, 
    /* Z=7 (N)  —  7 electrons */ 20.33, 20.33, 14.534, 14.534, 14.534, 
    /* Z=2 (He) —  2 electrons */ 24.588, 24.588, 
    /* Z=18 (Ar) — 18 electrons */ 15.76, 15.76, 15.76, 15.76 
}; 

// Binding energies for each ionization stage of H, N, He, Ar [eV] 
// Format: innermost shell first, progressing outward 
AMREX_GPU_HOST_DEVICE 
inline amrex::Real ionization_energy[12] = { 
    /* Z=1  H  — 1 stage  */ 13.598434, 
    /* Z=7  N  — 5 stages */ 14.53413, 29.60125, 47.4453, 77.4735, 97.89013, 
    /* Z=2  He — 2 stages */ 24.587388, 54.417763, 
    /* Z=18 Ar — 4 stages */ 15.759611, 27.62967, 40.735, 59.58 
}; 

// Average Gaunt factor fit coefficients for excitation ionization (H, N, He, Ar) 
AMREX_GPU_HOST_DEVICE 
inline amrex::Real gaunt_fit[num_atom_species][7] = { 
    /* H */  {-7.61094672e-06, 1.42014919e-04, -5.26821513e-04, 
              -2.27233338e-03, 2.38468040e-03, 5.36211118e-02, 2.56972926e-01}, 
    /* N */  { 1.28897560e-05, -6.31681622e-05, 3.94454841e-04, 
              -2.02240986e-03, -1.12428478e-02, 4.14377747e-02, 3.31875681e-01}, 
    /* He */ {-4.17177174e-07, 5.84617203e-05, -4.65772440e-04, 
              -1.06125512e-03, 4.16013477e-03, 5.01909786e-02, 2.23090873e-01}, 
    /* Ar */ { 1.12925396e-05, 7.83928108e-05, -8.07569320e-04, 
              -1.45478529e-03, 4.14430633e-03, 5.21310441e-02, 2.49330993e-01} 
}; 

AMREX_GPU_HOST_DEVICE 
inline amrex::Real oscillator_strength[num_atom_species] = {0.416, 0.276, 0.26, 0.12}; // Oscillator strengths for H, N, He, Ar 

AMREX_GPU_HOST_DEVICE 
inline amrex::Real ex_energy_atom[num_atom_species] = {10.6, 19.8, 11.6, 10.3}; // Excitation energies for H, N, He, Ar (eV) 

// Index mapping for reduced arrays 
// Returns the starting index in the energy arrays for a given Z 
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
int get_energy_index(int Z) 
{ 
    if (Z == 1) return 0;   // H — index 0 
    if (Z == 7) return 1;   // N — indices 1–5 
    if (Z == 2) return 6;   // He — indices 6–7 
    if (Z == 18) return 8;  // Ar — indices 8–11 
    return 0; 
} 

namespace coll_ion 
{ 
    using namespace amrex; 
    
    // Temperature grid 
    static constexpr int NT = 500; 
    static constexpr Real T_min = 1e-3;   // eV 
    static constexpr Real T_max = 3.0e2;  // eV 
    static constexpr Real K_to_eV = 8.617333262145e-5; 

    // Energy grid for cross-section integration 
    static constexpr int npoints = 500; 

    // Table storage 
    inline AMREX_GPU_MANAGED 
    Real rate_table[NumSpec][NT]; 

    inline AMREX_GPU_MANAGED 
    bool initialized = false; 

    /** * @brief Maxwellian energy-distribution weight. 
     * * Returns sqrt(E) * exp(-E / Te), the unnormalized weight of 
     * electron kinetic energy E in a Maxwellian characterised by 
     * temperature Te (both in eV). 
     */ 
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
    Real maxwell_weight(Real E, Real Te) 
    { 
        return std::sqrt(E) * std::exp(-E / Te); 
    } 

    AMREX_GPU_HOST_DEVICE 
    const int num_simulated_electrons[num_atom_species] = {1, 5, 2, 4}; // Matches the entries in binding_energy 

    // Cross section using RBEB model from Kim et al. (2000) 
    /** * @brief Relativistic Binary-Encounter-Bethe (RBEB) electron-impact 
     * ionization cross section. 
     * * Loops over all bound electrons of the ion (from innermost to the 
     * outermost occupied shell). Degenerate sub-shells sharing the same 
     * binding energy are grouped and counted together (factor `n`). 
     * * @param atomic_number Atomic number Z of the target element. 
     * @param zstar Current ion charge (number of removed electrons). 
     * @param ep Incident-electron kinetic energy [eV]. 
     * @return Ionization cross section [cm²]. 
     */ 
    AMREX_GPU_HOST_DEVICE inline 
    Real sigma_E(int atomic_number, int zstar, Real ep) 
    { 
        Real sigma = 0.0; 
        int n = 1; 

        // Map atomic_number to the local species index to find simulated bounds 
        int atom_idx = 0; 
        if (atomic_number == 7) atom_idx = 1; 
        if (atomic_number == 2) atom_idx = 2; 
        if (atomic_number == 18) atom_idx = 3; 

        int base_idx = get_energy_index(atomic_number); 
        const Real me_c2 = 510998.0; 
        const Real normalization = 2.0 * M_PI * std::pow(1.0/137.0, 4) * std::pow(5.29e-11, 2); 

        const Real ep_mc2 = ep / me_c2; 

        // NEW LOOP BOUND: Stop when we hit the simulated limit 
        int max_k = num_simulated_electrons[atom_idx] - zstar; 
        for (int k = 0; k < max_k; k++) { 
            int idx_k = base_idx + k; 
            int idx_k_p = base_idx + (k + 1); 

            Real bp = binding_energy[idx_k] / me_c2; 

            // Accumulate degenerate sub-shells (Adding a safety bounds check for idx_k_p) 
            if (k < max_k - 1 && 
                idx_k_p < base_idx + num_simulated_electrons[atom_idx] && 
                binding_energy[idx_k] == binding_energy[idx_k_p]) { 
                n++; 
                continue; 
            } 

            Real e = ep_mc2 / bp; // fix 2: use pre-converted ep_mc2 

            if (e > 1.0) { 
                Real betae2 = 1.0 - 1.0 / ((1.0 + ep_mc2) * (1.0 + ep_mc2)); 
                Real betab2 = 1.0 - 1.0 / ((1.0 + bp) * (1.0 + bp)); 
                Real betau2 = betab2; 

                Real s0 = normalization * n / 
                          (bp * (betae2 + betab2 + betau2)); 

                Real ep2 = 1.0 / (1.0 + 0.5 * ep_mc2); // fix 2 
                ep2 *= ep2; 

                Real a1c = (1.0 + 2.0 * ep_mc2) / (1.0 + e) * ep2; // fix 2 
                Real a2c = (e - 1.0) * bp * bp * 0.5 * ep2; 
                Real a3 = std::log(betae2 / (1.0 - betae2)) 
                          - betae2 
                          - std::log(2.0 * bp); 

                Real sk = s0 * (0.5 * a3 * (1.0 - 1.0 / (e * e)) 
                          + 1.0 - 1.0 / e 
                          + a2c 
                          - a1c * std::log(e)); 

                sigma += sk; 
            } 

            n = 1; 
        } 

        return sigma * 1e4; // convert to cm^2 
    } 

    /** * @brief Maxwell-averaged collisional-ionization rate coefficient. 
     * * Numerically integrates σ(E) · v(E) over a Maxwellian electron 
     * distribution using a simple Riemann sum on [E_min, E_max]. 
     * * @param atomic_number Atomic number Z. 
     * @param zstar Current ion charge. 
     * @param Te Electron temperature [eV]. 
     * @return Rate coefficient <σ v> [cm³ s⁻¹]. 
     */ 
    inline AMREX_GPU_HOST_DEVICE 
    Real compute_rate(int atomic_number, int zstar, Real Te) 
    { 
        Real sum = 0.0; 
        Real norm = 0.0; 

        const Real E_min = 0.0; 
        const Real E_max = 1e3; // upper integration bound [eV] 
        const Real dE = (E_max - E_min) / npoints; 

        for (int i = 0; i < npoints; i++) { 
            Real E = E_min + i * dE; // electron energy [eV] 
            Real w = maxwell_weight(E, Te); // distribution weight 
            Real sig = sigma_E(atomic_number, zstar, E); // cross section [cm²] 
            Real v = std::sqrt(2.0 * E * C::ev2erg / C::m_e); // velocity [cm/s] 

            sum += sig * v * w; 
            norm += w; 
        } 

        if (norm > 0.0) { 
            sum /= norm; 
        } 

        return sum; 
    } 

    /** * @brief Pre-compute the direct-ionization rate table. 
     * * Fills `rate_table[ispe][it]` for every species and every point on 
     * the logarithmic temperature grid. Called once at start-up; the 
     * `initialized` flag prevents redundant work. 
     */ 
    inline void initialize() 
    { 
        if (initialized) return; 
        
        for (int ispe = 0; ispe < NumSpec; ispe++) { 
            int Z = zatom[get_atom_index(ispe)]; 
            int zstar = zion[ispe]; 

            for (int it = 0; it < NT; it++) { 
                // Logarithmic spacing: T = T_min * (T_max/T_min)^(it/(NT-1)) 
                Real logT = std::log(T_min) 
                            + (std::log(T_max) - std::log(T_min)) * it / (NT - 1); 
                Real Te = std::exp(logT); 

                rate_table[ispe][it] = compute_rate(Z, zstar, Te); 
            } 
        } 

        initialized = true; 
    } 

    /** * @brief Direct-ionization rate from the pre-computed table. 
     * * Performs log-linear interpolation on the temperature grid. 
     * Clamps to the boundary values when Te falls outside [T_min, T_max]. 
     * * @param ispe Species index (into `rate_table`). 
     * @param Te Electron temperature [K]. 
     * @return Direct-ionization rate coefficient [cm³ s⁻¹]. 
     */ 
    AMREX_GPU_HOST_DEVICE inline 
    Real get_rate_direct_ion(int ispe, Real Te) 
    { 
        Te *= K_to_eV; // K → eV 

        if (Te <= T_min) return rate_table[ispe][0]; 
        if (Te >= T_max) return rate_table[ispe][NT - 1]; 

        // Fractional index on the log-spaced grid 
        Real x = (std::log(Te) - std::log(T_min)) 
                 / (std::log(T_max) - std::log(T_min)); 
        Real f = x * (NT - 1); 
        int i = static_cast<int>(f); 
        Real w = f - i; 
        return (1.0 - w) * rate_table[ispe][i] + w * rate_table[ispe][i + 1]; 
    } 

    /** * @brief Collisional excitation-ionization rate coefficient. 
     * * Applicable only to neutral atoms (zion == 0). Uses a 6th-order 
     * polynomial fit for the average Gaunt factor and the van Regemorter 
     * formula with species-specific oscillator strengths. 
     * * @param ispe Species index. 
     * @param Te Electron temperature [K]. 
     * @return Excitation-ionization rate coefficient [cm³ s⁻¹]. 
     */ 
    AMREX_GPU_HOST_DEVICE inline 
    Real get_rate_ex_ion(int ispe, Real Te) 
    { 
        Real rate = 0.0; 

        // Only neutral atoms undergo excitation-ionization 
        if (zion[ispe] != 0) return rate; 

        Te *= K_to_eV; // K → eV 

        // FIX: Shield against extremely low temperatures where rate is physically 0 
        // and log(Te) would cause polynomial overflow. 
        if (Te < T_min) return 0.0; 
        // Optional: Clamp to T_max if your fits blow up at high temperatures too 
        if (Te > T_max) Te = T_max; 

        int idx = get_atom_index(ispe); 

        // Evaluate the 6th-order polynomial for ln(Gaunt) in ln(Te) 
        Real x = std::log(Te); 
        Real poly = 0.0; 
        for (int n = 0; n < 7; n++) { 
            poly += gaunt_fit[idx][n] * std::pow(x, 6 - n); 
        } 
        Real Gaunt_value = std::exp(poly); 

        Real dE = ex_energy_atom[idx]; // excitation energy [eV] 
        Real f = oscillator_strength[idx]; // oscillator strength 
        Real prefactor = 1.57e-7; // van Regemorter constant 

        rate = prefactor * f * Gaunt_value 
               / (dE * std::sqrt(Te)) 
               * std::exp(-dE / Te); 
        return rate; 
    } 

    /** * @brief Three-body recombination rate coefficient. 
     * * Derived from the ionization rate through the principle of detailed 
     * balance: 
     * R_3body = R_forward × (λ_dB³ / a_i m_p) × (g_i / 2 g_{i+1}) 
     * * where λ_dB is the thermal de Broglie wavelength of the electron 
     * and g_i are the statistical weights (degeneracies). 
     * * @param ispe Species index. 
     * @param rho Mass density [g cm⁻³]. 
     * @param Te Electron temperature [K]. 
     * @return Three-body recombination rate coefficient. 
     */ 
    AMREX_GPU_HOST_DEVICE inline 
    Real get_rate_3body(int ispe, Real rho, Real Te) 
    { 
        if (ispe + 1 >= NumSpec) return 0.0; 
        
        // Thermal de Broglie wavelength of the electron 
        Real lambda_DeBroglie = C::hbar 
                                * std::sqrt(2.0 * M_PI / (C::m_e * C::k_B * Te)); 

        // Detailed-balance factor: ρ λ³ / (A m_p) · g_i / (2 g_{i+1}) 
        Real reverse_factor = rho 
                              * std::pow(lambda_DeBroglie, 3) 
                              / (aion[ispe] * C::m_p) 
                              * g_deg[ispe] / (2.0 * g_deg[ispe + 1]); 

        // Identify the binding and excitation energies for this ion stage 
        int idx_atom = get_atom_index(ispe); 
        int idx_Eion = get_energy_index(zatom[idx_atom]) + zion[ispe]; 

        // Forward rate = direct ionization + excitation ionization 
        // (each weighted by its Boltzmann factor) 
        Real rate_forward = get_rate_direct_ion(ispe, Te) * std::exp(ionization_energy[idx_Eion] / (Te * K_to_eV)) 
                            + get_rate_ex_ion(ispe, Te) * std::exp(ex_energy_atom[idx_atom] / (Te * K_to_eV)); 

        return rate_forward * reverse_factor; 
    } 
} 

#endif // COLLISIONAL_IONIZATION_H