#ifndef COLLISIONAL_IONIZATION_H 
#define COLLISIONAL_IONIZATION_H 

#include <AMReX_AmrCore.H> 
#include <AMReX_REAL.H> 
#include <map> 
#include <cmath>
#include <algorithm>

// --- SHARED ARRAYS (Used in multiple functions) ---
// Using inline AMREX_GPU_MANAGED ensures A100 visibility and no linker errors

inline AMREX_GPU_MANAGED amrex::Real g_deg[15] = { 
    2.0_rt, 1.0_rt, 4.0_rt, 9.0_rt, 6.0_rt, 1.0_rt, 2.0_rt, 1.0_rt, 
    1.0_rt, 2.0_rt, 1.0_rt, 1.0_rt, 4.0_rt, 5.0_rt, 4.0_rt 
};

constexpr int num_atom_species = 4; 
constexpr amrex::Real X_min = 1e-20_rt; 

inline AMREX_GPU_MANAGED int zatom[num_atom_species] = {1, 7, 2, 18};

inline AMREX_GPU_MANAGED amrex::Real ionization_energy[12] = { 
    13.598434_rt, 
    14.53413_rt, 29.60125_rt, 47.4453_rt, 77.4735_rt, 97.89013_rt, 
    24.587388_rt, 54.417763_rt, 
    15.759611_rt, 27.62967_rt, 40.735_rt, 59.58_rt 
}; 

inline AMREX_GPU_MANAGED amrex::Real gaunt_fit[num_atom_species][7] = { 
    {-7.61094672e-06_rt, 1.42014919e-04_rt, -5.26821513e-04_rt, -2.27233338e-03_rt, 2.38468040e-03_rt, 5.36211118e-02_rt, 2.56972926e-01_rt}, 
    { 1.28897560e-05_rt, -6.31681622e-05_rt, 3.94454841e-04_rt, -2.02240986e-03_rt, -1.12428478e-02_rt, 4.14377747e-02_rt, 3.31875681e-01_rt}, 
    {-4.17177174e-07_rt, 5.84617203e-05_rt, -4.65772440e-04_rt, -1.06125512e-03_rt, 4.16013477e-03_rt, 5.01909786e-02_rt, 2.23090873e-01_rt}, 
    { 1.12925396e-05_rt, 7.83928108e-05_rt, -8.07569320e-04_rt, -1.45478529e-03_rt, 4.14430633e-03_rt, 5.21310441e-02_rt, 2.49330993e-01_rt} 
}; 

inline AMREX_GPU_MANAGED amrex::Real oscillator_strength[num_atom_species] = {0.416_rt, 0.276_rt, 0.26_rt, 0.12_rt}; 

inline AMREX_GPU_MANAGED amrex::Real ex_energy_atom[num_atom_species] = {10.6_rt, 19.8_rt, 11.6_rt, 10.3_rt}; 

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
int get_atom_index(int mu) { 
    if (mu <= 1 && mu >= 0) return 0;   
    if (mu <= 7 && mu > 1)  return 1;   
    if (mu <= 10 && mu > 7) return 2;   
    if (mu <= 14 && mu > 10)return 3;   
    return 0; 
} 

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
int get_energy_index(int Z) 
{ 
    if (Z == 1) return 0;   
    if (Z == 7) return 1;   
    if (Z == 2) return 6;   
    if (Z == 18) return 8;  
    return 0; 
} 

namespace coll_ion 
{ 
    using namespace amrex; 
    
    static constexpr int NT = 100;  
    static constexpr Real T_min = 1e-3_rt;   
    static constexpr Real T_max = 3.0e2_rt;  
    static constexpr Real K_to_eV = 8.617333262145e-5_rt;  
    static constexpr int npoints = 100; 

    inline AMREX_GPU_MANAGED Real rate_table[NumSpec][NT];
    inline AMREX_GPU_MANAGED bool initialized = false; 

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
    Real maxwell_weight(Real E, Real Te) 
    { 
        return std::sqrt(E) * std::exp(-E / Te); 
    } 

    AMREX_GPU_HOST_DEVICE inline 
    Real sigma_E(int atomic_number, int zstar, Real ep) 
    { 
        // SAFETY GUARD: Prevent electron (-1) out of bounds crash
        if (zstar < 0) return 0.0_rt;

        Real sigma = 0.0_rt; 
        int n = 1; 

        int atom_idx = 0; 
        if (atomic_number == 7) atom_idx = 1; 
        if (atomic_number == 2) atom_idx = 2; 
        if (atomic_number == 18) atom_idx = 3; 

        // --- LOCAL ARRAYS (Used only in this function) ---
        // NVCC will load these into registers/constant memory cleanly
        static constexpr Real binding_energy[12] = { 
            13.5981_rt, 
            20.33_rt, 20.33_rt, 14.534_rt, 14.534_rt, 14.534_rt, 
            24.588_rt, 24.588_rt, 
            15.76_rt, 15.76_rt, 15.76_rt, 15.76_rt 
        }; 
        
        static constexpr int num_simulated_electrons[num_atom_species] = {1, 5, 2, 4}; 

        int base_idx = get_energy_index(atomic_number); 
        const Real me_c2 = 510998.0_rt; 
        const Real normalization = 2.0_rt * M_PI * std::pow(1.0_rt/137.0_rt, 4) * std::pow(5.29e-11_rt, 2); 

        const Real ep_mc2 = ep / me_c2; 

        int neutral_valence_idx = base_idx + num_simulated_electrons[atom_idx] - 1;
        Real neutral_valence_energy = binding_energy[neutral_valence_idx];

        int max_k = num_simulated_electrons[atom_idx] - zstar; 
        
        for (int k = 0; k < max_k; k++) { 
            int idx_k = base_idx + k; 
            int idx_k_p = base_idx + (k + 1); 

            Real bp_energy = binding_energy[idx_k];

            if (std::abs(bp_energy - neutral_valence_energy) < 1e-5_rt) {
                int idx_Eion = get_energy_index(atomic_number) + zstar;
                bp_energy = ionization_energy[idx_Eion];
            }

            Real bp = bp_energy / me_c2;

            if (k < max_k - 1 && 
                idx_k_p < base_idx + num_simulated_electrons[atom_idx] && 
                binding_energy[idx_k] == binding_energy[idx_k_p]) { 
                n++; 
                continue; 
            } 

            Real e = ep_mc2 / bp; 

            if (e > 1.0_rt) { 
                Real betae2 = 1.0_rt - 1.0_rt / ((1.0_rt + ep_mc2) * (1.0_rt + ep_mc2)); 
                Real betab2 = 1.0_rt - 1.0_rt / ((1.0_rt + bp) * (1.0_rt + bp)); 
                Real betau2 = betab2; 

                Real s0 = normalization * n / (bp * (betae2 + betab2 + betau2)); 
                Real ep2 = 1.0_rt / (1.0_rt + 0.5_rt * ep_mc2); 
                ep2 *= ep2; 

                Real a1c = (1.0_rt + 2.0_rt * ep_mc2) / (1.0_rt + e) * ep2; 
                Real a2c = (e - 1.0_rt) * bp * bp * 0.5_rt * ep2; 
                Real a3 = std::log(betae2 / (1.0_rt - betae2)) - betae2 - std::log(2.0_rt * bp); 

                Real sk = s0 * (0.5_rt * a3 * (1.0_rt - 1.0_rt / (e * e)) + 1.0_rt - 1.0_rt / e + a2c - a1c * std::log(e)); 
                sigma += sk; 
            } 
            n = 1; 
        } 
        return sigma * 1e4_rt;
    } 
    
    inline AMREX_GPU_HOST_DEVICE 
    Real compute_rate(int atomic_number, int zstar, Real Te) 
    { 
        Real sum = 0.0_rt; 
        Real norm = 0.0_rt; 

        const Real E_min = 0.0_rt; 
        const Real E_max = 1e3_rt; 
        const Real dE = (E_max - E_min) / npoints; 

        for (int i = 0; i < npoints; i++) { 
            Real E = E_min + i * dE; 
            Real w = maxwell_weight(E, Te); 
            Real sig = sigma_E(atomic_number, zstar, E); 
            Real v = std::sqrt(2.0_rt * std::abs(E) * C::ev2erg / C::m_e); 

            sum += sig * v * w; 
            norm += w; 
        } 

        if (norm > 0.0_rt) { 
            sum /= norm; 
        } 
        return sum; 
    } 
    
    inline void initialize() 
    { 
        if (initialized) return; 

        for (int ispe = 0; ispe < NumSpec; ispe++) { 
            int Z = zatom[get_atom_index(ispe)]; 
            int zstar = zion[ispe]; 

            for (int it = 0; it < NT; it++) { 
                Real logT = std::log(T_min) + (std::log(T_max) - std::log(T_min)) * it / (NT - 1); 
                Real Te = std::exp(logT); 
                rate_table[ispe][it] = compute_rate(Z, zstar, Te); 
            } 
        } 
        initialized = true; 
    } 
    
    AMREX_GPU_HOST_DEVICE inline 
    Real get_rate_direct_ion(int ispe, Real Te) 
    { 
        Te *= K_to_eV; 

        if (Te <= T_min) return rate_table[ispe][0]; 
        if (Te >= T_max) return rate_table[ispe][NT - 1]; 

        Real x = (std::log(Te) - std::log(T_min)) / (std::log(T_max) - std::log(T_min)); 
        Real f = x * (NT - 1); 
        int i = static_cast<int>(f); 
        Real w = f - i; 
        return (1.0_rt - w) * rate_table[ispe][i] + w * rate_table[ispe][i + 1]; 
    } 
    
    AMREX_GPU_HOST_DEVICE inline 
    Real get_rate_ex_ion(int ispe, Real Te) 
    { 
        Real rate = 0.0_rt; 

        if (zion[ispe] != 0) return rate; 

        Te *= K_to_eV; 

        if (Te < T_min) return 0.0_rt; 
        if (Te > T_max) Te = T_max; 

        int idx = get_atom_index(ispe); 

        Real x = std::log(Te); 
        Real poly = 0.0_rt; 
        for (int n = 0; n < 7; n++) { 
            poly += gaunt_fit[idx][n] * std::pow(x, 6 - n); 
        } 
        Real Gaunt_value = std::exp(poly); 

        Real dE = ex_energy_atom[idx]; 
        Real f = oscillator_strength[idx]; 
        Real prefactor = 1.57e-7_rt; 

        rate = prefactor * f * Gaunt_value / (dE * std::sqrt(Te)) * std::exp(-dE / Te); 
        return rate; 
    } 

    AMREX_GPU_HOST_DEVICE inline
    Real get_rate_3body(int ispe, Real rho, Real Te)
    {
        if (ispe + 1 >= NumSpec) return 0.0_rt;

        const Real Te_eV = Te * K_to_eV;
        if (Te_eV < T_min) return 0.0_rt;          
        
        Real lambda_DeBroglie = C::hbar * std::sqrt(2.0_rt * M_PI / (C::m_e * C::k_B * Te));

        Real reverse_factor = rho * std::pow(lambda_DeBroglie, 3)
                            / (aion[ispe] * C::m_p)
                            * g_deg[ispe] / (2.0_rt * g_deg[ispe + 1]);

        int idx_atom = get_atom_index(ispe);
        int idx_Eion = get_energy_index(zatom[idx_atom]) + zion[ispe];
        
        Real exp_ion = std::exp(std::min(ionization_energy[idx_Eion] / Te_eV, 50.0_rt));
        Real exp_ex  = std::exp(std::min(ex_energy_atom[idx_atom]    / Te_eV, 50.0_rt));

        Real rate_forward = get_rate_direct_ion(ispe, Te) * exp_ion
                          + get_rate_ex_ion(ispe, Te)     * exp_ex;

        return rate_forward * reverse_factor;
    }
} 

#endif // COLLISIONAL_IONIZATION_H