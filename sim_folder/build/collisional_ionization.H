#ifndef COLLISIONAL_IONIZATION_H 
#define COLLISIONAL_IONIZATION_H 

#include <AMReX_AmrCore.H> 
#include <AMReX_REAL.H> 
#include <map> 
#include <cmath>
#include <limits>
#include <algorithm>

// Quantum degeneracy of each species (H, N, He, Ar)
inline AMREX_GPU_MANAGED amrex::Real g_deg[21] = { 
    2.0_rt, 1.0_rt, 
    4.0_rt, 9.0_rt, 6.0_rt, 1.0_rt, 2.0_rt, 1.0_rt, 
    1.0_rt, 2.0_rt, 1.0_rt, 
    1.0_rt, 4.0_rt, 5.0_rt, 4.0_rt, 1.0_rt, 2.0_rt, 1.0_rt, 2.0_rt, 1.0_rt, 4.0_rt 
}; 
// Number of atom species
constexpr int num_atom_species = 4; 
constexpr amrex::Real X_min = 1e-20_rt; 

inline AMREX_GPU_MANAGED int atom_first_species[num_atom_species] = {0, 2, 8, 11};
// Helper function to get the index of the atom species from the atomic number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
int get_atom_index(int mu) { 
    if (mu <= 1 && mu >= 0) return 0; // H
    if (mu <= 7 && mu > 1)  return 1; // N
    if (mu <= 10 && mu > 7) return 2; // He
    if (mu <= 18 && mu > 10)return 3; // Ar
    return 0; 
} 
// Atomic numbers of each species (H, N, He, Ar)
inline AMREX_GPU_MANAGED int zatom[num_atom_species] = {1, 7, 2, 18};
// Ionization energies of each species (H, N, He, Ar)
inline AMREX_GPU_MANAGED amrex::Real ionization_energy[17] = { 
    13.598434_rt, 
    14.53413_rt, 29.60125_rt, 47.4453_rt, 77.4735_rt, 97.89013_rt, 
    24.587388_rt, 54.417763_rt, 
    15.759611_rt, 27.62967_rt, 40.735_rt, 59.81_rt, 75.02_rt, 91.009_rt, 124.323_rt, 143.46_rt, 422.45_rt 
}; 
// Gaunt factors polynomial coefficients for each species (H, N, He, Ar)
inline AMREX_GPU_MANAGED amrex::Real gaunt_fit[num_atom_species][7] = { 
    {-7.61094672e-06_rt, 1.42014919e-04_rt, -5.26821513e-04_rt, -2.27233338e-03_rt, 2.38468040e-03_rt, 5.36211118e-02_rt, 2.56972926e-01_rt}, 
    { 1.28897560e-05_rt, -6.31681622e-05_rt, 3.94454841e-04_rt, -2.02240986e-03_rt, -1.12428478e-02_rt, 4.14377747e-02_rt, 3.31875681e-01_rt}, 
    {-4.17177174e-07_rt, 5.84617203e-05_rt, -4.65772440e-04_rt, -1.06125512e-03_rt, 4.16013477e-03_rt, 5.01909786e-02_rt, 2.23090873e-01_rt}, 
    { 1.12925396e-05_rt, 7.83928108e-05_rt, -8.07569320e-04_rt, -1.45478529e-03_rt, 4.14430633e-03_rt, 5.21310441e-02_rt, 2.49330993e-01_rt} 
}; 
// Oscillator strengths for each species (H, N, He, Ar)
inline AMREX_GPU_MANAGED amrex::Real oscillator_strength[num_atom_species] = {0.416_rt, 0.276_rt, 0.26_rt, 0.12_rt}; 
// Excitation energy for each species (H, N, He, Ar) 
inline AMREX_GPU_MANAGED amrex::Real ex_energy_atom[num_atom_species] = {10.6_rt, 19.8_rt, 11.6_rt, 10.3_rt}; 

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
    
    static constexpr int NT = 100;  // Number of temperature points
    static constexpr Real T_min = 1e-3_rt;   // Minimum temperature
    static constexpr Real T_max = 3.0e2_rt;  // Maximum temperature
    static constexpr Real K_to_eV = 8.617333262145e-5_rt;  // Conversion factor from Kelvin to eV

    static constexpr int npoints = 1000; 

    // Elementary charge in CGS. Used for the Coulomb temperature floor.
    static constexpr Real q_elem_cgs = 4.803204712570263e-10_rt;

    // Largest / smallest exponent we can safely feed to std::exp() for the
    // active precision (double or single). Used purely as an overflow guard.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real log_huge()
    {
        constexpr Real ln2 = 0.6931471805599453_rt;
        return 0.99_rt * static_cast<Real>(std::numeric_limits<Real>::max_exponent) * ln2;
    }

    // The rate varies like exp(-E_th/Te) close to threshold, which cannot be
    // tabulated nor linearly interpolated accurately at low Te. We therefore store
    // log(rate) + E_th/Te, which is smooth in log(Te), and add back -E_th/Te analytically.
    inline AMREX_GPU_MANAGED Real log_reduced_rate_table[NumSpec][NT];
    // Lowest energy (in eV) at which the ionization cross-section is non-zero
    inline AMREX_GPU_MANAGED Real threshold_energy[NumSpec];
    // False when the species cannot be ionized (e.g. fully stripped ion)
    inline AMREX_GPU_MANAGED bool has_ionization[NumSpec];
    inline AMREX_GPU_MANAGED bool initialized = false; 
    // Electron impact ionization cross-section from Kim et al. (2000)
    AMREX_GPU_HOST_DEVICE inline 
    Real sigma_E(int atomic_number, int zstar, Real ep) 
    { 
        // Prevent electron (-1) out of bounds crash
        if (zstar < 0) return 0.0_rt;
        Real sigma = 0.0_rt; 
        int n = 1; 

        int atom_idx = 0; 
        if (atomic_number == 7) atom_idx = 1; 
        if (atomic_number == 2) atom_idx = 2; 
        if (atomic_number == 18) atom_idx = 3; 
        // Binding energies for each species (H, N, He, Ar)
        static constexpr Real binding_energy[17] = { 
            13.5981_rt, 
            20.33_rt, 20.33_rt, 14.534_rt, 14.534_rt, 14.534_rt, 
            24.588_rt, 24.588_rt, 
            422.45_rt, 29.3_rt, 29.3_rt, 15.76_rt, 15.76_rt, 15.76_rt, 15.76_rt, 15.76_rt, 15.76_rt 
        }; 
        // Number of levels to consider per atomic species
        static constexpr int num_simulated_electrons[num_atom_species] = {1, 5, 2, 9}; 

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
    // Finds the lowest energy (in eV) at which the ionization cross-section is non-zero.
    // Returns a negative value if the cross-section vanishes everywhere.
    inline Real find_threshold_energy(int atomic_number, int zstar)
    {
        // Coarse scan on a logarithmic grid, followed by a bisection
        constexpr int nscan = 4000;
        const Real log_E_lo = std::log(1.e-2_rt);
        const Real log_E_hi = std::log(1.e5_rt);
        Real E_prev = 0.0_rt;
        for (int i = 0; i < nscan; i++) {
            Real E = std::exp(log_E_lo + (log_E_hi - log_E_lo) * i / (nscan - 1));
            if (sigma_E(atomic_number, zstar, E) > 0.0_rt) {
                Real lo = E_prev, hi = E;
                for (int it = 0; it < 100; it++) {
                    Real mid = 0.5_rt * (lo + hi);
                    if (sigma_E(atomic_number, zstar, mid) > 0.0_rt) { hi = mid; } else { lo = mid; }
                }
                return lo;
            }
            E_prev = E;
        }
        return -1.0_rt;
    }
    // Computes log(rate) + E_th/Te, where rate is the ionization rate obtained
    // by integrating the electron impact ionization cross-section over a maxwellian.
    // The integration starts at the threshold E_th, with a step scaled to Te
    // (E = E_th + x*Te), so that it remains accurate even when Te << E_th.
    // The maxwellian normalization int sqrt(E) exp(-E/Te) dE = sqrt(pi)/2 Te^1.5 is analytical.
    inline Real compute_log_reduced_rate(int atomic_number, int zstar, Real E_th, Real Te)
    {
        constexpr Real x_max = 60.0_rt;
        const Real dx = x_max / npoints;
        Real sum = 0.0_rt;
        // Trapezoidal rule (the integrand vanishes at x=0)
        for (int i = 1; i <= npoints; i++) {
            Real x = i * dx;
            Real E = E_th + x * Te;
            Real sig = sigma_E(atomic_number, zstar, E);
            Real v = std::sqrt(2.0_rt * E * C::ev2erg / C::m_e);
            Real weight = (i == npoints) ? 0.5_rt : 1.0_rt;
            sum += weight * sig * v * std::sqrt(E) * std::exp(-x);
        }
        sum *= dx * Te;
        if (sum <= 0.0_rt) return -log_huge();
        return std::log(sum) - std::log(0.5_rt * std::sqrt(M_PI) * std::pow(Te, 1.5_rt));
    }
    // Precomputes the ionization rate for all species and temperatures
    // and stores them in the log_reduced_rate_table array to speed up the computation
    inline void initialize() 
    { 
        if (initialized) return; 

        for (int ispe = 0; ispe < NumSpec; ispe++) { 
            int Z = zatom[get_atom_index(ispe)]; 
            int zstar = zion[ispe]; 

            Real E_th = find_threshold_energy(Z, zstar);
            has_ionization[ispe] = (E_th > 0.0_rt);
            threshold_energy[ispe] = has_ionization[ispe] ? E_th : 0.0_rt;

            for (int it = 0; it < NT; it++) { 
                Real logT = std::log(T_min) + (std::log(T_max) - std::log(T_min)) * it / (NT - 1); 
                Real Te = std::exp(logT); 
                log_reduced_rate_table[ispe][it] = has_ionization[ispe] ?
                    compute_log_reduced_rate(Z, zstar, E_th, Te) : -log_huge();
            } 
        } 
        initialized = true; 
    } 
    // Returns log(rate) of the direct ionization rate for a given species and temperature (in K),
    // by interpolating the precomputed table in log space.
    // Returns -log_huge() if the species cannot be ionized.
    AMREX_GPU_HOST_DEVICE inline
    Real get_log_rate_direct_ion(int ispe, Real Te)
    {
        if (!has_ionization[ispe]) return -log_huge();

        Te *= K_to_eV;
        // Clamp the temperature at which the (smooth) reduced rate is evaluated,
        // but keep the exact exponential factor exp(-E_th/Te)
        Real Te_tab = amrex::Clamp(Te, T_min, T_max);

        Real x = (std::log(Te_tab) - std::log(T_min)) / (std::log(T_max) - std::log(T_min));
        Real f = x * (NT - 1);
        int i = amrex::min(static_cast<int>(f), NT - 2);
        Real w = f - i;
        Real log_reduced = (1.0_rt - w) * log_reduced_rate_table[ispe][i] + w * log_reduced_rate_table[ispe][i + 1];
        return log_reduced - threshold_energy[ispe] / Te;
    }
    // Accesses the ionization rate for a given species and temperature (in K)
    AMREX_GPU_HOST_DEVICE inline 
    Real get_rate_direct_ion(int ispe, Real Te) 
    { 
        Real log_rate = get_log_rate_direct_ion(ispe, Te);
        if (log_rate <= -log_huge()) return 0.0_rt;
        return std::exp(log_rate);
    } 
    // Accesses the excitation ionization rate for a given atomic number and temperature
    // by interpolating the precomputed rate_table array
    // Returns log(rate), or -log_huge() if the rate vanishes
    AMREX_GPU_HOST_DEVICE inline 
    Real get_log_rate_ex_ion(int ispe, Real Te) 
    { 
        if (zion[ispe] != 0) return -log_huge(); 

        Te *= K_to_eV; 

        if (Te < T_min) return -log_huge(); 
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

        // Computed in log space, since exp(-dE/Te) underflows at low Te
        return std::log(prefactor * f / (dE * std::sqrt(Te))) + poly - dE / Te;
    } 
    AMREX_GPU_HOST_DEVICE inline 
    Real get_rate_ex_ion(int ispe, Real Te) 
    { 
        Real log_rate = get_log_rate_ex_ion(ispe, Te);
        if (log_rate <= -log_huge()) return 0.0_rt;
        return std::exp(log_rate);
    } 

    // Effective temperature used to regularize the 3-body recombination
    // rate at low Te.
    AMREX_GPU_HOST_DEVICE inline
    Real get_effective_temperature(Real Te, Real n_e)
    {
        // Prevent n_e from being too small (in cm^-3)
        const Real n_e_safe = amrex::max(n_e * 1e-6_rt, 1.e-30_rt);
        // Wigner-Seitz radius and the associated Coulomb temperature
        const Real a_d = std::cbrt(3.0_rt / (4.0_rt * M_PI * n_e_safe));
        const Real T_coulomb = (q_elem_cgs * q_elem_cgs / a_d) / C::k_B;
        // Regularized evaluation temperature including Coulomb temperature
        return std::sqrt(Te * Te + T_coulomb * T_coulomb);
    }

    // Accesses the 3-body recombination rate for a given atomic number and
    // temperature with the Saha detailled balance.
    AMREX_GPU_HOST_DEVICE inline
    Real get_rate_3body(int ispe, Real n_e, Real rho, Real Te)
    {
        // Regularized evaluation temperature including Coulomb temperature
        Real Te_eff = get_effective_temperature(Te, n_e);
        const Real Te_eff_eV = Te_eff * K_to_eV;

        // Work with log(rate): the rates themselves underflow at low Te, while
        // the products rate*exp(E/Te) needed for detailed balance are finite
        const Real log_rate_dir = get_log_rate_direct_ion(ispe, Te_eff);
        const Real log_rate_ex  = get_log_rate_ex_ion(ispe, Te_eff);

        const int idx_atom = get_atom_index(ispe);
        const int idx_Eion = get_energy_index(zatom[idx_atom]) + zion[ispe];

        // Assemble the product in log space
        const Real ln_tiny = -log_huge();

        // lambda_DeBroglie = hbar * sqrt(2 pi / (m_e k_B Te))
        const Real log_lambda3 = 1.5_rt * std::log(2.0_rt * M_PI * C::hbar * C::hbar
                                                   / (C::m_e * C::k_B * Te_eff));

        const Real log_geom = std::log(rho) - std::log(aion[ispe] * C::m_p)
                            + std::log(g_deg[ispe]) - std::log(2.0_rt * g_deg[ispe + 1]);

        // Calculate the log of the rate for two different physical processes (ionization and excitation)

        // Add the energy contribution (energy divided by electron temperature)
        const Real a = log_rate_dir + ionization_energy[idx_Eion] / Te_eff_eV;

        // Same process for the excitation channel
        const Real b = log_rate_ex + ex_energy_atom[idx_atom] / Te_eff_eV;

        // Combine the two channels using the "log-sum-exp" trick
        // This is a numerically stable way to compute log(e^a + e^b) without overflow
        const Real max_val = amrex::max(a, b);  // Find the larger value
        const Real log_rate_forward = max_val + std::log(std::exp(a - max_val) + std::exp(b - max_val));

        // Add additional physical factors (lambda and geometry corrections)
        const Real log_rate = log_rate_forward + log_lambda3 + log_geom;

        // If the rate is too small (underflow), return zero
        if (log_rate < ln_tiny) return 0.0_rt;

        // Convert back from log space to actual rate, but cap at a maximum to prevent overflow
        return std::exp(amrex::min(log_rate, log_huge()));
    }
} 

#endif // COLLISIONAL_IONIZATION_H
