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
    static constexpr Real T_min = 1e-3_rt;   // Minimum temperature in eV
    static constexpr Real T_max = 5.0e2_rt;  // Maximum temperature in eV
    static constexpr Real K_to_eV = 8.617333262145e-5_rt;  // Conversion factor from Kelvin to eV

    static constexpr int  npoints = 4000;
    // x = (E - E_ion) / T, to prevent overflows
    static constexpr Real x_max   = 60.0_rt;

    // Stands for log(0): a stage that cannot ionise.  Finite, so that
    // interpolating between two such nodes never produces 0 * inf = NaN.
    static constexpr Real ln_rate_floor = -1.0e4_rt;

    // Elementary charge in CGS. Used for the Coulomb temperature floor.
    static constexpr Real q_elem_cgs = 4.803204712570263e-10_rt;

    // Largest / smallest exponent we can safely feed to std::exp() for the
    // active precision (double or single). Used purely as an overflow guard.
    AMREX_GPU_HOST_DEVICE inline
    Real log_huge()
    {
        constexpr Real ln2 = 0.6931471805599453_rt;
        return 0.99_rt * static_cast<Real>(std::numeric_limits<Real>::max_exponent) * ln2;
    }

    // ln R(T) of each stage, with R = k_ion(T) exp(+E_ion/T) the *reduced*
    // ionisation rate, on NT nodes uniform in ln T between T_min and T_max.
    inline AMREX_GPU_MANAGED Real ln_reduced_table[NumSpec][NT];

    // Reference energy of each stage's reduced rate: the lowest energy at
    // which its cross section is non-zero.
    inline AMREX_GPU_MANAGED Real E_ref_table[NumSpec];
    inline AMREX_GPU_MANAGED bool initialized = false; 
    // Maxwellian weight function
    AMREX_GPU_HOST_DEVICE inline 
    Real maxwell_weight(Real E, Real Te) 
    { 
        return std::sqrt(E) * std::exp(-E / Te); 
    } 
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
    // Ionisation energy of the stage ispe (eV).
    AMREX_GPU_HOST_DEVICE inline
    Real stage_ionization_energy(int ispe)
    {
        const int idx = get_energy_index(zatom[get_atom_index(ispe)])
                      + static_cast<int>(zion[ispe]);
        return ionization_energy[amrex::min(idx, 16)];
    }

    // ln of the reduced rate R = <sigma v> exp(E_ion/T), T and E in eV.
    AMREX_GPU_HOST_DEVICE 
    Real compute_log_reduced_rate(int atomic_number, int zstar, Real Te, Real E_ion)
    {
        const Real h = x_max / static_cast<Real>(npoints);
        Real sum = 0.0_rt;
        for (int i = 0; i <= npoints; i++) {
            const Real x = static_cast<Real>(i) * h;
            const Real E = E_ion + Te * x;
            const Real v = std::sqrt(2.0_rt * E * C::ev2erg / C::m_e);
            const Real wS = (i == 0 || i == npoints) ? 1.0_rt : ((i % 2) ? 4.0_rt : 2.0_rt);
            sum += wS * sigma_E(atomic_number, zstar, E) * v * std::sqrt(E) * std::exp(-x);
        }
        sum *= h / 3.0_rt;
        if (!(sum > 0.0_rt)) { return ln_rate_floor; }
        return std::log(2.0_rt / std::sqrt(M_PI) * sum / std::sqrt(Te));
    }

    // Lowest energy [eV] at which sigma_E(atomic_number, zstar, E) > 0, or a
    // negative value if the stage cannot ionise in the model.
    AMREX_GPU_HOST_DEVICE inline
    Real cross_section_threshold(int atomic_number, int zstar)
    {
        Real lo = 0.0_rt, hi = 2.0e3_rt;
        if (!(sigma_E(atomic_number, zstar, hi) > 0.0_rt)) { return -1.0_rt; }
        for (Real E = 1.0e-3_rt; E < hi; E *= 1.001_rt) {
            if (sigma_E(atomic_number, zstar, E) > 0.0_rt) { hi = E; break; }
        }
        for (int it = 0; it < 80; ++it) {
            const Real m = 0.5_rt * (lo + hi);
            if (sigma_E(atomic_number, zstar, m) > 0.0_rt) { hi = m; } else { lo = m; }
        }
        return hi;
    }

    // Maxwellian-averaged ionisation rate [cm^3/s] at Te [eV].
    AMREX_GPU_HOST_DEVICE inline
    Real compute_rate(int atomic_number, int zstar, Real Te)
    {
        const Real E_ref = cross_section_threshold(atomic_number, zstar);
        if (E_ref < 0.0_rt) { return 0.0_rt; }
        const Real lnR = compute_log_reduced_rate(atomic_number, zstar, Te, E_ref);
        const Real e = lnR - E_ref / Te;
        return (lnR <= ln_rate_floor || e < -log_huge()) ? 0.0_rt : std::exp(e);
    }

    // Precomputes ln R for all stages and temperatures
    inline void initialize()
    {
        if (initialized) return;
        for (int ispe = 0; ispe < NumSpec; ispe++) {
            const int Z = zatom[get_atom_index(ispe)];
            const int zstar = static_cast<int>(zion[ispe]);
            const Real E_ion = stage_ionization_energy(ispe);
            const Real thr = cross_section_threshold(Z, zstar);
            // snap to E_ion where the model is consistent, so that detailed
            // balance is exact there rather than good to the bisection
            E_ref_table[ispe] = (thr < 0.0_rt || std::abs(thr - E_ion) < 1.0e-9_rt * E_ion)
                              ? E_ion : thr;
            for (int it = 0; it < NT; it++) {
                const Real logT = std::log(T_min) + (std::log(T_max) - std::log(T_min)) * it / (NT - 1);
                ln_reduced_table[ispe][it] = (thr < 0.0_rt) ? ln_rate_floor
                    : compute_log_reduced_rate(Z, zstar, std::exp(logT), E_ref_table[ispe]);
            }
        }
        initialized = true;
    }

    // ln R of the direct (electron-impact) ionisation of stage ispe at Te [K],
    // R = k_ion exp(+E_ref/T), interpolated linearly in ln T.
    AMREX_GPU_HOST_DEVICE inline
    Real get_log_reduced_rate_direct(int ispe, Real Te)
    {
        const Real TeV = amrex::Clamp(Te * K_to_eV, T_min, T_max);
        const Real f = (std::log(TeV) - std::log(T_min))
                     / (std::log(T_max) - std::log(T_min)) * (NT - 1);
        const int  i = amrex::min(static_cast<int>(f), NT - 2);
        const Real w = f - static_cast<Real>(i);
        return (1.0_rt - w) * ln_reduced_table[ispe][i] + w * ln_reduced_table[ispe][i + 1];
    }

    // Direct ionisation rate [cm^3/s] of stage ispe at Te [K]
    AMREX_GPU_HOST_DEVICE inline
    Real get_rate_direct_ion(int ispe, Real Te)
    {
        const Real lnR = get_log_reduced_rate_direct(ispe, Te);
        if (lnR <= ln_rate_floor) { return 0.0_rt; }
        const Real TeV = amrex::max(Te * K_to_eV, 1.0e-300_rt);
        const Real e = lnR - E_ref_table[ispe] / TeV;
        return (e < -log_huge()) ? 0.0_rt : std::exp(e);
    }
    // Accesses the excitation ionization rate for a given atomic number and temperature
    // by interpolating the precomputed rate_table array
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

    // ln of the excitation channel's rate with its own maxwellian factor exp(-E_ex/T).
    AMREX_GPU_HOST_DEVICE inline
    Real get_log_reduced_rate_ex(int ispe, Real Te)
    {
        if (zion[ispe] != 0) { return ln_rate_floor; }
        Real TeV = Te * K_to_eV;
        if (TeV < T_min) { return ln_rate_floor; }
        if (TeV > T_max) { TeV = T_max; }
        const int idx = get_atom_index(ispe);
        const Real x = std::log(TeV);
        Real poly = 0.0_rt;
        for (int n = 0; n < 7; n++) { poly += gaunt_fit[idx][n] * std::pow(x, 6 - n); }
        return std::log(1.57e-7_rt * oscillator_strength[idx] / (ex_energy_atom[idx] * std::sqrt(TeV)))
             + poly;
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

    // 3-body recombination coefficient of stage ispe+1 -> ispe, as the
    // detailed-balance (Saha) inverse of the ionisation channels. 
    // channels: 1 = direct ionisation only, 2 = excitation channel only, 3 = both.
    AMREX_GPU_HOST_DEVICE inline
    Real get_rate_3body(int ispe, Real n_e, Real rho, Real Te, int channels = 3)
    {
        // Regularized evaluation temperature including Coulomb temperature
        const Real Te_eff = get_effective_temperature(Te, n_e);
        const Real Te_eff_eV = Te_eff * K_to_eV;
        const Real lnRd = (channels & 1)
            ? get_log_reduced_rate_direct(ispe, Te_eff)
              - amrex::max(E_ref_table[ispe] - stage_ionization_energy(ispe), 0.0_rt) / Te_eff_eV
            : ln_rate_floor;
        const Real lnRx = (channels & 2) ? get_log_reduced_rate_ex(ispe, Te_eff)     : ln_rate_floor;
        const Real mx = amrex::max(lnRd, lnRx);
        if (mx <= ln_rate_floor) { return 0.0_rt; }
        const Real lnR = mx + std::log(std::exp(lnRd - mx) + std::exp(lnRx - mx));

        // lambda_DeBroglie^3 = (2 pi hbar^2 / (m_e k_B Te))^{3/2}
        const Real log_lambda3 = 1.5_rt * std::log(2.0_rt * M_PI * C::hbar * C::hbar
                                                   / (C::m_e * C::k_B * Te_eff));

        const Real log_geom = std::log(rho) - std::log(C::m_u)
                            + std::log(g_deg[ispe]) - std::log(2.0_rt * g_deg[ispe + 1]);

        const Real log_rate = lnR + log_lambda3 + log_geom;
        if (log_rate < -log_huge()) return 0.0_rt;
        return std::exp(amrex::min(log_rate, log_huge()));
    }
} 

#endif // COLLISIONAL_IONIZATION_H
