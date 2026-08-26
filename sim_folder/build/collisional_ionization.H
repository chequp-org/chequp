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
// Binding energies for each species (H, N, He, Ar)
inline AMREX_GPU_MANAGED amrex::Real binding_energy[17] = {
    13.5981_rt,
    20.33_rt, 20.33_rt, 14.534_rt, 14.534_rt, 14.534_rt,
    24.588_rt, 24.588_rt,
    422.45_rt, 29.3_rt, 29.3_rt, 15.76_rt, 15.76_rt, 15.76_rt, 15.76_rt, 15.76_rt, 15.76_rt
};
// Number of levels to consider per atomic species (H, N, He, Ar)
inline AMREX_GPU_MANAGED int num_simulated_electrons[num_atom_species] = {1, 5, 2, 9};
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int get_atom_index_from_Z(int Z)
{
    if (Z == 7)  return 1;
    if (Z == 2)  return 2;
    if (Z == 18) return 3;
    return 0;
}

namespace coll_ion 
{ 
    using namespace amrex; 
    
    static constexpr int NT = 100;  // Number of temperature points
    static constexpr Real T_min = 1e-3_rt;   // Minimum temperature
    static constexpr Real T_max = 3.0e2_rt;  // Maximum temperature
    static constexpr Real K_to_eV = 8.617333262145e-5_rt;  // Conversion factor from Kelvin to eV

    static constexpr int n_gauss_laguerre = 16;
    static constexpr Real sqrt_pi_over_2 = 0.8862269254527580_rt;  // \int_0^inf sqrt(x) e^{-x} dx

    // Elementary charge in esu (statC). Used for the Coulomb temperature floor.
    static constexpr Real q_elem_esu = 4.803204712570263e-10_rt;

    // Largest / smallest exponent we can safely feed to std::exp() for the
    // active precision (double or single). Used purely as an overflow guard.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real log_huge()
    {
        constexpr Real ln2 = 0.6931471805599453_rt;
        return 0.99_rt * static_cast<Real>(std::numeric_limits<Real>::max_exponent) * ln2;
    }

    inline AMREX_GPU_MANAGED Real rate_table[NumSpec][NT];
    inline AMREX_GPU_MANAGED bool initialized = false; 
    // Maxwellian weight function. Unused by compute_rate(), kept for external callers.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE 
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

        const int atom_idx = get_atom_index_from_Z(atomic_number);
        const int base_idx = get_energy_index(atomic_number); 

        const int max_k = num_simulated_electrons[atom_idx] - zstar; 
        if (max_k <= 0) return 0.0_rt;

        const Real me_c2 = 510998.0_rt; 
        const Real normalization = 2.0_rt * M_PI * std::pow(1.0_rt/137.0_rt, 4) * std::pow(5.29e-11_rt, 2); 

        const Real ep_mc2 = ep / me_c2; 

        // Outermost shell of the current charge state, not of the neutral atom
        const int valence_idx = base_idx + max_k - 1;
        const Real valence_energy = binding_energy[valence_idx];

        for (int k = 0; k < max_k; k++) { 
            int idx_k = base_idx + k; 
            int idx_k_p = base_idx + (k + 1); 

            Real bp_energy = binding_energy[idx_k];

            if (std::abs(bp_energy - valence_energy) < 1e-5_rt) {
                int idx_Eion = base_idx + zstar;
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
    // Ionization rate <sigma v> = \int sqrt(E) e^{-E/Te} sigma v dE / \int sqrt(E) e^{-E/Te} dE.
    // sigma vanishes below E_th, so E = E_th + u*Te turns the numerator into
    // Te e^{-E_th/Te} \int_0^inf e^{-u} g(u) du: the Gauss-Laguerre form, with the
    // Maxwellian tail carried analytically and the nodes scaling with Te.
    // The denominator is sqrt(pi)/2 * Te^{3/2}, so the Te factors cancel to 1/sqrt(Te).
    inline AMREX_GPU_HOST_DEVICE 
    Real compute_rate(int atomic_number, int zstar, Real Te) 
    { 
        // 16-point Gauss-Laguerre rule (weight e^{-x})
        static constexpr Real gl_x[n_gauss_laguerre] = {
            8.764941047892776e-02_rt, 4.626963289150804e-01_rt,
            1.141057774831226e+00_rt, 2.129283645098381e+00_rt,
            3.437086633893207e+00_rt, 5.078018614549768e+00_rt,
            7.070338535048234e+00_rt, 9.438314336391938e+00_rt,
            1.221422336886616e+01_rt, 1.544152736878162e+01_rt,
            1.918015685675314e+01_rt, 2.351590569399191e+01_rt,
            2.857872974288214e+01_rt, 3.458339870228662e+01_rt,
            4.194045264768833e+01_rt, 5.170116033954333e+01_rt
        };
        static constexpr Real gl_w[n_gauss_laguerre] = {
            2.061517149578049e-01_rt, 3.310578549508783e-01_rt,
            2.657957776442144e-01_rt, 1.362969342963787e-01_rt,
            4.732892869412563e-02_rt, 1.129990008033960e-02_rt,
            1.849070943526327e-03_rt, 2.042719153082809e-04_rt,
            1.484458687398150e-05_rt, 6.828319330871331e-07_rt,
            1.881024841079722e-08_rt, 2.862350242973897e-10_rt,
            2.127079033224121e-12_rt, 6.297967002517880e-15_rt,
            5.050473700035608e-18_rt, 4.161462370372851e-22_rt
        };

        if (Te <= 0.0_rt) return 0.0_rt;
        if (zstar < 0) return 0.0_rt;

        const int atom_idx = get_atom_index_from_Z(atomic_number);
        const int base_idx = get_energy_index(atomic_number);

        if (zstar >= num_simulated_electrons[atom_idx]) return 0.0_rt;

        const Real E_th = ionization_energy[base_idx + zstar];
        const Real arg = E_th / Te;

        if (arg > log_huge()) return 0.0_rt;   // Boltzmann factor underflows

        Real acc = 0.0_rt;
        for (int i = 0; i < n_gauss_laguerre; i++) {
            const Real E = E_th + gl_x[i] * Te;
            const Real sig = sigma_E(atomic_number, zstar, E);
            if (sig > 0.0_rt) {
                const Real v = std::sqrt(2.0_rt * E * C::ev2erg / C::m_e);
                acc += gl_w[i] * std::sqrt(E) * sig * v;
            }
        }

        return acc * std::exp(-arg) / (sqrt_pi_over_2 * std::sqrt(Te));
    } 
    // Precomputes the ionization rate for all species and temperatures
    // and stores them in the rate_table array to speed up the computation
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
    // Accesses the ionization rate for a given atomic number and temperature
    // by interpolating the precomputed rate_table array
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

    // Effective temperature used to regularize the 3-body recombination
    // rate at low Te.
    AMREX_GPU_HOST_DEVICE inline
    Real get_effective_temperature(Real Te, Real rho, Real abar)
    {
        const Real n_e = amrex::max(rho / (abar * C::m_p), 1.e-30_rt);

        // Wigner-Seitz radius and the associated Coulomb temperature
        const Real a_d = std::cbrt(3.0_rt / (4.0_rt * M_PI * n_e));
        const Real T_coulomb = (q_elem_esu * q_elem_esu / a_d) / C::k_B;

        return std::sqrt(Te * Te + T_coulomb * T_coulomb);
    }

    // Accesses the 3-body recombination rate for a given atomic number and
    // temperature with the Saha detailled balance.
    AMREX_GPU_HOST_DEVICE inline
    Real get_rate_3body(int ispe, Real rho, Real Te)
    {
        if (ispe + 1 >= NumSpec) return 0.0_rt;
        if (rho <= 0.0_rt) return 0.0_rt;

        // Regularized evaluation temperature including Coulomb temperature
        Real Te_eff = get_effective_temperature(Te, rho, aion[ispe]);
        const Real Te_eff_eV = Te_eff * K_to_eV;

        const Real rate_dir = get_rate_direct_ion(ispe, Te_eff);
        const Real rate_ex  = get_rate_ex_ion(ispe, Te_eff);

        const int idx_atom = get_atom_index(ispe);
        const int idx_Eion = get_energy_index(zatom[idx_atom]) + zion[ispe];

        // Assemble the product in log space
        const Real ln_tiny = -log_huge();

        // lambda_DeBroglie = hbar * sqrt(2 pi / (m_e k_B Te))
        const Real log_lambda3 = 1.5_rt * std::log(2.0_rt * M_PI * C::hbar * C::hbar
                                                   / (C::m_e * C::k_B * Te_eff));

        const Real log_geom = std::log(rho) - std::log(aion[ispe] * C::m_p)
                            + std::log(g_deg[ispe]) - std::log(2.0_rt * g_deg[ispe + 1]);

        const Real a = (rate_dir > 0.0_rt ? std::log(rate_dir) : ln_tiny)
                     + ionization_energy[idx_Eion] / Te_eff_eV;
        const Real b = (rate_ex > 0.0_rt ? std::log(rate_ex) : ln_tiny)
                     + ex_energy_atom[idx_atom] / Te_eff_eV;

        // log-sum-exp of the two forward channels
        const Real m = amrex::max(a, b);
        const Real log_rate_forward = m + std::log(std::exp(a - m) + std::exp(b - m));

        const Real log_rate = log_rate_forward + log_lambda3 + log_geom;

        // Final guard against floating-point overflow only.
        if (log_rate < ln_tiny) return 0.0_rt;
        return std::exp(amrex::min(log_rate, log_huge()));
    }
} 

#endif // COLLISIONAL_IONIZATION_H