#ifndef SPECIES_DIFFUSION_H
#define SPECIES_DIFFUSION_H

#include <AMReX.H>
#include <AMReX_Math.H>
#include <fundamental_constants.H>
#include <prob_parameters.H>
#include <collision_frequencies.H>

using namespace amrex;

// Analytical Fits for Momentum Transfer Cross Sections (m^2)
// Data from: Krstic, P.S., and Schultz, D.R. (1998) 
// Atomic and Plasma-Material Interaction Data for Fusion, Vol. 8

/**
 * @brief Computes the momentum transfer cross section for H+ + H
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real get_sigma_mt_Hp_H(amrex::Real T_h)
{
    // Convert T_h to CM Energy (eV)
    amrex::Real E_cm = SI::k_B * T_h / SI::q_e; 
    
    // STRICT CLAMP: Prevent evaluation near the 0.75 eV asymptote and 38.6 eV root
    E_cm = amrex::min(amrex::max(E_cm, 0.8_rt), 35.0_rt); 
    amrex::Real lnE = std::log(E_cm);

    // H+ + H Momentum Transfer Coefficients 
    const amrex::Real a0 =  46.9060_rt;
    const amrex::Real a1 =  21.3026_rt;
    const amrex::Real a2 =  -6.71354_rt;
    const amrex::Real a3 =  -0.783829_rt;
    
    const amrex::Real b1 =  3.60031_rt;
    const amrex::Real b2 =  0.685113_rt;
    const amrex::Real b3 =  0.174344_rt;

    amrex::Real num = a0 + a1*lnE + a2*std::pow(lnE, 2) + a3*std::pow(lnE, 3);
    amrex::Real den = 1.0_rt + b1*lnE + b2*std::pow(lnE, 2) + b3*std::pow(lnE, 3);
    
    amrex::Real sigma_au = num / den;

    // Convert from atomic units (a_0^2) to m^2 (1 a.u. = 2.80028e-21 m^2)
    return amrex::max(sigma_au * 2.80028e-21_rt, 1.0e-24_rt); 
}

/**
 * @brief Computes the momentum transfer cross section for H + H
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real get_sigma_mt_H_H(amrex::Real T_h)
{
    amrex::Real E_cm = SI::k_B * T_h / SI::q_e; 
    
    E_cm = amrex::max(E_cm, 1.0e-5_rt); 
    amrex::Real lnE = std::log(E_cm);

    // H + H Momentum Transfer Coefficients
    const amrex::Real a0 =  43.7445_rt;
    const amrex::Real a1 =  0.412813_rt;
    const amrex::Real a2 =  -1.89227_rt;
    const amrex::Real a3 =  0.429745_rt;
    
    const amrex::Real b1 =  0.256749_rt;
    const amrex::Real b2 =  0.166570_rt;
    const amrex::Real b3 =  0.00211056_rt;
    const amrex::Real b4 =  0.0919928_rt;
    const amrex::Real b5 =  0.0411435_rt;

    amrex::Real num = a0 + a1*lnE + a2*std::pow(lnE, 2) + a3*std::pow(lnE, 3);
    amrex::Real den = 1.0_rt + b1*lnE + b2*std::pow(lnE, 2) + b3*std::pow(lnE, 3) 
                      + b4*std::pow(lnE, 4) + b5*std::pow(lnE, 5);
    
    amrex::Real sigma_au = num / den;

    return amrex::max(sigma_au * 2.80028e-21_rt, 1.0e-24_rt); 
}


// Helper: Binary Collision Frequencies

/**
 * @brief Computes the heavy-particle collision frequency nu_ss'
 * * Uses appropriate physics based on electrostatic interactions:
 * - Neutral-Neutral: Exact H-H fit or Hard-sphere fallback
 * - Ion-Ion: Spitzer Coulomb scattering
 * - Ion-Neutral: Exact Hp-H fit or Langevin fallback
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real get_heavy_collision_frequency(int mu, int nu, amrex::Real T_h, amrex::Real n_nu)
{
    // Prevent self-collision frequency calculation
    if (mu == nu || n_nu <= 1e-20_rt) return 1.0e-30_rt;

    const amrex::Real m_mu = aion[get_atom_index(mu)] * C::m_u * 1e-3_rt; // Mass in kg
    const amrex::Real m_nu = aion[get_atom_index(nu)] * C::m_u * 1e-3_rt; 
    const amrex::Real m_r = (m_mu * m_nu) / (m_mu + m_nu); // Reduced mass

    const int z_mu = zion[mu];
    const int z_nu = zion[nu];
    
    // Simplistic mass check to see if dealing with Hydrogen (~1 amu)
    // Adjust logic if you have specific species index enums (e.g., SPEC_H, SPEC_HP)
    const bool is_mu_hydrogen = (std::abs(m_mu - 1.008_rt * C::m_u * 1e-3_rt) < 0.1_rt * C::m_u * 1e-3_rt);
    const bool is_nu_hydrogen = (std::abs(m_nu - 1.008_rt * C::m_u * 1e-3_rt) < 0.1_rt * C::m_u * 1e-3_rt);

    if (z_mu > 0 && z_nu > 0) {
        // 1. Ion-Ion: Spitzer Formulation
        const amrex::Real lnLambda = 10.0_rt;
        const amrex::Real q_mu2 = amrex::Math::powi<2>(z_mu * SI::q_e);
        const amrex::Real q_nu2 = amrex::Math::powi<2>(z_nu * SI::q_e);
        const amrex::Real denom = 3.0_rt * std::sqrt(m_r) * std::pow(SI::k_B * T_h, 1.5_rt) 
                                * amrex::Math::powi<2>(4.0_rt * M_PI * SI::epsilon0);
        return (4.0_rt * std::sqrt(2.0_rt * M_PI) * n_nu * q_mu2 * q_nu2 * lnLambda) / denom;
    } 
    else if (z_mu == 0 && z_nu == 0) {
        // 2. Neutral-Neutral
        amrex::Real sigma_nn;
        if (is_mu_hydrogen && is_nu_hydrogen) {
            sigma_nn = get_sigma_mt_H_H(T_h); // Quantum Mechanical Momentum Transfer
        } else {
            sigma_nn = 3.0e-20_rt; // Hard-Sphere Approximation fallback
        }
        return n_nu * sigma_nn * std::sqrt((8.0_rt * SI::k_B * T_h) / (M_PI * m_r));
    } 
    else {
        // 3. Ion-Neutral
        if (is_mu_hydrogen && is_nu_hydrogen && ((z_mu == 1 && z_nu == 0) || (z_mu == 0 && z_nu == 1))) {
            amrex::Real sigma_in = get_sigma_mt_Hp_H(T_h); // Quantum Mechanical Momentum Transfer
            return n_nu * sigma_in * std::sqrt((8.0_rt * SI::k_B * T_h) / (M_PI * m_r));
        } else {
            // Langevin Polarization fallback
            const amrex::Real alpha = 1.64e-40_rt; 
            const amrex::Real denom = amrex::Math::powi<2>(4.0_rt * M_PI * SI::epsilon0) * m_r;
            return 2.0_rt * M_PI * n_nu * std::sqrt((alpha * amrex::Math::powi<2>(SI::q_e)) / denom);
        }
    }
}

// Helper: Local Cell Diffusion Properties

struct DiffusionState {
    amrex::Real rho;
    amrex::Real X[NumSpec];      // Mass fractions
    amrex::Real x[NumSpec];      // Mole fractions
    amrex::Real D_m[NumSpec];    // Mixture-averaged diffusivities
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_diffusion_properties(int i, int j, int k, Array4<const Real> const& state, DiffusionState& ds)
{
    ds.rho = state(i,j,k,URHO);
    amrex::Real T_h = state(i,j,k,UTEMP); 
    
    amrex::Real n_mix = 0.0_rt;
    amrex::Real n_num[NumSpec] = {0.0_rt};

    // Step 1: Calculate Mass Fractions and Total SI Number Density
    for (int mu = 0; mu < NumSpec; ++mu) {
        ds.X[mu] = state(i,j,k,UFS+mu);
        
        // FIX: Safely convert CGS rho to SI number density (m^-3)
        // using the same logic you used in problem_source.H
        n_num[mu] = ds.X[mu] * ds.rho * aion_inv[mu] / C::m_u * 1.0e6_rt; 
        n_mix += n_num[mu];
    }

    // Protect against vacuum
    if (n_mix <= 1e-20_rt) {
        for (int mu = 0; mu < NumSpec; ++mu) {
            ds.x[mu] = 0.0_rt;
            ds.D_m[mu] = 0.0_rt;
        }
        return;
    }

    // Step 2 & 3: Global Mole Fractions & Binary Diffusivities
    amrex::Real D_bin[NumSpec][NumSpec];
    for (int mu = 0; mu < NumSpec; ++mu) {
        ds.x[mu] = n_num[mu] / n_mix;
        if (i ==0 && j == 0 && k == 0){
            std::cout << "x"<<mu << " : " << ds.x[mu] << std::endl;
        }
        for (int nu = 0; nu < NumSpec; ++nu) {
            if (mu == nu) {
                D_bin[mu][nu] = 0.0_rt;
                continue;   
            }
            amrex::Real nu_ss = get_heavy_collision_frequency(mu, nu, T_h, n_num[nu]);
            amrex::Real m_mu = aion[get_atom_index(mu)] * C::m_u * 1e-3_rt; // SI mass
            amrex::Real m_nu = aion[get_atom_index(nu)] * C::m_u * 1e-3_rt; // SI mass
            amrex::Real m_r = (m_mu * m_nu) / (m_mu + m_nu);
            
            D_bin[mu][nu] = (SI::k_B * T_h) / (m_r * nu_ss) * 1.0e-4_rt; // D is in m^2/s
            if (i ==0 && j == 0 && k == 0 && mu == 0){
                std::cout <<"(mu) "<< mu << "(nu) "<< nu << " (Dbin) "<< D_bin[mu][nu] << std::endl;
            }
        }
    }

    // Step 4: Mixture-Averaged Diffusivities (and convert to CGS)
    for (int mu = 0; mu < NumSpec; ++mu) {
        if (ds.X[mu] < 1.0e-10_rt) {
            ds.D_m[mu] = 0.0_rt; 
            continue;
        }
        
        amrex::Real S_den = 0.0_rt;
        for (int nu = 0; nu < NumSpec; ++nu) {
            if (mu != nu && ds.x[nu] > 1.0e-10_rt) {
                S_den += ds.x[nu] / (1e-10_rt + D_bin[mu][nu]);
            }
        }

        std::cout << "(mu) " << mu << " (S_den) " << S_den << std::endl;
        
        ds.D_m[mu] = ((1.0_rt - ds.X[mu]) / S_den);
        std::cout << "(mu)" << mu << "(Dmix)" << ds.D_m[mu] << std::endl;
    }
}

// Main Function: Species Diffusion Source Term
/**
 * @brief Returns the term Div(j_s) at point (i,j,k) for a target species `spec_target`
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real compute_species_diffusion_term(
    int i, int j, int k, int spec_target, 
    Array4<const Real> const& state, const Real* problo, const Real* dx, const int coord)
{
    amrex::Real species_diffusion_term = 0.0_rt;
    
    // Get state for center cell
    DiffusionState ds_c;
    get_diffusion_properties(i, j, k, state, ds_c);

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        int il = i; int jl = j; int kl = k;
        int ir = i; int jr = j; int kr = k;

        amrex::Real dxinv = 1.0_rt / dx[idir];

        if (idir == 0) {
            il = i - 1; ir = i + 1;
        }
#if AMREX_SPACEDIM >= 2
        else if (idir == 1) {
            jl = j - 1; jr = j + 1;
        }
#endif
#if AMREX_SPACEDIM == 3
        else {
            kl = k - 1; kr = k + 1;
        }
#endif

        // Evaluate properties at left and right neighbors
        DiffusionState ds_r, ds_l;
        get_diffusion_properties(ir, jr, kr, state, ds_r);
        get_diffusion_properties(il, jl, kl, state, ds_l);

        // -- Compute Flux Right (i + 1/2) --
        amrex::Real rho_r_edge = 0.5_rt * (ds_c.rho + ds_r.rho);
        amrex::Real X_target_r = 0.5_rt * (ds_c.X[spec_target] + ds_r.X[spec_target]);
        
        // Target species diagonal gradient: -D_m^mu * grad(x^mu)
        amrex::Real D_m_target_r = 0.5_rt * (ds_c.D_m[spec_target] + ds_r.D_m[spec_target]);
        amrex::Real grad_x_target_r = (ds_r.x[spec_target] - ds_c.x[spec_target]) * dxinv;
        amrex::Real flux_vec_r = -D_m_target_r * grad_x_target_r;

        // Add off-diagonal components: sum_{nu != mu} (D_m^nu * X^nu * grad(x^nu))
        for (int nu = 0; nu < NumSpec; ++nu) {
            if (nu != spec_target) {
                amrex::Real D_m_nu_r = 0.5_rt * (ds_c.D_m[nu] + ds_r.D_m[nu]);
                amrex::Real X_nu_r = 0.5_rt * (ds_c.X[nu] + ds_r.X[nu]);
                amrex::Real grad_x_nu_r = (ds_r.x[nu] - ds_c.x[nu]) * dxinv;
                flux_vec_r += D_m_nu_r * X_nu_r * grad_x_nu_r;
            }
        }
        amrex::Real j_flux_r = rho_r_edge * X_target_r * flux_vec_r;

        // -- Compute Flux Left (i - 1/2) --
        amrex::Real rho_l_edge = 0.5_rt * (ds_l.rho + ds_c.rho);
        amrex::Real X_target_l = 0.5_rt * (ds_l.X[spec_target] + ds_c.X[spec_target]);
        
        amrex::Real D_m_target_l = 0.5_rt * (ds_l.D_m[spec_target] + ds_c.D_m[spec_target]);
        amrex::Real grad_x_target_l = (ds_c.x[spec_target] - ds_l.x[spec_target]) * dxinv;
        amrex::Real flux_vec_l = -D_m_target_l * grad_x_target_l;

        for (int nu = 0; nu < NumSpec; ++nu) {
            if (nu != spec_target) {
                amrex::Real D_m_nu_l = 0.5_rt * (ds_l.D_m[nu] + ds_c.D_m[nu]);
                amrex::Real X_nu_l = 0.5_rt * (ds_l.X[nu] + ds_c.X[nu]);
                amrex::Real grad_x_nu_l = (ds_c.x[nu] - ds_l.x[nu]) * dxinv;
                flux_vec_l += D_m_nu_l * X_nu_l * grad_x_nu_l;
            }
        }
        amrex::Real j_flux_l = rho_l_edge * X_target_l * flux_vec_l;

#if AMREX_SPACEDIM < 3
        // Apply geometric terms for curvilinear coordinates
        if ((coord != 0) && (idir == 0)) {
            Real rr = problo[idir] + static_cast<Real>(ir) * dx[idir];
            Real rl = problo[idir] + static_cast<Real>(i) * dx[idir];
            Real rc = problo[idir] + (static_cast<Real>(i) + 0.5_rt) * dx[idir];

            if (coord == 1) { // Cylindrical
                j_flux_r *= rr;
                j_flux_l *= rl;
                dxinv *= 1.0_rt / rc;
            } else { // Spherical
                j_flux_r *= rr * rr;
                j_flux_l *= rl * rl;
                dxinv *= 1.0_rt / (rc * rc);
            }
        } else if ((coord == 2) && (idir == 1)) {
            Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
            Real thetar = problo[idir] + static_cast<Real>(jr) * dx[idir];
            Real thetal = problo[idir] + static_cast<Real>(j) * dx[idir];
            Real thetac = problo[idir] + (static_cast<Real>(j) + 0.5_rt) * dx[idir];

            j_flux_r *= std::sin(thetar);
            j_flux_l *= std::sin(thetal);
            dxinv *= 1.0_rt / (rc * rc * std::sin(thetac));
        }
#endif
        // Sum the divergence: (j_{i+1/2} - j_{i-1/2}) / dx
        species_diffusion_term += dxinv * (j_flux_r - j_flux_l);
    }

    return species_diffusion_term;
}

#endif // SPECIES_DIFFUSION_H