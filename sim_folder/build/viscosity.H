#ifndef VISCOSITY_H
#define VISCOSITY_H

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <fundamental_constants.H>
#include <prob_parameters.H>

using namespace amrex;

namespace transport {

    /* =========================================================================
     * COLLISION CROSS-SECTIONS
     * ========================================================================= */

    /**
     * Evaluates a rational function fit for the Argon-Argon collision cross-section.
     * Takes the natural logarithm of energy in eV and returns the cross-section in m^2.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real get_sigma_m2_Ar_Ar(amrex::Real log_E_eV) {
        // Numerator and denominator coefficients for the rational fit
        const amrex::Real num[] = {-4.505866e+01_rt, -3.821852e+04_rt, -3.357927e+04_rt,
                                2.008318e+04_rt, 5.289070e+03_rt};
        const amrex::Real den[] = {8.493218e+02_rt, 7.254028e+02_rt, -4.605761e+02_rt, -1.039268e+02_rt};
        return eval_sigma_rational(log_E_eV, 3.453775e+00_rt, 3.405567e+00_rt, num, den);
    }


    /* =========================================================================
     * DYNAMIC VISCOSITY COMPUTATION
     * ========================================================================= */

    /**
     * Computes the mixture dynamic viscosity using the classical Spitzer (1962) 
     * prescription for the ion pool, with a hard-sphere fallback for neutral species.
     * Returns total dynamic viscosity in SI units (Pa*s).
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real compute_spitzer_viscosity(Real T_h, int n_active, const int* active_spec, const Real* n_s) 
    {
        using namespace problem; 

        // Accumulators for ion and electron properties
        Real n_ions = 0.0;
        Real sum_Z  = 0.0;
        Real sum_A  = 0.0;
        Real n_e    = 0.0;

        // Accumulators for neutral properties
        Real n_neutrals = 0.0;
        Real sum_A_neut = 0.0;

        // --- STEP 1: Gather average properties for the ion and neutral pools ---
        for (int i = 0; i < n_active; i++) {
            int s = active_spec[i];
            Real n_density = n_s[s];
            
            // Skip trace species to avoid division-by-zero or negligible computational overhead
            if (n_density < 1.0e-10) continue; 

            Real Z_s = zion[s]; // Species charge number
            Real A_s = aion[s]; // Species atomic mass

            if (Z_s > 0.0) {
                // Populate the ion/plasma pool
                n_ions += n_density;
                sum_Z  += n_density * Z_s;
                sum_A  += n_density * A_s;
                n_e    += n_density * Z_s; // Accumulate free electrons based on ionization state
            } 
            else if (Z_s == 0.0) {
                // Populate the neutral gas pool
                n_neutrals += n_density;
                sum_A_neut += n_density * A_s;
            }
        }

        Real mu_ion_si = 0.0;
        Real mu_neut_si = 0.0;

        // --- STEP 2: Compute Plasma/Ion Viscosity (Spitzer) ---
        if (n_ions > 1.0e-10 && n_e > 1.0e-10) {
            Real Z_avg = sum_Z / n_ions;
            Real A_avg = sum_A / n_ions;
            
            Real T_term = 8.8541878128e-12 * C::k_B * T_h;
            Real Lambda = 12.0 * M_PI * std::pow(T_term, 1.5) / 
                          (Z_avg * Z_avg * std::pow(C::q_e, 3.0) * std::sqrt(n_e));
            
            Real coulomb_log = amrex::max(std::log(Lambda), 2.0); 

            Real mu_cgs = 2.21e-15 * std::sqrt(A_avg) * std::pow(T_h, 2.5) / 
                          (std::pow(Z_avg, 4.0) * coulomb_log);

            // Convert CGS (Poise) to SI (Pa*s)
            mu_ion_si = mu_cgs * 0.1; 
        }

        // --- STEP 3: Compute Neutral Gas Viscosity Fallback ---
        if (n_neutrals > 1.0e-10_rt) {
            amrex::Real A_avg_neut = sum_A_neut / n_neutrals;
            amrex::Real m_neut = A_avg_neut * C::m_u;
            amrex::Real v_th = std::sqrt(8.0_rt * C::k_B * T_h / (M_PI * m_neut));
            amrex::Real E_bar_eV = (4.0_rt / M_PI) * C::k_B * T_h / C::q_e;
            amrex::Real log_E_eV = std::log(std::max(E_bar_eV, 1.0e-10_rt));
            amrex::Real sigma_nn = get_sigma_m2_Ar_Ar(log_E_eV);
            
            mu_neut_si = (C::k_B * T_h) / (v_th * sigma_nn); 
        }

        // --- STEP 4: Mole-Fraction Mixture Rule ---
        Real n_total = n_ions + n_neutrals + 1.0e-15; // Prevent div-by-zero
        Real x_ion = n_ions / n_total;
        Real x_neut = n_neutrals / n_total;
        
        Real mu_visc = (x_ion * mu_ion_si) + (x_neut * mu_neut_si);

        return amrex::max(150*mu_visc, 1.0e-15);

    }
    /* =========================================================================
     * VISCOUS SOURCE TERMS (MOMENTUM & ENERGY)
     * ========================================================================= */

    /**
     * Computes the divergence of the viscous stress tensor for momentum equations
     * (S_mx, S_my, S_mz) and the viscous heating term for the energy equation (S_visc).
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void compute_viscous_sources(int i, int j, int k,
                                Array4<const Real> const& state,
                                Real mu_visc,
                                const GeometryData& geomdata,
                                Real& S_mx, Real& S_my, Real& S_mz, Real& S_visc)
    {
        const Real* dx     = geomdata.CellSize();
        const Real* problo = geomdata.ProbLo();
        const int coord    = geomdata.Coord(); // 0 = Cartesian, 1 = R-Z (Cylindrical), 2 = Spherical

        // Spherical coordinates require a different geometric treatment of the stress tensor
        AMREX_ALWAYS_ASSERT(coord != 2); 

        // Compute local radius at cell center (only meaningful for R-Z coordinates where coord == 1)
        const Real r = problo[0] + (i + 0.5) * dx[0];

        // Helper lambda to fetch velocity components directly from the state array.
        // Note: Missing ghost cells will propagate NaNs here, which is expected behavior
        // to flag boundary condition errors upstream.
        auto get_vel = [&](int ii, int jj, int kk, int dir) -> Real {
            return (dir == 0) ? state(ii, jj, kk, UMX) :
                   (dir == 1) ? state(ii, jj, kk, UMY) :
                                state(ii, jj, kk, UMZ);
        };

        // Struct to hold all 9 components of the velocity gradient tensor (nabla_u)
        struct Grads {
            Real dudx = 0.0, dudy = 0.0, dudz = 0.0; // Gradients of u (x-velocity)
            Real dvdx = 0.0, dvdy = 0.0, dvdz = 0.0; // Gradients of v (y-velocity)
            Real dwdx = 0.0, dwdy = 0.0, dwdz = 0.0; // Gradients of w (z-velocity)
        };

        // Helper lambda: Computes velocity gradients at cell (ii, jj, kk) 
        // using 2nd-order central differencing.
        auto get_grads = [&](int ii, int jj, int kk) -> Grads {
            Grads g;
            
            // X-direction derivatives
            const Real inv_dx = 0.5 / dx[0];
            g.dudx = (get_vel(ii+1, jj, kk, 0) - get_vel(ii-1, jj, kk, 0)) * inv_dx;
            g.dvdx = (get_vel(ii+1, jj, kk, 1) - get_vel(ii-1, jj, kk, 1)) * inv_dx;
            g.dwdx = (get_vel(ii+1, jj, kk, 2) - get_vel(ii-1, jj, kk, 2)) * inv_dx;

    #if (AMREX_SPACEDIM > 1)
            // Y-direction derivatives
            const Real inv_dy = 0.5 / dx[1];
            g.dudy = (get_vel(ii, jj+1, kk, 0) - get_vel(ii, jj-1, kk, 0)) * inv_dy;
            g.dvdy = (get_vel(ii, jj+1, kk, 1) - get_vel(ii, jj-1, kk, 1)) * inv_dy;
            g.dwdy = (get_vel(ii, jj+1, kk, 2) - get_vel(ii, jj-1, kk, 2)) * inv_dy;
    #endif

    #if (AMREX_SPACEDIM > 2)
            // Z-direction derivatives
            const Real inv_dz = 0.5 / dx[2];
            g.dudz = (get_vel(ii, jj, kk+1, 0) - get_vel(ii, jj, kk-1, 0)) * inv_dz;
            g.dvdz = (get_vel(ii, jj, kk+1, 1) - get_vel(ii, jj, kk-1, 1)) * inv_dz;
            g.dwdz = (get_vel(ii, jj, kk+1, 2) - get_vel(ii, jj, kk-1, 2)) * inv_dz;
    #endif
            return g;
        };

        // Struct to hold the symmetric components of the Newtonian stress tensor (tau)
        struct Tau {
            Real xx = 0.0, yy = 0.0, zz = 0.0; // Normal stresses
            Real xy = 0.0, xz = 0.0, yz = 0.0; // Shear stresses
            Real theta_theta = 0.0;            // Hoop stress (only for R-Z coords)
        };

        // Helper lambda: Computes the Newtonian stress tensor at cell (ii, jj, kk)
        // using the constitutive relation: tau = mu * [grad(u) + grad(u)^T - (2/3)*div(u)*I]
        auto compute_tau = [&](int ii, int jj, int kk) -> Tau {
            const Grads g = get_grads(ii, jj, kk);
            
            // Velocity divergence: div(u) = du/dx + dv/dy + dw/dz
            const Real div_u = g.dudx + g.dvdy + g.dwdz;

            Tau t;
            // Normal viscous stresses
            t.xx = mu_visc * (2.0 * g.dudx - (2.0/3.0) * div_u);
            t.yy = mu_visc * (2.0 * g.dvdy - (2.0/3.0) * div_u);
            t.zz = mu_visc * (2.0 * g.dwdz - (2.0/3.0) * div_u);
            
            // Off-diagonal shear stresses (symmetric, so xy = yx)
            t.xy = mu_visc * (g.dudy + g.dvdx);
            t.xz = mu_visc * (g.dwdx + g.dudz);
            t.yz = mu_visc * (g.dvdz + g.dwdy);

            // Cylindrical coordinate geometry correction (Hoop Stress)
            if (coord != 0) { 
                const Real rr = problo[0] + (ii + 0.5) * dx[0]; // Local radius
                const Real u_r = get_vel(ii, jj, kk, 0);        // Radial velocity
                // Theta-theta component incorporates radial expansion (u_r / r)
                t.theta_theta = mu_visc * (2.0 * (u_r / rr) - (2.0/3.0) * div_u);
            }
            return t;
        };

        // --- STENCIL EVALUATION ---
        // Compute the stress tensor (tau) at the central cell and its immediate neighbors.
        // We do this to take the divergence of tau using central differences later.
        const Tau t_c   = compute_tau(i,   j,   k);
        const Tau t_ip1 = compute_tau(i+1, j,   k);
        const Tau t_im1 = compute_tau(i-1, j,   k);
        const Tau t_jp1 = compute_tau(i, j+1,   k);
        const Tau t_jm1 = compute_tau(i, j-1,   k);
    #if (AMREX_SPACEDIM > 2)
        const Tau t_kp1 = compute_tau(i, j, k+1);
        const Tau t_km1 = compute_tau(i, j, k-1);
    #endif

        // Inverse grid spacings for calculating the divergence of the stress tensor
        const Real inv_dx = 0.5 / dx[0];
        const Real inv_dy = (AMREX_SPACEDIM > 1) ? 0.5 / dx[1] : 0.0;
        const Real inv_dz = (AMREX_SPACEDIM > 2) ? 0.5 / dx[2] : 0.0;

        // --- MOMENTUM SOURCE TERMS --- 
        // Divergence of the stress tensor: S_m = div(tau)

        // 1. X-Momentum Source: d(tau_xx)/dx + d(tau_xy)/dy + d(tau_xz)/dz
        S_mx = (t_ip1.xx - t_im1.xx) * inv_dx
             + (t_jp1.xy - t_jm1.xy) * inv_dy;
    #if (AMREX_SPACEDIM > 2)
        S_mx += (t_kp1.xz - t_km1.xz) * inv_dz;
    #endif
        if (coord != 0) {
            // R-Z geometric correction for radial momentum divergence.
            // In cylindrical coords, div(tau)_r = (1/r) d(r*tau_xx)/dr - tau_theta_theta/r
            // Expanding by product rule gives: tau_xx/r + d(tau_xx)/dr - tau_theta_theta/r.
            // The d(tau_xx)/dr term is already handled by S_mx above.
            S_mx += (t_c.xx - t_c.theta_theta) / r;
        }

        // 2. Y-Momentum Source: d(tau_xy)/dx + d(tau_yy)/dy + d(tau_yz)/dz
        S_my = (t_ip1.xy - t_im1.xy) * inv_dx
             + (t_jp1.yy - t_jm1.yy) * inv_dy;
    #if (AMREX_SPACEDIM > 2)
        S_my += (t_kp1.yz - t_km1.yz) * inv_dz;
    #endif
        if (coord != 0) {
            // R-Z geometric correction for theta momentum divergence.
            // Adds the remaining (1/r) d(r*tau_xy)/dr term not covered by linear difference.
            S_my += t_c.xy / r; 
        }

        // 3. Z-Momentum Source: d(tau_xz)/dx + d(tau_yz)/dy + d(tau_zz)/dz
    #if (AMREX_SPACEDIM > 2)
        S_mz = (t_ip1.xz - t_im1.xz) * inv_dx
             + (t_jp1.yz - t_jm1.yz) * inv_dy
             + (t_kp1.zz - t_km1.zz) * inv_dz;
        if (coord != 0) {
            S_mz += t_c.xz / r;
        }
    #else
        S_mz = 0.0;
    #endif

        // --- ENERGY SOURCE TERM ---
        // Viscous heating (dissipation function): Phi = tau : grad(u)
        // This is a double dot product of the stress tensor and velocity gradient tensor
        // evaluated purely at the cell center (using the previously computed t_c and gc).
        const Grads gc = get_grads(i, j, k);
        S_visc = t_c.xx * gc.dudx + t_c.yy * gc.dvdy + t_c.zz * gc.dwdz
               + t_c.xy * (gc.dudy + gc.dvdx)
               + t_c.xz * (gc.dudz + gc.dwdx)
               + t_c.yz * (gc.dvdz + gc.dwdy);
               
        if (coord != 0) {
            const Real u_r = get_vel(i, j, k, 0);
            // Include hoop-stress heating contribution for cylindrical expansion
            S_visc += t_c.theta_theta * (u_r / r); 
        }
    }

} // namespace transport

#endif // VISCOSITY_H