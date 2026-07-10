#ifndef VISCOSITY_H
#define VISCOSITY_H

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <fundamental_constants.H>
#include <prob_parameters.H>

using namespace amrex;

namespace transport {

    /**
     * Computes the mixture dynamic viscosity using the classical Spitzer (1962) 
     * prescription for the ion pool, with a hard-sphere fallback for neutral species.
     * Returns dynamic viscosity in SI units (Pa*s).
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real compute_spitzer_viscosity(Real T_h, int n_active, const int* active_spec, const Real* n_s) 
    {
        using namespace problem; 

        Real n_ions = 0.0;
        Real sum_Z  = 0.0;
        Real sum_A  = 0.0;

        Real n_neutrals = 0.0;
        Real sum_A_neut = 0.0;

        // 1. Gather average properties for the ion pool and neutral pool
        for (int i = 0; i < n_active; i++) {
            int s = active_spec[i];
            Real n_density = n_s[s];
            
            if (n_density < 1.0e-10) continue; // Skip trace species

            Real Z_s = zion[s];
            Real A_s = aion[s];

            if (Z_s > 0.0) {
                n_ions += n_density;
                sum_Z  += n_density * Z_s;
                sum_A  += n_density * A_s;
            } else {
                n_neutrals += n_density;
                sum_A_neut += n_density * A_s;
            }
        }

        Real mu_visc = 0.0;

        // --- 2. Plasma/Ion Viscosity (Spitzer) ---
        if (n_ions > 1.0e-10) {
            Real Z_avg = sum_Z / n_ions;
            Real A_avg = sum_A / n_ions;
            
            // Coulomb logarithm (typically ~10 in many laboratory/astrophysical plasmas)
            const Real coulomb_log = 10.0; 

            // Equation 23.5 yields viscosity in CGS units (gm cm^-1 s^-1, also known as Poise)
            Real mu_cgs = 2.21e-15 * std::sqrt(A_avg) * std::pow(T_h, 2.5) / 
                          (std::pow(Z_avg, 4) * coulomb_log);

            // Convert CGS (Poise) to SI (Pa*s) -> 1 Poise = 0.1 Pa*s
            mu_visc += mu_cgs * 0.1; 
        }

        // --- 3. Neutral Gas Viscosity Fallback ---
        if (n_neutrals > 1.0e-10) {
            Real A_avg_neut = sum_A_neut / n_neutrals;
            Real m_neut = A_avg_neut * C::m_u;
            
            // Simple hard-sphere collision cross-section fallback (~30 Angstroms^2)
            const Real sigma_nn = 3.0e-19; 
            Real v_th = std::sqrt(8.0 * C::k_B * T_h / (M_PI * m_neut));
            
            // Baseline neutral viscosity approximation
            Real mu_neut = (C::k_B * T_h) / (v_th * sigma_nn); 
            
            mu_visc += mu_neut;
        }

        // Impose a tiny floor to prevent solver crashes from absolute zero viscosity
        return amrex::max(mu_visc, 1.0e-15);
    }

    // --- Explicit Viscous Momentum Sources & Heating ---
    // (This remains entirely unchanged from the previous setup)
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void compute_viscous_sources(int i, int j, int k,
                                 Array4<const Real> const& state,
                                 Real mu_visc,
                                 const Real* dx,
                                 Real& S_mx, Real& S_my, Real& S_mz, Real& S_visc)
    {
        const Real idx2 = 1.0 / (2.0 * dx[0]);
        const Real idy2 = 1.0 / (2.0 * dx[1]);
        const Real idz2 = 1.0 / (2.0 * dx[2]);
        
        const Real idx_sq = 1.0 / (dx[0] * dx[0]);
        const Real idy_sq = 1.0 / (dx[1] * dx[1]);
        const Real idz_sq = 1.0 / (dx[2] * dx[2]);

        auto get_vel = [&](int ii, int jj, int kk, int dir) {
            return state(ii, jj, kk, UMX + dir) / state(ii, jj, kk, URHO);
        };

        Real u_c = get_vel(i, j, k, 0);
        Real v_c = get_vel(i, j, k, 1);
        Real w_c = get_vel(i, j, k, 2);

        // 1. First Derivatives
        Real dudx = (get_vel(i+1, j, k, 0) - get_vel(i-1, j, k, 0)) * idx2;
        Real dudy = (get_vel(i, j+1, k, 0) - get_vel(i, j-1, k, 0)) * idy2;
        Real dudz = (get_vel(i, j, k+1, 0) - get_vel(i, j, k-1, 0)) * idz2;

        Real dvdx = (get_vel(i+1, j, k, 1) - get_vel(i-1, j, k, 1)) * idx2;
        Real dvdy = (get_vel(i, j+1, k, 1) - get_vel(i, j-1, k, 1)) * idy2;
        Real dvdz = (get_vel(i, j, k+1, 1) - get_vel(i, j, k-1, 1)) * idz2;

        Real dwdx = (get_vel(i+1, j, k, 2) - get_vel(i-1, j, k, 2)) * idx2;
        Real dwdy = (get_vel(i, j+1, k, 2) - get_vel(i, j-1, k, 2)) * idy2;
        Real dwdz = (get_vel(i, j, k+1, 2) - get_vel(i, j, k-1, 2)) * idz2;

        Real div_u = dudx + dvdy + dwdz;

        // 2. Stress Tensor & Viscous Heating
        Real tau_xx = mu_visc * (2.0 * dudx - (2.0/3.0) * div_u);
        Real tau_yy = mu_visc * (2.0 * dvdy - (2.0/3.0) * div_u);
        Real tau_zz = mu_visc * (2.0 * dwdz - (2.0/3.0) * div_u);
        Real tau_xy = mu_visc * (dudy + dvdx);
        Real tau_xz = mu_visc * (dudz + dwdx);
        Real tau_yz = mu_visc * (dvdz + dwdy);

        S_visc = tau_xx * dudx + tau_yy * dvdy + tau_zz * dwdz + 
                 tau_xy * (dudy + dvdx) + tau_xz * (dudz + dwdx) + tau_yz * (dvdz + dwdy);

        // 3. Second Derivatives (Laplacians)
        Real lap_u = (get_vel(i+1,j,k,0) - 2.0*u_c + get_vel(i-1,j,k,0)) * idx_sq +
                     (get_vel(i,j+1,k,0) - 2.0*u_c + get_vel(i,j-1,k,0)) * idy_sq +
                     (get_vel(i,j,k+1,0) - 2.0*u_c + get_vel(i,j,k-1,0)) * idz_sq;

        Real lap_v = (get_vel(i+1,j,k,1) - 2.0*v_c + get_vel(i-1,j,k,1)) * idx_sq +
                     (get_vel(i,j+1,k,1) - 2.0*v_c + get_vel(i,j-1,k,1)) * idy_sq +
                     (get_vel(i,j,k+1,1) - 2.0*v_c + get_vel(i,j,k-1,1)) * idz_sq;

        Real lap_w = (get_vel(i+1,j,k,2) - 2.0*w_c + get_vel(i-1,j,k,2)) * idx_sq +
                     (get_vel(i,j+1,k,2) - 2.0*w_c + get_vel(i,j-1,k,2)) * idy_sq +
                     (get_vel(i,j,k+1,2) - 2.0*w_c + get_vel(i,j,k-1,2)) * idz_sq;

        // 4. Cross Derivatives for grad(div u)
        Real d2v_dxdy = (get_vel(i+1,j+1,k,1) - get_vel(i-1,j+1,k,1) - get_vel(i+1,j-1,k,1) + get_vel(i-1,j-1,k,1)) * (idx2 * idy2);
        Real d2w_dxdz = (get_vel(i+1,j,k+1,2) - get_vel(i-1,j,k+1,2) - get_vel(i+1,j,k-1,2) + get_vel(i-1,j,k-1,2)) * (idx2 * idz2);
        Real grad_div_u_x = (get_vel(i+1,j,k,0) - 2.0*u_c + get_vel(i-1,j,k,0)) * idx_sq + d2v_dxdy + d2w_dxdz;

        Real d2u_dxdy = (get_vel(i+1,j+1,k,0) - get_vel(i-1,j+1,k,0) - get_vel(i+1,j-1,k,0) + get_vel(i-1,j-1,k,0)) * (idx2 * idy2);
        Real d2w_dydz = (get_vel(i,j+1,k+1,2) - get_vel(i,j-1,k+1,2) - get_vel(i,j+1,k-1,2) + get_vel(i,j-1,k-1,2)) * (idy2 * idz2);
        Real grad_div_u_y = d2u_dxdy + (get_vel(i,j+1,k,1) - 2.0*v_c + get_vel(i,j-1,k,1)) * idy_sq + d2w_dydz;

        Real d2u_dxdz = (get_vel(i+1,j,k+1,0) - get_vel(i-1,j,k+1,0) - get_vel(i+1,j,k-1,0) + get_vel(i-1,j,k-1,0)) * (idx2 * idz2);
        Real d2v_dydz = (get_vel(i,j+1,k+1,1) - get_vel(i,j-1,k+1,1) - get_vel(i,j+1,k-1,1) + get_vel(i,j-1,k-1,1)) * (idy2 * idz2);
        Real grad_div_u_z = d2u_dxdz + d2v_dydz + (get_vel(i,j,k+1,2) - 2.0*w_c + get_vel(i,j,k-1,2)) * idz_sq;

        // 5. Assign Momentum Source Terms: S = mu * (laplacian(u) + 1/3 * grad(div u))
        S_mx = mu_visc * (lap_u + (1.0/3.0) * grad_div_u_x);
        S_my = mu_visc * (lap_v + (1.0/3.0) * grad_div_u_y);
        S_mz = mu_visc * (lap_w + (1.0/3.0) * grad_div_u_z);
    }

} // namespace transport

#endif // VISCOSITY_H