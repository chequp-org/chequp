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
        Real n_e    = 0.0; // Track local electron density

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
                n_e    += n_density * Z_s; // Accumulate free electrons
            } else {
                n_neutrals += n_density;
                sum_A_neut += n_density * A_s;
            }
        }

        Real mu_visc = 0.0;

        // --- 2. Plasma/Ion Viscosity (Spitzer) ---
        if (n_ions > 1.0e-10 && n_e > 1.0e-10) {
            Real Z_avg = sum_Z / n_ions;
            Real A_avg = sum_A / n_ions;
            
            // Calculate dynamic Coulomb Logarithm (ln Lambda)
            // Vacuum permittivity eps_0 = 8.8541878128e-12 F/m
            Real T_term = 8.8541878128e-12 * C::k_B * T_h;
            Real Lambda = 12.0 * M_PI * std::pow(T_term, 1.5) / 
                          (Z_avg * Z_avg * std::pow(C::q_e, 3.0) * std::sqrt(n_e));
            
            // Protect against unphysical values in highly coupled (dense/cold) plasmas
            Real coulomb_log = amrex::max(std::log(Lambda), 2.0); 

            // Equation 23.5 yields viscosity in CGS units (gm cm^-1 s^-1, also known as Poise)
            Real mu_cgs = 2.21e-15 * std::sqrt(A_avg) * std::pow(T_h, 2.5) / 
                          (std::pow(Z_avg, 4.0) * coulomb_log);

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
        return amrex::max(3.0 * mu_visc, 1.0e-15);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void compute_viscous_sources(int i, int j, int k,
                                Array4<const Real> const& state,
                                Real mu_visc,
                                const GeometryData& geomdata,
                                Real& S_mx, Real& S_my, Real& S_mz, Real& S_visc)
    {
        const Real* dx     = geomdata.CellSize();
        const Real* problo = geomdata.ProbLo();
        const int coord     = geomdata.Coord(); // 0 = Cartesian, 1 = R-Z, 2 = Spherical

        AMREX_ALWAYS_ASSERT(coord != 2); // Spherical not yet implemented -- see note below

        // Radius at cell center (only meaningful for coord != 0)
        const Real r = problo[0] + (i + 0.5) * dx[0];

        // Plain velocity fetch. No NaN-masking: if state has NaNs, that is a bug
        // upstream (e.g. uninitialized ghost cells) that should be fixed there,
        // not hidden here. Fill ghost cells properly instead.
        auto get_vel = [&](int ii, int jj, int kk, int dir) -> Real {
            return (dir == 0) ? state(ii, jj, kk, UMX) :
                (dir == 1) ? state(ii, jj, kk, UMY) :
                                state(ii, jj, kk, UMZ);
        };

        struct Grads {
            Real dudx = 0.0, dudy = 0.0, dudz = 0.0;
            Real dvdx = 0.0, dvdy = 0.0, dvdz = 0.0;
            Real dwdx = 0.0, dwdy = 0.0, dwdz = 0.0;
        };

        // Full 3D velocity gradient tensor at (ii,jj,kk).
        auto get_grads = [&](int ii, int jj, int kk) -> Grads {
            Grads g;
            const Real inv_dx = 0.5 / dx[0];
            g.dudx = (get_vel(ii+1, jj, kk, 0) - get_vel(ii-1, jj, kk, 0)) * inv_dx;
            g.dvdx = (get_vel(ii+1, jj, kk, 1) - get_vel(ii-1, jj, kk, 1)) * inv_dx;
            g.dwdx = (get_vel(ii+1, jj, kk, 2) - get_vel(ii-1, jj, kk, 2)) * inv_dx;

    #if (AMREX_SPACEDIM > 1)
            const Real inv_dy = 0.5 / dx[1];
            g.dudy = (get_vel(ii, jj+1, kk, 0) - get_vel(ii, jj-1, kk, 0)) * inv_dy;
            g.dvdy = (get_vel(ii, jj+1, kk, 1) - get_vel(ii, jj-1, kk, 1)) * inv_dy;
            g.dwdy = (get_vel(ii, jj+1, kk, 2) - get_vel(ii, jj-1, kk, 2)) * inv_dy;
    #endif

    #if (AMREX_SPACEDIM > 2)
            const Real inv_dz = 0.5 / dx[2];
            g.dudz = (get_vel(ii, jj, kk+1, 0) - get_vel(ii, jj, kk-1, 0)) * inv_dz;
            g.dvdz = (get_vel(ii, jj, kk+1, 1) - get_vel(ii, jj, kk-1, 1)) * inv_dz;
            g.dwdz = (get_vel(ii, jj, kk+1, 2) - get_vel(ii, jj, kk-1, 2)) * inv_dz;
    #endif
            return g;
        };

        struct Tau {
            Real xx = 0.0, yy = 0.0, zz = 0.0;
            Real xy = 0.0, xz = 0.0, yz = 0.0;
            Real theta_theta = 0.0; // only meaningful for coord != 0
        };

        // Newtonian stress tensor, including R-Z geometric hoop term.
        auto compute_tau = [&](int ii, int jj, int kk) -> Tau {
            const Grads g = get_grads(ii, jj, kk);
            const Real div_u = g.dudx + g.dvdy + g.dwdz;

            Tau t;
            t.xx = mu_visc * (2.0 * g.dudx - (2.0/3.0) * div_u);
            t.yy = mu_visc * (2.0 * g.dvdy - (2.0/3.0) * div_u);
            t.zz = mu_visc * (2.0 * g.dwdz - (2.0/3.0) * div_u);
            t.xy = mu_visc * (g.dudy + g.dvdx);
            t.xz = mu_visc * (g.dudz + g.dwdx);
            t.yz = mu_visc * (g.dvdz + g.dwdy);

            if (coord != 0) { // R-Z: hoop stress from u_r / r
                const Real rr = problo[0] + (ii + 0.5) * dx[0];
                const Real u_r = get_vel(ii, jj, kk, 0);
                t.theta_theta = mu_visc * (2.0 * (u_r / rr) - (2.0/3.0) * div_u);
            }
            return t;
        };

        // Compute tau once per stencil point and reuse everywhere.
        const Tau t_c   = compute_tau(i,   j,   k);
        const Tau t_ip1 = compute_tau(i+1, j,   k);
        const Tau t_im1 = compute_tau(i-1, j,   k);
        const Tau t_jp1 = compute_tau(i, j+1,   k);
        const Tau t_jm1 = compute_tau(i, j-1,   k);
    #if (AMREX_SPACEDIM > 2)
        const Tau t_kp1 = compute_tau(i, j, k+1);
        const Tau t_km1 = compute_tau(i, j, k-1);
    #endif

        const Real inv_dx = 0.5 / dx[0];
        const Real inv_dy = (AMREX_SPACEDIM > 1) ? 0.5 / dx[1] : 0.0;
        const Real inv_dz = (AMREX_SPACEDIM > 2) ? 0.5 / dx[2] : 0.0;

        // S_mx = d(tau_xx)/dx + d(tau_xy)/dy + d(tau_xz)/dz + geometric term
        S_mx = (t_ip1.xx - t_im1.xx) * inv_dx
            + (t_jp1.xy - t_jm1.xy) * inv_dy;
    #if (AMREX_SPACEDIM > 2)
        S_mx += (t_kp1.xz - t_km1.xz) * inv_dz;
    #endif
        if (coord != 0) {
            // R-Z divergence in cylindrical coords: (1/r) d(r*tau_xx)/dr - tau_theta_theta/r
            // = tau_xx/r + d(tau_xx)/dr - tau_theta_theta/r  (product rule, dr term already in S_mx)
            S_mx += (t_c.xx - t_c.theta_theta) / r;
        }

        // S_my = d(tau_xy)/dx + d(tau_yy)/dy + d(tau_yz)/dz [+ R-Z curvature term]
        S_my = (t_ip1.xy - t_im1.xy) * inv_dx
            + (t_jp1.yy - t_jm1.yy) * inv_dy;
    #if (AMREX_SPACEDIM > 2)
        S_my += (t_kp1.yz - t_km1.yz) * inv_dz;
    #endif
        if (coord != 0) {
            S_my += t_c.xy / r; // (1/r) d(r*tau_xy)/dr extra term beyond the dr derivative above
        }

        // S_mz = d(tau_xz)/dx + d(tau_yz)/dy + d(tau_zz)/dz [+ R-Z curvature term]
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

        // Viscous heating: tau : grad(u), using the already-computed tau_c and grads at (i,j,k)
        const Grads gc = get_grads(i, j, k);
        S_visc = t_c.xx * gc.dudx + t_c.yy * gc.dvdy + t_c.zz * gc.dwdz
            + t_c.xy * (gc.dudy + gc.dvdx)
            + t_c.xz * (gc.dudz + gc.dwdx)
            + t_c.yz * (gc.dvdz + gc.dwdy);
        if (coord != 0) {
            const Real u_r = get_vel(i, j, k, 0);
            S_visc += t_c.theta_theta * (u_r / r); // hoop-stress heating contribution
        }
    }

} // namespace transport

#endif // VISCOSITY_H