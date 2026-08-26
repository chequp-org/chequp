#ifndef VISCOSITY_H
#define VISCOSITY_H

#include <fundamental_constants.H>
#include <prob_parameters.H>
#include <cmath>

using namespace amrex;

namespace transport {

    // =======================================================================
    // SI constants, declared locally.
    //
    // IMPORTANT: the collision-rate and viscosity fits below are all in SI.
    // Castro's C:: namespace is CGS (k_B in erg/K, q_e in esu, m_u in g), so
    // C::q_e / C::k_B would give ~3.5e6 "K/eV" instead of 11604.5.  Everything
    // in this namespace therefore uses these local SI values, and
    // compute_dynamic_viscosity() returns Pa*s.
    // =======================================================================
    namespace si {
        constexpr Real k_B   = 1.380649e-23_rt;      // J/K
        constexpr Real q_e   = 1.602176634e-19_rt;   // C
        constexpr Real m_e   = 9.1093837015e-31_rt;  // kg
        constexpr Real m_u   = 1.6605390666e-27_rt;  // kg
        constexpr Real eps_0 = 8.8541878128e-12_rt;  // F/m
    }

    constexpr Real K_PER_EV = si::q_e / si::k_B;              // ~11604.518 K/eV
    constexpr Real m_Ar     = 39.948_rt * si::m_u;            // Ar atom mass [kg]

    // electron-Ar reduced mass (mu_en_Ar in the python script)
    constexpr Real mu_en_Ar = si::m_e * m_Ar / (si::m_e + m_Ar);

    // Electron-screening coefficient (GAMMA_AR in the python script)
    constexpr Real GAMMA_AR = 3.0_rt;

    // Coulomb-log floor (LAMBDA_FLOOR in the python script)
    constexpr Real LAMBDA_FLOOR = 2.0_rt;

    // Floor on sum_{alpha'} nu_{Ar,alpha'} [s^-1]  (python: np.maximum(..., 1e-3))
    constexpr Real NU_H_FLOOR = 1.0e-3_rt;

    // =======================================================================
    // Heavy-heavy collision rate coefficients K_hg(T_h) = <Q v_rel> [m^3/s]
    // Fits from Phelps 1991 / Phelps 1994.
    // =======================================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real get_collision_rate_coeff(int species_idx, Real T_h_eV) {
        Real T_safe = amrex::max(T_h_eV, 1.0e-3_rt);

        if (species_idx == 0) { // Neutral Argon (Ar0): cross section Q00(T_K)
            const Real B0 = -3.83003443e+01_rt;
            const Real B1 = -1.07765213e+00_rt;
            const Real B2 =  1.22411293e-01_rt;
            const Real B3 = -1.16327055e-02_rt;
            const Real B4 =  4.15451004e-04_rt;

            Real T_K   = amrex::max(T_safe * K_PER_EV, 100.0_rt);
            Real log_T = std::log(T_K);
            Real log_Q = B0 + log_T * (B1 + log_T * (B2 + log_T * (B3 + log_T * B4)));
            Real Q00   = std::exp(log_Q);

            Real v_rel = std::sqrt(16.0_rt * si::q_e * T_safe / (M_PI * m_Ar));
            return Q00 * v_rel;
        }
        else if (species_idx == 1) { // Singly ionized Argon (Ar+)
            const Real C0 = -3.31849900e+01_rt;
            const Real C1 =  4.14999301e-01_rt;
            const Real C2 =  1.49955220e-02_rt;
            const Real C3 = -3.67722064e-03_rt;
            const Real C4 = -1.18958350e-03_rt;

            Real log_T = std::log(T_safe);
            Real log_K = C0 + log_T * (C1 + log_T * (C2 + log_T * (C3 + log_T * C4)));
            return std::exp(log_K);
        }
        else { // Doubly ionized Argon (Ar2+)
            const Real D0 = -3.39489334e+01_rt;
            const Real D1 =  3.63714701e-02_rt;
            const Real D2 =  1.68314237e-02_rt;
            const Real D3 =  3.39500066e-03_rt;
            const Real D4 = -6.09201455e-04_rt;

            Real log_T = std::log(T_safe);
            Real log_K = D0 + log_T * (D1 + log_T * (D2 + log_T * (D3 + log_T * D4)));
            return std::exp(log_K);
        }
    }

    // =======================================================================
    // Electron-neutral momentum-transfer cross section sigma_m(E) [m^2]
    // Rational (Pade) fit in log(E/eV); == get_sigma_m2_Ar() in python.
    // =======================================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real get_sigma_m2_Ar(Real log_E_eV) {
        constexpr Real offset = 2.128350e+00_rt;
        constexpr Real scale  = 6.152029e+00_rt;

        constexpr Real N0 = -4.341181e+01_rt;
        constexpr Real N1 =  6.670171e+01_rt;
        constexpr Real N2 = -4.380806e+02_rt;
        constexpr Real N3 = -2.230848e+03_rt;
        constexpr Real N4 = -2.086728e+03_rt;

        constexpr Real D1 = -1.425189e+00_rt;
        constexpr Real D2 =  9.207079e+00_rt;
        constexpr Real D3 =  4.881694e+01_rt;
        constexpr Real D4 =  4.621947e+01_rt;

        Real x = (log_E_eV - offset) / scale;

        Real num = N0 + x * (N1 + x * (N2 + x * (N3 + x * N4)));
        Real den = 1.0_rt + x * (D1 + x * (D2 + x * (D3 + x * D4)));

        return std::exp(num / den);
    }

    /**
     * Electron-neutral momentum transfer frequency nu_{e,Ar} [s^-1].
     * python: compute_nu_eAr()
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real compute_nu_eAr(Real n_Ar0, Real T_e_eV) {
        Real T_e_safe = amrex::max(T_e_eV, 1.0e-4_rt);

        Real E_bar_eV = (4.0_rt / M_PI) * T_e_safe;   // mean energy [eV]
        Real E_bar    = E_bar_eV * si::q_e;           // mean energy [J]

        Real sigma = get_sigma_m2_Ar(std::log(E_bar_eV));

        return amrex::max(n_Ar0, 0.0_rt) * (4.0_rt / 3.0_rt) *
               std::sqrt(2.0_rt * E_bar / mu_en_Ar) * sigma;
    }

    /**
     * Electron-ion Coulomb logarithm.  python: coulomb_log_ei()
     * NOTE: no Z-dependence, exactly as in the python script.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real coulomb_log_ei(Real n_e, Real T_e_eV, Real T_h_eV) {
        Real kT_e     = amrex::max(T_e_eV, 1.0e-4_rt) * si::q_e;   // k_B T_e [J]
        Real n_e_safe = amrex::max(n_e, 1.0e10_rt);
        Real T_ratio  = amrex::max(T_e_eV, 1.0e-4_rt) / amrex::max(T_h_eV, 1.0e-4_rt);

        Real A  = 4.0_rt * M_PI * si::eps_0 * kT_e;
        Real q2 = si::q_e * si::q_e;
        Real q6 = q2 * q2 * q2;

        Real inside = (A * A * A) / (q6 * n_e_safe * (1.0_rt + T_ratio));

        Real Lambda = std::log((3.0_rt / (2.0_rt * std::sqrt(M_PI))) *
                               std::sqrt(amrex::max(inside, 1.0e-100_rt)));

        return amrex::max(Lambda, LAMBDA_FLOOR);
    }

    /**
     * Electron-ion momentum transfer frequency nu_{e,ion} [s^-1].
     * python: compute_nu_eArplus()  ->  C * Z^2 * n_ion
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real compute_nu_eIon(Real n_ion, Real n_e, Real T_e_eV, Real T_h_eV, Real Z) {
        Real kT_e   = amrex::max(T_e_eV, 1.0e-4_rt) * si::q_e;
        Real Lambda = coulomb_log_ei(n_e, T_e_eV, T_h_eV);

        Real q2 = si::q_e * si::q_e;
        Real q4 = q2 * q2;

        Real four_pi_eps = 4.0_rt * M_PI * si::eps_0;

        Real Cfac = (4.0_rt / 3.0_rt) * std::sqrt(2.0_rt * M_PI / si::m_e) *
                    (q4 * Lambda) /
                    ((four_pi_eps * four_pi_eps) * (kT_e * std::sqrt(kT_e)));

        return Cfac * Z * Z * amrex::max(n_ion, 0.0_rt);
    }

    /**
     * Dynamic viscosity of the Ar (neutral) heavy species with thermal
     * electron screening [Pa*s].  Densities n_s in m^-3, temperatures in K.
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real compute_dynamic_viscosity(Real T_h_K, int n_active, const int* active_spec,
                                   const Real* n_s, Real T_e_K = -1.0_rt) {
        using namespace problem;

        Real T_h_eV = amrex::max(T_h_K / K_PER_EV, 1.0e-4_rt);
        Real T_e_eV = (T_e_K > 0.0_rt) ? amrex::max(T_e_K / K_PER_EV, 1.0e-4_rt) : T_h_eV;

        Real n_Ar0 = 0.0_rt;
        Real n_Ar1 = 0.0_rt;
        Real n_Ar2 = 0.0_rt;

        for (int i = 0; i < n_active; i++) {
            int s = active_spec[i];
            Real density = amrex::max(n_s[s], 0.0_rt);
            Real Z_s = zion[s];

            if (Z_s <= 1.0e-12_rt) {
                n_Ar0 += density;
            } else if (std::abs(Z_s - 1.0_rt) < 0.1_rt) {
                n_Ar1 += density;
            } else if (Z_s >= 1.9_rt) {
                n_Ar2 += density;
            }
        }

        Real n_e = amrex::max(n_Ar1 + 2.0_rt * n_Ar2, 1.0e10_rt);

        // Electron collision frequencies and the screening factor K_nu
        Real nu_eAr  = compute_nu_eAr(n_Ar0, T_e_eV);
        Real nu_eIon = compute_nu_eIon(n_Ar1, n_e, T_e_eV, T_h_eV, 1.0_rt)
                     + compute_nu_eIon(n_Ar2, n_e, T_e_eV, T_h_eV, 2.0_rt);

        Real sum_nu_e = nu_eAr + nu_eIon;
        Real ratio    = (sum_nu_e > 0.0_rt) ? (nu_eIon / sum_nu_e) : 0.0_rt;

        Real K_nu = 1.0_rt / (1.0_rt + GAMMA_AR * ratio);

        // Heavy-heavy collision frequencies nu_{Ar,alpha'} [s^-1]
        Real nu_Ar_Ar   = n_Ar0 * get_collision_rate_coeff(0, T_h_eV);
        Real nu_Ar_Arp  = n_Ar1 * get_collision_rate_coeff(1, T_h_eV);
        Real nu_Ar_Ar2p = n_Ar2 * get_collision_rate_coeff(2, T_h_eV);

        Real sum_nu_Ar = amrex::max(nu_Ar_Ar + nu_Ar_Arp + nu_Ar_Ar2p, NU_H_FLOOR);

        // mu_factor: k_B^2 * T[K] written as k_B * q_e * T[eV] (exactly equal)
        Real mu_factor = (15.0_rt / (2.0_rt * M_PI)) *
                         (si::k_B * si::q_e * T_h_eV * n_Ar0) /
                         (m_Ar * sum_nu_Ar);

        Real mu_H = 1.2e-2 * K_nu * mu_factor;

        return amrex::max(mu_H, 1.0e-20_rt);
    }

    // Any cell whose density is below this (or non-finite) is treated as
    // having no valid velocity.  Set this to something comparable to your
    // castro.small_dens -- the old 1e-30 floor was ~30 orders below the
    // physical density and converted garbage momentum into ~1e51 cm/s.
    constexpr Real RHO_FLOOR_VISC = 1.0e-20_rt;

    /**
     * Viscous stress divergence and viscous dissipation.
     *
     *   S_m    = div(tau)             -> momentum source
     *   S_visc = tau : grad(u) >= 0   -> internal-energy source
     *
     * mu_visc must be in the same unit system as the state (poise for a CGS
     * Castro state: multiply the SI value by PAS_TO_POISE).
     *
     * All stencil indices are clamped to the bounds of the Array4 itself, so
     * the routine degrades to a one-sided difference at the edge of the
     * available data instead of reading unfilled ghost cells.  Filling one
     * ghost cell is still strongly preferred: the one-sided form is only
     * first-order and does not represent a physical wall/outflow condition.
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
        const int coord    = geomdata.Coord();

        AMREX_ALWAYS_ASSERT(coord != 2);

        S_mx = 0.0_rt; S_my = 0.0_rt; S_mz = 0.0_rt; S_visc = 0.0_rt;

        const Real r = problo[0] + (i + 0.5_rt) * dx[0];

        // ---- bounds of the data we are actually allowed to touch ----
        const int lo_x = state.begin.x, hi_x = state.end.x - 1;
        const int lo_y = state.begin.y, hi_y = state.end.y - 1;
        const int lo_z = state.begin.z, hi_z = state.end.z - 1;

        auto cx = [&](int ii) { return amrex::min(amrex::max(ii, lo_x), hi_x); };
        auto cy = [&](int jj) { return amrex::min(amrex::max(jj, lo_y), hi_y); };
        auto cz = [&](int kk) { return amrex::min(amrex::max(kk, lo_z), hi_z); };
#if (AMREX_SPACEDIM < 2)
        amrex::ignore_unused(cy, lo_y, hi_y);
#endif
#if (AMREX_SPACEDIM < 3)
        amrex::ignore_unused(cz, lo_z, hi_z);
#endif

        // ---- is this cell's velocity usable at all? ----
        auto cell_ok = [&](int ii, int jj, int kk) -> bool {
            const Real rho_l = state(ii, jj, kk, URHO);
            return (rho_l > RHO_FLOOR_VISC) && std::isfinite(rho_l)
                && std::isfinite(state(ii, jj, kk, UMX))
                && std::isfinite(state(ii, jj, kk, UMY))
                && std::isfinite(state(ii, jj, kk, UMZ));
        };

        // UMX..UMZ are momentum densities -- divide by rho to get velocity.
        auto vel_raw = [&](int ii, int jj, int kk, int dir) -> Real {
            if (!cell_ok(ii, jj, kk)) return 0.0_rt;
            const Real rho_l = state(ii, jj, kk, URHO);
            const Real mom   = (dir == 0) ? state(ii, jj, kk, UMX) :
                               (dir == 1) ? state(ii, jj, kk, UMY) :
                                            state(ii, jj, kk, UMZ);
            return mom / rho_l;
        };

        if (!cell_ok(i, j, k)) return;   // nothing sensible to do in this cell

        struct Grads {
            Real dudx = 0.0_rt, dudy = 0.0_rt, dudz = 0.0_rt;
            Real dvdx = 0.0_rt, dvdy = 0.0_rt, dvdz = 0.0_rt;
            Real dwdx = 0.0_rt, dwdy = 0.0_rt, dwdz = 0.0_rt;
            Real div  = 0.0_rt;   // full divergence, including u_r/r
        };

        auto get_grads = [&](int ii, int jj, int kk) -> Grads {
            Grads g;

            // A neighbour with no usable velocity falls back to this cell,
            // which yields a zero gradient rather than a spurious huge one.
            auto v = [&](int i2, int j2, int k2, int dir) -> Real {
                if (!cell_ok(i2, j2, k2)) return vel_raw(ii, jj, kk, dir);
                return vel_raw(i2, j2, k2, dir);
            };

            {
                const int ip = cx(ii+1), im = cx(ii-1);
                const Real inv = (ip > im) ? 1.0_rt / ((ip - im) * dx[0]) : 0.0_rt;
                g.dudx = (v(ip, jj, kk, 0) - v(im, jj, kk, 0)) * inv;
                g.dvdx = (v(ip, jj, kk, 1) - v(im, jj, kk, 1)) * inv;
                g.dwdx = (v(ip, jj, kk, 2) - v(im, jj, kk, 2)) * inv;
            }

#if (AMREX_SPACEDIM > 1)
            {
                const int jp = cy(jj+1), jm = cy(jj-1);
                const Real inv = (jp > jm) ? 1.0_rt / ((jp - jm) * dx[1]) : 0.0_rt;
                g.dudy = (v(ii, jp, kk, 0) - v(ii, jm, kk, 0)) * inv;
                g.dvdy = (v(ii, jp, kk, 1) - v(ii, jm, kk, 1)) * inv;
                g.dwdy = (v(ii, jp, kk, 2) - v(ii, jm, kk, 2)) * inv;
            }
#endif

#if (AMREX_SPACEDIM > 2)
            {
                const int kp = cz(kk+1), km = cz(kk-1);
                const Real inv = (kp > km) ? 1.0_rt / ((kp - km) * dx[2]) : 0.0_rt;
                g.dudz = (v(ii, jj, kp, 0) - v(ii, jj, km, 0)) * inv;
                g.dvdz = (v(ii, jj, kp, 1) - v(ii, jj, km, 1)) * inv;
                g.dwdz = (v(ii, jj, kp, 2) - v(ii, jj, km, 2)) * inv;
            }
#endif

            g.div = g.dudx + g.dvdy + g.dwdz;

            // Cylindrical / spherical: div(u) picks up u_r / r
            if (coord != 0) {
                const Real rr = problo[0] + (ii + 0.5_rt) * dx[0];
                g.div += vel_raw(ii, jj, kk, 0) / rr;
            }

            return g;
        };

        struct Tau {
            Real xx = 0.0_rt, yy = 0.0_rt, zz = 0.0_rt;
            Real xy = 0.0_rt, xz = 0.0_rt, yz = 0.0_rt;
            Real theta_theta = 0.0_rt;
        };

        auto compute_tau = [&](int ii, int jj, int kk) -> Tau {
            const Grads g = get_grads(ii, jj, kk);
            const Real div_u = g.div;

            Tau t;
            t.xx = mu_visc * (2.0_rt * g.dudx - (2.0_rt/3.0_rt) * div_u);
            t.yy = mu_visc * (2.0_rt * g.dvdy - (2.0_rt/3.0_rt) * div_u);
            t.zz = mu_visc * (2.0_rt * g.dwdz - (2.0_rt/3.0_rt) * div_u);

            t.xy = mu_visc * (g.dudy + g.dvdx);
            t.xz = mu_visc * (g.dwdx + g.dudz);
            t.yz = mu_visc * (g.dvdz + g.dwdy);

            if (coord != 0) {
                const Real rr  = problo[0] + (ii + 0.5_rt) * dx[0];
                const Real u_r = vel_raw(ii, jj, kk, 0);
                t.theta_theta = mu_visc * (2.0_rt * (u_r / rr) - (2.0_rt/3.0_rt) * div_u);
            }
            return t;
        };

        const Tau t_c = compute_tau(i, j, k);

        // ---- radial / x momentum ----
        {
            const int ip = cx(i+1), im = cx(i-1);
            const Real inv = (ip > im) ? 1.0_rt / ((ip - im) * dx[0]) : 0.0_rt;
            const Tau t_p = compute_tau(ip, j, k);
            const Tau t_m = compute_tau(im, j, k);
            S_mx += (t_p.xx - t_m.xx) * inv;
            S_my += (t_p.xy - t_m.xy) * inv;
#if (AMREX_SPACEDIM > 2)
            S_mz += (t_p.xz - t_m.xz) * inv;
#endif
        }

#if (AMREX_SPACEDIM > 1)
        {
            const int jp = cy(j+1), jm = cy(j-1);
            const Real inv = (jp > jm) ? 1.0_rt / ((jp - jm) * dx[1]) : 0.0_rt;
            const Tau t_p = compute_tau(i, jp, k);
            const Tau t_m = compute_tau(i, jm, k);
            S_mx += (t_p.xy - t_m.xy) * inv;
            S_my += (t_p.yy - t_m.yy) * inv;
#if (AMREX_SPACEDIM > 2)
            S_mz += (t_p.yz - t_m.yz) * inv;
#endif
        }
#endif

#if (AMREX_SPACEDIM > 2)
        {
            const int kp = cz(k+1), km = cz(k-1);
            const Real inv = (kp > km) ? 1.0_rt / ((kp - km) * dx[2]) : 0.0_rt;
            const Tau t_p = compute_tau(i, j, kp);
            const Tau t_m = compute_tau(i, j, km);
            S_mx += (t_p.xz - t_m.xz) * inv;
            S_my += (t_p.yz - t_m.yz) * inv;
            S_mz += (t_p.zz - t_m.zz) * inv;
        }
#endif

        // ---- geometric (curvature) terms ----
        if (coord != 0) {
            S_mx += (t_c.xx - t_c.theta_theta) / r;   // (1/r)d(r tau_rr)/dr - tau_thth/r
            S_my += t_c.xy / r;
#if (AMREX_SPACEDIM > 2)
            S_mz += t_c.xz / r;
#endif
        }

        // ---- viscous dissipation tau : grad(u)  (>= 0) ----
        const Grads gc = get_grads(i, j, k);
        S_visc = t_c.xx * gc.dudx + t_c.yy * gc.dvdy + t_c.zz * gc.dwdz
               + t_c.xy * (gc.dudy + gc.dvdx)
               + t_c.xz * (gc.dudz + gc.dwdx)
               + t_c.yz * (gc.dvdz + gc.dwdy);

        if (coord != 0) {
            S_visc += t_c.theta_theta * (vel_raw(i, j, k, 0) / r);
        }

        // Dissipation is positive-definite in exact arithmetic; enforce it so
        // stencil noise can never cool the gas.
        S_visc = amrex::max(S_visc, 0.0_rt);

        // Final backstop: never hand a non-finite source to the integrator.
        if (!std::isfinite(S_mx) || !std::isfinite(S_my) ||
            !std::isfinite(S_mz) || !std::isfinite(S_visc)) {
            S_mx = 0.0_rt; S_my = 0.0_rt; S_mz = 0.0_rt; S_visc = 0.0_rt;
        }
    }

} // namespace transport

#endif // VISCOSITY_H