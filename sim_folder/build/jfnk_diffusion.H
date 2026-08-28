#ifndef jfnk_diffusion_H
#define jfnk_diffusion_H

#include <diffusion_operator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <eos.H>
#include <vector>

using namespace amrex;

namespace jfnk_diffusion {

    // ---------------------------------------------------------------
    // Tunable solver parameters. Expose these via _prob_params if you
    // want runtime control; hardcoded here for clarity.
    // ---------------------------------------------------------------
    static constexpr int    max_newton_iters   = 30;
    static constexpr Real   newton_rtol        = 1.0e-8_rt;
    static constexpr Real   newton_atol        = 1.0e-30_rt; // in erg/cm^3 units, problem-dependent
    static constexpr int    max_gmres_iters    = 50;
    static constexpr Real   gmres_rtol         = 1.0e-6_rt;
    static constexpr Real   jfnk_eps_base      = 1.0e-7_rt;

    // ---------------------------------------------------------------
    // Given a state MultiFab (with UTEMP holding the current Newton
    // iterate T^k, and URHO/UFS/UFX consistent with the *old* time
    // level, i.e. not evolved by diffusion), recompute UEINT/UEDEN
    // consistently with UTEMP via the EOS, at every cell in bx.
    // This keeps state "self-consistent" as Newton updates T.
    // ---------------------------------------------------------------
    inline void
    sync_energy_from_temp( amrex::MultiFab& state_mf )
    {
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(state_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.growntilebox(); // include ghosts: needed for stencil
            Array4<Real> const state = state_mf.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real rho = state(i,j,k,URHO);
                if (rho <= 0.0_rt) { return; }
                Real rhoinv = 1.0_rt / rho;

                eos_t eos_state;
                eos_state.rho = rho;
                eos_state.T   = state(i,j,k,UTEMP);
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = state(i,j,k,UFS+n) * rhoinv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = state(i,j,k,UFX+n) * rhoinv;
                }
#endif
                eos(eos_input_rt, eos_state);

                Real ke = 0.5_rt * rhoinv *
                    (state(i,j,k,UMX)*state(i,j,k,UMX) +
                     state(i,j,k,UMY)*state(i,j,k,UMY) +
                     state(i,j,k,UMZ)*state(i,j,k,UMZ));

                state(i,j,k,UEINT) = rho * eos_state.e;
                state(i,j,k,UEDEN) = rho * eos_state.e + ke;
            });
        }
    }

    // ---------------------------------------------------------------
    // Residual: F(T) = rho*e(T) - rho*e^n - dt * L(T)
    // state_mf    : current Newton iterate (UTEMP = T^k, energies made
    //               consistent via sync_energy_from_temp before calling)
    // rhoe_n_mf   : rho*e at the OLD time level (fixed data, not updated)
    // F_mf        : output residual, single component
    // Requires state_mf to have >=1 filled ghost cell.
    // ---------------------------------------------------------------
    inline void
    compute_residual( amrex::MultiFab& state_mf,
                       const amrex::MultiFab& rhoe_n_mf,
                       amrex::MultiFab& F_mf,
                       const amrex::Geometry& geom,
                       Real dt )
    {
        sync_energy_from_temp(state_mf);
        state_mf.FillBoundary(geom.periodicity());

        amrex::MultiFab L_mf(F_mf.boxArray(), F_mf.DistributionMap(), 1, 0);
        compute_diffusion_term_mf(state_mf, L_mf, geom);

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(F_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array4<Real const> const state = state_mf.const_array(mfi);
            Array4<Real const> const rhoe_n = rhoe_n_mf.const_array(mfi);
            Array4<Real const> const L = L_mf.const_array(mfi);
            Array4<Real> const F = F_mf.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                F(i,j,k) = state(i,j,k,UEINT) - rhoe_n(i,j,k) - dt * L(i,j,k);
            });
        }
    }

    // ---------------------------------------------------------------
    // Matrix-free Jacobian-vector product via forward differencing:
    //   J(T)*v  ~=  [F(T + eps*v) - F(T)] / eps
    // v_mf and Jv_mf are single-component (they represent perturbations
    // to UTEMP only). state_mf is the current Newton iterate (UTEMP=T^k);
    // it is copied internally so the caller's state_mf is left unmodified.
    // ---------------------------------------------------------------
    inline void
    jacobian_vector_product( const amrex::MultiFab& state_mf,
                              const amrex::MultiFab& F0_mf,
                              const amrex::MultiFab& v_mf,
                              amrex::MultiFab& Jv_mf,
                              const amrex::MultiFab& rhoe_n_mf,
                              const amrex::Geometry& geom,
                              Real dt )
    {
        Real norm_T = state_mf.norm2(UTEMP);
        Real norm_v = v_mf.norm0();
        Real eps = jfnk_eps_base * (norm_T + 1.0_rt) / (norm_v + 1.0e-300_rt);

        amrex::MultiFab state_pert(state_mf.boxArray(), state_mf.DistributionMap(),
                                    state_mf.nComp(), state_mf.nGrow());
        amrex::MultiFab::Copy(state_pert, state_mf, 0, 0, state_mf.nComp(), state_mf.nGrow());

        // Perturb UTEMP by eps*v (v has no ghost cells; only touch valid region,
        // then FillBoundary will be called inside compute_residual)
        for (MFIter mfi(state_pert, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array4<Real> const state = state_pert.array(mfi);
            Array4<Real const> const v = v_mf.const_array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                state(i,j,k,UTEMP) += eps * v(i,j,k);
            });
        }

        amrex::MultiFab Feps_mf(F0_mf.boxArray(), F0_mf.DistributionMap(), 1, 0);
        compute_residual(state_pert, rhoe_n_mf, Feps_mf, geom, dt);

        amrex::MultiFab::LinComb(Jv_mf, 1.0_rt/eps, Feps_mf, 0, -1.0_rt/eps, F0_mf, 0, 0, 1, 0);
    }

    // ---------------------------------------------------------------
    // Preconditioner: cheap Jacobi/diagonal scaling using the local
    // "stiffness" ~ dt * kappa / dx^2 to rescale the residual before
    // GMRES. This is NOT a full physics-based (MLMG) preconditioner --
    // it is a diagonal approximation, but it costs almost nothing and
    // significantly helps GMRES conditioning across regions where
    // kappa(T) ~ T^(5/2) varies by orders of magnitude.
    // For harder problems, replace this with an MLABecLaplacian/MLMG
    // solve using lagged kappa(T^k) as a real preconditioner -- see
    // note at the end of this file.
    // ---------------------------------------------------------------
    inline void
    apply_diagonal_preconditioner( const amrex::MultiFab& state_mf,
                                    const amrex::MultiFab& r_mf,
                                    amrex::MultiFab& z_mf,
                                    const amrex::Geometry& geom,
                                    Real dt )
    {
        const Real* dx = geom.CellSize();
        Real dx2min = dx[0]*dx[0];
#if AMREX_SPACEDIM >= 2
        dx2min = amrex::min(dx2min, dx[1]*dx[1]);
#endif
#if AMREX_SPACEDIM == 3
        dx2min = amrex::min(dx2min, dx[2]*dx[2]);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(z_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array4<Real const> const state = state_mf.const_array(mfi);
            Array4<Real const> const r = r_mf.const_array(mfi);
            Array4<Real> const z = z_mf.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real kth = get_conductivity_coeff(i,j,k,state);
                // Diagonal of (I - dt*L) acting on energy: 1 + ~2*dim*dt*kth/(rho*cv*dx^2)
                // We don't have cv here explicitly (residual is energy-based), so we use
                // a robust relative scaling instead of an exact diagonal.
                Real stiffness = 2.0_rt * AMREX_SPACEDIM * dt * kth / dx2min;
                Real diag = 1.0_rt + stiffness;
                z(i,j,k) = r(i,j,k) / diag;
            });
        }
    }

    // ---------------------------------------------------------------
    // Preconditioned GMRES(m) restarted solver for J*dT = -F
    // Matrix-free: only needs jacobian_vector_product and the
    // preconditioner above. dT_mf is the Newton update (output).
    // ---------------------------------------------------------------
    inline int
    gmres_solve( const amrex::MultiFab& state_mf,
                 const amrex::MultiFab& F0_mf,
                 const amrex::MultiFab& rhoe_n_mf,
                 amrex::MultiFab& dT_mf,
                 const amrex::Geometry& geom,
                 Real dt,
                 int max_iters,
                 Real rtol )
    {
        const BoxArray& ba = F0_mf.boxArray();
        const DistributionMapping& dm = F0_mf.DistributionMap();

        dT_mf.setVal(0.0_rt);

        amrex::MultiFab r_mf(ba, dm, 1, 0);
        amrex::MultiFab::Copy(r_mf, F0_mf, 0, 0, 1, 0);
        r_mf.mult(-1.0_rt);   // r0 = -F0  (solving J*dT = -F)

        amrex::MultiFab z_mf(ba, dm, 1, 0);
        apply_diagonal_preconditioner(state_mf, r_mf, z_mf, geom, dt);

        Real beta = z_mf.norm2(0);
        if (beta < 1.0e-300_rt) { return 0; }

        std::vector<std::unique_ptr<amrex::MultiFab>> V;
        V.push_back(std::make_unique<amrex::MultiFab>(ba, dm, 1, 0));
        amrex::MultiFab::Copy(*V[0], z_mf, 0, 0, 1, 0);
        V[0]->mult(1.0_rt/beta);

        std::vector<std::vector<Real>> H(max_iters+1, std::vector<Real>(max_iters, 0.0_rt));
        std::vector<Real> g(max_iters+1, 0.0_rt);
        std::vector<Real> cs(max_iters, 0.0_rt), sn(max_iters, 0.0_rt);
        g[0] = beta;

        int final_k = 0;
        for (int k = 0; k < max_iters; ++k) {
            final_k = k+1;

            amrex::MultiFab w_mf(ba, dm, 1, 0);
            jacobian_vector_product(state_mf, F0_mf, *V[k], w_mf, rhoe_n_mf, geom, dt);

            amrex::MultiFab w_pc(ba, dm, 1, 0);
            apply_diagonal_preconditioner(state_mf, w_mf, w_pc, geom, dt);
            amrex::MultiFab::Copy(w_mf, w_pc, 0, 0, 1, 0);

            for (int i = 0; i <= k; ++i) {
                H[i][k] = amrex::MultiFab::Dot(w_mf, 0, *V[i], 0, 1, 0);
                amrex::MultiFab::Saxpy(w_mf, -H[i][k], *V[i], 0, 0, 1, 0);
            }
            H[k+1][k] = w_mf.norm2(0);

            if (H[k+1][k] > 1.0e-300_rt) {
                V.push_back(std::make_unique<amrex::MultiFab>(ba, dm, 1, 0));
                amrex::MultiFab::Copy(*V[k+1], w_mf, 0, 0, 1, 0);
                V[k+1]->mult(1.0_rt/H[k+1][k]);
            }

            // Apply previous Givens rotations to new column of H
            for (int i = 0; i < k; ++i) {
                Real temp    =  cs[i]*H[i][k] + sn[i]*H[i+1][k];
                H[i+1][k]    = -sn[i]*H[i][k] + cs[i]*H[i+1][k];
                H[i][k]      = temp;
            }
            // New Givens rotation
            Real denom = std::sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
            if (denom < 1.0e-300_rt) denom = 1.0e-300_rt;
            cs[k] = H[k][k]/denom;
            sn[k] = H[k+1][k]/denom;
            H[k][k]   = cs[k]*H[k][k] + sn[k]*H[k+1][k];
            H[k+1][k] = 0.0_rt;

            Real g_temp = cs[k]*g[k];
            g[k+1]      = -sn[k]*g[k];
            g[k]        = g_temp;

            Real resid = std::abs(g[k+1]) / beta;
            if (resid < rtol) { break; }
        }

        // Back-substitution to get y, then dT = sum y_i * V_i
        int m = final_k;
        std::vector<Real> y(m, 0.0_rt);
        for (int i = m-1; i >= 0; --i) {
            Real sum = g[i];
            for (int j = i+1; j < m; ++j) {
                sum -= H[i][j]*y[j];
            }
            y[i] = sum / H[i][i];
        }

        for (int i = 0; i < m; ++i) {
            amrex::MultiFab::Saxpy(dT_mf, y[i], *V[i], 0, 0, 1, 0);
        }

        return m;
    }

    // ---------------------------------------------------------------
    // Top-level Newton driver.
    //
    // state_in_mf : the CURRENT full state (e.g. state_new as handed to
    //               fill_ext_source -- may be const in the caller; this
    //               function never writes into it). UTEMP is used as the
    //               initial Newton guess; URHO/UFS/UFX/UMom are held
    //               fixed through the diffusion sub-step (operator
    //               splitting -- diffusion only touches energy/UTEMP).
    // dsrc_mf     : OUTPUT, single component. On return, contains the
    //               effective volumetric energy source rate
    //                   dsrc = (rho*e_implicit^{n+1} - rho*e^n) / dt
    //               which the caller should ADD into ext_src(UEINT) and
    //               ext_src(UEDEN) so that Castro's own
    //               state_new += dt*ext_src reproduces the converged
    //               implicit result.
    // nGrow       : number of valid ghost cells already filled in
    //               state_in_mf (must be >= 1).
    //
    // Returns true if converged within max_newton_iters.
    // ---------------------------------------------------------------
    inline bool
    solve_implicit_diffusion( const amrex::MultiFab& state_in_mf,
                               amrex::MultiFab& dsrc_mf,
                               const amrex::Geometry& geom,
                               Real dt,
                               int nGrow )
    {
        const BoxArray& ba = state_in_mf.boxArray();
        const DistributionMapping& dm = state_in_mf.DistributionMap();

        // Working copy that JFNK is free to mutate. Owned locally --
        // never aliases the caller's (possibly const) state_new.
        amrex::MultiFab state_mf(ba, dm, state_in_mf.nComp(), nGrow);
        amrex::MultiFab::Copy(state_mf, state_in_mf, 0, 0, state_in_mf.nComp(), nGrow);

        // Freeze the old-time (pre-diffusion) energy as the RHS anchor
        // for the backward-Euler residual. This is rho*e at the state
        // handed in, i.e. AFTER any explicit sources already folded in
        // by the caller, BEFORE diffusion is applied.
        amrex::MultiFab rhoe_n_mf(ba, dm, 1, 0);
        amrex::MultiFab::Copy(rhoe_n_mf, state_mf, UEINT, 0, 1, 0);

        amrex::MultiFab F_mf(ba, dm, 1, 0);
        amrex::MultiFab dT_mf(ba, dm, 1, 0);

        state_mf.FillBoundary(geom.periodicity());
        compute_residual(state_mf, rhoe_n_mf, F_mf, geom, dt);

        Real F0norm = F_mf.norm2(0);
        Real tol = amrex::max(newton_atol, newton_rtol * F0norm);

        bool converged = (F0norm <= tol);

        for (int newton_it = 0; newton_it < max_newton_iters && !converged; ++newton_it)
        {
            gmres_solve(state_mf, F_mf, rhoe_n_mf, dT_mf, geom, dt,
                        max_gmres_iters, gmres_rtol);

            // Simple damped update with backtracking line search on ||F||
            Real lambda = 1.0_rt;
            amrex::MultiFab state_trial(ba, dm, state_mf.nComp(), state_mf.nGrow());
            amrex::MultiFab F_trial(ba, dm, 1, 0);

            bool step_ok = false;
            for (int ls = 0; ls < 8 && !step_ok; ++ls) {
                amrex::MultiFab::Copy(state_trial, state_mf, 0, 0, state_mf.nComp(), state_mf.nGrow());
                for (MFIter mfi(state_trial, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const Box& bx = mfi.tilebox();
                    Array4<Real> const st = state_trial.array(mfi);
                    Array4<Real const> const d = dT_mf.const_array(mfi);
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        st(i,j,k,UTEMP) += lambda * d(i,j,k);
                    });
                }
                state_trial.FillBoundary(geom.periodicity());
                compute_residual(state_trial, rhoe_n_mf, F_trial, geom, dt);

                Real Ftrial_norm = F_trial.norm2(0);
                if (Ftrial_norm < F0norm || Ftrial_norm < tol) {
                    step_ok = true;
                } else {
                    lambda *= 0.5_rt;
                }
            }

            amrex::MultiFab::Copy(state_mf, state_trial, 0, 0, state_mf.nComp(), state_mf.nGrow());
            amrex::MultiFab::Copy(F_mf, F_trial, 0, 0, 1, 0);
            F0norm = F_mf.norm2(0);

            converged = (F0norm <= tol);
        }

        // Back out the effective source: dsrc = (rho*e^{n+1} - rho*e_in) / dt
        // Note rho*e_in here is UEINT of state_in_mf (pre-diffusion), NOT
        // rhoe_n_mf necessarily -- they're the same by construction above,
        // but we read from state_in_mf directly for clarity/robustness.
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(dsrc_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array4<Real const> const rhoe_new = state_mf.const_array(mfi);
            Array4<Real const> const rhoe_old = state_in_mf.const_array(mfi);
            Array4<Real> const dsrc = dsrc_mf.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                dsrc(i,j,k) = (rhoe_new(i,j,k,UEINT) - rhoe_old(i,j,k,UEINT)) / dt;
            });
        }

        return converged;
    }

} // namespace jfnk_diffusion

#endif