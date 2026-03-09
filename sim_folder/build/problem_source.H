#ifndef problem_source_H
#define problem_source_H

#include <prob_parameters.H>
#include <fundamental_constants.H>
#include <collision_frequencies.H>
#include <conductivity.H>
#include <ionization_table.H>

using namespace amrex;

// Returns the thermal conductivity coefficient k_th at point (i,j,k)
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real get_conductivity_coeff( int i, int j, int k, Array4<const Real> const& state )
{
    // Reproduces the same code as `fill_temp_cond` in `diffusion_util.cpp`

    eos_t eos_state;
    eos_state.rho  = state(i,j,k,URHO);
    Real rhoinv = 1.0_rt/eos_state.rho;

    eos_state.T = state(i,j,k,UTEMP);   // needed as an initial guess
    eos_state.e = state(i,j,k,UEINT) * rhoinv;
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = state(i,j,k,UFS+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = state(i,j,k,UFX+n) * rhoinv;
    }
#endif

    if (eos_state.e < 0.0_rt) {
      eos_state.T = castro::small_temp;
      eos(eos_input_rt, eos_state);
    } else {
      eos(eos_input_re, eos_state);
    }

    if (eos_state.rho > castro::diffuse_cutoff_density) {
      conductivity(eos_state);

      if (eos_state.rho < castro::diffuse_cutoff_density_hi) {
        Real multiplier = (eos_state.rho - castro::diffuse_cutoff_density) /
          (castro::diffuse_cutoff_density_hi - castro::diffuse_cutoff_density);
        eos_state.conductivity = eos_state.conductivity * multiplier;
      }
    } else {
      eos_state.conductivity = 0.0_rt;
    }
    return castro::diffuse_cond_scale_fac * eos_state.conductivity;
}

// Returns the term ∇ ⋅(k_th ∇T) at point (i,j,k)
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real compute_temp_diffusion_term(
    int i, int j, int k, Array4<const Real> const& state, const Real* problo, const Real* dx, const int coord )
{
    // Reproduces the same code as `getTempDiffusionTerm` in `Castro_diffusion.cpp`

    amrex::Real temp_diffusion_term = 0.0;
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        int il = i;
        int jl = j;
        int kl = k;

        int ir = i;
        int jr = j;
        int kr = k;

        Real dxinv2 = 1.0/(dx[idir]*dx[idir]);
        Real kth_r;
        Real kth_l;

        if (idir == 0) {
            il = i - 1;
            ir = i + 1;
        }
#if AMREX_SPACEDIM >= 2
        else if (idir == 1) {
            jl = j - 1;
            jr = j + 1;
        }
#endif
#if AMREX_SPACEDIM == 3
        else {
            kl = k - 1;
            kr = k + 1;
        }
#endif
        // Conductivity at the cell edges: average of the conductivity from cell centers to the edges
        // TODO: This could be made more efficient by avoiding to recalculate
        // conductivity_coeffs(i,j,k,state) multiple times
        kth_r = 0.5*(get_conductivity_coeff(ir,jr,kr,state) + get_conductivity_coeff(i,j,k,state));
        kth_l = 0.5*(get_conductivity_coeff(il,jl,kl,state) + get_conductivity_coeff(i,j,k,state));

#if AMREX_SPACEDIM < 3
        // Apply geometric terms for curvilinear coordinates

        if ((coord != 0) && (idir == 0)) {
            // In curilinear radial direction

            Real rr = problo[idir] + static_cast<Real>(ir) * dx[idir];
            Real rl = problo[idir] + static_cast<Real>(i) * dx[idir];
            Real rc = problo[idir] + (static_cast<Real>(i) + 0.5_rt) * dx[idir];

            if (coord == 1) {
                // Cylindrical radial equation looks like: 1/r d(r kth dT/dr)/dr

                kth_r *= rr;
                kth_l *= rl;
                dxinv2 *= 1.0_rt / rc;
            } else {
                // Spherical radial equation looks like: 1/r^2 d(r^2 kth dT/dr)/dr

                kth_r *= rr * rr;
                kth_l *= rl * rl;
                dxinv2 *= 1.0_rt / (rc * rc) ;
            }
        } else if ((coord == 2) && (idir == 1)) {
            // In spherical theta direction
            // Spherical theta equation looks like: 1/r^2 sin(θ) d(sin(θ) kth dT/dθ)/dθ

            Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
            Real thetar = problo[idir] + static_cast<Real>(jr) * dx[idir];
            Real thetal = problo[idir] + static_cast<Real>(j) * dx[idir];
            Real thetac = problo[idir] + (static_cast<Real>(j) + 0.5_rt) * dx[idir];

            kth_r *= std::sin(thetar);
            kth_l *= std::sin(thetal);
            dxinv2 *= 1.0_rt / (rc * rc * std::sin(thetac));
        }
#endif
        temp_diffusion_term += dxinv2 *
            (kth_r * (state(ir,jr,kr,UTEMP) - state(i ,j ,k ,UTEMP)) -
             kth_l * (state(i ,j ,k ,UTEMP) - state(il,jl,kl,UTEMP)));
    }

    return temp_diffusion_term;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;

    // Extract density and electron temperature
    Real rho = state(i,j,k,URHO);

    // Electron temperature
    // In a single-temperature model (gamma_law), this will extract the average temperature
    // In a two-temperature model (e.g. gamma_law_2T), this will extract the electron temperature
    const amrex::Real T_e = state(i,j,k,UTEMP);

    // Initialize species mass fractions and identify active species
    // Initialize species mass fractions
    Real X_stage[NumSpec];
    
    // Create mask to identify active species
    bool active_mask[NumSpec] = {false};
    int active_spec[NumSpec];
    int n_active = 0;

    // Extract mass fractions and mark active species
    for (int mu = 0; mu < NumSpec; mu++) {
        X_stage[mu] = state(i,j,k,UFS+mu) / rho;
        
        // Mark species and neighbors as active if mass fraction is above threshold
        if (X_stage[mu] > X_min) {
            active_mask[mu] = true;
            if (mu > 0) active_mask[mu-1] = true;
            if (mu < NumSpec-1) active_mask[mu+1] = true;
        }
        
        // Build list of active species indices
        if (active_mask[mu]) {
            active_spec[n_active++] = mu;
        }
    }

    // Extract energy and ion fraction
    Real f = state(i,j,k,UFX) / rho;
    Real e = state(i,j,k,UEINT) / rho;

    // Calculate electron fraction from active species
    Real X_e = 0.0;
    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        X_e += zion[mu] * X_stage[mu];
    }
    // Initialize sources
    Real e_ion = 0.0;

    // Loop over ionizing species only once
    for (int ispe = 0; ispe < n_active; ispe++) {
        // Access active index
        int beta = active_spec[ispe];
        // Physical constants for three-body recombination
        // De Broglie wavelength for electrons (in cm)
        Real const lambda_DeBroglie = C::hbar * std::sqrt( 2*M_PI/(C::m_e*C::k_B*T_e) );
        Real const reverse_rate_factor = rho * std::pow(lambda_DeBroglie, 3) / (aion[ispe] * C::m_p * g_deg[ispe]);
        // Reaction rate
        Real coll_I = 0.0;
        Real three_body_R = 0.0;
        for (int step = 0; step < n_step; step++) {
            coll_I += A_I[beta][step] *
                std::pow(C::k_B*T_e / C::ev2erg, q_I[beta][step]) *
                std::exp(-eps_I[beta][step] * C::ev2erg / (C::k_B*T_e)) *
                X_stage[beta] * X_e;
            three_body_R += A_I[beta][step] *
                        std::pow(C::k_B*T_e / C::ev2erg, q_I[beta][step]) * 
                        X_e * X_e * X_e * reverse_rate_factor;
        }

        Real const coeff = (coll_I - three_body_R) * (rho * rho / C::m_p);
        // Loss
        src(i,j,k,UFS+beta) -= coeff;

        // Gain
        int beta_next = beta + 1;
        if (beta_next < NumSpec) {
            src(i,j,k,UFS+beta_next) += coeff;
        }

        // Energy lose by ionization
        e_ion -= (coll_I - three_body_R) * (rho * rho / C::m_p) *
                (eps_I[beta][0] * C::ev2erg / C::m_p);
    }

    // Update energy source terms
    src(i,j,k,UEDEN) = e_ion;
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

    // Calculate electron and neutral number densities (SI units)
    amrex::Real n_e = 0.0;
    amrex::Real n_n_species[num_atom_species] = {};

    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        n_e += zion[mu] * X_stage[mu] * rho / C::m_u * 1.e6;
        if (zion[mu] == 0) {
            int idx_atom = get_atom_index(mu);
            n_n_species[idx_atom] += X_stage[mu] * rho / C::m_u * 1.e6;
        }
    }

    // Compute collision frequencies
    amrex::Real v_ei;
    amrex::Real v_en[NumSpec];
    const amrex::Real T_h = T_e;
    get_collision_frequencies(n_e, n_n_species, T_e, T_h, v_ei, v_en);

    // Calculate momentum transfer collision frequencies
    amrex::Real v_e_E[NumSpec] = {};
    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        v_e_E[mu] = 2 * SI::m_e / mass_vec[get_atom_index(mu)] *
                    (std::pow(zion[mu], 4) * v_ei + v_en[mu]);
    }

    // Calculate energy transfer source term
    amrex::Real Sc = 0.0;
    for (int ispe = 0; ispe < n_active; ispe++) {
        int mu = active_spec[ispe];
        Sc += rho * v_e_E[mu] * (1 - (1 + X_stage[mu]) * f);
    }

    // Add contribution for temperature diffusion
    if (diffuse_temp) {
        const amrex::Real temp_diffusion_term = compute_temp_diffusion_term(i,j,k,state,prob_lo,dx,coord);
        src(i,j,k,UEDEN) += temp_diffusion_term;
        src(i,j,k,UEINT) += temp_diffusion_term;
    }

    src(i,j,k,UFX) = -f/e * src(i,j,k,UEDEN) + Sc;
}

#endif
