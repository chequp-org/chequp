#ifndef problem_initialize_H
#define problem_initialize_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void problem_initialize ()
{
    const Geometry& dgeom = DefaultGeometry();

    const int coord_type = dgeom.Coord();
    const Real* problo = dgeom.ProbLo();
    const Real* probhi = dgeom.ProbHi();

    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        if (coord_type == 1 || coord_type == 2) {
            problem::center[n] = 0.0_rt;
        }
        else {
            problem::center[n] = 0.5_rt * (problo[n] + probhi[n]);
        }
    }

    Real xn_zone[NumSpec];
    for (int n = 0; n < NumSpec; ++n) {
        xn_zone[n] = 0.0_rt;
    }
    xn_zone[0] = 1.0_rt;

    eos_t eos_state;

    // override the pressure with the temperature
    if (problem::temp_ambient > 0.0_rt) {

        eos_state.rho = problem::dens_ambient;
        eos_state.T = problem::temp_ambient;
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = xn_zone[n];
        }

        eos(eos_input_rt, eos_state);

        problem::p_ambient = eos_state.p;

    }

    // Calculate ambient state data

    eos_state.rho = problem::dens_ambient;
    eos_state.p   = problem::p_ambient;
    eos_state.T   = 1.e9_rt; // Initial guess for iterations
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn_zone[n];
    }

    eos(eos_input_rp, eos_state);

    problem::temp_ambient = eos_state.T;

}

#endif
