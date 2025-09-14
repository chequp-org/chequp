#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <prob_parameters.H>
#include <eos.H>
#include <AMReX_Math.H>
#include <AMReX_Print.H>
#include <fundamental_constants.H>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

// Static arrays for initial conditions that are read from a CSV file
// These arrays are initialized from a file in the function read_initial_conditions
// We then use interpolation to read these values for each cell,
// in the function problem_initialize_state_data (GPU kernel)
static std::vector<Real> initial_conditions_r; // radius in cm
static std::vector<Real> initial_conditions_ioniz; // fraction of atoms that are ionized
static std::vector<Real> initial_conditions_Te; // electron temperature in K

// Function to initialize initial_conditions_r, initial_conditions_ioniz, initial_conditions_Te
inline void read_initial_conditions()
{
    const std::string initial_conditions_filename = "initial_conditions.csv";
    std::ifstream file(initial_conditions_filename);
    if (!file.is_open()) {
        amrex::Abort("Failed to open " + initial_conditions_filename);
    }

    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        // Read radius
        if (std::getline(ss, value, ',')) {
            try {
                initial_conditions_r.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                amrex::Abort("Invalid radius value: " + value);
            }
        } else {
            amrex::Abort("Missing radius value");
        }

        // Read ionization fraction
        if (std::getline(ss, value, ',')) {
            try {
                initial_conditions_ioniz.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                amrex::Abort("Invalid ionization fraction value: " + value);
            }
        } else {
            amrex::Abort("Missing ionization fraction value");
        }

        // Read electron temperature
        if (std::getline(ss, value)) {
            try {
                initial_conditions_Te.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                amrex::Abort("Invalid electron temperature value: " + value);
            }
        } else {
            amrex::Abort("Missing electron temperature value");
        }
    }
    file.close();
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k, Array4<Real> const& state, const GeometryData& geomdata)
{
    int coord_type = geomdata.Coord();

    const Real* dx = geomdata.CellSize();
    const Real* problo = geomdata.ProbLo();

    Real x = 0;
    Real y = 0;
    Real z = 0;

    x = problo[0] + dx[0] * (static_cast<Real>(i)+0.5);
#if AMREX_SPACEDIM >= 2
    y = problo[1] + dx[1] * (static_cast<Real>(j)+0.5);
#endif
#if AMREX_SPACEDIM >= 3
    z = problo[2] + dx[2] * (static_cast<Real>(k)+0.5);
#endif

    Real r = sqrt(x*x + y*y + z*z);

    // Linear interpolation to find ionization fraction and electron temperature
    Real ioniz_fraction = 0.0;
    Real te = 0.0;

    // Check if plasma data has been loaded
    if (initial_conditions_r.empty()) {
        amrex::Abort("Plasma data not initialized. Make sure problem_initialize() is called first.");
    }

    if (r <= initial_conditions_r.front()) {
        ioniz_fraction = initial_conditions_ioniz.front();
        te = initial_conditions_Te.front();
    } else if (r >= initial_conditions_r.back()) {
        ioniz_fraction = initial_conditions_ioniz.back();
        te = initial_conditions_Te.back();
    } else {
        for (int idx = 1; idx < initial_conditions_r.size(); ++idx) {
            if (r < initial_conditions_r[idx]) {
                Real r_low = initial_conditions_r[idx-1];
                Real r_high = initial_conditions_r[idx];
                Real frac = (r - r_low) / (r_high - r_low);

                ioniz_fraction = initial_conditions_ioniz[idx-1] + frac * (initial_conditions_ioniz[idx] - initial_conditions_ioniz[idx-1]);
                te = initial_conditions_Te[idx-1] + frac * (initial_conditions_Te[idx] - initial_conditions_Te[idx-1]);
                break;
            }
        }
    }
    Real eint = problem::dens_ambient * (ioniz_fraction * 3./2 * C::k_B * te +  3./2 * C::k_B * problem::temp_ambient)/C::m_u;

    state(i,j,k,URHO) = problem::dens_ambient;
    state(i,j,k,UMX) = 0.e0_rt;
    state(i,j,k,UMY) = 0.e0_rt;
    state(i,j,k,UMZ) = 0.e0_rt;

    state(i,j,k,UEDEN) = eint +
                         0.5e0_rt * (state(i,j,k,UMX) * state(i,j,k,UMX) / state(i,j,k,URHO) +
                                     state(i,j,k,UMY) * state(i,j,k,UMY) / state(i,j,k,URHO) +
                                     state(i,j,k,UMZ) * state(i,j,k,UMZ) / state(i,j,k,URHO));

    state(i,j,k,UEINT) = eint;
    state(i,j,k,UFX) = eint; // internal energy of heavy ions

    // The temperature initialization will be done later
    // in the call to clean_state. We want to avoid
    // EOS calls in ca_initdata. So that there is valid
    // data for an initial guess for the EOS, we initialize
    // to the ambient temperature everywhere.

    state(i,j,k,UTEMP) = problem::temp_ambient;

    state(i,j,k,UFS) = state(i,j,k,URHO)*(1-ioniz_fraction);
    state(i,j,k,UFS+1) = state(i,j,k,URHO)*ioniz_fraction;
}

#endif
