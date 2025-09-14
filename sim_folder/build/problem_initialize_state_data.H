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
static amrex::Gpu::DeviceVector<Real> initial_conditions_r; // radius in cm
static amrex::Gpu::DeviceVector<Real> initial_conditions_ioniz; // fraction of atoms that are ionized
static amrex::Gpu::DeviceVector<Real> initial_conditions_Te; // electron temperature in K
static Real* initial_conditions_r_data = nullptr;
static Real* initial_conditions_ioniz_data = nullptr;
static Real* initial_conditions_Te_data = nullptr;
static int initial_conditions_size = 0;

// Function to fill out initial_conditions_r, initial_conditions_ioniz, initial_conditions_Te
inline void read_initial_conditions()
{
    const std::string initial_conditions_filename = "initial_conditions.csv";
    std::ifstream file(initial_conditions_filename);
    if (!file.is_open()) {
        amrex::Abort("Failed to open " + initial_conditions_filename);
    }

    // Vector to fill when reading the file
    static std::vector<Real> r; // radius in cm
    static std::vector<Real> ioniz; // fraction of atoms that are ionized
    static std::vector<Real> Te; // electron temperature in K

    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        // Read radius
        if (std::getline(ss, value, ',')) {
            try {
                r.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                amrex::Abort("Invalid radius value: " + value);
            }
        } else {
            amrex::Abort("Missing radius value");
        }

        // Read ionization fraction
        if (std::getline(ss, value, ',')) {
            try {
                ioniz.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                amrex::Abort("Invalid ionization fraction value: " + value);
            }
        } else {
            amrex::Abort("Missing ionization fraction value");
        }

        // Read electron temperature
        if (std::getline(ss, value)) {
            try {
                Te.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                amrex::Abort("Invalid electron temperature value: " + value);
            }
        } else {
            amrex::Abort("Missing electron temperature value");
        }
    }
    file.close();

    // Copy the vectors to the static variables
    // Resize the static variables to the size of the vectors
    initial_conditions_r.resize(r.size());
    initial_conditions_ioniz.resize(ioniz.size());
    initial_conditions_Te.resize(Te.size());
#ifdef AMREX_USE_GPU
    // Copy vectors to device
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r.begin(), r.end(),
                          initial_conditions_r.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, ioniz.begin(), ioniz.end(),
                          initial_conditions_ioniz.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Te.begin(), Te.end(),
                          initial_conditions_Te.begin());
    amrex::Gpu::streamSynchronize();
#else
    // Copy the vectors to the static variables
    std::copy(r.begin(), r.end(), initial_conditions_r.begin());
    std::copy(ioniz.begin(), ioniz.end(), initial_conditions_ioniz.begin());
    std::copy(Te.begin(), Te.end(), initial_conditions_Te.begin());
#endif

    // Initialize pointers to be used in the GPU kernel
    initial_conditions_r_data = initial_conditions_r.data();
    initial_conditions_ioniz_data = initial_conditions_ioniz.data();
    initial_conditions_Te_data = initial_conditions_Te.data();
    initial_conditions_size = initial_conditions_r.size();
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k, Array4<Real> const& state, const GeometryData& geomdata)
{
    int coord_type = geomdata.Coord();

    const Real* dx = geomdata.CellSize();
    const Real* problo = geomdata.ProbLo();

    // Get radius for this cell
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
    int const idx = amrex::bisect( initial_conditions_r_data, 0, initial_conditions_size-1, r );
    AMREX_ASSERT( initial_conditions_r_data[idx] <= r && initial_conditions_r_data[idx+1] > r);
    Real frac = (r - initial_conditions_r_data[idx]) / (initial_conditions_r_data[idx+1] - initial_conditions_r_data[idx]);
    Real ioniz_fraction = initial_conditions_ioniz[idx] + frac * (initial_conditions_ioniz[idx+1] - initial_conditions_ioniz[idx]);
    Real Te = initial_conditions_Te[idx] + frac * (initial_conditions_Te[idx+1] - initial_conditions_Te[idx]);

    Real eint = problem::dens_ambient * (ioniz_fraction * 3./2 * C::k_B * Te +  3./2 * C::k_B * problem::temp_ambient)/C::m_u;

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
