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
#ifdef AMREX_USE_GPU
#include <cuda_runtime.h>
#endif

// Static arrays for initial conditions that are read from a CSV file
// These arrays are initialized from a file in the function read_initial_conditions
// We then use interpolation to read these values for each cell,
// in the function problem_initialize_state_data (GPU kernel)
static amrex::Gpu::DeviceVector<Real> initial_conditions_r; // radius in cm
static amrex::Gpu::DeviceVector<Real> initial_conditions_ioniz; // fraction of atoms that are ionized
static amrex::Gpu::DeviceVector<Real> initial_conditions_Te; // electron temperature in K
// Device-accessible static variables
AMREX_GPU_DEVICE static Real* d_initial_conditions_r_data = nullptr;
AMREX_GPU_DEVICE static Real* d_initial_conditions_ioniz_data = nullptr;
AMREX_GPU_DEVICE static Real* d_initial_conditions_Te_data = nullptr;
AMREX_GPU_DEVICE static int d_initial_conditions_size = 0;

// Function to fill out initial_conditions_r, initial_conditions_ioniz, initial_conditions_Te
inline void read_initial_conditions()
{
    std::ifstream file(problem::initial_conditions_file);
    if (!file.is_open()) {
        amrex::Abort("Failed to open " + problem::initial_conditions_file);
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

    // Resize the static vectors
    int initial_conditions_size = r.size();
    initial_conditions_r.resize(initial_conditions_size);
    initial_conditions_ioniz.resize(initial_conditions_size);
    initial_conditions_Te.resize(initial_conditions_size);
    // Initialize pointers to be copied into static variables
    Real* initial_conditions_r_data = initial_conditions_r.data();
    Real* initial_conditions_ioniz_data = initial_conditions_ioniz.data();
    Real* initial_conditions_Te_data = initial_conditions_Te.data();

    // Copy all needed information into the static variables
#ifdef AMREX_USE_GPU
    // Copy simple variables
    cudaMemcpyToSymbol(d_initial_conditions_size, &initial_conditions_size, sizeof(int));
    cudaMemcpyToSymbol(d_initial_conditions_r_data, &initial_conditions_r_data, sizeof(Real*));
    cudaMemcpyToSymbol(d_initial_conditions_ioniz_data, &initial_conditions_ioniz_data, sizeof(Real*));
    cudaMemcpyToSymbol(d_initial_conditions_Te_data, &initial_conditions_Te_data, sizeof(Real*));
    // Copy vectors
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r.begin(), r.end(),
                          initial_conditions_r.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, ioniz.begin(), ioniz.end(),
                          initial_conditions_ioniz.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Te.begin(), Te.end(),
                          initial_conditions_Te.begin());
    amrex::Gpu::streamSynchronize();
#else
    // Copy simple variables
    d_initial_conditions_size = initial_conditions_size;
    d_initial_conditions_r_data = initial_conditions_r_data;
    d_initial_conditions_ioniz_data = initial_conditions_ioniz_data;
    d_initial_conditions_Te_data = initial_conditions_Te_data;
    // Copy vectors
    std::copy(r.begin(), r.end(), initial_conditions_r.begin());
    std::copy(ioniz.begin(), ioniz.end(), initial_conditions_ioniz.begin());
    std::copy(Te.begin(), Te.end(), initial_conditions_Te.begin());
#endif
}

AMREX_GPU_DEVICE AMREX_INLINE
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
    int const idx = amrex::bisect( d_initial_conditions_r_data, 0, d_initial_conditions_size-1, r );
    AMREX_ASSERT( d_initial_conditions_r_data[idx] <= r && d_initial_conditions_r_data[idx+1] > r);
    Real frac = (r - d_initial_conditions_r_data[idx]) / (d_initial_conditions_r_data[idx+1] - d_initial_conditions_r_data[idx]);
    Real read_ioniz_fraction = d_initial_conditions_ioniz_data[idx] + frac * (d_initial_conditions_ioniz_data[idx+1] - d_initial_conditions_ioniz_data[idx]);
    Real Te = d_initial_conditions_Te_data[idx] + frac * (d_initial_conditions_Te_data[idx+1] - d_initial_conditions_Te_data[idx]);

    // For the electron temperature to be well-defined everywhere, there needs to be a non-zero ionization fraction.
    // Thus, whenever read_ioniz_fraction is less than min_ioniz_fraction, artificially add electrons so that the ionization fraction is at least min_ioniz_fraction.
    // The added electrons are at the ambient temperature.
    Real added_ioniz_fraction = std::max(problem::min_ioniz_fraction - read_ioniz_fraction, 0.0);
    Real rho_e_heavies =  problem::dens_ambient * 3./2 * C::k_B * problem::temp_ambient / C::m_u;
    Real rho_e_electrons = problem::dens_ambient * 3./2 * C::k_B * (read_ioniz_fraction * Te + added_ioniz_fraction * problem::temp_ambient) / C::m_u;

    state(i,j,k,URHO) = problem::dens_ambient;
    state(i,j,k,UMX) = 0.e0_rt;
    state(i,j,k,UMY) = 0.e0_rt;
    state(i,j,k,UMZ) = 0.e0_rt;

    state(i,j,k,UEDEN) = rho_e_heavies + rho_e_electrons +
                         0.5e0_rt * (state(i,j,k,UMX) * state(i,j,k,UMX) / state(i,j,k,URHO) +
                                     state(i,j,k,UMY) * state(i,j,k,UMY) / state(i,j,k,URHO) +
                                     state(i,j,k,UMZ) * state(i,j,k,UMZ) / state(i,j,k,URHO));

    state(i,j,k,UEINT) = rho_e_heavies + rho_e_electrons;
    Real ioniz_fraction = read_ioniz_fraction + added_ioniz_fraction;
    state(i,j,k,UFS) = state(i,j,k,URHO)*(1-ioniz_fraction);
    state(i,j,k,UFS+1) = state(i,j,k,URHO)*ioniz_fraction;
    // UFX is rho*f, where f is the fraction of internal energy stored in heavies (ions + atoms)
    state(i,j,k,UFX) = problem::dens_ambient * rho_e_heavies / (rho_e_electrons + rho_e_heavies);

    // The temperature initialization will be done later
    // in the call to clean_state. We want to avoid
    // EOS calls in ca_initdata. So that there is valid
    // data for an initial guess for the EOS, we initialize
    // to the ambient temperature everywhere.
    state(i,j,k,UTEMP) = problem::temp_ambient;
}

#endif
