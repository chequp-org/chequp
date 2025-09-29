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
static amrex::Gpu::DeviceVector<Real> initial_conditions_table; // fraction of atoms that are ionized
// Device-accessible static variables
AMREX_GPU_DEVICE static Real* d_initial_conditions_r_data = nullptr;
AMREX_GPU_DEVICE static Real* d_initial_conditions_table_data = nullptr;
AMREX_GPU_DEVICE static int d_initial_conditions_size = 0;

inline void read_initial_conditions()
{
    std::ifstream file(problem::initial_conditions_file);
    if (!file.is_open()) {
        amrex::Abort("Failed to open " + problem::initial_conditions_file);
    }

    // Host-side vectors to temporarily store data from the file
    static std::vector<Real> r_host; // radius in cm
    static std::vector<Real> table_host_flat;
    int num_rows = 0;

    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        // Read radius
        if (std::getline(ss, value, ',')) {
            try {
                r_host.push_back(std::stod(value));
            } catch (const std::invalid_argument&) {
                amrex::Abort("Invalid radius value: " + value);
            }
        } else {
            amrex::Abort("Missing radius value");
        }

        // 2. Read the rest of the columns into the flattened table
        int current_cols = 0;
        while (std::getline(ss, value, ',')) {
            table_host_flat.push_back(std::stod(value));
            current_cols++;
        }

        if (current_cols != NumSpec+1) {
            amrex::Abort("Inconsistent number of columns in CSV file.");
        }
        num_rows++;
    }
    file.close();
    
    // print state to verify that the csv was read properly
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n[Initial Conditions] Successfully read file: "
                       << problem::initial_conditions_file << "\n";
        amrex::Print() << "    - Found " << num_rows << " data points (rows).\n";
        amrex::Print() << "    - Found " << NumSpec + 1 << " data columns (Te + species).\n";
        amrex::Print() << "    - First radius: " << r_host.front() << " cm, Last radius: " << r_host.back() << " cm.\n";
    }
    // Resize the device vectors
    initial_conditions_r.resize(num_rows);
    initial_conditions_table.resize(table_host_flat.size());
    // Get raw pointers for GPU copy
    Real* initial_conditions_r_data = initial_conditions_r.data();
    Real* d_table_ptr = initial_conditions_table.data();

    // Copy all needed information into the static variables
#ifdef AMREX_USE_GPU
    cudaMemcpyToSymbol(d_initial_conditions_size, &num_rows, sizeof(int));
    cudaMemcpyToSymbol(d_initial_conditions_r_data, &initial_conditions_r_data, sizeof(Real*));
    cudaMemcpyToSymbol(d_initial_conditions_table_data, &d_table_ptr, sizeof(Real*));

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r_host.begin(), r_host.end(),
                          initial_conditions_r.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, table_host_flat.begin(), table_host_flat.end(),
                          initial_conditions_table.begin());
    amrex::Gpu::streamSynchronize();
#else
    d_initial_conditions_size = num_rows;
    d_initial_conditions_r_data = initial_conditions_r_data;
    d_initial_conditions_table_data = d_table_ptr;
    std::copy(r_host.begin(), r_host.end(), initial_conditions_r.begin());
    std::copy(table_host_flat.begin(), table_host_flat.end(), initial_conditions_table.begin());
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
    if (coord_type == 0) {  // Cartesian
        y = problo[1] + dx[1] * (static_cast<Real>(j)+0.5);
    } else if (coord_type == 1) {  // Cylindrical
        y = 0;
    }
#endif
#if AMREX_SPACEDIM >= 3
    z = problo[2] + dx[2] * (static_cast<Real>(k)+0.5);
#endif
    Real r = sqrt(x*x + y*y + z*z);

    // Linear interpolation to find ionization fraction and electron temperature
    int const idx = amrex::bisect(d_initial_conditions_r_data, 0, d_initial_conditions_size - 1, r);
    AMREX_ASSERT( d_initial_conditions_r_data[idx] <= r && d_initial_conditions_r_data[idx+1] > r);
    Real frac = (r - d_initial_conditions_r_data[idx]) / (d_initial_conditions_r_data[idx+1] - d_initial_conditions_r_data[idx]);
    Real interp_values[NumSpec+1];
    for (int col = 0; col < NumSpec+1; ++col) {
        Real val_low = d_initial_conditions_table_data[idx * (NumSpec+1) + col];
        Real val_high = d_initial_conditions_table_data[(idx + 1) * (NumSpec+1) + col];
        interp_values[col] = val_low + frac * (val_high - val_low);
    }
    Real Te = interp_values[0];
    Real number_fractions[NumSpec];
    for (int s = 0; s < NumSpec; ++s) {
        number_fractions[s] = interp_values[s + 1];
    }

    // Assume uniform plasma with 0 velocity initially
    state(i,j,k,URHO) = problem::dens_ambient;
    state(i,j,k,UMX) = 0.0;
    state(i,j,k,UMY) = 0.0;
    state(i,j,k,UMZ) = 0.0;

    // Convert from number fraction to mass fraction
    Real avg_mass = 0;
    for (int s = 0; s < NumSpec; ++s) { avg_mass += number_fractions[s] * aion[s]; }
    for (int s = 0; s < NumSpec; ++s) {
        state(i,j,k,UFS+s) = state(i,j,k,URHO) * (number_fractions[s] * aion[s]) / avg_mass;
    }

    // Set the initial internal energy and total energy
    Real Ta = 348.; // 0.03 eV, in Kelvins
    Real avg_mass_grams = avg_mass * C::m_u;
    Real electrons_per_heavy = 0.0;
    Real rho_e_heavies =  state(i,j,k,URHO) * C::k_B * Ta / ( avg_mass_grams * (eos_rp::eos_gamma - 1.0) );
    for (int s = 0; s < NumSpec; ++s) {
        electrons_per_heavy += number_fractions[s] * zion[s];
    }
    Real rho_e_electrons = state(i,j,k,URHO) * electrons_per_heavy * C::k_B * Te / ( avg_mass_grams * (eos_rp::eos_gamma - 1.0) );

    state(i,j,k,UEINT) = rho_e_heavies + rho_e_electrons;
    state(i,j,k,UEDEN) = rho_e_heavies + rho_e_electrons +
                         0.5e0_rt * (state(i,j,k,UMX) * state(i,j,k,UMX) / state(i,j,k,URHO) +
                                     state(i,j,k,UMY) * state(i,j,k,UMY) / state(i,j,k,URHO) +
                                     state(i,j,k,UMZ) * state(i,j,k,UMZ) / state(i,j,k,URHO));

    // UFX is rho*f, where f is the fraction of internal energy stored in heavies (ions + atoms)
    state(i,j,k,UFX) = state(i,j,k,URHO) * rho_e_heavies / (rho_e_electrons + rho_e_heavies);

    // The temperature initialization will be done later
    // in the call to clean_state. We want to avoid
    // EOS calls in ca_initdata. So that there is valid
    // data for an initial guess for the EOS, we initialize
    // to the ambient temperature everywhere.
    state(i,j,k,UTEMP) = problem::temp_ambient;
}

#endif
