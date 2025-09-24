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
AMREX_GPU_DEVICE static int d_num_rows = 0;
AMREX_GPU_DEVICE static int d_num_cols = 0; // Number of columns excluding radius

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
    int num_cols = 0;

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

        if (num_rows == 0) {
            num_cols = current_cols;
        } else if (current_cols != num_cols) {
            amrex::Abort("Inconsistent number of columns in CSV file.");
        }
        num_rows++;
    }
    file.close();
    
    // +++ ADD THIS BLOCK FOR VERIFICATION +++
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n[Initial Conditions] Successfully read file: "
                       << problem::initial_conditions_file << "\n";
        amrex::Print() << "    - Found " << num_rows << " data points (rows).\n";
        amrex::Print() << "    - Found " << num_cols << " data columns (Te + species).\n";
        amrex::Print() << "    - First radius: " << r_host.front() << " cm, Last radius: " << r_host.back() << " cm.\n";
    }
    // Resize the device vectors
    initial_conditions_r.resize(num_rows);
    initial_conditions_table.resize(table_host_flat.size());
    // Get raw pointers for GPU copy
    Real* d_r_ptr = initial_conditions_r.data();
    Real* d_table_ptr = initial_conditions_table.data();

    // Copy all needed information into the static variables
#ifdef AMREX_USE_GPU
    cudaMemcpyToSymbol(d_num_rows, &num_rows, sizeof(int));
    cudaMemcpyToSymbol(d_num_cols, &num_cols, sizeof(int));
    cudaMemcpyToSymbol(d_initial_conditions_r_data, &d_r_ptr, sizeof(Real*));
    cudaMemcpyToSymbol(d_initial_conditions_table_data, &d_table_ptr, sizeof(Real*));

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r_host.begin(), r_host.end(),
                          initial_conditions_r.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, table_host_flat.begin(), table_host_flat.end(),
                          initial_conditions_table.begin());
    amrex::Gpu::streamSynchronize();
#else
    d_num_rows = num_rows;
    d_num_cols = num_cols;
    d_initial_conditions_r_data = d_r_ptr;
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
    y = problo[1] + dx[1] * (static_cast<Real>(j)+0.5);
#endif
#if AMREX_SPACEDIM >= 3
    z = problo[2] + dx[2] * (static_cast<Real>(k)+0.5);
#endif
    Real r = sqrt(x*x + y*y + z*z);

    // Linear interpolation to find ionization fraction and electron temperature
    int const idx = amrex::bisect(d_initial_conditions_r_data, 0, d_num_rows - 1, r);
    AMREX_ASSERT( d_initial_conditions_r_data[idx] <= r && d_initial_conditions_r_data[idx+1] > r);
    Real frac = (r - d_initial_conditions_r_data[idx]) / (d_initial_conditions_r_data[idx+1] - d_initial_conditions_r_data[idx]);
    Real interp_values[d_num_cols];
    for (int col = 0; col < d_num_cols; ++col) {
        Real val_low = d_initial_conditions_table_data[idx * d_num_cols + col];
        Real val_high = d_initial_conditions_table_data[(idx + 1) * d_num_cols + col];
        interp_values[col] = val_low + frac * (val_high - val_low);
    }
    const int num_species = 8; // Hn, Hp, Nn, N1, N2, N3, N4
    Real te_kelvin = interp_values[0];
    Real number_fractions[num_species];
    for (int s = 0; s < num_species; ++s) {
        number_fractions[s] = interp_values[s + 1];
    }
    Real mass_H = 1.0;
    Real mass_N = 14.0;
    Real mass_fractions[num_species];
    Real avg_mass = (number_fractions[0] + number_fractions[1]) * mass_H;
    for (int s = 2; s < num_species; ++s) { avg_mass += number_fractions[s] * mass_N; }
    if (avg_mass > 1.e-12) {
        mass_fractions[0] = (number_fractions[0] * mass_H) / avg_mass;
        mass_fractions[1] = (number_fractions[1] * mass_H) / avg_mass;
        for (int s = 2; s < num_species; ++s) {
            mass_fractions[s] = (number_fractions[s] * mass_N) / avg_mass;
        }
    } else {
        for (int s = 0; s < num_species; ++s) mass_fractions[s] = 0.0;
    }
    // --- State Variable Assignment ---
    state(i,j,k,URHO) = problem::dens_ambient;
    state(i,j,k,UMX) = 0.0;
    state(i,j,k,UMY) = 0.0;
    state(i,j,k,UMZ) = 0.0;
    for (int s = 0; s < num_species; ++s) {
        state(i,j,k,UFS+s) = state(i,j,k,URHO) * mass_fractions[s];
    }
    // Calculate internal energy
    Real electrons_per_heavy = number_fractions[1] * 1.0 +   // H+
                               number_fractions[3] * 1.0 +   // N+
                               number_fractions[4] * 2.0 +   // N2+
                               number_fractions[5] * 3.0 +   // N3+
                               number_fractions[6] * 4.0 +   // N4+
                               number_fractions[7] * 5.0;    // N5+

    Real temp_heavy = 348.0;
    Real avg_E_per_heavy = (3.0/2.0) * C::k_B * (electrons_per_heavy * te_kelvin + 1.0 * temp_heavy);
    Real avg_mass_grams = avg_mass * C::m_u;
    Real eint = (avg_mass_grams > 1.e-12) ? avg_E_per_heavy / avg_mass_grams : 0.0;
    state(i,j,k,UEINT) = eint;
    state(i,j,k,UEDEN) = state(i,j,k,URHO) * eint;


    // The temperature initialization will be done later
    // in the call to clean_state. We want to avoid
    // EOS calls in ca_initdata. So that there is valid
    // data for an initial guess for the EOS, we initialize
    // to the ambient temperature everywhere.
    state(i,j,k,UTEMP) = problem::temp_ambient;
}

#endif
