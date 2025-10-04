#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <hdf5.h>

#include <prob_parameters.H>
#include <eos.H>
#include <network.H>
#include <AMReX_Math.H>
#include <AMReX_Print.H>
#include <fundamental_constants.H>
#include <vector>
#ifdef AMREX_USE_GPU
#include <cuda_runtime.h>
#endif

// Static arrays for initial conditions that are read from an HDF5 file
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
    // Open HDF5 file
    hid_t file_id = H5Fopen(problem::initial_conditions_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Host-side vectors to temporarily store data from the file
    static std::vector<Real> r; // radius in cm
    static std::vector<Real> table_host_flat;

    // Read grid shape and attributes from T dataset
    hid_t dataset_T = H5Dopen(file_id, "data/0/meshes/T", H5P_DEFAULT);
    hid_t dataspace_T = H5Dget_space(dataset_T);
    hsize_t shape;
    H5Sget_simple_extent_dims(dataspace_T, &shape, NULL);
    hid_t attr_spacing = H5Aopen(dataset_T, "gridSpacing", H5P_DEFAULT);
    Real grid_spacing;
    H5Aread(attr_spacing, H5T_NATIVE_DOUBLE, &grid_spacing);
    H5Aclose(attr_spacing);
    hid_t attr_offset = H5Aopen(dataset_T, "gridGlobalOffset", H5P_DEFAULT);
    Real grid_offset;
    H5Aread(attr_offset, H5T_NATIVE_DOUBLE, &grid_offset);
    H5Aclose(attr_offset);
    H5Sclose(dataspace_T);
    H5Dclose(dataset_T);

    // Compute radius array from grid spacing (convert from m to cm)
    r.resize(shape);
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = (grid_offset + i * grid_spacing) * 100.0; // Convert m to cm
    }
    int num_rows = r.size();

    // Allocate table_host_flat: each row has (NumSpec+1) columns
    // Column 0: electron temperature, Columns 1 to NumSpec: species fractions
    table_host_flat.resize(num_rows * (NumSpec + 1));

    // Create list of dataset names: "T" first, then species fractions
    std::vector<std::string> dataset_names;
    dataset_names.push_back("T");
    for (int s = 0; s < NumSpec; ++s) {
        dataset_names.push_back(short_spec_names_cxx[s] + "_fraction");
    }

    // Loop over dataset names and fill table_host_flat
    for (int col = 0; col < dataset_names.size(); ++col) {
        std::string dataset_name = dataset_names[col];
        std::string dataset_path = std::string("data/0/meshes/") + dataset_name;

        // Open and read the dataset
        hid_t dataset = H5Dopen(file_id, dataset_path.c_str(), H5P_DEFAULT);
        hid_t dataspace = H5Dget_space(dataset);
        hsize_t dims;
        H5Sget_simple_extent_dims(dataspace, &dims, NULL);

        // Read data
        std::vector<Real> data(dims);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

        // Fill in the table (column col)
        for (int row = 0; row < num_rows; ++row) {
            table_host_flat[row * (NumSpec + 1) + col] = data[row];
        }

        H5Sclose(dataspace);
        H5Dclose(dataset);
    }

    H5Fclose(file_id);
    // Resize the device vectors
    initial_conditions_r.resize(num_rows);
    initial_conditions_table.resize(table_host_flat.size());
    // Get raw pointers for GPU copy
    Real* initial_conditions_r_data = initial_conditions_r.data();
    Real* initial_conditions_table_data = initial_conditions_table.data();

    // Copy all needed information into the static variables
#ifdef AMREX_USE_GPU
    cudaMemcpyToSymbol(d_initial_conditions_size, &num_rows, sizeof(int));
    cudaMemcpyToSymbol(d_initial_conditions_r_data, &initial_conditions_r_data, sizeof(Real*));
    cudaMemcpyToSymbol(d_initial_conditions_table_data, &initial_conditions_table_data, sizeof(Real*));

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r.begin(), r.end(),
                          initial_conditions_r.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, table_host_flat.begin(), table_host_flat.end(),
                          initial_conditions_table.begin());
    amrex::Gpu::streamSynchronize();
#else
    d_initial_conditions_size = num_rows;
    d_initial_conditions_r_data = initial_conditions_r_data;
    d_initial_conditions_table_data = initial_conditions_table_data;
    std::copy(r.begin(), r.end(), initial_conditions_r.begin());
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
    Real T = interp_values[0];
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
    Real mean_aion = 0;
    for (int s = 0; s < NumSpec; ++s) { mean_aion += number_fractions[s] * aion[s]; }
    for (int s = 0; s < NumSpec; ++s) {
        state(i,j,k,UFS+s) = state(i,j,k,URHO) * (number_fractions[s] * aion[s]) / mean_aion;
    }

    // Set the initial internal energy and total energy
    Real Ta = 348.; // 0.03 eV, in Kelvins
    Real avg_mass = mean_aion * C::m_u;
    Real electrons_per_heavy = 0.0;
    Real rho_e_heavies =  state(i,j,k,URHO) * C::k_B * Ta / ( avg_mass * (eos_rp::eos_gamma - 1.0) );
    for (int s = 0; s < NumSpec; ++s) {
        electrons_per_heavy += number_fractions[s] * zion[s];
    }
    Real rho_e_electrons = state(i,j,k,URHO) * electrons_per_heavy * C::k_B * T / ( avg_mass * (eos_rp::eos_gamma - 1.0) );

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
