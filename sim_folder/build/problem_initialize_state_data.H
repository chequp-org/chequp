#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <hdf5.h>

#include <prob_parameters.H>
#include <eos.H>
#include <network.H>
#include <AMReX_Math.H>
#include <AMReX_Print.H>
#include <AMReX_TableData.H>
#include <fundamental_constants.H>
#include <vector>
#ifdef AMREX_USE_GPU
#include <cuda_runtime.h>
#endif

// Static arrays for initial conditions that are read from an HDF5 file
// These arrays are initialized from a file in the function read_initial_conditions
// We then use interpolation to read these values for each cell,
// in the function problem_initialize_state_data (GPU kernel)
// - Memory-owning object
static amrex::Gpu::DeviceVector<Real> d_table_flat;
// - Multi-dimensional view of the memory
AMREX_GPU_DEVICE static amrex::Table2D<Real,amrex::Order::C> d_initial_conditions_table;
// Meta-data about the grid point positions for the above table
AMREX_GPU_DEVICE static RealVect d_grid_spacing;
AMREX_GPU_DEVICE static RealVect d_grid_offset;

inline void read_initial_conditions()
{
    // Open HDF5 file
    hid_t file_id = H5Fopen(problem::initial_conditions_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Read grid shape and attributes from T dataset
    hid_t dataset_T = H5Dopen(file_id, "data/0/meshes/T", H5P_DEFAULT);
    hid_t dataspace_T = H5Dget_space(dataset_T);
    hsize_t shape_array[AMREX_SPACEDIM];
    H5Sget_simple_extent_dims(dataspace_T, shape_array, NULL);

    hid_t attr_spacing = H5Aopen(dataset_T, "gridSpacing", H5P_DEFAULT);
    Real grid_spacing_array[AMREX_SPACEDIM];
    H5Aread(attr_spacing, H5T_NATIVE_DOUBLE, grid_spacing_array);
    H5Aclose(attr_spacing);
    hid_t attr_offset = H5Aopen(dataset_T, "gridGlobalOffset", H5P_DEFAULT);
    Real grid_offset_array[AMREX_SPACEDIM];
    H5Aread(attr_offset, H5T_NATIVE_DOUBLE, grid_offset_array);
    H5Aclose(attr_offset);
    H5Sclose(dataspace_T);
    H5Dclose(dataset_T);

    // Allocate table_flat: each row has (NumSpec+1) datasets stored contiguously
    // First dataset: electron temperature, next NumSpec datasets: species fractions
    // Calculate dataset_size as the product of shape_array from index 0 to AMREX_SPACEDIM-1
    int dataset_size = 1;
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
        dataset_size *= shape_array[dim];
    }
    amrex::Gpu::HostVector<Real> table_flat;
    table_flat.resize((NumSpec + 1) * dataset_size);

    // Create list of dataset names: "T" first, then species fractions
    std::vector<std::string> dataset_names;
    dataset_names.push_back("T");
    for (int s = 0; s < NumSpec; ++s) {
        dataset_names.push_back(short_spec_names_cxx[s] + "_fraction");
    }

    // Loop over dataset names and fill table_flat
    for (int col = 0; col < dataset_names.size(); ++col) {
        std::string dataset_name = dataset_names[col];
        std::string dataset_path = std::string("data/0/meshes/") + dataset_name;

        // Open and read the dataset
        hid_t dataset = H5Dopen(file_id, dataset_path.c_str(), H5P_DEFAULT);

        // Read data directly into the table column
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                table_flat.data() + col*dataset_size);

        H5Dclose(dataset);
    }
    H5Fclose(file_id);

    // Copy data on GPU, into static memory-owning object
    d_table_flat.resize(table_flat.size());
#ifdef AMREX_USE_GPU
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, table_flat.begin(), table_flat.end(), d_table_flat.begin());
    amrex::Gpu::streamSynchronize();
#else
    d_table_flat = table_flat;
#endif

    // Create multi-dimensional Table object that points to `d_table_flat`
    auto initial_conditions_table = amrex::Table2D<Real,amrex::Order::C>
        (d_table_flat.data(), {0, 0}, {NumSpec+1, dataset_size});
    // Copy grid spacing and offset into static RealVect objects, and convert from m to cm
    auto grid_spacing = RealVect(AMREX_D_DECL(1e2*grid_spacing_array[0], 1e2*grid_spacing_array[1], 1e2*grid_spacing_array[2]));
    auto grid_offset = RealVect(AMREX_D_DECL(1e2*grid_offset_array[0], 1e2*grid_offset_array[1], 1e2*grid_offset_array[2]));

    // Copy all needed information into the static variables
#ifdef AMREX_USE_GPU
    cudaMemcpyToSymbol(d_initial_conditions_table, &initial_conditions_table, sizeof(amrex::Table2D<Real,amrex::Order::C>));
    cudaMemcpyToSymbol(d_grid_spacing, &grid_spacing, sizeof(RealVect));
    cudaMemcpyToSymbol(d_grid_offset, &grid_offset, sizeof(RealVect));
    amrex::Gpu::streamSynchronize();
#else
    d_initial_conditions_table = initial_conditions_table;
    d_grid_spacing = grid_spacing;
    d_grid_offset = grid_offset;
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

    // Linear interpolation to find ionization fraction and electron temperature
    Real x_norm = (x - d_grid_offset[0]) / d_grid_spacing[0];
    int const idx = static_cast<int>(x_norm);
    Real frac = x_norm - idx;
    Real interp_values[NumSpec+1];
    for (int col = 0; col < NumSpec+1; ++col) {
        Real val_low = d_initial_conditions_table(col, idx);
        Real val_high = d_initial_conditions_table(col, (idx + 1));
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
    Real rho_e_electrons = state(i,j,k,URHO) * electrons_per_heavy * C::k_B * Te / ( avg_mass * (eos_rp::eos_gamma - 1.0) );

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
