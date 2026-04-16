#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <hdf5.h>
#include <filesystem>

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
// Layout of the "columns" (first index of the table):
//   col 0         : electron temperature T  (K)
//   col 1..NumSpec: species number densities (m^-3)
//   col NumSpec+1 : xmom  (kg m^-2 s^-1  -- momentum density in x)
//   col NumSpec+2 : ymom
//   col NumSpec+3 : zmom
// Total columns: NumSpec + 4
static constexpr int N_COLS = NumSpec + 4;

static amrex::Gpu::DeviceVector<Real> d_table_flat;

#if AMREX_SPACEDIM == 1
AMREX_GPU_DEVICE static amrex::Table2D<Real,amrex::Order::C> d_initial_conditions_table;
#elif AMREX_SPACEDIM == 2
AMREX_GPU_DEVICE static amrex::Table3D<Real,amrex::Order::C> d_initial_conditions_table;
#elif AMREX_SPACEDIM == 3
AMREX_GPU_DEVICE static amrex::Table4D<Real,amrex::Order::C> d_initial_conditions_table;
#endif

AMREX_GPU_DEVICE static RealVect d_grid_spacing;
AMREX_GPU_DEVICE static RealVect d_grid_offset;

inline void read_initial_conditions()
{
    if (!std::filesystem::exists(problem::initial_conditions_file)) {
        amrex::Abort("Initial conditions file does not exist: " + problem::initial_conditions_file);
    }

    hid_t file_id = H5Fopen(problem::initial_conditions_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // Read grid shape and attributes from T dataset (unchanged)
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

    // Compute flat size of one dataset
    int dataset_size = 1;
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
        dataset_size *= shape_array[dim];
    }

    // Allocate flat buffer: N_COLS datasets stored contiguously
    amrex::Gpu::HostVector<Real> table_flat;
    table_flat.resize(N_COLS * dataset_size);

    // Build ordered list of dataset names matching the column layout:
    //   T, species densities, xmom, ymom, zmom
    std::vector<std::string> dataset_names;
    dataset_names.push_back("T");
    for (int s = 0; s < NumSpec; ++s) {
        dataset_names.push_back(short_spec_names_cxx[s] + "_density");
    }
    dataset_names.push_back("xmom");
    dataset_names.push_back("ymom");
    dataset_names.push_back("zmom");
    // Sanity check
    AMREX_ASSERT(static_cast<int>(dataset_names.size()) == N_COLS);

    // Read each dataset into its column slice of the flat buffer
    for (int col = 0; col < N_COLS; ++col) {
        std::string dataset_path = "data/0/meshes/" + dataset_names[col];
        hid_t dataset = H5Dopen(file_id, dataset_path.c_str(), H5P_DEFAULT);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                table_flat.data() + col * dataset_size);
        H5Dclose(dataset);
    }
    H5Fclose(file_id);

    // Copy data to GPU
    d_table_flat.resize(table_flat.size());
#ifdef AMREX_USE_GPU
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, table_flat.begin(), table_flat.end(), d_table_flat.begin());
    amrex::Gpu::streamSynchronize();
#else
    d_table_flat = table_flat;
#endif

    // Build the multi-dimensional Table view over the flat buffer.
    // First index spans columns (N_COLS), remaining indices span spatial dims.
#if AMREX_SPACEDIM == 1
    auto initial_conditions_table = amrex::Table2D<Real,amrex::Order::C>
        (d_table_flat.data(),
         {0, 0},
         {N_COLS, static_cast<int>(shape_array[0])});
#elif AMREX_SPACEDIM == 2
    auto initial_conditions_table = amrex::Table3D<Real,amrex::Order::C>
        (d_table_flat.data(),
         {0, 0, 0},
         {N_COLS, static_cast<int>(shape_array[0]), static_cast<int>(shape_array[1])});
#elif AMREX_SPACEDIM == 3
    auto initial_conditions_table = amrex::Table4D<Real,amrex::Order::C>
        (d_table_flat.data(),
         {0, 0, 0, 0},
         {N_COLS, static_cast<int>(shape_array[0]), static_cast<int>(shape_array[1]), static_cast<int>(shape_array[2])});
#endif

    // Convert grid spacing and offset from SI (m) to CGS (cm)
    auto grid_spacing = RealVect(AMREX_D_DECL(
        1e2*grid_spacing_array[0],
        1e2*grid_spacing_array[1],
        1e2*grid_spacing_array[2]));
    auto grid_offset = RealVect(AMREX_D_DECL(
        1e2*grid_offset_array[0],
        1e2*grid_offset_array[1],
        1e2*grid_offset_array[2]));

    // Push Table view and grid metadata to GPU constant memory
#ifdef AMREX_USE_GPU
#if AMREX_SPACEDIM == 1
    cudaMemcpyToSymbol(d_initial_conditions_table, &initial_conditions_table, sizeof(amrex::Table2D<Real,amrex::Order::C>));
#elif AMREX_SPACEDIM == 2
    cudaMemcpyToSymbol(d_initial_conditions_table, &initial_conditions_table, sizeof(amrex::Table3D<Real,amrex::Order::C>));
#elif AMREX_SPACEDIM == 3
    cudaMemcpyToSymbol(d_initial_conditions_table, &initial_conditions_table, sizeof(amrex::Table4D<Real,amrex::Order::C>));
#endif
    cudaMemcpyToSymbol(d_grid_spacing, &grid_spacing, sizeof(RealVect));
    cudaMemcpyToSymbol(d_grid_offset,  &grid_offset,  sizeof(RealVect));
#else
    d_initial_conditions_table = initial_conditions_table;
    d_grid_spacing = grid_spacing;
    d_grid_offset  = grid_offset;
#endif
}

AMREX_GPU_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k,
                                    Array4<Real> const& state,
                                    const GeometryData& geomdata)
{
    int coord_type = geomdata.Coord();

    const Real* dx     = geomdata.CellSize();
    const Real* problo = geomdata.ProbLo();

    Real x0 = problo[0] + dx[0] * (static_cast<Real>(i) + 0.5);
#if AMREX_SPACEDIM >= 2
    Real x1 = problo[1] + dx[1] * (static_cast<Real>(j) + 0.5);
#endif
#if AMREX_SPACEDIM >= 3
    Real x2 = problo[2] + dx[2] * (static_cast<Real>(k) + 0.5);
#endif

    // Normalised coordinates and clamped lo/hi indices for interpolation
    Real x0_norm = (x0 - d_grid_offset[0]) / d_grid_spacing[0];
    int const idx0    = static_cast<int>(x0_norm);
    int const idx0_lo = std::clamp(idx0,     d_initial_conditions_table.begin[1], d_initial_conditions_table.end[1]-1);
    int const idx0_hi = std::clamp(idx0 + 1, d_initial_conditions_table.begin[1], d_initial_conditions_table.end[1]-1);
    Real frac0 = x0_norm - idx0;
#if AMREX_SPACEDIM >= 2
    Real x1_norm = (x1 - d_grid_offset[1]) / d_grid_spacing[1];
    int const idx1    = static_cast<int>(x1_norm);
    int const idx1_lo = std::clamp(idx1,     d_initial_conditions_table.begin[2], d_initial_conditions_table.end[2]-1);
    int const idx1_hi = std::clamp(idx1 + 1, d_initial_conditions_table.begin[2], d_initial_conditions_table.end[2]-1);
    Real frac1 = x1_norm - idx1;
#endif
#if AMREX_SPACEDIM >= 3
    Real x2_norm = (x2 - d_grid_offset[2]) / d_grid_spacing[2];
    int const idx2    = static_cast<int>(x2_norm);
    int const idx2_lo = std::clamp(idx2,     d_initial_conditions_table.begin[3], d_initial_conditions_table.end[3]-1);
    int const idx2_hi = std::clamp(idx2 + 1, d_initial_conditions_table.begin[3], d_initial_conditions_table.end[3]-1);
    Real frac2 = x2_norm - idx2;
#endif

    // Interpolate all N_COLS columns at once
    Real interp_values[N_COLS];
#if AMREX_SPACEDIM == 1
    for (int col = 0; col < N_COLS; ++col) {
        Real v0 = d_initial_conditions_table(col, idx0_lo);
        Real v1 = d_initial_conditions_table(col, idx0_hi);
        interp_values[col] = v0 + frac0 * (v1 - v0);
    }
#elif AMREX_SPACEDIM == 2
    for (int col = 0; col < N_COLS; ++col) {
        Real v00 = d_initial_conditions_table(col, idx0_lo, idx1_lo);
        Real v10 = d_initial_conditions_table(col, idx0_hi, idx1_lo);
        Real v01 = d_initial_conditions_table(col, idx0_lo, idx1_hi);
        Real v11 = d_initial_conditions_table(col, idx0_hi, idx1_hi);
        Real v0  = v00 + frac0 * (v10 - v00);
        Real v1  = v01 + frac0 * (v11 - v01);
        interp_values[col] = v0 + frac1 * (v1 - v0);
    }
#elif AMREX_SPACEDIM == 3
    for (int col = 0; col < N_COLS; ++col) {
        Real v000 = d_initial_conditions_table(col, idx0_lo, idx1_lo, idx2_lo);
        Real v100 = d_initial_conditions_table(col, idx0_hi, idx1_lo, idx2_lo);
        Real v010 = d_initial_conditions_table(col, idx0_lo, idx1_hi, idx2_lo);
        Real v110 = d_initial_conditions_table(col, idx0_hi, idx1_hi, idx2_lo);
        Real v001 = d_initial_conditions_table(col, idx0_lo, idx1_lo, idx2_hi);
        Real v101 = d_initial_conditions_table(col, idx0_hi, idx1_lo, idx2_hi);
        Real v011 = d_initial_conditions_table(col, idx0_lo, idx1_hi, idx2_hi);
        Real v111 = d_initial_conditions_table(col, idx0_hi, idx1_hi, idx2_hi);

        Real v00 = v000 + frac0 * (v100 - v000);
        Real v10 = v010 + frac0 * (v110 - v010);
        Real v01 = v001 + frac0 * (v101 - v001);
        Real v11 = v011 + frac0 * (v111 - v011);
        Real v0  = v00  + frac1 * (v10  - v00);
        Real v1  = v01  + frac1 * (v11  - v01);
        interp_values[col] = v0 + frac2 * (v1 - v0);
    }
#endif

    // Unpack interpolated columns using the agreed layout:
    //   [0]           -> T (K)
    //   [1..NumSpec]  -> number densities (m^-3)
    //   [NumSpec+1]   -> xmom  (kg m^-2 s^-1)
    //   [NumSpec+2]   -> ymom
    //   [NumSpec+3]   -> zmom
    Real Te = interp_values[0];

    Real number_densities[NumSpec];
    for (int s = 0; s < NumSpec; ++s) {
        number_densities[s] = interp_values[s + 1];
    }

    // Momentum densities from file are in SI (kg m^-2 s^-1).
    // Castro works in CGS (g cm^-2 s^-1): 1 kg m^-2 s^-1 = 0.1 g cm^-2 s^-1    
    Real xmom_cgs = interp_values[NumSpec + 1] * 0.1;
    Real ymom_cgs = interp_values[NumSpec + 2] * 0.1;
    Real zmom_cgs = interp_values[NumSpec + 3] * 0.1;

    // Mass density (CGS): number densities are in m^-3, convert to cm^-3
    Real rho = 0.0;
    for (int s = 0; s < NumSpec; ++s) {
        rho += number_densities[s] * 1.e-6 * aion[s] * C::m_u;
    }

    state(i,j,k,URHO) = rho;
    state(i,j,k,UMX)  = xmom_cgs;
    state(i,j,k,UMY)  = ymom_cgs;
    state(i,j,k,UMZ)  = zmom_cgs;

    for (int s = 0; s < NumSpec; ++s) {
        state(i,j,k,UFS+s) = number_densities[s] * 1.e-6 * aion[s] * C::m_u;
    }

    Real Ta = 348.;

    Real n_heavy     = 0.0;
    Real n_electrons = 0.0;
    for (int s = 0; s < NumSpec; ++s) {
        n_heavy     += number_densities[s] * 1.e-6;
        n_electrons += number_densities[s] * 1.e-6 * zion[s];
    }

    Real rho_e_heavies   = n_heavy     * C::k_B * Ta / (eos_rp::eos_gamma - 1.0);
    Real rho_e_electrons = n_electrons * C::k_B * Te / (eos_rp::eos_gamma - 1.0);

    state(i,j,k,UEINT) = rho_e_heavies + rho_e_electrons;
    // Kinetic energy now uses the actual momentum from the file
    state(i,j,k,UEDEN) = rho_e_heavies + rho_e_electrons
                        + 0.5_rt * (  xmom_cgs * xmom_cgs
                                    + ymom_cgs * ymom_cgs
                                    + zmom_cgs * zmom_cgs) / rho;

    state(i,j,k,UFX)   = state(i,j,k,URHO) * rho_e_heavies / (rho_e_electrons + rho_e_heavies);
    state(i,j,k,UTEMP) = problem::temp_ambient;
}

#endif