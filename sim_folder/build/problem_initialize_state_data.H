#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <hdf5.h>

#include <prob_parameters.H>
#include <eos.H>
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
    // Open HDF5 file
    hid_t file_id = H5Fopen(problem::initial_conditions_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    
    // Vector to fill when reading the file
    static std::vector<Real> r; // radius in cm
    static std::vector<Real> ioniz; // fraction of atoms that are ionized
    static std::vector<Real> Te; // electron temperature in K

    // Read H_1_fraction (ionization fraction)
    hid_t dataset_ioniz = H5Dopen(file_id, "data/0/meshes/H_1_fraction", H5P_DEFAULT);
    hid_t dataspace_ioniz = H5Dget_space(dataset_ioniz);
    hsize_t dims_ioniz;
    H5Sget_simple_extent_dims(dataspace_ioniz, &dims_ioniz, NULL);
    ioniz.resize(dims_ioniz);
    H5Dread(dataset_ioniz, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ioniz.data());
    
    // Read grid attributes from H_1_fraction dataset
    hid_t attr_spacing = H5Aopen(dataset_ioniz, "gridSpacing", H5P_DEFAULT);
    Real grid_spacing;
    H5Aread(attr_spacing, H5T_NATIVE_DOUBLE, &grid_spacing);
    H5Aclose(attr_spacing);
    
    hid_t attr_offset = H5Aopen(dataset_ioniz, "gridGlobalOffset", H5P_DEFAULT);
    Real grid_offset;
    H5Aread(attr_offset, H5T_NATIVE_DOUBLE, &grid_offset);
    H5Aclose(attr_offset);
    
    H5Sclose(dataspace_ioniz);
    H5Dclose(dataset_ioniz);

    // Read T (electron temperature)
    hid_t dataset_Te = H5Dopen(file_id, "data/0/meshes/T", H5P_DEFAULT);
    hid_t dataspace_Te = H5Dget_space(dataset_Te);
    hsize_t dims_Te;
    H5Sget_simple_extent_dims(dataspace_Te, &dims_Te, NULL);
    Te.resize(dims_Te);
    H5Dread(dataset_Te, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Te.data());
    H5Sclose(dataspace_Te);
    H5Dclose(dataset_Te);

    H5Fclose(file_id);

    // Compute radius array from grid spacing (convert from m to cm)
    r.resize(ioniz.size());
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = (grid_offset + i * grid_spacing) * 100.0; // Convert m to cm
    }

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
    int const idx = amrex::bisect( d_initial_conditions_r_data, 0, d_initial_conditions_size-1, r );
    AMREX_ASSERT( d_initial_conditions_r_data[idx] <= r && d_initial_conditions_r_data[idx+1] > r);
    Real frac = (r - d_initial_conditions_r_data[idx]) / (d_initial_conditions_r_data[idx+1] - d_initial_conditions_r_data[idx]);
    Real ioniz_fraction = d_initial_conditions_ioniz_data[idx] + frac * (d_initial_conditions_ioniz_data[idx+1] - d_initial_conditions_ioniz_data[idx]);
    Real Te = d_initial_conditions_Te_data[idx] + frac * (d_initial_conditions_Te_data[idx+1] - d_initial_conditions_Te_data[idx]);
    Real Ta = 348.; // 0.03 eV, in Kelvins

    // Assume uniform plasma with 0 velocity initially
    state(i,j,k,URHO) = problem::dens_ambient;
    state(i,j,k,UMX) = 0.e0_rt;
    state(i,j,k,UMY) = 0.e0_rt;
    state(i,j,k,UMZ) = 0.e0_rt;

    // Set the initial internal energy and total energy
    Real rho_e_electrons = state(i,j,k,URHO) * ioniz_fraction * C::k_B * Te / ( C::m_u * (eos_rp::eos_gamma - 1.0) );
    Real rho_e_heavies =  state(i,j,k,URHO) * C::k_B * Ta / ( C::m_u * (eos_rp::eos_gamma - 1.0) );
    state(i,j,k,UEINT) = rho_e_heavies + rho_e_electrons;
    state(i,j,k,UEDEN) = rho_e_heavies + rho_e_electrons +
                         0.5e0_rt * (state(i,j,k,UMX) * state(i,j,k,UMX) / state(i,j,k,URHO) +
                                     state(i,j,k,UMY) * state(i,j,k,UMY) / state(i,j,k,URHO) +
                                     state(i,j,k,UMZ) * state(i,j,k,UMZ) / state(i,j,k,URHO));

    // Set the initial mass fractions
    state(i,j,k,UFS) = state(i,j,k,URHO)*(1-ioniz_fraction);
    state(i,j,k,UFS+1) = state(i,j,k,URHO)*ioniz_fraction;

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
