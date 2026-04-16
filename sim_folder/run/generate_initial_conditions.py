import numpy as np
import openpmd_api as io

# Define constants
sigma1 = 38e-6  # in m
sigma2 = 35e-6  # in m
Te_max = 313322.  # 27 eV, in Kelvins
Ta = 348.  # 0.03 eV, in Kelvins
n_total = 1.e24  # Total number density in m^-3 (equivalent to 1e18 cm^-3)

# Create r array from 0 to 6e-4 with 1e-6 increment
r = np.arange(0, 6e-4 + 1e-6, 1e-6)
# Calculate ionization fraction
ioniz_fraction = (1. - 1.e-3)*np.exp(-np.power(r*r/(2*sigma1*sigma1), 12)) + 1.e-3
# Calculate electron temperature
Te = (Te_max - Ta) * np.exp(-np.power(r*r/(2*sigma2*sigma2), 3)) + Ta

# Number densities in m^-3
densities = {
    'H0': (1 - ioniz_fraction) * n_total,
    'H1': ioniz_fraction * n_total,
    'N0': np.zeros_like(ioniz_fraction),
    'N1': np.zeros_like(ioniz_fraction),
    'N2': np.zeros_like(ioniz_fraction),
    'N3': np.zeros_like(ioniz_fraction),
    'N4': np.zeros_like(ioniz_fraction),
    'N5': np.zeros_like(ioniz_fraction),
}

# create openpmd file
series = io.Series('example_1d_initial_conditions.h5', io.Access.create)
# only 1 iteration needed
it = series.iterations[0]

# Get spatial resolution
dr = np.diff(r).mean()
rmin = r.min()

# Save the temperature
T = it.meshes["T"]
T.grid_spacing = np.array([dr])
T.grid_global_offset = [rmin]
T.axis_labels = ['r']
T.unit_dimension = {io.Unit_Dimension.theta: 1}
dataset = io.Dataset(Te.dtype, Te.shape)
T_scalar = T[io.Mesh_Record_Component.SCALAR]
T_scalar.reset_dataset(dataset)
T_scalar.position = [0.0]
T_scalar.store_chunk(Te)

# Save the species densities
for species_key in densities.keys():
    dens = it.meshes[species_key + "_density"]
    dens.grid_spacing = np.array([dr])
    dens.grid_global_offset = [rmin]
    dens.axis_labels = ['r']
    dens.unit_dimension = {io.Unit_Dimension.L: -3}  # m^-3
    dataset = io.Dataset(densities[species_key].dtype, densities[species_key].shape)
    dens_scalar = dens[io.Mesh_Record_Component.SCALAR]
    dens_scalar.reset_dataset(dataset)
    dens_scalar.position = [0.0]
    dens_scalar.store_chunk(densities[species_key].copy())

series.flush()
del series
