"""
Generate example initial conditions for CHEQUP, in openPMD format.

The plasma is a hydrogen channel heated on axis, as created by an
optical-field-ionizing laser (similar to the benchmark of Mewes et al.,
Phys. Rev. Research 5, 033112, 2023):
- a super-Gaussian ionization fraction (minimal value of 1e-3, so that the
  electron temperature is defined everywhere)
- a super-Gaussian electron temperature peaking at 27 eV

Usage:
    python3 generate_initial_conditions.py           # 1D (r), for inputs.1d.cyl
    python3 generate_initial_conditions.py --dim 2   # 2D (r-z), for inputs.2d.cyl
"""
import argparse
import os
import re
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../initial_condition'))
from ionization_routines import save_to_openpmd

# Define constants
sigma1 = 38e-6  # Width of the ionization profile, in m
sigma2 = 35e-6  # Width of the temperature profile, in m
Te_max = 27.  # Peak electron temperature, in eV
Ta = 0.03  # Background temperature, in eV
n_total = 1.e24  # Total number density in m^-3 (equivalent to 1e18 cm^-3)
r_max = 6e-4  # Radial extent of the grid, in m (matches geometry.prob_hi in the inputs files)
z_max = 6e-4  # Longitudinal extent of the grid, in m (2D only)
z_ramp = 1e-4  # Length of the longitudinal ramp of the laser heating at each end of the grid, in m (2D only)


def get_species_keys():
    """Parse the name of the species for which CHEQUP is compiled"""
    species_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../build/species.net')
    with open(species_file, 'r') as f:
        return re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())


def radial_profiles(r):
    """Return the ionization fraction and the electron temperature (eV) along r"""
    ioniz_fraction = (1. - 1.e-3)*np.exp(-np.power(r*r/(2*sigma1*sigma1), 12)) + 1.e-3
    T_eV = (Te_max - Ta) * np.exp(-np.power(r*r/(2*sigma2*sigma2), 3)) + Ta
    return ioniz_fraction, T_eV


def generate_1d(output_file='example_1d_initial_conditions.h5'):
    species_keys = get_species_keys()
    r = np.arange(0, r_max + 1e-6, 1e-6)
    ioniz_fraction, T_eV = radial_profiles(r)

    # Number densities in m^-3, with shape (Nr, n_species)
    densities = np.zeros((len(r), len(species_keys)))
    densities[:, species_keys.index('H0')] = (1 - ioniz_fraction) * n_total
    densities[:, species_keys.index('H1')] = ioniz_fraction * n_total

    save_to_openpmd({'r': [r.min(), r.max()]}, densities, T_eV, output_file, species_keys)
    print(f'Wrote {output_file}')


def generate_2d(output_file='example_2d_initial_conditions.h5'):
    species_keys = get_species_keys()
    r = np.arange(0, r_max + 2e-6, 2e-6)
    z = np.arange(0, z_max + 2e-6, 2e-6)
    R, Z = np.meshgrid(r, z, indexing='ij')

    # Same radial profiles as in 1D, with a heating that smoothly
    # vanishes at both ends of the grid in z (focus/defocus of the laser)
    ioniz_fraction_r, T_eV_r = radial_profiles(R)
    envelope = np.sin(0.5*np.pi*np.clip(np.minimum(Z, z_max - Z)/z_ramp, 0., 1.))**2
    ioniz_fraction = (ioniz_fraction_r - 1.e-3) * envelope + 1.e-3
    T_eV = (T_eV_r - Ta) * envelope + Ta

    # Number densities in m^-3, with shape (Nr, Nz, n_species)
    densities = np.zeros((len(r), len(z), len(species_keys)))
    densities[:, :, species_keys.index('H0')] = (1 - ioniz_fraction) * n_total
    densities[:, :, species_keys.index('H1')] = ioniz_fraction * n_total

    save_to_openpmd({'r': [r.min(), r.max()], 'z': [z.min(), z.max()]},
                    densities, T_eV, output_file, species_keys)
    print(f'Wrote {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate example initial conditions for CHEQUP')
    parser.add_argument('--dim', type=int, choices=[1, 2], default=1,
                        help='Dimension of the initial conditions: 1 (r) or 2 (r-z)')
    parser.add_argument('--output', type=str, default=None, help='Output file name')
    args = parser.parse_args()

    if args.dim == 1:
        generate_1d(args.output or 'example_1d_initial_conditions.h5')
    else:
        generate_2d(args.output or 'example_2d_initial_conditions.h5')
