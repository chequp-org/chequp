"""
This script tests that the 1D code produce the correct Sedov-Taylor blast wave solution.
It assumes that the code has already been compiled in ../sim_folder/build/
"""
import subprocess
import re
import numpy as np
import sys
import yt
import glob
import os
import openpmd_api
import time
import h5py
sys.path.append("../../../initial_condition")
from ionization_routines import save_to_openpmd


def cleanup_outputs(extra_file = ""):
    # Remove previously generated plotfiles and checkpoints

    os.system("rm -rf plt_2d_* chk* amr_diag.out species_diag.out grid_diag.out Backtrace.0" + extra_file)
import numpy as np
import re

def generate_initial_conditions():
    """
    Generate 2D (r, z) initial conditions for a Sedov-Taylor-like blast wave test:
    - no ionization reactions (castro.add_ext_src=0)
    - no temperature diffusion (castro.diffuse_temp=0)
    """
    print("Generating 2D (r, z) initial conditions...")

    # Spatial grid
    r_max = 10e-6   # 10 microns
    z_max = 10e-6
    Nr, Nz = 256, 256
    r = np.linspace(0, r_max, Nr)
    z = np.linspace(0, z_max, Nz)
    R, Z = np.meshgrid(r, z, indexing='ij')

    # Gaussian temperature profile
    sigma_r = 4e-6
    sigma_z = 6e-6  # can be different if desired
    T0_eV = 1000
    T_eV = T0_eV * np.exp(-(R**2 / sigma_r**2 + Z**2 / sigma_z**2))

    # Parse the species names for which Castro has been compiled
    with open('../../build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())

    # Populations array: shape (Nr, Nz, Nspecies)
    populations = np.zeros((Nr, Nz, len(species_keys)))

    # Set fractions: mostly ionized hydrogen
    if 'H1' in species_keys and 'H0' in species_keys:
        i_H1 = species_keys.index('H1')
        i_H0 = species_keys.index('H0')
        populations[:, :, i_H1] = 1 - 1e-3
        populations[:, :, i_H0] = 1e-3
    else:
        raise ValueError("Expected species 'H1' and 'H0' not found in species.net")

    # Save file
    save_to_openpmd(
        {'r': [r.min(), r.max()], 'z': [z.min(), z.max()]},
        populations,
        T_eV,
        '2d_inits.h5',
        species_keys
    )

def run_castro_simulation(runtime_options):
    """
    Run the Castro simulation.
    Raise an error and print stdout/stderr if the command fails.
    """
    # Find the Castro executable
    build_dir = "../../build"
    executables = glob.glob( os.path.join(build_dir, "Castro2d*") )
    if len(executables) == 0:
        raise FileNotFoundError(f"No Castro2d executable found in {build_dir}")
    elif len(executables) > 1:
        raise RuntimeError(f"Multiple Castro2d executables found: {executables}")
    executable = executables[0]

    cleanup_outputs()

    # Run the code
    inputs = "inputs.2d.cyl"
    command = f"{executable} {inputs} {runtime_options}"
    try:
        subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

if __name__ == "__main__":
    generate_initial_conditions()
    run_castro_simulation("castro.add_ext_src=0 castro.diffuse_temp=0 problem.initial_conditions_file=2d_inits.h5")