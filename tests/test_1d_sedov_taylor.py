"""
This script tests that the 1D code produce the correct Sedov-Taylor blast wave solution.

It assumes that the code has already been compiled in ../sim_folder/build/
"""
import subprocess
import re
import numpy as np
import sys
import glob
import os
sys.path.append("../initial_condition")
from ionization_routines import save_to_openpmd
from checksum.checksumAPI import evaluate_checksum

def run_castro_simulation(runtime_options):
    """
    Run the Castro simulation.

    Raise an error and print stdout/stderr if the command fails.
    """
    # Find the Castro executable
    build_dir = "../sim_folder/build"
    executables = glob.glob( os.path.join(build_dir, "Castro1d*") )
    if len(executables) == 0:
        raise FileNotFoundError(f"No Castro1d executable found in {build_dir}")
    elif len(executables) > 1:
        raise RuntimeError(f"Multiple Castro1d executables found: {executables}")
    executable = executables[0]

    # Remove previously generated plotfiles and checkpoints
    os.system(f"rm -rf plt_* chck* amr_diag.out species_diag.out grid_diag.out")

    # Run the code
    inputs = "../sim_folder/run/inputs.1d.cyl"
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


def test_1d_sedov_taylor():
    """
    Test that the 1D code produce the correct Sedov-Taylor blast wave solution.
    """
    # Generate openPMD inital conditions:
    # 1000eV plasma in the first 5 microns, low-temperature plasma in the rest
    r = np.linspace(0, 10e-6, 1024)
    T_eV = np.ones_like(r) * 1000
    # Parse the species names for which Castro has been compiled
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())
    populations = np.zeros((len(r), len(species_keys)))
    # Set fraction to 1 for H+
    populations[:, species_keys.index('H1')] = 1
    # Save file
    save_to_openpmd( {'r': [r.min(), r.max()]}, populations,
        T_eV, '1d_sedov_taylor.h5', species_keys)

    # Run the code
    run_castro_simulation("castro.add_ext_src=0 castro.diffuse_temp=0 problem.initial_conditions_file=1d_sedov_taylor.h5")

    # Check the results
    # TODO: Compare the results with Sedov-Taylor theory
    # Evaluate checksum
    evaluate_checksum("1d_sedov_taylor", "plt_1d_*")

if __name__ == "__main__":
    test_1d_sedov_taylor()