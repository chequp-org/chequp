"""
This script tests that the 1D code produce the correct Sedov-Taylor blast wave solution.

It assumes that the code has already been compiled and that the executable
is in ../sim_folder/build/Castro1d.gnu.MPI.ex
"""
import subprocess
import re
import numpy as np
import sys
sys.path.append("../initial_condition")
from ionization_routines import save_to_openpmd

def run_castro_simulation(runtime_options):
    """
    Run the Castro simulation.

    Raise an error and print stdout/stderr if the command fails.
    """
    executable = "../sim_folder/build/Castro1d.gnu.MPI.ex"
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
    # High-temperature plasma in the first 5 microns, low-temperature plasma in the rest
    r = np.linspace(0, 100e-6, 1024)
    T_eV = np.ones_like(r) * 2000
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

    # TODO: Compare the results with the correct solution
    pass

if __name__ == "__main__":
    test_1d_sedov_taylor()