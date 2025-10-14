"""
This script tests that the 1D code produce the correct Sedov-Taylor blast wave solution.

It assumes that the code has already been compiled and that the executable
is in ../sim_folder/build/Castro1d.gnu.MPI.ex
"""
import os
import subprocess

def run_castro_simulation():
    """
    Run the Castro simulation.

    Raise an error and print stdout/stderr if the command fails.
    """
    executable = "../sim_folder/build/Castro1d.gnu.MPI.ex"
    inputs = "../sim_folder/run/inputs.1d.cyl"
    command = f"{executable} {inputs} castro.add_ext_src=0 castro.diffuse_temp=0"
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

    # Generate openPMD inital conditions

    # Run the code
    run_castro_simulation()

    # Compare the results with the correct solution
    pass

if __name__ == "__main__":
    test_1d_sedov_taylor()