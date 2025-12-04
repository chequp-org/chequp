import subprocess
import re
import numpy as np
import sys
import glob
import os
import time
sys.path.append("../initial_condition")
from ionization_routines import save_to_openpmd
sys.path.append('../sim_folder/analysis/')
from analysis_tool import CastroSimulation
sys.path.append('../theory/sedov_theory/python/')
from sedov_theory import SedovTalorProblem
from checksum.checksumAPI import evaluate_checksum
from scipy.constants import m_p, e
from scipy.optimize import curve_fit

def cleanup_outputs(extra_file=""):
    # Remove previously generated plotfiles and checkpoints

    os.system("rm -rf plt_1d_* chk* amr_diag.out species_diag.out grid_diag.out Backtrace.0" + extra_file)

def check_energy_conservation(sim_data, tol:float=1.0):
    """
    This function check that the total energy is conserved within a given tol
    Raise an assertion error if the test fails
    """
    t = sim_data.output_times
    E_tot = sim_data.get_energy(t, level=2)[0]
    rel_err = (E_tot - E_tot[0]) / E_tot[0] * 100.0
    test = np.all(rel_err < tol)
    value = np.max(rel_err)
    assert test, f"Energy conservation test failed: Avg. Deviation = {value:.1e} % > {tol}% tol."

def check_blast_radius_t_ST(sim_data, sol, tol:int=10):
    """
    This function check that the blast radius time evolution fit with the theory given by Sedov Taylor
    Raise an assertion error if the test fails
    """
    t = sim_data.output_times
    density_profiles = [sim_data.get_field(t_, 'density', level=2) for t_ in t]
    r_blast = [profile['r'][np.argmax(profile['q'])] for profile in density_profiles]
    popt, _ = curve_fit(lambda time, a: a * np.sqrt(time), t[1:], r_blast[1:])
    r_fit = popt[0] * np.sqrt(t)
    r_analytical = np.array(sol.blast_radius(t))
    rel_error = np.linalg.norm(r_fit*1e4 - r_analytical*1e4) / np.linalg.norm(r_analytical*1e4) * 100.
    assert rel_error < tol, f"Shock radius comparison to Sedov Taylor theory failed: rel. err. = {rel_error:.1f} % > {tol} % tol."

def check_density_profile_ST(sim_data, sol, tol:int=15):
    """
    This function check for different time (7, 8, 9 ns) that the density profile match with the one given by Sedov Taylor theory
    The function only compare radius below the blast radius (where the density peak is)
    """
    t = np.array([sim_data.output_times[int(len(sim_data.output_times)*f)] for f in [0.7, 0.8, 0.9]])
    r = np.array([sim_data.get_field(t_, 'density', level=2)['r'] for t_ in t])
    # This get the density profile rho(r) from the sim for the differents time
    rho_sim = np.array([sim_data.get_field(t_, 'density', level=2)['q'] for t_ in t]) 
    # This compute the theoretical density profile from Sedov Taylor for the time and the radius array
    rho_analytical = np.array([sol.evaluate('density', r_, t_) for r_, t_ in zip(r, t)])
    # compare up to the first peak present in both profiles
    for rho_a, rho_s, r_ in zip(rho_analytical, rho_sim, r):
        peak_idx = min(np.argmax(rho_a), np.argmax(rho_s))
        if r_[peak_idx] > 1e-2: # dont compared for low blast radius
            denom = np.linalg.norm(rho_s[:peak_idx])
            err = np.linalg.norm(rho_a[:peak_idx] - rho_s[:peak_idx]) / denom * 100.
            assert err < tol, f"Density profile comparison to Sedov Taylor theory failed: rel. err. = {err:.1f} % > {tol} % tol."

def run_castro_simulation(model='gamma_law', runtime_options=""):
    """
    Run the Castro simulation.
    Raise an error and print stdout/stderr if the command fails.
    """
    # Find the Castro executable
    build_dir = "../sim_folder/build"
    executables = glob.glob( os.path.join(build_dir, f"Castro1d*.{model}.ex") )
    if len(executables) == 0:
        raise FileNotFoundError(f"No Castro1d executable found in {build_dir}")
    elif len(executables) > 1:
        raise RuntimeError(f"Multiple Castro1d executables found: {executables}")
    executable = executables[0]

    cleanup_outputs()

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
    Test that code produce the exact Sedov-Taylor blast wave solution, in a simplified setup:
    - no ionization reactions (castro.add_ext_src=0)
    - no temperature diffusion (castro.diffuse_temp=0)
    """
    # Generate openPMD inital conditions for a small-radius plasma
    # Gaussian temperature profile with sigma=4 microns, peak T=1000 eV
    Twidth = 4e-6
    r = np.linspace(0, 5*Twidth, 1024)
    T0_eV = 1000
    # Gaussian profile to fasten convergence
    T_eV = np.ones_like(r) * T0_eV * np.exp(-r**2/Twidth**2) 
    # put last value to zero as this is used outside of 5*sigma
    T_eV[-1] = 0 
    # Parse the species names for which Castro has been compiled
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())
    populations = np.zeros((len(r), len(species_keys)))
    # Set fraction to 1 for H+
    # small neutral fraction to avoid issues with zero density
    populations[:, species_keys.index('H1')] = 1 - 1e-3
    populations[:, species_keys.index('H0')] = 1e-3
    # Save file
    save_to_openpmd( {'r': [r.min(), r.max()]}, populations,
        T_eV, '1d_sedov_taylor.h5', species_keys)

    # Run the code
    # The runtime options are the parameters that are temporary overwritten in the input file to lauch the simulation.
    # This avoid to modify each time we want to run with differents parameters
    run_castro_simulation(model='gamma_law', runtime_options="amr.n_cell=128 castro.add_ext_src=0 castro.diffuse_temp=0 problem.initial_conditions_file=1d_sedov_taylor.h5")
    # Physical tests #
    sim_data = CastroSimulation('.', 'plt_1d_') # load simulation data

    # Comparison with Sedov Taylor theory
    rho_initial = 1.67e-6  # in g.cm^-3
    mp_g = m_p*1e3 # in g
    sigma_cm = Twidth*1e2 # in cm
    deposited_energy = 3*np.pi/2 * T0_eV*e * sigma_cm**2 * rho_initial/mp_g * 1e7 # in erg/cm (computed by integrating initial conditions)
    analytical_data = SedovTalorProblem(5.0 / 3.0, deposited_energy, rho_initial)

    check_energy_conservation(sim_data, tol=1.0)
    check_blast_radius_t_ST(sim_data, analytical_data, tol=10)
    check_density_profile_ST(sim_data, analytical_data, tol=12)

    # Evaluate checksum
    evaluate_checksum("1d_sedov_taylor", "plt_1d_*", rtol=4.e-7)

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_sedov_taylor.h5')


def test_1d_desy_benchmark():
    """
    Test the code in the scenario that benchmarked with DESY team
    (close - but not identical - to the one from Mewes et al., PRR 5, 033112, 2023)
    """
    # Generate openPMD inital conditions according to the agreed-upon benchmark
    sigma1 = 38e-6  # in m
    sigma2 = 35e-6  # in m
    Te_max = 27 # in eV
    Ta = 0.03 # in eV
    # Create r array from 0 to 6e-4 with 1e-6 increment
    r = np.arange(0, 6e-4 + 1e-6, 1e-6)
    # Calculate ionization fraction, with minimal ionization fraction of 1e-3
    # (the minimal fraction is needed for the electron temperature to be defined everywhere)
    ioniz_fraction = (1. - 1.e-3)*np.exp(-np.power(r*r/(2*sigma1*sigma1), 12)) + 1.e-3
    # Calculate electron temperature, with a minimal temperature of 0.03 eV
    T_eV = (Te_max - Ta) * np.exp(-np.power(r*r/(2*sigma2*sigma2), 3)) + Ta
    # Parse the species names for which Castro has been compiled
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())
    populations = np.zeros((len(r), len(species_keys)))
    # Set H0 and H1 fractions
    populations[:, species_keys.index('H0')] = 1 - ioniz_fraction
    populations[:, species_keys.index('H1')] = ioniz_fraction
    # Save file
    save_to_openpmd( {'r': [r.min(), r.max()]}, populations,
        T_eV, '1d_desy_benchmark.h5', species_keys)

    # Run the code
    run_castro_simulation("problem.initial_conditions_file=1d_desy_benchmark.h5")
    # Evaluate checksum
    evaluate_checksum("1d_desy_benchmark", "plt_1d_*", rtol=4.e-7)

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_desy_benchmark.h5')

if __name__ == "__main__":
    test_1d_desy_benchmark()
    test_1d_sedov_taylor()
