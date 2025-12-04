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
from scipy.optimize import curve_fit
from scipy.constants import e, m_p

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



# Those data came from a COMSOL simulation done by Mathis Mewes using the model HYQUP (Mewes et al., PRR 5, 033112, 2023)
# The simulation only consider Atomic species H0, H1. The initial parameters are defined in test_1d_desy_benchmark.
# Units: m for length, m-3 for density, K for temperature
comsol_data = {
    "blast":{
        "t": np.array([0, 9.987819732034104e-10, 1.9975639464068208e-9, 5.006090133982948e-9, 8.002436053593179e-9, 9.987819732034105e-9]),
        "r": np.array([0.000056708333333333334, 0.00008966666666666666, 0.00011241666666666667, 0.00016491666666666667, 0.00020837499999999998, 0.00023375])
    },
        "2ns": {
        "r": np.array([0.0, 3.33e-05, 6.665999999999999e-05, 9.995999999999998e-05, 0.00013331999999999998, 0.00016661999999999997, 0.00019998, 0.00023328, 0.00026663999999999995, 0.0003]),
        "Te": np.array([23211.770254021863, 26268.06379376689, 25591.017258312797, 20765.247549951004, 18346.123342827552, 13993.043254526054, 9584.55472861077, 5462.684093868529, 2669.5177678165664, 2186.16848211767]),
        "na": np.array([4.573456505084193e23, 4.488322021597051e23, 4.716988950992047e23, 1.2950720870687254e24, 1.000191035605734e24, 1.0000108494331972e24, 1.0000074151960918e24, 1.0000038781767558e24, 1.0000005022939332e24, 1.0000000117352788e24])
    },
        "5ns": {
        "r": np.array([0.0, 3.33e-05, 6.665999999999999e-05, 9.995999999999998e-05, 0.00013331999999999998, 0.00016661999999999997, 0.00019998, 0.00023328, 0.00026663999999999995, 0.0003]),
        "Te": np.array([15298.083353951515, 15839.669679535817, 16230.07213040422, 16191.996802917163, 13554.72861167953, 12179.76963417542, 8297.040258967807, 5445.559752830032, 3581.2870669509766, 2873.4876147582386]),
        "na": np.array([1.845096259779847e23, 1.8600390663265706e23, 2.182668312667086e23, 2.9632927479752823e23, 8.15295462591155e23, 2.4862851947449127e24, 1.0000272133128904e24, 1.0000241194832156e24, 1.0000129103143216e24, 1.0000059737963808e24])
    },
        "8ns": {
        "r": np.array([0.0, 3.33e-05, 6.665999999999999e-05, 9.995999999999998e-05, 0.00013331999999999998, 0.00016661999999999997, 0.00019998, 0.00023328, 0.00026663999999999995, 0.0003]),
        "Te": np.array([15261.491790992808, 14799.523774366897, 14131.878282643287, 13239.936964127124, 11439.173495150188, 8950.546444133326, 7785.458674669467, 5495.694543677841, 3659.4758700778334, 3033.5011783160394]),
        "na": np.array([1.7173346635405725e23, 1.6882604000540383e23, 1.7749837289864806e23, 1.980970437470241e23, 2.929998504336976e23, 9.050084528474029e23, 2.80213515839221e24, 1.0000595793491873e24, 1.00004108600266e24, 1.0000290997686949e24])
    }
}

def check_blast_radius_t_Comsol(sim_data, tol:int=10):
    """
    Compare the time evolution of the blast radius with the one obtain from the COMSOL simulation.
    Raise an error if the L2 error is over the tol%.
    """
    # Comsol data
    t_comsol, r_comsol = comsol_data['blast']['t'], comsol_data['blast']['r']

    # Simulation data
    t_sim = sim_data.output_times
    r_sim = np.array([sim_data.get_field(t_, 'density', level=2)['r'][np.argmax(sim_data.get_field(t_, 'density', level=2)['q'])] for t_ in t_sim])
    r_comsol_interp = np.interp(t_sim[1:], t_comsol, r_comsol)
    rel_error = np.linalg.norm(r_sim[1:]*1e4 - r_comsol_interp*1e6) / np.linalg.norm(r_comsol_interp*1e6) * 100.
    assert rel_error < tol, f"Shock radius comparison to COMSOL failed: rel. err. = {rel_error:.1f} % > {tol} % tol."

def check_density_profile_r_Comsol(sim_data, tol:int=50):
    """
    Compare radial density profiles at several output times to the COMSOL solution.
    Raise an error if the L2 error is over the tol%.
    """
    for t, t_comsol in zip([2e-9, 5e-9, 8e-9], ['2ns', '5ns', '8ns']):
        r = np.array(sim_data.get_field(float(t), 'density', level=2)['r'], dtype=np.float64)
        # consider only r > 20 microns to avoid low-density noisy region and below the blast radius
        rho = np.array(sim_data.get_field(float(t), 'density', level=2)['q'], dtype=np.float64)
        mask = (r*1e4 > 20) & (r*1e4 > r[np.argmax(rho)]*1e4)
        na_comsol_interp = np.interp(r*1e4, comsol_data[t_comsol]['r']*1e6, comsol_data[t_comsol]['na']/1e24)
        error = np.linalg.norm(rho[mask]/1.67e-6 - na_comsol_interp[mask]) / np.linalg.norm(na_comsol_interp[mask]) * 100.
        assert error < tol, f"Shock radius comparison to COMSOL failed: rel. err. = {error:.1f} % > {tol} % tol."

def test_1d_desy_benchmark():
    """
    Test the code in the scenario that benchmarked with DESY team
    (close - but not identical - to the one from Mewes et al., PRR 5, 033112, 2023)
    """
    # Generate openPMD inital conditions according to the agreed-upon benchmark
    sigma1 = 38e-6  # in m
    sigma2 = 35e-6  # in m
    Te_max = 27 # in eV
    kb = 8.617333262145e-5  # eV/K
    # Background temperature in eV (constrain from COMSOL simulation)
    Ta = 2000 * kb  
    # Create r array from 0 to 6e-4 with 1e-6 increment
    r = np.arange(0, 6e-4 + 1e-6, 1e-6)
    # Calculate ionization fraction, with minimal ionization fraction of 1e-3
    # (the minimal fraction is needed for the electron temperature to be defined everywhere)
    ioniz_fraction = (1. - 1.e-3)*np.exp(-np.power(r*r/(2*sigma1*sigma1), 12)) + 1.e-3
    # Calculate electron temperature profile such that it match with the COMSOL one
    T_eV = Te_max * np.exp(-np.power(r*r/(2*sigma2*sigma2), 3)) + Ta
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
    # The runtime options are the parameters that are temporary overwritten in the input file to lauch the simulation.
    # This avoid to modify each time we want to run with differents parameters
    run_castro_simulation(model='gamma_law_2T', runtime_options="castro.add_ext_src=1 castro.diffuse_temp=1 amr.plot_int = 100 problem.initial_conditions_file=1d_desy_benchmark.h5")
    # Physical tests
    sim_data = CastroSimulation('.', 'plt_1d_*')

    check_energy_conservation(sim_data,tol=1.0)
    check_blast_radius_t_Comsol(sim_data, tol=12)
    check_density_profile_r_Comsol(sim_data, tol=25)

    # Evaluate checksum
    evaluate_checksum("1d_desy_benchmark", "plt_1d_*", rtol=4.e-7)

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_desy_benchmark.h5')

if __name__ == "__main__":
    test_1d_sedov_taylor()
    test_1d_desy_benchmark()
