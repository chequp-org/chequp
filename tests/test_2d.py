"""
This script tests that the 2D code produce the correct Sedov-Taylor blast wave solution.

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
from scipy.interpolate import RegularGridInterpolator
sys.path.append("../initial_condition")
from ionization_routines import save_to_openpmd
sys.path.append('../sim_folder/analysis/')
from analysis_tool import CastroSimulation
sys.path.append('../theory/sedov_theory/python/')
from sedov_theory import SedovTalorProblem
from checksum.checksumAPI import evaluate_checksum
from scipy.constants import m_p, k
from scipy.optimize import curve_fit

def cleanup_outputs(extra_file = ""):
    # Remove previously generated plotfiles and checkpoints
    os.system(f"rm -rf plt_2d_* chk* amr_diag.out species_diag.out grid_diag.out Backtrace.0 " + extra_file)

def check_r_iso_t(sim_data, sol, tol_r: int = 10, tol_iso: float = 0.5):

    def find_edge_radial_xy(data, n_angles=100, n_samples=1000):
        x, y = np.array(data['x'], dtype=float), np.array(data['y'], dtype=float)
        cx, cy = x[-1]/2, y[-1]/2

        # create interpolator on physical grid
        interp = RegularGridInterpolator((y, x), data['q'], bounds_error=False, fill_value=np.nan)

        # radial sampling
        thetas = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        radii = np.zeros(n_angles)
        x_edge = np.zeros(n_angles)
        y_edge = np.zeros(n_angles)

        # maximum possible radius (diagonal)
        r_max = np.hypot(x[-1]-x[0], y[-1]-y[0])

        for i, th in enumerate(thetas):
            rs = np.linspace(0, r_max, n_samples)
            xs_ray = cx + rs * np.cos(th)
            ys_ray = cy + rs * np.sin(th)
            pts = np.column_stack([ys_ray, xs_ray])  # interpolator expects (y,x)
            vals = interp(pts)
            valid = np.isfinite(vals)
            if valid.sum() < 5:
                radii[i] = np.nan
                x_edge[i] = np.nan
                y_edge[i] = np.nan
                continue
            rs = rs[valid]; vals = vals[valid]

            dv = np.gradient(vals, rs)
            idx = np.nanargmax(np.abs(dv))

            radii[i] = rs[idx]
            x_edge[i] = cx + radii[i] * np.cos(th)
            y_edge[i] = cy + radii[i] * np.sin(th)

        return x_edge, y_edge

    def fit_circle_radius(x, y, data):
        R0 = np.array(data['x'][-1], dtype=float)/2
        r = np.sqrt((x-R0)**2 + (y-R0)**2)
        R = np.mean(r)
        iso = np.std(r)/np.mean(r)
        return R, iso
    
    L_r, L_r_analytical, L_iso = [], [], []
    t_sim = sim_data.output_times
    for t_ in t_sim:
        data_fit = sim_data.get_field(t_, quantity ='density', level=2)
        x_max, y_max = find_edge_radial_xy(data_fit)
        R_fit, iso = fit_circle_radius(x_max, y_max, data_fit)
        r_analytical = sol.blast_radius(t_)
        L_r.append(R_fit)
        L_r_analytical.append(r_analytical)
        L_iso.append(iso)
    mask = np.array(t_sim) >= 1e-9 # avoid early times with poor resolution
    error_r = np.linalg.norm(np.array(L_r)[mask] - np.array(L_r_analytical)[mask]) / np.linalg.norm(np.array(L_r_analytical)[mask]) * 100.
    error_iso = np.mean(np.array(L_iso)[mask])
    test_r = error_r < tol_r
    test_iso = error_iso < tol_iso

    assert test_r, f"Shock radius test failed: rel. error = {error_r:.2f} % > {tol_r} %"
    assert test_iso, f"Shock isotropy test failed: mean isotropy = {error_iso:.2f} % > {tol_iso} %"

def check_energy_conservation(sim_data, tol: float = 1.0):
    t = sim_data.output_times
    E_tot = sim_data.get_energy(t, level=2, energy_type='total')[0]
    rel_err = np.abs(E_tot - E_tot[0]) / E_tot[0] * 100.0
    test = np.all(rel_err < tol)
    value = np.max(rel_err)
    assert test, f"Energy conservation test failed: Avg. Deviation = {value:.1e} % > {tol}% tol."

def check_rho_r(sim_data, sol, tol: int = 21):
    """
    Compare radial density profiles at several output times to the analytical solution.
    Returns True if the mean relative L2 error is below 15%.
    """
    errors = []
    t_sim = sim_data.output_times
    for t_ in t_sim[3::2]:
        m = sim_data.get_field(t_, quantity='density', level=2)
        rho_sim = m['q'][:, m['q'].shape[0]//2]
        x_center = np.linspace(-np.array(m['x'], dtype = float)[len(m['x'])//2], np.array(m['x'], dtype = float)[len(m['x'])//2], len(m['x']), dtype=float)
        rho_analytical = sol.evaluate('density', np.abs(x_center), t_)
        # compare up to the first peak present in both profiles
        peak_idx = min(np.argmax(rho_analytical), np.argmax(rho_sim))
        denom = np.linalg.norm(rho_sim[:peak_idx])
        err = np.linalg.norm(rho_analytical[:peak_idx] - rho_sim[:peak_idx])/denom
        errors.append(err)
        
    mean_error = float(np.mean(errors)) * 100
    test = mean_error < tol
    assert test, f"Density profile test failed: mean rel. L2 error = {mean_error:.2f} % > {tol} %"

def run_castro_simulation(model = 'gamma_law', runtime_options = ""):
    """
    Run the Castro simulation.
    Raise an error and print stdout/stderr if the command fails.
    """
    # Find the Castro executable
    build_dir = "../sim_folder/build"
    executables = glob.glob( os.path.join(build_dir, f"Castro2d*.{model}.ex") )
    if len(executables) == 0:
        raise FileNotFoundError(f"No Castro2d executable found in {build_dir}")
    elif len(executables) > 1:
        raise RuntimeError(f"Multiple Castro2d executables found: {executables}")
    executable = executables[0]

    cleanup_outputs()

    # Run the code
    inputs = "../sim_folder/run/inputs.2d.cyl_in_cartcoords"
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

def test_2d_sedov_taylor():
    """
    Test that code produce the exact Sedov-Taylor blast wave solution, in a simplified setup:
    - no ionization reactions (castro.add_ext_src=0)
    - no temperature diffusion (castro.diffuse_temp=0)
    """
    # Grid
    x = np.linspace(0.0, 600e-6, 256)
    y = np.linspace(0.0, 600e-6, 256)
    X, Y = np.meshgrid(x, y, indexing='ij')  # 2D grid 512x512
    sigma = 3e-6
    T_peak = 1000.0  # eV
    T_min = 1e-3     # eV, small temperature floor
    center = 300e-6
    T_eV = T_min + (T_peak - T_min) * np.exp(- ((X-center)**2 + (Y-center)**2) / (2 * sigma**2)) # Gaussian profile of temperature

    # Species keys
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())

    # Populations array
    f_ion = 1.0  # ionization fraction
    populations = np.zeros((X.shape[0], X.shape[1], len(species_keys)))
    populations[:, :, species_keys.index('H1')] = f_ion - 1e-3
    populations[:, :, species_keys.index('H0')] = f_ion
    # Save file
    save_to_openpmd({'x': [x.min(), x.max()], 'y': [y.min(), y.max()]},
                populations, T_eV, '2d_sedov_taylor.h5', species_keys)
    # Run the code
    run_castro_simulation(model = 'gamma_law', runtime_options = "amr.n_cell = 64 64 castro.add_ext_src = 0 castro.diffuse_temp = 0 amr.max_level  = 3 problem.initial_conditions_file=2d_sedov_taylor.h5")
    # Physical tests 
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dA_cm2 = dx * dy * 1e1
    eV_to_erg = 1.60218e-12
    rho_initial = 1.67e-6  # in g/cm^3
    deposited_energy = 3/2 * rho_initial / m_p * f_ion * eV_to_erg * np.sum(T_eV) * dA_cm2
    rho_initial = 1.67e-6  # in g/cm^3
    sim_data = CastroSimulation('.', 'plt_2d_')
    analytical_data = SedovTalorProblem(5.0 / 3.0, deposited_energy, rho_initial) # E0 in mJ/m, rho_0 in g/cm^3 (computed by integrating the initial profile of temperature ponderated by the populations)

    check_r_iso_t(sim_data, analytical_data, tol_r=10, tol_iso=0.5)
    check_energy_conservation(sim_data, tol=1.0)
    check_rho_r(sim_data, analytical_data, tol=21)

    # Evaluate checksum
    evaluate_checksum("2d_sedov_taylor", "plt_2d_*")

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('2d_sedov_taylor.h5')

def test_2d_desy_benchmark():
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
    evaluate_checksum("1d_desy_benchmark", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    cleanup_outputs('1d_desy_benchmark.h5')

if __name__ == "__main__":
    test_2d_sedov_taylor()
    test_2d_desy_benchmark()
