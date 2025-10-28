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

def clean_cluster_sim():
    # Remove previously generated plotfiles and checkpoints
    os.system(f"rm -rf plt_2d_*.temp.old.* plt_2d_*.temp.* plt_2d_*.old.* amr_diag.out species_diag.out grid_diag.out Backtrace.0 slurm-*.out slurm-*.err " )

class physical_test_2d:
    
    def __init__(self, rho_0 = 1.67e-6, E0 = 1.2e16, R0 = 50):
        self.rho_0 = rho_0  # g/cm^3
        self.R0 = R0  # microns
        self.E0 = E0  # mJ/m this value is computed by integrating the initial profile of temperature ponderated by the populations
        self.sol = SedovTalorProblem(5.0 / 3.0, E0, self.rho_0)
        self.time, self.data_rho, self.data_T = self.load_2D(end_time=10.e-9)
        
    def load_2D(self, end_time=2.0e-9):
        # List all directories that match your pattern
        folders = sorted(glob.glob("./plt_2d_*"))
        folders = [f for f in folders if os.path.exists(os.path.join(f, "Header"))]
        # Field to plot
        field_rho, field_T = ("boxlib", "density"), ("boxlib", "Temp")
        time, data_rho, data_T = [], [], []
        for folder in folders:
            try:
                ds = yt.load(folder)
            except Exception as e:
                continue
            curr_time = ds.current_time
            if (curr_time > end_time):
                break
            time.append(float(curr_time))
            # Extract the full 2D grid
            cg = ds.covering_grid(
                level=0,
                left_edge=ds.domain_left_edge,
                dims=ds.domain_dimensions
            )

            # Convert field to 2D array
            array_2d = cg[field_rho].to_ndarray()
            array_2d = np.squeeze(array_2d)  # remove singleton dimensions
            data_rho.append(array_2d)

            # Extract temperature field
            array_2d_T = cg[field_T].to_ndarray()
            array_2d_T = np.squeeze(array_2d_T)  # remove singleton dimensions
            data_T.append(array_2d_T)

        return time, data_rho, data_T
    
    def find_edge_radial_xy(self, data, n_angles=100, n_samples=1000):
        x, y = np.linspace(0, 100, data.shape[1]), np.linspace(0, 100, data.shape[0])
        cx, cy = 50, 50

        # create interpolator on physical grid
        interp = RegularGridInterpolator((y, x), data, bounds_error=False, fill_value=np.nan)

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
    
    def fit_circle_radius(self, x, y):
        r = np.sqrt((x-self.R0)**2 + (y-self.R0)**2)
        R = np.mean(r)
        iso = np.std(r)/np.mean(r)
        return R, iso
    
    def test_r_iso_t(self, tol_r: int = 10, tol_iso: float = 0.5):
        """
        Compare the fitted shock radius to the analytical Sedov-Taylor radius.

        Returns True if the relative L2 error is below `tol`, False otherwise.
        """
        L_time, L_r, L_r_analytical, L_iso = [], [], [], []
        for idx in range(1, len(self.time)):
            data_fit = self.data_rho[idx]
            x_max, y_max = self.find_edge_radial_xy(data_fit)
            R_fit, iso = self.fit_circle_radius(x_max, y_max)
            r_analytical = self.sol.blast_radius(self.time[idx])
            L_time.append(self.time[idx])
            L_r.append(R_fit)
            L_r_analytical.append(r_analytical)
            L_iso.append(iso)
        mask = np.array(L_time) >= 1e-9 # avoid early times with poor resolution
        rel_error = np.linalg.norm(np.array(L_r)[mask] - np.array(L_r_analytical)[mask]) / np.linalg.norm(np.array(L_r_analytical)[mask]) * 100.
        error_iso = np.mean(np.array(L_iso)[mask])
        if rel_error < tol_r:
            test_r = True
        else:
            test_r = False
        if error_iso < tol_iso:
            test_iso = True
        else:
            test_iso = False
        return test_r, rel_error, test_iso, error_iso
    
    def test_rho_r(self, tol: int = 5):
        """
        Compare radial density profiles at several output times to the analytical solution.
        Returns True if the mean relative L2 error is below 15%.
        """
        time_idx = np.arange(0.6, 0.99, 0.1) * len(self.time)
        r_binned_all, data_binned_all, r_analytical_all, data_analytical_all = [], [], [], []
        for idx in time_idx:
            idx = int(idx)
            x = np.linspace(-50, 50, self.data_rho[idx].shape[0])
            rho_fit_center = self.data_rho[idx][:, self.data_rho[idx].shape[0]//2]
            rho_analytical = self.sol.evaluate('density', np.abs(x), self.time[idx])
            r_binned_all.append(x)
            data_binned_all.append(rho_fit_center)
            r_analytical_all.append(x)
            data_analytical_all.append(rho_analytical)
        errors = []
        for k in range(len(time_idx)):
            rho_sim = data_binned_all[k]
            rho_analytical = data_analytical_all[k]
            # compare up to the first peak present in both profiles
            peak_idx = min(np.argmax(rho_analytical), np.argmax(rho_sim))
            denom = np.linalg.norm(rho_sim[:peak_idx])
            err = np.linalg.norm(rho_analytical[:peak_idx] - rho_sim[:peak_idx])/denom
            errors.append(err)
            
        mean_error = float(np.mean(errors))
        if mean_error < tol/100:
            test = True
        else:
            test = False
        return test , mean_error
    
    def test_energy(self, tol: float = 1.0):
        """
        Extract the total energy (thermal + kinetic + potential) as a function of time in 2D Cartesian geometry.
        The result is given per unit length along the z direction (J/m).
        """
        E_kin_thermal = []
        E_pot = []
        E_tot = []
        time = []
        yt_timeseries = yt.load('plt_2d_*')

        for ds in yt_timeseries:
            ad0 = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)

            # Coordinates (in meters)
            x = np.linspace(ds.domain_left_edge[0], ds.domain_right_edge[0], ds.domain_dimensions[0])
            y = np.linspace(ds.domain_left_edge[1], ds.domain_right_edge[1], ds.domain_dimensions[1])
            dx = (x[1] - x[0])
            dy = (y[1] - y[0])

            # Energy density [erg/cm^3] -> [J/m^3]
            E = ad0['rho_E'].to_ndarray() * 1e-7 * 1e6
            e_kin_thermal = np.sum(E) * dx * dy  # J/m

            # Hydrogen density [g/cm^3] -> [kg/m^3]
            rho_Hp = ad0['rho_H1'].to_ndarray() * 1e-3 * 1e6
            n_e = rho_Hp / m_p
            Ntot = np.sum(n_e) * dx * dy  # particles/m
            e = 1.60218e-19  # C
            # Potential energy (13.6 eV per electron)
            e_pot = 13.6 * e * Ntot  # J/m

            E_kin_thermal.append(e_kin_thermal)
            E_pot.append(e_pot)
            E_tot.append(e_kin_thermal + e_pot)
            time.append(float(ds.current_time))

        # Convert to arrays
        time = np.array(time)
        E_kin_thermal = np.array(E_kin_thermal) * 1e3  # mJ/m
        E_pot = np.array(E_pot) * 1e3  # mJ/m
        E_tot = np.array(E_tot) * 1e3  # mJ/m

        # Check energy conservation
        rel_error = np.max(np.abs(E_tot - E_tot[0]) / np.abs(E_tot[0]) * 100.)
        if rel_error < tol:
            test = True
        else:
            test = False
        value = rel_error
        return test, value
    

def run_castro_simulation(runtime_options, work_dir="sim_2D_128_10"):
    """
    Run the Castro simulation in another folder specified by `work_dir`.

    Raise an error and print stdout/stderr if the command fails.
    """
    # Find the Castro executable
    build_dir = "../sim_folder/build"
    executables = glob.glob(os.path.join(build_dir, "Castro2d*"))
    if len(executables) == 0:
        raise FileNotFoundError(f"No Castro2d executable found in {build_dir}")
    elif len(executables) > 1:
        raise RuntimeError(f"Multiple Castro2d executables found: {executables}")
    executable = executables[0]

    cleanup_outputs()

    # Run the code in a different directory
    inputs = os.path.join("../sim_folder/run/inputs.2d.cyl_in_cartcoords")
    command = f"{executable} {inputs} {runtime_options}"

    try:
        subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            cwd=work_dir,  # <<< run the command from this directory
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
    - the initial radius of the hot plasma is small (5 microns)
    """
    print("Generating initial conditions...")
    # Grid
    x = np.linspace(0.0, 600e-6, 256)
    y = np.linspace(0.0, 600e-6, 256)
    X, Y = np.meshgrid(x, y, indexing='ij')  # 2D grid 512x512
    sigma = 3e-6
    T_peak = 1000.0  # eV
    T_min = 1e-3     # eV, small temperature floor
    center = 300e-6
    T_eV = T_min + (T_peak - T_min) * np.exp(- ((X-center)**2 + (Y-center)**2) / (2 * sigma**2))

    # Species keys
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())

    # Populations array
    populations = np.zeros((X.shape[0], X.shape[1], len(species_keys)))
    populations[:, :, species_keys.index('H1')] = 1.0
    # Save file
    save_to_openpmd({'x': [x.min(), x.max()], 'y': [y.min(), y.max()]},
                populations, T_eV, '2d_sedov_taylor.h5', species_keys)
    print("Starting simulation...")
    # Run the code
    time_s = time.time()
    run_castro_simulation("problem.initial_conditions_file=2d_sedov_taylor.h5")
    time_e = time.time()
    print(f"Simulation completed in {time_e - time_s:.2f} seconds.")

    # Physical tests #
    print("Running physical tests...\n")
    phys_test = physical_test_2d(rho_0=1.67e-6, E0=1.19e16, R0=50)
    test_rho, val_rho = phys_test.test_rho_r(tol = 5)
    if test_rho :
        print(f"\t Test density profile : PASSED (rel. err. = {val_rho:.1f} % < 5 % tol.)")
    else :
        print(f"\t Test density profile : FAILED (rel. err. = {val_rho:.1f} % > 5 % tol.)")
    test_r_t, val_r_t, test_iso, value_iso = phys_test.test_r_iso_t(tol_r = 10., tol_iso = 0.5)
    if test_r_t :
        print(f"\t Test shock radius vs time : PASSED (rel. err. = {val_r_t:.1f} % < 10 % tol.)")
    else :
        print(f"\t Test shock radius vs time : FAILED (rel. err. = {val_r_t:.1f} % > 10 % tol.)")
    if test_iso :
        print(f"\t Test shock isotropy : PASSED (mean isotropy = {value_iso:.2f} % < 0.5 % tol.)")
    else :
        print(f"\t Test shock isotropy : FAILED (mean isotropy = {value_iso:.2f} % > 0.5 % tol.)")
    test_energy, val_energy = phys_test.test_energy(tol = 1)
    if test_energy is True :
        print(f"\t Test energy conservation : PASSED (Max. Deviation = {val_energy:.1e} % < 1 % tol.)")
    else :
        print(f"\t Test energy conservation : FAILED (Max. Deviation = {val_energy:.1e} % > 1 % tol.)")

    # Evaluate checksum
    #evaluate_checksum("1d_sedov_taylor", "plt_1d_*")

    # Remove generated plotfiles and checkpoints
    #cleanup_outputs('1d_sedov_taylor.h5')

def test_2d_desy_benchmark():
    """
    Test the code in the scenario that benchmarked with DESY team
    (close - but not identical - to the one from Mewes et al., PRR 5, 033112, 2023)
    """
    print("Generating initial conditions...")
    # Generate openPMD initial conditions according to the agreed-upon benchmark
    data = np.loadtxt("2D_xy_Init_Slice 3_2022_06.txt")
    x, y, Z_H1, T_eV = data.T

    interp_H1 = RegularGridInterpolator((np.unique(data[:,1]), np.unique(data[:,0])),data[:,2].reshape(300, 300),bounds_error=False,fill_value=None)
    interp_T = RegularGridInterpolator((np.unique(data[:,1]), np.unique(data[:,0])),data[:,3].reshape(300, 300),bounds_error=False,fill_value=None)
    # Grid
    x = np.linspace(0.0, 600e-6, 256)
    y = np.linspace(0.0, 600e-6, 256)
    center = 300e-6
    X_grid, Y_grid = np.meshgrid(x, y, indexing='ij')
    T_eV_interp = interp_T((Y_grid - center, X_grid - center)) 
    T_eV_interp = T_eV_interp / (1.1e4) # conversion from K to eV
    Z_H1_interp = interp_H1((Y_grid - center, X_grid - center))
    # Species keys
    with open('../sim_folder/build/species.net', 'r') as f:
        species_keys = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', f.read())
    populations = np.zeros((X_grid.shape[0], X_grid.shape[1], len(species_keys)))
    populations[:, :, species_keys.index('H1')] = Z_H1_interp
    populations[:, :, species_keys.index('H0')] = 1.0 - Z_H1_interp
    # Save file
    save_to_openpmd({'x': [x.min(), x.max()], 'y': [y.min(), y.max()]},
                populations, T_eV_interp, '2d_desy_benchmark.h5', species_keys)
    # Run the code
    time_s = time.time()
    print("Starting simulation...")
    run_castro_simulation("problem.initial_conditions_file=2d_desy_benchmark.h5")
    time_e = time.time()
    print(f"Simulation completed in {time_e - time_s:.2f} seconds.")

    # Remove generated plotfiles and checkpoints
    #cleanup_outputs('2d_desy_benchmark.h5')

if __name__ == "__main__":
    #test_2d_sedov_taylor()
    
    test_2d_desy_benchmark()
