import subprocess
import re
import numpy as np
import sys
import glob
import os
import pytest
from numba import njit
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

code_dir = '../../chequp'
sim_folder = f"{code_dir}/tests/"

sys.path.append(f"{code_dir}/initial_condition")
sys.path.append(f"{code_dir}/sim_folder/analysis")
sys.path.append(f"{code_dir}/theory/sedov_theory/python/")

from ionization_routines import save_to_openpmd
from analysis_tool import CastroSimulation

def get_species_indices():
    " Returns a list of species indices from the species.net file. "
    with open(f'{code_dir}/sim_folder/build/species.net', 'r') as f:
        content = f.read()
        species = re.findall(r'\n\s.*\s([A-Z][a-z]*\d)', content)
    return species

def run_castro_simulation(model='gamma_law_2T', runtime_options=''):
    " Runs a Castro simulation. "
    build_dir = os.path.abspath(f"{code_dir}/sim_folder/build/")
    executables = glob.glob(os.path.join(build_dir, f"Castro1d*.{model}.ex"))
    if len(executables) == 0:
        raise FileNotFoundError(f"No Castro1d executable found in {build_dir}")
    elif len(executables) > 1:
        raise RuntimeError(f"Multiple Castro1d executables found: {executables}")
    executable = os.path.abspath(executables[0])

    inputs = os.path.abspath(f"{code_dir}/sim_folder/run/inputs.1d.cyl")
    os.makedirs(sim_folder, exist_ok=True)

    command = f"rm -rf plt_* chk* amr_diag.out species_diag.out grid_diag.out Backtrace.0 && {executable} {inputs} {runtime_options}"

    try:
        subprocess.run(
            command,          
            shell=True,
            cwd=sim_folder,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise


# CONSTANTS & RATE TABLES 

# Extracted specifically for Hydrogen (Z=1) and Argon (Z=18)
BINDING_ENERGIES_H = np.array([13.5981])
BINDING_ENERGIES_AR = np.array([
    3206.2, 3206.2, 324.2, 324.2, 247.74, 247.74, 247.74, 247.74, 
    247.74, 247.74, 29.24, 29.24, 15.76, 15.76, 15.76, 15.76, 15.76, 15.76
])

gaunt_fit = {
    "H":  [-7.61094672e-06,  1.42014919e-04, -5.26821513e-04, -2.27233338e-03,  2.38468040e-03,  5.36211118e-02,  2.56972926e-01],
    "Ar": [ 1.12925396e-05,  7.83928108e-05, -8.07569320e-04, -1.45478529e-03,  4.14430633e-03,  5.21310441e-02,  2.49330993e-01]
}

# Same as in the CHEQUP code
OSC_STRENGTHS = {"H": 0.416, "Ar": 0.12}
EX_ENERGIES   = {"H": 10.6,  "Ar": 10.3}

# Expanded Ar to include Ar IV through Ar IX
ION_ENERGIES = {
    "H":  [13.598434],
    "Ar": [15.759611, 27.62967, 40.735, 59.81, 75.02, 91.009, 124.323, 143.46, 422.45]
}

# Expanded Ar degeneracies up to index 9
g_deg = {
    "H":  np.array([2.0, 1.0]),
    "Ar": np.array([1.0, 4.0, 5.0, 4.0, 1.0, 2.0, 1.0, 2.0, 1.0, 4.0])
}

HBAR_CGS  = 1.0545718e-27
M_E_CGS   = 9.1093837e-28
EV_TO_ERG = 1.602176634e-12

def _get_species_str(Z):
    " Returns the species string for a given atomic number. "
    if Z == 1:  return "H"
    if Z == 18: return "Ar"
    raise ValueError(f"Unknown Z={Z}")

def get_rate_ex_ion(Te_eV, Z, Zstar):
    " Returns the excitation ionization rate for a given atomic number and temperature. "
    rate = 0.0
    if Zstar != 0: return rate
    x = np.log(Te_eV)
    idx_str = _get_species_str(Z)
    poly = np.polyval(gaunt_fit[idx_str], x)
    gaunt_value = np.exp(poly)
    dE = EX_ENERGIES[idx_str]
    f  = OSC_STRENGTHS[idx_str]
    prefactor = 1.57e-7
    rate = (prefactor * f * gaunt_value) / (dE * np.sqrt(Te_eV)) * np.exp(-dE / Te_eV)
    return rate

def get_rate_3body(Te_eV, Z, Zstar):
    " Returns the 3-body recombination rate for a given atomic number and temperature. "
    idx_str = _get_species_str(Z)
    
    if Zstar + 1 >= len(g_deg[idx_str]):
        return 0.0
        
    lambda_db = HBAR_CGS * np.sqrt((2.0 * np.pi) / (M_E_CGS * Te_eV * EV_TO_ERG))
    reverse_factor = (lambda_db**3) * (g_deg[idx_str][Zstar] / (2.0 * g_deg[idx_str][Zstar + 1]))

    E_ion = ION_ENERGIES[idx_str][Zstar]
    E_ex = EX_ENERGIES[idx_str]
    
    rate_dir = _compute_ion_rate_scalar(Te_eV, Z, Zstar) * np.exp(min(E_ion / Te_eV, 500.0))
    rate_ex = get_rate_ex_ion(Te_eV, Z, Zstar) * np.exp(min(E_ex / Te_eV, 500.0))
    
    rate_forward = rate_dir + rate_ex
    return min(rate_forward * reverse_factor, rate_forward)

@njit(cache=True)
def fast_cross_section_jit(Z, Zstar, E_grid, me_c2, binding_arr, E_ion_true):
    " Computes the cross-section for a given atomic number and temperature. "
    normalization = 2 * np.pi * (1/137)**4 * (5.29e-11)**2
    sigma = np.zeros(len(E_grid))
    
    num_simulated = 1
    if Z == 1:   num_simulated = 1
    elif Z == 18: num_simulated = 9 # Expanded to 9
    
    for i in range(len(E_grid)):
        ep = E_grid[i] / me_c2
        if ep <= 0.0: continue
        
        N = 1
        max_k = num_simulated - Zstar
        k_start = Z - num_simulated
        
        for step in range(max_k):
            k = k_start + step
            bp = binding_arr[k] / me_c2
            neutral_valence_bp = binding_arr[Z - 1] / me_c2
            
            if step < (max_k - 1):
                if bp == binding_arr[k + 1] / me_c2:
                    N += 1
                    continue
            
            if abs(bp - neutral_valence_bp) < 1e-5:
                bp = E_ion_true / me_c2
                
            e = ep / bp
            if e > 1.0:
                betae2 = 1.0 - 1.0 / (1.0 + ep)**2
                betab2 = 1.0 - 1.0 / (1.0 + bp)**2
                betau2 = betab2 
                s0  = normalization * N / (bp * (betae2 + betab2 + betau2))
                ep2 = 1.0 / (1.0 + 0.5 * ep)**2
                A1  = (1.0 + 2.0 * ep) / (1.0 + e) * ep2
                A2  = 0.5 * (e - 1.0) * bp * bp * ep2
                A3  = np.log(betae2 / (1.0 - betae2)) - betae2 - np.log(2.0 * bp)
                sk  = s0 * (0.5 * A3 * (1.0 - 1.0/e**2) + 1.0 - 1.0/e + A2 - A1*np.log(e))
                sigma[i] += sk
            N = 1
            
    return sigma * 1e4

def _compute_ion_rate_scalar(Te_eV, Z, Zstar):
    " Computes the ionization rate for a given atomic number and temperature. "
    me_c2, me, eV_to_erg = 510998.9, 9.10938356e-28, 1.602176634e-12
    idx_str = _get_species_str(Z)
    E_ion_true = ION_ENERGIES[idx_str][Zstar]
    
    binding_arr = BINDING_ENERGIES_H if Z == 1 else BINDING_ENERGIES_AR
    
    E         = np.logspace(0.1, 6, 200)
    sigma     = fast_cross_section_jit(Z, Zstar, E, me_c2, binding_arr, E_ion_true)
    E_erg     = E * eV_to_erg
    Te_erg    = Te_eV * eV_to_erg
    integrand = sigma * E_erg * np.exp(-E_erg / Te_erg)
    integral  = np.trapz(integrand, E_erg)
    prefactor = np.sqrt(8 / (np.pi * me)) * Te_erg**(-3/2)
    return prefactor * integral

_TE_GRID = np.logspace(-1, 3, 800)

class RateTables:
    " Class that stores the ionization rates for a given temperature grid. "
    def __init__(self, Te_grid: np.ndarray = _TE_GRID):
        self.log_Te = np.log(Te_grid)
        self._ion, self._3b, self._ex = {}, {}, {}
        
        # Only compute the states needed for the H-Ar mix up to Ar8
        specs = [
            (1, 0),
            (18, 0), (18, 1), (18, 2), (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8)
        ]
        
        for Z, Zstar in specs:
            ion_vals = np.maximum(np.array([_compute_ion_rate_scalar(Te, Z, Zstar) for Te in Te_grid]), 1e-300)
            r3b_vals = np.maximum(np.array([get_rate_3body(Te, Z, Zstar) for Te in Te_grid]), 1e-300)
            ex_vals  = np.maximum(np.array([get_rate_ex_ion(Te, Z, Zstar) for Te in Te_grid]), 1e-300)
            
            self._ion[(Z, Zstar)] = np.log(ion_vals)
            self._3b[(Z, Zstar)]  = np.log(r3b_vals)
            self._ex[(Z, Zstar)]  = np.log(ex_vals)

    def get_H_tables(self):
        """Returns flat arrays for Hydrogen."""
        return self._ion[(1,0)], self._3b[(1,0)], self._ex[(1,0)]
        
    def get_Ar_tables(self):
        """Returns 2D C-contiguous arrays for Argon states (0 to 8)."""
        ion = np.ascontiguousarray(np.vstack(tuple(self._ion[(18,i)] for i in range(9))))
        r3b = np.ascontiguousarray(np.vstack(tuple(self._3b[(18,i)] for i in range(9))))
        ex  = np.ascontiguousarray(np.vstack(tuple(self._ex[(18,i)] for i in range(9))))
        return ion, r3b, ex

# NUMBA OPTIMIZED ODE BUILDER

import numpy as np
from numba import njit

@njit(cache=True)
def ode_rhs_ar_h_jit(t, state, log_Te_grid,
                     log_ion_H, log_3b_H, log_ex_H, E_ion_H, E_ex_H,
                     log_ion_Ar, log_3b_Ar, log_ex_Ar, E_ion_Ar, E_ex_Ar,
                     idx_H0, idx_H1, idx_Ar):
    
    "ODE Solver for the Argon-Hydrogen mixture."
    # Extract species densities
    n = state[:-1]
    
    # Extract electron temperature
    Te = max(state[-1], 0.01)
    log_Te = np.log(Te)

    # Calculate electron density (n_e) assuming macroscopic quasineutrality.
    n_e = 1e-10 + n[idx_H1]
    for i in range(1, 9):
        n_e += float(i) * n[idx_Ar[i]]

    # Initialize the derivative array to zero
    dstate = np.zeros_like(state)
    
    # Trackers for the electron energy equation
    total_ion_rate = 0.0
    total_energy_loss = 0.0

    # Hydrogen Kinetics
    
    # Interpolate log-scaled rate coefficients and convert back to linear scale.
    k_ion_H = np.exp(np.interp(log_Te, log_Te_grid, log_ion_H)) * 1e-6
    k_3b_H  = np.exp(np.interp(log_Te, log_Te_grid, log_3b_H)) * 1e-12
    k_ex_H  = np.exp(np.interp(log_Te, log_Te_grid, log_ex_H)) * 1e-6

    # Effective ionization rate (direct ionization + step-wise excitation-ionization)
    k_ion_H_tot = k_ion_H + k_ex_H
    
    # Energy lost by electrons due to hydrogen excitation
    total_energy_loss += n_e * n[idx_H0] * k_ex_H * E_ex_H

    # Net creation rate of H+ ions: (Ionization) - (3-body recombination)
    net_H = n_e * n[idx_H0] * k_ion_H_tot - k_3b_H * (n_e**2) * n[idx_H1]
    
    # Update state derivatives for neutral H and ionized H+
    dstate[idx_H0] -= net_H
    dstate[idx_H1] += net_H
    
    # Accumulate global source terms for the temperature equation
    total_ion_rate += net_H
    total_energy_loss += net_H * E_ion_H

    # Argon Kinetics
    
    # Iterate through Argon charge states (Ar0 -> Ar7)
    for i in range(8):
        k_ion = np.exp(np.interp(log_Te, log_Te_grid, log_ion_Ar[i])) * 1e-6
        k_3b  = np.exp(np.interp(log_Te, log_Te_grid, log_3b_Ar[i])) * 1e-18

        # Excitation is only considered for neutral Argon (i == 0) in this model
        if i == 0:
            k_ex = np.exp(np.interp(log_Te, log_Te_grid, log_ex_Ar[0])) * 1e-6
            k_ion += k_ex # Add excitation to total ionization pathway
            total_energy_loss += n_e * n[idx_Ar[0]] * k_ex * E_ex_Ar

        # Net creation rate of the next higher Argon charge state
        net_Ar = n_e * n[idx_Ar[i]] * k_ion - k_3b * (n_e**2) * n[idx_Ar[i+1]]
        
        # Update derivatives: deplete current state, populate next state
        dstate[idx_Ar[i]] -= net_Ar
        dstate[idx_Ar[i+1]] += net_Ar
        
        # Accumulate global source terms
        total_ion_rate += net_Ar
        total_energy_loss += net_Ar * E_ion_Ar[i]

    # Calculate dT_e/dt. 
    # Represents the cooling of the electron gas due to inelastic collisions 
    # and the energy cost of creating new electrons.
    dstate[-1] = -(total_energy_loss + 1.5 * Te * total_ion_rate) / max(1.5 * n_e, 1e-30)
    
    return dstate

def make_ode_ar_h(tables: RateTables, idx_H0, idx_H1, idx_Ar):
    """Factory that extracts variables and functions from the RateTables object
    and returns a function that can be used in the ODE solver.
    """
    log_Te_grid = tables.log_Te
    
    log_ion_H, log_3b_H, log_ex_H = tables.get_H_tables()
    log_ion_Ar, log_3b_Ar, log_ex_Ar = tables.get_Ar_tables()

    E_ion_H = ION_ENERGIES["H"][0]
    E_ex_H  = EX_ENERGIES["H"]
    E_ion_Ar = np.array(ION_ENERGIES["Ar"][:8]) # Extract first 8 for loop
    E_ex_Ar  = EX_ENERGIES["Ar"]

    def wrapper(t, state):
        return ode_rhs_ar_h_jit(
            t, state, log_Te_grid,
            log_ion_H, log_3b_H, log_ex_H, E_ion_H, E_ex_H,
            log_ion_Ar, log_3b_Ar, log_ex_Ar, E_ion_Ar, E_ex_Ar,
            idx_H0, idx_H1, idx_Ar
        )
    return wrapper
    
# PYTEST SETUP & ASSERTIONS

def assert_densities_match(t_chequp, n_chequp, t_ode, n_ode, tol, species_name):
    """Interpolates ODE solution and asserts deviation from CHEQUP is within tolerance."""
    interp_func = interp1d(t_ode, n_ode, kind='linear', fill_value='extrapolate')
    n_ode_mapped = interp_func(t_chequp)
    
    peak_density = np.max(n_ode_mapped)
    if peak_density < 1e-10:
        peak_density = 1e-10
        
    rel_diff = np.abs(n_chequp - n_ode_mapped) / peak_density
    max_diff = np.max(rel_diff) * 100
    
    assert max_diff <= tol, (
        f"Species {species_name} failed! Max relative difference {max_diff:.4e} "
        f"exceeds tolerance {tol}"
    )

# UNIT TESTS

def test_0D_Ar_H_mix(tol=11):
    " Unit test for the 0D Ar-H mixture. "
    r, n0, dt, t_max = np.linspace(0, 1e-6, 256), 3e22, 0.1e-9, 4e-9
    T_eV, species_keys = np.ones_like(r) * 600.0, get_species_indices()
    densities = np.zeros((len(r), len(species_keys)))
    # Initialize the densities (arbitrary values)
    densities[:, species_keys.index('H0')]  = 0.45 * n0
    densities[:, species_keys.index('H1')]  = 0.05 * n0
    densities[:, species_keys.index('Ar0')] = 0.45 * n0
    densities[:, species_keys.index('Ar1')] = 0.05 * n0

    save_to_openpmd({'r': [r.min(), r.max()]}, densities, T_eV,
                    f'{sim_folder}/init.h5', species_keys)
    # Run the CHEQUP simulation
    runtime_options = (
        f"max_step=100000000 geometry.is_periodic=0 castro.lo_bc=3 castro.hi_bc=3 "
        f"amr.n_cell=8 geometry.prob_hi=0.0001 amr.max_level=0 castro.add_ext_src=1 "
        f"stop_time={t_max} amr.plot_per={dt} amr.plot_int=1000000 "
        f"problem.initial_conditions_file=init.h5 castro.cfl=0.1 "
        f"castro.diffuse_temp=0 amr.derive_plot_vars=ALL"
    )
    run_castro_simulation(model='gamma_law_2T', runtime_options=runtime_options)
    sim_data = CastroSimulation(sim_folder, 'plt*')
    
    # Extract the densities from the HDF5 file
    nH0_chequp = np.array([sim_data.get_field(t, 'rho_H0', 0)['q'][0] for t in sim_data.output_times]) / 1.66e-30
    nH1_chequp = np.array([sim_data.get_field(t, 'rho_H1', 0)['q'][0] for t in sim_data.output_times]) / 1.66e-30
    
    nAr_chequp = np.array([
        np.array([sim_data.get_field(t, f'rho_Ar{i}', 0)['q'][0]
         for t in sim_data.output_times]) / (39.9 * 1.66e-30)
        for i in range(9) # Expanded loop to 9
    ]) 
    # Extract the indices of the species
    idx_H0  = species_keys.index('H0')
    idx_H1  = species_keys.index('H1')
    idx_Ar  = np.array([species_keys.index(f'Ar{i}') for i in range(9)]) # Expanded array to 9
    
    # Initialize the ODE directly via the optimized factory
    ode_func = make_ode_ar_h(RateTables(), idx_H0, idx_H1, idx_Ar)
    
    sol = solve_ivp(
        ode_func, [0, t_max], np.append(densities[0, :], T_eV[0]),
        method='Radau', t_eval=np.arange(0, t_max, dt), rtol=1e-2
    )
    # Extract the solution times and densities
    t_c = sim_data.output_times
    t_o = sol.t
    # Compare the solution to the CHEQUP solution
    assert_densities_match(t_c, nH0_chequp, t_o, sol.y[idx_H0], tol, "Mix H0")
    assert_densities_match(t_c, nH1_chequp, t_o, sol.y[idx_H1], tol, "Mix H1")
    
    for i in range(9): # Expanded assertion loop to 9
        assert_densities_match(t_c, nAr_chequp[i], t_o, sol.y[idx_Ar[i]], tol, f"Mix Ar{i}")

if __name__ == "__main__":
    test_0D_Ar_H_mix()