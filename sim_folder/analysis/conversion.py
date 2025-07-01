import numpy as np
import tqdm
from scipy.interpolate import RegularGridInterpolator
from analysis_tool import CastroSimulation
from scipy.constants import k, e

def interpolate_data(case):
    cs = CastroSimulation('../run/'+case, 'plt_1d_')
    path = 'Castro_benchmark/' + case

    for variable, path_end in zip(['density', 'rho_Hp', 'T_h', 'T_e'], ['_heavies', '_electrons', '_T_heavies', '_T_electrons']):

        # Extract data from different time
        # Note that time is not regularly spaced
        q_arr = []
        rmax_arr = []
        for time in tqdm.tqdm( cs.output_times ):
            r, q, t = cs.extract_data(time, variable, level=0)
            rmax = r[np.argmax(q)]
            rmax_arr.append(rmax)
            q_arr.append(q)
        q_arr = np.stack(q_arr)
        t_arr = cs.output_times.copy()
        r_arr = r

        # Interpolate on a grid with regularly-spaced time
        interp = RegularGridInterpolator(points=(t_arr, r_arr), values=q_arr, bounds_error=False, fill_value=None)
        t_interp, r_interp = np.meshgrid(
            np.linspace(0, t_arr.max(), 1001),
            np.linspace(0, r_arr.max(), 1001), indexing='ij')
        q_interp = interp((t_interp, r_interp))

        if variable.startswith('T'):
            q_interp = q_interp*k/e
        else:
            q_interp = q_interp/1.67e-6 * 1.e24

        np.save(path + path_end + '.npy', q_interp.T)

for case in [ '2T_yes_diff_noenergy_ioniz', '2T_yes_diff_yes_ioniz']:
    interpolate_data(case)