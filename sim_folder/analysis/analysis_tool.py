import os
import re
import numpy as np
import yt
import tqdm
yt.set_log_level("error")

class CastroSimulation(object):
    """
    Class to postprocess the results of a Castro simulation
    """

    def __init__(self, run_dir, file_start):
        """
        Create object to analyze/plot simulation results

        Parameters:
        -----------
        run_dir: string
            path to the folder that contains the output files

        file_start: string
            beginning of the filenames, e.g. 'plt_1d_'
        """
        # Then extract time for each output
        self.ts = yt.load(os.path.join(run_dir, file_start + '*'), hint="castro")
        self.output_times = np.array([ float(ds.current_time) for ds in tqdm.tqdm(self.ts) ])


    def extract_data( self, t, quantity, level ):
        """
        Extract the quantity `quantity` at time `t`, at the required refinement level

        Parameters:
        -----------
        TODO
        quantity: possible values: 'density', 'rho_Hn', 'rho_Hp', 'pressure', 'Temp'

        Note: actually returns at a different time
        """
        i_output = np.argmin( abs(t-self.output_times) )
        ds = self.ts[i_output]
        if ds.dimensionality==1:
            ad = ds.covering_grid( level=level,
                            left_edge=ds.domain_left_edge,
                            dims=[ds.domain_dimensions[0]*2**level, 1, 1] )
            q = ad[quantity].to_ndarray().squeeze()
            r = np.linspace(
                ds.domain_left_edge[0],
                ds.domain_right_edge[0],
                ds.domain_dimensions[0]*2**level)
        elif ds.dimensionality:
            ad = ds.covering_grid( level=level,
                                left_edge=ds.domain_left_edge,
                                dims=[ds.domain_dimensions[0]*2**level, ds.domain_dimensions[1]*2**level, 1] )
            q = ad[quantity].to_ndarray().squeeze()
            if ds.geometry == 'cylindrical':
                r = np.linspace(
                    ds.domain_left_edge[0],
                    ds.domain_right_edge[0],
                    ds.domain_dimensions[0]*2**level)
                q = q[:, q.shape[1]//2]
            else:
                q = q[q.shape[0]//2:,q.shape[1]//2]
                r = np.linspace(
                    0.5*(ds.domain_left_edge[0] + ds.domain_right_edge[0]),
                    ds.domain_right_edge[0],
                    ds.domain_dimensions[0]*2**level//2)
                r -= 0.5*(ds.domain_left_edge[0] + ds.domain_right_edge[0])
        return r.to_ndarray(), q, ds.current_time.to_value()
