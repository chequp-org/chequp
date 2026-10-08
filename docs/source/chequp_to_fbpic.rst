CHEQUP to FBPIC profile writer
==============================

``sim_folder/analysis/chequp_to_fbpic.py`` exports the plasma density profiles of a 2D r-z CHEQUP simulation
(plotfiles ``plt_2d_*``) at a given time into an openPMD file (``plasma_density.h5``) that can be used to
initialize the plasma in `FBPIC <https://fbpic.github.io>`_. Densities are written in m\ :sup:`-3`, on the
coarsest AMR level, for the neutral and ionized populations of H, He, N and Ar, and for the free electrons.

.. code-block:: python

    import sys
    sys.path.append("path/to/chequp/sim_folder/analysis")
    from chequp_to_fbpic import write_FBPIC_profile

    write_FBPIC_profile(input_path='./path_to_CHEQUP_sim',
                        output_path='./path_to_fbpic_input',
                        t_hydro=3.5e-9)

.. automodule:: chequp_to_fbpic
   :members:
   :show-inheritance:
   :undoc-members:
