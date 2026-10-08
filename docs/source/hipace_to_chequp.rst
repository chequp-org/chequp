Hipace++ to CHEQUP input writer
===============================

``sim_folder/analysis/hipace_to_chequp.py`` builds a CHEQUP initial-condition file (1D or 2D r-z) from a
HiPACE++ simulation of laser ionization: it reads the ion densities of each ionization level
(``grid_ionization_w_ion_<atom>_<level>``) and the electron temperature computed from the ionization-electron
momenta, interpolates them onto a zoom window, keeps ``r >= 0`` and writes the file with ``save_to_openpmd``.
Supported elements are H, He, N and Ar. It requires ``openpmd-viewer`` and ``pytools``.

.. code-block:: python

    from hipace_to_chequp import HipaceToChequpWriter

    writer = HipaceToChequpWriter(
        input="/data/runs/my_hipace_run/diags/hdf5",
        output="2d_input.h5",
        species=['H', 'Ar'],
        dim=2,
        r_zoom_um=(0, 50),
        z_zoom_cm=(10, 15),
        N_new=(200, 500)
    )
    writer.write_input(plot=True)

.. automodule:: hipace_to_chequp
   :members:
   :show-inheritance:
   :undoc-members:
