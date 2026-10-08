CHEQUP to WarpX density converter
=================================

``initial_condition/openpmd_conversion.py`` is a command-line tool that converts a 2D r-z CHEQUP plotfile into
an openPMD file containing a density profile (m\ :sup:`-3`) that can be used to initialize particles in
`WarpX <https://warpx.readthedocs.io>`_. The electron density is computed as the charge-weighted sum of
the ionized species.

.. code-block:: sh

    python initial_condition/openpmd_conversion.py --plotfile plt00300/
    python initial_condition/openpmd_conversion.py --plotfile plt00300/ --output my_density.h5
    python initial_condition/openpmd_conversion.py --plotfile plt00300/ --species heavy --level 3
    python initial_condition/openpmd_conversion.py --plotfile plt00300/ --all-species --verbose

Options:

* ``--plotfile``: path to the Castro plotfile (required).
* ``--output``: output file name (default ``openpmd_density.h5``).
* ``--level``: AMR level to extract (default: finest level).
* ``--species``: ``electron`` (default), ``H0``, ``H1`` or ``heavy`` (H0 + H1).
* ``--all-species``: write all of the above.
* ``--verbose``: print details during processing.

.. automodule:: openpmd_conversion
   :members:
   :undoc-members:
