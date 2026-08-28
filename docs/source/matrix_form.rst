Multi-species matrix formalism
------------------------------

CHEQUP is a plasma-hydrodynamics module built on top of the Castro AMR solver. It evolves a system using the material (Lagrangian) derivative defined as:

.. math::

   D_t \equiv \partial_t + \mathbf{u} \cdot \nabla


.. rubric:: Basic Equations

The simplified system of conservation laws is as follows:

1. **Mass continuity:** Total mass is conserved.

.. math::

   D_t\rho = 0

2. **Momentum:** Solves the inviscid Euler equations.

.. math::

   D_t(\rho\mathbf{u}) = -\nabla p

3. **Species mass fractions:** Driven by ionization and recombination.

.. math::

   D_t\boldsymbol{X}_s = \mathbf{S}_{r,s}\,\boldsymbol{X}_s

4. **Heavy-particle energy fraction:** Tracks the energy fraction :math:`f = e_h/e`.

.. math::

   D_t(\rho f) = S_{\text{Dth}} + S_{\text{ion}} + S_{\text{coll}}


.. rubric:: Matrix Scaling Example

For a mixture of Hydrogen and Nitrogen (ionized up to :math:`\text{N}^{2+}`), the state vectors for each level are stacked into a global vector. The system scales cleanly into a block-diagonal matrix:

.. math::

   D_t \begin{bmatrix} X_{\text{H}} \\ X_{\text{H}^+} \\ X_{\text{N}} \\ X_{\text{N}^+} \\ X_{\text{N}^{2+}} \end{bmatrix} = \begin{bmatrix} \mathbf{S}_{r,\text{H}} & \mathbf{0} \\ \mathbf{0} & \mathbf{S}_{r,\text{N}} \end{bmatrix} \begin{bmatrix} X_{\text{H}} \\ X_{\text{H}^+} \\ X_{\text{N}} \\ X_{\text{N}^+} \\ X_{\text{N}^{2+}} \end{bmatrix}

This block-diagonal structure occurs because electron-impact ionization and recombination couple levels *within* a specific species, but not between different species.

.. rubric:: The Reaction Matrix

The ionization-level populations of species :math:`s` obey :math:`D_t \boldsymbol{X}_s = \mathbf{S}_{r,s}(T_e, X_e)\,\boldsymbol{X}_s`. The reaction-rate matrix factors into :math:`\mathbf{S}_{r,s} = \frac{\rho^2}{m_p}\,\mathbf{R}_s`.

Because every reaction shifts the charge by exactly one unit, the dimensionless reaction kernel :math:`\mathbf{R}_s` is strictly **tridiagonal**:

.. math::

   (\mathbf{R}_s)_{\mu\mu} &= -X_e I_s^\mu - X_e^2 T_s^{\mu-1} \\
   (\mathbf{R}_s)_{\mu,\mu-1} &= X_e I_s^{\mu-1} \\
   (\mathbf{R}_s)_{\mu,\mu+1} &= X_e^2 T_s^\mu

**Conservation property:** Every column of :math:`\mathbf{R}_s` sums to zero, ensuring that ionization and recombination only redistribute mass between charge states without destroying or creating it.


.. rubric:: Reaction Rate Computation

Rate coefficients :math:`\langle\sigma v\rangle(T_e)` are obtained by averaging the collision cross sections over a Maxwellian velocity distribution:

.. math::

   R(T_e) = \frac{2\,T_e^{-3/2}}{\sqrt{\pi m_e}} \int_{\Delta E}^{\infty} \sigma(E)\,e^{-E/T_e}\,E\,\mathrm{d}E

.. rubric:: Direct Ionization

Calculated using the Relativistic Binary Encounter Bethe (BEB) model by summing contributions from all occupied orbitals :math:`k`:

.. math::

   \sigma_{\text{DI}}(E) = \sum_{k=1}^{Z-Z^*} \sigma_k(E)

.. rubric:: Excitation-Ionization

Modeled via the Van Regemorter formula for allowed dipole transitions (where :math:`f_{ij}` is the oscillator strength and :math:`g` is the Gaunt factor):

.. math::

   \sigma_{\text{EI}}(E) = \frac{8\pi^2 a_0^2}{\sqrt{3}} \left(\frac{\Delta E}{E}\right)^{\!2} \frac{g(E/\Delta E)\,f_{ij}}{E/\Delta E}

.. rubric:: Three-Body Recombination

Derived directly from ionization rates via microscopic reversibility (detailed balance), which guarantees automatic relaxation to the correct Saha equilibrium:

.. math::

   R_{3b}^{z\to z-1}(T_e) = \left(R_{\text{DI}}^{z-1\to z}\,e^{\varepsilon_{\text{ion}}/T_e} + R_{\text{EI}}^{z-1\to z}\,e^{\varepsilon_{\text{ex}}/T_e}\right) \frac{g_{z-1}}{g_z}\, \lambda_{\text{dB}}^{-3}(T_e)


.. rubric:: Energy Equations

Due to the massive difference between electron and ion mass (:math:`m_e/m_i \sim 10^{-4}`), thermal equilibration is slow. CHEQUP utilizes a two-temperature model that separates the electron and heavy-particle temperatures by tracking the heavy-particle energy fraction :math:`f = e_h/e`.

.. rubric:: Collision Frequencies

* **Electron-ion collisions** scale with the Coulomb logarithm :math:`\Lambda_{ei}` and a collision prefactor :math:`C(T_e)`:

  .. math::

     \nu_{ei}(s,\mu) = n_e\,C(T_e)\,(z_s^\mu)\,\Lambda_{ei}

* **Electron-neutral collisions** rely on momentum-transfer cross sections fitted from collision data:

  .. math::

     \nu_{en}(s) = n_{n,s}\,\frac{4}{3}\sqrt{\frac{8k_B T_e}{\pi M_{es}}} \,\sigma_{es}

.. rubric:: Energy Source Terms

The heavy-particle energy fraction evolves based on three primary source terms:

1. **Thermal Diffusion (** :math:`S_{\text{Dth}}` **)**: Electron thermal conduction deposited into the heavy-particle pool.

   .. math::

      S_{\text{Dth}} = -\frac{f}{e}\,\nabla\cdot(\lambda_e\,\nabla T_e)

2. **Ionization Energy Exchange (** :math:`S_{\text{ion}}` **)**: The energy extracted from (or returned to) the electron pool during ionization and recombination.

   .. math::

      S_{\text{ion}} = -\frac{f}{e}\sum_s \frac{\delta\boldsymbol{\varepsilon}_s^T\,\mathbf{R}_s\,\boldsymbol{X}_s}{m_s}

3. **Elastic Energy Exchange (** :math:`S_{\text{coll}}` **)**: The kinetic energy transfer driven by elastic collisions between electrons and heavy particles.

   .. math::

      s_{\text{coll}}(s,\mu) = \rho\,\nu_{es}^\varepsilon(s,\mu) \bigl[1 - (1 + X_s^\mu)f\bigr]

   .. math::

      S_{\text{coll}} = \sum_s\sum_{\mu=1}^{Z_s} s_{\text{coll}}(s,\mu)


.. rubric:: Notation and Variables

.. list-table:: 
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`D_t`
     - Material (Lagrangian) derivative
   * - :math:`\rho`
     - Total mass density
   * - :math:`\mathbf{u}`
     - Bulk velocity field
   * - :math:`p`
     - Pressure
   * - :math:`e`
     - Total specific internal energy
   * - :math:`f`
     - Fraction of internal energy in heavy particles (:math:`e_h/e \in [0,1]`)
   * - :math:`X_s^\mu`
     - Mass fraction of species :math:`s` at ionization level :math:`\mu`
   * - :math:`\boldsymbol{X}_s`
     - Column vector of mass fractions for species :math:`s`
   * - :math:`X_e`
     - Electron mass fraction
   * - :math:`T_e, T_h`
     - Electron temperature / heavy-particle temperature
   * - :math:`n_e`
     - Electron number density
   * - :math:`n_{n,s}`
     - Neutral number density of species :math:`s`
   * - :math:`z_s^\mu`
     - Charge (number of elementary charges) of species :math:`s` at level :math:`\mu`
   * - :math:`m_s, m_e, m_p`
     - Atomic mass of species :math:`s`, electron mass, and proton mass
   * - :math:`k_B`
     - Boltzmann constant
   * - :math:`\mathbf{S}_{r,s}`
     - Reaction-rate matrix for species :math:`s`
   * - :math:`\mathbf{R}_s`
     - Dimensionless reaction kernel
   * - :math:`I_s^\mu, T_s^\mu`
     - Ionization and three-body recombination rate coefficients
   * - :math:`R(T_e)`
     - Thermally averaged reaction rate coefficient
   * - :math:`\sigma(E)`
     - Collision cross-section as a function of energy
   * - :math:`E, \Delta E`
     - Electron kinetic energy and reaction threshold energy
   * - :math:`\varepsilon_{\text{ion}}, \varepsilon_{\text{ex}}`
     - Ionization and excitation threshold energies
   * - :math:`\delta\boldsymbol{\varepsilon}_s`
     - Vector of cumulative ionization energies for species :math:`s`
   * - :math:`\lambda_{\text{dB}}`
     - Thermal de Broglie wavelength
   * - :math:`g_z`
     - Statistical weight (spin degeneracy) of charge state :math:`z`
   * - :math:`\nu_{ei}, \nu_{en}, \nu_{es}`
     - Electron-ion, electron-neutral, and total electron-species collision frequencies
   * - :math:`\Lambda_{ei}`
     - Coulomb logarithm
   * - :math:`\lambda_e`
     - Electron thermal conductivity
   * - :math:`S_{\text{Dth}}, S_{\text{ion}}, S_{\text{coll}}`
     - Energy source terms: thermal diffusion, ionization, and elastic collisions