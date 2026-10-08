Multi-species matrix formalism
------------------------------

CHEQUP is a plasma-hydrodynamics module built on top of the Castro AMR solver. The plasma is described as a
single quasi-neutral fluid whose composition is given by the mass fractions of every ionization stage of every
element (the species of ``sim_folder/build/species.net``). Castro solves the compressible Euler equations;
CHEQUP adds the atomic-physics source terms (``problem_source.H``), the electron thermal conductivity
(``actual_conductivity.H``) and, in the two-temperature model, the transport of the heavy-particle energy fraction.

In what follows, :math:`D_t \equiv \partial_t + \mathbf{u} \cdot \nabla` is the material derivative,
so that :math:`\partial_t(\rho a) + \nabla\cdot(\rho a\,\mathbf{u}) = \rho D_t a` for any specific quantity :math:`a`.


.. rubric:: Governing Equations

1. **Mass continuity:**

.. math::

   \partial_t\rho + \nabla\cdot(\rho\mathbf{u}) = 0

2. **Momentum** (inviscid Euler equations):

.. math::

   \partial_t(\rho\mathbf{u}) + \nabla\cdot(\rho\mathbf{u}\mathbf{u}) = -\nabla p

3. **Total energy**, with electron thermal conduction (``castro.diffuse_temp = 1``) and the energy spent in
   ionization (``castro.add_ext_src = 1``):

.. math::

   \partial_t(\rho E) + \nabla\cdot\big[(\rho E + p)\,\mathbf{u}\big] = \nabla\cdot(\lambda_e\nabla T_e) + Q_{\text{ion}}

4. **Species mass fractions**, driven by ionization and recombination:

.. math::

   \rho\, D_t\boldsymbol{X}_s = \mathbf{S}_{r,s}\,\boldsymbol{X}_s

5. **Heavy-particle energy fraction** :math:`f = e_h/e` (auxiliary variable ``f_heavies``, two-temperature model):

.. math::

   \rho\, D_t f = S_{\text{Dth}} + S_{\text{ion}} + S_{\text{coll}}


.. rubric:: Equation of State

All species share the same adiabatic index :math:`\gamma` (``eos.eos_gamma``), and :math:`p = (\gamma-1)\rho e`.
The electron fraction (number of free electrons per atomic mass unit) follows from quasi-neutrality:

.. math::

   X_e = \sum_{s,\mu} \frac{z_s^\mu X_s^\mu}{A_s}

* With ``EOS_DIR=gamma_law`` (single temperature), electrons and heavy particles share the same temperature.
* With ``EOS_DIR=gamma_law_2T`` (two temperatures), the internal energy is split between the electrons,
  :math:`(1-f)\,e`, and the heavy particles, :math:`f\,e`, which gives

.. math::

   T_e = \frac{(\gamma-1)(1-f)\,e\,m_u}{k_B X_e}, \qquad
   T_h = \frac{(\gamma-1)\,f\,e\,m_u}{k_B \sum_{s,\mu} X_s^\mu/A_s}

The temperature state variable used by the source terms and by the thermal conductivity is the electron
temperature :math:`T_e` (the common temperature in the single-temperature model).


.. rubric:: Matrix Scaling Example

For a mixture of Hydrogen and Nitrogen (ionized up to :math:`\text{N}^{2+}`), the state vectors for each level are stacked into a global vector. The system scales cleanly into a block-diagonal matrix:

.. math::

   \rho\,D_t \begin{bmatrix} X_{\text{H}} \\ X_{\text{H}^+} \\ X_{\text{N}} \\ X_{\text{N}^+} \\ X_{\text{N}^{2+}} \end{bmatrix} = \begin{bmatrix} \mathbf{S}_{r,\text{H}} & \mathbf{0} \\ \mathbf{0} & \mathbf{S}_{r,\text{N}} \end{bmatrix} \begin{bmatrix} X_{\text{H}} \\ X_{\text{H}^+} \\ X_{\text{N}} \\ X_{\text{N}^+} \\ X_{\text{N}^{2+}} \end{bmatrix}

This block-diagonal structure occurs because electron-impact ionization and recombination couple levels *within* a specific species, but not between different species.

.. rubric:: The Reaction Matrix

The reaction-rate matrix factors into :math:`\mathbf{S}_{r,s} = \frac{\rho^2}{m_p}\,\mathbf{R}_s`.
Because every reaction shifts the charge by exactly one unit, the reaction kernel :math:`\mathbf{R}_s` is strictly **tridiagonal**:

.. math::

   (\mathbf{R}_s)_{\mu\mu} &= -X_e I_s^\mu - X_e^2 T_s^{\mu-1} \\
   (\mathbf{R}_s)_{\mu,\mu-1} &= X_e I_s^{\mu-1} \\
   (\mathbf{R}_s)_{\mu,\mu+1} &= X_e^2 T_s^\mu

where :math:`I_s^\mu = R_{\text{DI}} + R_{\text{EI}}` is the ionization rate coefficient of stage :math:`\mu`
(direct ionization plus excitation-ionization) and :math:`T_s^\mu` the three-body recombination coefficient from
stage :math:`\mu+1` to stage :math:`\mu`.

**Conservation property:** Every column of :math:`\mathbf{R}_s` sums to zero, ensuring that ionization and recombination only redistribute mass between charge states without destroying or creating it.

In practice, the source term is assembled as a sum over transitions. The net mass rate of the transition
:math:`\mu \to \mu+1` of element :math:`s` is

.. math::

   \Gamma_s^\mu = \frac{\rho^2}{m_p}\Big[X_e\,(R_{\text{DI}} + R_{\text{EI}})\,X_s^\mu - X_e^2\,T_s^\mu\,X_s^{\mu+1}\Big]

which is removed from stage :math:`\mu` and added to stage :math:`\mu+1`.

.. note::

   * Only the stages whose mass fraction exceeds :math:`X_{\min} = 10^{-20}`, and their direct neighbors, are
     included in the computation, which skips empty ionization stages.
   * The highest stage of each element in ``species.net`` (N\ :sup:`5+`, Ar\ :sup:`8+`, H\ :sup:`+`, He\ :sup:`2+`)
     is not ionized further.


.. rubric:: Reaction Rate Computation

Rate coefficients :math:`\langle\sigma v\rangle(T_e)` are obtained by averaging the collision cross sections over a Maxwellian electron distribution:

.. math::

   R(T_e) = \frac{\int_0^{\infty} \sigma(E)\,v(E)\,\sqrt{E}\,e^{-E/T_e}\,\mathrm{d}E}{\int_0^{\infty} \sqrt{E}\,e^{-E/T_e}\,\mathrm{d}E},
   \qquad v(E) = \sqrt{2E/m_e}

The integrals are evaluated numerically (1000 points between 0 and 1 keV). To keep the source term cheap, the
direct-ionization rates of every species are tabulated at start-up (``coll_ion::initialize()``) on 100
logarithmically spaced temperatures between :math:`10^{-3}` and 300 eV, and linearly interpolated at run time
(temperatures outside this range use the closest tabulated value).

.. rubric:: Direct Ionization

Calculated using the Relativistic Binary Encounter Bethe (RBEB) model :cite:p:`Kim_Santos_Parente_2000`, by summing the contributions of all occupied orbitals :math:`k` of the ion (orbitals with the same binding energy are grouped, and the binding energy of the outermost orbital is replaced by the ionization energy of the current stage):

.. math::

   \sigma_{\text{DI}}(E) = \sum_{k=1}^{Z-Z^*} \sigma_k(E)

.. rubric:: Excitation-Ionization

Ionization through an excited state is included for the **neutral** atoms only, using the Van Regemorter rate for allowed dipole transitions :cite:p:`Van_Regemorter_1962`:

.. math::

   R_{\text{EI}}(T_e) = 1.57\times10^{-7}\;\frac{f_{ij}\,\bar{g}(T_e)}{\Delta E\,\sqrt{T_e}}\;e^{-\Delta E/T_e}
   \quad [\mathrm{cm^3\,s^{-1}}], \qquad T_e, \Delta E \text{ in eV}

where the thermally averaged Gaunt factor is fitted as :math:`\bar{g}(T_e) = \exp\big[\sum_{n=0}^{6} a_n (\ln T_e)^{6-n}\big]`
:cite:p:`Gaunt_factor_1,Gaunt_factor_2`. The atomic parameters are:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - Element
     - :math:`f_{ij}`
     - :math:`\Delta E` (eV)
   * - H
     - 0.416
     - 10.6
   * - N
     - 0.276
     - 19.8
   * - He
     - 0.26
     - 11.6
   * - Ar
     - 0.12
     - 10.3

.. rubric:: Three-Body Recombination

Derived directly from the ionization rates via microscopic reversibility (detailed balance), which guarantees automatic relaxation to the correct Saha equilibrium:

.. math::

   T_s^\mu = \left(R_{\text{DI}}^{\mu\to\mu+1}\,e^{\varepsilon_{\text{ion}}/T_{\text{eff}}} + R_{\text{EI}}^{\mu\to\mu+1}\,e^{\varepsilon_{\text{ex}}/T_{\text{eff}}}\right) \frac{g_\mu}{2\,g_{\mu+1}}\, \lambda_{\text{dB}}^{3}(T_{\text{eff}})\;\frac{\rho}{A_s m_p}

with :math:`\lambda_{\text{dB}} = \hbar\sqrt{2\pi/(m_e k_B T)}` the thermal de Broglie wavelength. The last factor converts the
rate to the mass-fraction formulation of :math:`\mathbf{R}_s`. To avoid a divergence of the recombination rate at
low temperature, the rate is evaluated at the effective temperature

.. math::

   T_{\text{eff}} = \sqrt{T_e^2 + T_C^2}, \qquad T_C = \frac{e^2}{a\,k_B}, \qquad a = \left(\frac{3}{4\pi n_e}\right)^{1/3}

where :math:`T_C` is the Coulomb temperature associated with the Wigner-Seitz radius :math:`a`. The product is
evaluated in logarithmic form to prevent overflows at low temperature.


.. rubric:: Energy Equations

Due to the large difference between electron and ion mass (:math:`m_e/m_i \sim 10^{-4}`), thermal equilibration is slow. CHEQUP utilizes a two-temperature model that separates the electron and heavy-particle temperatures by tracking the heavy-particle energy fraction :math:`f = e_h/e`.

The energy spent to ionize (or released by recombination) is taken from the total and internal energy:

.. math::

   Q_{\text{ion}} = -\sum_s \sum_\mu \frac{\Gamma_s^\mu\,\varepsilon_s^\mu}{A_s m_p}

where :math:`\varepsilon_s^\mu` is the ionization energy of stage :math:`\mu` of element :math:`s`.

.. rubric:: Collision Frequencies

Following `arXiv:2305.16779 <https://arxiv.org/abs/2305.16779>`_ (``collision_frequencies.H``):

* **Electron-ion collisions** scale with the Coulomb logarithm :math:`\Lambda_{ei}`:

  .. math::

     \nu_{ei} = \frac{4}{3}\sqrt{\frac{2\pi}{m_e}}\,\frac{n_e\,e^4\,\Lambda_{ei}}{(4\pi\varepsilon_0)^2\,(k_B T_e)^{3/2}},
     \qquad
     \Lambda_{ei} = \max\left[\ln\left(\frac{3}{2\sqrt{\pi}}\sqrt{\frac{(4\pi\varepsilon_0 k_B T_e)^3}{e^6\,n_e\,(1+T_e/T_h)}}\right),\ \tfrac{1}{2}\ln 2\right]

  The contribution of the ions of element :math:`s` is weighted by their charge,
  :math:`\nu_{ei,s} = \nu_{ei}\,\sum_{\mu} (z_s^\mu)^2 n_s^\mu / n_e`.

* **Electron-neutral collisions** rely on momentum-transfer cross sections :math:`\sigma_{es}` fitted from
  LXCat data (rational functions of :math:`\ln E`), evaluated at the mean electron energy
  :math:`\bar{E} = 4k_B T_e/\pi`:

  .. math::

     \nu_{en,s} = n_{n,s}\,\frac{4}{3}\sqrt{\frac{2\bar{E}}{M_{es}}} \,\sigma_{es}(\bar{E})

  with :math:`M_{es}` the electron-atom reduced mass.

.. note::

   In the current implementation, :math:`T_h = T_e` is used in the Coulomb logarithm.

.. rubric:: Electron Thermal Conductivity

The conductivity used in the thermal diffusion term (``actual_conductivity.H``, Eq. B1 of arXiv:2305.16779) is

.. math::

   \lambda_e = \frac{k_B^2\,n_e\,T_e}{m_e\left(\frac{2\pi}{15}\sum_s \nu_{en,s} + \nu_{ei}\,Z_{\text{eff}}/\gamma_0\right)},
   \qquad
   Z_{\text{eff}} = \frac{\sum_{s,\mu} (z_s^\mu)^2 n_s^\mu}{n_e},
   \qquad
   \gamma_0 = 12.471\,\frac{Z_{\text{eff}} + 0.112}{Z_{\text{eff}} + 3.386}

where the dependence of :math:`\gamma_0` on :math:`Z_{\text{eff}}` is fitted from Braginskii's coefficients
(:math:`\gamma_0 \approx 3.16` for :math:`Z_{\text{eff}} = 1`). The conductivity is zero where there are no free electrons.

.. rubric:: Energy Source Terms

The heavy-particle energy fraction evolves based on three source terms:

1. **Thermal Diffusion (** :math:`S_{\text{Dth}}` **)**: Electron thermal conduction only changes the electron energy, which changes the fraction :math:`f`.

   .. math::

      S_{\text{Dth}} = -\frac{f}{e}\,\nabla\cdot(\lambda_e\,\nabla T_e)

2. **Ionization Energy Exchange (** :math:`S_{\text{ion}}` **)**: The energy extracted from (or returned to) the electron pool during ionization and recombination.

   .. math::

      S_{\text{ion}} = -\frac{f}{e}\,Q_{\text{ion}}

3. **Elastic Energy Exchange (** :math:`S_{\text{coll}}` **)**: The kinetic energy transfer driven by elastic collisions between electrons and heavy particles.

   .. math::

      S_{\text{coll}} = \rho\,\nu^\varepsilon \Big[1 - \Big(1 + \frac{n_e}{n_h}\Big)f\Big],
      \qquad
      \nu^\varepsilon = \sum_s \frac{2 m_e}{m_s}\big(\nu_{ei,s} + \nu_{en,s}\big)

   This term vanishes when :math:`f = n_h/(n_h + n_e)`, i.e. when :math:`T_e = T_h`.


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
   * - :math:`E, e`
     - Total and internal specific energy
   * - :math:`\gamma`
     - Adiabatic index
   * - :math:`f`
     - Fraction of internal energy in heavy particles (:math:`e_h/e \in [0,1]`)
   * - :math:`X_s^\mu`
     - Mass fraction of species :math:`s` at ionization level :math:`\mu`
   * - :math:`\boldsymbol{X}_s`
     - Column vector of mass fractions for species :math:`s`
   * - :math:`X_e`
     - Number of free electrons per atomic mass unit, :math:`\sum z_s^\mu X_s^\mu / A_s`
   * - :math:`T_e, T_h`
     - Electron temperature / heavy-particle temperature
   * - :math:`n_e, n_h`
     - Electron and heavy-particle (ions and neutrals) number densities
   * - :math:`n_s^\mu, n_{n,s}`
     - Number density of stage :math:`\mu` and of the neutral atoms of species :math:`s`
   * - :math:`z_s^\mu`
     - Charge (number of elementary charges) of species :math:`s` at level :math:`\mu`
   * - :math:`A_s, m_s`
     - Atomic mass number and mass of species :math:`s`
   * - :math:`m_e, m_p, m_u`
     - Electron mass, proton mass and atomic mass unit
   * - :math:`k_B`
     - Boltzmann constant
   * - :math:`\mathbf{S}_{r,s}`
     - Reaction-rate matrix for species :math:`s`
   * - :math:`\mathbf{R}_s`
     - Reaction kernel
   * - :math:`\Gamma_s^\mu`
     - Net mass rate of the transition :math:`\mu \to \mu+1`
   * - :math:`I_s^\mu, T_s^\mu`
     - Ionization and three-body recombination rate coefficients
   * - :math:`R(T_e)`
     - Thermally averaged reaction rate coefficient
   * - :math:`\sigma(E)`
     - Collision cross-section as a function of energy
   * - :math:`E, \Delta E`
     - Electron kinetic energy and excitation threshold energy
   * - :math:`f_{ij}, \bar{g}`
     - Oscillator strength and thermally averaged Gaunt factor
   * - :math:`\varepsilon_{\text{ion}}, \varepsilon_{\text{ex}}`
     - Ionization and excitation threshold energies
   * - :math:`Q_{\text{ion}}`
     - Energy source due to ionization and recombination
   * - :math:`\lambda_{\text{dB}}`
     - Thermal de Broglie wavelength
   * - :math:`T_{\text{eff}}, T_C`
     - Effective temperature for three-body recombination and Coulomb temperature
   * - :math:`g_\mu`
     - Statistical weight (degeneracy) of charge state :math:`\mu`
   * - :math:`\nu_{ei}, \nu_{en,s}`
     - Electron-ion and electron-neutral collision frequencies
   * - :math:`\nu^\varepsilon`
     - Energy-transfer collision frequency
   * - :math:`\Lambda_{ei}`
     - Coulomb logarithm
   * - :math:`\lambda_e`
     - Electron thermal conductivity
   * - :math:`Z_{\text{eff}}`
     - Effective ion charge
   * - :math:`S_{\text{Dth}}, S_{\text{ion}}, S_{\text{coll}}`
     - Source terms of the heavy-particle energy fraction equation
