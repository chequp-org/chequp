============
HYQUP model
============

**HYQUP** (HYdrodynamic QUasineutral Plasma) is a computational model designed to simulate plasma behavior without explicitly solving for electromagnetic fields. Fully implemented in COMSOL Multiphysics, it supports 1D, 2D, and 3D simulations.

.. rubric:: Core Assumptions & Features

The HYQUP model relies on several key physical approximations to simplify plasma dynamics while retaining essential thermodynamic and collisional behaviors:

* **Single, Quasi-neutral Fluid:** The plasma is modeled macroscopically as a single continuous fluid, operating under the assumption of quasi-neutrality.
* **Two Temperatures (2T):** The model separates the thermodynamics of the plasma into two distinct temperatures:
  
  * :math:`T_e`: Electron temperature.
  * :math:`T_a`: Heavy particle temperature (encompassing ions, atoms, and molecules).

* **Collisional Reaction Rates:** The ionization state of the plasma is not fixed; rather, it is dynamically tracked via specific collisional reaction rates.
* **No Explicit EM Fields:** The model simplifies the computational load by removing the requirement to explicitly solve for electric (:math:`E`) or magnetic (:math:`B`) fields.


.. rubric:: Governing Equations

The mathematical formulation of the HYQUP model is divided into three primary components: Flow Dynamics, Electron Energy, and Species Continuity.

.. rubric:: Flow Dynamics


The fluid motion is governed by the conservation of mass and momentum:

**Mass Continuity:**

.. math::

    \frac{\partial \rho}{\partial t} + \vec{\nabla} \cdot (\rho \vec{v}) = 0

**Momentum:**

.. math::

    \frac{\partial \rho \vec{v}}{\partial t} + \vec{\nabla} \cdot (\rho \vec{v} \vec{v}) = -\vec{\nabla} p - \vec{\nabla} \cdot (\eta \vec{\nabla} \vec{v})


.. rubric:: Electron Energy

The energy balance for the electrons incorporates convective transport, pressure work, thermal conduction, collisional energy transfer with heavy particles, and energy lost or gained through reactions. *(Note: A similar equation applies to the heavy particles).*

.. math::

    \frac{\partial (C_e T_e)}{\partial t} + \vec{\nabla} \cdot [(C_e T_e + p_e)\vec{v}] - \vec{v} \cdot \vec{\nabla} p_e - \vec{\nabla} \cdot (\lambda_e \vec{\nabla} T_e) = -k_{e-h}(T_e - T_a) - \sum_{j,\alpha} \Delta E_j r_j


.. rubric:: Species Continuity

The number density of each individual particle species (:math:`\alpha`) is tracked over time. It is updated by the advective flow and the sum of reaction rates creating or destroying that specific species:

.. math::

    \frac{\partial n_\alpha}{\partial t} + \vec{\nabla} \cdot (n_\alpha \vec{v}) = \sum_j c_{j\alpha} r_j


.. rubric:: Nomenclature

**State Variables**

* :math:`\rho`: Mass density
* :math:`v`: Flow velocity
* :math:`p`: Total pressure
* :math:`p_e`: Electron pressure
* :math:`n`: Number density (:math:`n_\alpha` for species :math:`\alpha`)
* :math:`T`: Temperature (:math:`T_e` for electrons, :math:`T_a` for heavy particles)

**Transport Coefficients & Properties**

* :math:`C`: Heat capacity (:math:`C_e` for electrons)
* :math:`\lambda`: Thermal conductivity (:math:`\lambda_e` for electrons)
* :math:`k_{e-h}`: Collisional energy transfer coefficient
* :math:`r`: Reaction rate (:math:`r_j` for reaction :math:`j`)
* :math:`\eta`: Viscosity coefficient (representing the viscous stress tensor :math:`\tau` in the momentum equation)

**Constants & Other Terms**

* :math:`c`: Stoichiometric constant (:math:`c_{j\alpha}` for species :math:`\alpha` in reaction :math:`j`)
* :math:`\Delta E`: Reaction energy (:math:`\Delta E_j` for reaction :math:`j`)
* :math:`B`: Magnetic field
* :math:`E`: Electric field


.. rubric:: References

1. Mewes, S. M. et al. Demonstration of tunability of HOFI waveguides via start-to-end simulations. *Phys. Rev. Research* **5**, 033112 (2023).