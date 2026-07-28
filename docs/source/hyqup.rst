HYQYUP model
------------

**HYQUP** (HYdrodynamic QUasineutral Plasma) is a computational model designed to simulate plasma behavior without explicitly solving for electromagnetic fields. Fully implemented in COMSOL Multiphysics, it supports 1D, 2D, and 3D simulations.

---

### Core Assumptions & Features

The HYQUP model relies on several key physical approximations to simplify plasma dynamics while retaining essential thermodynamic and collisional behaviors:

*   **Single, Quasi-neutral Fluid:** The plasma is modeled macroscopically as a single continuous fluid, operating under the assumption of quasi-neutrality.
*   **Two Temperatures (2T):** The model separates the thermodynamics of the plasma into two distinct temperatures:
    *   $T_e$: Electron temperature.
    *   $T_a$: Heavy particle temperature (encompassing ions, atoms, and molecules).
*   **Collisional Reaction Rates:** The ionization state of the plasma is not fixed; rather, it is dynamically tracked via specific collisional reaction rates.
*   **No Explicit EM Fields:** The model simplifies the computational load by removing the requirement to explicitly solve for electric ($E$) or magnetic ($B$) fields.

---

## Governing Equations

The mathematical formulation of the HYQUP model is divided into three primary components: Flow Dynamics, Electron Energy, and Species Continuity.

### Flow Dynamics

The fluid motion is governed by the conservation of mass and momentum:

**Mass Continuity:**
$$\frac{\partial \rho}{\partial t} + \vec{\nabla} \cdot (\rho \vec{v}) = 0$$

**Momentum:**
$$\frac{\partial \rho \vec{v}}{\partial t} + \vec{\nabla} \cdot (\rho \vec{v} \vec{v}) = -\vec{\nabla} p - \vec{\nabla} \cdot (\eta \vec{\nabla} \vec{v})$$

### Electron Energy

The energy balance for the electrons incorporates convective transport, pressure work, thermal conduction, collisional energy transfer with heavy particles, and energy lost or gained through reactions. *(Note: A similar equation applies to the heavy particles).*

$$\frac{\partial (C_e T_e)}{\partial t} + \vec{\nabla} \cdot [(C_e T_e + p_e)\vec{v}] - \vec{v} \cdot \vec{\nabla} p_e - \vec{\nabla} \cdot (\lambda_e \vec{\nabla} T_e) = -k_{e-h}(T_e - T_a) - \sum_{j,\alpha} \Delta E_j r_j$$

### Species Continuity

The number density of each individual particle species ($\alpha$) is tracked over time. It is updated by the advective flow and the sum of reaction rates creating or destroying that specific species:

$$\frac{\partial n_\alpha}{\partial t} + \vec{\nabla} \cdot (n_\alpha \vec{v}) = \sum_j c_{j\alpha} r_j$$

---

## Nomenclature

**State Variables**
*   $\rho$: Mass density
*   $v$: Flow velocity
*   $p$: Total pressure
*   $p_e$: Electron pressure
*   $n$: Number density ($n_\alpha$ for species $\alpha$)
*   $T$: Temperature ($T_e$ for electrons, $T_a$ for heavy particles)

**Transport Coefficients & Properties**
*   $C$: Heat capacity ($C_e$ for electrons)
*   $\lambda$: Thermal conductivity ($\lambda_e$ for electrons)
*   $k_{e-h}$: Collisional energy transfer coefficient
*   $r$: Reaction rate ($r_j$ for reaction $j$)
*   $\eta$: Viscosity coefficient (representing the viscous stress tensor $\tau$ in the momentum equation)

**Constants & Other Terms**
*   $c$: Stoichiometric constant ($c_{j\alpha}$ for species $\alpha$ in reaction $j$)
*   $\Delta E$: Reaction energy ($\Delta E_j$ for reaction $j$)
*   $B$: Magnetic field
*   $E$: Electric field

---

### References

1. Mewes, S. M. et al. Demonstration of tunability of HOFI waveguides via start-to-end simulations. *Phys. Rev. Research* **5**, 033112 (2023).