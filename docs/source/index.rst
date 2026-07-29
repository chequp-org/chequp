CHEQUP |release| Documentation
==============================

CHEQUP (Castro-based Hofi Expansion with QUasineutral Plasma) is a module built on the Castro code https://github.com/AMReX-Astro/Castro for high-performance physics simulations, and using the HYQUP model developed for simulating quasineutral plasmas. It is designed to handle complex plasma dynamics, including ionization, recombination, and energy transfer processes, in a computationally efficient manner.
.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card:: Getting Started
        :text-align: center

        Check this out for installation instructions, tutorials, and testing.

        +++

        .. button-ref:: installation
            :expand:
            :color: primary
            :click-parent:

            Getting Started

    .. grid-item-card:: Physics
        :text-align: center

        Understand the physical models, matrix formulations, and theoretical background.

        +++

        .. button-ref:: hyqup
            :expand:
            :color: primary
            :click-parent:

            Explore the Physics

    .. grid-item-card:: User guide
        :text-align: center

        Detailed guides on using the Python analysis tools and running HiPACE++ workflows.

        +++

        .. button-ref:: analysis_tool_python
            :expand:
            :color: primary
            :click-parent:

            Analysis tools

    .. grid-item-card:: Project Info
        :text-align: center

        Information about the authors, bibliography, and licensing.

        +++

        .. button-ref:: authors
            :expand:
            :color: primary
            :click-parent:

            Project Information

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   installation
   run
   analyze_basic
   example_tutorial
   test_suite

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Physics
   
   hyqup
   matrix_form

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User guide

   analysis_tool_python
   hipace_to_chequp

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project Info

   authors
   license
   bibliography