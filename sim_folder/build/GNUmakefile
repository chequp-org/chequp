PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE
USE_HDF5   = TRUE
USE_CUDA   = FALSE
CUDA_ARCH  = 80
USE_MHD    = FALSE

BL_NO_FORT := TRUE

# define the location of the CASTRO top directory
CASTRO_HOME  := ../../Castro

# This sets the EOS directory in $(MICROPHYSICS_HOME)/EOS
EOS_DIR     := gamma_law

# This sets the network directory in $(MICROPHYSICS_HOME)/Networks
NETWORK_DIR := general_null
NETWORK_INPUTS = species.net

# Activate thermal diffusion
USE_DIFFUSION = TRUE
CONDUCTIVITY_DIR := .

# Activate custom source terms for species
USE_SPECIES_SOURCES = TRUE

Bpack   := ./Make.package
Blocs   := .

USERSuffix = .$(EOS_DIR)

include $(CASTRO_HOME)/Exec/Make.Castro
