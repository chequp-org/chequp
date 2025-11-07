#!/usr/bin/env python3
"""
Convert Castro plotfile to OpenPMD format for WarpX particle initialization.
Calculates electron density using charge-weighted sum of all ionized species.

Usage:
    python convert_plotfile_to_openpmd.py --plotfile plt00300/
    python convert_plotfile_to_openpmd.py --plotfile plt00300/ --all-species
    python convert_plotfile_to_openpmd.py --plotfile plt00300/ --species heavy --level 3
    python convert_plotfile_to_openpmd.py --plotfile plt00300/ --all-species --verbose
"""
import os
import argparse
import numpy as np
import openpmd_api as io
import yt

yt.set_log_level("error")

# Atomic masses in grams
amu_to_g = 1.66053906660e-24
mass_h_g = 1.008 * amu_to_g
mass_n_g = 14.007 * amu_to_g


def extract_data_at_level(ds, quantity, level):
    """
    Extract quantity at specified refinement level for 2D RZ geometry.
    Uses covering_grid to get uniform resolution data.
    """
    if ds.dimensionality != 2:
        raise ValueError(f"This script requires 2D RZ data, got {ds.dimensionality}D")
    
    # Use covering_grid to get data at specified level with uniform resolution
    ad = ds.covering_grid(
        level=level,
        left_edge=ds.domain_left_edge,
        dims=[ds.domain_dimensions[0] * 2**level, 
              ds.domain_dimensions[1] * 2**level, 1]
    )
    
    q_full = ad[quantity].to_ndarray().squeeze()
    
    # Grid dimensions
    nr = ds.domain_dimensions[0] * 2**level
    nz = ds.domain_dimensions[1] * 2**level
    
    # Create 1D coordinate arrays
    r_1d = np.linspace(ds.domain_left_edge[0], ds.domain_right_edge[0], nr)
    z_1d = np.linspace(ds.domain_left_edge[1], ds.domain_right_edge[1], nz)
    
    # Transpose to (nz, nr) format
    q_transposed = q_full.T
    
    return r_1d.to_ndarray(), z_1d.to_ndarray(), q_transposed


def calculate_electron_density(ds, level, verbose=True):
    """
    Calculate electron density as charge-weighted sum of all ionized species.
    
    Returns:
        r_1d: 1D array of radial coordinates (CGS)
        z_1d: 1D array of axial coordinates (CGS)
        n_electrons: 2D array of electron density (cm^-3), shape (nz, nr)
        species_densities: dict of individual species densities
    """
    
    # Define all species with their masses and charges
    species_info = [
        {'name': 'H0', 'mass': mass_h_g, 'charge': 0},
        {'name': 'H1', 'mass': mass_h_g, 'charge': 1},
        {'name': 'N0', 'mass': mass_n_g, 'charge': 0},
        {'name': 'N1', 'mass': mass_n_g, 'charge': 1},
        {'name': 'N2', 'mass': mass_n_g, 'charge': 2},
        {'name': 'N3', 'mass': mass_n_g, 'charge': 3},
        {'name': 'N4', 'mass': mass_n_g, 'charge': 4},
        {'name': 'N5', 'mass': mass_n_g, 'charge': 5},
    ]
    
    if verbose:
        print(f"\nExtracting species densities at refinement level {level}...")
    
    r_1d = None
    z_1d = None
    n_electrons = None
    species_densities = {}
    
    for spec in species_info:
        field_name = ('boxlib', f"rho_{spec['name']}")
        
        try:
            r_1d_temp, z_1d_temp, mass_density = extract_data_at_level(
                ds, field_name, level
            )
            
            # Initialize arrays on first successful extraction
            if r_1d is None:
                r_1d = r_1d_temp
                z_1d = z_1d_temp
                n_electrons = np.zeros_like(mass_density)
                if verbose:
                    print(f"  Initialized grid: {mass_density.shape} (nz={len(z_1d)}, nr={len(r_1d)})")
            
            # Convert mass density to number density
            n_species = mass_density / spec['mass']
            species_densities[spec['name']] = n_species
            
            # Add contribution to electron density (charge-weighted)
            if spec['charge'] > 0:
                contribution = spec['charge'] * n_species
                n_electrons += contribution
                if verbose:
                    print(f"  {spec['name']} (charge={spec['charge']}): "
                          f"{n_species.min():.2e} to {n_species.max():.2e} cm^-3, "
                          f"contributes {contribution.min():.2e} to {contribution.max():.2e} e^-/cm^3")
            else:
                if verbose:
                    print(f"  {spec['name']} (neutral): "
                          f"{n_species.min():.2e} to {n_species.max():.2e} cm^-3")
                
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not process {spec['name']}: {e}")
    
    return r_1d, z_1d, n_electrons, species_densities


def save_density_to_openpmd(r_1d_m, z_1d_m, density_m3, output_filename, verbose=True):
    """
    Save density to OpenPMD file in WarpX-compatible format.
    
    Parameters:
        r_1d_m: 1D array of radial coordinates (m)
        z_1d_m: 1D array of axial coordinates (m)
        density_m3: 2D density array (m^-3), shape (nz, nr)
        output_filename: Output file name
    """
    
    # Transpose from (nz, nr) to (nr, nz) for WarpX
    # Then reshape to (1, nr, nz) as required for RZ geometry
    density_warpx = density_m3.T
    density_warpx = density_warpx.reshape(1, *density_warpx.shape)
    
    if verbose:
        print(f"\n  Data shape: {density_warpx.shape} (1, nr={len(r_1d_m)}, nz={len(z_1d_m)})")
    
    # Grid parameters
    dr = r_1d_m[1] - r_1d_m[0]
    dz = z_1d_m[1] - z_1d_m[0]
    grid_offset = [r_1d_m.min(), z_1d_m.min()]
    
    if verbose:
        print(f"  Grid spacing: dr={dr:.3e} m, dz={dz:.3e} m")
        print(f"  Grid offset: r_min={grid_offset[0]:.3e} m, z_min={grid_offset[1]:.3e} m")
    
    # Create OpenPMD file
    series = io.Series(output_filename, io.Access.create)
    it = series.iterations[1]
    
    # Set mesh metadata
    density = it.meshes["density"]
    density.grid_spacing = np.array([dr, dz])
    density.grid_global_offset = grid_offset
    density.axis_labels = ["r", "z"]
    density.geometry = io.Geometry.thetaMode
    density.geometry_parameters = "m=0;imag=+"
    density.unit_dimension = {io.Unit_Dimension.L: -3}
    
    # Store density data
    density_component = density[io.Mesh_Record_Component.SCALAR]
    density_component.position = [0.0, 0.0]
    
    dataset = io.Dataset(density_warpx.dtype, density_warpx.shape)
    density_component.reset_dataset(dataset)
    density_component.store_chunk(density_warpx)
    
    series.flush()
    del series


def main():
    parser = argparse.ArgumentParser(
        description='Convert Castro plotfile to OpenPMD format for WarpX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_plotfile_to_openpmd.py --plotfile plt00300/
  python convert_plotfile_to_openpmd.py --plotfile plt00300/ --all-species
  python convert_plotfile_to_openpmd.py --plotfile plt00300/ --output my_density.h5
  python convert_plotfile_to_openpmd.py --plotfile plt00300/ --species heavy --level 3
  python convert_plotfile_to_openpmd.py --plotfile plt00300/ --all-species --verbose
        """
    )
    
    parser.add_argument(
        '--plotfile',
        type=str,
        required=True,
        help='Path to Castro plotfile (e.g., plt00300/)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='openpmd_density.h5',
        help='Output filename (default: openpmd_density.h5)'
    )
    
    parser.add_argument(
        '--level',
        type=int,
        default=None,
        help='Refinement level to extract (default: use highest available level)'
    )
    
    parser.add_argument(
        '--species',
        type=str,
        choices=['electron', 'H0', 'H1', 'heavy'],
        default='electron',
        help='Species to output: electron (default), H0 (neutral H), H1 (ionized H), or heavy (H0+H1)'
    )
    
    parser.add_argument(
        '--all-species',
        action='store_true',
        help='Save all available species (electron, H0, H1, heavy) - overrides --species flag'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output during processing'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("="*70)
        print("Convert Castro Plotfile to OpenPMD for WarpX")
        print("="*70)
    
    # Load the dataset
    if args.verbose:
        print(f"\nLoading plotfile: {args.plotfile}")
    ds = yt.load(args.plotfile, hint="castro")
    
    # Determine refinement level
    if args.level is None:
        level = ds.index.max_level
        if args.verbose:
            print(f"Using highest available refinement level: {level}")
    else:
        level = min(args.level, ds.index.max_level)
        if args.verbose:
            if args.level > ds.index.max_level:
                print(f"Warning: Requested level {args.level} > max level {ds.index.max_level}")
                print(f"Using max level {ds.index.max_level} instead")
            else:
                print(f"Using refinement level: {level}")
    
    if args.verbose:
        print(f"Simulation time: {ds.current_time.to_value('ns'):.3f} ns")
        print(f"Domain: r=[{ds.domain_left_edge[0]:.3e}, {ds.domain_right_edge[0]:.3e}] cm")
        print(f"        z=[{ds.domain_left_edge[1]:.3e}, {ds.domain_right_edge[1]:.3e}] cm")
    
    # Calculate electron density and all species
    r_1d_cm, z_1d_cm, n_electrons_cm3, species_densities = calculate_electron_density(
        ds, level, verbose=args.verbose
    )
    
    # Convert to SI units
    r_1d_m = r_1d_cm * 0.01  # cm to m
    z_1d_m = z_1d_cm * 0.01  # cm to m
    
    # Build dictionary of all available densities in SI units
    density_dict = {}
    density_dict['electron'] = n_electrons_cm3 * 1e6
    
    if 'H0' in species_densities:
        density_dict['H0'] = species_densities['H0'] * 1e6
    
    if 'H1' in species_densities:
        density_dict['H1'] = species_densities['H1'] * 1e6
    
    if 'H0' in species_densities and 'H1' in species_densities:
        density_dict['heavy'] = (species_densities['H0'] + species_densities['H1']) * 1e6
    
    # Print summary
    if args.verbose:
        print("\n" + "="*70)
        print("Electron Density Summary")
        print("="*70)
        print(f"\nTotal electron density:")
        print(f"  Range: {n_electrons_cm3.min():.2e} to {n_electrons_cm3.max():.2e} cm^-3")
        if n_electrons_cm3.max() > 0:
            print(f"  Mean (non-zero): {n_electrons_cm3[n_electrons_cm3 > 0].mean():.2e} cm^-3")
        
        # Calculate contribution breakdown
        print(f"\nElectron contributions by species:")
        for spec_name in ['H1', 'N1', 'N2', 'N3', 'N4', 'N5']:
            if spec_name in species_densities:
                n_spec = species_densities[spec_name]
                charge = int(spec_name[1]) if spec_name[0] == 'N' else 1
                contribution = charge * n_spec
                if n_electrons_cm3.sum() > 0:
                    fraction = contribution.sum() / n_electrons_cm3.sum() * 100
                    print(f"  {spec_name} (×{charge}): {fraction:.1f}% of total electrons")
        
        # Quasi-neutrality check
        if 'H1' in species_densities:
            n_H1 = species_densities['H1']
            mask = (n_electrons_cm3 > 1e12) & (n_H1 > 1e12)
            if mask.any():
                ratio = n_electrons_cm3[mask].mean() / n_H1[mask].mean()
                print(f"\nQuasi-neutrality check (ionized regions, n > 1e12 cm^-3):")
                print(f"  n_e / n_H1 = {ratio:.3f}")
    
    # Determine which species to save
    if args.all_species:
        species_to_save = list(density_dict.keys())
        if args.verbose:
            print(f"\nSaving all available species: {species_to_save}")
    else:
        if args.species not in density_dict:
            print(f"\n✗ Species '{args.species}' not available in this plotfile")
            print(f"Available species: {list(density_dict.keys())}")
            return 1
        species_to_save = [args.species]
    
    # Save the requested species
    saved_files = []
    for species_name in species_to_save:
        density_m3 = density_dict[species_name]
        
        if args.all_species:
            # Auto-generate filenames for all-species mode
            base, ext = os.path.splitext(args.output)
            if base == 'openpmd_density':  # default name
                output_file = f'density_{species_name}.h5'
            else:
                output_file = f'{base}_{species_name}{ext}'
        else:
            output_file = args.output
        
        if args.verbose:
            print(f"\nSaving {species_name} to: {output_file}")
            print(f"  Density range: {density_m3.min():.2e} to {density_m3.max():.2e} m^-3")
        
        save_density_to_openpmd(r_1d_m, z_1d_m, density_m3, output_file, verbose=args.verbose)
        saved_files.append((species_name, output_file))
    


if __name__ == '__main__':
    exit(main())
