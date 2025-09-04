#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <prob_parameters.H>
#include <eos.H>
#include <AMReX_Math.H>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k, Array4<Real> const& state, const GeometryData& geomdata)
{
    const Real* dx = geomdata.CellSize();
    const Real* problo = geomdata.ProbLo();

    Real x = problo[0] + dx[0] * (static_cast<Real>(i) + 0.5);
#if AMREX_SPACEDIM >= 2
    Real y = problo[1] + dx[1] * (static_cast<Real>(j) + 0.5);
#else
    Real y = 0.0;
#endif
#if AMREX_SPACEDIM >= 3
    Real z = problo[2] + dx[2] * (static_cast<Real>(k) + 0.5);
#else
    Real z = 0.0;
#endif

    // We assume the laser is propagating along z, so we use the transverse radius.
    Real r = std::sqrt(x*x + y*y);

    // =========================================================================
    // Read the entire data table from multi-species CSV file.
    // =========================================================================
    std::vector<std::vector<Real>> data_table;
    std::ifstream file("plasma_data_multi_species.csv"); 

    if (file.is_open()) {
        std::string line;
        // Skip the header line
        std::getline(file, line);

        // Read all numerical data into a 2D vector
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<Real> row;
            while(std::getline(ss, value, ',')) {
                try {
                    row.push_back(std::stod(value));
                } catch (const std::invalid_argument&) {
                    amrex::Abort("Invalid floating point value in CSV: " + value);
                }
            }
            data_table.push_back(row);
        }
        file.close();
    } else {
        amrex::Abort("Failed to open plasma_data_multi_species.csv");
    }

    // =========================================================================
    // Interpolate all columns (Temp and all species number fractions).
    // =========================================================================
    int num_cols = data_table[0].size();
    std::vector<Real> interp_values(num_cols);

    // Get the radius column from the data table for easier access
    std::vector<Real> r_coords;
    for(const auto& row : data_table){
        r_coords.push_back(row[0]);
    }

    if (r <= r_coords.front()) {
        interp_values = data_table.front();
    } else if (r >= r_coords.back()) {
        interp_values = data_table.back();
    } else {
        for (int idx = 1; idx < r_coords.size(); ++idx) {
            if (r <= r_coords[idx]) {
                Real r_low = r_coords[idx-1];
                Real r_high = r_coords[idx];
                Real frac = (r - r_low) / (r_high - r_low);
                // Interpolate every column in the table
                for (int col = 0; col < num_cols; ++col) {
                    interp_values[col] = data_table[idx-1][col] + frac * (data_table[idx][col] - data_table[idx-1][col]);
                }
                break;
            }
        }
    }

    // Extract temperature and the number fractions for clarity
    // Note: The first column (index 0) is radius.
    Real te_kelvin = interp_values[1];
    // The species number fractions start from the second column (index 2).
    // We have 7 species defined: Hn, Hp, Nn, N1, N2, N3, N4
    const int num_species = 7;
    std::vector<Real> number_fractions(num_species);
    for (int s = 0; s < num_species; ++s) {
        // CSV has H_0, H_1, N_0, N_1, N_2, N_3, N_4, N_5
        // We map these to our 7 species.
        number_fractions[s] = interp_values[s + 2];
    }

    // =========================================================================
    // Convert from Number Fraction to Mass Fraction
    // =========================================================================
    Real mass_H = 1.0;
    Real mass_N = 14.0;
    std::vector<Real> mass_fractions(num_species);

    // First, calculate the average atomic mass based on the number fractions
    Real avg_mass = 0.0;
    avg_mass += (number_fractions[0] + number_fractions[1]) * mass_H; // H species
    for (int s = 2; s < num_species; ++s) {
        avg_mass += number_fractions[s] * mass_N; // N species
    }

    // Now, calculate the mass fraction for each species
    if (avg_mass > 1.e-12) {
        mass_fractions[0] = (number_fractions[0] * mass_H) / avg_mass; // H_neutral
        mass_fractions[1] = (number_fractions[1] * mass_H) / avg_mass; // H_ion
        for (int s = 2; s < num_species; ++s) {
            mass_fractions[s] = (number_fractions[s] * mass_N) / avg_mass; // N ions
        }
    } else {
        // Avoid division by zero, though this case should not be reached
        for (int s = 0; s < num_species; ++s) mass_fractions[s] = 0.0;
    }

    // =========================================================================
    // Set the state variables using the calculated mass fractions.
    // =========================================================================

    // Total density is ambient
    state(i,j,k,URHO) = problem::dens_ambient;

    // No initial velocity
    state(i,j,k,UMX) = 0.0;
    state(i,j,k,UMY) = 0.0;
    state(i,j,k,UMZ) = 0.0;

    // Set the species mass fractions
    // The order MUST match species.net file
    // UFS corresponds to the first species (H_neutral)
    // UFS+1 corresponds to the second species (H_ion)
    // UFS+2 corresponds to the third species (N_neutral), etc.
    for (int s = 0; s < num_species; ++s) {
        state(i,j,k,UFS+s) = state(i,j,k,URHO) * mass_fractions[s];
    }

    // Set thermodynamic variables
    eos_t eos_state;
    eos_state.rho = state(i,j,k,URHO);
    eos_state.T = te_kelvin; // Use the interpolated temperature
    for (int s = 0; s < num_species; ++s) {
        eos_state.xn[s] = mass_fractions[s];
    }

    eos(eos_input_rt, eos_state);

    state(i,j,k,UTEMP) = eos_state.T;
    state(i,j,k,UEINT) = eos_state.e;
    state(i,j,k,UEDEN) = state(i,j,k,URHO) * eos_state.e; // Internal energy only, no kinetic
}

#endif
