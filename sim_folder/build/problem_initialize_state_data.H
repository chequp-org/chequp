#ifndef problem_initialize_state_data_H
#define problem_initialize_state_data_H

#include <prob_parameters.H>
#include <eos.H>
#include <AMReX_Math.H>
#include <AMReX_Print.H>
#include <fundamental_constants.H>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

// Static plasma data arrays - initialized once in problem_initialize
static std::vector<Real> plasma_data_r;
static std::vector<Real> plasma_data_ioniz;
static std::vector<Real> plasma_data_te;

// Function to initialize plasma data from CSV file
inline void initialize_plasma_data()
{
    std::ifstream file("plasma_data.csv");
    if (file.is_open()) {
        std::string line;

        // Skip the header line
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;

            // Read radius
            if (std::getline(ss, value, ',')) {
                try {
                    plasma_data_r.push_back(std::stod(value));
                } catch (const std::invalid_argument&) {
                    amrex::Abort("Invalid radius value: " + value);
                }
            } else {
                amrex::Abort("Missing radius value");
            }

            // Read ionization fraction
            if (std::getline(ss, value, ',')) {
                try {
                    plasma_data_ioniz.push_back(std::stod(value));
                } catch (const std::invalid_argument&) {
                    amrex::Abort("Invalid ionization fraction value: " + value);
                }
            } else {
                amrex::Abort("Missing ionization fraction value");
            }

            // Read electron temperature
            if (std::getline(ss, value)) {
                try {
                    plasma_data_te.push_back(std::stod(value));
                } catch (const std::invalid_argument&) {
                    amrex::Abort("Invalid electron temperature value: " + value);
                }
            } else {
                amrex::Abort("Missing electron temperature value");
            }
        }
        file.close();
    } else {
        amrex::Abort("Failed to open plasma_data.csv");
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void problem_initialize_state_data (int i, int j, int k, Array4<Real> const& state, const GeometryData& geomdata)
{
    int coord_type = geomdata.Coord();

    const Real* dx = geomdata.CellSize();
    const Real* problo = geomdata.ProbLo();

    Real x = 0;
    Real y = 0;
    Real z = 0;

    x = problo[0] + dx[0] * (static_cast<Real>(i)+0.5);
#if AMREX_SPACEDIM >= 2
    y = problo[1] + dx[1] * (static_cast<Real>(j)+0.5);
#endif
#if AMREX_SPACEDIM >= 3
    z = problo[2] + dx[2] * (static_cast<Real>(k)+0.5);
#endif

    Real r = sqrt(x*x + y*y + z*z);

    // Linear interpolation to find ionization fraction and electron temperature
    Real ioniz_fraction = 0.0;
    Real te = 0.0;

    // Check if plasma data has been loaded
    if (plasma_data_r.empty()) {
        amrex::Abort("Plasma data not initialized. Make sure problem_initialize() is called first.");
    }

    if (r <= plasma_data_r.front()) {
        ioniz_fraction = plasma_data_ioniz.front();
        te = plasma_data_te.front();
    } else if (r >= plasma_data_r.back()) {
        ioniz_fraction = plasma_data_ioniz.back();
        te = plasma_data_te.back();
    } else {
        for (int idx = 1; idx < plasma_data_r.size(); ++idx) {
            if (r < plasma_data_r[idx]) {
                Real r_low = plasma_data_r[idx-1];
                Real r_high = plasma_data_r[idx];
                Real frac = (r - r_low) / (r_high - r_low);

                ioniz_fraction = plasma_data_ioniz[idx-1] + frac * (plasma_data_ioniz[idx] - plasma_data_ioniz[idx-1]);
                te = plasma_data_te[idx-1] + frac * (plasma_data_te[idx] - plasma_data_te[idx-1]);
                break;
            }
        }
    }
    Real eint = problem::dens_ambient * (ioniz_fraction * 3./2 * C::k_B * te +  3./2 * C::k_B * problem::temp_ambient)/C::m_u;

    state(i,j,k,URHO) = problem::dens_ambient;
    state(i,j,k,UMX) = 0.e0_rt;
    state(i,j,k,UMY) = 0.e0_rt;
    state(i,j,k,UMZ) = 0.e0_rt;

    state(i,j,k,UEDEN) = eint +
                         0.5e0_rt * (state(i,j,k,UMX) * state(i,j,k,UMX) / state(i,j,k,URHO) +
                                     state(i,j,k,UMY) * state(i,j,k,UMY) / state(i,j,k,URHO) +
                                     state(i,j,k,UMZ) * state(i,j,k,UMZ) / state(i,j,k,URHO));

    state(i,j,k,UEINT) = eint;
    state(i,j,k,UFX) = eint; // internal energy of heavy ions

    // The temperature initialization will be done later
    // in the call to clean_state. We want to avoid
    // EOS calls in ca_initdata. So that there is valid
    // data for an initial guess for the EOS, we initialize
    // to the ambient temperature everywhere.

    state(i,j,k,UTEMP) = problem::temp_ambient;

    state(i,j,k,UFS) = state(i,j,k,URHO)*(1-ioniz_fraction);
    state(i,j,k,UFS+1) = state(i,j,k,URHO)*ioniz_fraction;
}

#endif
