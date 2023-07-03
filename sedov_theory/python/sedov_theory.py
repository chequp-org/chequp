"""
This file computes quantities from the Sedov theory, based on the paper
"Evaluation of the Sedov-von Neumann-Taylor Blast Wave Solution"
by James R. Kamm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def sedov_alpha(gamma, j=2):
    """
    Returns the alpha coefficient, which is used to
    evaluate the blast radius in cylindrical geometry:

    r_2 = (E0/(rho0*alpha))**(1/4) * t**(1/2)
    """
    j = 2 # cylindrical geometry
    w = 0 # uniform background plasma

    a = (j+2-w)*(gamma+1)/4
    b = (gamma+1)/(gamma-1)
    c = (j+2-w)*gamma/2
    d = (j+2-w)*(gamma+1)/( (j+2-w)*(gamma+1) - 2*(2+j*(gamma-1)) )
    e = (2+j*(gamma-1))/2

    alpha_0 = 2/(j+2-w)
    alpha_2 = - (gamma-1)/( 2*(gamma-1) + j - gamma*w )
    alpha_1 = (j+2-w)*gamma/(2+j*(gamma-1)) * ( 2*(j*(2-gamma)-w)/(gamma*(j+2-w)**2) - alpha_2 )
    alpha_3 = (j-w)/( 2*(gamma-1) + j - gamma*w )
    alpha_4 = (j+2-w)*(j-w)/( j*(2-gamma) - w ) * alpha_1
    alpha_5 = ( w*(1+gamma) - 2*j )/( j*(2-gamma) - w )

    V0 = 2/((j+2-w)*gamma)
    V2 = 4/((j+2-w)*(gamma+1))

    def j1_integrand( V ):
        return - (gamma+1)/(gamma-1) * V**2 * ( alpha_0/V + alpha_2*c/(c*V-1) - alpha_1*e/(1-e*V) ) \
                * ( (a*V)**alpha_0 * (b*(c*V-1))**alpha_2 * (d*(1-e*V))**alpha_1 )**( -(j+2-w) ) \
                * (b*(c*V-1))**alpha_3 * (d*(1-e*V))**alpha_4 * (b*(1-c*V/gamma))**alpha_5

    def j2_integrand( V ):
        return - (gamma+1)/(2*gamma) * V**2 * ((c*V-gamma)/(1-c*V)) * ( alpha_0/V + alpha_2*c/(c*V-1) - alpha_1*e/(1-e*V) ) \
                * ( (a*V)**alpha_0 * (b*(c*V-1))**alpha_2 * (d*(1-e*V))**alpha_1 )**( -(j+2-w) ) \
                * (b*(c*V-1))**alpha_3 * (d*(1-e*V))**alpha_4 * (b*(1-c*V/gamma))**alpha_5

    J1 = quad( j1_integrand, V0+1.e-10, V2)[0]
    J2 = quad( j2_integrand, V0+1.e-10, V2)[0]

    assert j==2
    I1 = np.pi*J1
    I2 = 2/(gamma-1)*np.pi*J2
    alpha = I1+I2

    return alpha