"""
This file computes quantities from the Sedov theory, based on the paper
"Evaluation of the Sedov-von Neumann-Taylor Blast Wave Solution"
by James R. Kamm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
import tqdm

w = 0 # uniform background plasma
j = 2 # Cylindrical geometry

class SedovTalorProblem(object):

    def __init__(self, gamma, E0, rho0):
        self.gamma = gamma
        self.E0 = E0
        self.rho0 = rho0

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

        # Store coefficients that are useful in the Sedov-Taylor problem
        self.alpha = I1+I2
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.alpha_4 = alpha_4
        self.alpha_5 = alpha_5
        self.V0 = V0
        self.V2 = V2

    def blast_radius( self, t ):
        return (self.E0/(self.alpha*self.rho0))**(1./(j+2-w)) * t**(2./(j+2-w))

    def evaluate( self, quantity, r, t ):
        """
        quantity: string
            Either 'density' or 'pressure'

        r: 1darray
            Radii at which to evaluate the solution

        t: float
            Time at which to evaluate the solution
        """

        # First: find blast radius
        r2 = self.blast_radius( t )

        # Fill array with pre-shock values
        if quantity == 'density':
            q = self.rho0 * np.ones_like(r)
            q2 = (self.gamma+1)/(self.gamma-1) * self.rho0
        elif quantity == 'pressure':
            q = np.zeros_like(r)
            U = 2./(j + 2 - w) * (r2/t)
            q2 = 2./(self.gamma+1) * self.rho0 * U**2
        else:
            raise RuntimeError('Unknown quantity: %s' %quantity)

        # Fill post-shock values
        for i in tqdm.tqdm(range(len(r))):

            # Skip if this is post-shock
            if r[i] > r2:
                continue

            # Find value of V that corresponds to this value of r
            sol = root_scalar( lambda V:
                            (self.a*V) ** (-self.alpha_0) * \
                            (self.b*(self.c*V-1)) ** (-self.alpha_2) * \
                            (self.d*(1-self.e*V)) ** (-self.alpha_1) - r[i]/r2,
                            method='bisect',
                            bracket=[self.V0, self.V2] )
            V = sol.root

            # Evaluate Sedov solution at this point
            if quantity == 'density':
                q[i] = q2 * (self.a*V) ** (self.alpha_0*w) * \
                            (self.b*(self.c*V-1)) ** (self.alpha_3 + self.alpha_2*w) * \
                            (self.d*(1-self.e*V)) ** (self.alpha_4 + self.alpha_1*w) * \
                            (self.b*(1-self.c*V/self.gamma)) ** self.alpha_5
            elif quantity == 'pressure':
                q[i] = q2 * (self.a*V) ** (self.alpha_0*j) * \
                            (self.d*(1-self.e*V)) ** (self.alpha_4 + self.alpha_1*(w-2)) * \
                            (self.b*(1-self.c*V/self.gamma)) ** (1+self.alpha_5)

        return q
