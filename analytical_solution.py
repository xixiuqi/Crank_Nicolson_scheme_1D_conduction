
from scipy.constants import foot, day
from scipy.optimize import brentq, fsolve
from scipy.optimize import root
import scipy.sparse as sps
import seaborn as sns
import numpy as np

class plane_conv_analytical:
    def __init__(self, alpha, h, k, R, Ta, Ti):
        self.R = R
        self.alpha = alpha
        self.h = h
        self.R = R
        self.Bi = h*R/k
        self.Ti = Ti
        self.Ta = Ta

    def root_function(self, xi):
        return  self.Bi - xi*np.tan(xi)
#         return (1.0- self.Bi)*np.tan(xi) - xi

    '''
    def find_nth_root(self, i):
        xi_i = brentq(self.root_function,
                      np.pi*.5+np.pi*(i)+1e-6,
                      np.pi*.5+np.pi*(i+1)-1e-6,
#                       rtol = 9*np.finfo(float).eps
                     )
        assert np.abs(self.root_function(xi_i)) < 1e-8, "Root i = "+str(i)+" was not found"
        return xi_i
    '''

    def fond_root_eigenvalue(self):
        xi_i = []

        for init in np.arange(0,10000,0.2):

          sol2 = root(self.root_function, init)

          if sol2.success == True:
            xi_i.append(sol2.x)

        xi_i = np.asarray(xi_i)
        xi_i = np.unique(xi_i.round(decimals=4))
        xi_i = xi_i[xi_i > 0]

        print("The number of eigenvalues is " + str(len(xi_i)))

        return xi_i

    def constant(self,xi):
        return 4.0*np.sin(xi)/(2.0*xi + np.sin(2*xi))

    def theta(self, r_star, Fo):
        theta = 0
        xi_i = self.fond_root_eigenvalue()
        for xi_ in xi_i:
            constant_ = self.constant(xi_)
            theta += constant_*np.exp(-Fo*xi_**2)*np.cos(r_star*xi_)
        return theta

    def temperature(self, r, t):
        Fo = self.alpha*t/self.R**2
        r_star = r/self.R
        return self.Ta + (self.Ti - self.Ta)*self.theta(r_star, Fo)

    def _est_t_cooling(self,Tf):
#         return (self.R**2/(self.alpha*self.constant(self.find_nth_root(0))))*np.log((self.Ti - self.Ta)/(Tf-self.Ta))
        _xi = self.find_nth_root(0)
        _constant = self.constant(_xi)
        return (self.R**2/(self.alpha*_xi**2))*np.log(_constant*(self.Ti - self.Ta)/(Tf - self.Ta))

    def _temp_diff(self, t, Tf):
        return self.temperature(0,t) - Tf

    def t_cooling(self, Tf):
        t_est = self._est_t_cooling(Tf)
#         t_est = 30*day
        if self._temp_diff(t_est, Tf) < .1:
            return t_est
        else:
            t_ = fsolve(self._temp_diff, t_est+10*day, args = (Tf),
                        xtol = 1e-16
                       )[0]
            assert self._temp_diff(t_, Tf) < .0001, "Temperature diff is too large"
            return t_
