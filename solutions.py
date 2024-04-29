import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import linalg as LA
from scipy.constants import foot, day
from scipy.optimize import brentq, fsolve
from scipy.optimize import root
import scipy.sparse as sps
import seaborn as sns

from crank_nicolson_scheme import simulateT
from analytical_solution import plane_conv_analytical

convective_coeff = 15
  
sol = plane_conv_analytical(2E-5, convective_coeff, 0.2, 0.5, 593.15, 293.15)
analytical_surface_T = sol.temperature(0.5, np.linspace(0,20,1000))

different_grid_size = []

grid_matrix = np.geomspace(10,500,10).round().astype(int)

for gridsize in grid_matrix:

  T_record = simulateT(gridsize,1000,15)

  simulated_surface_T = []

  for T in T_record:
    simulated_surface_T.append(T[0])

  different_grid_size.append(simulated_surface_T)

plt.xlabel('Mesh Biot number')
plt.ylabel('Norm distance')
index = 0
error = []
for simulated_surface_T in different_grid_size:
  error.append(LA.norm(simulated_surface_T-analytical_surface_T, np.inf))
  index  = index+1

plt.plot(0.5/grid_matrix*convective_coeff/0.2, error, marker = '.', markersize = 10)

plt.legend()
plt.grid(True)
plt.xscale("log")
plt.yscale("log")

plt.show()
