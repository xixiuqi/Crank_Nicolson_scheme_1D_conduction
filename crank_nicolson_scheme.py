import numpy as np
import math
from numpy import linalg as LA

def simulateT(spatialNumer = 100, timeNumber = 1000, h_0 = 15):
  Length = 0.5
  J = spatialNumer
  dx = float(Length)/float(J-1)
  x_grid = np.array([j*dx for j in range(J)])

  Time = 20
  N = timeNumber
  dt = float(Time)/float(N-1)
  t_grid = np.array([n*dt for n in range(N)])

  alpha = 2E-5
  k = 0.2
  h_L = 0
  T_infity = 593.15

  sigma_T = float(alpha*dt)/float((2.*dx*dx))

  T_init = 293.15
  T =  np.array([T_init for i in range(J)])

  x_0_con = sigma_T*h_0*dx/k
  x_L_con = sigma_T*h_L*dx/k

  A_T = np.diagflat([-sigma_T for i in range(J-1)], -1) +\
        np.diagflat([1.+sigma_T+x_0_con]+[1.+2.*sigma_T for i in range(J-2)]+[1.+sigma_T-x_L_con]) +\
        np.diagflat([-sigma_T for i in range(J-1)], 1)

  B_T = np.diagflat([sigma_T for i in range(J-1)], -1) +\
        np.diagflat([1.-sigma_T-x_0_con]+[1.-2.*sigma_T for i in range(J-2)]+[1.-sigma_T+x_L_con]) +\
        np.diagflat([sigma_T for i in range(J-1)], 1)

  f_T = np.array([2*x_0_con*T_infity] + [0]*(J-2) + [-2*x_L_con*T_infity])

  T_record = []

  T_record.append(T)

  for ti in range(1,N):
      T_new = np.linalg.solve(A_T, B_T.dot(T) + f_T)

      T = T_new

      T_record.append(T)

  return T_record
