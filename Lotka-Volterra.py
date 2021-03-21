# Lotka-Volterra equations: predator-prey model #

# Isolated ecosystem: no migration, no other species present, no pests...
# The prey population in the absence of predators grows exponentially: the rate of reproduction is proportional to the number of individuals. 
#   Prey only die when they are killed by the predator
# The predator population in the absence of prey decreases exponentially.
# The predator population affects the prey population by decreasing it proportionally to the number of prey and predators.
# Prey population affects predator population also proportionally to the number of encounters, 
#   but with different proportionality constant

# Lions and zebras

# a system of two first-order, coupled, autonomous, non-linear, coupled first-order differential equations
# dx/dt = ax - bxy
# dy/dt = - cy + dyx

# Parameters
# x = nº prey (zebras)
# y = nº predator (lions)
# a = growth rate of prey
# b = success in predator hunting
# c = rate of decrease of predators
# d = hunting success and how much a predator is fed by hunting prey

# Info: https://pybonacci.org/2015/01/05/ecuaciones-de-lotka-volterra-modelo-presa-depredador/#:~:text=Las%20ecuaciones%20de%20Lotka%2DVolterra,bajo%20una%20serie%20de%20hip%C3%B3tesis%3A&text=La%20poblaci%C3%B3n%20de%20depredadores%20en%20ausencia%20de%20presas%20decrece%20de%20manera%20exponencial.

# Modelling of this phenomenon #

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Equation:
def df_dt (x, t, a, b, c, d):
    dx = a * x[0] - b * x[0] * x[1]
    dy = - c * x[1] + d * x[0] * x[1]

    return np.array([dx, dy])

# Parameters:
a = 0.15
b = 0.02
c = 0.3
d = 0.01

# Initial Conditions:
x0 = 40
y0 = 19
initial_cond = np.array([x0, y0])

# Conditions for Integration:
T = 200
time_steps = 800
t = np.linspace(0, T, time_steps)

# Solution:
solution = odeint(df_dt, initial_cond, t, args=(a, b, c, d))

# Plot:
plt.plot(t, solution[:, 0], label='prey')
plt.plot(t, solution[:,1], label='predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Lotka-Volterra')
plt.show()

# Other plot:
plt.figure("Prey vs predator")
plt.plot(solution[:, 0], solution[:, 1])
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.show()

# Directions field:
x_max = np.max(solution[:,0]) * 1.05
y_max = np.max(solution[:,1]) * 1.05

xy_step = 25
x = np.linspace(0, x_max, xy_step)
y = np.linspace(0, y_max, xy_step)
xx, yy = np.meshgrid(x, y)

uu, vv = df_dt((xx, yy), 0, a, b, c, d)
norm = np.sqrt(uu**2 + vv**2)
uu = uu / norm
vv = vv / norm

# Plot:
plt.figure("Field map")
plt.quiver(xx, yy, uu, vv, norm, cmap=plt.cm.gray)
plt.plot(solution[:, 0], solution[:, 1])
plt.xlim(0, x_max)
plt.ylim(0, y_max)
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.show()


