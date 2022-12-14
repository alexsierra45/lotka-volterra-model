import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from diferental_equations import deriv_basic, deriv, deriv_improved

# Initial number of preys and predators, R0 and F0.
R0, F0 = 40, 9
# Initialy there are a total population of 49 animals
r1 = 0.1 # Instantaneous rate of increase of preys in the absence of predators
b1 = 0.02 # Susceptibility of preys to being hunted
r2 = 0.3 # Instantaneous rate of decline of predators in the case of absence of preys
b2 = 0.01 # Ability of predators to predation
c = 0.001 # Intraspecific competition factor
parameters = {'r1': r1, 'r2': r2, 'b1': b1, 'b2': b2, 'c': c}

# A grid of time points (in months)
t = np.linspace(0, 200, 200)

# Initial conditions vector
y0 = np.array((R0, F0))
# Integrate the LV equations over the time grid, t.
ret = odeint(deriv_improved, y0, t, (parameters,))
R, F = ret.T

x_max = np.max(R) * 1.05
y_max = np.max(F) * 1.05
x = np.linspace(0, x_max, 25)
y = np.linspace(0, y_max, 25)
xx, yy = np.meshgrid(x, y)
uu, vv = deriv((xx, yy), 0, parameters)
norm = np.sqrt(uu**2 + vv**2)
uu = uu / norm
vv = vv / norm
plt.quiver(xx, yy, uu, vv, norm, cmap=plt.cm.gray)
plt.plot(R, F)