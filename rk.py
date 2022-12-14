import numpy as np
import matplotlib.pyplot as plt
from diferental_equations import deriv_basic, deriv, deriv_improved

# Runge Kutta 4th order method
def rk4(f, x, h, a, parameters):
    k1 = h * f(x, a, parameters)
    k2 = h * f(x + 0.5 * k1, a, parameters)
    k3 = h * f(x + 0.5 * k2, a, parameters)
    k4 = h * f(x + k3, a, parameters)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Initial number of preys and predators, R0 and F0.
R0, F0 = 40, 9
r1 = 0.1 # Instantaneous rate of increase of preys in the absence of predators
b1 = 0.02 # Susceptibility of preys to being hunted
r2 = 0.3 # Instantaneous rate of decline of predators in the case of absence of preys
b2 = 0.01 # Ability of predators to predation
c = 0.001 # Intraspecific competition factor
parameters = {'r1': r1, 'r2': r2, 'b1': b1, 'b2': b2, 'c': c}

# A grid of time points (in months)
t = np.linspace(0, 200, 200)

# Initial conditions vector
y0 = np.array([R0, F0])
# Integrate the LV equations over the time grid, t.
ret = np.array([y0])
for i in range(len(t) - 1):
    ret = np.vstack((ret, rk4(deriv, ret[-1], t[i + 1] - t[i], 0, parameters)))
R, F = ret.T

# Plot the data on two separate curves for R(t), and F(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, R, 'b', alpha=0.5, lw=2, label='Presas')
ax.plot(t, F, 'r', alpha=0.5, lw=2, label='Depredadores')
ax.set_xlabel('Tiempo / meses')
ax.set_ylabel('Poblacion')
ax.set_ylim(0,60)
# Limit the x axis range
ax.set_xlim(-0.1, 200)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()