import numpy as np
import matplotlib.pyplot as plt

# Runge Kutta 4th order method
def rk4(f, x, h, r1, r2, b1, b2):
    k1 = h * f(x, r1, r2, b1, b2)
    k2 = h * f(x + 0.5 * k1, r1, r2, b1, b2)
    k3 = h * f(x + 0.5 * k2, r1, r2, b1, b2)
    k4 = h * f(x + k3, r1, r2, b1, b2)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# LV model differential equations.
def deriv(y, r1, r2, b1, b2):
    R, F = y
    dRdt = r1 * R - b1 * R * F
    dFdt = -r2 * F + b2 * R * F
    return np.array([dRdt, dFdt])

# Total population, N.
N = 600
# Initial number of rabbits and foxes, R0 and F0.
R0, F0 = 40, 9
# Initialy there are a total population of 600 animals
r1 = 0.1
b1 = 0.02
r2 = 0.3
b2 = 0.01

# A grid of time points (in months)
t = np.linspace(0, 200, 200)

# Initial conditions vector
y0 = np.array([R0, F0])
# Integrate the SIR equations over the time grid, t.
ret = np.array([y0])
for i in range(len(t) - 1):
    ret = np.vstack((ret, rk4(deriv, ret[-1], t[i + 1] - t[i], r1, r2, b1, b2)))
R, F = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
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
