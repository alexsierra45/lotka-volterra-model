import matplotlib.pyplot as plt
import numpy as np
from mpl_interactions import ipyplot as iplt
from scipy import integrate

t = np.linspace(0, 200, 200)  # time
X0 = np.array([40, 9])

def f0(r1, r2, b1, b2):
    def dX_dt(X, t=0):
        rabbits, foxes = X
        dRabbit_dt = r1 * rabbits - b1 * foxes * rabbits
        dFox_dt = -r2 * foxes + b2 * rabbits * foxes
        return [dRabbit_dt, dFox_dt]

    X, _ = integrate.odeint(dX_dt, X0, t, full_output=True)
    return X

def f1(r1, r2, b1, b2, c=0.001):
    def dX_dt(X, t=0):
        rabbits, foxes = X
        dRabbit_dt = r1 * rabbits - b1 * foxes * rabbits - c * rabbits**2
        dFox_dt = -r2 * foxes + b2 * rabbits * foxes
        return [dRabbit_dt, dFox_dt]

    X, _ = integrate.odeint(dX_dt, X0, t, full_output=True)
    return X

func = input()
if func == 'f0' : func = f0
else : func = f1
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4.8))
controls = iplt.plot(
    func, ax=ax1, r1=(0.05, 0.2), b1=(0.01, 0.04), r2=(0.1, 0.5), b2=(0.005, 0.03), parametric=True
)
ax1.set_xlabel("rabbits")
ax1.set_ylabel("foxes")

iplt.plot(func, ax=ax2, controls=controls, label=["rabbits", "foxes"])
ax2.set_xlabel("time")
ax2.set_ylabel("population")
_ = ax2.legend()
plt.show()