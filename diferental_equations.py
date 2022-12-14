import numpy as np

# LV model whitout coexistence
def deriv_basic(y, a, parameters) :
    r1, r2 = parameters['r1'], parameters['r2']
    R, F = y
    dRdt = r1 * R
    dFdt = -r2 * F
    dTdt = dRdt + dFdt
    return np.array([dRdt, dFdt])

# LV model differential equations.
def deriv(y, a, parameters):
    r1, r2, b1, b2 = parameters['r1'], parameters['r2'], parameters['b1'], parameters['b2']
    R, F = y
    dRdt = r1 * R - b1 * R * F
    dFdt = -r2 * F + b2 * R * F
    dTdt = dRdt + dFdt
    return np.array([dRdt, dFdt])

# LV improved model
def deriv_improved(y, a, parameters):
    r1, r2, b1, b2, c = parameters['r1'], parameters['r2'], parameters['b1'], parameters['b2'], parameters['c']
    R, F = y
    dRdt = r1 * R - b1 * R * F - c * R**2
    dFdt = -r2 * F + b2 * R * F
    return np.array([dRdt, dFdt])