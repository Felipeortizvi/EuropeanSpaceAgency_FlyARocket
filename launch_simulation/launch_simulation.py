"""
Numerical calculation of rocket trajectory with air resistance.
In 2 dimensions, and with a time-dependent mass.

Created on Thu 07 Dec 2023 at 14:50:30.
Last modified [dd.mm.yyyy]: 14-11-2024.
@author: Felipe Ortiz Villegas.

This is a test change to see if ssh key is working.

Goals: To make this a Class, and along with that make it much more complex.
"""

import numpy as np # maths
import matplotlib.pyplot as plt # plotting


# ========================= Constants & parameters ========================== #

# Constants
g = 9.81                # gravitational acceleration [m/s^2]
rho_0 = 1.225           # air density at sea level [kg/m^3]
H = 7700.0              # scale height [m]
C_D = 0.51              # drag coefficient of the rocket [-]
A = 1.081e-2            # rocket body frontal area [m^2]. 1.081 dm^2 = 1.081e-2 m^2
m_0 = 19.765            # wet mass of the rocket [kg]
m_f = 11.269            # dry mass of the rocket [kg]
T_0 = 2501.8            # average rocket engine thrust [N]
t_b = 6.09              # burn time [s]
theta_0 = 75 * np.pi / 180  # launch angle [rad]

# Simulation parameters
dt = 0.001              # simulation time step [s]
t_f = 180.0             # simulation end time [s]


# ================================ Functions ================================ #

def m(t):
    """
    Rocket mass [kg]
    as a function of time t [s]
    PS! Assumes 0 <= t <= t_b.
    """
    return m_0 - (m_0 - m_f) * t / t_b


def rho(y):
    """
    Air density [kg/m^3]
    as a function of altitude y [m]
    """
    return rho_0 * np.exp(-y/H)


def D_i(y, v, v_i):
    """Drag in the i-Direction

    as a function of time [s], altitude y [m], and velocity v, v_i [m/s]
    """
    return -0.5 * C_D * A * rho(y) * v * v_i

# ======================== Numerical implementation ========================= #

# Calculate the number of data points in our simulation
N = int(np.ceil(t_f/dt))

# We assume constant thrust, so we calculate the components here.
#T_y = T_0*np.sin(theta_0)
T_y = T_0 # In 1 dimension, we are always firing straight up.

# Under the assumption of constant thrust:
T_x = T_0*np.cos(theta_0)
T_y = T_0*np.sin(theta_0)


# Create data lists
# Except for the time list, all lists are initialized as lists of zeros.
t = np.arange(t_f, step=dt) # runs from 0 to t_f with step length dt
y = np.zeros(N)
x = np.zeros(N)
v_x = np.zeros(N)
v_y = np.zeros(N)
a_x = np.zeros(N)
a_y = np.zeros(N)


# We will use while loops to iterate over our data lists. For this, we will use
# the auxillary variable n to keep track of which element we're looking at.
# The data points are numbered from 0 to N-1
n = 0
n_max = N - 1


# Thrusting phase
# ---------------------------------- #
# First, we iterate until the motor has finished burning, or until we reach the lists' end:
while t[n] < t_b and n < n_max:
    # Values needed for Euler's method
    # ---------------------------------- #
    # Speed
    v = np.sqrt(v_x[n]**2 + v_y[n]**2) 
    
    # Acceleration
    a_x[n] = ( T_x + D_i(y[n], v, v_x[n]) ) / m(t[n])
    a_y[n] = ( T_y + D_i(y[n], v, v_y[n]) ) / m(t[n]) - g  
    
    # Euler's method:
    # ---------------------------------- #
    # Position
    x[n+1] = x[n] + v_x[n]*dt
    y[n+1] = y[n] + v_y[n]*dt
    
    # Velocity
    v_x[n+1] = v_x[n] + a_x[n]*dt
    v_y[n+1] = v_y[n] + a_y[n]*dt
    
    # Advance n with 1
    n += 1



# Coasting phase
# ---------------------------------- #
# Then we iterate until the rocket has crashed, or until we reach the lists' end:
while y[n] >= 0 and n < n_max:
    # Values needed for Euler's method
    # ---------------------------------- #
    # Speed
    v = np.sqrt(v_x[n]**2 + v_y[n]**2) 
    
    # Acceleration
    a_x[n] = D_i(y[n], v, v_x[n]) / m_f
    a_y[n] = D_i(y[n], v, v_y[n]) / m_f - g
    
    
    # Euler's method:
    # ---------------------------------- #
    # Position
    y[n+1] = y[n] + v_y[n]*dt
    x[n+1] = x[n] + v_x[n]*dt
    
    # Velocity
    v_y[n+1] = v_y[n] + a_y[n]*dt
    v_x[n+1] = v_x[n] + a_x[n]*dt
    
    
    # Advance n with 1
    n += 1
 



# When we exit the loops above, our index n has reached a value where the rocket
# has crashed (or it has reached its maximum value). Since we don't need the
# data after n, we redefine our lists to include only the points from 0 to n:
t = t[:n]
x = x[:n]
y = y[:n]
v_x = v_x[:n]
v_y = v_y[:n]
a_x = a_x[:n]
a_y = a_y[:n]


# ============================== Data analysis ============================== #

# Burnout
n_b = np.argmin(np.abs(t - t_b)) # Index where t is closest to t_b

# Apogee
n_a = np.argmax(y) # Index at apogee

# Speed
v = np.sqrt(v_x**2 + v_y**2)




# =========================== Printing of results =========================== #

print('\n---------------------------------\n')
print('Burnout time:\t', t[n_b] ,'s') # Should be close or equal to t_b
print('... altitude:\t', round(y[n_b]), 'm')
print('... speed:\t\t', round(v[n_b]), 'm/s')
print('')
print('Apogee time:\t', t[n_a], 's')
print('... altitude:\t', round(y[n_a]), 'm')
print('... speed:\t\t', round(v[n_a]), 'm/s') # For debugging purposes
print('')
print('Final time:\t\t', t[-1], 's')
print('... altitude:\t', round(y[-1]), 'm')
print('... speed:\t\t', round(v[-1]), 'm/s')
print('\n---------------------------------\n')