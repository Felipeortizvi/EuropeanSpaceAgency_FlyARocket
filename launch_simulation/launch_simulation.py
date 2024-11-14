"""
Numerical calculation of rocket trajectory with air resistance.
In 2 dimensions, and with a time-dependent mass.

Created on Thu 07 Dec 2023 at 14:50:30.
Last modified [dd.mm.yyyy]: 14-11-2024.
@author: Felipe Ortiz Villegas.
"""

import numpy as np # maths
import matplotlib.pyplot as plt # plotting


# ========================= Constants & parameters ========================== #

# Constants
g = 9.81				# gravitational acceleration [m/s^2]
rho_0 = 1.225			# air density at sea level [kg/m^3]
H = np.inf	        	# scale height [m]. Hint to 3a: np.inf = positive infinity
C_D = 0.5				# drag coefficient of the rocket [-]
A = 1.0e-2          	# rocket body frontal area [m^2]. "e-2" is short for "*10**(-2)"
m_0 = 20.000			# wet mass of the rocket [kg]
m_f = 20.000			# dry mass of the rocket [kg]
T_0 = 9000.1		    # average rocket engine thrust [N]
t_b = 6.0		    	# burn time [s]
theta_0 = 90*np.pi/180  # launch angle [rad]. Not used in 1D.


# Simulation parameters
dt = 0.01				# simulation time step [s]
#t_0 = 0                 # simulation start [s]; not needed when we start at 0
t_f = 131				# simulation time end [s]



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


def D_y(t, y, v, v_y):
    """
    Acceleration in the y-direction due to air resistance [m/s^2]
    as a function of time [s], altitude y [m], and velocity v, v_y [m/s]
    """
    return -0.5 * C_D * A * rho(y) * v * v_y



# ======================== Numerical implementation ========================= #

# Calculate the number of data points in our simulation
N = int(np.ceil(t_f/dt))

# We assume constant thrust, so we calculate the components here.
#T_y = T_0*np.sin(theta_0)
T_y = T_0 # In 1 dimension, we are always firing straight up.


# Create data lists
# Except for the time list, all lists are initialized as lists of zeros.
t = np.arange(t_f, step=dt) # runs from 0 to t_f with step length dt
y = np.zeros(N)
v_y = np.zeros(N)
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
    v = np.sqrt(v_y[n]**2) # Powers, like a^2, is written a**2
    
    # Acceleration
    a_y[n] = ( T_y + D_y(t[n], y[n], v, v_y[n]) )/ m(t[n]) - g
    
    
    # Euler's method:
    # ---------------------------------- #
    # Position
    y[n+1] = y[n] + v_y[n]*dt
    
    # Velocity
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
    v = np.sqrt(v_y[n]**2)
    
    # Acceleration
    a_y[n] = D_y(t[n], y[n], v, v_y[n]) / m_f - g
    
    
    # Euler's method:
    # ---------------------------------- #
    # Position
    y[n+1] = y[n] + v_y[n]*dt
    
    # Velocity
    v_y[n+1] = v_y[n] + a_y[n]*dt
    
    
    # Advance n with 1
    n += 1
 



# When we exit the loops above, our index n has reached a value where the rocket
# has crashed (or it has reached its maximum value). Since we don't need the
# data after n, we redefine our lists to include only the points from 0 to n:
t = t[:n]
y = y[:n]
v_y = v_y[:n]
a_y = a_y[:n]


# ============================== Data analysis ============================== #

# Apogee
n_a = np.argmax(y) # Index at apogee


# =========================== Printing of results =========================== #

print('\n---------------------------------\n')
print('Apogee time:\t', t[n_a], 's')
print('... altitude:\t', round(y[n_a])/1000, 'km')
print('\n---------------------------------\n')


# =========================== Plotting of results =========================== #

# Close all currently open figures, so we avoid mixing up old and new figures.
plt.close('all')

# Trajectory
plt.figure('Trajectory')
plt.plot(t, y)
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.grid(linestyle='--')
plt.show()