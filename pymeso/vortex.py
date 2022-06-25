"""
Computational rankine vortex simulation 

Jordan Brook - 5 July 2018
"""
import numpy as np
from math import acos
from math import sqrt
from math import pi

#define small functions for linear algebra
def length(v):
    """
    return the magnitude of a 2D vector
    input: (2D tuple) both 2D vector components
    output: (float) magnitude of 2D vector
    """
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v, w):
    """
    return the dot product of a 2D vector
    input: (2 x 2D tuple) both 2D vector components of the two input vectors
    output: (float) dot product of two vectors
    """
    return v[0]*w[0]+v[1]*w[1]

def inner_angle(v, w):
    """
    returns the angle between two vectors
    input: (2 x 2D tuple) both 2D vector components of the two input vectors
    output: (float) angle between input vectors in degrees
    """
    cosx = dot_product(v, w)/(length(v)*length(w))
    rad  = acos(cosx) 
    return rad*180/pi 

def angle_clockwise(A, B):
    """
    returns the smallest angle between two vectors
    input: (2 x 2D tuple) both 2D vector components of the two input vectors
    output: (float) smallest angle between input vectors in degrees
    """
    inner = inner_angle(A, B)
    return(min(inner, 360-inner))

    
def polar_vortex(beam, gate_size, max_range, pos, radius, vmax, clockwise, noise):
    """
    Simulate a rankine vortex in polar coordinates as observed by a Doppler radar
    Parameters:
    ===========
    beam: float
        the azimuthal grid spacing (degrees)
    gate_size: float
        the radial grid spacing (km)
    max_range: float
        the maximum range of the calculation (km)
    pos: 2D array
        2D array containing the x and y positions of the vortex respectively 
        (relative to the radar at point [0,0], in km)
    radius: float
        radius of the Rankine vortex (km)
    clockwise: bool
        True for clockwise, False for anti-clockwise circulations
    noise: bool
        True to include Gaussian radar noise in resulting Doppler field
    Returns:
    ========
    x: array
        x positions of all grid points
    y: array
        y positions of all grid points
    U: array
        velocity in x direction
    V: array
        velocity in y direction
    doppler: array
        Doppler velocity (towards or away from radar)
    """
    
    #determine directionality of circulation
    if clockwise == True:
        power = 2
    else:
        power = 1
    
    #add wind velocity
    wind_v = 0
    #add wind direction
    wind_t = np.pi/4
    
    #redefining initial positions
    x0, y0 = pos[0], pos[1]
    
    #define radius and angle vectors from input variables
    r = np.linspace(0, max_range, int(max_range/gate_size-1))
    theta = np.linspace(0, 359.9, int(360/beam))
    theta = theta * np.pi/180.0
    
    #defining the cartesian grid from input variables
    r, theta = np.meshgrid(r, theta)
    x = r* np.cos(theta)
    y = r* np.sin(theta)
    
    #define the angle from the centre of the vortex
    thetav = np.arctan2((y-y0), (x-x0))
    thetav[thetav<0] += 2*np.pi
    
    #define radius from the centre of the vortex
    vortex = np.sqrt((x - x0)**2 + (y - y0)**2)
    
    #define the tangential velocities at each point according to Rankine Mechanics
    TV = np.zeros(vortex.shape)
    for i in range(vortex.shape[0]):
        for j in range(vortex.shape[1]):
            if vortex[i, j] <= radius:
                TV[i, j] = vmax*vortex[i, j]/radius
            elif vortex[i, j] > radius:
                TV[i, j] = vmax*radius/vortex[i, j]
    
    #calculate the x and y components of velocity
    U = (-1)**power*TV*np.sin(thetav) + wind_v*np.cos(wind_t)
    V = (-1)**(power+1)*TV*np.cos(thetav) + wind_v*np.sin(wind_t)
    
    #calculate the doppler return for each gridpoint
    angle   = np.zeros(r.shape)
    doppler = np.zeros(r.shape)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            angle[i, j]   = angle_clockwise((x[i,j], y[i,j]), (U[i,j], V[i,j]))*np.pi/180
            doppler[i, j] = length((U[i, j], V[i, j]))*np.cos(angle[i, j])
    
    #add radar noise
    if noise == True:
        noise   = np.random.normal(0, 2, doppler.shape)
        doppler = doppler + noise
    
    #return the required variables
    return x, y, U, V, doppler
