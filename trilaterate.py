import numpy as np
from scipy.optimize import least_squares
from math import pi 
from numpy.linalg import svd


def compute_residuals(pos, beacon_positions, distances):
    """
    Returns list of errors between a given position (pos) and all beacon positions
    where the error is simply the estimated distance minus actual distance 
    """
    residuals = []
    pos = np.array(pos)
    for b, d in zip(beacon_positions, distances):
        residuals.append(np.linalg.norm(pos-b)-d)
    return residuals 

def trilaterate(beacon_positions, beacon_distances, beacon_bearings):
    """
    Find point such that distances from it to known beacons are the same as the observed distances
    """
    if len(beacon_positions) < 3 or is_degenerate(beacon_positions):
        return 
    # Position
    beacon_positions = np.array(beacon_positions)
    beacon_distances = np.array(beacon_distances)
    # Initial guess: midpoints between beacons
    initial_guess = np.mean(beacon_positions, axis=0)
    # Least squares optimization
    result = least_squares(compute_residuals, initial_guess, args=(beacon_positions, beacon_distances))
    est_x, est_y = result.x
    # Orientation
    #  Calculate orientation using bearing, estimated robot position and actual beacon position
    #  using atan2. Take average of this calculation over all beacons (using cos, sin)
    angle_x, angle_y = 0, 0
    for (beacon_x, beacon_y), beacon_bearing in zip(beacon_positions, beacon_bearings):
        angle = np.atan2(beacon_y-est_y, beacon_x-est_x)-beacon_bearing
        # Wrap into interval [0, 2π]:
        angle %= 2*pi 
        angle_x += np.cos(angle)
        angle_y += np.sin(angle)
    angle_x = angle_x/len(beacon_positions)
    angle_y = angle_y/len(beacon_positions)
    est_angle = np.atan2(angle_y, angle_x)
    est_angle %= 2*pi 
    # Return trilaterated position 
    return est_x, est_y, est_angle 

def is_degenerate(beacon_positions, tol=1e-3):
    """
    Detect "degenerate configuration" of beacons
    Written with help of ChatGPT to detect if a beacon configuration is "degenerate"
    i.e. when three beacons are arranged such that it doesn't result in a unique solution 
    """
    if len(beacon_positions) < 3:
        return True
    # Check colinearity (pick first 3 for simplicity)
    if len(beacon_positions) == 3:
        a, b, c = beacon_positions
        area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
        if area < tol:
            return True
    # Compute Jacobian matrix at centroid
    center = np.mean(beacon_positions, axis=0)
    J = []
    for b in beacon_positions:
        diff = center - b
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return True  # overlapping beacon
        J.append(diff / dist)
    J = np.array(J)
    # Check condition number
    u, s, vh = svd(J)
    condition_number = s[0] / s[-1]
    return condition_number > 1e3  # adjust threshold as needed
