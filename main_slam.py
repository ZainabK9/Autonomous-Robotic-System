import os
import pygame
import math
from math import pi
import numpy as np
import random
import datetime as dt
from PIL import Image
import imageio.v2 as imageio
from trilaterate import trilaterate


# True map
NUM_CELLS = 20  # number of "world cells" along each axis
CELL_SIZE = 40 
OBSTACLE_DENSITY = 0.2  # 0.2
NUM_BEACONS = 100  # 100
# Screen 
SCREEN_SIZE = NUM_CELLS*CELL_SIZE 
SCREEN_CAPTURE = True 
# Predicted map 
NUM_GRID_CELLS = 40  # number of "grid cells" along each axis; choose s.t. SCREEN_SIZE is divisible by it
GRID_CELL_SIZE = SCREEN_SIZE//NUM_GRID_CELLS
# Colors
BACKGROUND_COLOR = (30, 30, 30)
ROBOT_COLOR = (200, 50, 50)
SENSOR_COLOR = (50, 200, 50)
OBSTACLE_COLOR = (100, 100, 100)
SENSOR_HIT_COLOR = (230, 0, 0)
PASSIVE_BEACON_COLOR = (130, 130, 130) 
ACTIVE_BEACON_COLOR = (0, 255, 0) 
# Robot
ROBOT_RADIUS = 10
MOVE_SPEED = 3  # linear speed factor 
ROTATE_SPEED = 1  # angular speed factor
# Robot sensors 
NUM_SENSORS = 12
SENSOR_RANGE = 120  # 100
SENSOR_ANGLE_STEP = 2*pi/NUM_SENSORS
SENSOR_STEP_SIZE = 4  # ideally, SENSOR_RANGE divisible by it 
OMNI_RANGE = SENSOR_RANGE  # range around robot where it can detect beacons 
OCCUPANCY_ERROR = 0.001
OCCUPANCY_LOG_ODDS = np.log((1-OCCUPANCY_ERROR)/OCCUPANCY_ERROR)  # log-odds of occupancy
# Visibility 
flag_changed = False 
obstacles_visible = False 
robot_info_visible = True 
trajectory_visible = False
beacon_visible = False 
grid_visible = True 
occupancy_values_visible = False
# Kalman filter 
V = np.diag([1e-6, 1e-6, 1e-6])  # initial state-estimate covariance 
R = np.diag([1e-6, 1e-6, 1e-6])  # motion-noise covariance
Q = np.diag([1e-6, 1e-6, 1e-6])  # sensor-noise covariance



# === EKF-SLAM Helpers ===
def initialize_slam_state(robot_pose, beacon_positions, robot_cov, landmark_cov):
    M = len(beacon_positions)
    dim = 3 + 2*M
    X = np.zeros(dim)
    X[0:3] = robot_pose
    for i, pos in enumerate(beacon_positions):
        X[3+2*i:3+2*i+2] = pos
    P = np.zeros((dim, dim))
    P[0:3, 0:3] = robot_cov
    for i in range(M):
        idx = 3 + 2*i
        P[idx:idx+2, idx:idx+2] = landmark_cov
    return X, P

def motion_jacobian(robot_state, u):
    x, y, theta = robot_state
    v_l, v_r = u
    v = (v_l + v_r) / 2
    F = np.eye(3)
    F[0,2] = -v * math.sin(theta)
    F[1,2] =  v * math.cos(theta)
    return F

# === EKF-SLAM Prediction ===
# Predict robot motion in EKF-SLAM
# X: state vector (x, y, theta, lx1, ly1, ..., lxM, lyM)
# P: covariance matrix
# u: control input (v_l, v_r)
# R: process noise covariance
# Returns: predicted state and covariance
def predict_slam(X, P, u, R):
    # Predict robot motion in EKF-SLAM
    x, y, theta = X[0:3]
    v_l, v_r = u
    v = (v_l + v_r) / 2
    omega = (v_r - v_l)
    # motion update
    X_pred = X.copy()
    X_pred[0] += v * math.cos(theta)
    X_pred[1] += v * math.sin(theta)
    X_pred[2] = (theta + omega) % (2*pi)
    # Jacobian
    Fx = motion_jacobian(X[0:3], u)
    M = (len(X)-3)//2
    F = np.eye(len(X))
    F[0:3,0:3] = Fx
    # process noise
    Q_full = np.zeros_like(P)
    Q_full[0:3,0:3] = R
    # covariance
    P_pred = F @ P @ F.T + Q_full
    return X_pred, P_pred

# === EKF-SLAM Measurement Update ===
def meas_pred(X, i):
    x, y, theta = X[0:3]
    idx = 3+2*i
    # landmark position
    lx, ly = X[idx], X[idx+1]
    dx, dy = lx - x, ly - y
    r = math.hypot(dx, dy)
    phi = (math.atan2(dy, dx) - theta) % (2*pi)
    return np.array([r, phi])

# Jacobian of the measurement model
def meas_jac(X, i):
    x, y, theta = X[0:3]
    idx = 3+2*i
    lx, ly = X[idx], X[idx+1]
    dx, dy = lx - x, ly - y
    q = dx*dx + dy*dy
    sqrt_q = math.sqrt(q)
    H = np.zeros((2, len(X)))
    # dr/dx, dr/dy
    H[0,0] = -dx/sqrt_q; H[0,1] = -dy/sqrt_q
    # dr/dlx, dr/dly
    H[0,idx] = dx/sqrt_q; H[0,idx+1] = dy/sqrt_q
    # dphi/dx, dphi/dy, dphi/dtheta
    H[1,0] = dy/q; H[1,1] = -dx/q; H[1,2] = -1
    # dphi/dlx, dphi/dly
    H[1,idx]   = -dy/q; H[1,idx+1] = dx/q
    return H

def ekf_slam_correction(X, P, measurements, Q):
    # measurements: list of (i, [r,phi])
    zs, zhs, Hs = [], [], []
    # loop through all measurements
    for i, z in measurements:
        # z is the measurement vector for landmark i
        zs.append(z)
        # zhat is the predicted measurement vector for landmark i
        zhs.append(meas_pred(X, i))
        Hs.append(meas_jac(X, i))
    z = np.hstack(zs)
    zhat = np.hstack(zhs)
    H = np.vstack(Hs)
    R_big = np.kron(np.eye(len(measurements)), Q)
    S = H @ P @ H.T + R_big
    K = P @ H.T @ np.linalg.inv(S)
    X_new = X + K @ (z - zhat)
    P_new = (np.eye(len(X)) - K @ H) @ P
    return X_new, P_new

class Environment:

    # Initialization ---------------------------------------------------------------------------------------------
        
    def __init__(self):
        pygame.init()
        self.reset()
    
    def reset(self):
        # Reset counters & visuals
        self.timestep = 0
        self.frames = []
        self.prev_frame = None

        # Build the true map & robot spawn
        self.initialize_map()
        self.initialize_beacons()
        self.initialize_grid()
        self.initialize_robot()

        # Initialize SLAM state
        robot_pose   = [self.spawn_x, self.spawn_y, self.robot_angle]
        # Landmark positions are the beacons' positions
        # If the red X’s are updating too slowly toward the true landmark positions,
        # try reducing the initial landmark covariance (e.g., landmark_cov = np.diag([1e2, 1e2]))
        # or increasing the measurement noise covariance Q
        # 1e3 is a good value for the initial landmark covariance.
        landmark_cov = np.diag([1e3, 1e3])
        self.X, self.P = initialize_slam_state(robot_pose, self.beacons, V, landmark_cov)

        # Now set up your belief variables
        # At this point, self.belief_mean isn’t defined, so define it from X
        self.belief_mean       = self.X[0:3].copy()
        self.belief_cov        = self.P[0:3, 0:3].copy()
        self.belief_trajectory = [ self.belief_mean[:2].copy() ]

    def update_belief(self, action):
        # predict
        self.X, self.P = predict_slam(self.X, self.P, action, R)
        # build measurements from omni-sensor
        meas = []
        for j, dist in enumerate(self.beacon_distances):
            if dist < math.inf:
                meas.append((j, np.array([dist, self.beacon_bearings[j]])))
        # correct
        if meas:
            self.X, self.P = ekf_slam_correction(self.X, self.P, meas, np.diag([1e-2,1e-2]))
        # unpack robot pose
        self.belief_mean = self.X[0:3]
        # store belief trajectory
        self.belief_trajectory.append(self.belief_mean[0:2])

    def initialize_map(self):
        """
        Generates a map with randomly placed blocks.
        """
        self.obstacles = []
        center_rect = pygame.Rect(SCREEN_SIZE/2-2*CELL_SIZE/2, SCREEN_SIZE/2-2*CELL_SIZE/2, 2*CELL_SIZE, 2*CELL_SIZE)
        # Calculate the number of obstacles based on density 
        n_obstacles = int((SCREEN_SIZE**2)/(CELL_SIZE**2)*OBSTACLE_DENSITY)
        while len(self.obstacles) < n_obstacles:
            x = random.randint(0, (SCREEN_SIZE//CELL_SIZE)-1)*CELL_SIZE
            y = random.randint(0, (SCREEN_SIZE//CELL_SIZE)-1)*CELL_SIZE
            # Create the obstacle rectangle
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            # Avoid overlapping obstacles and don't place obstacles in center
            if not (rect.colliderect(center_rect) or any(rect.colliderect(obstacle) for obstacle in self.obstacles)):
                self.obstacles.append(rect)
        # Add boundary walls to fully enclose the map
        self.obstacles.extend([
            pygame.Rect(0, 0, SCREEN_SIZE, 5),              # Top boundary
            pygame.Rect(0, 0, 5, SCREEN_SIZE),              # Left boundary
            pygame.Rect(0, SCREEN_SIZE-5, SCREEN_SIZE, 5),  # Bottom boundary
            pygame.Rect(SCREEN_SIZE-5, 0, 5, SCREEN_SIZE)   # Right boundary
        ])

    def initialize_grid(self):
        """
        """
        self.grid = []  # coordinates of grid cell centers 
        self.grid_distances = []  # current distances from robot to grid cell centers
        self.occupancy = [] 
        self.detected_occupancy = []
        for x in range(0, SCREEN_SIZE, GRID_CELL_SIZE):
            for y in range(0, SCREEN_SIZE, GRID_CELL_SIZE):
                xc, yc = x+GRID_CELL_SIZE//2, y+GRID_CELL_SIZE//2  # grid cell center coordinates
                self.grid.append((xc, yc))
                self.grid_distances.append(math.inf) 
                self.occupancy.append(0)  # log-odds of occupancy, prior is p=0.5 -> log(0.5/(1-0.5))=0
                self.detected_occupancy.append(0)  # 0 if not detected, 1 if detected (might be wrong due to sensor noise)
    
    def initialize_beacons(self):
        """
        Use obstacle corners as beacons.
        Only the squares are technically obstacles, and when two squares touch, 
        I do not count this as a corner and it is not a beacon.
        I simply take all squares' corners and then remove all points that 
        appear more than once.
        """
        self.beacons = []
        self.beacon_distances = []
        self.beacon_bearings = []
        
        '''
        # <temp>
        # Just for some testing, generate beacons completely randomly 
        #  and independently of obstacles!! (i.e. not as corners of obstacles!)
        if NUM_BEACONS > 0:
            MIN_BEACON_DISTANCE = 0.5 * OMNI_RANGE
            while True:
                x, y = np.random.uniform(5, SCREEN_SIZE-5, size=2)
                # Only actually add to self.beacons if the proposed beacons is sufficiently 
                #  far away from all existing beacons! 
                #  This minimum distance (MIN_BEACON_DISTANCE) should not be too large, because then
                #  the robot will never be in range of 3 beacons for trilateration!! 
                valid = True 
                for other_beacon in self.beacons:
                    if np.linalg.norm(np.array(other_beacon)-np.array([x, y])) < MIN_BEACON_DISTANCE:
                        valid = False 
                        break 
                if valid:
                    self.beacons.append((x, y))
                    self.beacon_distances.append(math.inf) 
                    self.beacon_bearings.append(0)
                # Once we have NUM_BEACONS beacons, leave the while loop
                if len(self.beacons) == NUM_BEACONS:
                    break 
        # </temp>
        '''

        # '''
        if (NUM_BEACONS > 0) or (NUM_BEACONS is None):
            corners = set()
            remove = set()
            for obstacle in self.obstacles:
                for dx in [0, CELL_SIZE]:
                    for dy in [0, CELL_SIZE]:
                        corner = (obstacle.x+dx, obstacle.y+dy)
                        # TODO: (only a small detail) rectangles that touch the edges of the screen should
                        #  also be removed 
                        if corner not in corners:
                            corners.add(corner)
                        else:
                            remove.add(corner)
            # Remove those corners that have appeared more than once 
            corners = list(corners - remove)
            # Go through these corners and turn then into pygame.Rect objects, 
            #  and then put them into self.beacons and self.beacon_distances
            #  The latter is simply the list of detected distances, which are initialized to infinity 
            # Only actually keep NUM_BEACONS random beacons: 
            if NUM_BEACONS is not None:
                np.random.shuffle(corners)
                corners = corners[:NUM_BEACONS]
            for i, (x, y) in enumerate(corners):
                self.beacons.append((x, y))
                self.beacon_distances.append(math.inf) 
                self.beacon_bearings.append(0)
        # '''

    def initialize_robot(self):
        # Pose 
        # Important: robot_x, robot_y store the robot's relative position!
        #  To get the absolute position, add the spawn position (spawn_x, spawn_y)
        self.robot_x = 0
        self.robot_y = 0
        self.robot_angle = 0  # orientation (in radians)
        # Find a good spawn point 
        self.spawn_x = SCREEN_SIZE/2  # x-position
        self.spawn_y = SCREEN_SIZE/2  # y-position 
        # TODO: make so that spawn angle could be anything, 
        #  and robot_angle is also relative!! 
        #  When the robot spawns, it doesn't necessarily know its absolute direction!! 
        #  Though, if we do local localization, we presumably also tell it its 
        #  initial orientation.
        # self.spawn_angle = 0 
        # Make sure the robot is not placed on top of or inside an obstacle: 
        while self.is_colliding():
            # Reposition the robot's spawn randomly within the map bounds:
            self.spawn_x = random.randint(ROBOT_RADIUS, SCREEN_SIZE-ROBOT_RADIUS)
            self.spawn_y = random.randint(ROBOT_RADIUS, SCREEN_SIZE-ROBOT_RADIUS)
        # List to keep track of robot's trajectory (position in each timestep)
        self.trajectory = [(0, 0)]
        # Sensors
        # Get initial sensor reading 
        self.compute_omni()
        self.compute_sensors()
        # Just for visualization, also keep track of these values inside self
        # (rather than just passing them as parameters to the move method)
        self.v_l = 0
        self.v_r = 0
    
    def initialize_belief(self):
        # Belief is the robot's belief of its own pose in absolute terms! 
        # Initialize belief at spawn point (local localization!) 
        #  and set small variance.
        #  For global localization, use random initial point and large variance.
        self.belief_mean = np.array([self.spawn_x, self.spawn_y, self.robot_angle])
        self.belief_cov = V 
        # Initialize belief trajectory 
        self.belief_trajectory = [self.belief_mean[:2]]

    # Step: transition to the next state -------------------------------------------------------------------------
    
    def step(self, action):
        """
        Transition from current environment state to the next 
        """
        global flag_changed
        self.compute_sensors()
        self.compute_omni()
        #  in pallarel to the robot's movement, compute the occupancy grid mapping and ekf_slam
        # occupancy grid mapping 
        self.compute_occupancy() 
        # ekf_slam 
        self.update_belief(action) 
        if SCREEN_CAPTURE:
            self.save_frame()
        self.render()
        self.timestep += 1
        flag_changed = False 
        clock.tick(60) 
    
    # Motion -----------------------------------------------------------------------------------------------------

    def move(self, action):
        """
        Parameter 'action' must be a tuple containing (v_l, v_r)
        Move robot at velocities v_l (left motor), v_r (right motor) 
        """
        if self.timestep > 0 and action == (0, 0) and not flag_changed:
            # If there is no movement, there is no need to calculate new pose
            #  Unless it's timestep 0 or the user toggled the visibility of something 
            #  This also means that, if no action is taken, time doesn't pass
            return 
        v_l, v_r = action
        # Keep track of v_l, v_r so we can put it on screen as text 
        #  (see draw_robot method)
        self.v_l, self.v_r = v_l, v_r 
        # Calculate linear and angular velocities from v_l, v_r 
        v_linear = MOVE_SPEED * (v_l + v_r) / 2  # Average speed of both motors
        v_angular = ROTATE_SPEED * (v_r - v_l) / ROBOT_RADIUS  # Differential rotation
        # Calculate new position based on current angle
        d_x = v_linear * math.cos(self.robot_angle)
        d_y = v_linear * math.sin(self.robot_angle)
        d_angle = v_angular
        # Calculate the new position
        new_x = self.robot_x + d_x
        new_y = self.robot_y + d_y
        new_angle = self.robot_angle + d_angle 
        # Make sure angle stays in range [0, 2π]:
        new_angle %= 2*pi 
        # Add motion noise (using matrix R) 
        noise_x, noise_y, noise_angle = np.random.multivariate_normal(np.zeros(3), R, size=1)[0]
        new_x += noise_x 
        new_y += noise_y 
        new_angle += noise_angle 
        # Collision check 
        robot_rect = pygame.Rect(self.spawn_x + new_x - ROBOT_RADIUS, 
                                 self.spawn_y + new_y - ROBOT_RADIUS, 
                                 ROBOT_RADIUS * 2, 
                                 ROBOT_RADIUS * 2)
        x_clear = True
        y_clear = True
        for obs in self.obstacles:
            if robot_rect.colliderect(obs):
                # Check for sliding along the wall
                # Try moving only in the x direction
                temp_x = self.robot_x + d_x
                temp_rect_x = pygame.Rect(self.spawn_x + temp_x - ROBOT_RADIUS, 
                                          self.spawn_y + self.robot_y - ROBOT_RADIUS, 
                                          ROBOT_RADIUS * 2, 
                                          ROBOT_RADIUS * 2)
                if temp_rect_x.colliderect(obs):
                    x_clear = False
                # Try moving only in the y direction
                temp_y = self.robot_y + d_y
                temp_rect_y = pygame.Rect(self.spawn_x + self.robot_x - ROBOT_RADIUS, 
                                          self.spawn_y + temp_y - ROBOT_RADIUS, 
                                          ROBOT_RADIUS * 2, 
                                          ROBOT_RADIUS * 2)
                if temp_rect_y.colliderect(obs):
                    y_clear = False
        # Resolve movement based on collision checks
        if not x_clear and not y_clear:
            # Block movement completely if both directions are blocked
            # Only change angle 
            self.robot_angle = new_angle 
        elif not x_clear:
            # Allow sliding along the y-axis
            self.robot_y = new_y 
            self.robot_angle = new_angle
        elif not y_clear:
            # Allow sliding along the x-axis
            self.robot_x = new_x 
            self.robot_angle = new_angle
        else:
            # If no collision, move to the new position
            self.robot_x = new_x
            self.robot_y = new_y
            self.robot_angle = new_angle
        # Save new position in trajectory
        self.trajectory.append((self.robot_x, self.robot_y))
        # Transition to next state 
        env.step(action) 

    # Sensors ----------------------------------------------------------------------------------------------------

    def compute_sensors(self):
        """
        Determine for each sensor: whether any point within the sensor's range 
        (i.e. point along the line from the robot to the sensor's maximum-range point) 
        collides with an obstacle, and, if so, at what distance.
        The corresponding readings are put in the self.sensors list.
        """
        self.sensors = []
        for i in range(NUM_SENSORS): 
            sensor_angle = self.robot_angle + i * SENSOR_ANGLE_STEP
            hit = False 
            distance = math.inf 
            sensor_x, sensor_y = self.robot_x, self.robot_y 
            # Step along the line to detect the closest intersection
            for d in range(0, SENSOR_RANGE, SENSOR_STEP_SIZE):  
                sensor_x = self.robot_x + d * math.cos(sensor_angle)
                sensor_y = self.robot_y + d * math.sin(sensor_angle)
                # point = int(sensor_x), int(sensor_y)
                # Use absolute position just to check collision 
                point = self.spawn_x + sensor_x, self.spawn_y + sensor_y 
                # Check collision with obstacles
                for obs in self.obstacles:
                    if obs.collidepoint(point):
                        hit = True
                        # Store the distance where collision occurs, in "sensor steps":
                        distance = max((d-ROBOT_RADIUS)//SENSOR_STEP_SIZE, 0) 
                        break
                if hit:
                    break
            # Important: these are ground-truth sensor measurements
            #  Robot only has access to noisy (hit, distance) data 
            #  Also, we only append relative positions!! 
            #  Otherwise, the robot would know its exact position from the sensor readings!
            self.sensors.append((sensor_x, sensor_y, hit, distance))
    
    def compute_omni(self):
        """
        Checks which beacons are in range of the robot's omnidirectional sensor 
        Computes the observation vector based on omni-sensor readings
        First applies trilateration to omni-sensor readings (detected beacon positions, distances, bearings) 
        and then outputs the estimated robot pose according to this trilateration.
        This estimated pose wil be used as the robot's "observation".
        """
        # Check which beacons are in range (detected)
        detected_beacon_positions = [] 
        detected_beacon_distances = []
        detected_beacon_bearings = []
        self.beacon_distances = [math.inf for _ in self.beacons]
        robot_pos = np.array([self.spawn_x+self.robot_x, self.spawn_y+self.robot_y])
        for j, (beacon_x, beacon_y) in enumerate(self.beacons):
            # Use ACTUAL beacon location and ACTUAL robot location to get distance measure 
            #  and only later add noise: 
            beacon_pos = np.array([beacon_x, beacon_y])
            distance = np.linalg.norm(beacon_pos-robot_pos)
            if distance < OMNI_RANGE:
                detected_beacon_positions.append((beacon_x, beacon_y))
                # Distance
                self.beacon_distances[j] = distance 
                detected_beacon_distances.append(distance)
                # Bearing 
                #  Robot doesn't actually know true self.robot_angle, but it knows the 
                #  relative angle between its orientation and a beacon that is perceives -> "bearing" 
                bearing = np.atan2(beacon_y-(self.spawn_y+self.robot_y), beacon_x-(self.spawn_x+self.robot_x))-self.robot_angle 
                # Wrap into interval [0, 2π]:
                bearing %= 2*pi 
                self.beacon_bearings[j] = bearing 
                detected_beacon_bearings.append(bearing)
        # Compute "observation" (trilaterated robot position in absolute coordinates) 
        self.observation = None 
        # Trilateration
        trilateration = trilaterate(detected_beacon_positions, detected_beacon_distances, detected_beacon_bearings) 
        if trilateration is not None:
            # Add sensor noise (using matrix Q) 
            noise = np.random.multivariate_normal(np.zeros(3), Q, size=1)[0]
            trilateration += noise 
            # Save this as the observation 
            self.observation = trilateration 

    def compute_occupancy(self):
        """
        Checks which grid cells are in range of the robot's omnidirectional sensor 
        Updates grid_distances with the distances to the detected grid cells
        """
        self.grid_distances = [math.inf for _ in self.grid]
        x, y = self.spawn_x+self.robot_x, self.spawn_y+self.robot_y
        robot_pos = np.array([x, y])
        for j, (cell_x, cell_y) in enumerate(self.grid):
            cell_pos = np.array([cell_x, cell_y])  # cell center point
            distance = np.linalg.norm(cell_pos-robot_pos)  # distance from robot to cell center point
            if distance < OMNI_RANGE:
                angle = np.atan2(cell_y-y, cell_x-x)  # absolute angle between robot and cell center point 
                cell_rect = pygame.Rect(cell_x-GRID_CELL_SIZE//2, cell_y-GRID_CELL_SIZE//2, GRID_CELL_SIZE, GRID_CELL_SIZE) 
                # First, determine, distance to closest edge of cell of interest 
                # Then, check if any point along the line from the robot to the cell center collides with an obstacle
                distance_to_cell_edge = int(distance)  # it will be at most the distance to the cell's center 
                for d in range(0, OMNI_RANGE, SENSOR_STEP_SIZE): 
                    sensor_x = x + d * math.cos(angle)
                    sensor_y = y + d * math.sin(angle)
                    point = sensor_x, sensor_y 
                    # Check collision with cell of interest
                    if cell_rect.collidepoint(point):
                        # Store the distance where collision occurs, in "absolute units" (not in "sensor steps"):
                        # distance_to_cell_edge = int(max(d-ROBOT_RADIUS, 0) )
                        distance_to_cell_edge = int(max(d, 0))
                        # As soon as the first (closest) point of collision is found, break 
                        break 
                # Then, check if any point along the line from the robot to the cell's border (!),
                #  collides with an obstacle. If so, the cell of interest is occluded! 
                is_occluded = False 
                for d in range(0, distance_to_cell_edge, SENSOR_STEP_SIZE):  
                    sensor_x = x + d * math.cos(angle)
                    sensor_y = y + d * math.sin(angle)
                    point = sensor_x, sensor_y 
                    # Check collision with obstacles (but not counting the cell of interest!!)
                    for obs in self.obstacles:
                        # and not obs.colliderect(cell_rect)
                        if obs.collidepoint(point) and not cell_rect.collidepoint(point):
                            is_occluded = True 
                            break
                    if is_occluded:
                        break
                # Check if cell is actually occupied (ground truth)
                is_occupied = False 
                for obs in self.obstacles:
                    if obs.collidepoint(cell_pos):
                        is_occupied = True 
                        break 
                # If the cell of interest is not occluded, update the occupancy value 
                #  based on (noisy) sensor reading 
                if not is_occluded:
                    self.grid_distances[j] = distance
                    # Adding noise to occupancy value: with chance OCCUPANCY_ERROR,
                    #  the occupancy value is flipped 
                    if np.random.rand() < OCCUPANCY_ERROR:
                        # Flip occupancy value
                        is_occupied = not is_occupied
                    # Based on occupancy sensor measurement, update occupancy log-odds: 
                    if is_occupied:
                        self.occupancy[j] += OCCUPANCY_LOG_ODDS
                        self.detected_occupancy[j] = 1
                    else:
                        self.occupancy[j] -= OCCUPANCY_LOG_ODDS
                        self.detected_occupancy[j] = 0

    # Kalman filter ----------------------------------------------------------------------------------------------
    # Helpers ----------------------------------------------------------------------------------------------------
    
    def is_colliding(self, xy=None):
        """
        Checks if a given position, (x, y), which might be the actual or a hypothetical position
        of the robot, collides with any obstacles
        """
        if xy is None:
            x, y = self.spawn_x + self.robot_x, self.spawn_y + self.robot_y 
        else:
            x, y = xy 
        robot_rect = pygame.Rect(x-ROBOT_RADIUS, y-ROBOT_RADIUS, ROBOT_RADIUS*2, ROBOT_RADIUS*2)
        for obs in self.obstacles:
            if robot_rect.colliderect(obs):
                return True
        return False

    def print_sensors(self):
        for i, (sx, sy, hit, distance) in enumerate(self.sensors):
            # sx, sy are sensor positions relative to the robot
            # to get absolute ones add spawn_x, spawn_y respectively 
            status = "HIT" if hit else "CLEAR"
            if hit: 
                print(f"Sensor {i+1}: {status} at ({sx:.1f}, {sy:.1f}), distance {distance}")
                # print(f"Sensor {i+1}: {status} at ({sx+spawn_x:.1f}, {sy+spawn_y:.1f}), distance {distance}")

    # Drawing ----------------------------------------------------------------------------------------------------
    
    def draw_obstacles(self):
        if not obstacles_visible:
            return 
        for obs in self.obstacles: 
            pygame.draw.rect(screen, OBSTACLE_COLOR, obs) 
    
    def draw_beacons(self):
        if not beacon_visible:
            return 
        for i, ((beacon_x, beacon_y), distance) in enumerate(zip(self.beacons, self.beacon_distances)):
            color = (0, 0, 0)
            if distance < math.inf:
                color = ACTIVE_BEACON_COLOR
                # Draw circle around beacon with radius = distance to robot 
                #  Since we know this is the distance between the robot and the beacon, the robot 
                #  must be somewhere on that circle (or close to it, due to noise)
                pygame.draw.circle(screen, color, (beacon_x, beacon_y), distance, 1)
            else:
                color = PASSIVE_BEACON_COLOR
            pygame.draw.circle(screen, color, (beacon_x, beacon_y), 3)
            '''
            # Draw beacon id 
            beacon_text = font.render(str(i), True, color)
            beacon_text_rect = beacon_text.get_rect(center=(beacon_x, beacon_y))
            screen.blit(beacon_text, beacon_text_rect)
            '''
    
    def draw_grid(self):
        if not grid_visible:
            return 
        # Draw occupancy values 
        # Instead of plotting the probability, I will plot the normalized log-odds, because then
        #  differences are more visible
        # p = 1-1/(1+np.exp(self.occupancy))
        # q = (self.occupancy-min(self.occupancy))/(max(self.occupancy)-min(self.occupancy))
        # Use constant min (-500) and max (500) values for q:
        q = (np.clip(self.occupancy, -500, 500)-(-500))/(500-(-500))
        color_steepness = 2 
        for i, cell_x in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
            for j, cell_y in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
                # Using inverse sigmoid function to map log-odds to nice colors values
                #  This exaggerates the differences between occupancy values close to 0 and 1, 
                #  because most of the time, occupancy values are actually very close to 0 or 1
                color = int(255 * np.clip(1/2-1/color_steepness*np.log(1/q[i*NUM_GRID_CELLS+j]-1), 0, 1))
                pygame.draw.rect(screen, (0, color//4, color), (cell_x, cell_y, GRID_CELL_SIZE, GRID_CELL_SIZE))
                # Occupancy value as probability
                if occupancy_values_visible:
                    cell_text = font_small.render(str(int(self.occupancy[i*NUM_GRID_CELLS+j])), True, (255, 255, 255))
                    cell_text_rect = cell_text.get_rect(center=(cell_x+GRID_CELL_SIZE//2, cell_y+GRID_CELL_SIZE//2))
                    screen.blit(cell_text, cell_text_rect)
        # Draw grid lines
        for i in range(0, SCREEN_SIZE, GRID_CELL_SIZE):
            pygame.draw.line(screen, (50, 50, 50), (i, 0), (i, SCREEN_SIZE))
            pygame.draw.line(screen, (50, 50, 50), (0, i), (SCREEN_SIZE, i))
        # Highlight grid cells that are in range
        for i, cell_x in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
            for j, cell_y in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
                distance = self.grid_distances[i*NUM_GRID_CELLS+j]
                if distance < math.inf:
                    # Draw line from robot to grid cell center
                    '''
                    pygame.draw.line(screen, (255, 255, 0), (self.spawn_x+self.robot_x, self.spawn_y+self.robot_y),
                                                            (cell_x+GRID_CELL_SIZE//2, cell_y+GRID_CELL_SIZE//2), 1)
                    '''
                    if self.detected_occupancy[i*NUM_GRID_CELLS+j]:
                        # In range and occupancy is detected
                        pygame.draw.rect(screen, (255, 255, 0), (cell_x, cell_y, GRID_CELL_SIZE, GRID_CELL_SIZE))
                    else:
                        # In range and NO occupancy is detected
                        pygame.draw.rect(screen, (255, 255, 0), (cell_x, cell_y, GRID_CELL_SIZE, GRID_CELL_SIZE), 1)

    def draw_robot(self):
        # Draw trajectory (past positions)
        if trajectory_visible:
            points = [(self.spawn_x+x, self.spawn_y+y) for x, y in self.trajectory]
            pygame.draw.lines(screen, (90, 30, 30), False, points, 3)
        # Get robot's current (absolute) position
        x, y = self.spawn_x + self.robot_x, self.spawn_y + self.robot_y 
        if robot_info_visible:
            '''
            # Draw sensors
            for sensor_x, sensor_y, hit, _ in self.sensors:
                color = SENSOR_HIT_COLOR if hit else SENSOR_COLOR
                pygame.draw.line(screen, color, (x, y), (self.spawn_x+sensor_x, self.spawn_y+sensor_y), 2)
                # pygame.draw.circle(screen, color, (self.spawn_x+sensor_x, self.spawn_y+sensor_y), 3)
            '''
            # Draw omnidirectional sensor (circle around robot indicating OMNI_RANGE)
            pygame.draw.circle(screen, ROBOT_COLOR, (x, y), OMNI_RANGE, 2)
            # Draw lines from robot to detected beacons 
            if beacon_visible:
                for (beacon_x, beacon_y), distance, bearing in zip(self.beacons, self.beacon_distances, self.beacon_bearings):
                    if distance < math.inf:
                        pygame.draw.line(screen, ACTIVE_BEACON_COLOR, (x, y), (beacon_x, beacon_y), 2)
                        '''
                        # Draw beacon distances
                        #  I'm dividing by SENSOR_STEP_SIZE so we get the same units as the sensor distances 
                        distance_text = font.render(str(int(distance//SENSOR_STEP_SIZE)), True, (255, 255, 255))
                        distance_text_rect = distance_text.get_rect(center=(beacon_x+10, beacon_y+10))
                        screen.blit(distance_text, distance_text_rect)
                        '''
                        '''
                        # Draw beacon bearings
                        bearing_text = font.render(str(int(np.rad2deg(2*pi-bearing)))+"°", True, (255, 255, 255))
                        bearing_text_rect = bearing_text.get_rect(center=(beacon_x+10, beacon_y+10))
                        screen.blit(bearing_text, bearing_text_rect)
                        '''
        # Draw the robot body (disk)
        pygame.draw.circle(screen, ROBOT_COLOR, (x, y), ROBOT_RADIUS)
        # Draw the direction line (heading)
        heading_x = x + ROBOT_RADIUS * math.cos(self.robot_angle)
        heading_y = y + ROBOT_RADIUS * math.sin(self.robot_angle)
        pygame.draw.line(screen, (0, 0, 0), (x, y), (heading_x, heading_y), 3)
        # Draw estimated position according to trilateration 
        if beacon_visible and self.observation is not None:
            est_x, est_y, _ = self.observation 
            pygame.draw.circle(screen, (255, 255, 255), (est_x, est_y), 3)
        '''
        # Draw motor speed text inside the robot body
        if robot_info_visible:
            # vel_text = font.render(f"[{self.v_l}, {self.v_r}]", True, (255, 255, 255))
            # screen.blit(vel_text, (x-11, y-6))  
            for i, (_, _, hit, distance) in enumerate(self.sensors):
                # Draw sensor distance values around the robot (text)
                sensor_angle = self.robot_angle + i * SENSOR_ANGLE_STEP
                text_x = x + (ROBOT_RADIUS + 20) * math.cos(sensor_angle)  # Position outside the robot
                text_y = y + (ROBOT_RADIUS + 20) * math.sin(sensor_angle)  
                # Only draw if finite distance 
                if distance < math.inf:
                    distance_text = font.render(str(int(distance)), True, (255, 255, 255) if hit else (100, 100, 100))
                    text_rect = distance_text.get_rect(center=(text_x, text_y))
                    screen.blit(distance_text, text_rect)
        '''

    def draw_belief(self):
        mean_x, mean_y, _ = self.belief_mean
        cov = self.belief_cov
        # compute ellipse width & height from cov diagonal
        n_std = 2
        w, h = 2 * n_std * np.sqrt(cov[0,0]), 2 * n_std * np.sqrt(cov[1,1])
        ellipse = pygame.Rect(mean_x - w/2, mean_y - h/2, w, h)
        pygame.draw.ellipse(screen, (0, 0, 255), ellipse, 2)

    def save_frame(self): 
        frame_surface = pygame.display.get_surface().copy()
        frame_array3d = pygame.surfarray.array3d(frame_surface) 
        # Only add if frame is not empty and different from last frame 
        if (frame_array3d.sum() > 0) and (self.prev_frame is None or (frame_array3d != self.prev_frame).any()):
            frame_array = pygame.image.tostring(frame_surface, "RGB")
            frame = Image.frombytes("RGB", frame_surface.get_size(), frame_array)
            self.frames.append(frame) 
            self.prev_frame = frame_array3d 
    
    def to_video(self):
        if len(self.frames) == 0: 
            return
        if not os.path.exists("screencapture"):
            os.mkdir("screencapture")
        timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"screencapture/main {timestamp}.mp4"
        print(f"Saving file as {filename}") 
        # Save first frame as image
        # with imageio.get_writer(f"screencapture/main {timestamp}.png") as writer: 
        #     writer.append_data(np.array(frames[1]))
        with imageio.get_writer(filename, fps=60, codec="libx264") as writer: 
            for frame in self.frames[1:]:
                writer.append_data(np.array(frame))
    
    def draw_slam_landmarks(self):
        # overlay EKF‐SLAM’s landmark estimates as red X’s
        M = len(self.beacons)
        for i in range(M):
            ex, ey = self.X[3+2*i], self.X[3+2*i+1]
            size = 4
            # draw a red cross at (ex, ey)
            pygame.draw.line(screen, (255, 0, 0), (ex-size, ey-size), (ex+size, ey+size), 2)
            pygame.draw.line(screen, (255, 0, 0), (ex-size, ey+size), (ex+size, ey-size), 2)


    def render(self):
        screen.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_obstacles()
        self.draw_beacons()
        self.draw_belief()
        self.draw_slam_landmarks()
        self.draw_robot()
        pygame.display.flip()


if __name__ == "__main__":
    # PyGame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Mobile Robot")
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont("Calibri", 12)
    font_small = pygame.font.SysFont("Calibri", 8)
    
    # Environment 
    env = Environment() 
    
    # Main loop
    running = True 
    while running:
        # env.print_sensors()

        # System controls
        mods = pygame.key.get_mods()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 
            elif event.type == pygame.KEYDOWN:
                match event.key: 
                    case pygame.K_ESCAPE: 
                        running = False 
                    case pygame.K_b:
                        # "B" key toggles beacon visibility 
                        beacon_visible = not beacon_visible
                        flag_changed = True
                    case pygame.K_g:
                        # "G" key toggles grid visibility
                        grid_visible = not grid_visible
                        flag_changed = True
                    case pygame.K_i:
                        # "I" key toggles robot info visibility 
                        robot_info_visible = not robot_info_visible
                        flag_changed = True
                    case pygame.K_o:
                        # "O" key toggles obstacle visibility
                        obstacles_visible = not obstacles_visible
                        flag_changed = True
                    case pygame.K_r:
                        # "R" resets the environment 
                        env.reset()
                    case pygame.K_s if (mods & (pygame.KMOD_META | pygame.KMOD_CTRL)):
                        # CMD + "S" key saves run so far as mp4 file 
                        env.to_video()
                    case pygame.K_t:
                        # "T" key toggles trajectory visibility
                        trajectory_visible = not trajectory_visible
                        flag_changed = True
                    case pygame.K_v:
                        # "V" key toggles occupancy values
                        occupancy_values_visible = not occupancy_values_visible
                        flag_changed = True
        
        # Robot controls 
        v_l, v_r = 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:  
            # "W" key controls the left motor
            v_l = 1
        if keys[pygame.K_UP]:  
            # UP arrow key controls the right motor
            v_r = 1
        # Move the robot according to (v_l, v_r)
        env.move((v_l, v_r))
    
    pygame.quit() 
