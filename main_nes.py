# Stripped down, optimized code for faster evolution 
import os 
import pygame
import math 
from math import pi, e, inf 
import numpy as np
import random 
from time import time 
from trilaterate import trilaterate
from scipy.spatial.distance import pdist


# True map
NUM_CELLS = 20  # number of "world cells" along each axis
CELL_SIZE = 40 
OBSTACLE_DENSITY = 0.2  # 0.2
# Screen 
SCREEN_SIZE = NUM_CELLS*CELL_SIZE 
# Predicted map 
NUM_GRID_CELLS = 10  # number of "grid cells" along each axis, e.g. for occupancy grid 
assert SCREEN_SIZE % NUM_GRID_CELLS == 0, "SCREEN_SIZE must be divisible by NUM_GRID_CELLS"
GRID_CELL_SIZE = SCREEN_SIZE//NUM_GRID_CELLS
# Robot
ROBOT_RADIUS = 10
MOVE_SPEED = 3  # linear speed factor 
ROTATE_SPEED = 1  # angular speed factor 
# Robot sensors 
NUM_SENSORS = 8
SENSOR_RANGE = 100 
SENSOR_ANGLE_STEP = 2*pi/NUM_SENSORS
SENSOR_STEP_SIZE = 10  # 4  # ideally, SENSOR_RANGE divisible by it 
OMNI_RANGE = SENSOR_RANGE 
OCCUPANCY_ERROR = 0.001  # 0.01, 0.001
OCCUPANCY_LOG_ODDS = np.log((1-OCCUPANCY_ERROR)/OCCUPANCY_ERROR)  # log-odds of occupancy
# Policy 
ACTION_FREQUENCY = 1/4  # how often to call the policy to take an action 
# Evolution 
NUM_GENERATIONS = 1000 
NUM_OFFSPRING = 24  # λ
assert NUM_OFFSPRING % 2 == 0, "NUM_OFFSPRING must be divisible by 2"
NUM_EVAL_EPISODES = 8  # 10
EPISODE_LENGTH = 1800  # 2000
NUM_HIDDEN_UNITS = 4  # 4 
NUM_OUTPUTS = 1  # control outputs
# NUM_TIMESTEPS_PER_EPISODE = 200
LR_MEAN = 0.2  # 0.1, 0.05
LR_LOGSTD = 0.2  # 0.1, 0.05
# Noise
R = np.diag([1e-6, 1e-6, 1e-6])  # motion-noise covariance


class Evolution:

    # Initialization ---------------------------------------------------------------------------------------------
    
    def __init__(self):
        """
        Initialize the environment by resetting all map and robot settings, 
        generating an ID and initializing the evolution procedure.
        """
        self.reset()
        self.initialize_id()
        self.initialize_evolution()

    def reset(self, reset_map=True):
        """
        Reset the map and robot, but not the evolution or 
        This method can be called for each new offspring to generate a new map and robot,
        without resetting the ID or evolutionary history. 
        """
        self.timestep = 0 
        if reset_map:
            self.initialize_map()
            self.initialize_target()
        self.initialize_robot()
        self.initialize_grid()

    def initialize_id(self):
        """
        Read in the "history" directory what the most recent ID was and simply increment it
        to obtain the ID for this run. 
        """
        past_ids = [int(f[:f.index("_")]) for f in os.listdir("history") 
                    if ".npy" in f and f[:f.index("_")].isnumeric()]
        most_recent_id = max([0, *past_ids])
        self.evol_id = most_recent_id + 1 
    
    def initialize_map(self):
        """
        Generates a map with randomly placed blocks.
        """
        self.obstacle_grid = np.zeros((NUM_CELLS, NUM_CELLS), dtype=int)  # grid of obstacles
        self.obstacles = []
        # Useful to define the rectangle at the center of the map
        center_rect = pygame.Rect(SCREEN_SIZE/2-2*CELL_SIZE/2, SCREEN_SIZE/2-2*CELL_SIZE/2, 2*CELL_SIZE, 2*CELL_SIZE)
        # Calculate the number of obstacles based on density 
        n_obstacles = int((SCREEN_SIZE**2)/(CELL_SIZE**2)*OBSTACLE_DENSITY)
        while len(self.obstacles) < n_obstacles:
            i = random.randint(0, (SCREEN_SIZE//CELL_SIZE)-1)
            j = random.randint(0, (SCREEN_SIZE//CELL_SIZE)-1)
            x = i*CELL_SIZE
            y = j*CELL_SIZE
            # Create the obstacle rectangle
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            # Avoid overlapping obstacles and don't place obstacles in center
            if not (rect.colliderect(center_rect) or any(rect.colliderect(obstacle) for obstacle in self.obstacles)):
                self.obstacles.append(rect)
                self.obstacle_grid[i,j] = 1
        # Add boundary walls to fully enclose the map
        self.obstacles.extend([
            pygame.Rect(0, 0, SCREEN_SIZE, 5),              # Top boundary
            pygame.Rect(0, 0, 5, SCREEN_SIZE),              # Left boundary
            pygame.Rect(0, SCREEN_SIZE-5, SCREEN_SIZE, 5),  # Bottom boundary
            pygame.Rect(SCREEN_SIZE-5, 0, 5, SCREEN_SIZE)   # Right boundary
        ])
    
    def initialize_robot(self):
        """
        Initialize robot's initial pose, initial sensor readings, initial action, 
        the robot's trajectory, and robot-related statistics, like the collision counter 
        and total distance traveled. 
        """
        # Pose 
        # Important: robot_x, robot_y store the robot's relative position!
        #  To get the absolute position, add the spawn position (spawn_x, spawn_y)
        self.robot_x = 0
        self.robot_y = 0
        self.robot_angle = np.random.uniform(0, 2*pi)  # orientation (in radians)
        # Find a good spawn point 
        self.spawn_x = SCREEN_SIZE/2  # x-position
        self.spawn_y = SCREEN_SIZE/2  # y-position 
        # Make sure the robot is not placed on top of or inside an obstacle: 
        while self.is_colliding():
            # Reposition the robot's spawn randomly within the map bounds:
            self.spawn_x = random.randint(ROBOT_RADIUS, SCREEN_SIZE-ROBOT_RADIUS)
            self.spawn_y = random.randint(ROBOT_RADIUS, SCREEN_SIZE-ROBOT_RADIUS)
        # List to keep track of robot's trajectory (position in each timestep)
        self.trajectory = [(0, 0)]
        # Sensors
        # Get initial sensor reading 
        self.compute_sensors()
        # Just for visualization, also keep track of these values inside self
        # (rather than just passing them as parameters to the move method)
        self.v_l = 0
        self.v_r = 0
        # Keep track of certain values for evolutionary fitness calculation 
        self.d_angle = 0
        self.sum_d_angle = 0
        self.collision_counter = 0
        self.distance_traveled = 0 
    
    def initialize_grid(self):
        """
        Initialize the occupancy grid and variables used to track the occupancy grid mapping, 
        like the robot's distance to each grid cell within sensor range. 
        """
        self.grid = np.zeros((NUM_GRID_CELLS, NUM_GRID_CELLS, 2))  # coordinates of grid cell centers 
        self.grid_distances = inf*np.ones((NUM_GRID_CELLS, NUM_GRID_CELLS))  # current distances from robot to grid cell centers
        self.occupancy = np.zeros((NUM_GRID_CELLS, NUM_GRID_CELLS))  # log-odds of occupancy, prior is p=0.5 -> log(0.5/(1-0.5))=0
        self.detected_occupancy = np.zeros((NUM_GRID_CELLS, NUM_GRID_CELLS))  # 0 if not detected, 1 if detected (might be wrong due to sensor noise)
        for i, cell_x in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
            for j, cell_y in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
                cell_center_x, cell_center_y = cell_x+GRID_CELL_SIZE//2, cell_y+GRID_CELL_SIZE//2  # grid cell center coordinates
                self.grid[i,j] = (cell_center_x, cell_center_y)

    def initialize_target(self):
        """
        Generate a random position in the map (only free cells and not in the map center area), 
        and place the target (rescuee) there.
        """
        obstacle_grid = self.obstacle_grid.copy()
        # Let's block the center region so the target is somewhere further away from the center 
        center = NUM_CELLS//2 
        halfway = center//2 
        obstacle_grid[center-halfway:center+halfway,center-halfway:center+halfway] = 1 
        # Random cell in NUM_CELLS x NUM_CELLS grid that is not occupied by an obstacle 
        free_cells = np.argwhere(1-obstacle_grid)
        idx = np.random.randint(len(free_cells))
        target_i, target_j = free_cells[idx]
        # Get coordinates of cell center and save as target_x, target_y
        self.target_x = target_i*CELL_SIZE + CELL_SIZE//2
        self.target_y = target_j*CELL_SIZE + CELL_SIZE//2

    def initialize_evolution(self):
        """
        Initialize all the parameters necessary for the evolutionary procedure. 
        """
        self.current_generation = 0
        self.current_eval_episode = 0
        self.current_offspring = 0
        self.hidden_state = np.zeros(NUM_HIDDEN_UNITS)
        self.num_features = len(self.compute_features())
        print(self.num_features, "features")
        self.num_weights_by_layer = (NUM_HIDDEN_UNITS * (self.num_features+1), NUM_OUTPUTS * (NUM_HIDDEN_UNITS+1))
        self.num_weights = sum(self.num_weights_by_layer)
        print(self.num_weights, "weights")
        self.w = np.zeros(self.num_weights) 

    # Step: transition to the next state -------------------------------------------------------------------------
    
    def step(self, action):
        """
        Transition from current environment state to the next 
        """
        self.compute_sensors()
        self.compute_occupancy()
        self.timestep += 1
    
    # Motion -----------------------------------------------------------------------------------------------------

    def move(self, action=None):
        """
        Parameter 'action' must be a tuple containing (v_l, v_r)
        Move robot at velocities v_l (left motor), v_r (right motor) 
        Ordinarily, v_l, v_r should be between -1 and 1 (they are then multiplied by respective SPEED constant)
        """
        if self.timestep > 0 and action == (0, 0):
            # If there is no movement, there is no need to calculate new pose
            #  Unless it's timestep 0 or the user toggled the visibility of something 
            #  This also means that, if no action is taken, time doesn't pass
            return 
        v_l, v_r = action
        # Keep track of v_l, v_r so we can put it on screen as text 
        #  (see draw_robot method)
        self.v_l, self.v_r = v_l, v_r 
        # old_angle = self.robot_angle
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
                self.collision_counter += 1
                # print([self.collision_counter], "Collision detected!")
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
        self.step(action) 

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
            distance = inf 
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

    def compute_occupancy(self):
        """
        Checks which grid cells are in range of the robot's omnidirectional sensor 
        Updates grid_distances with the distances to the detected grid cells
        """
        self.grid_distances = inf * np.ones((NUM_GRID_CELLS, NUM_GRID_CELLS))
        x, y = self.spawn_x+self.robot_x, self.spawn_y+self.robot_y
        robot_pos = np.array([x, y])
        for i in range(NUM_GRID_CELLS): 
            for j in range(NUM_GRID_CELLS):
                cell_x, cell_y = self.grid[i,j]
                cell_pos = np.array([cell_x, cell_y])  # cell center point
                distance = np.linalg.norm(self.grid[i,j]-robot_pos)  # distance from robot to cell center point
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
                        self.grid_distances[i,j] = distance
                        # Adding noise to occupancy value: with chance OCCUPANCY_ERROR,
                        #  the occupancy value is flipped 
                        if np.random.rand() < OCCUPANCY_ERROR:
                            # Flip occupancy value
                            is_occupied = not is_occupied
                        # Based on occupancy sensor measurement, update occupancy log-odds: 
                        if is_occupied:
                            self.occupancy[i,j] += OCCUPANCY_LOG_ODDS
                            self.detected_occupancy[i,j] = 1
                        else:
                            self.occupancy[i,j] -= OCCUPANCY_LOG_ODDS
                            self.detected_occupancy[i,j] = 0
    
    # Navigation -------------------------------------------------------------------------------------------------
    
    def load_weights(self, evol_id=None):
        """
        Load weights from file corresponding to the given evol_id.
        """
        if evol_id is None:
            evol_id = self.evol_id-1
        print("ID:", evol_id)
        self.w = np.load(f"weights/{evol_id}.npy")

    def angle_difference_to_velocities(self, d_angle):
        """
        Convert d_angle (e.g. from policy network) to (v_l, v_r) so it can be passed
        to the move method. 
        """
        v_linear = 1.0  # constant forward speed
        v_l = (v_linear - ROBOT_RADIUS * ROTATE_SPEED * d_angle) / MOVE_SPEED
        v_r = (v_linear + ROBOT_RADIUS * ROTATE_SPEED * d_angle) / MOVE_SPEED
        return v_l, v_r

    def move_autonomously(self, w=None):
        """
        This method calls the compute_features method and the policy network 
        in those timesteps in which an action is applied (depending on the action frequency). 
        """
        # Compute new action every 1/freq timesteps 
        if self.timestep % int(1/ACTION_FREQUENCY) == 0:
            if w is None:
                w = self.w 
            features = self.compute_features()
            control_signals, hidden_state = self.nn(features, w) 
            d_angle = control_signals[0] 
            self.d_angle = d_angle
            self.sum_d_angle += abs(d_angle) 
            self.hidden_state = hidden_state 
        else:
            # Use the last action
            d_angle = self.d_angle 
        action = self.angle_difference_to_velocities(d_angle) 
        self.move(action)
    
    def nn(self, features, w):
        """
        Recurrent neural network that computes the policy (state-to-action mapping)
        """
        # Extracting the layers' weight matrices, W0 and W1, from the flattened weight vector, w 
        n0, n1 = self.num_weights_by_layer
        W0 = w[:n0].reshape((NUM_HIDDEN_UNITS, self.num_features+1)) 
        W1 = w[n0:n0+n1].reshape((1, NUM_HIDDEN_UNITS+1))
        # Feedforward
        features = np.concatenate(([1.0], features))            # prepend constant feature to allow bias term
        hidden_state = (W0 @ features.T).clip(0)                # relu nonlinearity 
        _hidden_state = np.concatenate(([1.0], hidden_state))   # prepend constant feature to allow bias term
        d_angle = (W1 @ _hidden_state.T).clip(-pi/4, pi/4)      # range [-π/4, π/4], i.e. [-45°, 45°]
        return d_angle, hidden_state 

    def compute_features(self):
        """
        Compute feature vector that is used as input to the policy
        The feature vector is a concatenation of the following features:
         - Robot position (robot_x, robot_y)
         - Robot orientation (cos(robot_angle), sin(robot_angle))
         - Sensor readings (proximity to obstacles, per sensor)
         - Occupancy probabilities (per grid cell)
        (- Target position (target_x, target_y))
        """
        # Sensor features (proximity)
        dist = np.array([distance for _, _, _, distance in self.sensors]).clip(0, 2*SENSOR_RANGE)  # inf -> 2*SENSOR_RANGE
        prox = 1-(dist/(SENSOR_RANGE//SENSOR_STEP_SIZE))-0.5  # -0.5 to center non-inf entries around 0 
        # Occupancy features 
        # occ = self.obstacle_grid.flatten() 
        occ_probs = 1-1/(1+np.exp(self.occupancy.flatten())) 
        occ = (2*np.abs(occ_probs-0.5) > 0.9).astype(float)  # not just touched lightly, but actually visited -> high absolute cell entry 
        # occ = (self.occupancy != 0).astype(float).flatten()  # simplify occupancy grid to binary "coverage map" 
        # Robot pose 
        x, y, angle = self.robot_x, self.robot_y, self.robot_angle 
        # x, y, angle = self.belief_mean
        pos_min, pos_max = -SCREEN_SIZE/2, SCREEN_SIZE/2
        x = (x-pos_min)/(pos_max-pos_min)
        y = (y-pos_min)/(pos_max-pos_min)
        # Putting it all in a feature vector  
        features = np.array([
            x,
            y,
            np.cos(angle),
            np.sin(angle),
            *prox,
            *self.hidden_state, 
            self.d_angle,
            *occ,
        ])
        # print(features)
        return features 

    # Evolution --------------------------------------------------------------------------------------------------

    def evolve(self): 
        """
        Natural evolution strategy (NES) algorithm 
        Loops over specified number of generations, and all offspring per generation, 
        calls fitness evaluation method, and applied the parameter according to the NES algorithm.
        It saves the statistics and final weights of this run. 
        """
        print("ID:", self.evol_id)
        evolution_start_time = time()

        # Search distribution 
        search_mean = self.w  # "parent mean"
        search_logstd = -1.5 * np.ones(len(search_mean))  # (log std, std): (-2, 0.14), (-1, 0.37), (-1.5, 0.22)
        
        f = np.empty(NUM_OFFSPRING)  # fitness scores 
        loggrad_mean = np.empty((NUM_OFFSPRING, self.num_weights))
        loggrad_logstd = np.empty((NUM_OFFSPRING, self.num_weights))
        
        history_mean = []
        history_max = []
        history_diversity = []
        history_std = []
        
        # Run evolution
        for g in range(NUM_GENERATIONS):
            print("Generation", g)
            print(" ", [f"Current runtime: {round(time()-evolution_start_time, 1)} s"])
            self.current_generation = g 
            search_std = np.exp(search_logstd)
            # Create and evaluate offspring 
            offspring = []
            i = 0
            for _ in range(NUM_OFFSPRING//2):
                # Mirror sampling:
                noise = np.random.randn(self.num_weights)
                for j in [-1, 1]:
                    start_time = time()
                    print("  Offspring", i, end=" ") 
                    self.current_offspring = i 
                    self.reset() 
                    # Sample 
                    w = search_mean + search_std * j * noise 
                    offspring.append(w)
                    # Evaluate 
                    f[i] = self.evaluate(w) 
                    # Log-gradients of search parameters 
                    loggrad_mean[i] = (w-search_mean)/np.multiply(search_std, search_std)
                    loggrad_logstd[i] = np.multiply(w-search_mean, w-search_mean)/np.multiply(search_std, search_std)-1
                    print(f"{round(time()-start_time, 1)} s")
                    i += 1
            
            # Normalize fitness scores
            unnorm_f = f
            f = (f-np.mean(f))/(np.std(f)+1e-8)
            
            # Gradients of expected fitness
            grad_J_mean = (f[:, None] * loggrad_mean).mean(axis=0)
            grad_J_logstd = (f[:, None] * loggrad_logstd).mean(axis=0)
            
            # Fisher matrices
            fisher_mean = np.diag((loggrad_mean ** 2).mean(axis=0))
            fisher_logstd = np.diag((loggrad_logstd ** 2).mean(axis=0))
            
            # Update search parameters 
            search_mean += LR_MEAN * (np.linalg.inv(fisher_mean) @ grad_J_mean)
            search_logstd += LR_LOGSTD * (np.linalg.inv(fisher_logstd) @ grad_J_logstd)

            std = np.exp(search_logstd) 
            # Entropy as a diversity score
            # diversity = 0.5 * np.sum([np.log(2*pi*e*std[i]**2) for i in range(self.num_weights)])
            # Average pairwise distance as diversity score
            diversity = pdist(np.stack(offspring)).mean() 

            # Track statistics 
            history_mean.append(unnorm_f.mean())
            history_max.append(unnorm_f.max())
            history_diversity.append(diversity)
            history_std.append(std.mean())
            # Save statistics
            np.save(f"history/{self.evol_id}_mean.npy", history_mean)
            np.save(f"history/{self.evol_id}_max.npy", history_max)
            np.save(f"history/{self.evol_id}_diversity.npy", history_diversity)
            np.save(f"history/{self.evol_id}_std.npy", history_std)
            # Save weights of best candidate of this generation 
            best_candidate = offspring[f.argmax()]
            np.save(f"weights/{self.evol_id}.npy", best_candidate) 

    def evaluate(self, w): 
        """
        Fitness function evaluation
        Runs NUM_EVAL_EPISODES, extracts the relevant statistics, averages them across these episodes,
        and returns the corresponding fitness score for the given weight vector w.
        """
        # Run one or multiple evaluation episodes 
        # total_sensor_distance = 0.0 
        # sensor_max_squared_distance = 0.0
        for ep in range(NUM_EVAL_EPISODES):
            self.current_eval_episode = ep 
            while True:
                self.move_autonomously(w) 
                # sensor_squared_distances = [min(dist, SENSOR_RANGE//SENSOR_STEP_SIZE)**2 for _, _, _, dist in self.sensors] 
                # total_sensor_distance += sum(sensor_squared_distances) 
                # sensor_max_squared_distance += max([min(dist, SENSOR_RANGE//SENSOR_STEP_SIZE)**2 for _, _, _, dist in self.sensors] )
                if self.timestep >= EPISODE_LENGTH:
                    break 
        # Use data from evaluation episode to compute fitness score 
        # displacement_squared = self.robot_x**2 + self.robot_y**2  # squared distance between final position and spawn position 
        # displacement = np.sqrt(displacement_squared)
        num_visited_cells = (self.occupancy.flatten() != 0).sum() 
        # share_visited_cells = num_visited_cells/len(self.occupancy.flatten())
        fitness_components = [
            # 5.00 * displacement,             # go as far from starting point as possible
            1.00 * num_visited_cells,          # visit as many distinct grid cells as possible 
            0.001 * -self.collision_counter,   # collide as little as possible
            # 0.01 * -self.sum_d_angle,        # move (rotate) as little as possible 
            # 1.00 * -np.sum(w**2),            # L2 regularization
        ]
        score = 1/(EPISODE_LENGTH*NUM_EVAL_EPISODES) * sum(fitness_components)
        print("\t", [round(float(component), 4) for component in fitness_components], "\t", [round(float(score), 4)], end="\t")
        return score 

if __name__ == "__main__":
    # PyGame
    os.environ["SDL_VIDEODRIVER"] = "dummy" 
    pygame.init() 

    # Evolution environment 
    env = Evolution() 
    # env.load_weights()  # load most recent (or particular) weights to continue evolution from there 
    env.evolve()
    
    pygame.quit() 
