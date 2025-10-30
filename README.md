# Autonomous-Robotic-System
Autonomous robotic system simulating exploration, localization, mapping, and navigation using differential drive, Kalman Filter, EKF-SLAM, and a neural controller evolved via Natural Evolution Strategies (NES). Developed in Python (Pygame, NumPy, SciPy, Matplotlib) for the Autonomous Robotic Systems course.

This project implements a full **autonomous robot pipeline** — from simulation and localization to mapping and learned navigation — for a **differential-drive robot** in a 2D environment with obstacles.  
It was developed for the **Autonomous Robotic Systems (ARS)** course at **Maastricht University**.

---

## Project Overview

The robot autonomously explores an unknown environment, avoids obstacles, constructs an occupancy grid map, and finds a target (“rescue mission”) without human input.

The project was completed in **four milestones**:
1. **Simulation:** Differential-drive robot simulation with obstacle avoidance  
2. **Localization:** Pose estimation using a **Linear Kalman Filter**  
3. **Mapping:** Occupancy grid + **EKF-SLAM** for simultaneous localization and mapping  
4. **Navigation:** Learned neural controller trained via **Natural Evolution Strategies (NES)** for autonomous exploration

---

## System Components

| Module | Key Algorithm | Purpose |
|--------|----------------|----------|
| **Simulation** | Differential-drive kinematics | Robot motion & sensor data |
| **Localization** | Kalman Filter | Estimate robot pose under noise |
| **Mapping** | Occupancy grid, EKF-SLAM | Build map & reduce pose uncertainty |
| **Navigation** | RNN + NES optimization | Learn control policy for exploration |

---

## Technical Details

### Simulation
- Robot represented as a circle with **12 distance sensors** (30° spacing)  
- **Spawn validation** prevents initialization on obstacles  
- Manual testing with motor control (W + arrow keys)

### Localization (Kalman Filter)
- Combines **velocity-based motion model** and **trilateration landmarks**
- Corrects robot pose when ≥3 landmarks are visible
- Experiments on **sensor and motion noise (Q, R matrices)**

### Mapping (EKF-SLAM)
- Occupancy grid for static obstacle mapping  
- EKF-SLAM integrates localization & mapping  
- Occlusion handling and noise-sensitive occupancy probabilities

### Navigation (Neural Controller + NES)
- **Input:** 117 features (pose, sensors, coverage grid, hidden state)  
- **Output:** Δθ (turning angle)  
- Optimized using **Natural Evolution Strategies**  
- Fitness: maximize explored area, minimize collisions

---

## Tools & Libraries
- **Python 3.9+**
- `pygame` – visualization and simulation  
- `numpy`, `scipy` – math, trilateration, optimization  
- `matplotlib` – plotting  
- `random`, `math` – environment generation  

---

## Experiments & Results
- Robust spawn handling under random obstacle maps  
- Kalman filter localization stable under sensor noise  
- EKF-SLAM maps converge to real landmarks  
- NES-trained controller achieves reliable exploration behavior after ~400 generations  

---

## Example Outputs
- **Occupancy Grid Maps** with variable sensor noise  
- **EKF-SLAM Visualization:** landmark convergence  
- **NES Training Curves:** fitness and diversity evolution  


## References
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics* – MIT Press  
- Wierstra et al. (2011). *Natural Evolution Strategies*  
- Tipaldi & Burgard (2015). *Occupancy Mapping*  
- Pygame & SciPy documentation  

---

## 💬 Summary
This project demonstrates a fully functional **autonomous exploration and mapping pipeline** built from scratch, integrating **classical robotics algorithms** with **neural evolution-based control** for intelligent, adaptive navigation in unknown environments.
