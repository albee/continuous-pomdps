"""
A double integrator as a POMDP with a mass parameter estimator to boot!

S: 2D state space, continuous
A: 2D bounded x and y thrust
T: stochastic Gaussian weighted double integrator dynamics, discrete
R: user-defined
O: Gaussian weighted measurement, but can modify to include effect of EKF
Z: mass
gamma
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon
from shapely.affinity import translate

class DoubleInt:
    def __init__(self, x_min, y_min, x_max, y_max, offsets, obstacles, goals, dt, init_belief):
        self.mass_mean = 1.0
        self.mass_covar = 0.1
        self.mass_actual = 1.0
        self.dims = [x_min, x_max, y_min, y_max]

        self.state = init_belief.mean # initial condition, [x; y; theta; xdot; ydot; thetadot]
        self.param = [self.mass_mean]

        self.dt = dt
        self.A = np.vstack((
                    np.array([
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]),
                    np.zeros((2,4))
                ))

        self.B = np.vstack((
                    np.array([
                    [1/self.mass_mean*dt, 0],
                    [0, 1/self.mass_mean*dt],
                    [1/self.mass_mean, 0],
                    [0, 1/self.mass_mean]]) 
                ))

        # Self geometry
        self.offsets = offsets  # 2D geometry offsets (minx, miny, maxx, maxy)
        self.geometry = box(self.state[0]+offsets[0], self.state[1]+offsets[1], self.state[0]+offsets[2], self.state[1]+offsets[3])
        self.obstacles = obstacles
        self.goals = goals

        # For plotting
        self.last_robot = None

        self.gamma = 0.99

    # Reach the goal, but learn too. Call after transition.
    def reward(self, last_state, action):
        reward = 0
        if self.goal_check():
            reward = 1000
        elif self.collision_check():
            reward = -50
        else:
            reward = -1
        if not self.in_bounds():
            reward = -10

        return reward

    def set_state_and_get_reward(self, state, action, state_next):
        self.state = state_next
        return self.reward(state, action)

    # Return the stochastic observation of the state variables. Parameter is NOT directly observed.
    def observe_stoc(self):
        var = 0.01
        z = np.add(self.state, np.random.normal(0, var, 4).reshape(-1,1))
        return z, 0

    # Creates a feasible action (within thruster constraints)
    def generate_action(self):
        u_min = -5.0
        u_max = 5.0
        diff = u_max-u_min
        rand_action = np.random.rand(2,1)*(diff) - diff/2
        return rand_action

    # Transition the state, return new state
    def dynamics_prop(self, uk, certain=True):
        var = 0.01
        A = self.A; B = self.B; xk = self.state; dt = self.dt

        if certain == True:
            dx = (np.matmul(A, xk) + np.matmul(B, uk))*dt  # Euler integration
            xk1 = xk + dx
        else:
            dx = (np.matmul(A, xk) + np.matmul(B, uk))*dt  # Euler integration
            xk1 = np.add(xk + dx, np.random.normal(0, var, 4).reshape(-1,1))
        return xk1, dx

    # Transition the state and update the internal value
    def transition_certain(self, uk): 
        xk = self.state    
        xk1, dx = self.dynamics_prop(uk, True) 
        self.state = xk1
        
        # Evaluate reward
        reward = self.reward(xk, uk)

        # Goal and obstacle checks
        if self.collision_check():
            self.state = xk
            return xk
        self.geometry = translate(self.geometry, xoff=dx[0], yoff=dx[1])  # update the robot geometry position
        if self.goal_check():
            return None  # signals complete
        return xk1

    # Transition the state and update the internal value, uncertain
    def transition_stoc(self, uk):   
        xk = self.state    
        xk1, dx = self.dynamics_prop(uk, False) 
        self.state = xk1

        # Evaluate reward
        reward = self.reward(xk, uk)

        # Goal and obstacle checks
        if self.collision_check():
            self.state = xk
            return self.state, reward, self.param
        self.geometry = translate(self.geometry, xoff=dx[0], yoff=dx[1])  # update the robot geometry position

        return self.state, reward, self.param

        # Can add in "action" to resolve uncertainty?

    def set_state_and_simulate(self, state, action):
        self.state = state
        self.geometry = box(self.state[0]+self.offsets[0], self.state[1]+self.offsets[1], self.state[0]+self.offsets[2], self.state[1]+self.offsets[3])
        state_next, reward, _ = self.transition_stoc(action)
        obs, _ = self.observe_stoc()
        return state_next, obs, reward

    """
    Collision checking and plotting
    """

    def init_plot(self):
        # Unpack state bounds
        x_min = self.dims[0]
        x_max = self.dims[1]
        y_min = self.dims[2]
        y_max = self.dims[3]
    
        # Initialize figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.set_xlim(xmin=x_min, xmax=x_max)
        ax.set_ylim(ymin=y_min, ymax=y_max)
        plt.ion() # make non-blocking
        plt.autoscale(False)
        fig.tight_layout()
        ax.set_aspect('equal', 'box')

        # Plot obstacles, self, and goal
        for obstacle in self.obstacles:
            self.plot_obstacle(obstacle, fig)
        self.plot_robot(fig)
        for goal in self.goals:
            self.plot_goal(goal, fig)

        return fig

    def plot_obstacle(self, obstacle, fig):
        plt.figure(fig.number)
        x,y = obstacle.exterior.xy
        plt.fill(x, y, "r")

    def plot_robot(self, fig):
        plt.figure(fig.number)
        if self.last_robot != None:
            self.last_robot.remove()
        x,y = self.geometry.exterior.xy
        last_robot = plt.fill(x, y, "b")
        self.last_robot = last_robot[0]

    def plot_goal(self, goal, fig):
        plt.figure(fig.number)
        x,y = goal.exterior.xy
        plt.fill(x, y, "g")

    def update_plot(self, fig):
        plt.ion()  # make non-blocking
        # Updates
        self.plot_robot(fig)

        plt.show()
        plt.pause(0.03)
        return fig

    def show_plot(self, fig):
        plt.figure(fig.number)
        plt.show()

    # Collision check against geometry and specified obstacles
    def collision_check(self):
        for obstacle in self.obstacles:
            if self.collision_check_single(obstacle):
                return True
        return False

    def collision_check_single(self, rectangle):
        if self.geometry.intersects(rectangle):
            return True
        return False

    def goal_check(self):
        for goal in self.goals:
            if self.collision_check_single(goal) and self.state[2]+self.state[3] < 1.0:
                return True
            return False

    def in_bounds(self):
        if self.state[0] > self.dims[0] and self.state[0] < self.dims[1] and \
           self.state[1] > self.dims[2] and self.state[1] < self.dims[3]:
           return True
        return False