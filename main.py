"""
Creates the POMDP, calls up a solution method, and interprets the result.
"""
from DoubleInt import DoubleInt
from Belief import Belief
import numpy as np
from shapely.geometry import box, Polygon
from common import plan
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# moviewriter setup
def setup_movie_writer(fps):
    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
        comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer

def replay(states):
    return None

dt = .2

# Obstcles and self geometry
offsets = [-1, -1, 1, 1]  # 2D geometry offsets (minx, miny, maxx, maxy)

obstacles = []
goals = []

obstacle_coords = np.array([
    [0.0, 7.0, 8.5, 9.0],  # lower node
    [11.5, 7.0, 20.0, 9.0],
    [20.0, 7.0, 22.0, 25.0]
    ])

goal_coords = np.array([
    [46, 2, 48, 4],
    [12, 18, 14, 20]
    ])

for i in range(obstacle_coords.shape[0]):
    obstacle = box(obstacle_coords[i][0], obstacle_coords[i][1], obstacle_coords[i][2], obstacle_coords[i][3])
    obstacles.append(obstacle)
for i in range(goal_coords.shape[0]):
    goal = box(goal_coords[i][0], goal_coords[i][1], goal_coords[i][2], goal_coords[i][3])
    goals.append(goal)

# Create the simulator and true system
init_belief = Belief(np.array([2, 2, 0, 0]).reshape(-1, 1), 0.1*np.identity(4))
sim = DoubleInt(0.0, 0.0, 50.0, 25.0, offsets, obstacles, goals, dt, init_belief)
true_system = DoubleInt(0.0, 0.0, 50.0, 25.0, offsets, obstacles, goals, dt, init_belief)

fig = true_system.init_plot()
true_system.show_plot(fig)

# # Simulation loop
# for i in range(500):
#     # action = np.random.rand(2,1)*0.1
#     action = np.array([4,1]).reshape(-1,1)
#     sim.transition_stoc(action)
#     sim.update_plot(fig)


# Perform a receding horizon simulation
belief = init_belief
# writer = setup_movie_writer(10)
# with writer.saving(plt.figure(fig.number), "blargh.mp4", 300):
for i in range(100):
    # action = np.random.rand(2,1)*0.1
    n = 100
    best_action, root = plan(belief, sim, n)
    print best_action

    true_system.transition_stoc(best_action)
    true_system.update_plot(fig)
    belief.mean = true_system.state
    print "traversing"
    # root.traverse_tree()
    # writer.grab_frame()
    # root.plot_tree(fig)
    # plt.pause(20)