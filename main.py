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

dt = .2

# Self geometry
offsets = [-1, -1, 1, 1]  # 2D geometry offsets (minx, miny, maxx, maxy)

# Obstacles
# o1 = box(12.0, 7.0, 14.0, 25.0)  # 2d geometry absolutes (minx, miny, maxx, maxy)
# o2 = box(38.0, 0.0, 40.0, 18.0)

o1 = box(0.0, 7.0, 8.5, 9.0)  # 2d geometry absolutes (minx, miny, maxx, maxy)
o2 = box(11.5, 7.0, 20.0, 9.0)
o3 = box(20.0, 7.0, 22.0, 25.0)

obstacles = [o1, o2, o3]

# Goal regions
g1 = box(46, 2, 48, 4)
g2 = box(12, 18, 14, 20)
# g3 = box(5, 5, 7, 7)
goals = [g1, g2]

# Create the simulator
init_belief = Belief(np.array([2, 2, 0, 0]).reshape(-1, 1), 0.1*np.identity(4))

sim = DoubleInt(0.0, 50.0, 0.0, 25.0, offsets, obstacles, goals, dt, init_belief)

true_system = DoubleInt(0.0, 50.0, 0.0, 25.0, offsets, obstacles, goals, dt, init_belief)
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
    n = 40
    best_action, root = plan(belief, sim, n)
    print best_action

    true_system.transition_stoc(best_action)
    true_system.update_plot(fig)
    belief.mean = true_system.state

    # writer.grab_frame()
    root.plot_tree(fig)
    # plt.pause(20)