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

def main():
    dt = .5

    # Obstcles and self geometry
    offsets = [-0.5, -0.5, 0.5, 0.5]  # 2D geometry offsets (minx, miny, maxx, maxy)

    obstacles = []
    goals = []

    obstacle_coords = np.array([
        [0.0, 0.0, 2.0, 25.0],  # lower node
        [0.0, 23.0, 15.0, 25.0],
        [0.0, 0.0, 15.0, 2.0],
        [13.0, 0.0, 15.0, 9.5],
        [13.0, 14.5, 15.0, 25.0],
        ])

    # goal_coords = np.array([
    #     [42, 10, 47, 15],
    #     [4, 10, 9, 15]
    #     ])

    goal_coords = np.array([
        [4, 10, 9, 15]
        ])

    for i in range(obstacle_coords.shape[0]):
        obstacle = box(obstacle_coords[i][0], obstacle_coords[i][1], obstacle_coords[i][2], obstacle_coords[i][3])
        obstacles.append(obstacle)
    for i in range(goal_coords.shape[0]):
        goal = box(goal_coords[i][0], goal_coords[i][1], goal_coords[i][2], goal_coords[i][3])
        goals.append(goal)

    # Perform a receding horizon simulation

    # writer = setup_movie_writer(10)
    # with writer.saving(plt.figure(fig.number), "blargh.mp4", 300):
    
    number_of_reductions = []
    total_reward = []

    for trial in range(50):
        trial_reward = 0
        reduction = 0
        init_belief = Belief(np.array([25.0, 12.5, 0, 0]).reshape(-1, 1), 0.1*np.identity(4))
        sim = DoubleInt(0.0, 0.0, 50.0, 25.0, offsets, obstacles, goals, dt, init_belief)
        true_system = DoubleInt(0.0, 0.0, 50.0, 25.0, offsets, obstacles, goals, dt, init_belief)
        belief = init_belief
        covar_factor = 1.0

        # Create the simulator and true system

        fig = true_system.init_plot()
        true_system.show_plot(fig)

        for i in range(50):
            # action = np.random.rand(2,1)*0.1
            n = 100
            best_action, root = plan(belief, sim, n)
            print "action: ",best_action
            print "Q: ", root.Q

            _, _, reward, _, _ = true_system.set_state_and_simulate(true_system.state, best_action, 0)
            true_system.update_plot(fig)
            belief.mean = true_system.state
            trial_reward += reward
            if best_action[0] == "reduce":
                reduction+=1
            # print covar_factor
            # root.traverse_tree()
            # writer.grab_frame()
            root.plot_tree(fig)
            # plt.pause(20)
        total_reward.append(trial_reward)
        number_of_reductions.append(reduction)
        print total_reward
        print number_of_reductions
if __name__ == '__main__':
    main()