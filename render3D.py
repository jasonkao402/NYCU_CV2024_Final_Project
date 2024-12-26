import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from point import load_points_from_csv, save_points_to_csv
from time import strftime
from tqdm import tqdm
import matplotlib.animation as animation_module

output_dir = 'output/'
# read x,y,z coordinates and timestamp from csv file
trace = load_points_from_csv('Model3D.csv')
trace = np.stack([x.toXYZT() for x in trace], axis=0)
print(trace.shape)

def moving_average(data, window_size=5):
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return np.concatenate((data[:window_size-1], smoothed))  # Keep length consistent

for i in range(3):
    trace[:, i] = moving_average(trace[:, i], window_size=9)
# Calculate velocity and acceleration
velocity = np.diff(trace[:, :3], axis=0, prepend=np.zeros((1, 3)))
# acceleration = np.diff(velocity, axis=0, prepend=np.zeros((1, 3)))

# plot 3D scatter plot animation
FPS = 120
interval = 1000 / FPS
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
axis_min = np.min(trace, axis=0)
axis_max = np.max(trace, axis=0)
print(axis_min, axis_max)

N = 30  # Number of past frames to show
# Apply a simple moving average for smoothing

def update(frame):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(axis_min[0], axis_max[0])
    ax.set_ylim(axis_min[1], axis_max[1])
    ax.set_zlim(axis_min[2], axis_max[2])
    ax.view_init(elev=10, azim=45+frame/100)
    for i in range(max(0, frame - N), frame + 1):
        alpha = (i - max(0, frame - N)) / N
        ax.scatter(trace[i, 0], trace[i, 1], trace[i, 2], c='r', alpha=alpha)
        if i > 0:
            ax.quiver(trace[i-1, 0], trace[i-1, 1], trace[i-1, 2], 
                      velocity[i, 0], velocity[i, 1], velocity[i, 2], 
                      color='b', alpha=alpha, length=0.25, normalize=True)
            # ax.quiver(trace[i-1, 0], trace[i-1, 1], trace[i-1, 2], 
            #           acceleration[i, 0], acceleration[i, 1], acceleration[i, 2], 
            #           color='g', alpha=alpha, length=0.5, normalize=True)
    ax.title.set_text('Frame: ' + str(frame))

def save_animation_to_video(animation, filename):
    

    writer = animation_module.writers['ffmpeg'](fps=FPS, extra_args=['-vcodec', 'libx264'])
    with tqdm(total=animation._save_count, desc="Saving video") as pbar:
        animation.save(filename, writer=writer, progress_callback=lambda i, n: pbar.update(1))

start_frame = 400
end_frame = min(900, trace.shape[0])
# end_frame = trace.shape[0]
ani = FuncAnimation(fig, update, frames=range(start_frame, end_frame), repeat=False, interval=interval)
# plt.show()

# Save the animation to a video file
timestamp = strftime('%Y_%m%d_%H%M')
save_animation_to_video(ani, f'{output_dir}trace_{timestamp}.mp4')
