import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from point import load_points_from_csv, save_points_to_csv
from time import strftime
from tqdm import tqdm
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.animation as animation_module

# Argument parser to choose CSV file and output directory
parser = argparse.ArgumentParser(description="3D Trajectory Visualization")
parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save the output video")
args = parser.parse_args()

output_dir = args.output_dir
trace = load_points_from_csv(args.csv)
trace = np.stack([x.toXYZ() for x in trace], axis=0)
print(trace.shape)
# (1438, 3)

def interpolate_trajectory(points, method="linear"):
    """
    Fill missing 3D trajectory values based on interpolation.
    Args:
        trace (np.ndarray): Input array with shape (n, 3) where columns are [X, Y, Z].
        method (str): Interpolation method (e.g., 'linear', 'spline', 'polynomial').
    Returns:
        np.ndarray: Array with missing values filled.
    """
    # Identify missing points
    missing = np.all(points == 0, axis=1)
    # Do not interpolate missing values at the beginning
    for i in range(len(missing)):
        if missing[i] == True:
            missing[i] = False
        else:
            missing[i] = False
            for j in range(0, i):
                points[j] = points[i]
            break
    valid = ~missing
    
    valid_indices = np.where(valid)[0]
    
    # Create interpolation function
    if method == "linear":
        f = interp1d(valid_indices, points[valid], axis=0, kind="linear", fill_value="extrapolate")
    elif method == "spline":
        f = UnivariateSpline(valid_indices, points[valid], axis=0, k=2, s=0)
    
    # Interpolate missing values
    trace_interp = f(np.arange(len(points)))
    points[missing] = trace_interp[missing]
    return points

def moving_average(data, window_size=5):
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return np.concatenate((data[:window_size-1], smoothed))  # Keep length consistent

trace = interpolate_trajectory(trace, method="linear")
for i in range(3):
    trace[:, i] = moving_average(trace[:, i], window_size=9)
    
# Calculate velocity and acceleration
velocity = np.diff(trace, axis=0, prepend=np.zeros((1, 3)))
acceleration = np.diff(velocity, axis=0, prepend=np.zeros((1, 3)))

# plot 3D scatter plot animation
FPS = 120
interval = 1000 / FPS
TRAIL = 30  # Number of past frames to show
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111, projection='3d')
axis_min = np.min(trace, axis=0)
axis_max = np.max(trace, axis=0)
print(axis_min, axis_max, sep='\n')

# Initialize scatter plot and quiver objects
scatter = ax1.scatter([], [], [], c='r')

def best_fit_plane(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[2, :]
    return centroid, normal

last_normal = None

def update(frame):
    global last_normal
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim3d(axis_min[0], axis_max[0])
    ax1.set_ylim3d(axis_min[1], axis_max[1])
    ax1.set_zlim3d(axis_min[2], axis_max[2])
    
    # Calculate the best fit plane for the current frame
    current_points = trace[max(0, frame - 7):frame + 7, :3]
    _, normal = best_fit_plane(current_points)
    
    # Interpolate the normal to reduce spazzing camera movement
    normal = normal if last_normal is None else normal if np.dot(normal, last_normal) > 0 else -normal
    if last_normal is not None:
        normal = 0.99 * last_normal + 0.01 * normal
    last_normal = normal
    
    # Set the view to the best fit plane's normal
    ax1.view_init(elev=10, azim=np.degrees(np.arctan2(normal[1], normal[0])))
    
    # Update scatter plot
    trail_indices = range(max(0, frame - TRAIL), frame + 1)
    scatter._offsets3d = (trace[trail_indices, 0], 
                          trace[trail_indices, 1], 
                          trace[trail_indices, 2])
    
    # Update scatter plot sizes to make the trail width gradually get smaller
    sizes = 10 ** np.linspace(0, 2, len(trail_indices))

    # if (x,y,z) == (0,0,0), set size to 0
    for i in range(len(trail_indices)):
        if np.all(trace[trail_indices[i]] == 0):
            sizes[i] = 0

    scatter.set_sizes(sizes)
    
    ax1.title.set_text(f'Frame: {frame}, Vel: {np.linalg.norm(velocity[frame]*FPS):6.3f}, Acc: {np.linalg.norm(acceleration[frame]*FPS**2):6.3f}')

def save_animation_to_video(animation, filename):
    writer = animation_module.writers['ffmpeg'](fps=FPS, extra_args=['-vcodec', 'libx264'])
    with tqdm(total=animation._save_count, desc="Saving video") as pbar:
        animation.save(filename, writer=writer, progress_callback=lambda i, n: pbar.update(1))

start_frame = 0
end_frame = trace.shape[0]
ani = FuncAnimation(fig, update, frames=range(start_frame, end_frame), repeat=False, interval=interval)

# Save the animation to a video file
timestamp = strftime('%Y_%m%d_%H%M')
save_animation_to_video(ani, os.path.join(output_dir, f'3D_trajectory.mp4'))
