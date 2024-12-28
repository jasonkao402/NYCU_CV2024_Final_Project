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

def draw_court_lines(ax):
    """
    在 z=0 平面上，繪製整個「雙打」羽球場的所有主要標線 + 立面網子(網高約 1.55 m)。
    
    假設座標原點在球場正中央，所以：
      - x 軸範圍為 -3.05 ~ 3.05 (球場寬 6.10 m)
      - y 軸範圍為 -6.7 ~ 6.7 (球場長 13.4 m)
      - z=0 為地面。
      - 網子在 y=0，高度約 1.55 m。
    """

    #---------------------------------------------
    # 幫助避免線條被 surface 或 pane 遮住：
    # 可以將線條稍微提升一點 z=0.001，
    # 以免與地板 (z=0) 在 3D 場景發生 z-fighting。
    #---------------------------------------------
    offset_z = 0

    #---------------------------------------------
    # 1. 外框 (雙打邊界)
    #   x = ±3.05, y = ±6.7
    #---------------------------------------------
    outer_x = [-3.05,  3.05,  3.05, -3.05, -3.05]
    outer_y = [-6.70, -6.70,  6.70,  6.70, -6.70]
    outer_z = [offset_z]*5
    ax.plot(outer_x, outer_y, outer_z, color='white', linewidth=2)

    #---------------------------------------------
    # 2. 單打邊線
    #   x = ±(半場寬度 2.53), y = ±6.7
    #   若需要顯示單打場地，可以補上這些線
    #---------------------------------------------
    single_x = [-2.53,  2.53,  2.53, -2.53, -2.53]
    single_y = [-6.70, -6.70,  6.70,  6.70, -6.70]
    single_z = [offset_z]*5
    ax.plot(single_x, single_y, single_z, color='white', linewidth=1.5)

    #---------------------------------------------
    # 3. 短發球線 (Short Service Line)
    #   與網子距離約 1.98 m -> y= ±1.98
    #---------------------------------------------
    short_service_line_x = [-3.05, 3.05]
    short_service_line_y1 = [1.98, 1.98]
    short_service_line_y2 = [-1.98, -1.98]
    short_service_line_z  = [offset_z, offset_z]

    ax.plot(short_service_line_x, short_service_line_y1, short_service_line_z,
            color='white', linewidth=2)
    ax.plot(short_service_line_x, short_service_line_y2, short_service_line_z,
            color='white', linewidth=2)

    #---------------------------------------------
    # 4. 雙打長發球線 (Doubles Long Service Line)
    #   距離底線約 0.76 m => y= ±(6.70 - 0.76) = ±5.94
    #---------------------------------------------
    doubles_long_service_line_x = [-3.05, 3.05]
    doubles_long_service_line_y1 = [5.94, 5.94]
    doubles_long_service_line_y2 = [-5.94, -5.94]
    doubles_long_service_line_z  = [offset_z, offset_z]

    ax.plot(doubles_long_service_line_x, doubles_long_service_line_y1, doubles_long_service_line_z,
            color='white', linewidth=2)
    ax.plot(doubles_long_service_line_x, doubles_long_service_line_y2, doubles_long_service_line_z,
            color='white', linewidth=2)

    #---------------------------------------------
    # 5. 中心線 (Center Line)
    #   分隔左右發球區，從短發球線 y=±1.98 畫到底線 y=±6.7
    #   這樣各發球區被分成左右半場。
    #---------------------------------------------
    # 上半場
    center_line_x = [0, 0]
    center_line_y_upper = [1.98, 6.7]   # y=1.98到6.7
    center_line_y_lower = [-1.98, -6.7] # y=-1.98到-6.7
    center_line_z = [offset_z, offset_z]

    ax.plot(center_line_x, center_line_y_upper, center_line_z,
            color='white', linewidth=2)
    ax.plot(center_line_x, center_line_y_lower, center_line_z,
            color='white', linewidth=2)

    #---------------------------------------------
    # 6. 網子 (以一個面表示)
    #   在 y=0，網高約 1.55 m
    #---------------------------------------------
    net_height = 1.55
    x_net = np.linspace(-3.05, 3.05, 2)   # 網子與雙打場寬相同
    z_net = np.linspace(0.79, net_height, 2)
    X_net, Z_net = np.meshgrid(x_net, z_net)
    Y_net = np.zeros_like(X_net)  # 全為 y=0
    # 以半透明黑色表示網子
    ax.plot_surface(X_net, Y_net, Z_net, color='black', alpha=0.5)

    # 7. 兩根柱子 (Poles)  
    #    假設在 x=±3.05, y=0, 高度 ~1.55 m
    pole_left_x = [-3.05, -3.05]
    pole_left_y = [0, 0]
    pole_left_z = [0, net_height]
    ax.plot(pole_left_x, pole_left_y, pole_left_z,
            color='black', linewidth=2)

    pole_right_x = [3.05, 3.05]
    pole_right_y = [0, 0]
    pole_right_z = [0, net_height]
    ax.plot(pole_right_x, pole_right_y, pole_right_z,
            color='black', linewidth=2)


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
ax1.set_xlim3d(-3.2, 3.2)
ax1.set_ylim3d(-6.9, 6.9)
ax1.set_zlim3d(0, 6)
ax1.zaxis.set_pane_color((0.098, 0.537, 0.392, 1.0))
ax1.set_box_aspect([6.4, 13.8, 6])
draw_court_lines(ax1)
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
