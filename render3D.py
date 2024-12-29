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
parser.add_argument("--output_vid", type=str, default="output/3D_trajectory.mp4", help="Directory to save the output video")
args = parser.parse_args()

output_vid = args.output_vid
trace = load_points_from_csv(args.csv)
trace = np.stack([x.toXYZ() for x in trace], axis=0)
start_frame = 0
end_frame = trace.shape[0]
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
        if missing[i]:
            missing[i] = False
        else:
            for j in range(0, i):
                points[j] = points[i]
            break
    # Do the same for the end
    for i in range(len(missing) - 1, -1, -1):
        if missing[i]:
            missing[i] = False
        else:
            for j in range(len(missing) - 1, i, -1):
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
    繪製雙打球場主要標線、網子以及柱子。
    """
    offset_z = 0

    # 外框 (雙打)
    outer_x = [-3.05,  3.05,  3.05, -3.05, -3.05]
    outer_y = [-6.70, -6.70,  6.70,  6.70, -6.70]
    outer_z = [offset_z]*5
    ax.plot(outer_x, outer_y, outer_z, color='white', linewidth=2)

    # 單打邊線
    single_x = [-2.53,  2.53,  2.53, -2.53, -2.53]
    single_y = [-6.70, -6.70,  6.70,  6.70, -6.70]
    single_z = [offset_z]*5
    ax.plot(single_x, single_y, single_z, color='white', linewidth=1.5)

    # 短發球線 (±1.98)
    short_service_line_x = [-3.05, 3.05]
    short_service_line_y1 = [1.98, 1.98]
    short_service_line_y2 = [-1.98, -1.98]
    short_service_line_z  = [offset_z, offset_z]
    ax.plot(short_service_line_x, short_service_line_y1, short_service_line_z,
            color='white', linewidth=2)
    ax.plot(short_service_line_x, short_service_line_y2, short_service_line_z,
            color='white', linewidth=2)

    # 雙打長發球線 (±5.94)
    doubles_long_service_line_x = [-3.05, 3.05]
    doubles_long_service_line_y1 = [5.94, 5.94]
    doubles_long_service_line_y2 = [-5.94, -5.94]
    doubles_long_service_line_z  = [offset_z, offset_z]
    ax.plot(doubles_long_service_line_x, doubles_long_service_line_y1, doubles_long_service_line_z,
            color='white', linewidth=2)
    ax.plot(doubles_long_service_line_x, doubles_long_service_line_y2, doubles_long_service_line_z,
            color='white', linewidth=2)

    # 中心線 (±1.98 到 ±6.7)
    center_line_x = [0, 0]
    ax.plot(center_line_x, [1.98, 6.7], [offset_z, offset_z],
            color='white', linewidth=2)
    ax.plot(center_line_x, [-1.98, -6.7], [offset_z, offset_z],
            color='white', linewidth=2)

    # 網子 (y=0, 高度~1.55)
    net_height = 1.55
    x_net = np.linspace(-3.05, 3.05, 2)
    z_net = np.linspace(0.79, net_height, 2)
    X_net, Z_net = np.meshgrid(x_net, z_net)
    Y_net = np.zeros_like(X_net)  
    ax.plot_surface(X_net, Y_net, Z_net, color='black', alpha=0.5)

    # 兩根柱子
    pole_left_x = [-3.05, -3.05]
    pole_left_y = [0, 0]
    pole_left_z = [0, net_height]
    ax.plot(pole_left_x, pole_left_y, pole_left_z, color='black', linewidth=2)

    pole_right_x = [3.05, 3.05]
    pole_right_y = [0, 0]
    pole_right_z = [0, net_height]
    ax.plot(pole_right_x, pole_right_y, pole_right_z, color='black', linewidth=2)

# -- 前處理 --
trace = interpolate_trajectory(trace, method="linear")
for i in range(3):
    trace[:, i] = moving_average(trace[:, i], window_size=5)

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

# axis_min = np.min(trace, axis=0)
# axis_max = np.max(trace, axis=0)
# print(axis_min, axis_max, sep='\n')

# Initialize scatter plot and quiver objects
scatter = ax1.scatter([], [], [], c='r')

# def best_fit_plane(points):
#     centroid = np.mean(points, axis=0)
#     centered_points = points - centroid
#     _, _, vh = np.linalg.svd(centered_points)
#     normal = vh[2, :]
#     return centroid, normal

# 新增：判斷誰得分的變數 + 閾值
landing_threshold = 0.05
winner = None
dest_angle = 0
curr_angle = 0

def update(frame):
    global dest_angle, curr_angle, winner

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Calculate the destination angle based on the ball's position to the judge's position
    judge_pos = np.array([20, 0, 0])
    x, y, z = trace[frame] - judge_pos
    dest_angle = np.degrees(np.arctan2(y, x))
    dest_angle = np.clip(dest_angle, -15, 15)
    
    curr_angle = 0.97 * curr_angle + 0.03 * dest_angle
    
    ax1.view_init(elev=10, azim=curr_angle)

    # Update scatter plot
    trail_indices = range(max(0, frame - TRAIL), frame + 1)
    scatter._offsets3d = (
        trace[trail_indices, 0], 
        trace[trail_indices, 1], 
        trace[trail_indices, 2]
    )
    
    # Update scatter plot sizes to make the trail width gradually get smaller
    sizes = 10 ** np.linspace(0, 2, len(trail_indices))
    for i in range(len(trail_indices)):
        if np.all(trace[trail_indices[i]] == 0):
            sizes[i] = 0
    scatter.set_sizes(sizes)

    # 取得當前球位置
    x, y, z = trace[frame]

    # ---------------------------------------------------
    # 新增: 出界 & 落地 的簡易判斷 (只會發生一次)
    # ---------------------------------------------------
    if winner is None:
        # 1) 檢查是否 "出界"
        #    球超過外框 => (abs(x) > 3.05) or (abs(y) > 6.7)
        if (abs(x) > 3.05 or abs(y) > 6.7) and z < landing_threshold:
            # 如果 y>0 => 出界在 B方 => A得分
            # 如果 y<0 => 出界在 A方 => B得分
            if y > 0:
                winner = "A gets the point (out-of-bound by B)"
            else:
                winner = "B gets the point (out-of-bound by A)"

        # 2) 若沒出界，再檢查是否落地 (z < threshold)
        elif z < landing_threshold:
            # y>0 => B方半場 => A得分
            # y<0 => A方半場 => B得分
            if y > 0:
                winner = "A gets the point (landed in B-court)"
            else:
                winner = "B gets the point (landed in A-court)"

    # 顯示在標題
    current_vel = np.linalg.norm(velocity[frame] * FPS)
    current_acc = np.linalg.norm(acceleration[frame] * FPS**2)
    
    if winner is None:
        title_text = (
            f"Frame: {frame:4d}, "
            f"Vel: {current_vel:7.3f}, "
            f"Acc: {current_acc:7.3f}, "
            "No winner yet"
        )
    else:
        title_text = (
            f"Frame: {frame:4d}, "
            f"Vel: {current_vel:7.3f}, "
            f"Acc: {current_acc:7.3f}, "
            f"Result: {winner}"
        )

    ax1.title.set_text(title_text)

def save_top_down_image(ax, trace, frame, filename):
    """
    Save a top-down image of the court and ball position at the given frame.
    """
    ax.view_init(elev=90, azim=0)  # Top-down view
    ball_x, ball_y, ball_z = trace[frame]
    
    # Set limits to zoom in on the ball
    zoom_margin = 3.0
    ax.set_box_aspect([2*zoom_margin, 2*zoom_margin, 6])
    ax.set_xlim3d(ball_x - zoom_margin, ball_x + zoom_margin)
    ax.set_ylim3d(ball_y - zoom_margin, ball_y + zoom_margin)
    ax.set_zlim3d(0, 3)
    
    # Update scatter plot for the specific frame
    scatter._offsets3d = (
        trace[frame:frame+1, 0], 
        trace[frame:frame+1, 1], 
        trace[frame:frame+1, 2]
    )
    scatter.set_sizes([100])  # Set the point size larger
    plt.savefig(filename)
    print(f"Top-down image saved to {filename}")

def save_animation_to_video(animation, filename):
    writer = animation_module.writers['ffmpeg'](fps=FPS, extra_args=['-vcodec', 'libx264'])
    with tqdm(total=animation._save_count, desc="Saving video") as pbar:
        animation.save(filename, writer=writer, progress_callback=lambda i, n: pbar.update(1))
    
    # Save the top-down image at the end of the animation
    save_top_down_image(ax1, trace, end_frame - 1, filename.replace('.mp4', '_top_down.png'))

ani = FuncAnimation(fig, update, frames=range(end_frame), repeat=False, interval=interval)

# Save the animation to a video file
save_animation_to_video(ani, output_vid)
