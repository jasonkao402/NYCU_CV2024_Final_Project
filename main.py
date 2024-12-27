"""
Triangulation : To combine two 2D points into 3D point
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional
import csv

import cv2
import numpy as np
import configparser
import pandas as pd


from point import Point, load_points_from_csv, save_points_to_csv

from MultiCamTriang import MultiCamTriang


os.path.abspath(__file__)

def loadConfig(cfg_file):
    try:
        config = configparser.ConfigParser()
        config.optionxform = str
        with open(cfg_file) as f:
            config.read_file(f)
    except IOError as e:
        logging.error(e)
        sys.exit()
    return config


def triangluation():
    global fps

    ks = []
    poses = []          # Only Used by Shao-Ping Method
    eye = []            # Only Used by Shao-Ping Method
    dist = []           # Our Method
    newcameramtx = []   # Our Method
    projection_mat = [] # Our Method

    # all points in (sorted by Timestamp)
    all_points_2d = []

    cam_idx = 0

    for i, folder in enumerate(os.listdir(data_folder)):
        print(f'{i:3d}: {folder}')
    num = int(input('Please select the rally folder: '))
    rally_folder = os.path.join(data_folder, os.listdir(data_folder)[num])


    # get fps from the config of camera
    fps = 1e9
    for c in cameras:
        mp4 = c + '.mp4'
        points_2d = load_points_from_csv(os.path.join(rally_folder ,mp4.split('.')[0]+'_ball.csv'))

        camera_config_path = os.path.join(camera_config_folder, c +'.cfg')

        camera_config = loadConfig(camera_config_path)

        fps = min(fps, float(camera_config['Camera']['fps']))

        ks.append(np.array(json.loads(camera_config['Other']['ks']),np.float32))
        poses.append(np.array(json.loads(camera_config['Other']['poses']),np.float32))
        eye.append(np.array(json.loads(camera_config['Other']['eye']),np.float32))
        dist.append(np.array(json.loads(camera_config['Other']['dist']),np.float32))
        newcameramtx.append(np.array(json.loads(camera_config['Other']['newcameramtx']),np.float32))
        projection_mat.append(np.array(json.loads(camera_config['Other']['projection_mat']),np.float32))

        for p in points_2d:
            p.cam_idx = cam_idx

        all_points_2d = all_points_2d + points_2d

        cam_idx += 1

    ks = np.stack(ks, axis=0)
    poses = np.stack(poses, axis=0)
    eye = np.stack(eye, axis=0)
    dist = np.stack(dist, axis=0)
    newcameramtx = np.stack(newcameramtx, axis=0)
    projection_mat = np.stack(projection_mat, axis=0)

    # Sort by Timestamp
    all_points_2d.sort()

    # csv3DWriter = CSVWriter(name="", filename=output_csv)

    multiCamTriang = MultiCamTriang(poses, eye, newcameramtx)

    output_points = []

    while len(all_points_2d) > 0:
        # Choose all_points_2d[0] as the point
        idx_pop_all_points_2d = [0]
        # Only Several Cameras detect the ball, [0,1,3] means idx 0,1,3 cams detect, 2 misses
        cam_detected_ball = [all_points_2d[0].cam_idx]

        points_2d_triangulated = [all_points_2d[0]]

        # Find other points that ts in 1/(fps*2)
        j = 1
        while j < len(all_points_2d):
            if all_points_2d[j].timestamp - all_points_2d[0].timestamp >= 1/(fps*2):
                break
            elif all_points_2d[j].cam_idx in cam_detected_ball: # same camera
                pass
            else:
                idx_pop_all_points_2d.append(j)
                points_2d_triangulated.append(all_points_2d[j])
                cam_detected_ball.append(all_points_2d[j].cam_idx)
            j += 1

        # 3D point fid
        # debug
        # if len(points_2d_triangulated) == 2:
        #     print('[P1]:', 'cam:', points_2d_triangulated[0].cam_idx, 'fid:', points_2d_triangulated[0].fid, '[P2]:', 'cam:', points_2d_triangulated[1].cam_idx, 'fid:', points_2d_triangulated[1].fid)
        #     print(points_2d_triangulated[0].timestamp, points_2d_triangulated[1].timestamp)
        # elif len(points_2d_triangulated) == 1:
        #     print('[P1]:', 'cam:', points_2d_triangulated[0].cam_idx, 'fid:', points_2d_triangulated[0].fid)
        point3d_fid = points_2d_triangulated[0].fid
        tmp_p = points_2d_triangulated[0]
        # check if >= 2 frames are visible
        vis_cnt = 0
        for p in points_2d_triangulated:
            if p.cam_idx < tmp_p.cam_idx:
                point3d_fid = p.fid
                tmp_p = p
            if p.visibility != 0:
                vis_cnt += 1

        # 3D point ts
        point3d_ts = np.mean([p.timestamp for p in points_2d_triangulated])

        # write invisible points
        if vis_cnt < 2:
            invis_point = Point(fid=point3d_fid,
                            timestamp=point3d_ts,
                            visibility=0,
                            x=0,
                            y=0,
                            z=0,
                            color='white')
            output_points.append(invis_point)
            # print('-->>invis', invis_point.fid, invis_point.timestamp)
            # print()
            # Remove triangulated points
            for index in sorted(idx_pop_all_points_2d, reverse=True):
                del all_points_2d[index]
            continue

        # remove invisible points and camera
        n_cam_in_ts = len(cam_detected_ball)
        new_points_2d_triangulated = []
        new_cam_detected_ball = []
        for i in range(n_cam_in_ts):
            if points_2d_triangulated[i].visibility != 0:
                new_points_2d_triangulated.append(points_2d_triangulated[i])
                new_cam_detected_ball.append(cam_detected_ball[i])
            # print(f'{i} -> detect{cam_detected_ball[i]}, vis = {points_2d_triangulated[i].visibility}')

        points_2d_triangulated = new_points_2d_triangulated
        cam_detected_ball = new_cam_detected_ball

        # Undistort
        undistort_points_2D = []
        for k in range(len(points_2d_triangulated)):
            temp = cv2.undistortPoints(points_2d_triangulated[k].toXY(),
                                        ks[cam_detected_ball[k]],
                                        dist[cam_detected_ball[k]],
                                        None,
                                        newcameramtx[cam_detected_ball[k]]) # shape:(1,num_frame,2), num_frame=1
            temp = temp.reshape(-1,2) # shape:(num_frame,2), num_frame=1
            undistort_points_2D.append(temp)
        undistort_points_2D = np.stack(undistort_points_2D, axis=0) # shape:(num_cam,num_frame,2), num_frame=1

        # Triangluation
        if undistort_points_2D.shape[0] >= 2:

            multiCamTriang.setTrack2Ds(undistort_points_2D)
            # print(projection_mat[cam_detected_ball])
            multiCamTriang.setProjectionMats(projection_mat[cam_detected_ball])
            track_3D = multiCamTriang.calculate3D() # shape:(num_frame,3), num_frame=1

            # Use Timestamp to triangulation, so fid is not correct [*]
            point3d = Point(fid=point3d_fid,
                            timestamp=point3d_ts,
                            visibility=1,
                            x=track_3D[0][0],
                            y=track_3D[0][1],
                            z=track_3D[0][2],
                            color='white')
            output_points.append(point3d)
            # print('-->>point', point3d.fid, point3d.timestamp)
            # print()

        # Remove triangulated points
        for index in sorted(idx_pop_all_points_2d, reverse=True):
            del all_points_2d[index]

    output_csv = os.path.join(rally_folder, 'Model3D.csv')
    save_points_to_csv(points=output_points, csv_file=output_csv)
    print(f'Output: {output_csv}') 

    return True


if __name__ == '__main__':
    cameras = ['CameraReader_1', 'CameraReader_2']
    camera_config_folder = 'config'
    data_folder = 'data'
    triangluation()

