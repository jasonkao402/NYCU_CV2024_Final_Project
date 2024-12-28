import argparse
import json
import logging
import os
import sys
import cv2
import numpy as np
import configparser
import pandas as pd

from point import Point, load_points_from_csv, save_points_to_csv
from MultiCamTriang import MultiCamTriang

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

def triangluation(data_folder, output_dir):
    ks = []
    poses = []     
    eye = []
    dist = []
    newcameramtx = []
    projection_mat = []

    all_points_2d = []

    cam_idx = 0

    cameras = [f.split('.')[0] for f in os.listdir(data_folder) if f.endswith('.cfg')]
    print(cameras)

    fps = 1e9
    for c in cameras:
        mp4 = c + '.mp4'
        points_2d = load_points_from_csv(os.path.join(data_folder, mp4.split('.')[0] + '.csv'))

        camera_config_path = os.path.join(data_folder, c + '.cfg')
        camera_config = loadConfig(camera_config_path)

        fps = min(fps, float(camera_config['Camera']['fps']))

        ks.append(np.array(json.loads(camera_config['Other']['ks']), np.float32))
        poses.append(np.array(json.loads(camera_config['Other']['poses']), np.float32))
        eye.append(np.array(json.loads(camera_config['Other']['eye']), np.float32))
        dist.append(np.array(json.loads(camera_config['Other']['dist']), np.float32))
        newcameramtx.append(np.array(json.loads(camera_config['Other']['newcameramtx']), np.float32))
        projection_mat.append(np.array(json.loads(camera_config['Other']['projection_mat']), np.float32))

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

    all_points_2d.sort()

    multiCamTriang = MultiCamTriang(poses, eye, newcameramtx)

    output_points = []

    while len(all_points_2d) > 0:
        idx_pop_all_points_2d = [0]
        cam_detected_ball = [all_points_2d[0].cam_idx]
        points_2d_triangulated = [all_points_2d[0]]

        j = 1
        while j < len(all_points_2d):
            if all_points_2d[j].timestamp - all_points_2d[0].timestamp >= 1 / (fps * 2):
                break
            elif all_points_2d[j].cam_idx in cam_detected_ball:
                pass
            else:
                idx_pop_all_points_2d.append(j)
                points_2d_triangulated.append(all_points_2d[j])
                cam_detected_ball.append(all_points_2d[j].cam_idx)
            j += 1

        point3d_fid = points_2d_triangulated[0].fid
        tmp_p = points_2d_triangulated[0]
        vis_cnt = 0
        for p in points_2d_triangulated:
            if p.cam_idx < tmp_p.cam_idx:
                point3d_fid = p.fid
                tmp_p = p
            if p.visibility != 0:
                vis_cnt += 1

        point3d_ts = np.mean([p.timestamp for p in points_2d_triangulated])

        if vis_cnt < 2:
            invis_point = Point(fid=point3d_fid,
                                timestamp=point3d_ts,
                                visibility=0,
                                x=0,
                                y=0,
                                z=0,
                                color='white')
            output_points.append(invis_point)
            for index in sorted(idx_pop_all_points_2d, reverse=True):
                del all_points_2d[index]
            continue

        n_cam_in_ts = len(cam_detected_ball)
        new_points_2d_triangulated = []
        new_cam_detected_ball = []
        for i in range(n_cam_in_ts):
            if points_2d_triangulated[i].visibility != 0:
                new_points_2d_triangulated.append(points_2d_triangulated[i])
                new_cam_detected_ball.append(cam_detected_ball[i])

        points_2d_triangulated = new_points_2d_triangulated
        cam_detected_ball = new_cam_detected_ball

        undistort_points_2D = []
        for k in range(len(points_2d_triangulated)):
            temp = cv2.undistortPoints(points_2d_triangulated[k].toXY(),
                                       ks[cam_detected_ball[k]],
                                       dist[cam_detected_ball[k]],
                                       None,
                                       newcameramtx[cam_detected_ball[k]])
            temp = temp.reshape(-1, 2)
            undistort_points_2D.append(temp)
        undistort_points_2D = np.stack(undistort_points_2D, axis=0)

        if undistort_points_2D.shape[0] >= 2:
            multiCamTriang.setTrack2Ds(undistort_points_2D)
            multiCamTriang.setProjectionMats(projection_mat[cam_detected_ball])
            track_3D = multiCamTriang.calculate3D()

            point3d = Point(fid=point3d_fid,
                            timestamp=point3d_ts,
                            visibility=1,
                            x=track_3D[0][0],
                            y=track_3D[0][1],
                            z=track_3D[0][2],
                            color='white')
            output_points.append(point3d)

        for index in sorted(idx_pop_all_points_2d, reverse=True):
            del all_points_2d[index]

    output_csv = os.path.join(output_dir, 'Model3D.csv')
    save_points_to_csv(points=output_points, csv_file=output_csv)
    print(f'Output: {output_csv}')

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triangulate 3D points from 2D data.")
    parser.add_argument('--data_folder', required=True, help="Path to the data folder.")
    parser.add_argument('--output_dir', default='output', help="Path to the output directory.")
    args = parser.parse_args()

    triangluation(data_folder=args.data_folder, output_dir=args.output_dir)
