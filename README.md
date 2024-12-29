# 3D Shuttlecock Trajectory Reconstruction

## Project Overview
This project reconstructed the 3D trajectory of a badminton shuttlecock using multiple camera views, and determine the scroing of single rally based on the reconstructed 3D trajectory.

## Requirements
Python 3.12.6
```
pip install -r requirements.txt
```

## Usage
The video and csv name must me the same as camera config file names under data folder
```
(Assumes two camera configurations c1.cfg, c2.cfg)
data_folder/
├── c1.cfg
├── c2.cfg
├── c1.mp4
├── c2.mp4
├── c1.csv
└── c2.csv
```
reconstruct3D.py reads a data folder containing the camera configurations, videos, and labeled csv files,
and outputs the reconstructed 3D coordinates to a csv file.
```bash
python reconstruct3D.py --data_folder <path_to_data_folder> --output_csv <filepath_to_output_csv>
```
render3D.py reads the output csv file from reconstructed csv file, and outputs a video containing the 3d trajectory and analysis on the badminton field .
```bash
python render3D.py --csv <path_to_input_csv> --output_vid <filepath_to_output_video>
```

## Web Demo
You can try our online demo to generate the 3D trajectory using our data.
```bash
python app.py
```
This page contains a page to load data folder, and will show the 3d trajectory visualization (might take some time to generate).
It also contains another page to calculate extrinsic matrix for both cameras (intrinsic parameters are required).

## Result
![./output/3D_trajectory.gif](./output/3D_trajectory.gif)



