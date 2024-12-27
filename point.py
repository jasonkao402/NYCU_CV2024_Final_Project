"""
Object: for each point of trajectory
"""
import numpy as np
import pandas as pd

class Point():
    def __init__(self, fid=-1, timestamp=-1, visibility=0, x=0, y=0, z=0, event=0, speed=0.0, color='white'):
        self.fid = int(float(fid))
        self.timestamp = float(timestamp)
        self.visibility = int(float(visibility))
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.event = int(float(event))
        self.speed = float(speed)
        self.color = color
        self.cam_idx = -1

    def __str__(self):
        s = (f"\nPoint Fid: {self.fid}\n"
             f"Timestamp: {self.timestamp:>.3f}\n"
             f"Vis: {self.visibility}\n"
             f"({self.x:.2f},{self.y:.2f},{self.z:.2f})\n"
             f"Event: {self.event}\n"
             f"Speed: {self.speed:.0f}\n"
             f"Camera Idx: {self.cam_idx}\n")
        return s

    def toXY(self):
        return np.array([self.x, self.y])

    def toXYT(self):
        return np.array([self.x, self.y, self.timestamp])

    def toXYZ(self):
        return np.array([self.x, self.y, self.z])

    def toXYZT(self):
        return np.array([self.x, self.y, self.z, self.timestamp])

    def __lt__(self, other):
        # sorted by timestamp
        return self.timestamp < other.timestamp

def load_points_from_csv(csv_file) -> list:
    # return list of Points
    # if you want to get ndarray of (N,4), just do np.stack([x.toXYZT() for x in list], axis=0)
    ret = []
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        point = Point(fid=row['Frame'], timestamp=row['Timestamp'], visibility=row['Visibility'],
                        x=row['X'], y=row['Y'], z=row['Z'],
                        event=row['Event'])
        ret.append(point)

    return ret

def save_points_to_csv(points: list, csv_file):
    # points: list of Point
    df = {}
    for i,p in enumerate(points):
        df[i] = {}
        df[i]['Frame'] = p.fid
        df[i]['Visibility'] = p.visibility
        df[i]['X'] = p.x
        df[i]['Y'] = p.y
        df[i]['Z'] = p.z
        df[i]['Event'] = p.event
        df[i]['Timestamp'] = p.timestamp
    COLUMNS = ['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp']
    pd_df = pd.DataFrame.from_dict(df, orient='index', columns=COLUMNS)
    pd_df.to_csv(csv_file, encoding = 'utf-8',index = False)

def np2Point(i: np.array, vis=1, fid=-1) -> Point:
    return Point(fid=fid, visibility=vis,x=i[0],y=i[1],z=i[2],timestamp=i[3])
    # [np2Point(p) for p in x]
