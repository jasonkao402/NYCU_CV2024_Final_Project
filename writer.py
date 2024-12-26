import pandas as pd
import csv
import logging
from operator import attrgetter

from point import Point


class CSVWriter():
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.csvfile = open(filename, 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp'])

    def close(self):
        self.csvfile.flush()
        self.csvfile.close()

        df = pd.read_csv(self.filename)
        df = df.sort_values(by=["Frame"])
        df.to_csv(self.filename, mode='w+', index=False)

    def writePoints(self, points):
        if isinstance(points, Point): # A point
            self.writer.writerow([points.fid, points.visibility, points.x, points.y, points.z, points.event, points.timestamp])
        elif isinstance(points, list): # A point list
            for p in points:
                self.writer.writerow([p.fid, p.visibility, p.x, p.y, p.z, p.event, p.timestamp])

        self.csvfile.flush()

    def setEventByTimestamp(self, event, timestamp): # Bug TODO
        df = pd.read_csv(self.filename)
        df.loc[df['Timestamp'] == timestamp, 'Event'] = event

        # writing into the file
        df.to_csv(self.filename, mode='w+', index=False)

