import numpy as np
import cv2

class MultiCamTriang(object):
    def __init__(self, poses, eye, Ks):
        super(MultiCamTriang, self).__init__()
        self.poses = poses
        self.eye = eye
        self.Ks = Ks
        self.f = (Ks[:,0,0] + Ks[:,1,1]) / 2
        self.p = Ks[:,0:2,2]
        # self.projection_mat = projection_mat

    def setTrack2Ds(self, newtrack2Ds):     # must set, then run calculate3D
        #logging.debug('setting track2Ds:{newtrack2Ds}')
        self.track2Ds = newtrack2Ds

    def setProjectionMats(self, ProjectionMats):
        self.projection_mat = ProjectionMats

    def calculate3D(self):
        total_track3Ds = np.zeros((1,3))
        track_3Ds = []
        for i in range(0,len(self.track2Ds)-1):
            for j in range(i+1,len(self.track2Ds)):
                self.track3D_homo = cv2.triangulatePoints(self.projection_mat[i],self.projection_mat[j],self.track2Ds[i][0],self.track2Ds[j][0]) # shape:(4,num_frame), num_frame=1
                self.track3D = self.track3D_homo[:3] / self.track3D_homo[3] # shape:(3,num_frame), num_frame=1
                self.track3D = np.stack(self.track3D, axis=1) # shape:(num_frame,3), num_frame=1
                track_3Ds.append(self.track3D)
                # logging.debug('i:{}, j:{}, self.track3D:{}'.format(i, j, self.track3D))

        track_3Ds = np.array(track_3Ds)
        n=1.5
        #IQR = Q3-Q1
        for i in range(0,3):
            IQR = np.percentile(track_3Ds[:,:,i],75) - np.percentile(track_3Ds[:,:,i],25)
            track_3Ds = track_3Ds[track_3Ds[:,:,i] <= np.percentile(track_3Ds[:,:,i],75)+n*IQR]
            track_3Ds = track_3Ds[:,np.newaxis,:]
        for i in track_3Ds:
            total_track3Ds += i
        return total_track3Ds/len(track_3Ds)