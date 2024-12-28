import logging
import cv2
import numpy as np
from math import cos, sin, pi

np.set_printoptions(suppress=True)

class CameraMatrixFinder(object):
    """docstring for Hfinder"""
    def __init__(self, camera_ks, dist, court3D, court2D):
        super(CameraMatrixFinder, self).__init__()
        self.camera_ks = camera_ks
        self.dist = dist

        self.court2D = court2D

        self.court3D = court3D
        self.calculate()


    def calculate(self):
        self.court2D = np.array(self.court2D)
        self.court3D = np.array(self.court3D)

        logging.debug("{} points: {}\np.s. excluding padding\n".format(len(self.court3D),self.court2D))

        ## use solvePnP to calculate pose and eye
        ret, rvec, tvec = cv2.solvePnP(np.array(self.court3D,np.float32), np.array(self.court2D,np.float32), np.array(self.camera_ks,np.float32), np.array(self.dist,np.float32),flags = cv2.SOLVEPNP_ITERATIVE)
        #logging.debug('rain tvec:{}'.format(tvec))
        T = np.array(tvec,np.float32)
        R = np.array(cv2.Rodrigues(rvec)[0],np.float32)
        self.R = R
        self.T = T
        self.projection_mat = self.camera_ks@np.concatenate((R,T),axis=1)

    def getProjection_mat(self):
        return self.projection_mat

    def getExtrinsic_mat(self):
        return np.concatenate((self.R,self.T),axis=1)

