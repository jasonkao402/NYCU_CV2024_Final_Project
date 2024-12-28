import logging
import cv2
import numpy as np
from math import cos, sin, pi

# Testing Code
import sys
import json
# from lib.h2pose.H2Pose import H2Pose
from numpy.linalg import inv
###########

np.set_printoptions(suppress=True)

class Hfinder(object):
    """docstring for Hfinder"""
    def __init__(self, camera_ks, dist, nmtx, court3D, court2D):
        super(Hfinder, self).__init__()
        self.camera_ks = camera_ks
        self.dist = dist
        self.nmtx = nmtx

        self.court2D = court2D

        self.court3D = court3D
        self.H = np.zeros((3,3)) # mapping 2D pixel to wcs 3D plane
        self.calculateH()

    def getH(self):
        return self.H


    def calculateH(self):
        self.court2D = np.array(self.court2D)
        self.court3D = np.array(self.court3D)

        logging.debug("{} points: {}\np.s. excluding padding\n".format(len(self.court3D),self.court2D))

        undistort_track2D = cv2.undistortPoints(np.array(np.expand_dims(self.court2D, 1), np.float32),
                                                np.array(self.camera_ks,np.float32),
                                                np.array(self.dist,np.float32),
                                                None,
                                                np.array(self.nmtx,np.float32))

        ## use solvePnP to calculate pose and eye
        ret, rvec, tvec = cv2.solvePnP(np.array(self.court3D,np.float32), np.array(self.court2D,np.float32), np.array(self.camera_ks,np.float32), np.array(self.dist,np.float32),flags = cv2.SOLVEPNP_ITERATIVE)
        #logging.debug('rain tvec:{}'.format(tvec))
        T = np.array(tvec,np.float32)
        R = np.array(cv2.Rodrigues(rvec)[0],np.float32)
        self.R = R
        self.T = T
        self.projection_mat = self.nmtx@np.concatenate((R,T),axis=1)
        self.H, status = cv2.findHomography(np.squeeze(undistort_track2D, axis=1), self.court3D)

    def getProjection_mat(self):
        return self.projection_mat

    def getExtrinsic_mat(self):
        return np.concatenate((self.R,self.T),axis=1)

