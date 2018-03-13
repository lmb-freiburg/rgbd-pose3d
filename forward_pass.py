#!/usr/bin/env python
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

from PoseNet3D import *
from utils.Camera import *

# VALUES YOU MIGHT WANT TO CHANGE
OPE_DEPTH = 1  # in [1, 5]; Number of stages for the 2D network. Smaller number makes the network faster but less accurate
VPN_TYPE = 'fast'  # in {'fast', 'default'}; which 3D architecture to use
CONF_THRESH = 0.25  # threshold for a keypoint to be considered detected (for visualization)
GPU_ID = 0  # id of gpu device
GPU_MEMORY = None  # in [0.0, 1.0 - eps]; percentage of gpu memory that should be used; None for no limit
# NO CHANGES BEYOND THIS LINE

if __name__ == '__main__':
    """  APPROX. RUNTIMES (measured on a GTX 1080 Ti, frame with 4 people)
    VPN=fast, OPE=1: 0.51sec = 1.96 Hz
    VPN=fast, OPE=5: 0.56sec = 1.79 Hz
    VPN=default, OPE=1: 0.59sec = 1.70 Hz
    VPN=default, OPE=5: 0.64sec = 1.57 Hz

    APPROX. RUNTIMES (measured on a GTX 970, frame with 4 people)
    VPN=fast, OPE=1: 1.20 = 0.84 Hz
    VPN=fast, OPE=5: 1.30 sec = 0.77 Hz
    VPN=default, OPE=1: 1.41sec = 0.71 Hz
    VPN=default, OPE=5: 1.54sec = 0.65 Hz

    NOTE: Runtime scales with the number of people in the scene.
    """
    # load data
    color = scipy.misc.imread('./color.png')  # color image
    color = scipy.misc.imresize(color, (1080, 1920))
    depth_w = scipy.misc.imread('./depth.png').astype('float32')  # depth map warped into the color frame

    # intrinsic calibration data
    ratio = np.array([1920.0/512.0, 1080.0/424.0])
    K = np.array([[3.7132019636619111e+02 * ratio[0], 0.0, 2.5185416982679811e+02 * ratio[0]],
                   [0.0, 3.7095047063504268e+02 * ratio[1], 2.1463524817996452e+02 * ratio[1]],
                   [0.0, 0.0, 1.0]])
    cam = Camera(K)

    # create algorithm
    poseNet = PoseNet3D(ope_depth=OPE_DEPTH, vpn_type=VPN_TYPE,
                        gpu_id=GPU_ID, gpu_memory_limit=GPU_MEMORY, K=K)

    # loop
    mask = np.logical_not(depth_w == 0.0)

    # run algorithm
    coords_pred, det_conf = poseNet.detect(color, depth_w, mask)

    # visualize
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(color)
    for i in range(coords_pred.shape[0]):
        coord2d = cam.project(coords_pred[i, :, :])
        vis = det_conf[i, :] > CONF_THRESH
        ax.plot(coord2d[vis, 0], coord2d[vis, 1], 'ro')
    plt.show()
