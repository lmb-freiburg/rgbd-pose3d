import numpy as np
import matplotlib.cm


def draw_person_limbs_2d_coco(axis, coords, vis=None, color=None, order='hw', with_face=True):
    """ Draws a 2d person stick figure in a matplotlib axis. """
    import matplotlib.cm
    if order == 'uv':
        pass
    elif order == 'hw':
        coords = coords[:, ::-1]
    else:
        assert 0, "Unknown order."

    LIMBS_COCO = np.array([[1, 2], [2, 3], [3, 4],  # right arm
                           [1, 8], [8, 9], [9, 10],  # right leg
                           [1, 5], [5, 6], [6, 7],  # left arm
                           [1, 11], [11, 12], [12, 13],  # left leg
                           [1, 0], [2, 16], [0, 14], [14, 16], [0, 15], [15, 17], [5, 17]])  # head

    if type(color) == str:
        if color == 'sides':
            blue_c = np.array([[0.0, 0.0, 1.0]])  # side agnostic
            red_c = np.array([[1.0, 0.0, 0.0]])  # "left"
            green_c = np.array([[0.0, 1.0, 0.0]])  # "right"
            color = np.concatenate([np.tile(green_c, [6, 1]),
                                    np.tile(red_c, [6, 1]),
                                    np.tile(blue_c, [7, 1])], 0)
            if not with_face:
                color = color[:13, :]

    if not with_face:
        LIMBS_COCO = LIMBS_COCO[:13, :]

    if vis is None:
        vis = np.ones_like(coords[:, 0]) == 1.0

    if color is None:
        color = matplotlib.cm.jet(np.linspace(0, 1, LIMBS_COCO.shape[0]))[:, :3]

    for lid, (p0, p1) in enumerate(LIMBS_COCO):
        if (vis[p0] == 1.0) and (vis[p1] == 1.0):
            if type(color) == str:
                axis.plot(coords[[p0, p1], 0], coords[[p0, p1], 1], color, linewidth=2)
            else:
                axis.plot(coords[[p0, p1], 0], coords[[p0, p1], 1], color=color[lid, :], linewidth=2)


def draw_person_limbs_3d_coco(axis, coords, vis=None, color=None, orientation=None, orientation_val=None, with_face=True, rescale=True):
    """ Draws a 3d person stick figure in a matplotlib axis. """

    import matplotlib.cm

    LIMBS_COCO = np.array([[1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
                           [6, 7], [1, 8], [8, 9], [9, 10],
                           [1, 11], [11, 12], [12, 13], [1, 0],
                           [2, 16], [0, 14], [14, 16], [0, 15], [15, 17], [5, 17]])

    if not with_face:
        LIMBS_COCO = LIMBS_COCO[:13, :]

    if vis is None:
        vis = np.ones_like(coords[:, 0]) == 1.0

    vis = vis == 1.0

    if color is None:
        color = matplotlib.cm.jet(np.linspace(0, 1, LIMBS_COCO.shape[0]))[:, :3]

    for lid, (p0, p1) in enumerate(LIMBS_COCO):
        if (vis[p0] == 1.0) and (vis[p1] == 1.0):
            if type(color) == str:
                axis.plot(coords[[p0, p1], 0], coords[[p0, p1], 1], coords[[p0, p1], 2], color, linewidth=2)
            else:
                axis.plot(coords[[p0, p1], 0], coords[[p0, p1], 1], coords[[p0, p1], 2], color=color[lid, :], linewidth=2)

    if np.sum(vis) > 0 and rescale:
        min_v, max_v, mean_v = np.min(coords[vis, :], 0), np.max(coords[vis, :], 0), np.mean(coords[vis, :], 0)
        range = np.max(np.maximum(np.abs(max_v-mean_v), np.abs(mean_v-min_v)))
        axis.set_xlim([mean_v[0]-range, mean_v[0]+range])
        axis.set_ylim([mean_v[1]-range, mean_v[1]+range])
        axis.set_zlim([mean_v[2]-range, mean_v[2]+range])

    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_zlabel('z')
    axis.view_init(azim=-90., elev=-90.)


def detect_hand_keypoints(scoremaps):
    """ Performs detection per scoremap for the hands keypoints. """
    if len(scoremaps.shape) == 4:
        scoremaps = np.squeeze(scoremaps)
    s = scoremaps.shape
    assert len(s) == 3, "This function was only designed for 3D Scoremaps."
    assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    keypoint_coords = np.zeros((s[2], 2))
    for i in range(s[2]):
        v, u = np.unravel_index(np.argmax(scoremaps[:, :, i]), (s[0], s[1]))
        keypoint_coords[i, 0] = v
        keypoint_coords[i, 1] = u
    return keypoint_coords


def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)


def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    axis.view_init(azim=-90.0, elev=-90.0)
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_zlabel('z')