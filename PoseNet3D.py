""" Algorithm to be evaluated: VoxelPoseNet. """
from __future__ import unicode_literals, print_function
import sys
import pickle

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import cv2
import scipy.ndimage as ndimage

from utils.VoxelizationUtil import VoxelizationUtil as vu
from utils.Camera import *
from utils.paf_utils import *

from nets.VoxelPoseNet import *
from nets.OpenPoseCoco import *


class PoseNet3D(object):
    def __init__(self, ope_depth=5, gpu_id=0, gpu_memory_limit=None, vpn_type=None, K=None):
        self._session = None  # tensorflow session
        self._image = None  # input to the 2D network
        self._scoremaps_kp = None  # output of the 2D network
        self._scoremaps_paf = None  # output of the 2D network
        self._depth_vox = None  # input to the 3D network
        self._kp_uv = None  # input to the 3D network
        self._voxel_root_xyz = None  # input to the 3D network
        self._voxel_scale = None  # input to the 3D network
        self._cam_mat = None  # input to the 3D network
        self._kp_vox = None  # output of the 3D network

        # parameters
        self._intermediate_scoremap_size = (100, 100)  # map size used for warping 2D->3D
        self.conf2d_thresh = 0.5  # minimal confidence of 2d detection
        self.cam = Camera(K)

        self.ope_depth = ope_depth
        self.gpu_id = gpu_id
        self.gpu_memory_limit = gpu_memory_limit

        if vpn_type == 'fast':
            self.use_fast_vpn = True
        else:
            self.use_fast_vpn = False

        # create network (this sets some member variables)
        self._setup_network()  # creates the tensorflow graph)
        self._init_open_pose()  # loads weights

    def detect(self, image, depth_w, mask):
        """ Given RGBD input it predicts 3D human pose. """
        ## A) 2D network on the RGB input
        # 1. Preprocessing RGB: Get image to the right size
        image_proc, image_s, scale_ratio = self._preproc_color(image)
        #self._show_input(image_s, depth_w, block=False)

        # 2. Run 2D network
        scoremaps_kp_v, scoremaps_paf_v = self._session.run([self._scoremaps_kp, self._scoremaps_paf],
                                                            {self._image: image_proc})

        # 3. Postprocessing: Detect keypoints and use PAF to work out person instances
        keypoint_det = detect_keypoints(np.squeeze(scoremaps_kp_v, 0))  # detect keypoints in the one scoremap
        paf_u, paf_v = self._split_paf_scoremap(np.squeeze(scoremaps_paf_v, 0))  # split representation
        pairwise_scores = calculate_pair_scores(keypoint_det, paf_u, paf_v)  # Calculate matching scores with the pafs

        # upscale because we dont upsample the scoremaps anymore
        keypoint_det_fs = list()
        for x in keypoint_det:
            if x is not None:
                x[:, :2] *= 8.0
            keypoint_det_fs.append(x)
        person_det = group_keypoints(keypoint_det_fs, pairwise_scores)  # Use detections and pairwise scores to get final estimation
        # print('Found %d persons' % len(person_det))  # coords in: person_det['person0']['kp'] = None,  person_det['person1']['kp'] = (u, v)
        # self._show_openpose_det(image_s, person_det, block=False)

        ## B) 3D network: VOXELPOSENET
        coord_uv, coord2d_conf = self._trafo_dict2array(person_det)
        coord_vis = coord2d_conf > self.conf2d_thresh
        coord_uv_fs = coord_uv / scale_ratio

        coord_xyz, det_conf = list(), list()
        for pid in range(len(person_det)):
            # Check if neck keypoint is visible
            if coord_vis[pid, 1] == 1.0:
                root_id = 1  # neck keypoint
                coord2d_root = coord_uv_fs[pid, root_id, :]

                #asymmetric grid
                grid_size_m = np.array([[-1.1, -0.4, -1.1],
                                        [1.1, 1.8, 1.1]])

            # if not try R-hip
            elif coord_vis[pid, 8] == 1.0:
                root_id = 8  # R-hip keypoint
                coord2d_root = coord_uv_fs[pid, root_id, :]

                #symmetric grid
                grid_size_m = np.array([[-1.1, -1.1, -1.1],
                                        [1.1, 1.1, 1.1]])

            # if not try L-hip
            elif coord_vis[pid, 11] == 1.0:
                root_id = 11  # L-hip keypoint
                coord2d_root = coord_uv_fs[pid, root_id, :]

                #symmetric grid
                grid_size_m = np.array([[-1.1, -1.1, -1.1],
                                        [1.1, 1.1, 1.1]])
            else:
                continue

            # find approx. depth for root
            z_value = self._get_depth_value(depth_w / 1000.0, coord2d_root[0], coord2d_root[1])
            # self._show_sparse_depth(depth_w, coord_uv_fs[pid, 1, :], block=False)

            if z_value == 0.0:
                print("Could not extract depth value. Skipping sample.")
                continue

            # create voxel occupancy grid from the warped depth map
            voxelgrid, coord2d_s, trafo_params = vu.voxelize_person(self.cam, depth_w, mask,
                                                            coord_uv_fs[pid, :, :], coord_vis[pid, :],
                                                            z_value, (64, 64, 64), f=1.2,
                                                            grid_size_m=grid_size_m,
                                                            root_id=root_id, coord2d_root=coord2d_root)

            # 5. Run VoxelPoseNet
            feed_dict_vpn = {self._depth_vox: voxelgrid, self._kp_uv: coord2d_s, self._kp_vis: coord_vis[pid, :]}
            kp_scorevol_v = self._session.run(self._kp_vox, feed_dict_vpn)

            # 6. Postprocessing: Detect keypoints, transform back to XYZ
            keypoints_xyz_vox, det_conf_vox = self._detect_scorevol(kp_scorevol_v)
            keypoints_xyz_pred = vu.trafo_vox_coords_to_xyz_new(keypoints_xyz_vox, trafo_params)  # xyz from voxel
            keypoints_xyz_pred_proj = self.cam.backproject(coord_uv_fs[pid, :, :], keypoints_xyz_pred[:, -1:])  # xyz from backprojected uv

            # assemble solution from voxel result and backprojected solution
            cond = coord2d_conf[pid, :] > det_conf_vox  # use backproj only when 2d was visible and 2d/3d roughly matches
            keypoints_xyz_pred[cond, :] = keypoints_xyz_pred_proj[cond, :]
            coord_xyz.append(keypoints_xyz_pred)
            det_conf.append(det_conf_vox)
            # self._show_voxelposenet_det(voxelgrid, keypoints_xyz_vox)

        # output numpy arrays
        if len(coord_xyz) > 0:
            coord_xyz = np.stack(coord_xyz)
        else:
            coord_xyz = np.zeros((0, 18, 3))

        if len(det_conf) > 0:
            det_conf = np.stack(det_conf)
        else:
            det_conf = np.zeros((0, 18))

        return coord_xyz, det_conf

    def _setup_network(self):
        """ Creates the tensorflow graph structure. """
        # input placeholder
        self._image = tf.placeholder(tf.float32, (1, 376, 656, 3), 'image')
        self._depth_vox = tf.placeholder(tf.float32, (64, 64, 64), 'depth_vox')
        self._kp_uv = tf.placeholder(tf.float32, (18, 2), 'kp_uv')
        self._kp_vis = tf.placeholder(tf.float32, (18), 'kp_vis')

        self._voxel_root_xyz = tf.placeholder(tf.float32, (1, 3), 'voxel_root_xyz') # only for warped
        self._voxel_scale = tf.placeholder(tf.float32, (1, 3), 'voxel_scale') # only for warped
        self._cam_mat = tf.placeholder(tf.float32, (3, 3), 'cam_mat') # only for warped
        evaluation = tf.placeholder_with_default(True, shape=(), name='evaluation')

        # OpenPose colornet
        color_net = OpenPoseCoco()
        scoremaps_kp_list, scoremaps_paf_list = color_net.inference_pose(self._image, train=False, upsample=False,
                                                                         gpu_id=self.gpu_id)
        self._scoremaps_kp, self._scoremaps_paf = scoremaps_kp_list[self.ope_depth], scoremaps_paf_list[self.ope_depth]

        # VoxelPoseNet for Person Kp
        net = VoxelPoseNet(use_slim=self.use_fast_vpn)

        # calculate scoremap
        kp_map_uv = self._create_multiple_gaussian_map(self._kp_uv, (64, 64), 3.0, tf.expand_dims(self._kp_vis, -1))

        # tile 2D scoremap into a scorevolume
        det2d_vox = tf.tile(tf.expand_dims(tf.expand_dims(kp_map_uv, 0), 3), [1, 1, 1, 64, 1])

        self._kp_vox = net.inference(tf.expand_dims(tf.expand_dims(self._depth_vox, 0), -1),
                                     det2d_vox,
                                     evaluation, gpu_id=self.gpu_id)

        # start session load weights
        if self.gpu_memory_limit is None:
            self._session = tf.Session()
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
            self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self._init_voxelposenet()

    def _init_open_pose(self):
        """ Loads weights for the OpenPose network. """
        # create dict for renaming the weights
        name_dict = {'conv1_1': 'conv1_1',
                     'conv1_2': 'conv1_2',
                     'conv2_1': 'conv2_1',
                     'conv2_2': 'conv2_2',
                     'conv3_1': 'conv3_1',
                     'conv3_2': 'conv3_2',
                     'conv3_3': 'conv3_3',
                     'conv3_4': 'conv3_4',
                     'conv4_1': 'conv4_1',
                     'conv4_2': 'conv4_2',
                     'conv4_3_CPM': 'conv4_3',
                     'conv4_4_CPM': 'conv4_4'}

        for type_old, type_new in [('L1', 'paf'), ('L2', 'kp')]:
            for rep_id in range(1, 8):
                name_dict['conv5_%d_CPM_%s' % (rep_id, type_old)] = 'conv5_%d_%s' % (rep_id, type_new)

            for stage_id in range(2, 7):
                for rep_id in range(1, 8):
                    name_dict['Mconv%d_stage%d_%s' % (rep_id, stage_id, type_old)] = 'conv%d_%d_%s' % (stage_id + 4, rep_id, type_new)

        weight_dict = dict()
        with open('./weights/openpose-coco.pkl', 'rb') as fi:
            if sys.version_info[0] == 3:
                weight_dict_raw = pickle.load(fi, encoding='latin1')  # for python3
            else:
                weight_dict_raw = pickle.load(fi)  # for python2

            for k, v in weight_dict_raw.items():
                if k in name_dict.keys():
                    new_name = name_dict[k]
                    weight_dict['CocoPoseNet/' + new_name + '/weights'] = v[0]
                    weight_dict['CocoPoseNet/' + new_name + '/biases'] = v[1]
                else:
                    print('Skipping: ', k)

            init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
            self._session.run(init_op, init_feed)
            print('Initialized 2D network with %d variables' % len(weight_dict))

    def _init_voxelposenet(self):
        """ Initializes the VoxelPoseNet from a snapshot. """
        if self.use_fast_vpn:
            checkpoint_path = './weights/snapshots_pose_run194/'
        else:
            checkpoint_path = './weights/snapshots_pose_run191/'
        rename_dict = {}
        discard_list = ['Adam', 'global_step', 'beta']
        self._load_all_variables_from_snapshot(checkpoint_path, rename_dict, discard_list)

    def _load_all_variables_from_snapshot(self, checkpoint_path, rename_dict=None, discard_list=None):
        """ Initializes certain tensors from a snapshot. """
        last_cpt = tf.train.latest_checkpoint(checkpoint_path)
        assert last_cpt is not None, "Could not locate snapshot to load."
        reader = pywrap_tensorflow.NewCheckpointReader(last_cpt)
        var_to_shape_map = reader.get_variable_to_shape_map()  # var_to_shape_map

        # for name in var_to_shape_map.keys():
        #     print(name, reader.get_tensor(name).shape)

        # Remove everything from the discard list
        num_disc = 0
        var_to_shape_map_new = dict()
        for k, v in var_to_shape_map.items():
            good = True
            for dis_str in discard_list:
                if dis_str in k:
                    good = False

            if good:
                var_to_shape_map_new[k] = v
            else:
                num_disc += 1
        var_to_shape_map = dict(var_to_shape_map_new)
        print('Discarded %d items' % num_disc)
        # print('Vars in checkpoint', var_to_shape_map.keys(), len(var_to_shape_map))

        # rename everything according to rename_dict
        num_rename = 0
        if rename_dict is not None:
            var_to_shape_map_new = dict()
            for name in var_to_shape_map.keys():
                rename = False
                for rename_str in rename_dict.keys():
                    if rename_str in name:
                        new_name = name.replace(rename_str, rename_dict[rename_str])
                        var_to_shape_map_new[new_name] = reader.get_tensor(name)
                        rename = True
                        num_rename += 1
                        break
                if not rename:
                    var_to_shape_map_new[name] = reader.get_tensor(name)
            var_to_shape_map = dict(var_to_shape_map_new)
        print('Renamed %d items' % num_rename)
        # print('(Possibly) renamed vars', var_to_shape_map.keys(), len(var_to_shape_map))

        init_op, init_feed = tf.contrib.framework.assign_from_values(var_to_shape_map)
        self._session.run(init_op, init_feed)
        print('Initialized %d variables from %s.' % (len(var_to_shape_map), last_cpt))

    @staticmethod
    def _preproc_color(image):
        """ Preprocessing the color image"""
        output_shape = np.array([376.0, 656.0], np.float32)

        # reshape by trafo
        ratio = np.min(output_shape / np.array(image.shape[:2], dtype=np.float32))
        M = np.array([[ratio, 0.0, 0.0], [0.0, ratio, 0.0]])
        image_s = cv2.warpAffine(image, M, (output_shape[1], output_shape[0]), flags=cv2.INTER_AREA)
        
        # subtract mean and rgb -> bgr
        image = image_s[:, :, 0:3].astype('float32')
        image = image[:, :, ::-1]
        image = image / 256.0 - 0.5
        image = np.expand_dims(image, 0)
        return image, image_s, ratio

    def _detect_keypoints(self, scoremap):
        """
        Takes a scoremap and finds locations for keypoints.
        Returns a KxNx2 matrix with the (u, v) coordinates of the N maxima found for the K keypoints.
        """
        assert len(scoremap.shape) == 3, "Needs to be a 3D scoremap."

        keypoint_loc = list()
        for kid in range(scoremap.shape[2]):
            num_kp, maxima = self._find_maxima(scoremap[:, :, kid])
            if num_kp > 0:
                keypoint_loc.append(maxima)
            else:
                keypoint_loc.append(None)
        return keypoint_loc

    @staticmethod
    def _find_maxima(scoremap):
        """
        Takes a scoremap and detect the peaks using the local maximum filter.
        Returns a Nx2 matrix with the (u, v) coordinates of the N maxima found.
        """
        assert len(scoremap.shape) == 2, "Needs to be a 2D scoremap."

        # apply the local maximum filter; all pixel of maximal value
        local_max = maximum_filter(scoremap, size=3)
        mask_max = scoremap == local_max

        # mask out background
        mask_bg = ((np.max(scoremap) - np.min(scoremap)) * 0.25) > scoremap
        mask_max[mask_bg] = False

        # find distinct objects in map
        labeled, num_objects = ndimage.label(mask_max)
        slices = ndimage.find_objects(labeled)

        # create matrix of found objects with their location
        maxima = np.zeros((num_objects, 3), dtype=np.float32)
        for oid, (dy, dx) in enumerate(slices):
            maxima[oid, :2] = [(dx.start + dx.stop - 1)/2, (dy.start + dy.stop - 1)/2]
            u, v = int(maxima[oid, 0] + 0.5), int(maxima[oid, 1] + 0.5)
            maxima[oid, 2] = scoremap[v, u]

        return num_objects, maxima

    @staticmethod
    def _split_paf_scoremap(scoremap_paf):
        """ The network outputs u1, v1, u2, v2, ... """
        paf_u, paf_v = scoremap_paf[:, :, ::2], scoremap_paf[:, :, 1::2]
        return paf_u, paf_v

    @staticmethod
    def _show_input(image, depth, block=True):
        """ Shows the input to the pipeline. """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(image)
        ax2.imshow(depth)
        plt.show(block=block)

    @staticmethod
    def _show_sparse_depth(depth, coord_uv, block=True):
        """ Shows the where from the sparse depth map we take a value. """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(depth)
        ax1.plot(coord_uv[0], coord_uv[1], 'ro')
        print('coord_uv', coord_uv)
        plt.show(block=block)

    @staticmethod
    def _show_openpose_det(image_s, person_det, block=True):
        """ Shows the detections of openpose in the color image. """
        import matplotlib.pyplot as plt
        from utils.DrawUtil import draw_person_limbs_2d_coco
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(image_s)
        fmt_list = ['ro', 'go', 'co', 'mo']
        for anno, fmt in zip(person_det.values(), fmt_list):
            coords = np.zeros((18, 2))
            vis = np.zeros((18, ))
            for i, kp in enumerate(anno['kp']):
                if (kp is not None) and (kp[2] > 0.5):
                    # ax1.plot(kp[0], kp[1], fmt)
                    coords[i, :] = np.array([kp[0], kp[1]])
                    vis[i] = 1.0

            draw_person_limbs_2d_coco(ax1, coords, vis, color='sides', order='uv')
        plt.show(block=block)

    @staticmethod
    def _show_voxelposenet_det(voxelgrid, keypoints_xyz_vox, block=True):
        """ Shows the detections of VoxelPoseNet in the input voxelgrid. """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from utils.DrawUtil import draw_person_limbs_3d_coco

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y, Z = np.where(voxelgrid)
        ax.scatter(X, Y, Z, c='g')
        draw_person_limbs_3d_coco(ax, keypoints_xyz_vox, color='r')
        plt.show(block=block)

    def _get_depth_value(self, map, u, v, crop_size=25):
        """ Extracts a depth value from a map.
            Checks for the closest value in a given neighborhood. """
        coord = np.array([[u, v]])

        while True:
            # get crop
            map_c, min_c, _ = self._crop_at(map, coord, crop_size)
            center = coord - min_c

            # find valid depths
            X, Y = np.where(np.not_equal(map_c, 0.0))

            if not X.shape[0] == 0:
                break
            crop_size *= 2  # if not successful use larger crop

        # calculate distance
        grid = np.stack([X, Y], 1) - center
        dist = np.sqrt(np.sum(np.square(grid), 1))

        # find element with minimal distance
        nn_ind = np.argmin(dist)
        x, y = X[nn_ind], Y[nn_ind]
        z_val = map_c[x, y]
        return z_val

    @staticmethod
    def _crop_at(map, center_coord, size):
        """ Crop a given map at the given center coordinate with the crop size specified.
            If the cropped area would partially reside outside of the map it is translated accordingly. """
        expand = False
        s = map.shape
        if len(s) == 2:
            map = np.expand_dims(map, 2)
            expand = True
            s = map.shape
        assert len(s) == 3, "Map has to be of Dimension 2 or 3."

        size = np.round(size).astype('int')

        # make sure crop size cant exceed image dims
        if s[0] <= size:
            size = s[0]
        elif s[1] <= size:
            size = s[1]

        center_coord = np.array(center_coord)
        center_coord = np.reshape(center_coord, [2])

        # work out the coords to actually lie in the crop
        c_min = np.round(center_coord - size // 2).astype('int')
        c_max = c_min + size

        # check if we left map
        for dim in [0, 1]:
            if c_min[dim] < 0.0:
                c_max[dim] -= c_min[dim]
                c_min[dim] = 0.0
            if c_max[dim] > s[1-dim]:
                c_min[dim] -= (c_max[dim]-s[1-dim])
                c_max[dim] = s[1-dim]

        # perform crop
        map_crop = map[c_min[1]:c_max[1], c_min[0]:c_max[0], :]

        if expand:
            map_crop = np.squeeze(map_crop)

        return map_crop, c_min, c_max

    @staticmethod
    def _trafo_dict2array(person_det, kp_conf_thresh=0.25):
        """ Transforms the dictionary returned from openpose into an array. """
        coord_uv = np.zeros((len(person_det), 18, 2))
        coord_vis = np.zeros((len(person_det), 18))

        for pid, anno in enumerate(person_det.values()):
            for kid, kp in enumerate(anno['kp']):
                if (kp is not None) and (kp[2] > kp_conf_thresh):
                    coord_vis[pid, kid] = kp[2]
                    coord_uv[pid, kid, :] = np.array([kp[0], kp[1]])
                else:
                    coord_vis[pid, kid] = 0.0
        return coord_uv, coord_vis

    @staticmethod
    def _create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None):
        """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
            with variance sigma for multiple coordinates."""
        with tf.name_scope('create_multiple_gaussian_map'):
            sigma = tf.cast(sigma, tf.float32)
            assert len(output_size) == 2
            s = coords_uv.get_shape().as_list()
            coords_uv = tf.cast(coords_uv, tf.int32)
            if valid_vec is not None:
                valid_vec = tf.cast(valid_vec, tf.float32)
                valid_vec = tf.squeeze(valid_vec)
                cond_val = tf.greater(valid_vec, 0.5)
            else:
                cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
                cond_val = tf.greater(cond_val, 0.5)

            cond_1_in = tf.logical_and(tf.less(coords_uv[:, 0], output_size[0]-1), tf.greater(coords_uv[:, 0], 0))
            cond_2_in = tf.logical_and(tf.less(coords_uv[:, 1], output_size[1]-1), tf.greater(coords_uv[:, 1], 0))
            cond_in = tf.logical_and(cond_1_in, cond_2_in)
            cond = tf.logical_and(cond_val, cond_in)

            coords_uv = tf.cast(coords_uv, tf.float32)

            # create meshgrid
            x_range = tf.expand_dims(tf.range(output_size[0]), 1)
            y_range = tf.expand_dims(tf.range(output_size[1]), 0)

            X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
            Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

            X.set_shape((output_size[0], output_size[1]))
            Y.set_shape((output_size[0], output_size[1]))

            X = tf.expand_dims(X, -1)
            Y = tf.expand_dims(Y, -1)

            X_b = tf.tile(X, [1, 1, s[0]])
            Y_b = tf.tile(Y, [1, 1, s[0]])

            X_b -= coords_uv[:, 0]
            Y_b -= coords_uv[:, 1]

            dist = tf.square(X_b) + tf.square(Y_b)

            scoremap = tf.exp(-dist / tf.square(sigma)) * tf.cast(cond, tf.float32)

            return scoremap

    @staticmethod
    def _detect_scorevol(scorevolume):
        """ Finds maximum volumewise. Tensor scorevolume is [1, D, H, W, C]. """
        scorevolume = np.squeeze(scorevolume)
        s = scorevolume.shape
        assert len(s) == 4, "Tensor must be 4D"

        coord_det = list()
        coord_conf = list()
        for i in range(s[3]):
            max_val = np.amax(scorevolume[:, :, :, i])

            ind = np.where(scorevolume[:, :, :, i] == max_val)
            ind = [np.median(x) for x in ind]  # this eliminates, when there are multiple maxima
            ind = [int(x) for x in ind]
            coord_conf.append(max_val)
            coord_det.append(ind)
        coord_det = np.reshape(np.array(coord_det), [-1, 3])
        coord_conf = np.array(coord_conf)
        return coord_det, coord_conf

