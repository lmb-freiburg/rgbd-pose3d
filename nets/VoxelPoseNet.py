import tensorflow as tf

from utils import NetworkOps

"""
    Network for estimating a 3D skeleton from voxelized depth maps.
"""
class VoxelPoseNet:
    def __init__(self, use_slim=False):
        """ Contains parameters used by the network. """
        # define some parameters here
        self.num_kp = 18
        self.use_slim = use_slim

    def inference(self, depth_vox, det2d_vox, evaluation, gpu_id):
        """ Inference part of the network. """
        total_input = tf.concat([depth_vox, det2d_vox], 4)
        scorevolume_list = self.inference_keypoint(total_input, evaluation, gpu_id)
        return scorevolume_list[-1]

    def inference_keypoint(self, depth_vox, evaluation, gpu_id, train=False):
        """ Infer normalized depth coordinate. """
        ops = NetworkOps.NetworkOps
        with tf.device('/device:GPU:%d' % gpu_id):
            with tf.variable_scope('VoxelPoseNet'):
                scorevolume_list = list()
                # input
                s = depth_vox.get_shape().as_list()
                x = depth_vox

                # ENCODER: DenseNet style
                if self.use_slim:
                    conv_per_block = [1, 1, 2]
                else:
                    conv_per_block = [2, 2, 2]
                out_chan_per_block = [64, 128, 256]
                pool_list = [True, True, True]

                skips = list()
                # ENCODER:
                for block_id, (layer_num, chan_num, pool) in enumerate(zip(conv_per_block, out_chan_per_block, pool_list), 1):
                    x_list = [x]
                    for layer_id in range(layer_num):
                        x = tf.concat(x_list, 4)
                        x = ops.conv3_relu(x, 'conv%d_%d' % (block_id, layer_id+1),
                                           kernel_size=3, stride=1, out_chan=chan_num)
                        x_list.append(x)

                    if pool:
                        x = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID', name='pool%d' % block_id)

                    skips.append(x)

                # DECODER: Use skip connections to get the details right
                skips.pop()
                skips = skips[::-1]
                scorevol_init = None
                num_skips = len(skips)
                for block_id, skip_x in enumerate(skips, 1):
                    # upconv to full size (for loss)
                    kernel_size = 2**(num_skips + 3 - block_id)
                    stride = 2**(num_skips + 2 - block_id)
                    # print('Intermediate loss from', x)
                    x_det = ops.conv3(x, 'det%d' % block_id, kernel_size=1, stride=1, out_chan=self.num_kp)
                    # print('Upconv with', kernel_size, stride)
                    scorevolume_fs = ops.upconv3d(x_det, 'scorevolume%d' % block_id, kernel_size=kernel_size, stride=stride,
                                                  output_shape=[s[0], s[1], s[2], s[3], self.num_kp])
                    # print('scorevolume_fs %d' % block_id, scorevolume_fs)
                    if scorevol_init is None:
                        scorevol_init = scorevolume_fs
                        scorevolume_list.append(scorevolume_fs)
                    else:
                        scorevol_init = scorevol_init + scorevolume_fs
                        scorevolume_list.append(scorevol_init)

                    # upconv to next layer and incorporate the skip connection
                    s_x = x.get_shape().as_list()
                    s_s = skip_x.get_shape().as_list()
                    x_up = ops.upconv3d(x, 'upconv%d' % block_id, kernel_size=4, stride=2,
                                        output_shape=[s_s[0], s_s[1], s_s[2], s_s[3], s_x[4]])
                    x = tf.concat([x_up, skip_x], 4)
                    x = ops.conv3_relu(x, 'conv_decoder%d' % block_id,
                                       kernel_size=3, stride=1, out_chan=s_s[4])

                # Final estimation
                x = ops.upconv3d(x, 'upconv_final', kernel_size=4, stride=2,
                                 output_shape=[s[0], s[1], s[2], s[3], s_s[4]])
                scorevolume = ops.conv3(x, 'scorevolume_final', kernel_size=1, stride=1, out_chan=self.num_kp)
                scorevol_init = scorevol_init + scorevolume
                scorevolume_list.append(scorevol_init)

                return scorevolume_list

