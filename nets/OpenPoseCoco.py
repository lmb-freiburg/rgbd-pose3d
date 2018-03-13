import tensorflow as tf

from utils import NetworkOps

"""
    Network for person keypoint detection based on color information
"""
class OpenPoseCoco:
    def __init__(self):
        """ Contains parameters used by the network. """
        # define some parameters here
        self.num_kp = 19
        self.num_limbs = 38

    def inference_pose(self, image_crop, train=True, upsample=True, gpu_id=0):
        """ Inference part of the network. """
        ops = NetworkOps.NetworkOps
        ops.neg_slope_of_relu = 0.0
        with tf.device('/device:GPU:%d' % gpu_id):
            with tf.variable_scope('CocoPoseNet'):
                scoremap_kp_list = list()
                scoremap_paf_list = list()
                layers_per_block = [2, 2, 4, 2]
                out_chan_list = [64, 128, 256, 512]
                pool_list = [True, True, True, False]
                retrain_layers = [False, False, False, False]

                # learn some feature representation, that describes the image content well
                x = image_crop

                for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                    for layer_id in range(layer_num):
                        x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1,
                                          out_chan=chan_num, trainable=retrain_layers[block_id-1])
                    if pool:
                        x = ops.max_pool(x, 'pool%d' % block_id)

                x = ops.conv_relu(x, 'conv4_3', kernel_size=3, stride=1, out_chan=256, trainable=train)
                encoding = ops.conv_relu(x, 'conv4_4', kernel_size=3, stride=1, out_chan=128, trainable=train)

                # use encoding to detect initial scoremaps kp
                x = ops.conv_relu(encoding, 'conv5_1_kp', kernel_size=3, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv5_2_kp', kernel_size=3, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv5_3_kp', kernel_size=3, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv5_4_kp', kernel_size=1, stride=1, out_chan=512, trainable=train)
                scoremap_kp = ops.conv(x, 'conv5_5_kp', kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
                scoremap_kp_list.append(scoremap_kp)

                # use encoding to detect initial scoremaps kp
                x = ops.conv_relu(encoding, 'conv5_1_paf', kernel_size=3, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv5_2_paf', kernel_size=3, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv5_3_paf', kernel_size=3, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv5_4_paf', kernel_size=1, stride=1, out_chan=512, trainable=train)
                scoremap_paf = ops.conv(x, 'conv5_5_paf', kernel_size=1, stride=1, out_chan=self.num_limbs, trainable=train)
                scoremap_paf_list.append(scoremap_paf)

                # iterate refinement part a couple of times
                num_recurrent_units = 5
                layers_per_recurrent_unit = 5
                for pass_id in range(num_recurrent_units):
                    # keypoints
                    x = tf.concat([scoremap_paf_list[-1], scoremap_kp_list[-1], encoding], 3)
                    for layer_id in range(layers_per_recurrent_unit):
                        x = ops.conv_relu(x, 'conv%d_%d_kp' % (pass_id+6, layer_id+1), kernel_size=7, stride=1, out_chan=128, trainable=train)
                    x = ops.conv_relu(x, 'conv%d_%d_kp' % (pass_id+6, 6), kernel_size=1, stride=1, out_chan=128, trainable=train)
                    scoremap_kp = ops.conv(x, 'conv%d_%d_kp' % (pass_id+6, 7), kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)

                    # paf
                    x = tf.concat([scoremap_paf_list[-1], scoremap_kp_list[-1], encoding], 3)
                    for layer_id in range(layers_per_recurrent_unit):
                        x = ops.conv_relu(x, 'conv%d_%d_paf' % (pass_id+6, layer_id+1), kernel_size=7, stride=1, out_chan=128, trainable=train)
                    x = ops.conv_relu(x, 'conv%d_%d_paf' % (pass_id+6, 6), kernel_size=1, stride=1, out_chan=128, trainable=train)
                    scoremap_paf = ops.conv(x, 'conv%d_%d_paf' % (pass_id+6, 7), kernel_size=1, stride=1, out_chan=self.num_limbs, trainable=train)

                    scoremap_kp_list.append(scoremap_kp)
                    scoremap_paf_list.append(scoremap_paf)

                # upsample to full size
                if upsample:
                    s = image_crop.get_shape().as_list()
                    scoremap_kp_list = [tf.image.resize_images(x, (s[1], s[2]), tf.image.ResizeMethod.BICUBIC, align_corners=True) for x in scoremap_kp_list]
                    scoremap_paf_list = [tf.image.resize_images(x, (s[1], s[2]), tf.image.ResizeMethod.BICUBIC, align_corners=True) for x in scoremap_paf_list]

        return scoremap_kp_list, scoremap_paf_list

    @staticmethod
    def _index_scoremaps(scoremap):
        scoremap_min_list = list()
        for i in range(21):
            min_val = tf.reduce_min(scoremap[:, :, :, i])
            max_val = tf.reduce_max(scoremap[:, :, :, i])
            thresh = (max_val-min_val) * 0.8 + min_val
            det_map = tf.cast(tf.greater(scoremap[:, :, :, i], thresh), tf.float32)
            scoremap_min_list.append(det_map)

        scoremap_min = tf.reduce_sum(tf.pack(scoremap_min_list, 3), 3, keep_dims=True)
        return scoremap_min


if __name__ == '__main__':
    import pickle
    import data.Types

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

    # Create network and load weights to create a snapshot from it
    net = OpenPoseCoco()

    image_tf = tf.placeholder(tf.float32, shape=(1, 368, 368, 3))
    evaluation = tf.placeholder_with_default(False, shape=(), name='evaluation')
    net.inference({data.Types.data_gt_t.image: image_tf}, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.initialize_all_variables()
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=0)

    weight_dict = dict()
    with open('./weights/openpose/openpose-coco.pkl', 'rb') as fi:
        weight_dict_raw = pickle.load(fi, encoding='latin1')

        for k, v in weight_dict_raw.items():
            if k in name_dict.keys():
                new_name = name_dict[k]
                weight_dict['CocoPoseNet/' + new_name + '/weights'] = v[0]
                weight_dict['CocoPoseNet/' + new_name + '/biases'] = v[1]
            else:
                print('Skipping: ', k)
            # print('new_weight', k, new_name, v[0].shape, v[1].shape)
        # print('weight dict', weight_dict.keys())

        vars = tf.trainable_variables()
        init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
        sess.run(init_op, init_feed)

    saver.save(sess, './weights/OpenPoseCoco-Init')
    print('Saved initial snapshot to disk.')
