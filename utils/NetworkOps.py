import tensorflow as tf
import numpy as np
import math


class NetworkOps(object):
    neg_slope_of_relu = 0.01

    @classmethod
    def leaky_relu(cls, tensor, name='relu'):
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu*tensor, name=name)
        return out_tensor

    @classmethod
    def conv3(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            tf.add_to_collection('shapes_for_memory', in_tensor)

            strides = [1, stride, stride, stride, 1]
            kernel_shape = [kernel_size, kernel_size, kernel_size, in_size[4], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                    uniform=True)
                                     , trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv3d(in_tensor, kernel, strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[4]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv3_relu(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        tensor = cls.conv3(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def conv3_z_stride_only(cls, in_tensor, layer_name, kernel_size_xy, kernel_size_z, stride_z, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            tf.add_to_collection('shapes_for_memory', in_tensor)

            strides = [1, 1, 1, stride_z, 1]
            kernel_shape = [kernel_size_xy, kernel_size_xy, kernel_size_z, in_size[4], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                    uniform=True)
                                     , trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv3d(in_tensor, kernel, strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[4]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv3_z_stride_only_relu(cls, in_tensor, layer_name, kernel_size_xy, kernel_size_z, stride_z, out_chan, trainable=True):
        tensor = cls.conv3_z_stride_only(in_tensor, layer_name, kernel_size_xy, kernel_size_z, stride_z, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def conv(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            tf.add_to_collection('shapes_for_memory', in_tensor)

            strides = [1, stride, stride, 1]
            kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv2d(in_tensor, kernel, strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv_sparse(cls, in_tensor, in_mask, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            # mask out invalid elements
            in_tensor = tf.mul(in_tensor, in_mask)

            in_size = in_tensor.get_shape().as_list()
            tf.add_to_collection('shapes_for_memory', in_tensor)

            strides = [1, stride, stride, 1]
            kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv2d(in_tensor, kernel, strides, padding='SAME')

            # weighting
            weight_kernel = tf.ones((kernel_size, kernel_size, 1, 1))  #this is a kernel of 1's same spatial shape as kernel
            weights = tf.nn.conv2d(in_mask, weight_kernel, strides, padding='SAME')  # simply calcs the num of 1's in mask
            tmp_result = tmp_result * kernel_size*kernel_size / (weights + 1e-8)  # normalize result

            # bias
            biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            # out_mask
            out_mask = tf.nn.max_pool(in_mask, ksize=[1, kernel_size, kernel_size, 1], strides=strides, padding='SAME', name='mask_pool')
            out_mask = tf.stop_gradient(out_mask)

            return out_tensor, out_mask

    @classmethod
    def conv_relu(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        tensor = cls.conv(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def conv_sparse_relu(cls, in_tensor, in_mask, layer_name, kernel_size, stride, out_chan, trainable=True):
        tensor, out_mask = cls.conv_sparse(in_tensor, in_mask, layer_name, kernel_size, stride, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor, out_mask

    @classmethod
    def conv_vgg(cls, in_tensor, layer_name, trainable=True, init_dict=None):
        with tf.variable_scope(layer_name):
            tf.add_to_collection('shapes_for_memory', in_tensor)

            # conv
            kernel = tf.Variable(init_dict[layer_name][0], name='weights', trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv2d(in_tensor, kernel, [1, 1, 1, 1], padding='SAME')

            # bias
            biases = tf.Variable(init_dict[layer_name][1], name='biases', trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv_relu_vgg(cls, in_tensor, layer_name, trainable=True, init_dict=None):
        tensor = cls.conv_vgg(in_tensor, layer_name, trainable, init_dict)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def max_pool(cls, bottom, name='pool'):
        pooled = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='VALID', name=name)
        return pooled

    @classmethod
    def avg_pool(cls, bottom, name='pool'):
        pooled = tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='VALID', name=name)
        return pooled

    @classmethod
    def upconv_vgg(cls, in_tensor, layer_name, output_shape, stride, trainable=True, init_dict=None):
        with tf.variable_scope(layer_name):
            tf.add_to_collection('shapes_for_memory', in_tensor)

            strides = [1, stride, stride, 1]
            # conv
            kernel = tf.Variable(init_dict[layer_name][0], name='weights', trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv2d_transpose(value=in_tensor, filter=kernel, output_shape=output_shape,
                                                strides=strides, padding='SAME')

            # bias
            biases = tf.Variable(init_dict[layer_name][1], name='biases', trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases)
            return out_tensor

    @classmethod
    def upconv_relu_vgg(cls, in_tensor, layer_name, output_shape, stride, trainable=True, init_dict=None):
        tensor = cls.upconv_vgg(in_tensor, layer_name, output_shape, stride, trainable, init_dict)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def upconv(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            tf.add_to_collection('shapes_for_memory', in_tensor)

            kernel_shape = [kernel_size, kernel_size, in_size[3], in_size[3]]
            strides = [1, stride, stride, 1]

            # conv
            kernel = cls.get_deconv_filter(kernel_shape, trainable)
            tmp_result = tf.nn.conv2d_transpose(value=in_tensor, filter=kernel, output_shape=output_shape,
                                                strides=strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[2]], tf.float32,
                                     tf.constant_initializer(0.0), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases)
            return out_tensor

    @classmethod
    def upconv_relu(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        tensor = cls.upconv(in_tensor, layer_name, output_shape, kernel_size, stride, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def upconv3d(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            kernel_shape = [kernel_size, kernel_size, kernel_size, in_size[4], in_size[4]]
            strides = [1, stride, stride, stride, 1]

            # conv
            kernel = cls.get_deconv_filter3d(kernel_shape, trainable)

            tmp_result = tf.nn.conv3d_transpose(value=in_tensor, filter=kernel, output_shape=output_shape,
                                                strides=strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                     tf.constant_initializer(0.0), trainable=trainable, collections=['wd', 'variables', 'biases'])

            out_tensor = tf.nn.bias_add(tmp_result, biases)
            return out_tensor

    @classmethod
    def upconv3d_relu(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        tensor = cls.upconv3d(in_tensor, layer_name, output_shape, kernel_size, stride, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @staticmethod
    def get_deconv_filter(f_shape, trainable):
        width = f_shape[0]
        height = f_shape[1]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init,
                               shape=weights.shape, trainable=trainable, collections=['wd', 'variables', 'filters'])

    @staticmethod
    def get_deconv_filter3d(f_shape, trainable):
        width = f_shape[0]
        height = f_shape[1]
        depth = f_shape[2]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        trilinear = np.zeros([f_shape[0], f_shape[1], f_shape[2]])
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    value = (1 - abs(x / f - c)) * (1 - abs(y / f - c)) * (1 - abs(z / f - c))
                    trilinear[x, y, z] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[3]):
            weights[:, :, :, i, i] = trilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init,
                               shape=weights.shape, trainable=trainable, collections=['wd', 'variables', 'filters'])

    @classmethod
    def conv_dil(cls, in_tensor, layer_name, kernel_size, dilation, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.atrous_conv2d(in_tensor, kernel, dilation, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv_dil_relu(cls, in_tensor, layer_name, kernel_size, dilation, out_chan, trainable=True):
        tensor = cls.conv_dil(in_tensor, layer_name, kernel_size, dilation, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @staticmethod
    def conv_sep(in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            strides = [1, stride, stride, 1]
            kernel1_shape = [kernel_size, 1, in_size[3], out_chan]
            kernel2_shape = [1, kernel_size, out_chan, out_chan]

            # conv1
            kernel1 = tf.get_variable('weights1', kernel1_shape, tf.float32,
                                      tf.contrib.layers.xavier_initializer(), trainable=trainable)
            tmp_result = tf.nn.conv2d(in_tensor, kernel1, strides, padding='SAME')

            # conv2
            kernel2 = tf.get_variable('weights2', kernel2_shape, tf.float32,
                                      tf.contrib.layers.xavier_initializer(), trainable=trainable)
            tmp_result2 = tf.nn.conv2d(tmp_result, kernel2, strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [out_chan], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable)
            out_tensor = tf.nn.bias_add(tmp_result2, biases, name='out')

            return out_tensor

    @classmethod
    def conv_sep_relu(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        tensor = cls.conv_sep(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @staticmethod
    def fully_connected(in_tensor, layer_name, out_chan, trainable=True, bias_boost=1.0, use_bias=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            assert len(in_size) == 2, 'Input to a fully connected layer must be a vector.'
            weights_shape = [in_size[1], out_chan]

            # weight matrix
            weights = tf.get_variable('weights', weights_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer(), trainable=trainable)
            weights = tf.check_numerics(weights, 'weights: %s' % layer_name)

            out_tensor = tf.matmul(in_tensor, weights)

            # bias
            if use_bias:
                biases = tf.get_variable('biases', [out_chan], tf.float32,
                                         tf.constant_initializer(0.0001), trainable=trainable)
                biases = tf.check_numerics(biases, 'biases: %s' % layer_name)

                out_tensor += bias_boost * biases
            return out_tensor

    @classmethod
    def fully_connected_relu(cls, in_tensor, layer_name, out_chan, trainable=True, use_bias=True):
        tensor = cls.fully_connected(in_tensor, layer_name, out_chan, trainable, use_bias=use_bias)
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu*tensor, name='out')
        return out_tensor


    @staticmethod
    def batch_norm(tensor, is_eval, scope='bn'):
        """
        Batch normalization on convolutional maps.
        Args:
            tensor:           Tensor, 4D BHWD input maps
            is_eval:    boolean tensor, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope(scope):
            assert len(tensor.get_shape().as_list()) == 4, "Tensor needs to be 4D BHWD input maps"
            n_out = tensor.get_shape().as_list()[3]
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                         name='beta', trainable=True, collections=['wd', 'variables'])
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True, collections=['wd', 'variables'])
            batch_mean, batch_var = tf.nn.moments(tensor, [0, 1, 2], name='moments')

            ema = tf.train.ExponentialMovingAverage(0.5)

            def mean_var_with_update():
                ema_assign = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_assign]):
                    return tf.identity(ema.average(batch_mean)), tf.identity(ema.average(batch_var))      # note the identity.

            def mean_var_no_update():
                return ema.average(batch_mean), ema.average(batch_var)

            mean, var = tf.cond(tf.logical_not(is_eval), mean_var_with_update, mean_var_no_update)
            tensor_bn = tf.nn.batch_normalization(tensor, mean, var, beta, gamma, 1e-3)

        return tensor_bn


    @staticmethod
    def batch_norm_3d(tensor, is_eval, scope='bn'):
        """
        Batch normalization on convolutional maps.
        Args:
            tensor:           Tensor, 5D BHWD input maps
            is_eval:    boolean tensor, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope(scope):
            assert len(tensor.get_shape().as_list()) == 5, "Tensor needs to be %D BXYZD input maps"
            n_out = tensor.get_shape().as_list()[4]
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                         name='beta', trainable=True, collections=['wd', 'variables'])
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True, collections=['wd', 'variables'])
            batch_mean, batch_var = tf.nn.moments(tensor, [0, 1, 2, 3], name='moments')

            ema = tf.train.ExponentialMovingAverage(0.5)

            def mean_var_with_update():
                ema_assign = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_assign]):
                    return tf.identity(ema.average(batch_mean)), tf.identity(ema.average(batch_var))      # note the identity.

            def mean_var_no_update():
                return ema.average(batch_mean), ema.average(batch_var)

            mean, var = tf.cond(tf.logical_not(is_eval), mean_var_with_update, mean_var_no_update)
            tensor_bn = tf.nn.batch_normalization(tensor, mean, var, beta, gamma, 1e-3)

        return tensor_bn

    @staticmethod
    def dropout(in_tensor, keep_prob, evaluation):
        """ Dropout: Each neuron is dropped independently. """
        with tf.variable_scope('dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=tensor_shape))
            return out_tensor

    @staticmethod
    def spatial_dropout(in_tensor, keep_prob, evaluation):
        """ Spatial dropout: Not each neuron is dropped independently, but feature map wise. """
        with tf.variable_scope('spatial_dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=[tensor_shape[0], 1, 1, tensor_shape[3]]))
            return out_tensor

    @classmethod
    def residual_unit(cls, x, out_filter, stride, evaluation, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        # print('Residual unit with:\nin=', x)
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = cls.batch_norm(x, evaluation, 'init_bn')
                x = cls.leaky_relu(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = cls.batch_norm(x, evaluation, 'init_bn')
                x = cls.leaky_relu(x)

        with tf.variable_scope('sub1'):
            x = cls.conv(x, 'conv1', 3, stride, out_filter)

        with tf.variable_scope('sub2'):
            x = cls.batch_norm(x, evaluation, 'bn2')
            x = cls.leaky_relu(x)
            x = cls.conv(x, 'conv2', 3, 1, out_filter)

        with tf.variable_scope('sub_add'):
            in_filter = orig_x.get_shape().as_list()[3]
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, [1, stride, stride, 1], [1, stride, stride, 1], 'SAME')
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])

            x += orig_x
        return x

    @classmethod
    def residual_unit_dil(cls, x, out_filter, dilation, evaluation, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = cls.batch_norm(x, evaluation, 'init_bn')
                x = cls.leaky_relu(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = cls.batch_norm(x, evaluation, 'init_bn')
                x = cls.leaky_relu(x)

        with tf.variable_scope('sub1'):
            x = cls.conv_dil(x, 'conv1', 3, dilation, out_filter)

        with tf.variable_scope('sub2'):
            x = cls.batch_norm(x, evaluation, 'bn2')
            x = cls.leaky_relu(x)
            x = cls.conv_dil(x, 'conv2', 3, dilation, out_filter)

        with tf.variable_scope('sub_add'):
            in_filter = orig_x.get_shape().as_list()[3]
            if in_filter != out_filter:
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])

            x += orig_x
        return x


    @classmethod
    def bottleneck_residual(cls, x, in_filter, out_filter, stride, evaluation, activate_before_residual=False):
        """Bottleneck resisual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = cls.batch_norm(x, evaluation, 'init_bn')
                x = cls.leaky_relu(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = cls.batch_norm(x, evaluation, 'init_bn')
                x = cls.leaky_relu(x)

        with tf.variable_scope('sub1'):
            x = cls.conv(x, 'conv1', 1, stride, out_filter/4)

        with tf.variable_scope('sub2'):
            x = cls.batch_norm(x, evaluation, 'bn2')
            x = cls.leaky_relu(x)
            x = cls.conv(x, 'conv2', 3, 1, out_filter/4)

        with tf.variable_scope('sub3'):
            x = cls.batch_norm(x, evaluation, 'bn3')
            x = cls.leaky_relu(x)
            x = cls.conv(x, 'conv3', 3, 1, out_filter)

        with tf.variable_scope('sub_add'):
            in_filter = x.get_shape().as_list()[3]
            if in_filter != out_filter:
                orig_x = cls.conv(orig_x, 'project', 1, out_filter, stride)
            x += orig_x

        return x