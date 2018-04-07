""" This file defines Meta Imitation Learning (MIL). """
from __future__ import division

import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import flags
from tf_utils import *
from utils import Timer
from natsort import natsorted

FLAGS = flags.FLAGS

class MIL(object):
    """ Initialize MIL. Need to call init_network to contruct the architecture after init. """
    def __init__(self, dU, state_idx=None, img_idx=None, network_config=None):
        # added
        self.step_size = FLAGS.train_update_lr
        self.loss_multiplier = FLAGS.loss_multiplier
        self.final_eept_loss_eps = FLAGS.final_eept_loss_eps
        self.act_loss_eps = FLAGS.act_loss_eps
        self.use_whole_traj = FLAGS.learn_final_eept_whole_traj
        # MIL hyperparams
        self.num_updates = FLAGS.num_updates
        self.update_batch_size = FLAGS.update_batch_size
        self.meta_batch_size = FLAGS.meta_batch_size
        self.meta_lr = FLAGS.meta_lr
        self.activation_fn = tf.nn.relu # by default, we use relu
        self.T = FLAGS.T
        self.network_params = network_config
        self.norm_type = FLAGS.norm
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.state_idx, self.img_idx = state_idx, img_idx
        # Dimension of input and output of the model
        self._dO = len(img_idx) + len(state_idx)
        self._dU = dU

    # [checked]
    def init_network(self, graph, input_tensors=None, restore_iter=0, prefix='Training_'):
        # sets self.state_ph, self.obs_ph, self.label_ph 
        self.add_placeholders(input_tensors)
        # sets self.weights
        self.add_weights(prefix=prefix, dim_input=self._dO, dim_output=self._dU, network_config=self.network_params)
        # sets self.minimize_op, self.loss
        self.add_loss(input_tensors=input_tensors, prefix=prefix, dim_input=self._dO, dim_output=self._dU, network_config=self.network_params)
        if 'Training' in prefix:
            self.train_summ_op = tf.summary.scalar(prefix+'loss', self.loss)
        elif 'Validation' in prefix:
            self.val_summ_op = tf.summary.scalar(prefix+'loss', self.loss)

    # [checked]
    def add_placeholders(self, input_tensors=None):
        # TODO: change input to be tf.concat(axis=2, values=[statea, obsa]) elsewhere
        # Note: this is both create_placeholders and add_placeholders
        if input_tensors is None:
            self.state_ph = tf.placeholder(tf.float32, name='statea')
            self.obs_ph   = tf.placeholder(tf.float32, name='obsa')
            self.label_ph = tf.placeholder(tf.float32, name='actiona')

    def construct_image_input(self, nn_input, state_idx, img_idx, network_config=None):
        """Preprocess images;
        takes in state_idx (list of indices for state data in observation), img_idx (list of indices for image data in observation), 
        and nn_input; pretty much just preprocesses input; returns preprocessed image input, flattened image input, and the state input
        """
        state_input = nn_input[:, 0:state_idx[-1]+1]
        flat_image_input = nn_input[:, state_idx[-1]+1:img_idx[-1]+1]

        # image goes through 3 convnet layers
        num_filters = network_config['num_filters']
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        image_input = tf.reshape(flat_image_input, [-1, num_channels, im_width, im_height])
        image_input = tf.transpose(image_input, perm=[0,3,2,1])
        if FLAGS.pretrain_weight_path != 'N/A':
            image_input = image_input * 255.0 - tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], np.float32))
            # 'RGB'->'BGR'
            image_input = image_input[:, :, :, ::-1]
        return image_input, flat_image_input, state_input

    def construct_weights(self, dim_input=27, dim_output=7, network_config=None):
        """ 
        Construct weights for the network;
        takes in input dim and final output dim and just builds conv and fc weights, including augmented bias thing 
        (and 2-headed architecture if FLAGS.two_head is true); returns weights
        """
        weights = {}
        num_filters = network_config['num_filters']
        strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
        filter_sizes = network_config.get('filter_size', [3]*len(strides)) # used to be 2
        if type(filter_sizes) is not list:
            filter_sizes = len(strides)*[filter_sizes]
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        is_dilated = network_config.get('is_dilated', False)
        use_fp = FLAGS.fp
        pretrain = FLAGS.pretrain_weight_path != 'N/A'
        train_pretrain_conv1 = FLAGS.train_pretrain_conv1
        initialization = network_config.get('initialization', 'random')
        if pretrain:
            num_filters[0] = 64
        pretrain_weight_path = FLAGS.pretrain_weight_path
        n_conv_layers = len(num_filters)
        downsample_factor = 1
        for stride in strides:
            downsample_factor *= stride[1]
        if use_fp:
            self.conv_out_size = int(num_filters[-1]*2)
        else:
            self.conv_out_size = int(np.ceil(im_width/(downsample_factor)))*int(np.ceil(im_height/(downsample_factor)))*num_filters[-1]

        # conv weights
        fan_in = num_channels
        if FLAGS.conv_bt:
            fan_in += num_channels
        if FLAGS.conv_bt:
            weights['img_context'] = safe_get('img_context', initializer=tf.zeros([im_height, im_width, num_channels], dtype=tf.float32))
            weights['img_context'] = tf.clip_by_value(weights['img_context'], 0., 1.)
        for i in xrange(n_conv_layers):
            if not pretrain or i != 0:
                if self.norm_type == 'selu':
                    weights['wc%d' % (i+1)] = init_conv_weights_snn([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                elif initialization == 'xavier':
                    weights['wc%d' % (i+1)] = init_conv_weights_xavier([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                elif initialization == 'random':
                    weights['wc%d' % (i+1)] = init_weights([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                else:
                    raise NotImplementedError
                weights['bc%d' % (i+1)] = init_bias([num_filters[i]], name='bc%d' % (i+1))
                fan_in = num_filters[i]
            else:
                import h5py

                assert num_filters[i] == 64
                vgg_filter_size = 3
                weights['wc%d' % (i+1)] = safe_get('wc%d' % (i+1), [vgg_filter_size, vgg_filter_size, fan_in, num_filters[i]], dtype=tf.float32, trainable=train_pretrain_conv1)
                weights['bc%d' % (i+1)] = safe_get('bc%d' % (i+1), [num_filters[i]], dtype=tf.float32, trainable=train_pretrain_conv1)
                pretrain_weight = h5py.File(pretrain_weight_path, 'r')
                conv_weight = pretrain_weight['block1_conv%d' % (i+1)]['block1_conv%d_W_1:0' % (i+1)][...]
                conv_bias = pretrain_weight['block1_conv%d' % (i+1)]['block1_conv%d_b_1:0' % (i+1)][...]
                weights['wc%d' % (i+1)].assign(conv_weight)
                weights['bc%d' % (i+1)].assign(conv_bias)
                fan_in = conv_weight.shape[-1]

        # fc weights
        in_shape = self.conv_out_size
        if not FLAGS.no_state:
            in_shape += len(self.state_idx)
        if FLAGS.learn_final_eept:
            final_eept_range = range(FLAGS.final_eept_min, FLAGS.final_eept_max)
            final_eept_in_shape = self.conv_out_size
            if FLAGS.fc_bt:
                weights['context_final_eept'] = safe_get('context_final_eept', initializer=tf.zeros([FLAGS.bt_dim], dtype=tf.float32))
                final_eept_in_shape += FLAGS.bt_dim
            weights['w_ee'] = init_weights([final_eept_in_shape, len(final_eept_range)], name='w_ee')
            weights['b_ee'] = init_bias([len(final_eept_range)], name='b_ee')
            if FLAGS.two_head and FLAGS.no_final_eept:
                weights['w_ee_two_heads'] = init_weights([final_eept_in_shape, len(final_eept_range)], name='w_ee_two_heads')
                weights['b_ee_two_heads'] = init_bias([len(final_eept_range)], name='b_ee_two_heads')
            in_shape += (len(final_eept_range))
        if FLAGS.fc_bt:
            in_shape += FLAGS.bt_dim
        if FLAGS.fc_bt:
            weights['context'] = safe_get('context', initializer=tf.zeros([FLAGS.bt_dim], dtype=tf.float32))
        fc_weights = self.construct_fc_weights(in_shape, dim_output, network_config=network_config)
        self.conv_out_size_final = in_shape
        weights.update(fc_weights)
        return weights

    def construct_fc_weights(self, dim_input=27, dim_output=7, network_config=None):
        """
        same as above just for only fc weights
        """
        n_layers = network_config.get('n_layers', 4)
        dim_hidden = network_config.get('layer_size', [100]*(n_layers-1))
        if type(dim_hidden) is not list:
            dim_hidden = (n_layers - 1)*[dim_hidden]
        dim_hidden.append(dim_output)
        weights = {}
        in_shape = dim_input
        for i in xrange(n_layers):
            if FLAGS.two_arms and i == 0:
                if self.norm_type == 'selu':
                    weights['w_%d_img' % i] = init_fc_weights_snn([in_shape-len(self.state_idx), dim_hidden[i]], name='w_%d_img' % i)
                    weights['w_%d_state' % i] = init_fc_weights_snn([len(self.state_idx), dim_hidden[i]], name='w_%d_state' % i)
                else:
                    weights['w_%d_img' % i] = init_weights([in_shape-len(self.state_idx), dim_hidden[i]], name='w_%d_img' % i)
                    weights['w_%d_state' % i] = init_weights([len(self.state_idx), dim_hidden[i]], name='w_%d_state' % i)
                    weights['b_%d_state_two_arms' % i] = init_bias([dim_hidden[i]], name='b_%d_state_two_arms' % i)
                weights['b_%d_img' % i] = init_bias([dim_hidden[i]], name='b_%d_img' % i)
                weights['b_%d_state' % i] = init_bias([dim_hidden[i]], name='b_%d_state' % i)
                in_shape = dim_hidden[i]
                continue
            if i > 0 and FLAGS.all_fc_bt:
                in_shape += FLAGS.bt_dim
                weights['context_%d' % i] = init_bias([FLAGS.bt_dim], name='context_%d' % i)
            if self.norm_type == 'selu':
                weights['w_%d' % i] = init_fc_weights_snn([in_shape, dim_hidden[i]], name='w_%d' % i)
            else:
                weights['w_%d' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d' % i)
            weights['b_%d' % i] = init_bias([dim_hidden[i]], name='b_%d' % i)
            if (i == n_layers - 1 or (i == 0 and FLAGS.zero_state and not FLAGS.two_arms)) and FLAGS.two_head:
                weights['w_%d_two_heads' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d_two_heads' % i)
                weights['b_%d_two_heads' % i] = init_bias([dim_hidden[i]], name='b_%d_two_heads' % i)
            in_shape = dim_hidden[i]
        return weights

    def forward(self, image_input, state_input, weights, meta_testing=False, is_training=True, testing=False, network_config=None):
        """ 
        Perform the forward pass;
        given image input, state input, and weight dict, perform standard forward pass in net, except only through conv layers really, 
        then call fc_forward and output final result through that 
        """
        if FLAGS.fc_bt:
            im_height = network_config['image_height']
            im_width = network_config['image_width']
            num_channels = network_config['image_channels']
            flatten_image = tf.reshape(image_input, [-1, im_height*im_width*num_channels])
            context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(FLAGS.bt_dim)))
            context += weights['context']
            if FLAGS.learn_final_eept:
                context_final_eept = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(FLAGS.bt_dim)))
                context_final_eept += weights['context_final_eept']
        norm_type = self.norm_type
        decay = network_config.get('decay', 0.9)
        strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
        downsample_factor = strides[0][1]
        n_strides = len(strides)
        n_conv_layers = len(strides)
        use_dropout = FLAGS.dropout
        prob = FLAGS.keep_prob
        is_dilated = network_config.get('is_dilated', False)
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        conv_layer = image_input
        if FLAGS.conv_bt:
            img_context = tf.zeros_like(conv_layer)
            img_context += weights['img_context']
            conv_layer = tf.concat(axis=3, values=[conv_layer, img_context])
        for i in xrange(n_conv_layers):
            if not use_dropout:
                conv_layer = norm(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
                                norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=self.activation_fn)
            else:
                conv_layer = dropout(norm(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
                                norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=self.activation_fn), keep_prob=prob, is_training=is_training, name='dropout_%d' % (i+1))
        if FLAGS.fp:
            _, num_rows, num_cols, num_fp = conv_layer.get_shape()
            if is_dilated:
                num_rows = int(np.ceil(im_width/(downsample_factor**n_strides)))
                num_cols = int(np.ceil(im_height/(downsample_factor**n_strides)))
            num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
            x_map = np.empty([num_rows, num_cols], np.float32)
            y_map = np.empty([num_rows, num_cols], np.float32)

            for i in range(num_rows):
                for j in range(num_cols):
                    x_map[i, j] = (i - num_rows / 2.0) / num_rows
                    y_map[i, j] = (j - num_cols / 2.0) / num_cols

            x_map = tf.convert_to_tensor(x_map)
            y_map = tf.convert_to_tensor(y_map)

            x_map = tf.reshape(x_map, [num_rows * num_cols])
            y_map = tf.reshape(y_map, [num_rows * num_cols])

            # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
            features = tf.reshape(tf.transpose(conv_layer, [0,3,1,2]),
                                  [-1, num_rows*num_cols])
            softmax = tf.nn.softmax(features)

            fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
            fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

            conv_out_flat = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
        else:
            conv_out_flat = tf.reshape(conv_layer, [-1, self.conv_out_size])
        fc_input = tf.add(conv_out_flat, 0)
        if FLAGS.learn_final_eept:
            final_eept_range = range(FLAGS.final_eept_min, FLAGS.final_eept_max)
            if testing:
                T = 1
            else:
                T = self.T
            conv_out_flat = tf.reshape(conv_out_flat, [-1, T, self.conv_out_size])
            conv_size = self.conv_out_size
            if FLAGS.fc_bt:
                context_dim = FLAGS.bt_dim
                conv_out_flat = tf.concat(axis=2, values=[conv_out_flat, tf.reshape(context_final_eept, [-1, T, context_dim])])
                conv_size += context_dim
            # only predict the final eept using the initial image
            final_ee_inp = tf.reshape(conv_out_flat, [-1, conv_size])
            # use video for preupdate only if no_final_eept
            if (not FLAGS.learn_final_eept_whole_traj) or meta_testing:
                final_ee_inp = conv_out_flat[:, 0, :]
            if FLAGS.two_head and not meta_testing and FLAGS.no_final_eept:
                final_eept_pred = tf.matmul(final_ee_inp, weights['w_ee_two_heads']) + weights['b_ee_two_heads']
            else:
                final_eept_pred = tf.matmul(final_ee_inp, weights['w_ee']) + weights['b_ee']
            if (not FLAGS.learn_final_eept_whole_traj) or meta_testing:
                final_eept_pred = tf.reshape(tf.tile(tf.reshape(final_eept_pred, [-1]), [T]), [-1, len(final_eept_range)])
                final_eept_concat = tf.identity(final_eept_pred)
            else:
                # Assume tbs == 1
                # Only provide the FC layers with final_eept_pred at first time step
                final_eept_concat = final_eept_pred[0]
                final_eept_concat = tf.reshape(tf.tile(tf.reshape(final_eept_concat, [-1]), [T]), [-1, len(final_eept_range)])
            fc_input = tf.concat(axis=1, values=[fc_input, final_eept_concat])
        else:
            final_eept_pred = None
        if FLAGS.fc_bt:
            fc_input = tf.concat(axis=1, values=[fc_input, context])
        return self.fc_forward(fc_input, weights, state_input=state_input, meta_testing=meta_testing, is_training=is_training, testing=testing, network_config=network_config), final_eept_pred

    def fc_forward(self, fc_input, weights, state_input=None, meta_testing=False, is_training=True, testing=False, network_config=None):
        '''
        fc_forward: completes forward pass for conv net; performs forward pass given special fc input (flexible for multiple architectures, including 2-headed, etc)
        '''
        n_layers = network_config.get('n_layers', 4)
        use_dropout = FLAGS.dropout
        prob = FLAGS.keep_prob
        fc_output = tf.add(fc_input, 0)
        use_selu = self.norm_type == 'selu'
        norm_type = self.norm_type
        if state_input is not None and not FLAGS.two_arms:
            fc_output = tf.concat(axis=1, values=[fc_output, state_input])
        for i in xrange(n_layers):
            if i > 0 and FLAGS.all_fc_bt:
                context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(fc_output)), range(FLAGS.bt_dim)))
                context += weights['context_%d' % i]
                fc_output = tf.concat(axis=1, values=[fc_output, context])
            if (i == n_layers - 1 or (i == 0 and FLAGS.zero_state and not FLAGS.two_arms)) and FLAGS.two_head and not meta_testing:
                fc_output = tf.matmul(fc_output, weights['w_%d_two_heads' % i]) + weights['b_%d_two_heads' % i]
            elif i == 0 and FLAGS.two_arms:
                assert state_input is not None
                if FLAGS.two_arms:
                    state_part = weights['b_%d_state_two_arms' % i]
                else:
                    state_part = tf.matmul(state_input, weights['w_%d_state' % i]) + weights['b_%d_state' % i]
                if not meta_testing:
                    fc_output = tf.matmul(fc_output, weights['w_%d_img' % i]) + weights['b_%d_img' % i] + state_part
                else:
                    fc_output = tf.matmul(fc_output, weights['w_%d_img' % i]) + weights['b_%d_img' % i] + \
                                tf.matmul(state_input, weights['w_%d_state' % i]) + weights['b_%d_state' % i]
            else:
                fc_output = tf.matmul(fc_output, weights['w_%d' % i]) + weights['b_%d' % i]
            if i != n_layers - 1:
                if use_selu:
                    fc_output = selu(fc_output)
                else:
                    fc_output = self.activation_fn(fc_output)
                # only use dropout for post-update
                if use_dropout:
                    fc_output = dropout(fc_output, keep_prob=prob, is_training=is_training, name='dropout_fc_%d' % i, selu=use_selu)
        return fc_output

    # sets self.weights using construct_weights [checked]
    def add_weights(self, dim_input=27, dim_output=7, network_config=None):
        with tf.variable_scope('model', reuse=None) as training_scope:
            # Construct layers weight & bias
            if 'weights' not in dir(self):
                if FLAGS.learn_final_eept:
                    final_eept_range = range(FLAGS.final_eept_min, FLAGS.final_eept_max)
                    self.weights = weights = self.construct_weights(dim_input, dim_output-len(final_eept_range), network_config=network_config)
                else:
                    self.weights = weights = self.construct_weights(dim_input, dim_output, network_config=network_config)
                self.sorted_weight_keys = natsorted(self.weights.keys())
            else:
                training_scope.reuse_variables()
                weights = self.weights
    
    # sets minimize op using input_tensors [checked]
    def add_loss(self, input_tensors=None, prefix='Training_', dim_input=27, dim_output=7, network_config=None):
        # observations are being fed in for testing
        if input_tensors is None:
            obsa = self.obs_ph
        # using queue
        else:
            obsa = input_tensors['inputa'] 

        # concat states and observations in as input to model; this provides more info that just obs
        inputa = tf.concat(axis=2, values=[self.state_ph, obsa])
        inputa = tf.reshape(inputa, [-1, dim_input])
        actiona = tf.reshape(self.label_ph, [-1, dim_output])
        testing = 'Testing' in prefix

        # for learning end effector pose
        final_eepta = None
        if FLAGS.learn_final_eept:
            final_eept_range = range(FLAGS.final_eept_min, FLAGS.final_eept_max)
            final_eepta = actiona[:, final_eept_range[0]:final_eept_range[-1]+1]
            actiona = actiona[:, :final_eept_range[0]]
            if FLAGS.no_final_eept:
                final_eepta = tf.zeros_like(final_eepta)

        if FLAGS.no_action:
            actiona = tf.zeros_like(actiona)

        # Convert to image dims
        inputa, _, state_inputa = self.construct_image_input(inputa, self.state_idx, self.img_idx, network_config=network_config)
        if FLAGS.zero_state:
            state_inputa = tf.zeros_like(state_inputa)
        if FLAGS.no_state:
            state_inputa = None

        # compute loss
        weights = self.weights
        if 'Training' in prefix:
            local_outputa, final_eept_preda = self.forward(inputa, state_inputa, weights, network_config=network_config)
        else:
            local_outputa, final_eept_preda = self.forward(inputa, state_inputa, weights, is_training=False, network_config=network_config)
        if FLAGS.learn_final_eept:
            final_eept_lossa = euclidean_loss_layer(final_eept_preda, final_eepta, multiplier=self.loss_multiplier, use_l1=FLAGS.use_l1_l2_loss)
        else:
            final_eept_lossa = tf.constant(0.0)
        local_lossa = self.act_loss_eps * euclidean_loss_layer(local_outputa, actiona, multiplier=self.loss_multiplier, use_l1=FLAGS.use_l1_l2_loss)
        if FLAGS.learn_final_eept:
            local_lossa += self.final_eept_loss_eps * final_eept_lossa
        self.loss = local_lossa
        self.minimize_op = tf.train.AdamOptimizer(self.step_size).minimize(local_lossa)
