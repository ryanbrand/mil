import numpy as np
import random
import tensorflow as tf
import logging
import imageio

from data_generator import DataGenerator
from new_data_generator_reptile import MILDataGenerator
from model import MIL
from evaluation.eval_reach import evaluate_vision_reach
from evaluation.reptile_eval_push import evaluate_push
from tensorflow.python.platform import flags
from tensorflow.python import debug as tf_debug

# added
import os
from reptile_new_data import Reptile
from datetime import datetime

FLAGS = flags.FLAGS
LOGGER = logging.getLogger(__name__)

## Dataset/method options
flags.DEFINE_string('experiment', 'sim_reach', 'sim_vision_reach or sim_push')
flags.DEFINE_string('demo_file', None, 'path to the directory where demo files that containing robot states and actions are stored')
flags.DEFINE_string('demo_gif_dir', None, 'path to the videos of demonstrations')
flags.DEFINE_string('gif_prefix', 'object', 'prefix of the video directory for each task, e.g. object_0 for task 0')
flags.DEFINE_integer('im_width', 100, 'width of the images in the demo videos,  125 for sim_push, and 80 for sim_vision_reach')
flags.DEFINE_integer('im_height', 90, 'height of the images in the demo videos, 125 for sim_push, and 64 for sim_vision_reach')
flags.DEFINE_integer('num_channels', 3, 'number of channels of the images in the demo videos')
flags.DEFINE_integer('T', 50, 'time horizon of the demo videos, 50 for reach, 100 for push')
flags.DEFINE_bool('hsv', False, 'convert the image to HSV format')
flags.DEFINE_bool('use_noisy_demos', False, 'use noisy demonstrations or not (for domain shift)')
flags.DEFINE_string('noisy_demo_gif_dir', None, 'path to the videos of noisy demonstrations')
flags.DEFINE_string('noisy_demo_file', None, 'path to the directory where noisy demo files that containing robot states and actions are stored')
flags.DEFINE_bool('no_action', False, 'do not include actions in the demonstrations for inner update')
flags.DEFINE_bool('no_state', False, 'do not include states in the demonstrations during training')
flags.DEFINE_bool('no_final_eept', False, 'do not include final ee pos in the demonstrations for inner update')
flags.DEFINE_bool('zero_state', False, 'zero-out states (meta-learn state) in the demonstrations for inner update')
flags.DEFINE_bool('two_arms', False, 'use two-arm structure when state is zeroed-out')
flags.DEFINE_integer('training_set_size', -1, 'size of the training set, 1500 for sim_reach, 693 for sim push, and \
                                                -1 for all data except those in validation set')
flags.DEFINE_integer('val_set_size', 150, 'size of the training set, 150 for sim_reach and 76 for sim push')

## Training options
flags.DEFINE_integer('metatrain_iterations', 50000, 'number of metatraining iterations.') # 30k for pushing, 50k for reaching and placing
flags.DEFINE_integer('meta_batch_size', 12, 'number of tasks sampled per meta-update') # 5 for reaching, 15 for pushing, 12 for placing
flags.DEFINE_float('meta_lr', 1.000, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('train_update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.001 for reaching, 0.01 for pushing and placing
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.') # 5 for placing
flags.DEFINE_bool('clip', False, 'use gradient clipping for fast gradient')
flags.DEFINE_float('clip_max', 10.0, 'maximum clipping value for fast gradient')
flags.DEFINE_float('clip_min', -10.0, 'minimum clipping value for fast gradient')
flags.DEFINE_bool('fc_bt', True, 'use bias transformation for the first fc layer')
flags.DEFINE_bool('all_fc_bt', False, 'use bias transformation for all fc layers')
flags.DEFINE_bool('conv_bt', True, 'use bias transformation for the first conv layer, N/A for using pretraining')
flags.DEFINE_integer('bt_dim', 10, 'the dimension of bias transformation for FC layers')
flags.DEFINE_string('pretrain_weight_path', 'N/A', 'path to pretrained weights') # pretrained using MIL to be fine-tuned with gradient step
flags.DEFINE_bool('train_pretrain_conv1', False, 'whether to finetune the pretrained weights')
flags.DEFINE_bool('two_head', False, 'use two-head architecture')
flags.DEFINE_bool('learn_final_eept', False, 'learn an auxiliary loss for predicting final end-effector pose')
flags.DEFINE_bool('learn_final_eept_whole_traj', False, 'learn an auxiliary loss for predicting final end-effector pose \
                                                         by passing the whole trajectory of eepts (used for video-only models)')
flags.DEFINE_integer('final_eept_min', 6, 'first index of the final eept in the action array')
flags.DEFINE_integer('final_eept_max', 8, 'last index of the final eept in the action array')
flags.DEFINE_float('final_eept_loss_eps', 0.1, 'the coefficient of the auxiliary loss')
flags.DEFINE_float('act_loss_eps', 1.0, 'the coefficient of the action loss')
flags.DEFINE_float('loss_multiplier', 100.0, 'the constant multiplied with the loss value, 100 for reach and 50 for push')
flags.DEFINE_bool('use_l1_l2_loss', False, 'use a loss with combination of l1 and l2')
flags.DEFINE_float('l2_eps', 0.01, 'coeffcient of l2 loss')
flags.DEFINE_bool('shuffle_val', False, 'whether to choose the validation set via shuffling or not')


## Model options
flags.DEFINE_integer('random_seed', 0, 'random seed for training')
flags.DEFINE_bool('fp', True, 'use spatial soft-argmax or not')
flags.DEFINE_string('norm', 'layer_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('dropout', False, 'use dropout for fc layers or not')
flags.DEFINE_float('keep_prob', 0.5, 'keep probability for dropout')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 64 for placing, 16 for pushing, 40 for reaching.') # seems like pushing is easiest?
flags.DEFINE_integer('filter_size', 3, 'filter size for conv nets -- 3 for placing, 5 for pushing, 3 for reaching.')
flags.DEFINE_integer('num_conv_layers', 5, 'number of conv layers -- 5 for placing, 4 for pushing, 3 for reaching.')
flags.DEFINE_integer('num_strides', 3, 'number of conv layers with strided filters -- 3 for placing, 4 for pushing, 3 for reaching.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_integer('num_fc_layers', 3, 'number of fully-connected layers')
flags.DEFINE_integer('layer_size', 100, 'hidden dimension of fully-connected layers')
flags.DEFINE_string('init', 'random', 'initializer for conv weights. Choose among random, xavier, and he')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('log_dir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('restore_iter', 0, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_integer('test_update_batch_size', 1, 'number of demos used during test time')
flags.DEFINE_float('gpu_memory_fraction', 1.0, 'fraction of memory used in gpu')
flags.DEFINE_bool('record_gifs', True, 'record gifs during evaluation')

## flags added for reptile
flags.DEFINE_integer('inner_batch_size_reptile', 10, 'inner batch size')
flags.DEFINE_integer('num_shots_reptile', 15, 'number of training shots to use for reptile')
flags.DEFINE_integer('inner_iters_reptile', 8, 'number of inner loop iterations')
flags.DEFINE_integer('meta_batch_size_reptile', 5, 'how many inner loops to run')
flags.DEFINE_float('meta_step_size_reptile', 1e-1, 'meta lr')
flags.DEFINE_integer('num_classes_reptile', 15, 'number of classes to sample') # TODO: consider trying 5
flags.DEFINE_integer('reptile_iterations', 1000, 'number of metatraining iterations.') # 30k for pushing, 50k for reaching and placing
flags.DEFINE_bool('transductive_reptile', False, 'is transductive, see reptile file for more info')
flags.DEFINE_string('reptile_log_dir', '/home/rmb2208/mil/logs/sim_push/20180423_183906_sim_push._num_shots.15_inner_iters.8_meta_batch_size.5_meta_step_size.1.0_num_classes.15_reptile_new_data/', 'model file to restore') # TODO: change after testing

def train(graph, model, saver, sess, log_dir, restore_itr=0, network_config=None):
    """
    Train the model.
    """
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 10
    TOTAL_ITERS = FLAGS.reptile_iterations
    save_dir = log_dir + '/model'
    train_writer = tf.summary.FileWriter(log_dir, graph)
    # actual training.
    if restore_itr == 0:
        training_range = range(TOTAL_ITERS)
    else:
        training_range = range(restore_itr+1, TOTAL_ITERS)

    # added
    with graph.as_default():
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
        variables = [v for v in variables if ('b' in v.name or 'w' in v.name)]
    reptile = Reptile(sess, graph, variables=variables, transductive=FLAGS.transductive_reptile, pre_step_op=None)
    dataset = MILDataGenerator()

    meta_step_size = FLAGS.meta_step_size_reptile
    meta_step_size_final = 0.0 # decay to step size to 0

    # for each training iteration
    for itr in training_range:
        frac_done = itr / float(TOTAL_ITERS)
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        print('cur meta step size:', cur_meta_step_size, 'iter:', itr)

        reptile.train_step(
            dataset.train_data,
            state_ph=model.state_ph,
            obs_ph=model.obs_ph,
            label_ph=model.label_ph,
            minimize_op=model.minimize_op,
            loss_op=model.loss,
            writer=train_writer,
            num_classes=FLAGS.num_classes_reptile, # note you probably want to play with this, consider 5
            num_shots=FLAGS.num_shots_reptile, # train shots
            inner_batch_size=FLAGS.inner_batch_size_reptile,
            inner_iters=FLAGS.inner_iters_reptile, # number of inner loop iterations
            step=itr,
            meta_step_size=cur_meta_step_size, # this needs to be linearly annealed
            meta_batch_size=FLAGS.meta_batch_size_reptile   # how many times to run the inner loop, TODO
        )

        if itr != 0 and (itr % SAVE_INTERVAL == 0 or itr == training_range[-1]):
            print 'Saving model to: %s' % (save_dir + '_%d' % itr)
            with graph.as_default():
                saver.save(sess, save_dir + '_%d' % itr)

def generate_test_demos(data_generator):
    if not FLAGS.use_noisy_demos:
        n_folders = len(data_generator.demos.keys())
        demos = data_generator.demos
    else:
        n_folders = len(data_generator.noisy_demos.keys())
        demos = data_generator.noisy_demos
    policy_demo_idx = [np.random.choice(n_demo, replace=False, size=FLAGS.test_update_batch_size) \
                        for n_demo in [demos[i]['demoX'].shape[0] for i in xrange(n_folders)]]
    selected_demoO, selected_demoX, selected_demoU = [], [], []
    for i in xrange(n_folders):
        selected_cond = np.array(demos[i]['demoConditions'])[np.arange(len(demos[i]['demoConditions'])) == policy_demo_idx[i]]
        Xs, Us, Os = [], [], []
        for idx in selected_cond:
            if FLAGS.use_noisy_demos:
                demo_gif_dir = data_generator.noisy_demo_gif_dir
            else:
                demo_gif_dir = data_generator.demo_gif_dir
            O = np.array(imageio.mimread(demo_gif_dir + data_generator.gif_prefix + '_%d/cond%d.samp0.gif' % (i, idx)))[:, :, :, :3]
            O = np.transpose(O, [0, 3, 2, 1]) # transpose to mujoco setting for images
            O = O.reshape(FLAGS.T, -1) / 255.0 # normalize
            Os.append(O)
        Xs.append(demos[i]['demoX'][np.arange(demos[i]['demoX'].shape[0]) == policy_demo_idx[i]].squeeze())
        Us.append(demos[i]['demoU'][np.arange(demos[i]['demoU'].shape[0]) == policy_demo_idx[i]].squeeze())
        selected_demoO.append(np.array(Os))
        selected_demoX.append(np.array(Xs))
        selected_demoU.append(np.array(Us))
    print "Finished collecting demos for testing"
    selected_demo = dict(selected_demoX=selected_demoX, selected_demoU=selected_demoU, selected_demoO=selected_demoO)
    data_generator.selected_demo = selected_demo

def main():
    print('STARTING MAIN')
    tf.set_random_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)
    # Build up environment to prevent segfault
    if not FLAGS.train:
        if 'reach' in FLAGS.experiment:
            env = gym.make('ReacherMILTest-v1')
            ob = env.reset()
            # import pdb; pdb.set_trace()
    # setup session
    print('MAKING SESS')
    graph = tf.Graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=graph, config=tf_config)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    print('MADE SESS')
    network_config = {
        'num_filters': [FLAGS.num_filters]*FLAGS.num_conv_layers,
        'strides': [[1, 2, 2, 1]]*FLAGS.num_strides + [[1, 1, 1, 1]]*(FLAGS.num_conv_layers-FLAGS.num_strides),
        'filter_size': FLAGS.filter_size,
        'image_width': FLAGS.im_width,
        'image_height': FLAGS.im_height,
        'image_channels': FLAGS.num_channels,
        'n_layers': FLAGS.num_fc_layers,
        'layer_size': FLAGS.layer_size,
        'initialization': FLAGS.init,
    }
    # generate expert demonstrations
    img_idx = range(20, 20+125*125*3)
    # need to compute x_idx and img_idx from data_generator
    model = MIL(7, state_idx=range(20), img_idx=img_idx, network_config=network_config)
    # TODO: figure out how to save summaries and checkpoints
    exp_string = FLAGS.experiment+ '.' + FLAGS.init + '_init.' + str(FLAGS.num_conv_layers) + '_conv' + '.' + str(FLAGS.num_strides) + '_strides' + '.' + str(FLAGS.num_filters) + '_filters' + \
                '.' + str(FLAGS.num_fc_layers) + '_fc' + '.' + str(FLAGS.layer_size) + '_dim' + '.bt_dim_' + str(FLAGS.bt_dim) + '.mbs_'+str(FLAGS.meta_batch_size) + \
                '.ubs_' + str(FLAGS.update_batch_size) + '.numstep_' + str(FLAGS.num_updates) + '.updatelr_' + str(FLAGS.train_update_lr)

    if FLAGS.clip:
        exp_string += '.clip_' + str(int(FLAGS.clip_max))
    if FLAGS.conv_bt:
        exp_string += '.conv_bt'
    if FLAGS.all_fc_bt:
        exp_string += '.all_fc_bt'
    if FLAGS.fp:
        exp_string += '.fp'
    if FLAGS.learn_final_eept:
        exp_string += '.learn_ee_pos'
    if FLAGS.no_action:
        exp_string += '.no_action'
    if FLAGS.zero_state:
        exp_string += '.zero_state'
    if FLAGS.two_head:
        exp_string += '.two_heads'
    if FLAGS.two_arms:
        exp_string += '.two_arms'
    if FLAGS.training_set_size != -1:
        exp_string += '.' + str(FLAGS.training_set_size) + '_trials'

    reptile_exp_string =  FLAGS.experiment + '.' + '_num_shots.' + str(FLAGS.num_shots_reptile) + '_inner_iters.' + str(FLAGS.inner_iters_reptile) + \
                          '_meta_batch_size.'   + str(FLAGS.meta_batch_size_reptile) + '_meta_step_size.' + str(FLAGS.meta_step_size_reptile) + \
                          '_num_classes.'       + str(FLAGS.num_classes_reptile) + '_reptile_iterations.' + str(FLAGS.reptile_iterations)
    date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
    log_dir = FLAGS.log_dir + '/' + str(date_time) + '_' + reptile_exp_string + '_reptile_new_data'

    # put here for now
    if FLAGS.train:
        print('FLAGS.use_noisy_demos:', FLAGS.use_noisy_demos)
        model.init_network(graph, restore_iter=FLAGS.restore_iter)
        #model.init_network(graph, input_tensors=val_input_tensors, restore_iter=FLAGS.restore_iter, prefix='Validation_')
    else:
        model.init_network(graph, prefix='Testing')
    with graph.as_default():
        # Set up saver.
        saver = tf.train.Saver(max_to_keep=3) # TODO: change later
        # Initialize variables.
        init_op = tf.global_variables_initializer()
        sess.run(init_op, feed_dict=None)
    if FLAGS.resume:
        if not os.path.exists(FLAGS.reptile_log_dir):
            model_file = tf.train.latest_checkpoint(log_dir)
        else:
            model_file = tf.train.latest_checkpoint(FLAGS.reptile_log_dir)
        if FLAGS.restore_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model_' + str(FLAGS.restore_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+6:])
            print("Restoring model weights from " + model_file)
            with graph.as_default():
                saver.restore(sess, model_file)
    if FLAGS.train:
        train(graph, model, saver, sess, log_dir, restore_itr=FLAGS.restore_iter, network_config=network_config)
    else:
        data_generator = DataGenerator()
        if 'reach' in FLAGS.experiment:
            generate_test_demos(data_generator)
            evaluate_vision_reach(env, graph, model, data_generator, sess, exp_string, FLAGS.record_gifs, log_dir)
        elif 'push' in FLAGS.experiment:
            evaluate_push(sess, graph, model, data_generator, exp_string, log_dir, FLAGS.demo_file + '/', save_video=FLAGS.record_gifs)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    main()
