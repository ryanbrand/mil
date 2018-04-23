import glob
import os
import imageio
import joblib
import pickle
import numpy as np
import random
import tensorflow as tf
from PIL import Image
from sampler_utils import rollout
from utils import mkdir_p, load_scale_and_bias
import sys

XML_PATH = '/home/rmb2208/gym/gym/envs/mujoco/assets/sim_push_xmls/'
SCALE_FILE_PATH = 'data/scale_and_bias_sim_push.pkl'
CROP = False
sys.path.append('/home/rmb2208/gym/gym')
from gym.envs.mujoco.pusher import PusherEnv

# added for reptile
from reptile_new_data import *

class TFAgent(object):
    def __init__(self, model, scale_bias_file, sess, graph):
        self.sess = sess
        self.model = model

        with graph.as_default():
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            variables = [v for v in variables if ('b' in v.name or 'w' in v.name)]
        # TODO: try transductive and non-transductive options
        self.reptile = Reptile(sess, graph, variables=variables, transductive=False, pre_step_op=None)

        if scale_bias_file:
            self.scale, self.bias = load_scale_and_bias(scale_bias_file)
        else:
            self.scale = None

    def reset(self):
        pass

    def set_demo(self, demo_gif, demoX, demoU):
        #import pdb; pdb.set_trace()

        # concatenate demos in time
        demo_gif = np.array(demo_gif)
        N, T, H, W, C = demo_gif.shape
        self.update_batch_size = N
        self.T = T
        demo_gif = np.reshape(demo_gif, [N*T, H, W, C])
        demo_gif = np.array(demo_gif)[:,:,:,:3].transpose(0,3,2,1).astype('float32') / 255.0
        self.demoVideo = demo_gif.reshape(1, N*T, -1)
        self.demoX = demoX
        self.demoU = demoU

    # TODO: this is probably wrong, you need to do some sort of reptile update here
    def get_action(self, obs):
        print('WARNING: case not considered')
        obs = obs.reshape((1,1,23))
        # old feed dict
        # {self.model.statea: self.demoX.dot(self.scale) + self.bias,
        #  self.model.actiona: self.demoU,
        #  self.model.stateb: obs.dot(self.scale) + self.bias}
        # new feed dict
        #evaluate(dataset, input_ph, label_ph, minimize_op, predictions, num_classes, num_shots, inner_batch_size, inner_iters)
        # get v1 working just pass in the data
        feed_dict = {
            self.model.state_ph : obs.dot(self.scale) + self.bias,
            self.model.label_ph : None  #TODO: not sure if this should be here
        }
        action = self.sess.run(self.model.test_act_op, feed_dict=feed_dict)
        return action, dict()

    # TODO: this is probably wrong, you need to do some sort of reptile update here
    def get_vision_action(self, image, obs, t=-1):
        if CROP:
            image = np.array(Image.fromarray(image).crop((40,25,120,90)))

        image = np.expand_dims(image, 0).transpose(0,3,2,1).astype('float32') / 255.0
        image = image.reshape((1, 1, -1))

        obs = obs.reshape((1,1,20))

        # added, we expact two dims because we don't take input for all task, just a single task at a time
        image = np.squeeze(image, 0)
        obs   = np.squeeze(obs, 0)
        if self.scale is not None:
            # old feed dict
            # {self.model.statea: self.demoX.dot(self.scale) + self.bias,
            #  self.model.obsa: self.demoVideo,
            #  self.model.actiona: self.demoU,
            #  self.model.stateb: obs.dot(self.scale) + self.bias,
            #  self.model.obsb: image}
            # new feed dict

            # TODO: check that training data only comes from one class
            # train
            statea =  np.squeeze(self.demoX.dot(self.scale) + self.bias, axis=0)
            obsa =    np.squeeze(self.demoVideo, axis=0)
            actiona = np.squeeze(self.demoU, axis=0)
            train_data = (statea, obsa, actiona)
            # test
            stateb = obs.dot(self.scale) + self.bias
            obsb = image
            test_data = (stateb, obsb)
            dataset = (train_data, test_data)

            # do an update on the model parameters
            #TODO: somewhere create reptile class instance
            action = self.reptile.evaluate(
                dataset,
                self.model.state_ph,
                self.model.obs_ph,
                self.model.label_ph,
                self.model.minimize_op,
                self.model.test_act_op,
                num_classes=None,      # not used
                num_shots=None,        # not used
                inner_batch_size=None, # not used
                inner_iters=1 # TODO: see if you want to change this
            )
        else:
            print('WARNING: scale is None, case not considered')
            # old feed dict
            # {self.imagea: self.demoVideo,
            #  self.actiona: self.demoU,
            #  self.imageb: image}
            # new feed dict
            feed_dict = {
                self.obs_ph : image
            }
            action = self.sess.run(self.test_act_op, feed_dict=feed_dict)
        return action, dict()

def load_env(demo_info):
    xml_filepath = demo_info['xml']
    suffix = xml_filepath[xml_filepath.index('pusher'):]
    prefix = XML_PATH + 'test2_ensure_woodtable_distractor_'
    xml_filepath = str(prefix + suffix)

    env = PusherEnv(**{'xml_file':xml_filepath, 'distractors': True})
    return env

def load_demo(task_id, demo_dir, demo_inds):
    demo_info = pickle.load(open(demo_dir+task_id+'.pkl', 'rb'))
    demoX = demo_info['demoX'][demo_inds,:,:]
    demoU = demo_info['demoU'][demo_inds,:,:]
    d1, d2, _ = demoX.shape
    demoX = np.reshape(demoX, [1, d1*d2, -1])
    demoU = np.reshape(demoU, [1, d1*d2, -1])

    # read in demo video
    if CROP:
        demo_gifs = [imageio.mimread(demo_dir+'crop_object_'+task_id+'/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]
    else:
        demo_gifs = [imageio.mimread(demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]

    return demoX, demoU, demo_gifs, demo_info


def eval_success(path):
      obs = path['observations']
      target = obs[:, -3:-1]
      obj = obs[:, -6:-4]
      dists = np.sum((target-obj)**2, 1)  # distances at each timestep
      return np.sum(dists < 0.017) >= 10

def evaluate_push(sess, graph, model, data_generator, exp_string, log_dir, demo_dir, id=-1, save_video=True, num_input_demos=1):
    scale_file = SCALE_FILE_PATH
    files = glob.glob(os.path.join(demo_dir, '*.pkl'))
    all_ids = [int(f.split('/')[-1][:-4]) for f in files]
    all_ids.sort()
    num_success = 0
    num_trials = 0
    trials_per_task = 5

    if id == -1:
        task_ids = all_ids
    else:
        task_ids = [id]
    # random.seed(5)
    # np.random.seed(5)

    for task_id in task_ids:
        demo_inds = [1] # for consistency of comparison
        if num_input_demos > 1:
            demo_inds += range(12, 12+int(num_input_demos / 2))
            demo_inds += range(2, 2+int((num_input_demos-1) / 2))
        assert len(demo_inds) == num_input_demos

        demoX, demoU, demo_gifs, demo_info = load_demo(str(task_id), demo_dir, demo_inds)

        # load xml file
        env = load_env(demo_info)
        policy = TFAgent(model, scale_file, sess, graph)
        policy.set_demo(demo_gifs, demoX, demoU)

        returns = []
        gif_dir = log_dir + '/evaluated_gifs_task_' + str(task_id) + '/'
        mkdir_p(gif_dir)

        while True:
            video_suffix = gif_dir + 'demo_' + str(num_input_demos) + '_' + str(len(returns)) + '.gif'
            path = rollout(env, policy, max_path_length=100, env_reset=True,
                           animated=True, speedup=1, always_return_paths=True, save_video=save_video, video_filename=video_suffix, vision=True)
            num_trials += 1
            if eval_success(path):
                num_success += 1
            print('Return: '+str(path['rewards'].sum()))
            returns.append(path['rewards'].sum())
            print('Average Return so far: ' + str(np.mean(returns)))
            print('Success Rate so far: ' + str(float(num_success)/num_trials))
            sys.stdout.flush()
            if len(returns) > trials_per_task:
                break
        tf.reset_default_graph()
    success_rate_msg = "Final success rate is %.5f" % (float(num_success)/num_trials)
    with open('logs/log_sim_push.txt', 'a') as f:
        f.write(exp_string + ':\n')
        f.write(success_rate_msg + '\n')
