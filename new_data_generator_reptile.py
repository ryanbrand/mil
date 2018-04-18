import numpy as np
import os
import pickle
import imageio
from data_generator import *
from natsort import natsorted
from glob import glob

SIM_PUSH_PATH = '/home/rmb2208/mil/data/sim_push'
SIM_PUSH_TEST_PATH = '/home/rmb2208/mil/data/sim_push_test'
SIM_PUSH_SCALE_BIAS_PATH = '/home/rmb2208/mil/data/scale_and_bias_sim_push.pkl'

class MILDataGenerator(object):

    # data_gen is the mil data generator
    def __init__(self, train_dir=SIM_PUSH_PATH, test_dir=SIM_PUSH_TEST_PATH, scale_and_bias_path=SIM_PUSH_SCALE_BIAS_PATH):
        # load scale and bias stats
        if scale_and_bias_path != None:
            with open(scale_and_bias_path, 'rb') as f:
                scale_bias_dict = pickle.load(f)
                self.scale = scale_bias_dict['scale']
                self.bias  = scale_bias_dict['bias']
        else:
            self.scale = None
            self.bias  = None

        self.train_data = self._load_data(train_dir)
        self.test_data  = self._load_data(test_dir)

    def _load_data(self, data_dir):
        # get all of the .pkl file paths
        pickle_paths = glob(os.path.join(data_dir, '*.pkl'))
        pickle_paths = natsorted(pickle_paths)

        # get all of the object_ dirs
        demo_gif_paths = glob(os.path.join(data_dir, 'object_*'))
        demo_gif_paths = natsorted(demo_gif_paths)

        for pickle, demo_gif in zip(pickle_paths, demo_gif_paths):
            yield  MILTask(demo_gif, pickle, self.scale, self.bias)


class MILTask(object):

    def __init__(self, gif_demo_path, pickle_path, scale=None, bias=None):
        gif_paths = natsorted([os.path.join(gif_demo_path, f) for f in glob(os.path.join(gif_demo_path, '*.gif'))])
        self.num_trials = len(gif_paths)
        self.gifs = self._read_gifs(gif_paths)
        #print('gifs shape:', self.gifs.shape)
        self.states, self.actions = self._read_pickle(pickle_path, scale, bias)
        self.state_dim = self.states.shape[-1]
        self.action_dim = self.actions.shape[-1]
        self.gif_dim    = self.gifs.shape[-3:]
        self.T          = self.states.shape[1]

    # dimension of input to construct model: 20 + 46875 = dim_input
    # forward called on [update_batch_size, dim_input]
    # when they do multiple gradient steps they re-use the same data
    def sample(self, num_samples):
        '''
        samples num_trials shots from the task

        returns: (gifs, states, actions) tuple where gifs: num_samples X 100 X H X W X C
                 states: num_samples X 100 X 20, actions: num_samples X 100 X 7
        '''
        # TODO: figure out if each time-step is treated as a seperate example?
        H, W, C = self.gif_dim
        sample_idxs = np.random.choice(self.num_trials, size=num_samples)
        gifs = self.gifs[sample_idxs].reshape(num_samples*self.T, -1)
        states = self.states[sample_idxs].reshape(num_samples*self.T, self.state_dim)
        actions = self.actions[sample_idxs].reshape(num_samples*self.T, self.action_dim)
        return (gifs, states, actions)

    def _read_gifs(self, gif_paths):
        gifs = np.stack([imageio.mimread(gif_path) for gif_path in gif_paths])
        # move the channel indicies to the front of the images to match format
        gifs = gifs[:, :, :, :, :3].transpose(0, 1, 4, 3, 2).astype('float32') / 255.0
        return gifs

    def _read_pickle(self, pickle_path, scale, bias):
        # returns a tuple of states and actions where
        # actions: (24, 100, 7), states: (24, 100, 20)
        demo_info = pickle.load(open(pickle_path, 'rb'))
        states    = demo_info['demoX']
        actions   = demo_info['demoU']
        N, T, S   = states.shape
        if isinstance(scale, np.ndarray) and isinstance(bias, np.ndarray):
            states = states.reshape(-1, S)
            states = states.dot(scale) + bias
            states = states.reshape(-1, T, S)
        return states, actions

