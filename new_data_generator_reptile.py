import numpy as np
import os
import pickle
import imageio
from data_generator import *
from natsort import natsorted
from glob import glob


class MILDataGenerator(object):
    
    # data_gen is the mil data generator
    def __init__(self, train_dir, test_dir, scale_and_bias_path=None):
        print('in init')
        # load scale and bias stats 
        if scale_and_bias_path != None:
            with open(scale_and_bias_path, 'rb') as f:
                scale_bias_dict = pickle.load(f)
                self.scale = scale_bias_dict['scale']
                self.bias  = scale_bias_dict['bias']
        else:
            self.scale = None
            self.bias  = None

        print('scale:', self.scale.shape, 'bias:', self.bias.shape)
        print('about to load data')
        self.train_data = self._load_data(train_dir)
        print('loaded data')
        #self.test_data  = self._load_data(test_dir)

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
        self.gifs = np.stack([self._read_gif(gif) for gif in gif_paths])
        self.states, self.actions = self._read_pickle(pickle_path, scale, bias)
        print('gif_demo_path:', gif_demo_path, 'pickle_path:', pickle_path)
        print('states:', self.states.shape, 'actions:', self.actions.shape, 'gifs:', self.gifs.shape)

    def sample(self, num_samples):
        '''
        samples num_trials shots from the task

        returns: (gifs, states, actions) tuple where gifs: num_samples X 100 X H X W X C
                 states: num_samples X 100 X 20, actions: num_samples X 100 X 7
        '''
        # TODO: figure out if each time-step is treated as a seperate example?
        # get the indexes
        sample_idxs = np.random.choice(self.num_trials, size=num_samples)
        gifs = self.gifs[sample_idxs]
        states = self.states[sample_idxs]
        actions = self.actions[sample_idxs]
        return (gifs, states, actions)

    def _read_gif(self, gif_path):
        return imageio.mimread(gif_path)

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

