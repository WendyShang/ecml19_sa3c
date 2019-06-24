from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gym.spaces.box import Box
from collections import namedtuple
import os

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class Env(object):
    def __init__(self, args, env_ind=0):            
        try:
            import scipy.misc
            self.imsave = scipy.misc.imsave
        except ImportError as e: self.logger.warning("WARNING: scipy.misc not found")
        self.logger     = args.logger
        self.ind        = env_ind  
        self.img_dir    = args.save_model_dir + "/imgs/"
        self.frame_ind  = 0
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        self.seed       = args.seed + self.ind 
        self.env_type   = 'atari'
        self.game       = args.game
        self._reset_experience()
        self.logger.warning("<-----------------------------------> Env")
        self.logger.warning("Creating {" + self.env_type + " | " + self.game + "} w/ Seed: " + str(self.seed))

    def _reset_experience(self):
        self.exp_state0 = None  
        self.exp_action = None
        self.exp_reward = None
        self.exp_state1 = None
        self.exp_terminal1 = None

    def _get_experience(self):
        return Experience(state0 = self.exp_state0, # NOTE: here state0 is always None
                          action = self.exp_action,
                          reward = self.exp_reward,
                          state1 = self._preprocessState(self.exp_state1),
                          terminal1 = self.exp_terminal1)

    def _preprocessState(self, state):
        raise NotImplementedError("not implemented in base calss")

    @property
    def state_shape(self):
        raise NotImplementedError("not implemented in base calss")

    @property
    def action_dim(self):
        if isinstance(self.env.action_space, Box):
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n

    def render(self):       # render using the original gl window
        raise NotImplementedError("not implemented in base calss")

    def reset(self):
        raise NotImplementedError("not implemented in base calss")

    def step(self, action):
        raise NotImplementedError("not implemented in base calss")

    def get_id(self):
        return self.ind
