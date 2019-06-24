from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.preprocess import preprocessAtari, preprocessAtariRGB
from collections import namedtuple
from envs.env import Env

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class AtariEnv(Env):  # pixel-level inputs
    def __init__(self, args, env_ind=0):
        super(AtariEnv, self).__init__(args, env_ind)

        assert self.env_type == "atari"
        try: import gym
        except ImportError as e: self.logger.warning("WARNING: gym not found")

        self.env = gym.make(self.game)
        self.env.seed(self.seed)    # NOTE: so each env would be different
        
        self.actions     = range(self.action_dim)
        self.logger.warning("Action Space: %s", self.actions)
        self.hei_state = args.hei_state
        self.wid_state = args.wid_state
        self.crop1 = args.crop1
        self.crop2 = args.crop2
        self.mean = args.mean
        self.std = args.std
        self.new_mean = args.mean
        self.new_std = args.std
        self.preprocess_mode = args.preprocess_mode
        # currently only support 80x80
        assert self.hei_state == self.wid_state and self.hei_state == 80
        self.logger.warning("State  Space: (" + str(self.state_shape) + " * " + str(self.state_shape) + ")")

    def _preprocessState(self, state):
        if self.preprocess_mode == 4:   
            state, self.new_mean, self.new_std = preprocessAtariRGB(state, self.crop1, self.crop2, self.mean, self.std)
            return state.reshape(3, self.hei_state, self.wid_state)
        elif self.preprocess_mode == 3:   
            state = preprocessAtari(state)
            return state.reshape(self.hei_state * self.wid_state)
        else: 
            self.logger.warning("ERROR: wrong preprocessing code")
            raise AssertionError()

    @property
    def state_shape(self):
        return self.hei_state

    def render(self):
        return self.env.render('rgb_array')

    def visual(self):
        frame_name = self.img_dir + "frame_%05d.jpg" % self.frame_ind
        self.imsave(frame_name, self.exp_state1)
        self.frame_ind += 1

    def sample_random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self.frame_ind = 0
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.actions[self.exp_action])
        return self._get_experience()
