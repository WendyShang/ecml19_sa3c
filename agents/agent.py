from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import torch
import torch.optim as optim
import os.path

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class Agent(object):
    def __init__(self, args, env_prototype, model_prototype):
        # logging
        self.logger = args.logger
        self.model = None

        if args.gpu >= 0:
            self.use_cuda = True
            self.dtype = torch.cuda.FloatTensor
        else:
           self.logger.warning("ERROR: only support CUDA!")
           raise AssertionError()

        if args.optimizer == 'adam':
            self.optim = optim.Adam
        else:
            self.logger.warning("ERROR: only support ADAM!")
            raise AssertionError()

        # prototypes for env & model 
        self.env_prototype = env_prototype          # NOTE: instantiated in fit_model() of inherited Agents
        self.env_params = args
        self.model_prototype = model_prototype      # NOTE: instantiated in fit_model() of inherited Agents
        self.model_params = args
        self.max_episode = args.max_episode

        # model and optimization loading and saving
        self.model_name = args.save_model_dir + '/model.pth'         # NOTE: will save the current model to model_name
        self.model_path = args.save_model_dir
        self.optimizer_name = args.save_model_dir + '/optimizer.pth' # NOTE: will save the current model to model_name
        self.trained = args.trained
        if args.trained is not None:
            self.model_file = args.trained + '/model.pth'            # NOTE: will load pretrained model_file if not None
            self.optimizer_file = args.trained + '/optimizer.pth'    # NOTE: will load pretrained model_file if not None
        else:
            self.model_file = None
            self.optimizer_file = None
        self.save_best = args.save_best
        if self.save_best:
            self.best_step   = None                 # NOTE: achieves best_reward at this step
            self.best_reward = None                 # NOTE: only save a new model if achieves higher reward

        # hyperparameters
        self.episode_number = args.eps_number                           
        self.early_stop = None
        self.gamma = args.gamma
        self.clip_grad = args.clip_grad
        self.lr = args.lr
        self.eval_freq = 100
        self.enable_log_at_train_step = True
        self.rollout_steps = args.num_steps
        self.tau = args.tau
        self.beta = args.beta
        self.lam =args.lam
        self.skip_frame = args.skip_frame
        self.hist_len = 1
        self.hidden_dim = args.hidden_dim
        self.model_params.hist_len = self.hist_len


    def _reset_experience(self):
        self.experience = Experience(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False)

    def _load_model(self, model_file):
        if model_file:
            self.logger.warning("Loading Model: " + self.model_file + " ...")
            self.model.load_state_dict(torch.load(model_file))
            self.logger.warning("Loaded  Model: " + self.model_file + " ...")
        else:
            self.logger.warning("No Pretrained Model. Will Train From Scratch.")

    def _load_optimizer(self, optimizer_file):
        if optimizer_file:
            if not os.path.exists(optimizer_file):
                self.logger.warning("Can NOT find optimizer file")
            else:
                self.logger.warning("Loading Model: " + self.optimizer_file + " ...")
                self.optimizer.load_state_dict(torch.load(optimizer_file))
                self.logger.warning("Loaded  Model: " + self.optimizer_file + " ...")
        else:
            self.logger.warning("No Previous Optimizer Record. Will Train From Scratch.")

    def _save_model(self, step, curr_reward):
        self.logger.warning("curr reward     @ Step: " + str(step) + ": " + str(curr_reward))
        torch.save(self.model.state_dict(), self.model_name)
        torch.save(self.optimizer.state_dict(), self.optimizer_name)
        self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ".")
        if self.save_best:
            if self.best_step is None:
                self.best_step   = step
                self.best_reward = curr_reward
            if curr_reward >= self.best_reward:
                self.best_step   = step
                self.best_reward = curr_reward
                torch.save(self.model.state_dict(), self.model_path + '/model_' + str(step) + '.pth')
                torch.save(self.optimizer.state_dict(), self.model_path + '/optimizer_' + str(step) + '.pth')
                self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ". {Best Step: " + str(self.best_step) + " | Best Reward: " + str(self.best_reward) + "}")
            else:
                self.logger.warning("Not saving model")

    def _forward(self, observation):
        raise NotImplementedError("not implemented in base class")

    def _backward(self, reward, terminal):
        raise NotImplementedError("not implemented in base class")

    def _eval_model(self):  # evaluation during training
        raise NotImplementedError("not implemented in base class")

    def fit_model(self):    # training
        raise NotImplementedError("not implemented in base class")

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError("not implemented in base class")
