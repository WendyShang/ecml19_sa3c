from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.tools import *
from agents.agent import Agent
from collections import namedtuple
from multiprocessing.pool import ThreadPool
import torch.nn as nn
import numpy
import time

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
A3C_Experience = namedtuple('A3C_Experience', 'staten ,state0, action, reward, state1, terminal1, policy_vb, value0_vb')


def preprocess_state(state, dtype, is_volatile=False):
    if isinstance(state, list):
        state_vb = []
        for i in range(len(state)):
            if is_volatile:
                with torch.no_grad():
                    state_vb.append(Variable(torch.from_numpy(state[i]).unsqueeze(0).type(dtype)))
            else:
                state_vb.append(Variable(torch.from_numpy(state[i]).unsqueeze(0).type(dtype)))
    else:
        if is_volatile:
            with torch.no_grad():
                state_vb = Variable(torch.from_numpy(state).unsqueeze(0).type(dtype))
        else:
            state_vb = Variable(torch.from_numpy(state).unsqueeze(0).type(dtype))
    return state_vb

def get_experience(exp_state0=None, exp_action=None, exp_reward=None, exp_state1=None, exp_terminal1=None):
    return Experience(state0 = exp_state0, # NOTE: here state0 is always None
                      action = exp_action,
                      reward = exp_reward,
                      state1 = exp_state1,
                      terminal1 = exp_terminal1)

def reset_rollout():
    # for storing the experiences collected through one rollout
    return A3C_Experience(staten = [],
                          state0 = [],
                          action = [],
                          reward = [],
                          state1 = [],
                          terminal1 = [],
                          policy_vb = [],
                          value0_vb = [],)


class A3CAgent(Agent):
    def __init__(self, args, env_prototype, model_prototype):
        super(A3CAgent, self).__init__(args, env_prototype, model_prototype)
        self.logger.warning("<===================================> A3C-Master {Env(dummy) & Model}")
        self.clip_reward  = not args.not_clip_reward
        self.episode_start = args.eps_start
        self.mean = args.mean
        self.std = args.std
        self.reconLoss = nn.MSELoss()
        self.reconLoss.size_average = True
        self.psi = args.psi
        self.batch_size = args.batch_size
        self.sampling_method = 'mul'
        self.lr_adjusted  = self.lr # adjusted lr

        # dummy_env just to get state_shape & action_dim
        self.dummy_env   = self.env_prototype(self.env_params,1)
        self.state_shape = self.dummy_env.state_shape
        self.action_dim  = self.dummy_env.action_dim
        del self.dummy_env

        # model setup
        self.model_params.state_shape = self.state_shape
        self.model_params.action_dim  = self.action_dim
        self.model_params.dtype = self.dtype
        if args.gpu >= 0:
            self.model_params.cuda = True
            self.gpu = args.gpu
        else:
            self.logger.warning("ERROR: Please use CUDA!")
            raise AssertionError()
        self.model = self.model_prototype(self.model_params)
        self.model.cuda()

        # load pretrained model if provided
        if self.model_file is not None: self._load_model(self.model_file)
        self.save_model_dir = args.save_model_dir

        # learning algorithm
        self.optimizer    = self.optim(self.model.parameters(), lr = self.lr, eps=1e-3)
        if self.optimizer_file is not None: self._load_optimizer(self.optimizer_file)

        # reset variables:
        self.frame_step   = 0 # global frame step counter
        self.train_step   = 0 # global train step counter
        self._reset_training_loggings()
        self.A3C_Experiences = []
        self._reset_lstm_hidden_vb_episode() 
        self._reset_lstm_hidden_vb_rollout() 

    def _reset_training_loggings(self):
        self.p_loss_avg   = 0.
        self.v_loss_avg   = 0.
        self.loss_avg     = 0.
        self.loss_counter = 0

    def _reset_lstm_hidden_vb_episode(self, training=True): # seq_len, batch_size, hidden_dim
        not_training = not training
        if not_training:
            with torch.no_grad():
                self.lstm_hidden_vb = (Variable(torch.zeros(self.batch_size, self.hidden_dim).type(self.dtype)),
                                       Variable(torch.zeros(self.batch_size, self.hidden_dim).type(self.dtype)))
        else:
            self.lstm_hidden_vb = (Variable(torch.zeros(self.batch_size, self.hidden_dim).type(self.dtype)),
                                   Variable(torch.zeros(self.batch_size, self.hidden_dim).type(self.dtype)))


    def _reset_lstm_hidden_vb_rollout(self):
        self.lstm_hidden_vb = (Variable(self.lstm_hidden_vb[0].data),
                               Variable(self.lstm_hidden_vb[1].data))

    def fit_model(self):
        self.logger.warning("<===================================> Training ...")
        self.model.train(True)
        self.optimizer.zero_grad()

        self.start_time = time.time()

        experiences = []
        for i in range(self.batch_size):
            experiences.append(get_experience())

        test_game = self.env_prototype(self.env_params, self.batch_size+1)

        games = []
        for i in range(self.batch_size):
            games.append(self.env_prototype(self.env_params, i))

        pool = ThreadPool()
        def get_state(game):
            id = game.get_id()
            experiences[id] = game.reset()

        def get_state_reinit(game):
            id = game.get_id()
            if experiences[id].terminal1 or eps_counts[id] >= self.max_episode:
                experiences[id] = game.reset()
                self.lstm_hidden_vb[0].data[id] = 0
                self.lstm_hidden_vb[1].data[id] = 0
                eps_counts[id] = 0

            self.A3C_Experiences[id].staten.append(numpy.zeros_like(experiences[id].state1)) 
            
        pool.map(get_state, games)
        eps_counts = [0.] * self.batch_size

        for episode in range(self.episode_start, self.episode_number):
            self.training = True
            self.model.train(self.training)
            self.model.sample_noise()
            self._reset_lstm_hidden_vb_rollout() # detach variable from previous rollout
            self.A3C_Experiences = []
            for i in range(self.batch_size):
                self.A3C_Experiences.append(reset_rollout())
            
            pool.map(get_state_reinit, games)
            for step in range(self.rollout_steps):
                states = torch.cat([preprocess_state(experiences[i].state1, self.dtype) for i in range(self.batch_size)])
                action, p_vb, v_vb  = self._forward(states)
                def step_game(game):
                    id = game.get_id()
                    if not experiences[id].terminal1:
                        if step < self.rollout_steps - 1:
                            self.A3C_Experiences[id].staten.append(experiences[id].state1)
                        self.A3C_Experiences[id].state0.append(experiences[id].state1)
                        accum_reward = 0
                        for _ in range(self.skip_frame):
                            experiences[id] = game.step(action[id][0])
                            accum_reward += experiences[id].reward
                            if experiences[id].terminal1:
                                break
                        eps_counts[id] += self.skip_frame
                        self.A3C_Experiences[id].action.append(action[id])
                        if self.clip_reward:
                            if accum_reward > 0:
                                self.A3C_Experiences[id].reward.append(min(accum_reward,1))
                            elif accum_reward == 0:
                                self.A3C_Experiences[id].reward.append(0)
                            else:
                                self.A3C_Experiences[id].reward.append(max(accum_reward, -1))
                        else:
                            if accum_reward > 0:
                                self.A3C_Experiences[id].reward.append(accum_reward)
                        self.A3C_Experiences[id].state1.append(experiences[id].state1)
                        self.A3C_Experiences[id].terminal1.append(experiences[id].terminal1)
                        self.A3C_Experiences[id].policy_vb.append(p_vb[id].unsqueeze(0))
                        self.A3C_Experiences[id].value0_vb.append(v_vb[id].unsqueeze(0))
                pool.map(step_game, games)
                

            # update model
            final_states = torch.cat([preprocess_state(experiences[i].state1, self.dtype, True) for i in range(self.batch_size)])
            self._backward(final_states)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.train_step += 1


            # test model if good enough then save
            if (episode+1) % self.eval_freq == 0:
                self._test_and_save(episode, test_game)

    def _test_and_save(self, episode, test_game):
        self.training = False
        self.model.train(self.training)
        total_reward = 0
        with torch.no_grad():
            test_lstm_hidden_vb = (Variable(torch.zeros(1, self.hidden_dim).type(self.dtype), volatile=True),
                                   Variable(torch.zeros(1, self.hidden_dim).type(self.dtype), volatile=True))
            current_experience = test_game.reset()
            test_game.visual()
            eps_count = 0
            test_actions = []
            test_done = False
            while eps_count < self.max_episode and not test_done:
                test_p_vb, test_v_vb, test_lstm_hidden_vb = self.model(
                    preprocess_state(current_experience.state1, self.dtype, is_volatile=True), test_lstm_hidden_vb)
                test_action = test_p_vb.max(1)[1].data[0]
                test_actions.append(test_action)
                for _ in range(self.skip_frame):
                    current_experience = test_game.step(test_action)
                    test_game.visual()
                    total_reward = total_reward + current_experience.reward
                    if current_experience.terminal1:
                        test_done = True
                        break
                eps_count = eps_count + self.skip_frame

            self._save_model(episode, total_reward)

    def _forward(self, state_vb):
        p_vb, v_vb, self.lstm_hidden_vb = self.model(state_vb, self.lstm_hidden_vb)
        if self.training:
            if self.sampling_method == 'mul':
                action = p_vb.multinomial(1).data
            else:
                self.logger.warning("ERROR: Wrong sampling method")
                raise AssertionError()
        else:
            action = p_vb.max(1)[1].data
        return action,torch.clamp(p_vb, min=0.000001, max=0.999999),v_vb

    def _backward(self, sT_vb):
        self.optimizer.zero_grad()
        # preparation
        _, valueT_vb, _ = self.model(sT_vb, self.lstm_hidden_vb)
        for i in range(self.batch_size):
            if self.A3C_Experiences[i].terminal1[-1]:
                valueT_vb.data[i] = 0
        valueT_vb = Variable(valueT_vb.data)
        rollout_steps = [len(self.A3C_Experiences[i].reward) for i in range(self.batch_size)]
        policy_vb = [self.A3C_Experiences[i].policy_vb for i in range(self.batch_size)]
        action_batch_vb = [self.A3C_Experiences[i].action for i in range(self.batch_size)]
        policy_log_vb = [[torch.log(policy_vb[i][j]) for j in range(len(policy_vb[i]))] for i in range(len(policy_vb))]
        entropy_vb    = [[- (policy_log_vb[i][j] * policy_vb[i][j]).sum(1) for j in range(len(policy_vb[i]))] for i in range(len(policy_vb))]
        policy_log_vb = [[policy_log_vb[i][j].gather(1,Variable(action_batch_vb[i][j]).unsqueeze(0).detach()) for j in range(len(action_batch_vb[i]))] for i in range(len(action_batch_vb))]
        for i in range(self.batch_size):
            # NOTE: only this last entry is Volatile, all others are still in the graph
            self.A3C_Experiences[i].value0_vb.append(Variable(valueT_vb.data[i]))
        gae_ts = torch.zeros(self.batch_size, 1)
        gae_ts = gae_ts.cuda()

        # compute loss
        policy_loss_vb = [0. for i in range(self.batch_size)]
        value_loss_vb  = [0. for i in range(self.batch_size)]
        loss_model_vb = 0
        for j in range(self.batch_size):
            for i in reversed(range(rollout_steps[j])):
                valueT_vb[j]     = self.gamma * valueT_vb[j] + self.A3C_Experiences[j].reward[i]
                advantage_vb  = valueT_vb[j] - self.A3C_Experiences[j].value0_vb[i]
                value_loss_vb[j] = value_loss_vb[j] + 0.5 * advantage_vb.pow(2)
                tderr_ts = self.A3C_Experiences[j].reward[i] + self.gamma * self.A3C_Experiences[j].value0_vb[i + 1].data - self.A3C_Experiences[j].value0_vb[i].data
                gae_ts[j]   = gae_ts[j] * self.tau * self.gamma + tderr_ts
                policy_loss_vb[j] = policy_loss_vb[j] - (policy_log_vb[j][i] * Variable(gae_ts[j]) + self.beta * entropy_vb[j][i])
            loss_model_vb = loss_model_vb + (policy_loss_vb[j] + self.lam * value_loss_vb[j])/rollout_steps[j] 

        loss_model_vb.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

        p_loss_avg = 0
        v_loss_avg = 0
        loss_avg = loss_model_vb.data.cpu().numpy()
        for i in range(self.batch_size):
            p_loss_avg += policy_loss_vb[i].data.cpu().numpy()/self.batch_size
            v_loss_avg += value_loss_vb[i].data.cpu().numpy()/self.batch_size

        # log training stats
        self.p_loss_avg   +=  p_loss_avg
        self.v_loss_avg   +=  v_loss_avg
        self.loss_avg     += loss_model_vb.data.cpu().numpy()
        self.loss_counter += 1

        self.logger.warning("Reporting       @ Step: " + str(self.train_step) + " | Elapsed Time: " + str(time.time() - self.start_time))
        self.logger.warning("Iteration: {}; lr: {}".format(self.train_step, self.lr_adjusted))
        self.logger.warning("Iteration: {}; current p_loss: {}; average p_loss: {}".format(self.train_step, p_loss_avg, self.p_loss_avg/self.loss_counter))
        self.logger.warning("Iteration: {}; current v_loss: {}; average v_loss: {}".format(self.train_step, v_loss_avg, self.v_loss_avg/self.loss_counter))
        self.logger.warning("Iteration: {}; current loss  : {}; average loss  : {}".format(self.train_step, loss_avg,   self.loss_avg/self.loss_counter))

    def testing(self, num_iter):
        self.training = False
        self.model.train(self.training)
        total_reward_list = []
        for i in range(num_iter):
            test_game = self.env_prototype(self.env_params, i)
            current_reward = 0
            test_lstm_hidden_vb = (Variable(torch.zeros(1, self.hidden_dim).type(self.dtype), volatile=True),
                                   Variable(torch.zeros(1, self.hidden_dim).type(self.dtype), volatile=True))
            current_experience = test_game.reset()
            test_done = False
            while not test_done:
                test_p_vb, test_v_vb, test_lstm_hidden_vb = self.model(preprocess_state(current_experience.state1, self.dtype, is_volatile=True), test_lstm_hidden_vb)
                test_action = test_p_vb.max(1)[1].data[0]
                for _ in range(self.skip_frame):
                    current_experience = test_game.step(test_action)
                    current_reward = current_reward + current_experience.reward
                    if current_experience.terminal1:
                        test_done = True
                        break
     
            self.logger.warning("Iteration: {}; reward: {}".format(i, current_reward))
            total_reward_list.append(current_reward)

        total_reward_list = np.array(total_reward_list)
        self.logger.warning("mean reward; {}; std reward: {}".format(total_reward_list.mean(), total_reward_list.std()))
