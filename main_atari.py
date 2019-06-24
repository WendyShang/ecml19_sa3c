from utils.tools import loggerConfig, read_config
from envs.atari_env import AtariEnv
from models.a3c_lstm_baseline import A3C_LSTM_BASELINE, A3C_LSTM_BASELINE_CRELU
from models.a3c_lstm_sa3c    import A3C_LSTM_SA3C,   A3C_LSTM_SA3C_CRELU
from models.a3c_lstm_fsa3c    import A3C_LSTM_FSA3C,   A3C_LSTM_FSA3C_CRELU
from models.a3c_lstm_hpa3c    import A3C_LSTM_HPA3C,   A3C_LSTM_HPA3C_CRELU
from models.a3c_lstm_nn       import A3C_LSTM_NN,   A3C_LSTM_NN_CRELU
import argparse
import os
import torch

parser = argparse.ArgumentParser(description='A3C LSTM for Atari')
parser.add_argument( '--lr',             type=float,    default=0.0005,         help='learning rate')
parser.add_argument( '--gamma',          type=float,    default=0.95,           help='discount factor for rewards')
parser.add_argument( '--tau',            type=float,    default=1.00,           help='parameter for GAE')
parser.add_argument( '--beta',           type=float,    default=0.01,           help='coefficient for entropy')
parser.add_argument( '--lam',            type=float,    default=5,              help='coefficient for value')
parser.add_argument( '--clip_grad',      type=float,    default=40.00,          help='gradient clipping')
parser.add_argument( '--psi',            type=float,    default=0.00001,        help='coefficient for KL')
parser.add_argument( '--sig',            type=float,    default=6,              help='-log(var) of the stochastic units')
parser.add_argument( '--eps_start',      type=int,      default=0,              help='number or episode to start')
parser.add_argument( '--seed',           type=int,      default=896,            help='random seed')
parser.add_argument( '--num_steps',      type=int,      default=20,             help='number of forward steps in A3C')
parser.add_argument( '--batch_size',     type=int,      default=64,             help='batch size, currently equal the number of workers')
parser.add_argument( '--max_episode',    type=int,      default=10000,          help='maximum length of an episode')
parser.add_argument( '--eps_number',     type=int,      default=150000,         help='number of episodes')
parser.add_argument( '--hidden_dim',     type=int,      default=512,            help='hidden dim for LSTM')
parser.add_argument( '--skip_frame',     type=int,      default=1,              help='number of frames given by the environment to skip')
parser.add_argument( '--gpu',            type=int,      default=0,              help='0 with gpu, -1 with cpu but currently not supported')
parser.add_argument( '--preprocess_mode',type=int,      default=4,              help='grey 3 | rgb 4')
parser.add_argument( '--model_type',                    default='baseline',     help='baseline|sa3c|fsa3c|hpa3c|nn')
parser.add_argument( '--game',                          default='Seaquest-v4',  help='game to train on')
parser.add_argument( '--env_config',                    default='config.json',  help='environment to crop and resize info')
parser.add_argument( '--optimizer',                     default='adam',         help='type of optimizer, currently only support adam')
parser.add_argument( '--trained',                       default=None,           help='folder to load trained models from')
parser.add_argument( '--save_model_dir',                default='save/',      help='folder to save models and optimization states')
parser.add_argument( '--log_dir',                       default='logs/',        help='folder to save logs')
parser.add_argument( '--save_best',                     action='store_true',    help='save the best model')
parser.add_argument( '--crelu',                         action='store_true',    help='enable CRELU or not')
parser.add_argument( '--not_clip_reward',               action='store_true',    help='clip reward or not')
parser.add_argument( '--verbose',                       action='store_true',    help='verbose or not')

args = parser.parse_args()

args.run_name = args.model_type + '_' + args.game + '_lr_' + str(args.lr) + '_gamma_' + str(args.gamma) + '_tau_' + str(args.tau) + '_psi_' + str(args.psi) + '_lam_' + str(args.lam) + '_seed_' + str(args.seed) + '_sig_' + str(args.sig) + '_start_' + str(args.eps_start)
if args.crelu:
    args.run_name = 'crelu_' + args.run_name 
if args.not_clip_reward:
    args.run_name = 'norewardclip_' + args.run_name

args.save_model_dir = args.save_model_dir + args.run_name + '/'
if not os.path.exists(args.save_model_dir): os.makedirs(args.save_model_dir)
if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)


args.logger = loggerConfig(args.log_dir + '/' + args.run_name + '.log', args.verbose)
#torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(args.seed)
setup_json = read_config(args.env_config)
for i in setup_json.keys():
    if i in args.game:
        args.crop1 = setup_json[i]['crop1']
        args.crop2 = setup_json[i]['crop2']
        # currently only support 80 x 80
        args.hei_state = 80              
        args.wid_state = 80
        args.mean = []
        args.mean.append(setup_json[i]['mean_red'])
        args.mean.append(setup_json[i]['mean_green'])
        args.mean.append(setup_json[i]['mean_blue'])
        args.std = []
        args.std.append(setup_json[i]['std_red'])
        args.std.append(setup_json[i]['std_green'])
        args.std.append(setup_json[i]['std_blue'])
        

args.logger.warning("<-----------------------------------> args")
args.logger.warning(args)

if args.model_type == 'baseline':
    from agents.a3c_baseline import A3CAgent
    if args.crelu:
        my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_BASELINE_CRELU)
    else:
        my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_BASELINE)
elif args.model_type == 'sa3c':
    from agents.a3c_sa3c import A3CAgent
    if args.crelu:
        my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_SA3C_CRELU)
    else:
        my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_SA3C)
elif args.model_type == 'fsa3c':
    from agents.a3c_sa3c import A3CAgent
    if args.crelu:
        my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_SA3C_CRELU)
    else:
        my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_SA3C)
elif args.model_type == 'hpa3c':
    from agents.a3c_hpa3c import A3CAgent
    my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_HPA3C)
elif args.model_type == 'nn':
    from agents.a3c_nn import A3CAgent
    if args.crelu:
        my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_NN_CRELU)
    else:
        my_agent = A3CAgent(args, AtariEnv, A3C_LSTM_NN)

#to reproduce exact training process
#torch.backends.cudnn.deterministic=True
my_agent.fit_model()
