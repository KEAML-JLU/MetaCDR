# Date: 16 December, 2020

import argparse
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Cross-domain Recommendation via Meta-Learning')
    parser.add_argument('--seed', type=int, default=66, help='set random seed for model')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=16, help='number of tasks in each batch per meta-update')
    
    parser.add_argument('--lr_inner', type=float, default=0.01, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=0.005, help='outer-loop learning rate (used with Adam optimiser)')
    
    parser.add_argument('--num_grad_steps_inner', type=int, default=5, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=1, help='number of gradient updates at test time (for evaluation)')

    parser.add_argument('--data_root', type=str, default="datasets/ml-1m/", help='path to data root')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--test', action='store_true', default=False, help='[train] or test')
    
    parser.add_argument('--embedding_dim', type=int, default=32, help='num of workers to use')
    # parser.add_argument('--layers', nargs='?', default='[64,64]', help='Size of each layer')
    parser.add_argument('--num_epoch', type=int, default=30, help='num of epoch')
    parser.add_argument('--gamma', type=float, default=1., help='gamma')
    parser.add_argument('--lamda', type=float, default=0., help='lamda')

    parser.add_argument('--num_genre', type=int, default=25, help='num of')
    parser.add_argument('--num_director', type=int, default=2186, help='num of')
    parser.add_argument('--num_actor', type=int, default=8030, help='num of')
    parser.add_argument('--num_rate', type=int, default=6, help='num of')
    parser.add_argument('--num_gender', type=int, default=2, help='num of')
    parser.add_argument('--num_age', type=int, default=7, help='num of')
    parser.add_argument('--num_occupation', type=int, default=21, help='num of')
    parser.add_argument('--num_zipcode', type=int, default=3402, help='num of')

    parser.add_argument('--source_min', type=int, default=13, help='The number of movies the user has watched the least in the source domain')
    parser.add_argument('--source_max', type=int, default=100, help='The number of movies the user has watched the most in the source domain')
    parser.add_argument('--target_min', type=int, default=13, help='The number of movies the user has watched the least in the target domain')
    parser.add_argument('--target_max', type=int, default=100, help='The number of movies the user has watched the most in the target domain')

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # please chose your own cuda
    print('Running on device: {}'.format(args.device))
    return args