# Date: 18 December, 2020
# Description: A PyTorch implementation of dataloader for Cross-Domain Recommendation System via Meta-learning(MetaCDR)

# Python imports
import pickle
import json
import numpy as np
import random
from tqdm import tqdm
import gc

# Workspace imports
from argslist import parse_args

# Pytorch imports
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Set random seed
def set_seed(seed=66):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class DomainData(Dataset):
    def __init__(self, args, test=False):
        super(DomainData, self).__init__()
        self.dataset_path = args.data_root  #  source file path

        self.movie_dict = pickle.load(open("{}m_movie_dict.pkl".format(self.dataset_path), "rb"))
        self.user_dict = pickle.load(open("{}m_user_dict.pkl".format(self.dataset_path), "rb"))
        '''
        movie_dict(dict): {movie_id(int): tensor([[movie vector(dim=10242)]])}
        user_dict(dict): {user_id(int): tensor([[user vector(dim=4)]])}
        '''
        state = ['source', 'target']
        with open('{}{}.json'.format(self.dataset_path, state[0]), encoding='utf-8') as f:
            self.dataset_split_source = json.loads(f.read())
        with open('{}{}_y.json'.format(self.dataset_path, state[0]), encoding='utf-8') as f:
            self.dataset_split_source_y = json.loads(f.read())
            # print('source:{}'.format(len(self.dataset_split_source)))
        with open('{}{}.json'.format(self.dataset_path, state[1]), encoding='utf-8') as f:
            self.dataset_split_target = json.loads(f.read())
        with open('{}{}_y.json'.format(self.dataset_path, state[1]), encoding='utf-8') as f:
            self.dataset_split_target_y = json.loads(f.read())
            # print('target:{}'.format(len(self.dataset_split_target)))

        '''
        dataset_split_source(dict): {user_id(str): [movie_id(str),...]}
        '''
        
        # self.final_index_source = []
        # for _, user_id in tqdm(enumerate(list(self.dataset_split_source.keys()))):
        #     u_id = int(user_id)
        #     seen_movie_len = len(self.dataset_split_source[str(u_id)])
        #     if seen_movie_len < args.source_min or seen_movie_len > args.source_max:
        #         continue
        #     else:
        #         self.final_index_source.append(user_id)
        # self.final_index_target = []
        # for _, user_id in tqdm(enumerate(list(self.dataset_split_target.keys()))):
        #     u_id = int(user_id)
        #     seen_movie_len = len(self.dataset_split_target[str(u_id)])
        #     if seen_movie_len < args.target_min or seen_movie_len > args.target_max:
        #         continue
        #     else:
        #         self.final_index_target.append(user_id)
        # self.final_index = []
        # for user in self.final_index_target:
        #     if user in self.final_index_source:
        #         self.final_index.append(user)
        
        if test == False:
            with open('{}final_index.txt'.format(self.dataset_path), 'r', encoding='utf-8') as f:
                self.final_index = eval(f.read())[:1187]
            print('train dataset')
        elif test == True:
            with open('{}final_index.txt'.format(self.dataset_path), 'r', encoding='utf-8') as f:
                self.final_index = eval(f.read())[1188:]
            print('test dataset')
        '''
        final_index(list): [user_id(str),...]
        '''
        domain_gap = []
        for u_id in self.final_index:
            seen_movie_source = len(self.dataset_split_source[str(u_id)])
            seen_movie_target = len(self.dataset_split_target[str(u_id)])
            gap = seen_movie_source - seen_movie_target
            domain_gap.append(gap)

        print('Final user:{} Average user gap:{} '.format(len(self.final_index), np.mean(domain_gap)))
        
    def __len__(self):
        return len(self.final_index)

    def __getitem__(self, index):
        user_id = self.final_index[index]
        u_id = int(user_id)
        seen_movie_len_source = len(self.dataset_split_source[str(u_id)])
        seen_movie_len_target = len(self.dataset_split_target[str(u_id)])

        indices_source = list(range(seen_movie_len_source))
        indices_target = list(range(seen_movie_len_target))
        random.shuffle(indices_source)
        random.shuffle(indices_target)

        tmp_x_source = np.array(self.dataset_split_source[str(u_id)])
        tmp_y_source = np.array(self.dataset_split_source_y[str(u_id)])
        tmp_x_target = np.array(self.dataset_split_target[str(u_id)])
        tmp_y_target = np.array(self.dataset_split_target_y[str(u_id)])
        '''
        x.dtype: numpy.str_
        y.dtype: numpy.int32
        '''
        support_x_source, support_x_target = [], []
        support_y_source, support_y_target = [], []

        for m_id_source in tmp_x_source[indices_source[:-10]]:
            for m_id_target in tmp_x_target[indices_target[:-10]]:
                m_id_source = int(m_id_source)
                m_id_target = int(m_id_target)
                tmp_x_converted_source = torch.cat((self.movie_dict[m_id_source], self.user_dict[u_id]), 1)
                tmp_x_converted_target = torch.cat((self.movie_dict[m_id_target], self.user_dict[u_id]), 1)

                support_x_source.append(tmp_x_converted_source)
                support_x_target.append(tmp_x_converted_target)
                '''
                len(support_x_source) = len(support_x_target)
                support_x_source(list): [tensor([[movie user vector(dim = 10246)]]) * 5353]
                '''
        
        for m_id_source in tmp_y_source[indices_source[:-10]]:
            for m_id_target in tmp_y_target[indices_target[:-10]]:
                support_y_source.append(m_id_source.astype(np.float32))
                support_y_target.append(m_id_target.astype(np.float32))

        query_x_source, query_x_target = [], []
        query_y_source, query_y_target = [], []

        for m_id_source in tmp_x_source[indices_source[-10:]]:
            for m_id_target in tmp_x_target[indices_target[-10:]]:
                m_id_source = int(m_id_source)
                m_id_target = int(m_id_target)
                tmp_x_converted_source = torch.cat((self.movie_dict[m_id_source], self.user_dict[u_id]), 1)
                
                tmp_x_converted_target = torch.cat((self.movie_dict[m_id_target], self.user_dict[u_id]), 1)

                query_x_source.append(tmp_x_converted_source)
                query_x_target.append(tmp_x_converted_target)

        for m_id_source in tmp_y_source[indices_source[-10:]]:
            for m_id_target in tmp_y_target[indices_target[-10:]]:
                query_y_source.append(m_id_source.astype(np.float32))
                query_y_target.append(m_id_target.astype(np.float32))

        sample_indices = list(range(len(support_x_source)))
        random.shuffle(sample_indices)

        final_support_x_source, final_support_x_target = [], []
        final_support_y_source, final_support_y_target = [], []
        
        for indice in sample_indices:
            final_support_x_source.append(support_x_source[indice])
            final_support_x_target.append(support_x_target[indice])           
            final_support_y_source.append(support_y_source[indice])
            final_support_y_target.append(support_y_target[indice])

        final_support_x_source = torch.cat(final_support_x_source, 0)
        final_support_x_target = torch.cat(final_support_x_target, 0) 

        sample_indices = list(range(len(query_x_source)))
        random.shuffle(sample_indices)

        final_query_x_source, final_query_x_target = [], []
        final_query_y_source, final_query_y_target = [], []
        
        for indice in sample_indices:
            final_query_x_source.append(query_x_source[indice])
            final_query_x_target.append(query_x_target[indice])
            final_query_y_source.append(query_y_source[indice])
            final_query_y_target.append(query_y_target[indice])

        final_query_x_source = torch.cat(final_query_x_source, 0)
        final_query_x_target = torch.cat(final_query_x_target, 0)

        # del support_x_source, support_x_target, support_y_source, support_y_target, query_x_source, query_x_target, query_y_source, query_y_target
        # gc.collect()

        return final_support_x_source, final_support_x_target, np.array(final_support_y_source).reshape(-1, 1), np.array(final_support_y_target).reshape(-1,1), final_query_x_source, final_query_x_target, np.array(final_query_y_source).reshape(-1,1), np.array(final_query_y_target).reshape(-1,1)



