# Author: Pang Haoyu
# Date: 19 December, 2020
# Description: A PyTorch implementation of Cross-Domain Recommendation System via Meta-learning(MetaCDR)

# Python imports
import copy
import os
import sys
import numpy as np
import time
from tqdm import tqdm
import random
import gc


# Pytorch imports
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Workspace imports
from argslist import parse_args
from data import DomainData
from model import RecNet
from logger import Logger

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # please choose your own cuda

# Set random seed
def set_seed(seed=66):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args, load_model=False, model_path='./'):
    if load_model:
        pass

    start_time = time.time()
    set_seed(args.seed)

    #-----------------------train---------------------
    #-------------------------------------------------

    model = RecNet(args).to(args.device)
    model.train()

    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
    logger = Logger()
    logger.args = args
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    dataloader_train = DataLoader(DomainData(args), batch_size=1, num_workers=args.num_workers)
    
    for epoch in range(args.num_epoch):

        x_spt_source, y_spt_source, x_qry_source, y_qry_source = [], [], [], []
        x_spt_target, y_spt_target, x_qry_target, y_qry_target = [], [], [], []
        iter_counter = 0
        for step, batch in enumerate(dataloader_train):
            if len(x_spt_source)<args.tasks_per_metaupdate:
                x_spt_source.append(batch[0][0].to(args.device))
                x_spt_target.append(batch[1][0].to(args.device))
                y_spt_source.append(batch[2][0].to(args.device))
                y_spt_target.append(batch[3][0].to(args.device))
                x_qry_source.append(batch[4][0].to(args.device))
                x_qry_target.append(batch[5][0].to(args.device))
                y_qry_source.append(batch[6][0].to(args.device))
                y_qry_target.append(batch[7][0].to(args.device))
                if not len(x_spt_source) == args.tasks_per_metaupdate:
                    continue
            if len(x_spt_source) != args.tasks_per_metaupdate:
                continue

            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init) 
            loss_pre_source = []
            loss_pre_target = []
            loss_pre = [] 
            loss_after_source = []
            loss_after_target = []
            loss_after = [] 
            for i in range(args.tasks_per_metaupdate):
                model.eval()
                with torch.no_grad():
                    qry_out_source, qry_out_target = model(x_qry_source[i], x_qry_target[i])
                    qry_loss_source = F.mse_loss(qry_out_source, y_qry_source[i]).item()
                    qry_loss_target = F.mse_loss(qry_out_target, y_qry_target[i]).item()
                    l1_reg = 0
                    for name, param in model.final_part.named_parameters():
                        if name == 'fc1.param.h' or name == 'fc2.param.h':
                            l1_reg += torch.sum(abs(param))
                    qry_loss_pre = qry_loss_source + qry_loss_target + args.lamda * l1_reg.item()
                loss_pre_source.append(qry_loss_source)
                loss_pre_target.append(qry_loss_target)
                loss_pre.append(qry_loss_pre)
                model.train()
                fast_parameters = model.final_part.parameters()
                for weight in model.final_part.parameters():
                    weight.fast = None
                for k in range(args.num_grad_steps_inner):
                    logits_source, logits_target = model(x_spt_source[i], x_spt_target[i])  # orign param/ fast param
                    loss_source = F.mse_loss(logits_source, y_spt_source[i])
                    loss_target = F.mse_loss(logits_target, y_spt_target[i])
                    l1_reg = 0
                    for name, param in model.final_part.named_parameters():
                        if name == 'fc1.param.h' or name == 'fc2.param.h':
                            if param.fast is not None:
                                l1_reg += torch.sum(abs(param.fast))
                            else:
                                l1_reg += torch.sum(abs(param))
                    loss = loss_source + loss_target + args.lamda * l1_reg
                    grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                    fast_parameters = []
                    for k, weight in enumerate(model.final_part.parameters()):
                        if weight.fast is None:
                            weight.fast = weight - args.lr_inner * grad[k]
                        else:
                            weight.fast = weight.fast - args.lr_inner * grad[k]
                        fast_parameters.append(weight.fast)

                logits_q_source, logits_q_target = model(x_qry_source[i], x_qry_target[i])  # fast param
                loss_q_source = F.mse_loss(logits_q_source, y_qry_source[i])
                loss_q_target = F.mse_loss(logits_q_target, y_qry_target[i])
                l1_reg = 0
                for name, param in model.final_part.named_parameters():
                    if name == 'fc1.param.h' or name == 'fc2.param.h':
                        if param.fast is not None:
                            l1_reg += torch.sum(abs(param.fast))
                        else:
                            l1_reg += torch.sum(abs(param))
                loss_q = loss_q_source + loss_q_target + args.lamda * l1_reg

                loss_after_source.append(loss_q_source.item())
                loss_after_target.append(loss_q_target.item())
                loss_after.append(loss_q.item())
                task_grad_test = torch.autograd.grad(loss_q, model.parameters())

                for g in range(len(task_grad_test)):
                    meta_grad[g] += task_grad_test[g].detach()

            # -------------------------------- meta update ------------------------------------

            meta_optimiser.zero_grad()

            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                param.grad.data.clamp_(-10,10)
           
            meta_optimiser.step()
            
            # del x_spt_source, x_spt_target, y_spt_source, y_spt_target, x_qry_source, x_qry_target, y_qry_source, y_qry_target
            # gc.collect()

            x_spt_source, x_spt_target, y_spt_source, y_spt_target = [], [], [], []
            x_qry_source, x_qry_target, y_qry_source, y_qry_target = [], [], [], []

            loss_pre = np.array(loss_pre)
            loss_after = np.array(loss_after)
            logger.train_loss.append(np.mean(loss_pre))
            logger.valid_loss.append(np.mean(loss_after))
            logger.train_conf.append(1.96*np.std(loss_pre, ddof=0)/np.sqrt(len(loss_pre)))
            logger.valid_conf.append(1.96*np.std(loss_after, ddof=0)/np.sqrt(len(loss_after)))
            logger.test_loss.append(0)
            logger.test_conf.append(0)

            logger.print_info(epoch, iter_counter, start_time)
            start_time = time.time()

            iter_counter += 1

        torch.save(model.state_dict(), 'model_save/parameter_{}epoch.pkl'.format(epoch))
        if epoch%5==4:
            model_test = copy.deepcopy(model)
            dataloader_test = DataLoader(DomainData(args, test=True), batch_size=1, num_workers=args.num_workers)
            evaluate_test(args, model_test, dataloader_test)

    return model


def evaluate_test(args, model, dataloader):
    
    loss_all = []
    loss_source, loss_target = [], []
    for c, batch in enumerate(dataloader):
        x_spt_source = batch[0].to(args.device)
        x_spt_target = batch[1].to(args.device)
        y_spt_source = batch[2].to(args.device)
        y_spt_target = batch[3].to(args.device)
        x_qry_source = batch[4].to(args.device)
        x_qry_target = batch[5].to(args.device)
        y_qry_source = batch[6].to(args.device)
        y_qry_target = batch[7].to(args.device)
        for i in range(x_spt_source.shape[0]):
            # -----------------inner update---------------
            model.train()
            fast_parameters = model.final_part.parameters()
            for weight in model.final_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                logits_source, logits_target = model(x_spt_source[i], x_spt_target[i])
                loss = F.mse_loss(logits_source, y_spt_source[i]) + F.mse_loss(logits_target, y_spt_target[i])
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.final_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad[k]
                    else: 
                        weight.fast = weight.fast - args.lr_inner * grad[k]
                    fast_parameters.append(weight.fast)
            model.eval()
            with torch.no_grad():
                out_source, out_target = model(x_qry_source[i], x_qry_target[i])

                loss_source.append(F.l1_loss(out_source, y_qry_source[i]).item())
                loss_target.append(F.l1_loss(out_target, y_qry_target[i]).item())
                
    loss_source = np.array(loss_source)
    loss_target = np.array(loss_target)

    print('source domain loss{}+/-{}'.format(np.mean(loss_source), 1.96*np.std(loss_source,0)/np.sqrt(len(loss_source))))
    print('target domain loss{}+/-{}'.format(np.mean(loss_target), 1.96*np.std(loss_target,0)/np.sqrt(len(loss_target))))




if __name__ == '__main__':
    args = parse_args()
    if not args.test:
        set_seed(args.seed)
        print(args)
        model = run(args)
        dataloader_test = DataLoader(DomainData(args, test=True), batch_size=1, num_workers=args.num_workers)
        evaluate_test(args, model, dataloader_test)
    else:
        set_seed(args.seed)
        model = RecNet(args)
        model.load_state_dict(torch.load('model_save/parameter_29epoch.pkl'))
        model.to(args.device)
        dataloader_test = DataLoader(DomainData(args, test=True), batch_size=1, num_workers=args.num_workers)       
        evaluate_test(args, model, dataloader_test)


