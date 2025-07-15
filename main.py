import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from dataloader.supervise_movienet import load_data
from model.CRCSD import ClueNet
from loss import bce, sigmoid_focal
from warm_up import warmup_decay_cosine
from metric import metric
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
torch.cuda.set_device(0)
import logging
import sys

import os
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#from newsclue_ss_ii_ss_t_si_t
def train_epoch(
        trainload,
        model,
        opti,
        lr_sh,
        gpu=0
):
    model.train()
    progress = tqdm(trainload)
    for i, sample in enumerate(progress):
        #data, graphs, inxs, clue, label = sample[0], sample[1], sample[2], sample[3], sample[4]
        data, data_t, graphs, graphs_t,  clue, clue_t, label = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6]

        data = data.cuda(gpu)
        data_t = data_t.cuda(gpu)

        hop_gh = trans_graph(graphs, gpu)
        hop_gh_t = trans_graph(graphs_t, gpu)


        clue = clue.cuda(gpu)
        clue_t = clue_t.cuda(gpu)

        label = label.cuda(gpu)

        pred = model(data, data_t, hop_gh, hop_gh_t, clue, clue_t)
        loss = bce(pred, label)

        opti.zero_grad()
        loss.backward()
        opti.step()

        lr_sh.step()
        progress.set_postfix(loss=f'{loss.item():.8f}')

    return 1


def test_epoch(
        testload,
        model,
        need,
        m,
        gpu=0,
):
    predlist = []
    labelist = []
    pathlist = []
    scos = 0
    dcos = 0
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate((tqdm(testload))):
           paths, data, data_t, graphs, graphs_t,  clue, clue_t, label = sample[0], sample[1], sample[2], sample[3], \
           sample[4], sample[5], sample[6], sample[7]

           data = data.cuda(gpu)
           data_t = data_t.cuda(gpu)

           hop_gh = trans_graph(graphs, gpu)
           hop_gh_t = trans_graph(graphs_t, gpu)

           clue = clue.cuda(gpu)
           clue_t = clue_t.cuda(gpu)

           pred = model(data, data_t, hop_gh, hop_gh_t,  clue, clue_t)

           predlist.append(pred.data.cpu().numpy())
           labelist.append(label.data.cpu().numpy())
           pathlist.append(paths)
    met, moviePL = metric(pathlist, predlist, labelist, m,needs=need)
    return met, moviePL

def main(
        sample_path,
        split_path=None,
        r = 0.5,
        m = 0.9,
        seg_sz = 12,
        batch=64,
        epoch=10,
        gpu=0,
        model_path=None,
        save_path=None,
):
    trainload = load_data(sample_path, split_path, batch, topk=5)
    testload = load_data(sample_path, split_path, 512, mode='test', topk=5)

    model = ClueNet(2048, t_dim=768, embed_dim=1024, att_drop=0.1, topk=5, seg_sz=seg_sz, mode='fine')
    re_inin = 'detect'
    if model_path is not None:
        pretrain = torch.load(model_path, map_location='cpu')['state_dict']
        # new_weight = {}
        # for key in pretrain.keys():
        #     if re_inin not in key.split('.')[:2]:
        #         new_weight.update({key:pretrain[key]})
        # model.load_state_dict(new_weight, strict=False)
        model.load_state_dict(pretrain)
    model.cuda(gpu)

    if model_path is None:
        for para in model.parameters():
            para.requires_grad = True
    else:
        for name, para in model.named_parameters():
            if name.split('.')[0] in re_inin:
                para.requires_grad = True
            else:
                para.requires_grad = False

    max_miou = 0
    max_map = 0
    opti = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
    iter_num = len(trainload)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opti,
        warmup_decay_cosine(iter_num, iter_num * (epoch - 1))
    )

    for i in range(epoch):
        train_epoch(trainload, model, opti, lr_scheduler, gpu)
        met, moviePL = test_epoch(testload, model, ['map', 'miou', 'f1'], m,gpu=gpu)

        # save model
        if save_path is not None:
            if max_miou < met['mIoU'] or max_map < met['mAP']:
                max_miou = met['mIoU']
                max_map = met['mAP']
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'miou': max_miou,
                    'map': max_map,
                    'f1':met['F1'],
                    'optim': opti.state_dict()
                },
                    save_path + '/epoch_{}.pth.tar'.format(i+1)
                )

        print('{} epoch: mAP:{:.5f}, mIoU:{:.3f}'.format(i+1, met['mAP'], met['mIoU']))
        print('{} epoch: F1:{:.3f}'.format(i + 1, met['F1']))

    return 1



def adjust_lr(optimizer):
    for param in optimizer.param_groups:
        param['lr'] *= 0.1
    return 1


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def write_pkl(path: str, data: dict):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    return 1


# ------------- Utilize -------------  #
def trans_graph(graph, gpu):
    graph = graph.to(torch.float)
    graph = graph.cuda(gpu)
    graph = Variable(graph, requires_grad=False)
    return graph


if __name__=='__main__':
    random.seed(16)
    np.random.seed(32)
    torch.manual_seed(64)
    torch.cuda.manual_seed(128)
    torch.cuda.manual_seed_all(128)  # 多GPU时使用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    data_path = 'G:/third/权重和工具/gendata_all'
    split_path = 'G:/data/movieNet/scene318/meta/split318.json'
    save_path = 'G:/data/movieNet/scene318/meta'
    model_path = None
    r = 0.5
    seg_sz = 20
    m = 0.9
    _ = main(data_path, split_path, r, m, seg_sz = seg_sz, batch=512, epoch=10, save_path=save_path, model_path=None)

