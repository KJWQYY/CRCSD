import numpy as np
import torch
import os
import json as js
import pickle as pkl
#需要完成文本的相似检测

#from .BaseDataset import BaseDataset
from tqdm import tqdm

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data





def gen_dataSet(ft_path, lb_path, gph_path, seg_sz=20, dim=2048, save_path=None):
    feats = read_pkl(ft_path)
    mnames = gen_labelName(lb_path)

    for name in tqdm(mnames):
        if name not in feats.keys():
            continue
        feat_m = feats[name]
        label_m = read_label(lb_path + '/' + name + '.txt')
        n_shot = len(label_m.keys())
        for c_id in range(n_shot):
            ctx, c_label  = sampleCtx(feat_m, label_m, c_id, seg_sz, dim=dim)

            shot_path = save_path + '/{}_shot{}.pkl'.format(name, c_id)
            hop_link= read_pkl(gph_path+ '/{}_shot{}.pkl'.format(name, c_id))['hop']
            sample = {'data': ctx, 'label': c_label, 'hop': hop_link}
            write_pkl(shot_path, sample)



    return 1


def sampleCtx(data:dict, labels:dict, center_id, seg_sz, dim=768):
    """
    Collecting shots centred in centre_id in the window at scale of seg_sz
    :param data:
    :param pairs:
    :param labels
    :param center_id:
    :param seg_sz: is a even number
    :return:
        ctx: (seg_sz, 2048)
        ctx_pair: (seg_sz, 5), second dim is shot index in a movie
        ctx_lab: (seg_sz,), same as above
        ctx_id: a list, where each elem. is shot's index in a movie
    """
    max_id = len(data.keys())
    half = seg_sz//2
    ctx_id = np.arange(center_id-half+1, center_id+half+1)
    ctx_id = np.clip(ctx_id, 0, max_id-1)

    if dim ==768:
        ctx = np.zeros((seg_sz, dim))
    else:
        ctx = np.zeros((seg_sz, 5, dim//5))
    ctx_lab = []

    for i, shot_id in enumerate(ctx_id):
        # if f'{shot_id:04d}' in data:
        #     ctx[i] = data[f'{shot_id:04d}'][None]
        ctx[i] = data[f'{shot_id:04d}'][None]
        if shot_id in labels.keys():
            ctx_lab.append(labels[shot_id])
        else:
            ctx_lab.append(0)

    if dim != 768:
        ctx = np.reshape(ctx, (seg_sz, dim))
    ctx_lab = np.stack(ctx_lab)
    return ctx, ctx_lab[seg_sz//2-1]


def read_label(path):
    """
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        labelDict = {}
        while 1:
            line = f.readline()
            if not line:
                break
            line = line.split('\n')[0]
            shot_id, label = line.split(' ')
            labelDict.update({int(shot_id): int(label)})
    return labelDict

def gen_labelName(path):
    files = os.listdir(path)
    names = [file.split('.')[0] for file in files]

    return names

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def write_pkl(path: str, data: dict):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    return 1


def topKNN(feat:np.ndarray, win_size=5, top=5):
    # feat: (T, dim), T is 20, dim is feature dimension
    #  win_size: int, length L in eq.(1)
    #  top: top-k
    # simat: (T, top)
    T = feat.shape[0]
    temp_mask = np.zeros((T, T))

    for i in range(T):
        if i-win_size < 0:
            i_ser = [0, win_size*2]
        elif i+win_size >= T:
            i_ser = [T-2*win_size, T]
        else:
            i_ser = [i-win_size, i+win_size+1]
        temp_mask[i, i_ser[0]:i_ser[1]] = 1

    simat = similarity(feat, feat, temp_mask)
    simat = np.argsort(simat, axis=-1)[:, -1:-(top+1):-1]
    return simat
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
def similarity(feat1, feat2, tmask=None):
    feat1 = normalized(feat1)
    feat2 = normalized(feat2)
    mat = np.matmul(feat1, feat2.transpose(1, 0))
    mask = np.eye(feat1.shape[0])
    mat = mat - mask
    if tmask is not None:
        mat = mat * tmask

    return mat

def get_sim_path(lb_path, ft_path, l, seg_sz=20, dim=2048, save_path=None):
    mnames = gen_labelName(lb_path)
    feats = read_pkl(ft_path)
    for name in tqdm(mnames):
        if name not in feats.keys():
            continue
        feat_m = feats[name]
        label_m = read_label(lb_path + '/' + name + '.txt')
        n_shot = len(label_m.keys())
        for c_id in range(n_shot):
            ctx, c_label = sampleCtx(feat_m, label_m, c_id, seg_sz, dim=dim)
            sim = topKNN(ctx, l, l*2)
            shot_path = save_path + '/{}_shot{}.pkl'.format(name, c_id)
            sample = {'hop': sim}
            write_pkl(shot_path, sample)

if __name__=='__main__':


    ft_path = 'G:/data/movieNet/Gemma3_shot.pkl'
    lb_path = 'G:/data/movieNet/scene318/label318'
    path = 'G:/third/tool/N/N_20_t_gemma3'
    seg_sz = 20
    l = 10
    get_sim_path(lb_path, ft_path,l, seg_sz=seg_sz, dim=768, save_path=path)






