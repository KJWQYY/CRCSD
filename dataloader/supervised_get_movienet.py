import numpy as np
import torch
import os
import json as js
import pickle as pkl
from tqdm import tqdm

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def sampleCtx_t(data:dict, labels:dict, center_id, seg_sz, dim=768):

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
        ctx[i] = data[f'{shot_id:04d}'][None]
        if shot_id in labels.keys():
            ctx_lab.append(labels[shot_id])
        else:
            ctx_lab.append(0)

    if dim != 768:
        ctx = np.reshape(ctx, (seg_sz, dim))
    ctx_lab = np.stack(ctx_lab)
    return ctx, ctx_lab[seg_sz//2-1]

def sampleCtx(data:dict, labels:dict, center_id, seg_sz, dim=2048):

    max_id = len(data.keys())
    half = seg_sz//2
    ctx_id = np.arange(center_id-half+1, center_id+half+1)
    ctx_id = np.clip(ctx_id, 0, max_id-1)

    if dim ==2048:
        ctx = np.zeros((seg_sz, dim))
    else:
        ctx = np.zeros((seg_sz, 5, dim//5))
    ctx_lab = []

    for i, shot_id in enumerate(ctx_id):
        ctx[i] = data[f'{shot_id:04d}'][None]
        if shot_id in labels.keys():
            ctx_lab.append(labels[shot_id])
        else:
            ctx_lab.append(0)

    if dim != 2048:
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
def getclue(simat, win_size=2, k=5):
    T = len(simat)
    self_clue = []

    for i in range(T):
        topk = simat[i]
        if i - win_size < 0:
            i_ser = [0, win_size*2]
        elif i+win_size >= T:
            i_ser = [T-2*win_size, T]
        else:
            i_ser = [i-win_size, i+win_size]
        intersection = []
        for ele in topk:
            if (ele >= i_ser[0]) & (ele <= i_ser[1]):
                intersection.append(ele)
        if len(intersection)<2:
            intersection = [-1 for _ in range(win_size)]
        elif 2 <= len(intersection) < k:
            padnum = k - len(intersection)
            for j in range(padnum):
                intersection.append(-1)
        else:
            intersection = intersection[0:k]

        self_clue.append(intersection)
    return self_clue
def getinterclue(simat, self_clue, win_size=2):
    T = len(simat)
    interclue_temp = [[0 for _ in range(T)] for _ in range(T)]
    for i in range(T):
        topk = simat[i]
        if i - win_size < 0:
            i_ser = [0, win_size*2]
        elif i+win_size >= T:
            i_ser = [T-2*win_size, T]
        else:
            i_ser = [i-win_size, i+win_size]
        for j in list(range(i_ser[0], i_ser[1])):
            interclue_temp[i][j] = 1

        for j in range(len(self_clue[i])):
            if self_clue[i][j] != -1:
                interclue_temp[i][self_clue[i][j]] = 0



    interclue = [[0 for _ in range(T)] for _ in range(T)]
    for i in range(T):
        indices = [index for index, value in enumerate(interclue_temp[i]) if value == 1]
        max_inter = 0
        clue_i = []
        for j in range(T):
            intersection = np.intersect1d(self_clue[j], indices)
            inter_num = len(intersection)
            if inter_num > max_inter:
                max_inter = inter_num
                clue_i = intersection
        for j in clue_i:
            interclue[i][j] = 1
    W = len(simat[0])
    interclue_indeices = []
    for i in interclue:
        indices = [index for index, value in enumerate(i) if value == 1]
        if len(indices) < 2:
            indices = [-1 for _ in range(W)]
        elif 2 <= len(indices) < W:
            padnum = W - len(indices)
            for j in range(padnum):
                indices.append(-1)
        else:
            indices = indices[0:W]
        interclue_indeices.append(indices)
    return interclue_indeices

def gen_clue(ft_path, lb_path, gph_path, win_size=2, seg_sz=20, dim=2048, save_path=None):
    mnames = gen_labelName(lb_path)
    feats = read_pkl(ft_path)
    for name in tqdm(mnames):
        if name not in feats.keys():
            continue
        feat_m = feats[name]
        label_m = read_label(lb_path + '/' + name + '.txt')
        n_shot = len(label_m.keys())
        for c_id in range(n_shot):
            ctx, c_label  = sampleCtx(feat_m, label_m, c_id, seg_sz, dim=dim)
            #
            shot_path = save_path + '/{}_shot{}.pkl'.format(name, c_id)
            hop_link= read_pkl(gph_path+ '/{}_shot{}.pkl'.format(name, c_id))['hop']
            self_clue = getclue(hop_link[0], win_size)
            inter_clue = getinterclue(hop_link[0], self_clue, win_size)
            #temp = 10
            sample = {'data': ctx, 'label': c_label, 'hop': hop_link, 'self_clue': self_clue, 'inter_clue': inter_clue}
            write_pkl(shot_path, sample)
def gen_clue_all(ft_path, ft_path_t, lb_path, gph_path, gph_path_t, win_size=2, k=5,  seg_sz=20, dim=2048, dim_t=768, save_path=None):
    best_config = {
        'k' : 5 ,
        'seg_sz' : 20 ,
        'win_size' : 5
    }
    mnames = gen_labelName(lb_path)
    feats = read_pkl(ft_path)
    feats_t = read_pkl(ft_path_t)
    for name in tqdm(mnames):
        if name not in feats.keys():
            continue
        feat_m = feats[name]
        feat_m_t = feats_t[name]
        label_m = read_label(lb_path + '/' + name + '.txt')
        n_shot = len(label_m.keys())
        for c_id in range(n_shot):
            ctx, c_label = sampleCtx(feat_m, label_m, c_id, seg_sz, dim=dim)
            ctx_t, c_label_t = sampleCtx_t(feat_m_t, label_m, c_id, seg_sz, dim=dim_t)
            shot_path = save_path + '/{}_shot{}.pkl'.format(name, c_id)
            # if os.path.exists(shot_path):
            #     continue
            hop_link = read_pkl(gph_path + '/{}_shot{}.pkl'.format(name, c_id))['hop']
            hop_link_t = read_pkl(gph_path_t + '/{}_shot{}.pkl'.format(name, c_id))['hop']

            if k == best_config['k'] and seg_sz == best_config['seg_sz'] and win_size == best_config['win_size']:
                _pkl = 'G:/third/tool/sample_20' + '/{}_shot{}.pkl'.format(name, c_id)
                self_clue = read_pkl(_pkl)['hop'][0]
            else:
                self_clue = getclue(hop_link, win_size, k)
            inter_clue = getinterclue(hop_link, self_clue, win_size)

            if k == best_config['k'] and seg_sz == best_config['seg_sz'] and win_size == best_config['win_size']:
                _pkl = 'G:/third/tool/sample_20' + '/{}_shot{}.pkl'.format(name, c_id)
                self_clue_t = read_pkl(_pkl)['hop']
            else:
                self_clue_t = getclue(hop_link_t, win_size, k)
            inter_clue_t = getinterclue(hop_link_t, self_clue_t, win_size)
            sample = {'data': ctx, 'data_t': ctx_t, 'label': c_label, 'hop': hop_link, 'hop_t': hop_link_t, 'self_clue': self_clue, 'inter_clue': inter_clue, 'self_clue_t': self_clue_t, 'inter_clue_t': inter_clue_t}
            write_pkl(shot_path, sample)


if __name__=='__main__':

    ft_path = 'G:/data/movieNet/ImageNet_shot.pkl'
    ft_path_t = 'G:/data/movieNet/Gemma3_Bert_shot.pkl'
    lb_path = 'G:/data/movieNet/scene318/label318'
    gph_path = 'G:/third/tool/N/N_20'
    gph_path_t = 'G:/third/tool/N/N_20_t_gemma3'
    path = 'G:/third/tool/path'
    seg_sz = 20
    win_size = 5
    k = 5
    dataset = gen_clue_all(ft_path, ft_path_t, lb_path, gph_path, gph_path_t, win_size, k=k, seg_sz=seg_sz, dim=2048, dim_t = 768, save_path=path)




