import numpy as np
import torch
import os
import json as js
import pickle as pkl
from dataloader.BaseDataset import BaseDataset
import random
#from .BaseDataset import BaseDataset
from tqdm import tqdm

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


class MovienetDataset(BaseDataset):
    def __init__(self, samplelist:list, mode='train', topk=5):
        super(MovienetDataset, self).__init__(samplelist)
        self.mode = mode
        self.topk = topk

    def __getitem__(self, ind):
        data = self._read_pkl(self.samplist[ind])
        sample = data['data']
        label = data['label']
        # if self.mode == 'unsupervised':
        #     label = None
        # else:
        #     label = data['label']
        s_clue = data['self_clue']
        i_clue = data['inter_clue']

        if label != 1 and label !=0:
            label = 1
        sample_t = data['data_t']
        s_clue_t = data['self_clue_t']
        i_clue_t = data['inter_clue_t']

        sample = torch.from_numpy(sample)
        sample = sample.to(torch.float)

        sample_t = torch.from_numpy(sample_t)
        sample_t = sample_t.to(torch.float)

        if len(data['hop']) == 1:
            alink = np.array(data['hop'][0])
        else:
            alink = data['hop']
        alink_t = data['hop_t']
        hop = []
        hop_t = []



        #t
        #
        gh_sc = self._build_graph_sclue(s_clue, i_clue)
        gh_ic = self._build_graph_iclue(s_clue, i_clue)
        #t
        gh_sc_t = self._build_graph_sclue(s_clue_t, i_clue_t)
        gh_ic_t = self._build_graph_iclue(s_clue_t, i_clue_t)
        gh_sic_t = self._build_graph_siclue(s_clue_t, i_clue_t)

        sc_gh = self._clue_matric(s_clue, n_top=self.topk)
        ic_gh = self._clue_matric(i_clue, n_top=self.topk)
        #
        sc_gh_t = self._clue_matric(s_clue_t, n_top=self.topk)
        ic_gh_t = self._clue_matric(i_clue_t, n_top=self.topk)

        clue = []
        clue_t = []

        #ap
        #hop.append(gh[None])
        #hop_t.append(gh_t[None])

        # inxs.append(inx_gh[None])
        # inxs_t.append(inx_gh_t[None])

        #
        clue.append(sc_gh[None])
        clue.append(ic_gh[None])
        clue_t.append(sc_gh_t[None])
        clue_t.append(ic_gh_t[None])

        hop.append(gh_sc[None])
        hop.append(gh_ic[None])
        hop_t.append(gh_sc_t[None])
        hop_t.append(gh_ic_t[None])
        hop_t.append(gh_sic_t[None])

        hop = np.concatenate(hop, axis=0)
        hop_t = np.concatenate(hop_t, axis=0)

        # inxs = np.concatenate(inxs, axis=0)
        # inxs_t = np.concatenate(inxs_t, axis=0)

        #
        clue = np.concatenate(clue, axis=0)
        clue_t = np.concatenate(clue_t, axis=0)

        hop = torch.from_numpy(hop)
        hop_t = torch.from_numpy(hop_t)
        # inxs = torch.from_numpy(inxs).float()
        # inxs_t = torch.from_numpy(inxs_t).float()

        #
        clue = torch.from_numpy(clue).float()
        clue_t = torch.from_numpy(clue_t).float()

        label = torch.from_numpy(np.array(label))
        label = label.to(torch.float)
        if self.mode == 'train' or self.mode == 'unsupervised':
            return sample, sample_t, hop, hop_t, clue, clue_t, label
        else:
            return self.samplist[ind], sample, sample_t, hop, hop_t,  clue, clue_t, label


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


def sampleCtx(data:dict, labels:dict, center_id, seg_sz, dim=2048):
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



def worker_init_fn(worker_id):

    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def load_data(data_path, split_path, batch, mode='train', topk=5):
    with open(split_path, 'r') as f:
        data = js.load(f)
        trainSet = data['train'] + data['val']
        testSet = data['test']

    # testSet = read_pkl(split_path)
    # testlist = [data_path+'/'+name for name in testSet]

    samplelist = os.listdir(data_path)
    if mode == 'train':
        trainlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in trainSet]
        trainDataset = MovienetDataset(trainlist, mode=mode, topk=topk)
        dataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch,
                                                  shuffle=True, drop_last=True, num_workers=4, worker_init_fn=worker_init_fn)
        # dataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch,
        #                                           shuffle=True, drop_last=True, num_workers=4)
    if mode == 'test':
        testlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in testSet]
        testDataset = MovienetDataset(testlist, mode=mode, topk=topk)
        dataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch,
                                                 shuffle=False, drop_last=False, num_workers=4, worker_init_fn=worker_init_fn)

    return dataLoader
def load_pseudo_data(data_path, split_path, sample_path, batch, mode='train', topk=5):
    with open(split_path, 'r') as f:
        data = js.load(f)
        data_all = data['all']


    samplelist = os.listdir(data_path)

    trainSet = data_all

    trainlist = [data_path + '/' + sample for sample in samplelist if sample.split('_')[0] in trainSet]

    trainDataset = MovienetDataset(trainlist, mode=mode, topk=topk)
    dataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch,
                                             shuffle=True, drop_last=True, num_workers=4)

    return dataLoader

def load_data_abl(data_path, split_path, batch, mode='train', inx_abl=0):
    with open(split_path, 'r') as f:
        data = js.load(f)
        trainSet = data['train'] + data['val']
        testSet = data['test']

    samplelist = os.listdir(data_path)
    if mode == 'train':
        trainlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in trainSet]
        trainDataset = MovienetDataset_abl(trainlist, mode=mode, inx_abl=inx_abl)
        dataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch,
                                                  shuffle=True, drop_last=True, num_workers=1)
    if mode == 'test':
        testlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in testSet]
        testDataset = MovienetDataset_abl(testlist, mode=mode, inx_abl=inx_abl)
        dataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch,
                                                 shuffle=False, drop_last=False, num_workers=1)

    return dataLoader


def load_transfer(data_path, batch):
    samplelist = os.listdir(data_path)
    datalist = [data_path+'/'+sample for sample in samplelist]
    Dataset = MovienetDataset(datalist, mode='test')
    dataLoader = torch.utils.data.DataLoader(Dataset, batch_size=batch,
                                             shuffle=False, drop_last=False, num_workers=4)
    return dataLoader
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
def getclue(simat, win_size=2):
    T = len(simat)
    self_clue = [[0 for _ in range(T)] for _ in range(T)]

    for i in range(T):
        topk = simat[i]
        if i - win_size < 0:
            i_ser = [0, win_size*2+1]
        elif i+win_size >= T:
            i_ser = [T-2*win_size-1, T]
        else:
            i_ser = [i-win_size, i+win_size+1]

        intersection = np.intersect1d(topk, list(range(i_ser[0], i_ser[1])))
        if intersection is not None:

            for j in intersection:
                self_clue[i][j] = 1
        self_clue[i][i] = 1
    W = len(simat[0])
    self_clue_indeices = []
    for i in self_clue:
        indices = [index for index, value in enumerate(i) if value == 1]
        if len(indices) <2:
            indices = [-1 for _ in range(W)]
        elif 2 <= len(indices) <W:
            padnum = W - len(indices)
            for j in range(padnum):
                indices.append(-1)
        self_clue_indeices.append(indices)

    return self_clue_indeices
def getinterclue(simat, self_clue, win_size=2):
    T = len(simat)
    interclue_temp = [[0 for _ in range(T)] for _ in range(T)]
    for i in range(T):
        topk = simat[i]
        if i - win_size < 0:
            i_ser = [0, win_size*2+1]
        elif i+win_size >= T:
            i_ser = [T-2*win_size-1, T]
        else:
            i_ser = [i-win_size, i+win_size+1]
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
        interclue_indeices.append(indices)
    return interclue_indeices

def gen_clue(lb_path, gph_path, win_size=2, seg_sz=20, dim=2048, save_path=None):
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
            #
            shot_path = save_path + '/{}_shot{}.pkl'.format(name, c_id)
            hop_link = read_pkl(gph_path+ '/{}_shot{}.pkl'.format(name, c_id))['hop']
            self_clue = getclue(hop_link[0], win_size)
            inter_clue = getinterclue(hop_link[0], self_clue, win_size)
            #temp = 10
            sample = {'data': ctx, 'label': c_label, 'hop': hop_link, 'self_clue': self_clue, 'inter_clue': inter_clue}
            write_pkl(shot_path, sample)

if __name__=='__main__':

    ft_path = '/media/sda1/data/movieNet/ImageNet_shot.pkl'
    lb_path = '/media/sda1/data/movieNet/scene318/label318'
    gph_path = '/media/sda1/third/gph_path'
    path = '/media/sda1/third/path'
    win_size = 2
    dataset = gen_clue(lb_path, gph_path, win_size, seg_sz=20, dim=2048, save_path=path)




