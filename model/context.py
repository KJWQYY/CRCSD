import torch, vit_pytorch
import torch.nn as nn
from torch.autograd import Variable


from .base import Normalization, ClueFeat, FeedForward


class ClueFusion(nn.Module):
    def __init__(self, shot_dim, att_drop, seg_sz, topk, mode='self'):
        super(ClueFusion, self).__init__()
        self.cue_fea = ClueFeat(shot_dim, dropout=att_drop, seg_sz=seg_sz, topk=topk)

        self.relu = nn.ReLU()
        self.norm = Normalization(shot_dim, normalization='ln')

        self.ffc = FeedForward(shot_dim, int(shot_dim*1.5), drop=att_drop)
        self.ffc_norm = Normalization(shot_dim, normalization='ln')




    def forward(self, x, x_t, adj_sc, adj_ic, adj_sc_t, adj_sic_t,  sclue_1, iclue_1, sclue_1_t, iclue_1_t):

        _x = x
        _x_t = x_t

        x, x_t = self.cue_fea(x, x_t, adj_sc, adj_ic,  adj_sc_t, adj_sic_t,  sclue_1, iclue_1, sclue_1_t, iclue_1_t)#all

        x = self.relu(x)
        x = self.norm(_x + x)
        x_t = self.relu(x_t)
        x_t = self.norm(_x_t + x_t)

        x_a = x + 0.2 * x_t
        _x = x_a
        x_a = self.ffc(x_a)
        x_a = self.ffc_norm(_x + x_a)

        return x_a

    def _build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask > 0, float(0))
        return mask

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)



