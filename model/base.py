import torch, einops
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable


class SelfAttention(nn.Module):
    def __init__(self, in_ch, dropout=0.1, is_scale=True):
        super(SelfAttention, self).__init__()

        out_ch = in_ch
        self.q = nn.Linear(in_ch, out_ch, bias=False)
        self.k = nn.Linear(in_ch, out_ch, bias=False)
        self.v = nn.Linear(in_ch, out_ch, bias=False)

        self.sqrt_dim = np.sqrt(out_ch)
        self.is_scale = is_scale

        self.final_drop = nn.Dropout(dropout)

        self.initialize_weight(self.q)
        self.initialize_weight(self.k)
        self.initialize_weight(self.v)

    def forward(self, ft_q, ft_k, ft_v, mask=None):
        """
        :param ft_q: (B, T, dim)
        :param ft_k: same
        :param ft_v: same
        :param mask: (B, T, T)
        :param reweight: (B, T, T)
        :return:
        """
        batch, T, _ = ft_k.shape
        q = self.q(ft_q)
        k = self.k(ft_k)
        v = self.v(ft_v)

        if self.is_scale:
            sim = torch.matmul(q, k.transpose(2, 1)) * self.sqrt_dim
        else:
            sim = torch.matmul(q, k.transpose(2, 1))
        if mask is not None:
            sim = sim + mask
        sim = F.softmax(sim, dim=-1)
        v = torch.matmul(sim, v)

        v = self.final_drop(v)

        return v
    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)



class ClueFeat(nn.Module):
    def __init__(self, in_ch, seg_sz=20, topk=4, dropout=0.1):
        super().__init__()
        self.seg_sz = seg_sz
        self.topk = topk

        out_ch = in_ch
        self.scale = out_ch ** (-0.5)

        self.q = nn.Linear(in_ch, out_ch, bias=False)
        self.k = nn.Linear(in_ch, out_ch, bias=False)
        self.v = nn.Linear(in_ch, out_ch, bias=False)

        #self.qs = nn.Linear(in_ch, out_ch, bias=False)

        self.ss = nn.Linear(self.topk, 1, bias=False)
        self.ii = nn.Linear(self.topk, 1, bias=False)

        self.initialize_weight(self.q)
        self.initialize_weight(self.k)
        self.initialize_weight(self.v)

        #self.initialize_weight(self.qs)
        self.initialize_weight(self.ss)
        self.initialize_weight(self.ii)

        nn.utils.weight_norm(self.ss, name='weight')
        nn.utils.weight_norm(self.ii, name='weight')
        #t
        self.q_t = nn.Linear(in_ch, out_ch, bias=False)
        self.k_t = nn.Linear(in_ch, out_ch, bias=False)
        self.v_t = nn.Linear(in_ch, out_ch, bias=False)

        self.ss_t = nn.Linear(self.topk, 1, bias=False)
        self.si_t = nn.Linear(self.topk, 1, bias=False)

        self.initialize_weight(self.q_t)
        self.initialize_weight(self.k_t)
        self.initialize_weight(self.v_t)

        self.initialize_weight(self.ss_t)
        self.initialize_weight(self.si_t)

        nn.utils.weight_norm(self.ss_t, name='weight')
        nn.utils.weight_norm(self.si_t, name='weight')

        self.maxpool = nn.AdaptiveMaxPool2d((self.topk, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((self.topk, 1))

        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        #nn.utils.weight_norm(self.ii_t, name='weight')
    def forward(self, x, x_t,  adj_sc, adj_ec,  adj_sc_t, adj_sec_t, sclue_1, eclue_1, sclue_1_t, eclue_1_t):

        """
        :param x: (B, T, dim)
        :param adj: (B, T, T)
        :param inxs: (B, T, nei, T)
        :return:
        """
        B, T = x.shape[0], x.shape[1]
        #mask = self.build_mask(adj) #测试selfclue
        mask = self.build_mask(adj_sc)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        #------------------build_mask----------------
        ssmask = self.build_mask_clue(adj_sc)
        iimask = self.build_mask_clue(adj_ec)
        #-----------------ss------------------
        qn = torch.einsum('bxnt,bxtc -> bxnc', (sclue_1, q.unsqueeze(dim=1)))
        sim_ss = torch.einsum('biuxd, bujyd -> bijxy', (qn.unsqueeze(dim=2), qn.unsqueeze(dim=1)))
        sim_ss = sim_ss.reshape(B, -1, sim_ss.shape[-2], sim_ss.shape[-1])
        sim_ss = self.avgpool(sim_ss).squeeze()
        sim_ss = self.ss(sim_ss).squeeze()
        sim_ss = sim_ss.reshape(B, T, T)
        sim_ss = sim_ss * ssmask

        #---------------ec-------------------
        ecn = torch.einsum('bxnt,bxtc -> bxnc', (eclue_1, k.unsqueeze(dim=1)))
        sim_ee = torch.einsum('biuxd, bujyd -> bijxy', (ecn.unsqueeze(dim=2), ecn.unsqueeze(dim=1)))
        sim_ee = sim_ee.reshape(B, -1, sim_ee.shape[-2], sim_ee.shape[-1])
        sim_ee = self.avgpool(sim_ee).squeeze()
        sim_ee = self.ii(sim_ee).squeeze()
        sim_ee = sim_ee.reshape(B, T, T)
        sim_ee = sim_ee * iimask
        #t--------------------------------------------
        B_t, T_t = x_t.shape[0], x_t.shape[1]
        mask_t = self.build_mask(adj_sec_t)
        q_t = self.q_t(x_t)
        k_t = self.k_t(x_t)
        v_t = self.v_t(x_t)

        q_t = F.normalize(q_t, dim=-1)
        k_t = F.normalize(k_t, dim=-1)
        ssmask_t = self.build_mask_clue(adj_sc_t)
        simask_t = self.build_mask_clue(adj_sec_t)
        qn_t = torch.einsum('bxnt,bxtc -> bxnc', (sclue_1_t, q_t.unsqueeze(dim=1)))
        sim_ss_t = torch.einsum('biuxd, bujyd -> bijxy', (qn_t.unsqueeze(dim=2), qn_t.unsqueeze(dim=1)))
        sim_ss_t = sim_ss_t.reshape(B_t, -1, sim_ss_t.shape[-2], sim_ss_t.shape[-1])
        sim_ss_t = self.avgpool(sim_ss_t).squeeze()
        sim_ss_t = self.ss_t(sim_ss_t).squeeze()
        sim_ss_t = sim_ss_t.reshape(B_t, T_t, T_t)
        sim_ss_t = sim_ss_t * ssmask_t


        scn_t = torch.einsum('bxnt,bxtc -> bxnc', (sclue_1_t, k_t.unsqueeze(dim=1)))
        ecn_t = torch.einsum('bxnt,bxtc -> bxnc', (eclue_1_t, k_t.unsqueeze(dim=1)))
        sim_se_t = torch.einsum('biuxd, bujyd -> bijxy', (scn_t.unsqueeze(dim=2), ecn_t.unsqueeze(dim=1)))
        sim_se_t = sim_se_t.reshape(B_t, -1, sim_se_t.shape[-2], sim_se_t.shape[-1])
        sim_se_t = self.avgpool(sim_se_t).squeeze()

        sim_se_t = self.si_t(sim_se_t).squeeze()
        sim_se_t = sim_se_t.reshape(B_t, T_t, T_t)
        sim_se_t = sim_se_t * simask_t


        #mask
        sim_ = sim_ss + sim_ee
        sim_t = sim_ss_t + sim_se_t


        rmask1 = self.build_mask_r1(sim_)
        rmask2 = self.build_mask_r2(sim_t)
        sim_mask = rmask1 * rmask2 * self.sigmoid(sim_)

        rmask1_t = self.build_mask_r1(sim_t)
        rmask2_t = self.build_mask_r2(sim_)
        sim_t_mask = rmask1_t * rmask2_t * self.sigmoid(sim_t)


        sim = self.softmax(sim_ss + sim_ee + sim_t_mask + mask)#all

        sim_t = self.softmax(sim_ss_t + sim_se_t + sim_mask + mask_t)#all
        #sim_t = self.softmax(sim_ss_t + mask_t)




        v = torch.einsum('bkt, btd->bkd', (sim, v))
        v_t = torch.einsum('bkt, btd->bkd', (sim_t, v))

        v = self.drop(v)
        v_t = self.drop(v_t)

        return v, v_t


    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(1e22)).masked_fill_(mask>0, float(0))
        return mask
    def build_mask_clue(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(0)).masked_fill_(mask>0, float(1))
        return mask
    def build_mask_r1(self, m_t):
        mask = Variable(m_t.clone(), requires_grad=False)
        mask = mask.masked_fill_((-0.9 <= mask) & (mask <= 0.9), float(0)).masked_fill_(mask>0.9, float(1)).masked_fill_(mask<-0.9, float(1))
        return mask

    def build_mask_r2(self, m_t):
        mask = Variable(m_t.clone(), requires_grad=False)
        mask = mask.masked_fill_((-0.1 <= mask) & (mask <= 0.1) & (mask != 0.0), float(1)).masked_fill_((0.1 <= mask) & (mask != 1.0),float(0)).masked_fill_((-0.1 > mask) & (mask != 1.0), float(0))
        return mask

class Normalization(nn.Module):

    def __init__(self, embed_dim, timesz=20, k=3, normalization='batch'):
        super(Normalization, self).__init__()
        if normalization == 'batch':
            self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)
        else:
            self.normalizer = nn.LayerNorm(embed_dim, eps=1e-6)

        self.type = normalization
        self.T = timesz

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.type == 'batch':
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        else:
            return self.normalizer(input)





class SineActivation(nn.Module):
    def __init__(self, in_features=1, out_features=256, n_shot=20):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

        self.pos_ids = torch.arange(n_shot, dtype=torch.float, device='cuda:0')[:, None]

    def forward(self, B):
        """
        :return:
        """
        v1 = self.f(torch.matmul(self.pos_ids, self.w) + self.b)
        v2 = torch.matmul(self.pos_ids, self.w0) + self.b0
        v = torch.cat([v1, v2], -1)

        v = einops.repeat(v, 't n -> b t n', b=B)
        return v


class FeedForward(nn.Module):
    def __init__(self, in_ch, hid_ch, drop=0.5):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(in_ch, hid_ch)
        self.linear_2 = nn.Linear(hid_ch, in_ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.linear_2(x)

        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class RelativePosition(nn.Module):
    def __init__(self, scale, seg_sz=20):
        super().__init__()
        self.scale = scale
        self.seg_sz = seg_sz
        self.w = nn.parameter.Parameter(0.5*torch.ones(seg_sz, seg_sz))
        self.b = nn.parameter.Parameter(torch.randn(seg_sz, 1))
        q_idx = torch.arange(self.seg_sz, dtype=torch.long, device="cuda:0")
        self.rel_pos = (q_idx[None] - q_idx[:, None])**2

        # nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.b)

    def forward(self):
        rel_pos = torch.exp(-self.rel_pos * self.w + self.b)

        return rel_pos



class Embedding(nn.Module):
    def __init__(self, in_ch, in_ch_t, out_ch, drop=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, out_ch)
        self.linear_2 = nn.Linear(out_ch, out_ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)
        #t
        self.linear_1_t = nn.Linear(in_ch_t, out_ch)
        self.linear_2_t = nn.Linear(out_ch, out_ch)
        self.drop_t = nn.Dropout(drop)
        self.relu_t = nn.ReLU()
        self.initialize_weight(self.linear_1_t)
        self.initialize_weight(self.linear_2_t)
    def forward(self, x, x_t):
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.linear_2(x)

        x_t = self.relu_t(self.linear_1_t(x_t))
        x_t = self.drop_t(x_t)
        x_t = self.linear_2_t(x_t)
        return x, x_t

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class CAttention(nn.Module):
    def __init__(self, input_dim):
        super(CAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # Linear transformations
        Q = self.query(x)  # (512, 20, 2048)
        K = self.key(x)    # (512, 20, 2048)
        V = self.value(x)

        # Scale dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.shape[-1])  # (512, 20, 20)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (512, 20, 20)
        #attention_weights = self.dropout(attention_weights)

        # Weighted sum
        attention_output = torch.matmul(attention_weights, V)  # (512, 20, 2048)
        return attention_output


