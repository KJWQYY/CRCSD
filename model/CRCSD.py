import torch
import torch.nn as nn
import numpy as np
from tslearn import metrics

from .base import SineActivation, Embedding
from .context import ClueFusion
from .detector import LatentDetector
from loss import bce, sigmoid_focal, pseudo_bce


class ClueNet(nn.Module):
    def __init__(self, shot_dim=1920, t_dim=768, embed_dim=1920, att_drop=0.1, seg_sz=20, topk=4, mode='pretrain'):
        super().__init__()
        self.embed_pos = SineActivation(n_shot=seg_sz, out_features=256)
        self.proj = Embedding(shot_dim, t_dim, embed_dim)
        shot_dim = embed_dim+256
        self.cf = ClueFusion(shot_dim, att_drop, topk=topk, seg_sz=seg_sz, mode='self')
        self.relu = nn.ReLU()
        self.detect = LatentDetector(seg_sz)

        self.n_sparse = 2
        self.n_dense = 18
        self.num_neg_sample = 1


    def forward(self, x, x_t, adj, adj_t, clue, clue_t):

        x, x_t = self.proj(x, x_t)

        pos = self.embed_pos(x.shape[0])
        x = torch.concat((x, pos), dim=-1)

        pos_t = self.embed_pos(x_t.shape[0])
        x_t = torch.concat((x_t, pos_t), dim=-1)

        ctx = self.cf(x, x_t, adj[:, 0], adj[:, 1],  adj_t[:, 0], adj_t[:, 2], clue[:, 0], clue[:, 1], clue_t[:, 0], clue_t[:, 1])
        out = self.detect(ctx)
        return out

    @torch.no_grad()
    def _compute_dtw_path(self, s_emb, d_emb):
        """ compute alignment between two sequences using DTW """
        cost = (
            (1 - torch.bmm(s_emb, d_emb.transpose(1, 2)))
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # shape: [b n_sparse n_dense]
        dtw_path = []
        for bi in range(cost.shape[0]):
            _path, _ = metrics.dtw_path_from_metric(cost[bi], metric="precomputed")
            dtw_path.append(np.asarray(_path))  # [n_dense 2]

        return dtw_path
    def _compute_boundary(self, dtw_path, nshot):
        """ get indices of boundary shots
        return:
            bd_idx: list of size B each of which means index of boundary shot
        """
        # dtw_path: list of B * [ndense 2]
        # find boundary location where the last index of first group (0)
        np_path = np.asarray(dtw_path)
        bd_idx = [np.where(path[:, 0] == 0)[0][-1] for path in np_path]

        return bd_idx
    def _compute_pp_loss(self, crn_repr_wo_mask, bd_idx):
        """ compute pseudo-boundary prediction loss """
        # bd_idx: list of B elements
        B, nshot, _ = crn_repr_wo_mask.shape  # nshot == ndense

        # sample non-boundary shots
        nobd_idx = []
        for bi in range(B):
            cand = np.delete(np.arange(nshot), bd_idx[bi])
            nobd_idx.append(
                np.random.choice(cand, size=self.num_neg_sample, replace=False)
            )
        nobd_idx = np.asarray(nobd_idx)

        # get representations of boundary and non-boundary shots
        # shape of shot_repr: [B*(num_neg_sample+1) D]
        # where first B elements correspond to boundary shots
        b_idx = torch.arange(0, B, device=crn_repr_wo_mask.device)
        bd_shot_repr = crn_repr_wo_mask[b_idx, bd_idx]  # [B D]
        nobd_shot_repr = [
            crn_repr_wo_mask[b_idx, nobd_idx[:, ni]]
            for ni in range(self.num_neg_sample)
        ]  # [B num_neg_sample D]
        #dele 0

        shot_repr = torch.cat([bd_shot_repr, torch.cat(nobd_shot_repr, dim=0)], dim=0)

        # compute boundaryness loss
        # bd_pred = self.head_pp(shot_repr)  # [B*(num_neg_sample+1) D]
        bd_pred = self.detect(shot_repr)

        # bd_label = torch.ones(
        #     (bd_pred.shape[0]), dtype=torch.long, device=crn_repr_wo_mask.device
        # )

        bd_label = torch.ones(
            (bd_pred.shape[0]), dtype=torch.float32, device=crn_repr_wo_mask.device
        )

        bd_label[B:] = 0

        pp_loss = pseudo_bce(bd_pred, bd_label)

        return pp_loss
