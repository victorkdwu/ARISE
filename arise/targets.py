"""Thin target-model adapters.

Each adapter exposes what ARISE needs from a target NIDS, computed exactly as
the target computes it at test time:

  prep(X_raw)            raw feature matrix -> model input tensor (CPU)
  new_model(T_train)     fresh, untrained model (KitNET fits its feature map here)
  load(path)/save(m,p)   checkpoints
  score_t(m, x)          per-sample anomaly score WITH gradient
  latent(m, x)           representation used for anchor-mining distances
  rec_loss(m, x)         the target's own training objective on a benign batch
  finetune_mode(m)       mode during fine-tuning: BatchNorm statistics frozen,
                         dropout off, so training computes the test-time score
  for_finetune(m)        the base model as the starting point of fine-tuning
                         (identity, except KitNET with KITNET_UNTIE)
  base_mode(m)           mode during base training (the target's own recipe)

The scaler is the target's MinMaxScaler in params_dir (frozen once written).
"""
import os
import pickle
import types

import numpy as np
import torch
from torch import nn

from .common import TARGET, cfg
from . import data as D

ZERO_IDX = list(cfg('FEATURE_ZERO_IDX', []))


def _scaler():
    return pickle.load(open(os.path.join(D.PARAMS_DIR, 'scaler.pkl'), 'rb'))


class _Base:
    clip = None                  # gradient-norm clipping in fine-tuning (None = off)

    def __init__(self):
        self._sc = None

    @property
    def scaler(self):
        if self._sc is None:
            self._sc = _scaler()
        return self._sc

    def prep(self, X):
        Z = self.scaler.transform(X)
        if ZERO_IDX:                     # a feature group removed (zeroed after scaling)
            Z = np.array(Z, copy=True)
            Z[:, ZERO_IDX] = 0
        return torch.tensor(Z, dtype=torch.float32)

    def load(self, path):
        m = self.new_model(None)
        m.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
        return m.cuda()

    def save(self, m, path):
        torch.save(m.state_dict(), path)

    def base_mode(self, m):
        m.train()

    def for_finetune(self, m):
        return m

    def finetune_mode(self, m):
        m.train()
        for mod in m.modules():
            if isinstance(mod, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
                mod.eval()

    @torch.no_grad()
    def scores(self, m, data_loader):
        m.eval()
        s, y = [], []
        for xb, yb in data_loader:
            xb = xb.cuda()
            s.append(self.score_t(m, xb).cpu().numpy())
            y.append(yb.numpy())
        return np.concatenate(s), np.concatenate(y)


class HorusEye(_Base):
    """Magnifier (1-D CNN AE over 5 x 21 channels); score = per-row RMSE."""

    def __init__(self):
        super().__init__()
        import utils  # targets/horuseye/arise/utils.py: reshape_channel
        from AE import Magnifier
        self._reshape, self._cls = utils.reshape_channel, Magnifier
        self.input_size = cfg('INPUT_SIZE', 105)

    def prep(self, X):
        return torch.tensor(self._reshape(self.scaler.transform(X)), dtype=torch.float32)

    def new_model(self, T_train):
        return self._cls(input_size=self.input_size)

    def score_t(self, m, x):
        return m.rmse_t(m(x), x)

    def latent(self, m, x):
        return m.forward_latent(x)[0]

    def rec_loss(self, m, x):
        return torch.sqrt(nn.functional.mse_loss(m(x), x))


class SimpleAE(_Base):
    """MLP autoencoder 79 -> 64 -> 32 -> 64 -> 79 with BatchNorm; score = per-row RMSE."""

    def __init__(self):
        super().__init__()
        from AE import autoencoder
        self._cls = autoencoder
        self.input_size = cfg('INPUT_SIZE', 79)

    def new_model(self, T_train):
        return self._cls(input_size=self.input_size, hidden_size=32)

    def _rec(self, m, x):
        return m.decoder(m.encoder(x))           # forward() without the .squeeze()

    def score_t(self, m, x):
        return torch.sqrt((self._rec(m, x) - x).pow(2).sum(1) / self.input_size)

    def latent(self, m, x):
        return m.encoder(x)

    def rec_loss(self, m, x):
        return torch.sqrt(nn.functional.mse_loss(self._rec(m, x), x))


class KitNET(_Base):
    """Kitsune ensemble: tied-weight tail AEs over feature groups + head AE over
    the tail errors.  The released score detaches the tail errors before the
    head; score_t is the same arithmetic without the detach (identical values,
    gradients reach the tails)."""
    clip = 2.0

    def __init__(self):
        super().__init__()
        from models import FeatureMapper, Kitsune
        self._fm_cls, self._cls = FeatureMapper, Kitsune

    def new_model(self, T_train):
        """Feature map fitted on the whole training set at once.  Equivalent to
        FeatureMapper.fit(T_train) (a single partial_fit); the correlation sum
        is a matrix product instead of a per-row Python loop."""
        x = T_train.double()
        fm = self._fm_cls(x.shape[1], 10)
        fm.seen_samples = x.shape[0]
        fm.running_sum = x.sum(0)
        r = x - fm.running_sum / float(fm.seen_samples)
        fm.running_residuals = r.sum(0)
        fm.running_squared_residuals = r.pow(2).sum(0)
        fm.running_corr_mat = r.t() @ r
        fm._update_clusters()
        fm._fitted = True
        print('kitnet: feature map with %d groups over %d features' % (fm.n_clusters_, x.shape[1]))
        return self._cls(feature_mapper=fm, compression_rate=0.6, dropout_rate=0.1).cuda()

    untie = cfg('KITNET_UNTIE', False)

    @staticmethod
    def _untied_forward(ae, x):
        x = ae.dropout(x)
        hidden = (x @ ae.w + ae.hidden_bias).tanh()
        return (hidden @ ae.w_de + ae.reconstruct_bias).sigmoid()

    def _untie(self, m):
        """Give every tail and the head its own decoder matrix, initialised to the
        transpose of the encoder: the function is unchanged at the start."""
        for ae in list(m.tails) + [m.head]:
            if not hasattr(ae, 'w_de'):
                ae.w_de = nn.Parameter(ae.w.detach().t().clone())
                ae.forward = types.MethodType(self._untied_forward, ae)
        return m

    def for_finetune(self, m):
        return self._untie(m) if self.untie else m

    def load(self, path):
        chkp = torch.load(path, map_location='cuda:0')
        m = self._cls(**chkp['config'])
        if any(k.endswith('.w_de') for k in chkp['weights']):
            self._untie(m)
        m.load_state_dict(chkp['weights'])
        return m.eval().cuda()

    def save(self, m, path):
        m.save(path)

    def _tail_errors(self, m, x):
        return torch.stack([m.mse(tail(f), f).mean(-1)
                            for f, tail in zip(m.fm.transform(x), m.tails)], 1)

    def score_t(self, m, x):
        tl = self._tail_errors(m, x)
        return m.mse(m.head(tl), tl).mean(-1)

    def latent(self, m, x):
        return torch.cat([(f @ t.w + t.hidden_bias).tanh()
                          for f, t in zip(m.fm.transform(x), m.tails)], 1)

    def rec_loss(self, m, x):
        """The released training objective: summed tail errors + head error on
        the detached tail errors (engine.train_single_epoch)."""
        out = m(x)
        return out['tails_losses'].sum(-1).mean(0) + out['head_loss'].mean(0)


def get():
    return {'horuseye': HorusEye, 'simpleae': SimpleAE, 'kitnet': KitNET}[TARGET]()
