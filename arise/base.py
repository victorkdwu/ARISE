"""Train a target model's base checkpoint.

Fits the MinMax scaler on benign train, builds a fresh model (KitNET fits its
feature map on the same rows), trains it with the target's own objective and
recipe, and keeps the epoch with the best PR-AUC on the selection set
(benign valid + all valid-split attacks), the same rule as fine-tuning.
Writes <params_dir>/scaler.pkl and <model_load_path>; refuses to overwrite.

    ARISE_PRESET=<preset> python -m arise.base
Recipe keys (preset): BASE_EPOCHS, BASE_LR, BASE_WD, BASE_BATCH, and BASE_PATIENCE
(stop after that many epochs without a better selection PR-AUC; unset = no early stop).
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from .common import cfg
from . import data as D
from . import targets

_num = lambda v, t: None if v is None else t(v)       # env overrides arrive as strings
EPOCHS = _num(cfg('BASE_EPOCHS', None), int)
LR = _num(cfg('BASE_LR', None), float)
WD = _num(cfg('BASE_WD', None), float)
BATCH = _num(cfg('BASE_BATCH', None), int)
PATIENCE = int(cfg('BASE_PATIENCE', 0))


def main():
    if None in (EPOCHS, LR, WD, BATCH):
        raise SystemExit('base: set BASE_EPOCHS, BASE_LR, BASE_WD, BASE_BATCH in the preset')
    scaler_path = os.path.join(D.PARAMS_DIR, 'scaler.pkl')
    if not any(k in os.path.abspath(D.PARAMS_DIR) + '/' for k in ('/params_v2/', '/params_v3/')):
        raise SystemExit('base: params_dir must be a params_v2/ or params_v3/ directory, got %s' % D.PARAMS_DIR)
    for p in (scaler_path, D.MODEL_LOAD_PATH):
        if os.path.exists(p):
            raise SystemExit('base: %s exists; remove it first to retrain' % p)
    os.makedirs(D.PARAMS_DIR, exist_ok=True)
    D.setup_seed()
    T = targets.get()
    Xtr = D.features(D.benign('train'))
    sc = MinMaxScaler().fit(Xtr)
    pickle.dump(sc, open(scaler_path, 'wb'))
    T._sc = sc
    Ttr = T.prep(Xtr)
    del Xtr
    model = T.new_model(Ttr).cuda()
    train_loader = D.loader(Ttr, np.zeros(len(Ttr)), BATCH, True)
    Xs, ys = D.selection_rows(list(D.BLUR))
    sel_loader = D.loader(T.prep(Xs), ys, 4096, False)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, best_ep, hist, t0 = -1.0, -1, [], time.time()
    for ep in range(EPOCHS):
        T.base_mode(model)
        tot, n = 0.0, 0
        for b_x, _ in train_loader:
            b_x = b_x.cuda()
            opt.zero_grad()
            loss = T.rec_loss(model, b_x)
            loss.backward()
            if T.clip:
                nn.utils.clip_grad_norm_(model.parameters(), T.clip)
            opt.step()
            tot += loss.item()
            n += 1
        m = D.metrics(*T.scores(model, sel_loader)[::-1])
        rec = dict(epoch=ep, train_loss=tot / n, **m)
        if m['pr_auc'] > best:
            best = m['pr_auc']
            best_ep = ep
            rec['saved'] = True
            T.save(model, D.MODEL_LOAD_PATH)
        hist.append(rec)
        print('epoch %d %.0fs loss %.5f sel_pr_auc %.4f sel_tpr1%% %.4f'
              % (ep, time.time() - t0, rec['train_loss'], m['pr_auc'], m['tpr_fpr1e-2']), flush=True)
        if PATIENCE and ep - best_ep >= PATIENCE:
            print('early stop: no better selection PR-AUC for %d epochs (best epoch %d)' % (PATIENCE, best_ep))
            break
    pd.DataFrame(hist).to_csv(os.path.join(D.PARAMS_DIR, 'base_history.csv'), index=False)
    D.run_info(os.path.join(D.PARAMS_DIR, 'base_run_info.json'), files=[scaler_path, D.MODEL_LOAD_PATH],
               extra={'recipe': {'epochs': EPOCHS, 'lr': LR, 'wd': WD, 'batch': BATCH, 'clip': T.clip,
                          'patience': PATIENCE}, 'best_epoch': best_ep})


if __name__ == '__main__':
    main()
