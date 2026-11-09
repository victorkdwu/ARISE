"""Knowledge integration: anchor-constrained fine-tuning of any target model.

  L = (1 - alpha) * L_rec + alpha * mean_a [ max(0, (1 + R) * tau0 - s(a)) ]^2

L_rec is the target's own training objective on benign batches; s is the
target's test-time score with gradient; tau0 is the base model's 99th
percentile on benign valid (fixed).  alpha = 0 ('none') is benign-only
fine-tuning.  Fine-tuning keeps BatchNorm statistics frozen and dropout off.
The best epoch is chosen by PR-AUC on the selection set (benign valid + valid
attacks of the anchor classes, anchors removed).

Data are loaded once; --grid runs several (alpha, margin) settings in sequence,
each from the base checkpoint with a fresh optimizer and the same seed.

    ARISE_PRESET=<preset> python -m arise.train --anchor_type all --prefix x_ --test \
        --grid anchor:0.9:1.0 none:0:0
Outputs: <RESULTS_V2>/runs/<prefix><loss>_a<alpha>_m<margin>/
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from .common import cfg
from . import data as D
from . import targets

ap = argparse.ArgumentParser()
ap.add_argument('--grid', nargs='+', required=True, help='anchor:alpha:margin | none:0:0')
ap.add_argument('--anchor_type', default='all')
ap.add_argument('--anchor_subdir', default='anchor')
ap.add_argument('--prefix', default='')
ap.add_argument('--test', action='store_true')
ap.add_argument('--skip_base', action='store_true')
ap.add_argument('--test_each_epoch', action='store_true',
                help='also score every test set after every epoch (analysis only; selection is unchanged)')
ap.add_argument('--last', action='store_true', help='keep the last epoch (no selection on the selection set)')
ap.add_argument('--lr', type=float, default=cfg('FT_LR', 1e-4))
ap.add_argument('--weight_decay', type=float, default=cfg('FT_WD', 1e-4))
ap.add_argument('--epochs', type=int, default=cfg('FT_EPOCHS', 20))
ap.add_argument('--accumulation_steps', type=int, default=cfg('FT_ACCUM', 12))
ap.add_argument('--batch_size', type=int, default=cfg('FT_BATCH', 256))
args = ap.parse_args()

T = targets.get()
D.setup_seed()
classes = list(D.BLUR) if args.anchor_type == 'all' else [args.anchor_type]
anchor_dir = os.path.join(D.RESULTS_DIR, args.anchor_subdir)
Xa_raw, ya, rids = D.load_anchors(anchor_dir, classes)
D.assert_disjoint_from_test(rids)
Xtr = D.features(D.benign('train'))
BENIGN_FRAC = float(cfg('FT_BENIGN_FRAC', 1.0))      # only part of the target's benign training data
if BENIGN_FRAC < 1:
    Xtr = Xtr[np.sort(np.random.RandomState(20).choice(len(Xtr), int(round(BENIGN_FRAC * len(Xtr))), replace=False))]
    print('benign train subsampled to %.0f%%: %d rows' % (100 * BENIGN_FRAC, len(Xtr)))
train_loader = D.loader(T.prep(Xtr), np.zeros(len(Xtr)), args.batch_size, True)
del Xtr
Xs, ys = D.selection_rows(classes, rids)
sel_loader = D.loader(T.prep(Xs), ys, 4096, False)
del Xs
X_anchor = T.prep(Xa_raw).cuda()
anchor_loader = D.loader(T.prep(Xa_raw), np.ones(len(Xa_raw)), 4096, False)
bv = D.features(D.benign('valid'))
bv_loader = D.loader(T.prep(bv), np.zeros(len(bv)), 4096, False)
del bv
test_loaders = None
if args.test:
    test_loaders = {k: D.loader(T.prep(X), y, 4096, False) for k, (X, y) in D.test_sets().items()}

D.release()                      # raw matrices are now tensors in the loaders
import gc
gc.collect()
tau0 = float(np.quantile(T.scores(T.load(D.MODEL_LOAD_PATH), bv_loader)[0], 0.99))
print('tau0 (base benign-valid q99) = %.6f' % tau0)


def run(loss_name, alpha, margin):
    tag = '%s%s_a%g_m%g' % (args.prefix, loss_name, alpha, margin)
    out_dir = os.path.join(D.RESULTS_DIR, 'runs', tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, 'model.pkl')
    D.setup_seed()
    model = T.for_finetune(T.load(D.MODEL_LOAD_PATH))
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def evaluate():
        s, y = T.scores(model, sel_loader)
        m = D.metrics(y, s)
        m['anchor_frac_above_tau0'] = float((T.scores(model, anchor_loader)[0] > tau0).mean())
        m['benign_valid_frac_above_tau0'] = float((s[y == 0] > tau0).mean())
        if args.test_each_epoch and test_loaders is not None:
            for k, l in test_loaders.items():
                mt = D.metrics(*T.scores(model, l)[::-1])
                m['test_%s_pr_auc' % k], m['test_%s_tpr1' % k] = mt['pr_auc'], mt['tpr_fpr1e-2']
        return m

    records = [dict(epoch=-1, train_loss=np.nan, **evaluate())]
    best, t0 = -1.0, time.time()
    for epoch in range(args.epochs):
        T.finetune_mode(model)
        tot, n = 0.0, 0
        for step, (b_x, _) in enumerate(train_loader):
            b_x = b_x.cuda()
            loss = T.rec_loss(model, b_x)
            if alpha > 0:
                hinge = torch.relu((1 + margin) * tau0 - T.score_t(model, X_anchor))
                loss = (1 - alpha) * loss + alpha * (hinge ** 2).mean()
            tot += loss.item()
            n += 1
            (loss / args.accumulation_steps).backward()
            if ((step + 1) % args.accumulation_steps == 0) or (step + 1 == len(train_loader)):
                if T.clip:
                    nn.utils.clip_grad_norm_(model.parameters(), T.clip)
                opt.step()
                opt.zero_grad()
        rec = dict(epoch=epoch, train_loss=tot / n, **evaluate())
        if rec['pr_auc'] > best or args.last:
            best = rec['pr_auc']
            T.save(model, ckpt)
        records.append(rec)
        print('[%s] epoch %d %.0fs sel_pr_auc %.4f sel_tpr1%% %.4f anchors>tau0 %.3f benign>tau0 %.4f'
              % (tag, epoch, time.time() - t0, rec['pr_auc'], rec['tpr_fpr1e-2'],
                 rec['anchor_frac_above_tau0'], rec['benign_valid_frac_above_tau0']), flush=True)
    hist = pd.DataFrame(records)
    hist.to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    best_row = hist.iloc[-1] if args.last else hist[hist.epoch >= 0].sort_values('pr_auc', ascending=False).iloc[0]
    extra = {'args': vars(args), 'loss': loss_name, 'alpha': alpha, 'margin': margin, 'tau0': tau0,
             'n_anchors': len(rids), 'anchor_classes': classes, 'best_epoch': int(best_row.epoch),
             'best_selection_metrics': best_row.to_dict()}
    if test_loaders is not None:
        model = T.load(ckpt)
        rows = [dict(set=k, **D.metrics(*T.scores(model, l)[::-1])) for k, l in test_loaders.items()]
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    D.run_info(os.path.join(out_dir, 'run_info.json'),
               files=[D.MODEL_LOAD_PATH, os.path.join(D.PARAMS_DIR, 'scaler.pkl')]
               + [os.path.join(anchor_dir, '%s.csv' % c) for c in classes], extra=extra)
    print('[%s] best epoch %d  sel pr_auc %.4f' % (tag, best_row.epoch, best_row.pr_auc), flush=True)


if test_loaders is not None and not args.skip_base:
    base_dir = os.path.join(D.RESULTS_DIR, 'runs', args.prefix + 'base')
    os.makedirs(base_dir, exist_ok=True)
    bm = T.load(D.MODEL_LOAD_PATH)
    pd.DataFrame([dict(set=k, **D.metrics(*T.scores(bm, l)[::-1])) for k, l in test_loaders.items()]
                 ).to_csv(os.path.join(base_dir, 'test.csv'), index=False)
    print('base test written', flush=True)

for spec in args.grid:
    name, a, m = spec.split(':')
    if name == 'none':
        name, a = 'rec', '0'
    elif name != 'anchor':
        raise SystemExit('unknown loss %r (anchor | none)' % name)
    run(name, float(a), float(m))
