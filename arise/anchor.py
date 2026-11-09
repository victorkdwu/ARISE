"""Anchor selection, Eq. (1).

Inside every behaviour cluster, selects anchors by optimising the coverage, importance and sparsity
terms of Eq. (1) under a Gumbel-sigmoid (Concrete) relaxation, then keeps the top-B points.
The budget is ANCHOR_RATIO x class size, allocated to clusters by size.
ANCHOR_SELECT=random draws the same number of anchors uniformly from the class's labelled rows, and
ANCHOR_SELECT=nocluster runs Eq. (1) on the whole class as one cluster (ablations).  Default eq1.

Output: <RESULTS_V2>/<ANCHOR_SUBDIR>/<class>.csv (f0.., subtype, rid, cluster,
score = base score) and anchor_summary.json.

    ARISE_PRESET=<preset> python -m arise.anchor
"""
import json
import math
import os

import numpy as np
import pandas as pd
import torch

from .common import cfg
from . import data as D
from . import targets

OBJECTIVE = cfg('OBJECTIVE', 'v2')
ANCHOR_RATIO = cfg('ANCHOR_RATIO', 0.001)
TAU = cfg('tau', 0.2)
LR = cfg('learning_rate', 5e-3)
EPOCHS = cfg('epoches', 10000)
OPT_BATCH = cfg('opt_batch_size', 2000)
PAIR_SAMPLE = cfg('PAIR_SAMPLE', 200000)
ANCHOR_SUBDIR = cfg('ANCHOR_SUBDIR', 'anchor' if OBJECTIVE == 'v2' else 'anchor_objv1')
W_DIST, W_RMSE, W_SIZE = (1.0, 1.0, 1.0) if OBJECTIVE == 'v2' else (1.0, 2.0, 50.0)
CLUSTER_DIR = os.path.join(D.RESULTS_DIR, 'cluster')
OUT = os.path.join(D.RESULTS_DIR, ANCHOR_SUBDIR)
SELECT = cfg('ANCHOR_SELECT', 'eq1')            # eq1 | random | nocluster
RANDOM_SEED = cfg('ANCHOR_RANDOM_SEED', 20)
T = targets.get()


@torch.no_grad()
def encode(model, X, bs=4096):
    model.eval()
    codes, sc = [], []
    X = T.prep(X)
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda()
        codes.append(T.latent(model, xb))
        sc.append(T.score_t(model, xb))
    return torch.cat(codes), torch.cat(sc)


def allocate(sizes, budget):
    tot = sum(sizes.values())
    raw = {c: budget * n / tot for c, n in sizes.items()}
    alloc = {c: int(math.floor(v)) for c, v in raw.items()}
    rest = budget - sum(alloc.values())
    for c in sorted(raw, key=lambda c: raw[c] - alloc[c], reverse=True)[:rest]:
        alloc[c] += 1
    return {c: min(a, sizes[c]) for c, a in alloc.items()}


def optimise(codes, r):
    n = len(r)
    g = torch.Generator(device='cuda').manual_seed(D.SEED)
    i = torch.randint(0, n, (PAIR_SAMPLE,), device='cuda', generator=g)
    j = torch.randint(0, n, (PAIR_SAMPLE,), device='cuda', generator=g)
    keep = i != j
    d = (codes[i[keep]] - codes[j[keep]]).norm(dim=1)
    mu, sd = d.mean(), d.std().clamp_min(1e-12)
    alpha = torch.randn(n, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([alpha], lr=LR)
    nb = math.ceil(n / OPT_BATCH)
    for ep in range(EPOCHS):
        order = torch.randperm(n, device='cuda') if OBJECTIVE == 'v2' else torch.arange(n, device='cuda')
        opt.zero_grad()
        tot = {'loss': 0.0, 'dist': 0.0, 'rmse': 0.0, 'size': 0.0}
        for b in range(nb):
            idx = order[b * OPT_BATCH:(b + 1) * OPT_BATCH]
            Dm = (torch.cdist(codes[idx], codes[idx]) - mu) / sd
            gum = -torch.log(-torch.log(torch.rand(len(idx), device='cuda') + 1e-9) + 1e-9)
            z = torch.sigmoid((alpha[idx] + gum) / TAU)
            W = z.unsqueeze(0) * z.unsqueeze(1)
            if OBJECTIVE == 'v2':
                W = W - torch.diag(torch.diag(W))
                dist = (W * Dm).sum() / (W.sum() + 1e-9)
            else:
                dist = (W * Dm).sum() / ((z ** 2).sum() + 1e-9)
            rm = (r[idx] * z).sum() / (z.sum() + 1e-9)
            size = z.sum() / len(idx)
            loss = (-W_DIST * dist + W_RMSE * rm + W_SIZE * size) / nb
            loss.backward()
            for k, v in (('loss', loss.item()), ('dist', dist.item()), ('rmse', rm.item()),
                         ('size', size.item())):
                tot[k] += v
        opt.step()
        if ep % 2000 == 0 or ep == EPOCHS - 1:
            print('    epoch %5d  loss %.4f  dist %.4f  rmse %.4f  size %.4f'
                  % (ep, tot['loss'], tot['dist'] / nb, tot['rmse'] / nb, tot['size'] / nb))
    return alpha.detach()


def main():
    D.setup_seed()
    os.makedirs(OUT, exist_ok=True)
    model = T.load(D.MODEL_LOAD_PATH)
    _, rb = encode(model, D.features(D.benign('valid')))
    tau0 = float(torch.quantile(rb.float().cpu(), 0.99))
    summary = {'objective': OBJECTIVE, 'select': SELECT, 'random_seed': RANDOM_SEED, 'weights': [W_DIST, W_RMSE, W_SIZE], 'epochs': EPOCHS,
               'tau': TAU, 'ratio': ANCHOR_RATIO, 'tau0_benign_valid_q99': tau0, 'classes': {}}
    for cls in D.BLUR:
        df = pd.read_csv(os.path.join(CLUSTER_DIR, '%s.csv' % cls))
        if SELECT == 'nocluster':
            df = df.assign(cluster=0)
        cand = df[df.cluster >= 0].reset_index(drop=True)
        fcols = [c for c in df.columns if c.startswith('f')]
        sizes = cand.groupby('cluster').size().to_dict()
        budget = max(1, int(round(ANCHOR_RATIO * D.class_size(cls))))
        alloc = allocate(sizes, budget)
        print('%s: %d candidates in %d clusters, budget %d' % (cls, len(cand), len(sizes), budget))
        picked = []
        if SELECT == 'random':
            alloc = {}
            pool = df.reset_index(drop=True)
            sel = np.sort(np.random.RandomState(RANDOM_SEED).choice(len(pool), min(budget, len(pool)), replace=False))
            a = pool.iloc[sel].copy()
            a['score'] = encode(model, a[fcols].values.astype(np.float64))[1].cpu().numpy()
            picked.append(a)
        for c, k in alloc.items():
            if k == 0:
                continue
            sub = cand[cand.cluster == c].reset_index(drop=True)
            codes, sc = encode(model, sub[fcols].values.astype(np.float64))
            if len(sub) <= k:
                sel = np.arange(len(sub))
            else:
                r = (sc - sc.mean()) / sc.std().clamp_min(1e-12)
                print('  cluster %d: n=%d, select %d' % (c, len(sub), k))
                sel = torch.topk(optimise(codes, r), k).indices.cpu().numpy()
            a = sub.iloc[sel].copy()
            a['score'] = sc.cpu().numpy()[sel]
            picked.append(a)
        anchors = pd.concat(picked, ignore_index=True)
        anchors.to_csv(os.path.join(OUT, '%s.csv' % cls), index=False)
        codes_a, _ = encode(model, anchors[fcols].values.astype(np.float64))
        pd_ = torch.pdist(codes_a) if len(anchors) > 1 else torch.zeros(1)
        _, sc_c = encode(model, cand[fcols].values.astype(np.float64))
        summary['classes'][cls] = {
            'n_candidates': len(cand), 'budget': budget, 'n_anchors': len(anchors),
            'allocation': {int(k): int(v) for k, v in alloc.items()},
            'frac_anchor_below_tau0': float((anchors['score'] < tau0).mean()),
            'frac_candidate_below_tau0': float((sc_c.cpu().numpy() < tau0).mean()),
            'anchor_score_median': float(anchors['score'].median()),
            'candidate_score_median': float(sc_c.median()),
            'anchor_pairwise_latent_mean': float(pd_.mean()),
            'anchor_subtypes': anchors['subtype'].value_counts().to_dict(),
            'candidate_subtypes': cand['subtype'].value_counts().to_dict()}
        print('  ->', json.dumps(summary['classes'][cls]))
    D.assert_disjoint_from_test(pd.concat(
        [pd.read_csv(os.path.join(OUT, '%s.csv' % c)) for c in D.BLUR])['rid'].tolist())
    with open(os.path.join(OUT, 'anchor_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    D.run_info(os.path.join(OUT, 'run_info.json'),
               files=[D.MODEL_LOAD_PATH, os.path.join(D.PARAMS_DIR, 'scaler.pkl')]
               + [os.path.join(CLUSTER_DIR, '%s.csv' % c) for c in D.BLUR])


if __name__ == '__main__':
    main()
