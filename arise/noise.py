"""Controlled label noise in the labelled attack pool ARISE mines anchors from.
Installed by data.py when NOISE_MODE != 'none'; set it for the cluster step (the pool) and the
train step (anchors are fetched from that pool), not for the anchor step, so that the anchor budget
(clean class size) stays clean.  The selection set is rebuilt from the clean pools and the test
sets are untouched, so both are identical across noise levels.  Each row keeps its true subtype,
so anchor purity follows from anchor_summary.json ('benign' = injected benign row; a subtype
outside the class = flipped).
  NOISE_MODE=flip    each candidate row is relabelled, with probability NOISE_RATE, to one of the
                     other coarse classes (uniformly)
  NOISE_MODE=benign  benign training rows labelled as the class are added until they make up
                     NOISE_RATE of the class's pool
"""
import numpy as np
import pandas as pd

from .common import cfg

MODE = cfg('NOISE_MODE', 'none')
RATE = float(cfg('NOISE_RATE', 0.0))
SEED = int(cfg('NOISE_SEED', 20))


def install(ns):
    orig, feats, classes = ns['class_rows'], ns['features'], list(ns['BLUR'])
    cache = {}

    def frame(X, cls_rows):
        f = pd.DataFrame(X, columns=['f%d' % i for i in range(X.shape[1])])
        f['class'] = 1
        f['rid'] = cls_rows['rid'].values
        f['subtype'] = cls_rows['subtype'].values
        return f

    def build():
        rng = np.random.RandomState(SEED)
        pools = {c: orig(c, 'valid') for c in classes}
        if MODE == 'flip':
            parts = {c: [] for c in classes}
            for c in classes:
                df = pools[c]
                m = rng.rand(len(df)) < RATE
                parts[c].append(df[~m])
                others = [o for o in classes if o != c]
                tgt = rng.randint(0, len(others), size=int(m.sum()))
                for j, o in enumerate(others):
                    parts[o].append(df[m][tgt == j])
            for c in classes:
                d = pd.concat(parts[c], ignore_index=True)
                cache[c] = frame(feats(d), d)
        elif MODE == 'benign':
            B = feats(ns['benign']('train'))
            used = np.zeros(len(B), bool)
            for c in classes:
                df = pools[c]
                n = int(round(RATE / (1 - RATE) * len(df)))
                idx = rng.choice(np.flatnonzero(~used), n, replace=False)
                used[idx] = True
                inj = pd.DataFrame({'rid': ['benign:%d' % i for i in idx], 'subtype': 'benign'})
                Xa = feats(df)
                assert Xa.shape[1] == B.shape[1], (Xa.shape, B.shape)
                cache[c] = pd.concat([frame(Xa, df), frame(B[idx], inj)], ignore_index=True)
        else:
            raise SystemExit('noise: unknown NOISE_MODE %r' % MODE)
        for c in classes:
            st = cache[c]['subtype']
            print('noise %s %.2f: %s pool %d rows, %d clean' % (MODE, RATE, c, len(st),
                  int(st.isin(ns['BLUR'][c]).sum())), flush=True)

    def class_rows(cls, part):
        if part != 'valid':
            return orig(cls, part)
        if not cache:
            build()
        return cache[cls]

    def selection_rows(classes, exclude_rids=()):
        att = pd.concat([orig(c, 'valid') for c in classes], ignore_index=True)
        att = att[~att['rid'].isin(set(exclude_rids))]
        Xbv = feats(ns['benign']('valid'))
        print('data: selection set (clean labels) benign %d + attack %d (%s, anchors removed)'
              % (len(Xbv), len(att), ','.join(classes)))
        return np.vstack([Xbv, feats(att)]), np.r_[np.zeros(len(Xbv)), np.ones(len(att))]

    ns['class_rows'] = class_rows
    ns['selection_rows'] = selection_rows
