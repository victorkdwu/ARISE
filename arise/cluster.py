"""Behaviour clustering for anchor mining.

Valid-split attack rows only; DBSCAN in the target's own MinMax space; eps at
the knee of the k-distance curve (k = min_samples) unless DBSCAN_EPS is set.
Output: <RESULTS_V2>/cluster/<class>.csv (f0.., subtype, rid, cluster) and
cluster_summary.json.

    ARISE_PRESET=<preset> python -m arise.cluster
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .common import cfg
from . import data as D
from . import targets

MIN_SAMPLES = cfg('DBSCAN_MIN_SAMPLES', 10)
EPS = cfg('DBSCAN_EPS', 0.0)
KNEE_SAMPLE = cfg('KNEE_SAMPLE', 20000)
# per-class eps taken from another run's cluster_summary.json (label-noise runs: the knee rule is
# dominated by far-away mislabelled rows, so eps is held at its clean-data value)
EPS_FROM = cfg('DBSCAN_EPS_FROM', '')
OUT = os.path.join(D.RESULTS_DIR, 'cluster')


def knee_eps(X, k, n_sample, seed):
    idx = np.random.RandomState(seed).permutation(len(X))[:n_sample]
    d = np.sort(NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[idx])[0][:, -1])
    if d[-1] - d[0] <= 0:
        return float(d[-1]) or 1e-6
    x = np.linspace(0, 1, len(d))
    return float(d[int(np.argmax(x - (d - d[0]) / (d[-1] - d[0])))])


def main():
    D.setup_seed()
    os.makedirs(OUT, exist_ok=True)
    scaler = targets.get().scaler
    summary = {}
    for cls in D.BLUR:
        df = D.class_rows(cls, 'valid')
        Xraw = D.features(df)
        X = scaler.transform(Xraw)
        eps = json.load(open(EPS_FROM))[cls]['eps'] if EPS_FROM else \
            (EPS or knee_eps(X, MIN_SAMPLES, KNEE_SAMPLE, D.SEED))
        lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X)
        sizes = {int(c): int(n) for c, n in zip(*np.unique(lab, return_counts=True))}
        print('%s: %d valid rows, eps=%.4f, %d clusters, noise %.3f'
              % (cls, len(X), eps, len([c for c in sizes if c >= 0]), sizes.get(-1, 0) / len(X)))
        out = pd.DataFrame(Xraw, columns=['f%d' % i for i in range(Xraw.shape[1])])
        out['subtype'] = df['subtype'].values
        out['rid'] = df['rid'].values
        out['cluster'] = lab
        out.to_csv(os.path.join(OUT, '%s.csv' % cls), index=False)
        per_sub = out[out.cluster >= 0].groupby(['cluster', 'subtype']).size()
        summary[cls] = {'n_valid': len(X), 'n_total': D.class_size(cls), 'eps': eps,
                        'min_samples': MIN_SAMPLES, 'sizes': sizes,
                        'cluster_subtypes': {'%s|%s' % k: int(v) for k, v in per_sub.items()}}
    with open(os.path.join(OUT, 'cluster_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    D.run_info(os.path.join(OUT, 'run_info.json'), files=[os.path.join(D.PARAMS_DIR, 'scaler.pkl')])


if __name__ == '__main__':
    main()
