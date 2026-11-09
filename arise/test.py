"""Detection test on the test split.

    ARISE_PRESET=<preset> python -m arise.test --ckpt <model> --out <dir>
    ARISE_PRESET=<preset> python -m arise.test --base --out <dir>
"""
import argparse
import os

import pandas as pd

from . import data as D
from . import targets

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--ckpt')
g.add_argument('--base', action='store_true')
ap.add_argument('--out', required=True)
args = ap.parse_args()

D.setup_seed()
T = targets.get()
ckpt = D.MODEL_LOAD_PATH if args.base else args.ckpt
model = T.load(ckpt)
rows = []
for name, (X, y) in D.test_sets().items():
    m = D.metrics(*T.scores(model, D.loader(T.prep(X), y, 4096, False))[::-1])
    rows.append(dict(set=name, **m))
    print(name, m)
os.makedirs(args.out, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(args.out, 'test.csv'), index=False)
D.run_info(os.path.join(args.out, 'test_run_info.json'),
           files=[ckpt, os.path.join(D.PARAMS_DIR, 'scaler.pkl')], extra={'ckpt': ckpt})
