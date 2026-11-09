"""Table II summary and comparison with the numbers reported in the paper (expected/table2.csv).
  python scripts/summarize.py runs <column ...>       full-pipeline runs (results/<col>/runs/)
  python scripts/summarize.py released <column ...>   released checkpoints (results/<col>/released/)
For 'runs', the ARISE row is the anchor configuration with the best selection-set PR-AUC (validation only)."""
import glob
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {'horuseye': 'HorusEye', 'kitnet': 'KitNET', 'simpleae': 'AE',
         'iot': 'IoT', 'cic2017': 'CIC-IDS2017', 'cic23': 'CIC-IoT2023', 'kitsune': 'Kitsune'}
mode, cols = sys.argv[1], sys.argv[2:]
exp = pd.read_csv(os.path.join(ROOT, 'expected/table2.csv'))
rows = []
for col in cols:
    res = os.path.join(ROOT, 'results', col)
    if mode == 'released':
        base, arise, chosen = res + '/released/base', res + '/released/arise', 'released'
    else:
        best, sel = None, -1
        for d in glob.glob(res + '/runs/t3_anchor_*'):
            if os.path.exists(d + '/test.csv'):
                s = json.load(open(d + '/run_info.json'))['best_selection_metrics']['pr_auc']
                if s > sel:
                    best, sel = d, s
        base, arise, chosen = res + '/runs/t3_base', best, os.path.basename(best)[3:]
    b = pd.read_csv(base + '/test.csv').set_index('set')
    a = pd.read_csv(arise + '/test.csv').set_index('set')
    t, ds = col.split('_')
    e = exp[(exp.target == NAMES[t]) & (exp.dataset == NAMES[ds])].set_index('cls')
    for cls in b.index:
        r = dict(column=col, cls=cls, chosen=chosen, base_pr=b.loc[cls, 'pr_auc'], base_tpr=b.loc[cls, 'tpr_fpr1e-2'],
                 arise_pr=a.loc[cls, 'pr_auc'], arise_tpr=a.loc[cls, 'tpr_fpr1e-2'])
        if cls in e.index:
            r['max_abs_diff_vs_paper'] = max(abs(r[k] - e.loc[cls, k]) for k in ('base_pr', 'base_tpr', 'arise_pr', 'arise_tpr'))
        rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(df.round(4).to_string(index=False))
out = os.path.join(ROOT, 'results', 'table2_%s.csv' % mode)
df.to_csv(out, index=False)
print('\nwritten', out, '| largest difference from the paper: %.2e' % df.max_abs_diff_vs_paper.max())
