"""Data splits, target-independent.

  benign   the dataset's own train/test rule (NORMAL_SPLIT_MODE), then the
           train part split 80/20 (random_state=20) into train / valid.
  attack   per sub-type, in file order: the first ATTACK_VALID_FRAC rows are
           valid, the rest test (ATTACK_SPLIT_MODE=chrono), or a seeded random
           split (=random).  Independent of which other sub-types are loaded.

Attack rows carry a row id "<subtype>:<index>"; anchors are valid rows by id.
The CSV readers below follow the HorusEye utils.load_seq_data /
utils._normal_split behaviour (file order, slicing, header rows, fillna).

Preset keys:
  FEATURE_DROP_COLS  positional columns that are not features
                     (default [0] if FEATURE_HAS_FLOWID else [])
  INF_POLICY         none | drop (rows with +-inf removed after loading)
  NORMAL_SPLIT_MODE  file_slice | subdir | random | random_train_subdir_test |
                     files (NORMAL_TRAIN_FILES / NORMAL_TEST_FILES, relative to
                     normal_dir)
With PROTOCOL: v1 (all Table II presets) the test sets come from the target's own
loaders instead (data_v1.py).
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .common import cfg, describe

BLUR = cfg('blur_attack_dict', None)
if not BLUR or not all(BLUR.values()):
    raise SystemExit('arise: blur_attack_dict must be set in the preset and non-empty')
NORMAL_DIR = cfg('normal_dir', None)
ATTACK_DIR = cfg('attack_dir', None)
PARAMS_DIR = cfg('params_dir', None)
MODEL_LOAD_PATH = cfg('model_load_path', None)
ATTACK_SPLIT_MODE = cfg('ATTACK_SPLIT_MODE', 'chrono')
ATTACK_VALID_FRAC = cfg('ATTACK_VALID_FRAC', 0.2)
BENIGN_VALID_FRAC = cfg('BENIGN_VALID_FRAC', 0.2)
BENIGN_TEST_RATIO = cfg('BENIGN_TEST_RATIO', 2)
BENIGN_OVERLAP_MAX = cfg('BENIGN_OVERLAP_MAX', 0.01)
RESULTS_DIR = cfg('RESULTS_V2', None)
SEED = cfg('SEED', 20)
SPLIT_SEED = cfg('SPLIT_SEED', SEED)   # random attack split; fix it to vary only the training seed
DROP_HEADER_ROWS = cfg('DROP_HEADER_ROWS', 1)
FEATURE_DROP_COLS = cfg('FEATURE_DROP_COLS', [0] if cfg('FEATURE_HAS_FLOWID', True) else [])
INF_POLICY = cfg('INF_POLICY', 'none')
NORMAL_SPLIT_MODE = cfg('NORMAL_SPLIT_MODE', 'file_slice')
for _name, _val in (('normal_dir', NORMAL_DIR), ('attack_dir', ATTACK_DIR),
                    ('params_dir', PARAMS_DIR), ('model_load_path', MODEL_LOAD_PATH),
                    ('RESULTS_V2', RESULTS_DIR)):
    if not _val:
        raise SystemExit('arise: %s must be set in the preset' % _name)
BENIGN_CACHE = cfg('BENIGN_CACHE', True)
BENIGN_CACHE_DIR = cfg('BENIGN_CACHE_DIR', os.path.join(RESULTS_DIR, 'cache'))

META = ['subtype', 'rid']


def setup_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------ CSV reading ----

def _csv_walk(file_dir):
    out = []
    for root, dirs, files in os.walk(file_dir):
        for f in files:
            if os.path.splitext(f)[1] == '.csv':
                out.append('{}/{}'.format(root, f))
    return out


def read_rows(dataset_path, type_list=(), label=0, begin=0, end=20):
    """HorusEye utils.load_seq_data, verbatim in behaviour."""
    df = pd.DataFrame()
    type_list = list(type_list) or os.listdir(dataset_path)
    for type_name in type_list:
        if '.csv' in type_name:
            files = [dataset_path + '{:}'.format(type_name)]
        else:
            files = _csv_walk(dataset_path + '{:}'.format(type_name))
        files.sort()
        df_type = pd.DataFrame()
        for fp in files[begin:end]:
            try:
                tmp = pd.read_csv(fp, header=None).iloc[DROP_HEADER_ROWS:, :]
                df_type = df_type.append(tmp, ignore_index=True)
            except Exception:
                print(fp)
        df = df.append(df_type, ignore_index=True)
    df['class'] = label
    df.fillna(0, inplace=True)
    return df


def features(df):
    """Float64 feature matrix: non-feature, class and meta columns removed."""
    drop = [c for c in list(FEATURE_DROP_COLS) + ['class'] + META if c in df.columns]
    return df.drop(columns=drop).values.astype(np.float64)


def _finite_rows(df):
    if INF_POLICY == 'none':
        return df
    if INF_POLICY != 'drop':
        raise SystemExit('arise: unknown INF_POLICY %r' % INF_POLICY)
    ok = np.isfinite(features(df)).all(1)
    if (~ok).any():
        print('data: dropped %d rows with +-inf' % int((~ok).sum()))
    return df[ok]


def _normal_split(which):
    """HorusEye utils._normal_split, plus the explicit 'files' mode."""
    mode = NORMAL_SPLIT_MODE
    if mode == 'subdir':
        return read_rows(NORMAL_DIR + which + '/', label=0)
    if mode == 'files':
        names = cfg('NORMAL_TRAIN_FILES' if which == 'train' else 'NORMAL_TEST_FILES', None)
        if not names:
            raise SystemExit('arise: NORMAL_SPLIT_MODE=files needs NORMAL_TRAIN_FILES/NORMAL_TEST_FILES')
        return read_rows(NORMAL_DIR, type_list=names, label=0)
    if mode == 'random_train_subdir_test':
        if which == 'test':
            return read_rows(NORMAL_DIR + 'test/', label=0)
        tr, _ = train_test_split(read_rows(NORMAL_DIR, label=0),
                                 test_size=float(cfg('NORMAL_TEST_SIZE', 0.2)), random_state=20)
        return tr
    if mode == 'random':
        tr, te = train_test_split(read_rows(NORMAL_DIR, label=0),
                                  test_size=float(cfg('NORMAL_TEST_SIZE', 0.2)), random_state=20)
        return tr if which == 'train' else te
    if mode != 'file_slice':
        raise SystemExit('arise: unknown NORMAL_SPLIT_MODE %r' % mode)
    if which == 'train':
        b, e = int(cfg('NORMAL_TRAIN_BEGIN', 0)), int(cfg('NORMAL_TRAIN_END', 4))
    else:
        b, e = int(cfg('NORMAL_TEST_BEGIN', 4)), int(cfg('NORMAL_TEST_END', 6))
    return read_rows(NORMAL_DIR, label=0, begin=b, end=e)


# ---------------------------------------------------------------- attack ----

_ATTACK_CACHE = {}


def subtype_split(subtype):
    """(valid, test) frames of one attack sub-type, with subtype/rid columns."""
    if subtype in _ATTACK_CACHE:
        return _ATTACK_CACHE[subtype]
    if not os.path.isdir(os.path.join(ATTACK_DIR, subtype)):
        raise SystemExit('arise: sub-type %r not found under %s' % (subtype, ATTACK_DIR))
    df = _finite_rows(read_rows(ATTACK_DIR, type_list=[subtype], label=1))
    if len(df) == 0:
        raise SystemExit('arise: sub-type %r loaded 0 rows' % subtype)
    df = df.reset_index(drop=True)
    df['subtype'] = subtype
    df['rid'] = ['%s:%d' % (subtype, i) for i in range(len(df))]
    n_valid = int(round(ATTACK_VALID_FRAC * len(df)))
    if ATTACK_SPLIT_MODE == 'chrono':
        valid, test = df.iloc[:n_valid], df.iloc[n_valid:]
    elif ATTACK_SPLIT_MODE == 'random':
        perm = np.random.RandomState(SPLIT_SEED).permutation(len(df))
        valid, test = df.iloc[np.sort(perm[:n_valid])], df.iloc[np.sort(perm[n_valid:])]
    else:
        raise SystemExit('arise: unknown ATTACK_SPLIT_MODE %r' % ATTACK_SPLIT_MODE)
    _ATTACK_CACHE[subtype] = (valid, test)
    return valid, test


def class_rows(cls, part):
    k = 0 if part == 'valid' else 1
    return pd.concat([subtype_split(t)[k] for t in BLUR[cls]], ignore_index=True)


def class_size(cls):
    return sum(len(subtype_split(t)[0]) + len(subtype_split(t)[1]) for t in BLUR[cls])


# ---------------------------------------------------------------- benign ----

_BENIGN = {}
_PARTS = ('train', 'valid', 'test')


def _benign_key():
    key = {'normal_dir': os.path.abspath(NORMAL_DIR), 'valid_frac': BENIGN_VALID_FRAC,
           'drop_cols': list(FEATURE_DROP_COLS), 'header_rows': DROP_HEADER_ROWS,
           'inf': INF_POLICY, 'mode': NORMAL_SPLIT_MODE}
    for k in ('NORMAL_TRAIN_BEGIN', 'NORMAL_TRAIN_END', 'NORMAL_TEST_BEGIN', 'NORMAL_TEST_END',
              'NORMAL_TEST_SIZE', 'NORMAL_TRAIN_FILES', 'NORMAL_TEST_FILES'):
        key[k] = cfg(k, None)
    return key, hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def _as_frame(X):
    # string column names: features() drops nothing, and DataFrame.sample draws
    # the same row positions as on the original frame of equal length
    return pd.DataFrame(X, columns=['f%d' % i for i in range(X.shape[1])])


def _row_hashes(X):
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values


def _overlap(Xtr, Xte):
    tr = set(_row_hashes(Xtr).tolist())
    th = _row_hashes(Xte)
    frac = float(np.mean([h in tr for h in th])) if len(th) else 0.0
    print('data: benign test rows whose exact feature vector also occurs in benign train: '
          '%.4f (limit %.4f)' % (frac, BENIGN_OVERLAP_MAX))
    if frac > BENIGN_OVERLAP_MAX:
        raise SystemExit('arise: benign train/test overlap %.4f exceeds the limit' % frac)
    return frac


def benign(part):
    """'train', 'valid' or 'test' benign rows (feature columns only), cached as .npy."""
    if not _BENIGN:
        key, h = _benign_key()
        files = {p: os.path.join(BENIGN_CACHE_DIR, 'benign_%s_%s.npy' % (h, p)) for p in _PARTS}
        meta = os.path.join(BENIGN_CACHE_DIR, 'benign_%s.json' % h)
        if BENIGN_CACHE and os.path.isfile(meta) and all(os.path.isfile(f) for f in files.values()):
            for p in _PARTS:
                _BENIGN[p] = _as_frame(np.load(files[p]))
            print('data: benign cache %s loaded (%s); overlap %.4f'
                  % (h, ', '.join('%s %d' % (p, len(_BENIGN[p])) for p in _PARTS),
                     json.load(open(meta))['overlap']))
        else:
            tr = _finite_rows(_normal_split('train'))
            te = _finite_rows(_normal_split('test'))
            tr, va = train_test_split(tr, test_size=BENIGN_VALID_FRAC, random_state=20)
            arrs = {'train': features(tr), 'valid': features(va), 'test': features(te)}
            ov = _overlap(arrs['train'], arrs['test'])
            if BENIGN_CACHE:
                os.makedirs(BENIGN_CACHE_DIR, exist_ok=True)
                tag = '.%d.tmp' % os.getpid()      # concurrent builders never share a temp file
                for p in _PARTS:
                    np.save(files[p] + tag + '.npy', arrs[p])
                    os.replace(files[p] + tag + '.npy', files[p])
                with open(meta + tag, 'w') as fh:
                    json.dump({'key': key, 'overlap': ov,
                               'rows': {p: len(arrs[p]) for p in _PARTS}}, fh, indent=1)
                os.replace(meta + tag, meta)
                print('data: benign cache %s written' % h)
            for p in _PARTS:
                _BENIGN[p] = _as_frame(arrs[p])
    return _BENIGN[part]


# --------------------------------------------------------------- anchors ----

def load_anchors(anchor_dir, classes):
    """(X_raw, class labels, rids) of the anchors, fetched by row id from the valid split."""
    Xs, ys, rids = [], [], []
    for cls in classes:
        path = os.path.join(anchor_dir, '%s.csv' % cls)
        if not os.path.isfile(path):
            raise SystemExit('arise: no anchor file %s' % path)
        a = pd.read_csv(path)
        valid = class_rows(cls, 'valid').set_index('rid')
        missing = sorted(set(a['rid']) - set(valid.index))
        if missing:
            raise SystemExit('arise: %d anchors of %s are not valid-split rows, e.g. %s'
                             % (len(missing), cls, missing[:3]))
        X = features(valid.loc[a['rid']].reset_index())
        stored = a[[c for c in a.columns if c.startswith('f')]].values.astype(np.float64)
        if stored.shape != X.shape or not np.allclose(stored, X, rtol=0, atol=1e-6):
            raise SystemExit('arise: anchor features in %s disagree with the data' % path)
        Xs.append(X)
        ys.extend([cls] * len(X))
        rids.extend(a['rid'].tolist())
    return np.vstack(Xs), np.array(ys), rids


def assert_disjoint_from_test(rids):
    test = set()
    for cls in BLUR:
        test.update(class_rows(cls, 'test')['rid'])
    bad = test.intersection(rids)
    if bad:
        raise SystemExit('arise: %d anchor rows are test rows' % len(bad))
    print('data: %d anchors, 0 in the test split' % len(rids))


# ------------------------------------------------------------- loaders ----

def loader(T, y, batch_size, shuffle):
    """DataLoader over an already-prepared input tensor T."""
    return DataLoader(TensorDataset(T, torch.tensor(np.asarray(y))),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def selection_rows(classes, exclude_rids=()):
    """Raw (X, y) of the model-selection set: benign valid + valid attacks
    of `classes`, minus the anchor rows."""
    att = pd.concat([class_rows(c, 'valid') for c in classes], ignore_index=True)
    att = att[~att['rid'].isin(set(exclude_rids))]
    Xbv = features(benign('valid'))
    print('data: selection set benign %d + attack %d (%s, anchors removed)'
          % (len(Xbv), len(att), ','.join(classes)))
    return np.vstack([Xbv, features(att)]), np.r_[np.zeros(len(Xbv)), np.ones(len(att))]


def _balanced_class_test(cls):
    parts = [subtype_split(t)[1] for t in BLUR[cls]]
    if len(parts) == 1:
        return parts[0]
    med = int(np.median([len(p) for p in parts]))
    out = []
    for p in parts:
        out.append(p if len(p) == med else p.sample(n=med, replace=len(p) < med, random_state=20))
    return pd.concat(out, ignore_index=True)


def test_sets():
    """{name: (X_raw, y)}: one balanced set per class (benign = ratio x attack,
    sampled from the full benign test split), plus 'overall'."""
    bt = benign('test')
    out = {}
    for cls in BLUR:
        att = _balanced_class_test(cls)
        need = BENIGN_TEST_RATIO * len(att)
        bn = bt.sample(n=need, replace=len(bt) < need, random_state=20)
        out[cls] = (np.vstack([features(bn), features(att)]),
                    np.r_[np.zeros(len(bn)), np.ones(len(att))])
    allatt = pd.concat([class_rows(c, 'test') for c in BLUR], ignore_index=True)
    out['overall'] = (np.vstack([features(bt), features(allatt)]),
                      np.r_[np.zeros(len(bt)), np.ones(len(allatt))])
    return out


# --------------------------------------------------------------- metrics ----

def tpr_at(fpr, tpr, limit):
    ok = np.where(fpr <= limit + 1e-6)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    p, r, _ = precision_recall_curve(y, s)
    return {'pr_auc': float(auc(r, p)), 'roc_auc': float(auc(fpr, tpr)),
            'tpr_fpr1e-3': tpr_at(fpr, tpr, 1e-3), 'tpr_fpr1e-2': tpr_at(fpr, tpr, 1e-2),
            'n_benign': int((y == 0).sum()), 'n_attack': int((y == 1).sum())}


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run_info(path, files=(), extra=None):
    here = os.path.dirname(os.path.abspath(__file__))
    git = lambda *a: subprocess.run(['git', '-C', here] + list(a), capture_output=True,
                                    text=True).stdout.strip()
    info = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'git_commit': git('rev-parse', 'HEAD'),
            'git_dirty_tracked': len([l for l in git('status', '--porcelain').splitlines()
                                      if not l.startswith('??')]),
            'config': describe(),
            'files': {f: _md5(f) for f in files if f and os.path.isfile(f)}}
    info.update(extra or {})
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(info, fh, indent=1, default=str)


def release():
    """Drop cached raw matrices once the caller holds tensors."""
    _BENIGN.clear()


# ------------------------------------------------------------ protocol ----
# PROTOCOL v1: the evaluation protocol of each Table II column, through the
# target's own data loaders (data_v1.py).
PROTOCOL = cfg('PROTOCOL', 'v2')
if PROTOCOL == 'v1':
    from .data_v1 import override as _v1_override
    _v1_override(globals())
elif PROTOCOL != 'v2':
    raise SystemExit('arise: unknown PROTOCOL %r' % PROTOCOL)

# Controlled label noise in the anchor-mining pool (noise.py).
if cfg("NOISE_MODE", "none") != "none":
    from .noise import install as _noise_install
    _noise_install(globals())
