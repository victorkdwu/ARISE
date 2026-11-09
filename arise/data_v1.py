"""The evaluation protocol of each Table II column: the target's own data loaders,
with row identity added.

The test sets are produced by the target's own loaders, unmodified; only two
hooks are added:
  * load_seq_data tags every attack row with rid = "<subtype>:<row index>".  The
    extra column changes no sampling or split: those depend only on row counts
    and random_state.  The rid is removed before the feature matrix is built.
  * preprocess_test is replaced by a capture of the exact rows that are scored.
So every attack row that any test set contains is known.

Anchor candidates (class_rows(cls, 'valid')) are the attack rows of the class
that appear in NO test set (per-class or Overall).  The fine-tuning benign
train/valid split is the target's benign train rows, 80/20 with random_state=20.

Applied by data.py when the preset sets PROTOCOL: v1.  Preset keys:
  V1_LOADER          blur (get_test_dataloaders_per_blur_attack) | per_attack
  V1_OVERALL_FILTER  true: Overall attack set = sub-types of blur_attack_dict only
  plus NORMAL_SPLIT_MODE, DROP_HEADER_ROWS, FEATURE_HAS_FLOWID.
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .common import TARGET, cfg

V1_LOADER = cfg('V1_LOADER', 'blur')
V1_OVERALL_FILTER = cfg('V1_OVERALL_FILTER', False)


def override(ns):
    """Replace the protocol functions of arise.data (namespace ns) with v1 ones."""
    if TARGET != 'horuseye':
        return _override_replica(ns)
    # the HorusEye utils read the benign split from os.environ
    for k in ('NORMAL_SPLIT_MODE', 'NORMAL_TRAIN_BEGIN', 'NORMAL_TRAIN_END', 'NORMAL_TEST_BEGIN',
              'NORMAL_TEST_END', 'NORMAL_TEST_SIZE'):
        v = cfg(k, None)
        if v is not None:
            os.environ['ARISE_' + k] = str(v)
    import utils as U

    blur = ns['BLUR']
    if U.blur_attack_dict != blur:
        raise SystemExit('data_v1: utils.blur_attack_dict differs from the preset')
    feat_drop = U._feature_cols()
    orig_load = U.load_seq_data
    state = {'tests': None, 'pool': {}, 'benign': {}}

    def tagged_load(dataset_path, type_list=[], label=0, begin=0, end=20, drop_header_rows=None):
        if label != 1:
            return orig_load(dataset_path, type_list=type_list, label=label, begin=begin, end=end,
                             drop_header_rows=drop_header_rows)
        types = list(type_list) or os.listdir(dataset_path)
        if V1_OVERALL_FILTER and not type_list:
            keep = {t for v in blur.values() for t in v}
            types = [t for t in types if t in keep]
        parts = []
        for t in types:
            d = orig_load(dataset_path, type_list=[t], label=1, begin=begin, end=end,
                          drop_header_rows=drop_header_rows)
            d['rid'] = ['%s:%d' % (t, i) for i in range(len(d))]
            parts.append(d)
        df = pd.DataFrame()
        for d in parts:                          # same row order as one loader call
            df = df.append(d, ignore_index=True)
        df.fillna(0, inplace=True)               # the loader fills after concatenating all types
        return df

    def capture(df_normal_test, df_attack, params_dir):
        att, _ = train_test_split(df_attack, test_size=0.2, random_state=20)
        test = pd.concat([df_normal_test, att], axis=0)
        X = test.drop(columns=[c for c in feat_drop + ['rid'] if c in test.columns]).values.astype(np.float64)
        y = test['class'].values.astype(np.int64)
        state['_cap'].append((X, y, att['rid'].tolist()))
        return torch.zeros((1, 5, 21)), torch.zeros(1)     # placeholders: the loader wraps them in a DataLoader

    def build_tests():
        if state['tests'] is not None:
            return state['tests']
        U.load_seq_data = tagged_load
        U.preprocess_test = capture
        out = {}
        try:
            state['_cap'] = []
            fn = U.get_test_dataloaders_per_blur_attack if V1_LOADER == 'blur' else U.get_test_dataloaders_per_attack
            d = fn(ns['NORMAL_DIR'], ns['ATTACK_DIR'], 256, './')
            if len(d) != len(state['_cap']):
                raise SystemExit('data_v1: captured %d sets for %d classes' % (len(state['_cap']), len(d)))
            for name, cap in zip(d.keys(), state['_cap']):
                out[name] = cap
            state['_cap'] = []
            U.get_test_dataloaders(ns['NORMAL_DIR'], ns['ATTACK_DIR'], 256, './')
            out['overall'] = state['_cap'][0]
        finally:
            U.load_seq_data = orig_load
        state['tests'] = out
        print('data_v1: test sets %s' % ', '.join('%s %d (attack %d)' % (k, len(v[1]), len(v[2]))
                                                  for k, v in out.items()))
        return out

    def test_rids():
        s = set()
        for _, _, r in build_tests().values():
            s.update(r)
        return s

    def class_rows(cls, part):
        if part not in ('valid', 'test'):
            raise ValueError(part)
        key = (cls, part)
        if key not in state['pool']:
            df = tagged_load(ns['ATTACK_DIR'], type_list=list(blur[cls]), label=1)
            df['subtype'] = df['rid'].str.split(':').str[0]
            inside = df['rid'].isin(test_rids())
            state['pool'][(cls, 'valid')] = df[~inside].reset_index(drop=True)
            state['pool'][(cls, 'test')] = df[inside].reset_index(drop=True)
            state['pool'][(cls, 'size')] = len(df)
        return state['pool'][key]

    def class_size(cls):
        class_rows(cls, 'valid')
        return state['pool'][(cls, 'size')]

    def benign(part):
        if not state['benign']:
            tr = U._normal_split(ns['NORMAL_DIR'], 'train')
            tr, va = train_test_split(tr, test_size=0.2, random_state=20)   # preprocess_train_contra
            for p, d in (('train', tr), ('valid', va)):
                state['benign'][p] = ns['_as_frame'](
                    d.drop(columns=[c for c in feat_drop if c in d.columns]).values.astype(np.float64))
        if part == 'test':
            raise SystemExit('data_v1: benign test rows come only through test_sets()')
        return state['benign'][part]

    def test_sets():
        return {k: (X, y) for k, (X, y, _) in build_tests().items()}

    def assert_disjoint_from_test(rids):
        bad = test_rids().intersection(rids)
        if bad:
            raise SystemExit('data_v1: %d anchor rows are test rows' % len(bad))
        print('data_v1: %d anchors, 0 in any test set' % len(rids))

    def features(df):
        drop = [c for c in list(feat_drop) + ['class', 'rid', 'subtype'] if c in df.columns]
        return df.drop(columns=drop).values.astype(np.float64)

    def release():
        """Drop the raw matrices once they are tensors (row ids are kept)."""
        if state['tests'] is not None:
            state['tests'] = {k: (None, None, r) for k, (_, _, r) in state['tests'].items()}
        state['benign'].clear()

    ns.update(class_rows=class_rows, class_size=class_size, benign=benign, test_sets=test_sets,
              assert_disjoint_from_test=assert_disjoint_from_test, features=features,
              v1_test_rids=test_rids, release=release)


# ----------------------------------------------------------------------------
# simple AE and KitNET: their own loaders, replicated in pandas, with row ids added.
#
# preset key V1R (mapping):
#   reader         keep (header=None, header row kept as data) | drop (header=None,
#                  first row dropped) | named (pd.read_csv default, header = names)
#   drop_cols      non-feature columns (ints or names), 'class' is added
#   file_end       per-type file slice end of load_seq_data (KitNET 20, AE 5)
#   benign_test    {mode: dir, begin, end[, subdir]} | {mode: split20}   (split20: 20% of all
#                  benign rows, random_state=20)
#   benign_train   {mode: dir, begin, end} | {mode: split80}   then 80/20, rs=20
#   train_drop_inf drop +-inf rows from the benign training rows (AE train.py)
#   perclass_mult  benign = mult x attack in the per-class sets (shrinking pool kept)
#   overall_types  all | blur          overall_benign  raw | mult2
# ----------------------------------------------------------------------------

def _override_replica(ns):
    R = cfg('V1R', None)
    if not R:
        raise SystemExit('data_v1: the preset must set V1R for target %s' % TARGET)
    blur = ns['BLUR']
    reader_kind = R['reader']
    drop_cols = list(R.get('drop_cols', [])) + ['class']
    file_end = int(R.get('file_end', 20))
    mult = R.get('perclass_mult', 2)
    state = {'tests': None, 'pool': {}, 'benign': {}, 'files': {}}

    attack_root = os.path.abspath(ns['ATTACK_DIR'])

    def read(f):
        if f not in state['files'] or not os.path.abspath(f).startswith(attack_root):
            if reader_kind == 'keep':
                d = pd.read_csv(f, header=None)
            elif reader_kind == 'drop':
                d = pd.read_csv(f, header=None).iloc[1:, :]
            elif reader_kind == 'named':
                d = pd.read_csv(f)
            else:
                raise SystemExit('data_v1: unknown V1R reader %r' % reader_kind)
            if os.path.abspath(f).startswith(attack_root):   # attack files are re-read; benign are not cached
                state['files'][f] = d
            return d
        return state['files'][f]

    def walk(d):
        out = []
        for root, dirs, files in os.walk(d):
            for f in files:
                if os.path.splitext(f)[1] == '.csv':
                    out.append('{}/{}'.format(root, f))
        return out

    def load(path, types, label, begin=0, end=None, tag=False):
        end = file_end if end is None else end
        types = list(types) or os.listdir(path)
        parts = []
        for t in types:
            fl = [path + t] if '.csv' in t else walk(path + t)
            fl.sort()
            frames = [read(f) for f in fl[begin:end]]
            d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if tag:
                d = d.copy()
                d['rid'] = ['%s:%d' % (t, i) for i in range(len(d))]
            parts.append(d)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        df['class'] = label
        df.fillna(0, inplace=True)
        return df

    def feats(df):
        return df.drop(columns=[c for c in drop_cols + ['rid', 'subtype'] if c in df.columns]).values.astype(np.float64)

    def finalize(df_normal_test, df_attack):
        att, _ = train_test_split(df_attack, test_size=0.2, random_state=20)
        test = pd.concat([df_normal_test, att], axis=0)
        return feats(test), test['class'].values.astype(np.int64), att['rid'].tolist()

    def benign_frame(spec):
        if spec['mode'] == 'dir':
            return load(ns['NORMAL_DIR'] + spec.get('subdir', ''), [], 0, spec.get('begin', 0), spec.get('end'))
        allb = load(ns['NORMAL_DIR'], [], 0)
        tr, te = train_test_split(allb, test_size=0.2, random_state=20)
        return {'split20': te, 'split80': tr}[spec['mode']]

    def build_tests():
        if state['tests'] is not None:
            return state['tests']
        bt = benign_frame(R['benign_test'])
        out, pool = {}, bt
        attack_dir = ns['ATTACK_DIR']
        listing = os.listdir(attack_dir)
        for cls, members in blur.items():
            dfs = [load(attack_dir, [t], 1, tag=True) for t in listing if t in members]
            if len(dfs) > 1:
                med = int(np.median([len(d) for d in dfs]))
                bal = [d.sample(n=med, random_state=20) if len(d) > med else
                       (d.sample(n=med, replace=True, random_state=20) if len(d) < med else d) for d in dfs]
                df_attack = pd.concat(bal, ignore_index=True)
            elif len(dfs) == 1:
                df_attack = dfs[0]
            else:
                raise SystemExit('data_v1: no attack data for %s' % cls)
            need = int(len(df_attack) * mult)
            pool = pool.sample(n=need, replace=len(pool) < need, random_state=20)   # as in the target's own loader
            out[cls] = finalize(pool, df_attack)
        types = [] if R.get('overall_types', 'all') == 'all' else \
            [t for t in listing if t in {x for v in blur.values() for x in v}]
        df_attack = load(attack_dir, types, 1, tag=True)
        if R.get('overall_benign', 'raw') == 'mult2':
            ob = bt.sample(n=df_attack.shape[0] * 2, replace=True, random_state=20)
        else:
            ob = bt
        out['overall'] = finalize(ob, df_attack)
        state['tests'] = out
        print('data_v1: test sets %s' % ', '.join('%s %d (attack %d)' % (k, len(v[1]), len(v[2]))
                                                  for k, v in out.items()))
        return out

    def test_rids():
        s = set()
        for _, _, r in build_tests().values():
            s.update(r)
        return s

    def class_rows(cls, part):
        if (cls, part) not in state['pool']:
            df = load(ns['ATTACK_DIR'], list(blur[cls]), 1, tag=True)
            df['subtype'] = df['rid'].str.split(':').str[0]
            inside = df['rid'].isin(test_rids())
            state['pool'][(cls, 'valid')] = df[~inside].reset_index(drop=True)
            state['pool'][(cls, 'test')] = df[inside].reset_index(drop=True)
            state['pool'][(cls, 'size')] = len(df)
        return state['pool'][(cls, part)]

    def class_size(cls):
        class_rows(cls, 'valid')
        return state['pool'][(cls, 'size')]

    def benign(part):
        if not state['benign']:
            tr = benign_frame(R['benign_train'])
            if R.get('train_drop_inf'):
                tr = tr[~tr.isin([np.inf, -np.inf]).any(axis=1)]
            tr, va = train_test_split(tr, test_size=0.2, random_state=20)
            for p, d in (('train', tr), ('valid', va)):
                state['benign'][p] = ns['_as_frame'](feats(d))
        if part == 'test':
            raise SystemExit('data_v1: benign test rows come only through test_sets()')
        return state['benign'][part]

    def assert_disjoint_from_test(rids):
        bad = test_rids().intersection(rids)
        if bad:
            raise SystemExit('data_v1: %d anchor rows are test rows' % len(bad))
        print('data_v1: %d anchors, 0 in any test set' % len(rids))

    def release():
        if state['tests'] is not None:
            state['tests'] = {k: (None, None, r) for k, (_, _, r) in state['tests'].items()}
        state['benign'].clear()
        state['files'].clear()

    def subtype_split(t):
        """All rows of one sub-type in this target's own CSV reading, tagged (used by the
        AIM v1 split, which re-splits every attack row itself); the second part is empty."""
        d = load(ns['ATTACK_DIR'], [t], 1, tag=True)
        d['subtype'] = t
        return d, d.iloc[0:0]

    ns.update(class_rows=class_rows, class_size=class_size, benign=benign, subtype_split=subtype_split,
              test_sets=lambda: {k: (X, y) for k, (X, y, _) in build_tests().items()},
              assert_disjoint_from_test=assert_disjoint_from_test, features=feats,
              v1_test_rids=test_rids, release=release)
