"""Validate TOSICA benchmark dataset splits against expected counts."""
import numpy as np
import os

# Expected counts from each dataset module's docstring
EXPECTED = {
    'hArtery': {
        'label_col': 'Celltype',
        'train': {'T': 2769, 'Fib': 1648, 'EC': 2843, 'SMC': 1080, 'Myeloid': 1166,
                  'B': 439, 'NK': 394, 'MSC': 388, 'Plasma': 155, 'Mast': 78},
        'test': {'T': 13323, 'Fib': 2816, 'EC': 3117, 'SMC': 2801, 'Myeloid': 8941,
                 'B': 1169, 'NK': 1110, 'MSC': 857, 'Plasma': 814, 'Mast': 451},
    },
    'hBone': {
        'label_col': 'Celltype',
        'train': {'HomC': 6930, 'preFC': 497, 'RegC': 3385, 'RepC': 669,
                  'HTC': 2474, 'preHTC': 201, 'FC': 459},
        'test': {'HomC': 397, 'preFC': 4423, 'RegC': 184, 'RepC': 2211,
                 'HTC': 50, 'preHTC': 1994, 'FC': 2266},
    },
    'hPancreas': {
        'label_col': 'Celltype',
        'train': {'Alpha': 3136, 'Beta': 2966, 'Ductal': 1290, 'Acinar': 1144,
                  'Delta': 793, 'PSC': 524, 'PP': 356, 'Endothelial': 273,
                  'Macrophage': 52, 'Mast': 25, 'Epsilon': 21, 'Schwann': 13,
                  'T_cell': 7, 'MHC class II': 0},
        'test': {'Alpha': 2011, 'Beta': 1006, 'Ductal': 414, 'PP': 282,
                 'Acinar': 209, 'Delta': 188, 'PSC': 73, 'Endothelial': 16,
                 'Epsilon': 7, 'Mast': 7, 'MHC class II': 5,
                 'Macrophage': 0, 'Schwann': 0, 'T_cell': 0},
    },
    'mPancreas': {
        'label_col': 'Celltype',
        'train': {'Alpha': 947, 'Beta': 646, 'Delta': 33, 'Ductal': 3769,
                  'Epsilon': 89, 'Fev+ Alpha': 339, 'Fev+ Beta': 902,
                  'Fev+ Delta': 59, 'Fev+ Epsilon': 66, 'Fev+ Pyy': 100,
                  'Mat. Acinar': 108, 'Multipotent': 1224, 'Ngn3 High early': 771,
                  'Ngn3 High late': 1090, 'Ngn3 low EP': 2354, 'Prlf. Acinar': 3194,
                  'Prlf. Ductal': 1890, 'Prlf. Tip': 2284, 'Prlf. Trunk': 1168,
                  'Tip': 3815, 'Trunk': 617},
        'test': {'Alpha': 481, 'Beta': 591, 'Delta': 70, 'Ductal': 499,
                 'Epsilon': 142, 'Fev+ Alpha': 6, 'Fev+ Beta': 457,
                 'Fev+ Delta': 51, 'Fev+ Epsilon': 46, 'Fev+ Pyy': 32,
                 'Mat. Acinar': 6959, 'Multipotent': 1, 'Ngn3 High early': 0,
                 'Ngn3 High late': 642, 'Ngn3 low EP': 262, 'Prlf. Acinar': 171,
                 'Prlf. Ductal': 417, 'Prlf. Tip': 0, 'Prlf. Trunk': 37,
                 'Tip': 6, 'Trunk': 16},
    },
    'mAtlas': {
        'label_col': 'cell_ontology_class',
        # Too many types to list — just check totals
        'train_total': 34027,
        'test_total': 76797,
    },
    'mBrain': {
        'label_col': 'cell_ontology_class',
        'train': {'astrocyte': 3843, 'brain pericyte': 767, 'endothelial cell': 2987,
                  'ependymal cell': 119, 'macrophage': 170, 'microglial cell': 4669,
                  'neuron': 28241, 'olfactory ensheathing cell': 16,
                  'oligodendrocyte': 7178, 'oligodendrocyte precursor cell': 811},
        'test': {'astrocyte': 1059, 'brain pericyte': 281, 'endothelial cell': 392,
                 'ependymal cell': 79, 'macrophage': 96, 'microglial cell': 275,
                 'neuron': 3584, 'olfactory ensheathing cell': 107,
                 'oligodendrocyte': 1485, 'oligodendrocyte precursor cell': 36},
    },
}


def validate(data_root='data'):
    results = {}
    for name, exp in EXPECTED.items():
        data_dir = os.path.join(data_root, name)
        try:
            cn = np.load(os.path.join(data_dir, 'class_names.npy'), allow_pickle=True)
            yt = np.load(os.path.join(data_dir, 'y_train.npy'))
            yv = np.load(os.path.join(data_dir, 'y_val.npy'))
            ye = np.load(os.path.join(data_dir, 'y_test.npy'))
        except FileNotFoundError:
            print(f'\n{"="*60}\n{name}: NOT FOUND\n{"="*60}')
            results[name] = 'MISSING'
            continue

        y_trainval = np.concatenate([yt, yv])
        actual_train = {str(cn[i]): int((y_trainval == i).sum()) for i in range(len(cn))}
        actual_test = {str(cn[i]): int((ye == i).sum()) for i in range(len(cn))}

        print(f'\n{"="*60}')
        print(f'{name}: {len(cn)} classes, '
              f'train+val={len(y_trainval):,}, test={len(ye):,}')
        print(f'{"="*60}')

        # Check per-class counts
        if 'train' in exp:
            exp_train, exp_test = exp['train'], exp['test']
            exp_total_train = sum(exp_train.values())
            exp_total_test = sum(exp_test.values())
            all_types = sorted(set(list(exp_train.keys()) + list(actual_train.keys())))

            ok = True
            for ct in all_types:
                et = exp_train.get(ct, 0)
                at = actual_train.get(ct, 0)
                ee = exp_test.get(ct, 0)
                ae = actual_test.get(ct, 0)
                t_match = '✓' if et == at else f'✗ (got {at})'
                e_match = '✓' if ee == ae else f'✗ (got {ae})'
                if et != at or ee != ae:
                    ok = False
                print(f'  {ct:35s}  train: {et:6d} {t_match:15s}  test: {ee:6d} {e_match}')

            print(f'  {"TOTAL":35s}  train: {exp_total_train:6d} '
                  f'{"✓" if exp_total_train == len(y_trainval) else f"✗ (got {len(y_trainval)})"}'
                  f'            test: {exp_total_test:6d} '
                  f'{"✓" if exp_total_test == len(ye) else f"✗ (got {len(ye)})"}')
            results[name] = 'MATCH' if ok else 'MISMATCH'
        else:
            # Just check totals
            et = exp.get('train_total', 0)
            ee = exp.get('test_total', 0)
            t_ok = et == len(y_trainval)
            e_ok = ee == len(ye)
            print(f'  train+val total: {len(y_trainval):,} (expected {et:,}) {"✓" if t_ok else "✗"}')
            print(f'  test total:      {len(ye):,} (expected {ee:,}) {"✓" if e_ok else "✗"}')
            print(f'  classes: {list(cn[:5])}...')
            results[name] = 'MATCH' if t_ok and e_ok else 'MISMATCH'

    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    for name, status in results.items():
        print(f'  {name:12s}  {status}')


if __name__ == '__main__':
    validate()