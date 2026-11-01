# summarize_demos.py (Isaac Lab Python으로 실행)

import h5py, numpy as np


for filename in ["team10_lee_s3"]:
    f = h5py.File(f'datasets/{filename}.hdf5', 'r')
    D = f['data']

    demos = sorted(D, key=lambda k: int(k.split('_')[1]))
    print(f'filename: atasets/{filename}.hdf5')
    print('episodes:', len(demos), '| total:', D.attrs['total'])

    for k in demos:
        a = D[k]['actions'][:]
        n = D[k].attrs['num_samples']
        ok = int(D[k].attrs['success'])
        sw = int((np.diff(a[:, 6]) != 0).sum())   # 그리퍼 전환
        idle = (a[:, :6] == 0).all(1).mean()       # 멈춤 비율
        print(f'{k:<8s} len={n:4d} ({n/20:.1f}s) '
            f'sw={sw:2d} idle={idle:.0%} ok={ok}')
