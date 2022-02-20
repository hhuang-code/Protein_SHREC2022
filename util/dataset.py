import os
import h5py
import numpy as np
from torch.utils.data import Dataset

# import SharedArray as SA

try:
#     from util.data_util import sa_create
    from util.data_util import data_prepare
except:
#     from data_util import sa_create
    from data_util import data_prepare

import pdb


def load_h5_data_label_seg(h5_filename, kwargs):
    # f = h5py.File(h5_filename)
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]     # (N, 3)
        curvatureM = f['curvatureM'][:][:, np.newaxis]   # (N,)
        curvatureG = f['curvatureG'][:][:, np.newaxis]   # (N,)
        curvatureMAX = f['curvatureMAX'][:][:, np.newaxis]   # (N,)
        curvatureMIN = f['curvatureMIN'][:][:, np.newaxis]   # (N,)
        seg = f['label'][:]     # (N,)
        start = f['start'][:]   # (n_shapes,)
        end = f['end'][:]       # (n_shapes,)
        pid = f['pid'][:]       # (n_shapes,)

        feat = []
        if kwargs['curvatureM']:
            if len(feat) == 0:
                feat = curvatureM
            else:
                feat = np.concatenate([feat, curvatureM], axis=1)
        if kwargs['curvatureG']:
            if len(feat) == 0:
                feat = curvatureG
            else:
                feat = np.concatenate([feat, curvatureG], axis=1)
        if kwargs['curvatureMAX']:
            if len(feat) == 0:
                feat = curvatureMAX
            else:
                feat = np.concatenate([feat, curvatureMAX], axis=1)
        if kwargs['curvatureMIN']:
            if len(feat) == 0:
                feat = curvatureMIN
            else:
                feat = np.concatenate([feat, curvatureMIN], axis=1)

    return data, feat, seg, start, end, pid


def load_h5_data(h5_filename, kwargs):
    # f = h5py.File(h5_filename)
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]     # (N, 3)
        curvatureM = f['curvatureM'][:][:, np.newaxis]   # (N,)
        curvatureG = f['curvatureG'][:][:, np.newaxis]   # (N,)
        curvatureMAX = f['curvatureMAX'][:][:, np.newaxis]  # (N,)
        curvatureMIN = f['curvatureMIN'][:][:, np.newaxis]  # (N,)
        start = f['start'][:]   # (n_shapes,)
        end = f['end'][:]       # (n_shapes,)
        pid = f['pid'][:]       # (n_shapes,)

        feat = []
        if kwargs['curvatureM']:
            if len(feat) == 0:
                feat = curvatureM
            else:
                feat = np.concatenate([feat, curvatureM], axis=1)
        if kwargs['curvatureG']:
            if len(feat) == 0:
                feat = curvatureG
            else:
                feat = np.concatenate([feat, curvatureG], axis=1)
        if kwargs['curvatureMAX']:
            if len(feat) == 0:
                feat = curvatureMAX
            else:
                feat = np.concatenate([feat, curvatureMAX], axis=1)
        if kwargs['curvatureMIN']:
            if len(feat) == 0:
                feat = curvatureMIN
            else:
                feat = np.concatenate([feat, curvatureMIN], axis=1)

    return data, feat, start, end, pid


class SHREC2022(Dataset):
    def __init__(self, data_root='../hdf5_data', split='train', transform=None, shuffle_index=False, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.kwargs = kwargs

        if self.split == 'train' or self.split == 'val':
            TRAINING_FILE_LIST = os.path.join(self.data_root, 'train_hdf5_file_list.txt')
            train_file_list = [os.path.join(self.data_root, line.rstrip()) for line in open(TRAINING_FILE_LIST)]

            self.data_list, self.feat_list, self.seg_list, self.start_list, self.end_list, self.pid_list = [], [], [], [], [], []
            for filename in train_file_list:
                data, feat, seg, start, end, pid = load_h5_data_label_seg(filename, self.kwargs)
                self.data_list.append(data)
                self.feat_list.append(feat)
                self.seg_list.append(seg)
                self.start_list.append(start)
                self.end_list.append(end)
                self.pid_list.append(pid)
        else:
            TEST_FILE_LIST = os.path.join(self.data_root, 'test_hdf5_file_list.txt')
            test_file_list = [os.path.join(self.data_root, line.rstrip()) for line in open(TEST_FILE_LIST)]

            self.data_list, self.feat_list, self.start_list, self.end_list, self.pid_list = [], [], [], [], []
            for filename in test_file_list:
                data, feat, start, end, pid = load_h5_data(filename, self.kwargs)
                self.data_list.append(data)
                self.feat_list.append(feat)
                self.start_list.append(start)
                self.end_list.append(end)
                self.pid_list.append(pid)

        self.num_shape_list = [x.shape[0] for x in self.pid_list]

        # self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = \
        #     split, voxel_size, transform, voxel_max, shuffle_index, loop
        #
        # data_list = sorted(os.listdir(data_root))
        # data_list = [item[:-4] for item in data_list if 'Area_' in item]
        #
        # if split == 'train':
        #     self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        # else:
        #     self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        #
        # for item in self.data_list:
        #     if not os.path.exists("/dev/shm/{}".format(item)):
        #     # if not os.path.exists("/scratch/hh1811/data/s3dis/shared/{}".format(item)):
        #         data_path = os.path.join(data_root, item + '.npy')
        #         data = np.load(data_path)  # xyzrgbl, N*7
        #         sa_create("shm://{}".format(item), data)
        #         # sa_create("file:///scratch/hh1811/data/s3dis/shared/{}".format(item), data)
        # self.data_idx = np.arange(len(self.data_list))
        #
        # print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):

        quo = np.where(idx // np.cumsum(self.num_shape_list) == 0)[0][0]
        rmd = idx if quo == 0 else idx - np.cumsum(self.num_shape_list)[quo - 1]

        pid = self.pid_list[quo][rmd]
        start = self.start_list[quo][rmd]
        end = self.end_list[quo][rmd]
        data = self.data_list[quo][start:end]
        feat = self.feat_list[quo][start:end]

        if self.split == 'train' or self.split == 'val':
            seg = self.seg_list[quo][start:end]
            data, feat, seg = data_prepare(data, feat, seg, self.split, self.transform, self.shuffle_index)  # (n_pts, 3), (n_pts, 3), (n_pts,)
        else:
            seg = None
            data, feat = data_prepare(data, feat, seg, self.split, self.transform, self.shuffle_index)  # (n_pts, 3), (n_pts, 3), (n_pts,)

        # data_idx = self.data_idx[idx % len(self.data_idx)]
        #
        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        # # data = SA.attach("file:///scratch/hh1811/data/s3dis/shared/{}".format(self.data_list[data_idx])).copy()
        #
        # coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        # coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        # return coord, feat, label

        if seg is not None:
            return data, feat, seg, pid
        else:
            return data, feat, pid

    def __len__(self):
        # return len(self.data_idx) * self.loop
        return sum(self.num_shape_list)


if __name__ == '__main__':

    import torch

    try:
        from util import transform as t
        from util.data_util import collate_fn
    except:
        import transform as t
        from data_util import collate_fn

    data_root = '../hdf5_data'

    transform = t.Compose([t.RandomScale([0.9, 1.1]),
                           # t.ChromaticAutoContrast(),
                           # t.ChromaticTranslation(),
                           # t.ChromaticJitter(),
                           # t.HueSaturationTranslation()
                           ])

    train_data = SHREC2022(data_root=data_root, transform=transform, shuffle_index=True)

    batch_size = 2
    workers = 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True, sampler=None, drop_last=True, collate_fn=collate_fn)

    for i, (coord, feat, target, offset, pid) in enumerate(train_loader):
        pdb.set_trace()
