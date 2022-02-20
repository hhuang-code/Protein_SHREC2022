import os
import random
import numpy as np
import logging
import argparse
import collections
import open3d as o3d

import sys

print(os.path.abspath(__file__))
sys.path.append(".")

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
# from util.voxelize import voxelize
from util.dataset import SHREC2022
from util.data_util import collate_fn

import pdb

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/shrec2022_pointtransformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/shrec2022_pointtransformer.yaml for all options',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)

    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    return logger


def main():
    global args, logger
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)

    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointtransformer_seg':
        from model.pointtransformer_seg import pointtransformer_seg as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

    if args.curvatureM:
        args.fea_dim += 1
    if args.curvatureG:
        args.fea_dim += 1
    if args.curvatureMAX:
        args.fea_dim += 1
    if args.curvatureMIN:
        args.fea_dim += 1

    model = Model(c=args.fea_dim, k=args.classes).cuda()
    logger.info(model)

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    # names = [line.rstrip('\n') for line in open(args.names_path)]

    # test(model, criterion, names)
    test(model)


# def data_prepare():
#     if args.data_name == 's3dis':
#         data_list = sorted(os.listdir(args.data_root))
#         data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
#     else:
#         raise Exception('dataset not supported yet'.format(args.data_name))
#     print("Totally {} samples in val set.".format(len(data_list)))
#
#     return data_list


# def data_load(data_name):
#     data_path = os.path.join(args.data_root, data_name + '.npy')
#     data = np.load(data_path)  # xyzrgbl, N*7
#     coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
#
#     idx_data = []       # length equals to the maximum number of points in any voxel
#     if args.voxel_size:
#         coord_min = np.min(coord, 0)
#         coord -= coord_min
#         idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
#         for i in range(count.max()):
#             idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
#             idx_part = idx_sort[idx_select]
#             idx_data.append(idx_part)
#     else:
#         idx_data.append(np.arange(label.shape[0]))
#
#     return coord, feat, label, idx_data


# def input_normalize(coord, feat):
#     coord_min = np.min(coord, 0)
#     coord -= coord_min
#     feat = feat / 255.
#
#     return coord, feat


# colors
colors = {0: [202, 202, 202], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [255, 204, 153],
          5: [204, 0, 204], 6: [247, 255, 0], 7: [255, 0, 255], 8: [0, 255, 255], 9: [204, 229, 255]}


THRESHOLD = 2.0


# def test(model, criterion, names):
def test(model):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    # batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    # args.batch_size_test = 10
    model.eval()

    check_makedirs(args.save_folder)

    # test_data = SHREC2022(data_root=args.data_root, split='val', transform=None, shuffle_index=False)
    test_data = SHREC2022(data_root=args.data_root, split='val', transform=None, shuffle_index=False,
                          curvatureM=args.curvatureM, curvatureG=args.curvatureG,
                          curvatureMAX=args.curvatureMAX, curvatureMIN=args.curvatureMIN)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False,
                                              collate_fn=collate_fn)

    pred_save, label_save = [], []
    # data_list = data_prepare()
    # for idx, item in enumerate(data_list):
    #     end = time.time()

    # pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
    # label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(item, args.epoch))
    # if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
    #     logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
    #     pred, label = np.load(pred_save_path), np.load(label_save_path)
    # else:
    #     coord, feat, label, idx_data = data_load(item)

    # pred = torch.zeros((label.size, args.classes)).cuda()
    # idx_size = len(idx_data)    # length equals to the maximum number of points in any voxel
    # idx_list, coord_list, feat_list, offset_list = [], [], [], []
    # for i in range(idx_size):
    #     logger.info(
    #         '{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
    #     idx_part = idx_data[i]
    #     coord_part, feat_part = coord[idx_part], feat[idx_part]
    #     if args.voxel_max and coord_part.shape[0] > args.voxel_max:
    #         coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
    #         while idx_uni.size != idx_part.shape[0]:
    #             init_idx = np.argmin(coord_p)
    #             dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
    #             idx_crop = np.argsort(dist)[:args.voxel_max]
    #             coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
    #             dist = dist[idx_crop]
    #             delta = np.square(1 - dist / np.max(dist))
    #             coord_p[idx_crop] += delta
    #             coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
    #             idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(
    #                 feat_sub), offset_list.append(idx_sub.size)
    #             idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
    #     else:
    #         coord_part, feat_part = input_normalize(coord_part, feat_part)
    #         idx_list.append(idx_part), coord_list.append(coord_part), \
    #         feat_list.append(feat_part), offset_list.append(idx_part.size)
    #
    # batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
    # for i in range(batch_num):
    #     s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
    #     idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], \
    #                                                    feat_list[s_i:e_i], offset_list[s_i:e_i]
    #     idx_part = np.concatenate(idx_part)
    #     coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
    #     feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
    #     offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
    #
    #     with torch.no_grad():
    #         pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
    #     torch.cuda.empty_cache()
    #     pred[idx_part, :] += pred_part
    #     logger.info(
    #         'Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i,
    #                                            len(idx_list), args.voxel_max, idx_part.shape[0]))
    # # loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
    # pred = pred.max(1)[1].data.cpu().numpy()

    for i, (coord, feat, label, offset, pid) in enumerate(test_loader):  # (n, 3), (n,), (b,), (b,)

        coord, feat, label, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), \
                                     label.cuda(non_blocking=True), offset.cuda(non_blocking=True)

        # change to binary segmentation
        if args.classes == 2:
            label[label != 0] = 1

        with torch.no_grad():
            output = model([coord, feat, offset])

        # pred = output.max(1)[1].cpu().numpy()
        pred = (output[:, 1] - output[:, 0] > THRESHOLD).cpu().numpy().astype(int)
        label = label.cpu().numpy()

        # calculation 1: add per room predictions
        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        # logger.info('Test: [{}/{}] '
        #             'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #             'Accuracy {accuracy:.4f}.'.format(i + 1, len(test_loader), batch_time=batch_time,
        #                                               accuracy=accuracy))

        logger.info('Test: [{}/{}] Accuracy {accuracy:.4f}.'.format(i + 1, len(test_loader), accuracy=accuracy))

        pred_save.append(pred)
        label_save.append(label)
        # np.save(pred_save_path, pred)
        # np.save(label_save_path, label)

        # save prediction
        for j in range(offset.shape[0]):
            if j == 0:
                st, ed = 0, offset[j]
            else:
                st, ed = offset[j - 1], offset[j]
            s_coord, s_label, s_pred, s_pid = coord[st:ed].cpu().numpy(), label[st:ed], pred[st:ed], pid[j].item()

            filename = os.path.join(args.save_folder, str(s_pid) + '_pred.ply')
            save_prediction(s_coord, s_label, s_pred, filename)

    # with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
    #     pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
    #     pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save),
                                                       np.concatenate(label_save), args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        # logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
        #                                                                             names[i]))
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def save_prediction(coord, label, pred, filename):
    pred_file = filename
    label_file = filename.replace('pred', 'label')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)

    point_colors = np.stack([colors[v] for v in pred], axis=0) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(pred_file, pcd)

    point_colors = np.stack([colors[v] for v in label], axis=0) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(label_file, pcd)


if __name__ == '__main__':
    main()
