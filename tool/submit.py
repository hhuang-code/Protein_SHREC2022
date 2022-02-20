import os
import random
import numpy as np
import trimesh
import logging
import argparse
import collections
import open3d as o3d
from sklearn.cluster import DBSCAN

import sys

print(os.path.abspath(__file__))
sys.path.append(".")

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.dataset import SHREC2022
from util.data_util import collate_fn_submit

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

    test(model)


THRESHOLD = 1.     # default value 1.5
EXCEPT_LIST = {2.0: [1034, 1056, 273, 290, 616, 76, 878, 881], 1.5: [1056, 273, 616, 881], 1.0: []}

# def test(model, criterion, names):
def test(model):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    model.eval()

    check_makedirs(args.save_folder)

    # test_data = SHREC2022(data_root=args.data_root, split='test', transform=None, shuffle_index=False)
    test_data = SHREC2022(data_root=args.data_root, split='test', transform=None, shuffle_index=False,
                          curvatureM=args.curvatureM, curvatureG=args.curvatureG,
                          curvatureMAX=args.curvatureMAX, curvatureMIN=args.curvatureMIN)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False,
                                              collate_fn=collate_fn_submit)

    for i, (coord, feat, offset, pid) in enumerate(test_loader):  # (n, 3), (n,), (b,), (b,)

        coord, feat, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), offset.cuda(
            non_blocking=True)

        with torch.no_grad():
            output = model([coord, feat, offset])

        # pred = output.max(1)[1].cpu().numpy()
        pred = (output[:, 1] - output[:, 0] > THRESHOLD).cpu().numpy()

        logger.info('Test: [{}/{}].'.format(i + 1, len(test_loader)))

        for j in range(offset.shape[0]):
            if j == 0:
                st, ed = 0, offset[j]
            else:
                st, ed = offset[j - 1], offset[j]
            # s_coord, s_pred, s_pid = coord[st:ed].cpu().numpy(), pred[st:ed], pid[j].item()
            s_output, s_pred, s_pid = output[st:ed].cpu().numpy(), pred[st:ed], pid[j].item()

            pred_filename = os.path.join(args.save_folder, 'test_' + str(s_pid) + '_pred.ply')

            if not (os.path.isfile(pred_filename) or s_pid in EXCEPT_LIST[THRESHOLD]):
                # save_prediction(s_coord, s_pred, pred_filename)
                save_prediction(s_output, s_pred, s_pid, pred_filename)

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


colors = {0: [202, 202, 202], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [255, 204, 153], 5: [204, 0, 204],
          6: [247, 255, 0], 7: [255, 0, 255], 8: [0, 255, 255], 9: [204, 229, 255], 10: [204, 255, 204]}


def save_prediction(output, pred, pid, pred_filename):
    # load original mesh
    off_filename = os.path.join(args.data_root, 'test', str(pid), 'triangulatedSurf.off')
    mesh = trimesh.load(off_filename, process=False, maintain_order=True)

    # statistics of distance between pairs of vertices
    vertices = mesh.vertices
    edges = mesh.edges

    st_pt = vertices[edges[:, 0], :]
    ed_pt = vertices[edges[:, 1], :]

    dist = np.linalg.norm(st_pt - ed_pt, ord=2, axis=1)

    # cluster on the positive prediction vertices
    dbscan = DBSCAN(eps=max(dist), min_samples=5)
    try:
        dbscan.fit(vertices[pred])
    except:
        pdb.set_trace()
    clusters = dbscan.fit_predict(vertices[pred])

    # project cluster id to the initial all vertices
    labels = np.zeros(pred.shape) - 1   # set default to -1
    labels[np.nonzero(pred)[0]] = clusters
    labels += 1

    # rank the cluster
    scores = dict()     # key: original cluster id; value: average score
    ids, cnts = np.unique(labels, return_counts=True)
    for id, cnt in zip(ids, cnts):
        if id != 0:
            score = ((output[labels == id, 1] - output[labels == id, 0]) ** 2).sum() / cnt
            scores[id] = score

    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

    ranks = dict()  # key: original cluster id: value: new cluster id
    for i, (id, score) in enumerate(sorted_scores.items()):
        ranks[id] = i + 1

    for o_id, n_id in ranks.items():
        if n_id <= 10:
            labels[labels == o_id] = n_id * 100
        else:
            labels[labels == o_id] = 0
    labels /= 100

    vertex_colors = np.stack([colors[v] for v in labels], axis=0)
    mesh.visual.vertex_colors = vertex_colors
    mesh.export(pred_filename)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array(vertices))
    #
    # point_colors = np.stack([colors[v] for v in labels], axis=0) / 255.0
    # pcd.colors = o3d.utility.Vector3dVector(point_colors)
    # o3d.io.write_point_cloud(pred_filename, pcd)

    submit_filename = os.path.join(os.path.dirname(pred_filename).rsplit('/', 1)[0], 'submit', str(pid) + '.txt')
    save_submission(labels.astype(int), submit_filename)

    print(pid, np.unique(labels))


# def save_prediction(coord, pred, filename):
#     pred_file = filename
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(coord)
#
#     point_colors = np.stack([colors[v] for v in pred], axis=0) / 255.0
#     pcd.colors = o3d.utility.Vector3dVector(point_colors)
#     o3d.io.write_point_cloud(pred_file, pcd)


def save_submission(prediction, pred_filename):
    with open(pred_filename, 'w') as f:
        f.write('\n'.join(map(str, prediction.tolist())))


if __name__ == '__main__':
    main()
