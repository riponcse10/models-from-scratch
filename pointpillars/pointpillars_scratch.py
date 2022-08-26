from dataloader_kitti.dataloader_pytorch import Dataset3D, collate
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
# from open3d.ml.torch.ops import voxelize, ragged_to_dense
from mmdet3d.ops import Voxelization
from mmcv.ops import nms_bev
import torch.nn.functional as F
import numpy as np
from anchors import Anchors, anchor_target, anchors2bboxes
from utils import limit_period, keep_bbox_from_lidar_range
from vis_o3d import vis_pc

root = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti"
split = "training"

dataset = Dataset3D(root, split)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate)

# for data in dataloader:
#     print(data.keys())
#     break


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size,point_cloud_range, max_num_points, max_voxels)

    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            res_voxels, res_coors, res_num_points = self.voxel_layer(pts)
            # res_voxels : (num_voxels, max_num_points_in_a_voxel, channel) -> initially 4
            # res_coors : (num_voxels, 3)
            # res_num_points : (num_voxels,)
            pillars.append(res_voxels)
            coors.append(res_coors)
            npoints_per_pillar.append(res_num_points)
            # print(res_voxels.size(), res_coors.size(), res_num_points.size())

        pillars = torch.cat(pillars, dim=0)  # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)  # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)  # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = voxel_size[0]/2 + point_cloud_range[0]
        self.y_offset = voxel_size[1]/2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

        print("X-l y_l",self.x_l, self.y_l)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
                pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
                coors_batch: (p1 + p2 + ... + pb, 1 + 3)
                npoints_per_pillar: (p1 + p2 + ... + pb, )
                return:  (bs, out_channel, y_l, x_l)
                '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar) - 3 features
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:,
                                                                                                   None,
                                                                                                   None]  # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center - 2 features
        x_offset_pi_center = pillars[:, :, :1] - (
                    coors_batch[:, None, 1:2] * self.vx + self.x_offset)  # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (
                    coors_batch[:, None, 2:3] * self.vy + self.y_offset)  # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder - total: 4+3+2 = 9 features
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center],
                             dim=-1)  # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center  # tmp
        features[:, :, 1:2] = y_offset_pi_center  # tmp
        # In consitent with mmdet3d.
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)  # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]  # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous()  # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0]  # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]
            # print(cur_features.size())

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1].long(), cur_coors[:, 2].long()] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, in_channel, self.y_l, self.x_l)

        print(batched_canvas.size())
        return batched_canvas

class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2,2,2]):
        super().__init__()
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _  in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
                x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
                return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
                '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)

        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], out_channels[i], upsample_strides[i],
                                                    stride=upsample_strides[i], bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors=6, n_classes=3):
        super().__init__()
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return:
              bbox_cls_pred: (bs, n_anchors*3, 248, 216)
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class PointPillars(nn.Module):
    def __init__(self, nclasses, voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points = 32,
                 max_voxels=(16000, 40000)):
        super().__init__()
        self.nclasses = nclasses
        self.voxel_size = voxel_size
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points, max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, point_cloud_range=point_cloud_range,
                                            in_channel=9, out_channel=64)
        self.backbone = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3,5,5])
        self.neck = Neck(in_channels=[64, 128, 256], upsample_strides=[1,2,4], out_channels=[128,128,128])
        self.head = Head(in_channel=384)

        #anchors
        ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                  [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                  [0, -39.68, -1.78, 69.12, 39.68, -1.78]]

        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations = [0, 1.57]

        self.anchors_generator = Anchors(ranges=ranges,
                                         sizes=sizes,
                                         rotations=rotations)

        # train
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        # val and test
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216)
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return:
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, )
        '''
        # 0. pre-process
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)

        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1)  # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]

            # 3.2 nms core
            keep_inds = nms_bev(boxes=cur_bbox_pred2d,
                                 scores=cur_bbox_cls_pred,
                                 thresh=self.nms_thr,
                                 pre_max_size=None,
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(
                cur_bbox_pred)  # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result

    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216)
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return:
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ]
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i],
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i],
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results


    def forward(self, batched_pts, mode="test", batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        # xs:  [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        xs = self.backbone(pillar_features)

        # x: (bs, 384, 248, 216)
        x = self.neck(xs)

        # bbox_cls_pred: (bs, n_anchors*3, 248, 216)
        # bbox_pred: (bs, n_anchors*7, 248, 216)
        # bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # print("bbox pred ",bbox_pred.size())

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred,
                                            bbox_pred=bbox_pred,
                                            bbox_dir_cls_pred=bbox_dir_cls_pred,
                                            batched_anchors=batched_anchors)

        # anchor_target_dict = anchor_target(batched_anchors=batched_anchors,
        #                                    batched_gt_bboxes=batched_gt_bboxes,
        #                                    batched_gt_labels=batched_gt_labels,
        #                                    assigners=self.assigners,
        #                                    nclasses=self.nclasses)

        # print(results)
        return results


def train():
    pointpillars = PointPillars(3).cuda()

    epochs = 1

    for epoch in range(epochs):
        for i, data_dict in enumerate(tqdm(dataloader)):
            for key in data_dict.keys():
                # print(key)
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].cuda()

            batched_pts = data_dict['batched_pts']
            batched_gt_boxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            pointpillars(batched_pts=batched_pts, mode="train",
                         batched_gt_bboxes=batched_gt_boxes, batched_gt_labels=batched_labels)

            break

def test():
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
    model = PointPillars(nclasses=3).cuda()
    # print(model)
    model.load_state_dict(torch.load("hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth"), strict=False)
    for i, data_dict in enumerate(dataloader):
        print(data_dict.keys())
        pts = data_dict['batched_pts']
        gt_label = data_dict['batched_labels']
        print("gt label",gt_label)
        pc_torch = pts[0].cuda()
        result_filter = model(batched_pts=[pc_torch],
                              mode='test')[0]
        result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']
        print(labels, scores)
        # vis_pc(pts[0].numpy(), bboxes=lidar_bboxes, labels=labels)
        break

test()

