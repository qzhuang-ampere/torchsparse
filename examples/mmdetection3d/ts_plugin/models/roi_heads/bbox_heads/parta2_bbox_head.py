# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmengine.model import normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from ...layers.sparse_block import make_sparse_convmodule_ts
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.utils.typing_utils import InstanceList

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import (SparseConvTensor, SparseMaxPool3d,
                                SparseSequential)
else:
    from mmcv.ops import SparseConvTensor, SparseMaxPool3d, SparseSequential

from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.models.layers import nms_bev, nms_normal_bev
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d import (LiDARInstance3DBoxes,
                                        rotation_3d_in_axis, xywhr2xyxyr)
from mmdet3d.utils.typing_utils import SamplingResultList

import torchsparse
from torchsparse import SparseTensor

import os

savepath = os.environ.get("PT_SAVE_PATH")

class Flag(torch.nn.Module):
    def __init__(self, msg) -> None:
        super().__init__()
        self.msg = msg
    def forward(self, x):
        print(f"Layer: {self.msg}")
        return x

class MaxPool3DWrap(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # self.pool = nn.MaxPool3d(kernel_size, stride)
        self.pool = SparseMaxPool3d(kernel_size, stride)

    def forward(self, tensor: SparseTensor):
        sptensor = SparseConvTensor(tensor.feats, tensor.coords, tensor.spatial_range[1:4], tensor.spatial_range[0]) # tensor.spatial_range[0]
        pooled = self.pool(sptensor)
        spatial_range = (tensor.spatial_range[0],) + tuple(pooled.spatial_shape)  # tensor.spatial_range[0]  # 100 in testing
        tstensor = SparseTensor(pooled.features, pooled.indices, spatial_range=spatial_range)
        return tstensor
    
        # dense_tensor = tensor.dense()
        # pooled = self.pool(dense_tensor)

        # batch_size = pooled.shape[0]

        # # transform to sparse tensors
        # sparse_shape = pooled.shape[1:4]

        # # (non_empty_num, 4) ==> [bs_idx, x_idx, y_idx, z_idx]
        # sparse_idx = pooled.sum(dim=-1).nonzero(as_tuple=False)

        # pooled_1 = pooled[sparse_idx[:, 0], sparse_idx[:, 1],
        #                            sparse_idx[:, 2], sparse_idx[:, 3]]
        # coords = sparse_idx.int().contiguous()
        # spatial_range = (coords[:, 0].max().item() + 1,) + tuple(sparse_shape)
        # pooled_1 = torchsparse.SparseTensor(pooled_1, coords, spatial_range=spatial_range, batch_size=batch_size)
        # return  pooled_1

@MODELS.register_module("PartA2BboxHeadTS")
class PartA2BboxHeadTS(BaseModule):
    """PartA2 RoI head.

    Args:
        num_classes (int): The number of classes to prediction.
        seg_in_channels (int): Input channels of segmentation
            convolution layer.
        part_in_channels (int): Input channels of part convolution layer.
        seg_conv_channels (list(int)): Out channels of each
            segmentation convolution layer.
        part_conv_channels (list(int)): Out channels of each
            part convolution layer.
        merge_conv_channels (list(int)): Out channels of each
            feature merged convolution layer.
        down_conv_channels (list(int)): Out channels of each
            downsampled convolution layer.
        shared_fc_channels (list(int)): Out channels of each shared fc layer.
        cls_channels (list(int)): Out channels of each classification layer.
        reg_channels (list(int)): Out channels of each regression layer.
        dropout_ratio (float): Dropout ratio of classification and
            regression layers.
        roi_feat_size (int): The size of pooled roi features.
        with_corner_loss (bool): Whether to use corner loss or not.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for box head.
        conv_cfg (dict): Config dict of convolutional layers
        norm_cfg (dict): Config dict of normalization layers
        loss_bbox (dict): Config dict of box regression loss.
        loss_cls (dict, optional): Config dict of classifacation loss.
    """

    def __init__(self,
                 num_classes: int,
                 seg_in_channels: int,
                 part_in_channels: int,
                 seg_conv_channels: List[int] = None,
                 part_conv_channels: List[int] = None,
                 merge_conv_channels: List[int] = None,
                 down_conv_channels: List[int] = None,
                 shared_fc_channels: List[int] = None,
                 cls_channels: List[int] = None,
                 reg_channels: List[int] = None,
                 dropout_ratio: float = 0.1,
                 roi_feat_size: int = 14,
                 with_corner_loss: bool = True,
                 bbox_coder: dict = dict(type='DeltaXYZWLHRBBoxCoder'),
                 conv_cfg: dict = dict(type='Conv1d'),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 loss_bbox: dict = dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_cls: dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=1.0),
                 init_cfg: dict = None) -> None:
        super(PartA2BboxHeadTS, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.with_corner_loss = with_corner_loss
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_cls = MODELS.build(loss_cls)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        sparse_norm_cfg = dict(type='TorchSparseBatchNorm', eps=1e-3, momentum=0.01)

        assert down_conv_channels[-1] == shared_fc_channels[0]

        # init layers
        part_channel_last = part_in_channels
        part_conv = []
        for i, channel in enumerate(part_conv_channels):
            part_conv.append(
                make_sparse_convmodule_ts(
                    part_channel_last,
                    channel,
                    3,
                    padding=1,
                    norm_cfg=sparse_norm_cfg,
                    # indice_key=f'rcnn_part{i}',  # @Yingqi: commented it out
                    conv_type='TorchSparseConv3d'))  # @Yingqi: changed to TorchSparseConv3d
            part_channel_last = channel
        self.part_conv = nn.Sequential(*part_conv)

        seg_channel_last = seg_in_channels
        seg_conv = []
        for i, channel in enumerate(seg_conv_channels):
            seg_conv.append(
                make_sparse_convmodule_ts(
                    seg_channel_last,
                    channel,
                    3,
                    padding=1,
                    norm_cfg=sparse_norm_cfg,
                    # indice_key=f'rcnn_seg{i}',
                    conv_type='TorchSparseConv3d'))
            seg_channel_last = channel
        self.seg_conv = nn.Sequential(*seg_conv)

        self.conv_down = nn.Sequential()

        merge_conv_channel_last = part_channel_last + seg_channel_last
        merge_conv = []
        for i, channel in enumerate(merge_conv_channels):
            merge_conv.append(
                make_sparse_convmodule_ts(
                    merge_conv_channel_last,
                    channel,
                    3,
                    padding=1,
                    norm_cfg=sparse_norm_cfg))
                    # indice_key='rcnn_down0'))
            merge_conv_channel_last = channel

        down_conv_channel_last = merge_conv_channel_last
        conv_down = []
        for i, channel in enumerate(down_conv_channels):
            conv_down.append(
                make_sparse_convmodule_ts(
                    down_conv_channel_last,
                    channel,
                    3,
                    padding=1,
                    norm_cfg=sparse_norm_cfg))
                    # indice_key='rcnn_down1'))
            down_conv_channel_last = channel
        # print(self)

        #self.conv_down.add_module("flag_before_merge_conv", Flag("before merge conv"))
        self.conv_down.add_module('merge_conv', nn.Sequential(*merge_conv))
        #self.conv_down.add_module("flag_after_merge_conv", Flag("after merge conv"))
        self.conv_down.add_module('max_pool3d', MaxPool3DWrap(kernel_size=2, stride=2))
        #self.conv_down.add_module("flag_after_max_pool3d", Flag("after max pool3d"))
        self.conv_down.add_module('down_conv', nn.Sequential(*conv_down))
        #self.conv_down.add_module("flag_after_down_conv", Flag("after down conv"))

        shared_fc_list = []
        pool_size = roi_feat_size // 2
        pre_channel = shared_fc_channels[0] * pool_size**3
        for k in range(1, len(shared_fc_channels)):
            shared_fc_list.append(
                ConvModule(
                    pre_channel,
                    shared_fc_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    inplace=True))
            pre_channel = shared_fc_channels[k]

            if k != len(shared_fc_channels) - 1 and dropout_ratio > 0:
                shared_fc_list.append(nn.Dropout(dropout_ratio))

        self.shared_fc = nn.Sequential(*shared_fc_list)

        # Classification layer
        channel_in = shared_fc_channels[-1]
        cls_channel = 1
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, len(cls_channels)):
            cls_layers.append(
                ConvModule(
                    pre_channel,
                    cls_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    inplace=True))
            pre_channel = cls_channels[k]
        cls_layers.append(
            ConvModule(
                pre_channel,
                cls_channel,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                act_cfg=None))
        if dropout_ratio >= 0:
            cls_layers.insert(1, nn.Dropout(dropout_ratio))

        self.conv_cls = nn.Sequential(*cls_layers)

        # Regression layer
        reg_layers = []
        pre_channel = channel_in
        for k in range(0, len(reg_channels)):
            reg_layers.append(
                ConvModule(
                    pre_channel,
                    reg_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    inplace=True))
            pre_channel = reg_channels[k]
        reg_layers.append(
            ConvModule(
                pre_channel,
                self.bbox_coder.code_size,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                act_cfg=None))
        if dropout_ratio >= 0:
            reg_layers.insert(1, nn.Dropout(dropout_ratio))

        self.conv_reg = nn.Sequential(*reg_layers)

        if init_cfg is None:
            self.init_cfg = dict(
                type='Xavier',
                layer=['Conv2d', 'Conv1d'],
                distribution='uniform')
        print(self.conv_down)
        pass

    def init_weights(self):
        super().init_weights()
        normal_init(self.conv_reg[-1].conv, mean=0, std=0.001)

    def forward(self, seg_feats: Tensor, part_feats: Tensor) -> Tuple[Tensor]:
        """Forward pass.

        Args:
            seg_feats (torch.Tensor): Point-wise semantic features.
            part_feats (torch.Tensor): Point-wise part prediction features.

        Returns:
            tuple[torch.Tensor]: Score of class and bbox predictions.
        """
        # (B * N, out_x, out_y, out_z, 4)
        rcnn_batch_size = part_feats.shape[0]

        # transform to sparse tensors
        sparse_shape = part_feats.shape[1:4]
        # (non_empty_num, 4) ==> [bs_idx, x_idx, y_idx, z_idx]
        # (non_empty_num, 4) ==> [bs_idx, x_idx, y_idx, z_idx]
        sparse_idx = part_feats.sum(dim=-1).nonzero(as_tuple=False)

        part_features = part_feats[sparse_idx[:, 0], sparse_idx[:, 1],
                                   sparse_idx[:, 2], sparse_idx[:, 3]]
        seg_features = seg_feats[sparse_idx[:, 0], sparse_idx[:, 1],
                                 sparse_idx[:, 2], sparse_idx[:, 3]]
        coords = sparse_idx.int().contiguous()
        spatial_range = (rcnn_batch_size,) + tuple(sparse_shape)  # first index become 99??? # coords[:, 0].max().item() + 1
        part_features = torchsparse.SparseTensor(part_features, coords, spatial_range=spatial_range)
        seg_features = torchsparse.SparseTensor(seg_features, coords, spatial_range=spatial_range)

        # forward rcnn network
        x_part = self.part_conv(part_features)
        x_rpn = self.seg_conv(seg_features)

        merged_feature = torch.cat((x_rpn.feats, x_part.feats),
                                   dim=1)  # (N, C)
        shared_feature = torchsparse.SparseTensor(merged_feature, coords, spatial_range=spatial_range)  # spatial_range = spatial_range  # (100,14,14,14)
        # spatial range becomes 99 above

        x = self.conv_down(shared_feature)

        shared_feature = x.dense()  # rcnn_batch_size # shared_feature.spatial_range[0]

        N, D, H, W, C = shared_feature.shape
        shared_feature = shared_feature.permute(0,4,1,2,3).contiguous()
        shared_feature = shared_feature.view(N, C * D, H, W)
        shared_feature = shared_feature.view(rcnn_batch_size, -1, 1)
        
        shared_feature = self.shared_fc(shared_feature)

        cls_score = self.conv_cls(shared_feature).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (B, 1)
        bbox_pred = self.conv_reg(shared_feature).transpose(
            1, 2).contiguous().squeeze(dim=1)  # (B, C)
        return cls_score, bbox_pred

    def loss(self, cls_score: Tensor, bbox_pred: Tensor, rois: Tensor,
             labels: Tensor, bbox_targets: Tensor, pos_gt_bboxes: Tensor,
             reg_mask: Tensor, label_weights: Tensor,
             bbox_weights: Tensor) -> Dict:
        """Computing losses.

        Args:
            cls_score (torch.Tensor): Scores of each roi.
            bbox_pred (torch.Tensor): Predictions of bboxes.
            rois (torch.Tensor): Roi bboxes.
            labels (torch.Tensor): Labels of class.
            bbox_targets (torch.Tensor): Target of positive bboxes.
            pos_gt_bboxes (torch.Tensor): Ground truths of positive bboxes.
            reg_mask (torch.Tensor): Mask for positive bboxes.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_weights (torch.Tensor): Weights of bbox loss.

        Returns:
            dict: Computed losses.

            - loss_cls (torch.Tensor): Loss of classes.
            - loss_bbox (torch.Tensor): Loss of bboxes.
            - loss_corner (torch.Tensor): Loss of corners.
        """
        losses = dict()
        rcnn_batch_size = cls_score.shape[0]

        # calculate class loss
        cls_flat = cls_score.view(-1)
        loss_cls = self.loss_cls(cls_flat, labels, label_weights)
        losses['loss_cls'] = loss_cls

        # calculate regression loss
        code_size = self.bbox_coder.code_size
        pos_inds = (reg_mask > 0)
        if pos_inds.any() == 0:
            # fake a part loss
            losses['loss_bbox'] = loss_cls.new_tensor(0) * loss_cls.sum()
            if self.with_corner_loss:
                losses['loss_corner'] = loss_cls.new_tensor(0) * loss_cls.sum()
        else:
            pos_bbox_pred = bbox_pred.view(rcnn_batch_size, -1)[pos_inds]
            bbox_weights_flat = bbox_weights[pos_inds].view(-1, 1).repeat(
                1, pos_bbox_pred.shape[-1])
            loss_bbox = self.loss_bbox(
                pos_bbox_pred.unsqueeze(dim=0), bbox_targets.unsqueeze(dim=0),
                bbox_weights_flat.unsqueeze(dim=0))
            losses['loss_bbox'] = loss_bbox

            if self.with_corner_loss:
                pos_roi_boxes3d = rois[..., 1:].view(-1, code_size)[pos_inds]
                pos_roi_boxes3d = pos_roi_boxes3d.view(-1, code_size)
                batch_anchors = pos_roi_boxes3d.clone().detach()
                pos_rois_rotation = pos_roi_boxes3d[..., 6].view(-1)
                roi_xyz = pos_roi_boxes3d[..., 0:3].view(-1, 3)
                batch_anchors[..., 0:3] = 0
                # decode boxes
                pred_boxes3d = self.bbox_coder.decode(
                    batch_anchors,
                    pos_bbox_pred.view(-1, code_size)).view(-1, code_size)

                pred_boxes3d[..., 0:3] = rotation_3d_in_axis(
                    pred_boxes3d[..., 0:3].unsqueeze(1),
                    pos_rois_rotation,
                    axis=2).squeeze(1)

                pred_boxes3d[:, 0:3] += roi_xyz

                # calculate corner loss
                loss_corner = self.get_corner_loss_lidar(
                    pred_boxes3d, pos_gt_bboxes)
                losses['loss_corner'] = loss_corner

        return losses

    def get_targets(self,
                    sampling_results: SamplingResultList,
                    rcnn_train_cfg: dict,
                    concat: bool = True) -> Tuple[Tensor]:
        """Generate targets.

        Args:
            sampling_results (list[:obj:`SamplingResult`]):
                Sampled results from rois.
            rcnn_train_cfg (:obj:`ConfigDict`): Training config of rcnn.
            concat (bool): Whether to concatenate targets between batches.

        Returns:
            tuple[torch.Tensor]: Targets of boxes and class prediction.
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        iou_list = [res.iou for res in sampling_results]
        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            cfg=rcnn_train_cfg)

        (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
         bbox_weights) = targets

        if concat:
            label = torch.cat(label, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            pos_gt_bboxes = torch.cat(pos_gt_bboxes, 0)
            reg_mask = torch.cat(reg_mask, 0)

            label_weights = torch.cat(label_weights, 0)
            label_weights /= torch.clamp(label_weights.sum(), min=1.0)

            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_weights /= torch.clamp(bbox_weights.sum(), min=1.0)

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def _get_target_single(self, pos_bboxes: Tensor, pos_gt_bboxes: Tensor,
                           ious: Tensor, cfg: dict) -> Tuple[Tensor]:
        """Generate training targets for a single sample.

        Args:
            pos_bboxes (torch.Tensor): Positive boxes with shape
                (N, 7).
            pos_gt_bboxes (torch.Tensor): Ground truth boxes with shape
                (M, 7).
            ious (torch.Tensor): IoU between `pos_bboxes` and `pos_gt_bboxes`
                in shape (N, M).
            cfg (dict): Training configs.

        Returns:
            tuple[torch.Tensor]: Target for positive boxes.
                (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)
        """
        cls_pos_mask = ious > cfg.cls_pos_thr
        cls_neg_mask = ious < cfg.cls_neg_thr
        interval_mask = (cls_pos_mask == 0) & (cls_neg_mask == 0)

        # iou regression target
        label = (cls_pos_mask > 0).float()
        label[interval_mask] = ious[interval_mask] * 2 - 0.5
        # label weights
        label_weights = (label >= 0).float()

        # box regression target
        reg_mask = pos_bboxes.new_zeros(ious.size(0)).long()
        reg_mask[0:pos_gt_bboxes.size(0)] = 1
        bbox_weights = (reg_mask > 0).float()
        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach()
            roi_center = pos_bboxes[..., 0:3]
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi)

            # canonical transformation
            pos_gt_bboxes_ct[..., 0:3] -= roi_center
            pos_gt_bboxes_ct[..., 6] -= roi_ry
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1), -roi_ry,
                axis=2).squeeze(1)

            # flip orientation if rois have opposite orientation
            ry_label = pos_gt_bboxes_ct[..., 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (
                2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = ry_label > np.pi
            ry_label[flag] = ry_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
            pos_gt_bboxes_ct[..., 6] = ry_label

            rois_anchor = pos_bboxes.clone().detach()
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            bbox_targets = self.bbox_coder.encode(rois_anchor,
                                                  pos_gt_bboxes_ct)
        else:
            # no fg bbox
            bbox_targets = pos_gt_bboxes.new_empty((0, 7))

        return (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)

    def get_corner_loss_lidar(self,
                              pred_bbox3d: Tensor,
                              gt_bbox3d: Tensor,
                              delta: float = 1.0) -> Tensor:
        """Calculate corner loss of given boxes.

        Args:
            pred_bbox3d (torch.FloatTensor): Predicted boxes in shape (N, 7).
            gt_bbox3d (torch.FloatTensor): Ground truth boxes in shape (N, 7).
            delta (float, optional): huber loss threshold. Defaults to 1.0

        Returns:
            torch.FloatTensor: Calculated corner loss in shape (N).
        """
        assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

        # This is a little bit hack here because we assume the box for
        # Part-A2 is in LiDAR coordinates
        gt_boxes_structure = LiDARInstance3DBoxes(gt_bbox3d)
        pred_box_corners = LiDARInstance3DBoxes(pred_bbox3d).corners
        gt_box_corners = gt_boxes_structure.corners

        # This flip only changes the heading direction of GT boxes
        gt_bbox3d_flip = gt_boxes_structure.clone()
        gt_bbox3d_flip.tensor[:, 6] += np.pi
        gt_box_corners_flip = gt_bbox3d_flip.corners

        corner_dist = torch.min(
            torch.norm(pred_box_corners - gt_box_corners, dim=2),
            torch.norm(pred_box_corners - gt_box_corners_flip,
                       dim=2))  # (N, 8)
        # huber loss
        abs_error = corner_dist.abs()
        quadratic = abs_error.clamp(max=delta)
        linear = (abs_error - quadratic)
        corner_loss = 0.5 * quadratic**2 + delta * linear

        return corner_loss.mean(dim=1)

    def get_results(self,
                    rois: Tensor,
                    cls_score: Tensor,
                    bbox_pred: Tensor,
                    class_labels: Tensor,
                    class_pred: Tensor,
                    input_metas: List[dict],
                    cfg: dict = None) -> InstanceList:
        """Generate bboxes from bbox head predictions.

        Args:
            rois (torch.Tensor): Roi bounding boxes.
            cls_score (torch.Tensor): Scores of bounding boxes.
            bbox_pred (torch.Tensor): Bounding boxes predictions
            class_labels (torch.Tensor): Label of classes
            class_pred (torch.Tensor): Score for nms.
            input_metas (list[dict]): Point cloud and image's meta info.
            cfg (:obj:`ConfigDict`): Testing config.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        roi_batch_id = rois[..., 0]
        roi_boxes = rois[..., 1:]  # boxes without batch id
        batch_size = int(roi_batch_id.max().item() + 1)

        # decode boxes
        roi_ry = roi_boxes[..., 6].view(-1)
        roi_xyz = roi_boxes[..., 0:3].view(-1, 3)
        local_roi_boxes = roi_boxes.clone().detach()
        local_roi_boxes[..., 0:3] = 0
        rcnn_boxes3d = self.bbox_coder.decode(local_roi_boxes, bbox_pred)
        rcnn_boxes3d[..., 0:3] = rotation_3d_in_axis(
            rcnn_boxes3d[..., 0:3].unsqueeze(1), roi_ry, axis=2).squeeze(1)
        rcnn_boxes3d[:, 0:3] += roi_xyz

        # post processing
        result_list = []
        for batch_id in range(batch_size):
            cur_class_labels = class_labels[batch_id]
            cur_cls_score = cls_score[roi_batch_id == batch_id].view(-1)

            cur_box_prob = class_pred[batch_id]
            cur_rcnn_boxes3d = rcnn_boxes3d[roi_batch_id == batch_id]
            keep = self.multi_class_nms(cur_box_prob, cur_rcnn_boxes3d,
                                        cfg.score_thr, cfg.nms_thr,
                                        input_metas[batch_id],
                                        cfg.use_rotate_nms)
            selected_bboxes = cur_rcnn_boxes3d[keep]
            selected_label_preds = cur_class_labels[keep]
            selected_scores = cur_cls_score[keep]

            results = InstanceData()
            results.bboxes_3d = input_metas[batch_id]['box_type_3d'](
                selected_bboxes, self.bbox_coder.code_size)
            results.scores_3d = selected_scores
            results.labels_3d = selected_label_preds

            result_list.append(results)
        return result_list

    def multi_class_nms(self,
                        box_probs: Tensor,
                        box_preds: Tensor,
                        score_thr: float,
                        nms_thr: float,
                        input_meta: dict,
                        use_rotate_nms: bool = True) -> Tensor:
        """Multi-class NMS for box head.

        Note:
            This function has large overlap with the `box3d_multiclass_nms`
            implemented in `mmdet3d.core.post_processing`. We are considering
            merging these two functions in the future.

        Args:
            box_probs (torch.Tensor): Predicted boxes probabitilies in
                shape (N,).
            box_preds (torch.Tensor): Predicted boxes in shape (N, 7+C).
            score_thr (float): Threshold of scores.
            nms_thr (float): Threshold for NMS.
            input_meta (dict): Meta information of the current sample.
            use_rotate_nms (bool, optional): Whether to use rotated nms.
                Defaults to True.

        Returns:
            torch.Tensor: Selected indices.
        """
        if use_rotate_nms:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev

        assert box_probs.shape[
            1] == self.num_classes, f'box_probs shape: {str(box_probs.shape)}'
        selected_list = []
        selected_labels = []
        boxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            box_preds, self.bbox_coder.code_size).bev)

        score_thresh = score_thr if isinstance(
            score_thr, list) else [score_thr for x in range(self.num_classes)]
        nms_thresh = nms_thr if isinstance(
            nms_thr, list) else [nms_thr for x in range(self.num_classes)]
        for k in range(0, self.num_classes):
            class_scores_keep = box_probs[:, k] >= score_thresh[k]

            if class_scores_keep.int().sum() > 0:
                original_idxs = class_scores_keep.nonzero(
                    as_tuple=False).view(-1)
                cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
                cur_rank_scores = box_probs[class_scores_keep, k]

                cur_selected = nms_func(cur_boxes_for_nms, cur_rank_scores,
                                        nms_thresh[k])

                if cur_selected.shape[0] == 0:
                    continue
                selected_list.append(original_idxs[cur_selected])
                selected_labels.append(
                    torch.full([cur_selected.shape[0]],
                               k + 1,
                               dtype=torch.int64,
                               device=box_preds.device))

        keep = torch.cat(
            selected_list, dim=0) if len(selected_list) > 0 else []
        return keep
