# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *


@BBOX_CODERS.register_module()
class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def decode_box(self, box):
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        box = torch.cat(
            [
                box[:, [X, Y, Z]],
                box[:, [W, L, H]].exp(),
                yaw[:, None],
                box[:, VX:],
            ],
            dim=-1,
        )
        return box

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        qulity=None,
        output_idx=-1,
    ):
        # 判断是否为tracking模式（有instance_id时通常每个query只对应一个类别）
        squeeze_cls = instance_id is not None

        # 取指定decoder层输出并进行sigmoid得到分类概率
        cls_scores = cls_scores[output_idx].sigmoid()

        # 如果是tracking模式，则对每个query取最大类别概率
        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            # 为统一shape，扩展一个类别维度
            cls_scores = cls_scores.unsqueeze(dim=-1)

        # 取对应decoder层的box预测
        box_preds = box_preds[output_idx]
        # 获取batch大小、query数量和类别数
        bs, num_pred, num_cls = cls_scores.shape
        # 将(query, class)展平后选取topk最高分
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        # 如果不是tracking模式，则根据索引恢复类别id
        if not squeeze_cls:
            cls_ids = indices % num_cls
        # 如果设置了score阈值，则生成过滤mask
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        # 如果存在quality分支（例如centerness）
        if qulity is not None:
            # 取centerness预测
            centerness = qulity[output_idx][..., CNS]
            # 根据query索引获取对应centerness
            centerness = torch.gather(centerness, 1, indices // num_cls)
            # 保存原始分类分数
            cls_scores_origin = cls_scores.clone()
            # 使用centerness对分类分数进行加权
            cls_scores *= centerness.sigmoid()
            # 按新的分数重新排序
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            # 同步更新类别id
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            # 同步更新mask
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            # 同步更新indices
            indices = torch.gather(indices, 1, idx)

        # 初始化输出列表
        output = []
        # 遍历batch
        for i in range(bs):
            # 获取当前batch的类别id
            category_ids = cls_ids[i]
            # tracking模式下需要根据indices重新排序类别
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            # 获取当前batch的score
            scores = cls_scores[i]
            # 根据indices获取对应的box预测
            box = box_preds[i, indices[i] // num_cls]
            # 如果设置了score阈值，则过滤低分结果
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            # 如果存在quality分支，获取原始分类分数
            if qulity is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            # 将网络输出的box参数解码为真实3D框
            box = self.decode_box(box)
            # 保存当前batch的检测结果
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
            # 如果有quality分支，额外保存原始分类分数
            if qulity is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            # 如果存在instance id，则输出tracking id
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids
        # 返回最终结果
        return output
