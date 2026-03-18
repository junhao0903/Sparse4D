import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
# InstanceBank用于管理Sparse4D中的instance query，包括初始化anchor、缓存历史instance、时序更新以及instance id管理
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(InstanceBank, self).__init__()
        # 调用父类初始化
        self.embed_dims = embed_dims
        # instance feature维度
        self.num_temp_instances = num_temp_instances
        # 用于temporal建模的instance数量
        self.default_time_interval = default_time_interval
        # 默认时间间隔
        self.confidence_decay = confidence_decay
        # temporal instance置信度衰减系数
        self.max_time_interval = max_time_interval
        # 最大允许时间间隔

        if anchor_handler is not None:
        # 如果存在anchor处理模块
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            # 构建anchor_handler模块
            assert hasattr(anchor_handler, "anchor_projection")
            # 确保存在anchor_projection函数
        self.anchor_handler = anchor_handler
        # 保存anchor处理模块
        if isinstance(anchor, str):
        # 如果anchor是文件路径
            anchor = np.load(anchor)
            # 从文件加载anchor
        elif isinstance(anchor, (list, tuple)):
        # 如果anchor是list或tuple
            anchor = np.array(anchor)
            # 转换为numpy数组
        self.num_anchor = min(len(anchor), num_anchor)
        # 最终anchor数量
        anchor = anchor[:num_anchor]
        # 截取anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        # anchor作为可学习参数
        self.anchor_init = anchor
        # 保存anchor初始值
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        # 初始化instance feature参数
        self.reset()
        # 初始化缓存状态

    def init_weight(self):
    # 初始化权重
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        # 恢复anchor初始值
        if self.instance_feature.requires_grad:
        # 如果instance feature可训练
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)
            # 使用Xavier初始化instance feature

    def reset(self):
    # 重置缓存信息
        self.cached_feature = None
        # 缓存的instance feature
        self.cached_anchor = None
        # 缓存的anchor
        self.metas = None
        # 缓存的meta信息
        self.mask = None
        # temporal有效mask
        self.confidence = None
        # instance置信度
        self.temp_confidence = None
        # 临时置信度
        self.instance_id = None
        # instance id
        self.prev_id = 0
        # id计数器

    def get(self, batch_size, metas=None, dn_metas=None):
    # 获取当前batch的instance feature和anchor
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        # 将learnable instance feature扩展到batch维度
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        # 将anchor扩展到batch维度

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
        # 如果存在历史缓存
            history_time = self.metas["timestamp"]
            # 历史时间戳
            time_interval = metas["timestamp"] - history_time
            # 当前帧与历史帧时间差
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            # 转换数据类型
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            # 判断时间差是否有效
            if self.anchor_handler is not None:
            # 如果存在anchor处理模块
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                # 计算历史帧到当前帧的坐标变换
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                # 将历史anchor投影到当前帧

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
            # 如果存在denoising anchor
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                # 获取dn anchor数量
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                # 投影dn anchor到当前帧
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
                # 恢复dn anchor形状
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
            # 处理无效时间间隔
        else:
        # 如果没有历史缓存
            self.reset()
            # 重置缓存
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )
            # 使用默认时间间隔

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )
        # 返回instance feature、anchor以及历史缓存

    def update(self, instance_feature, anchor, confidence):
    # 使用历史instance更新当前instance
        if self.cached_feature is None:
        # 如果没有历史instance
            return instance_feature, anchor

        num_dn = 0
        # dn instance数量
        if instance_feature.shape[1] > self.num_anchor:
        # 如果存在dn instance
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            # 提取dn feature
            dn_anchor = anchor[:, -num_dn:]
            # 提取dn anchor
            instance_feature = instance_feature[:, : self.num_anchor]
            # 截取正常instance
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        # 当前帧需要保留的instance数量
        confidence = confidence.max(dim=-1).values
        # 计算instance置信度
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        # 选择置信度最高的instance
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        # 拼接历史instance feature
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        # 拼接历史anchor
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        # 根据时间mask更新instance feature
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        # 根据时间mask更新anchor
        if self.instance_id is not None:
        # 如果存在instance id
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )
            # 对无效instance id置为-1

        if num_dn > 0:
        # 如果存在dn instance
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            # 恢复dn feature
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            # 恢复dn anchor
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
    # 缓存当前帧instance用于下一帧temporal建模
        if self.num_temp_instances <= 0:
        # 如果不使用temporal instance
            return
        instance_feature = instance_feature.detach()
        # 断开梯度
        anchor = anchor.detach()
        # 断开梯度
        confidence = confidence.detach()
        # 断开梯度

        self.metas = metas
        # 保存meta信息
        confidence = confidence.max(dim=-1).values.sigmoid()
        # 计算instance置信度
        if self.confidence is not None:
        # 如果存在历史置信度
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
            # 历史instance置信度衰减
        self.temp_confidence = confidence
        # 保存临时置信度

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
        # 选取topk instance作为temporal memory

    def get_instance_id(self, confidence, anchor=None, threshold=None):
    # 生成instance id用于tracking
        confidence = confidence.max(dim=-1).values.sigmoid()
        # 计算分类置信度
        instance_id = confidence.new_full(confidence.shape, -1).long()
        # 初始化id为-1

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
        # 如果存在历史instance id
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id
            # 继承历史id

        mask = instance_id < 0
        # 标记未分配id的instance
        if threshold is not None:
        # 如果设置置信度阈值
            mask = mask & (confidence >= threshold)
            # 只给高置信度instance分配id
        num_new_instance = mask.sum()
        # 新instance数量
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        # 生成新id
        instance_id[torch.where(mask)] = new_ids
        # 分配id
        self.prev_id += num_new_instance
        # 更新id计数器
        if self.num_temp_instances > 0:
        # 如果使用temporal instance
            self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
    # 更新temporal instance id
        if self.temp_confidence is None:
        # 如果没有临时置信度
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
            # 如果是多类别置信度
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
        # 使用缓存置信度
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        # 选取topk instance id
        instance_id = instance_id.squeeze(dim=-1)
        # 去除多余维度
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
        # padding到num_anchor长度
