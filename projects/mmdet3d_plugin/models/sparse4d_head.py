# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
# Sparse4D检测头，负责instance query管理、时序交互、特征更新、预测以及loss计算
    def __init__(
        self,
        instance_bank: dict,
        # instance库配置，用于管理和缓存query以及历史instance
        anchor_encoder: dict,
        # anchor位置编码器
        graph_model: dict,
        # instance之间空间关系建模的图注意力模块
        norm_layer: dict,
        # 归一化层
        ffn: dict,
        # 前馈网络
        deformable_model: dict,
        # deformable attention模块，用于从多尺度特征中采样
        refine_layer: dict,
        # bbox refinement模块
        num_decoder: int = 6,
        # decoder层数
        num_single_frame_decoder: int = -1,
        # 单帧decoder数量
        temp_graph_model: dict = None,
        # temporal graph模块，用于跨帧关系建模
        loss_cls: dict = None,
        # 分类loss
        loss_reg: dict = None,
        # 回归loss
        decoder: dict = None,
        # bbox decoder
        sampler: dict = None,
        # 正负样本采样器
        gt_cls_key: str = "gt_labels_3d",
        # GT类别字段
        gt_reg_key: str = "gt_bboxes_3d",
        # GT bbox字段
        reg_weights: List = None,
        # 回归各维度权重
        operation_order: Optional[List[str]] = None,
        # decoder内部模块执行顺序
        cls_threshold_to_reg: float = -1,
        # 分类置信度阈值，低于该值不参与回归loss
        dn_loss_weight: float = 5.0,
        # denoising loss权重
        decouple_attn: bool = True,
        # 是否使用attention解耦
        init_cfg: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        # 调用父类初始化
        self.num_decoder = num_decoder
        # decoder层数
        self.num_single_frame_decoder = num_single_frame_decoder
        # 单帧decoder数量
        self.gt_cls_key = gt_cls_key
        # GT类别key
        self.gt_reg_key = gt_reg_key
        # GT bbox key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        # 分类阈值
        self.dn_loss_weight = dn_loss_weight
        # denoising loss权重
        self.decouple_attn = decouple_attn
        # 是否使用attention解耦

        if reg_weights is None:
        # 如果未设置回归权重
            self.reg_weights = [1.0] * 10
            # 默认10维回归权重全部为1
        else:
            self.reg_weights = reg_weights
            # 使用自定义回归权重

        if operation_order is None:
        # 如果没有指定decoder操作顺序
            operation_order = [
                "temp_gnn",
                # 时序图注意力
                "gnn",
                # instance图注意力
                "norm",
                # 归一化
                "deformable",
                # deformable attention采样
                "norm",
                "ffn",
                # 前馈网络
                "norm",
                "refine",
                # bbox refinement
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
        # 每个decoder block包含这些操作
            operation_order = operation_order[3:]
            # 第一层删除gnn和norm
        self.operation_order = operation_order

        # =========== build modules ===========
        # 保存decoder操作顺序
        def build(cfg, registry):
        # 通用模块构建函数
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        # 构建instance管理模块
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        # 构建anchor位置编码器
        self.sampler = build(sampler, BBOX_SAMPLERS)
        # 构建采样器
        self.decoder = build(decoder, BBOX_CODERS)
        # 构建bbox decoder
        self.loss_cls = build(loss_cls, LOSSES)
        # 构建分类loss
        self.loss_reg = build(loss_reg, LOSSES)
        # 构建回归loss
        self.op_config_map = {
        # 不同操作对应模块配置
            "temp_gnn": [temp_graph_model, ATTENTION],
            # 时序graph attention
            "gnn": [graph_model, ATTENTION],
            # instance graph attention
            "norm": [norm_layer, NORM_LAYERS],
            # 归一化层
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            # 前馈网络
            "deformable": [deformable_model, ATTENTION],
            # deformable attention
            "refine": [refine_layer, PLUGIN_LAYERS],
            # bbox refinement
        }
        self.layers = nn.ModuleList(
        # 根据operation_order构建decoder层
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = self.instance_bank.embed_dims
        # instance特征维度
        if self.decouple_attn:
        # 如果使用attention解耦
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            # attention前扩展value维度
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
            # attention后压缩回原维度
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
    # 初始化网络权重
        for i, op in enumerate(self.operation_order):
        # 遍历decoder中的每个操作
            if self.layers[i] is None:
            # 如果该层为空则跳过
                continue
            elif op != "refine":
            # refine层通常有自己初始化方式
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                    # 对权重矩阵使用Xavier初始化
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
        # 遍历所有模块
            if hasattr(m, "init_weight"):
            # 如果模块定义了init_weight方法则调用
                m.init_weight()

    # sparse query主要体现在graph_model
    # graph_model数量很少，与anchor直接对应，只有900；key数量也很少，与历史anchor直接对应，只有600
    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
    # graph attention模块封装，用于instance之间关系建模
        if self.decouple_attn:
        # 如果启用attention解耦
            query = torch.cat([query, query_pos], dim=-1)
            # 将query与位置编码拼接
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
                # key同样拼接位置编码
            query_pos, key_pos = None, None
            # 拼接后不再单独使用pos
        if value is not None:
            value = self.fc_before(value)
            # value在attention前进行维度扩展
        return self.fc_after(
        # attention输出后再压缩回原维度
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
    # Sparse4D head前向传播
        if isinstance(feature_maps, torch.Tensor):
        # 如果输入为tensor则转为list
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        # 获取batch size

        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
        # 如果dn缓存batch不匹配则清空
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
        # 训练阶段使用denoising
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
                # 获取GT instance id
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
            # 生成denoising anchors
        if dn_metas is not None:
        # 如果存在denoising训练
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            # 解析dn信息
            num_dn_anchor = dn_anchor.shape[1]
            # dn anchor数量
            if dn_anchor.shape[-1] != anchor.shape[-1]:
            # 如果维度不一致则补零
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            # 将dn anchor拼接到正常anchor
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            # 为dn anchor添加对应feature
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            # 计算正常instance数量
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            # 初始化attention mask
            attn_mask[:num_free_instance, :num_free_instance] = False
            # 正常instance之间允许attention
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask
            # dn instance使用指定mask

        anchor_embed = self.anchor_encoder(anchor)
        # 对anchor进行位置编码
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
            # temporal anchor位置编码
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        # 保存每层bbox预测
        classification = []
        # 保存每层分类预测
        quality = []
        # 保存每层质量预测
        for i, op in enumerate(self.operation_order):
        # 按顺序执行decoder操作
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
            # 时序graph attention，关联anchor与历史anchor
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
            # instance之间graph attention,anchor自观察
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
            # 归一化或前馈网络
            # 归一化，让特征分布稳定，训练更容易收敛，避免梯度爆炸或梯度消失
            # 前馈网络，增加非线性表达能力
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
            # deformable attention从多尺度特征采样
            # 只在少量关键位置做 attention，而不是在整张特征图上做 attention
            # 假如feature_maps有40000pixels，anchor数量为900，那attn要做40000*900次计算
            # 但feature_maps并非全部pixel都和anchor相关，我们选离anchor较近的的pixel做attn
            # 如果每个anchor只有13个关键pixel，那attn只用做13*900次计算
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
            # bbox refinement模块
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                # 保存bbox预测
                classification.append(cls)
                # 保存分类预测
                quality.append(qt)
                # 保存质量预测
                if len(prediction) == self.num_single_frame_decoder:
                # 到达单帧decoder结束位置
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        # 缓存当前instance用于下一帧
        if not self.training:
        # 推理阶段
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            # 获取instance id
            output["instance_id"] = instance_id
        return output

    @force_fp32(apply_to=("model_outs"))
    # 强制将model_outs转换为FP32进行loss计算
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        # 每一层decoder输出的分类结果
        reg_preds = model_outs["prediction"]
        # 每一层decoder输出的bbox预测
        quality = model_outs["quality"]
        # 每一层decoder输出的质量预测
        output = {}
        # 保存loss结果
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
        # 遍历每一层decoder输出
            reg = reg[..., : len(self.reg_weights)]
            # 只保留需要回归的维度
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            # 根据GT进行正负样本匹配
            reg_target = reg_target[..., : len(self.reg_weights)]
            # GT bbox维度对齐
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            # 标记有效回归目标
            mask_valid = mask.clone()

            # 保存有效mask
            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            # 计算正样本数量
            if self.cls_threshold_to_reg > 0:
            # 如果设置了分类阈值
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )
                # 只有分类置信度高于阈值的样本参与回归

            cls = cls.flatten(end_dim=1)
            # 展平分类预测
            cls_target = cls_target.flatten(end_dim=1)
            # 展平分类GT
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)
            # 计算分类loss

            mask = mask.reshape(-1)
            # 展平mask
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            # 加入回归维度权重
            reg_target = reg_target.flatten(end_dim=1)[mask]
            # 选取正样本GT
            reg = reg.flatten(end_dim=1)[mask]
            # 选取正样本预测
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            # 选取正样本权重
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            # nan替换为0
            cls_target = cls_target[mask]
            # 选取正样本分类标签
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]
                # 选取正样本质量预测

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )
            # 计算bbox回归loss

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            # 保存分类loss
            output.update(reg_loss)
            # 保存回归loss

        if "dn_prediction" not in model_outs:
        # 如果没有denoising训练
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        # denoising分类预测
        dn_reg_preds = model_outs["dn_prediction"]
        # denoising回归预测

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        # 准备denoising loss所需数据
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
        # 遍历每一层decoder的denoising预测
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
            # 如果存在temporal dn
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")
                # 使用temporal dn数据

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            # 计算dn分类loss
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            # 计算dn回归loss
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            # 保存dn分类loss
            output.update(reg_loss)
            # 保存dn回归loss
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
    # 准备denoising loss所需数据
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        # 展平dn有效mask
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        # 获取dn分类GT
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        # 获取dn回归GT
        dn_pos_mask = dn_cls_target >= 0
        # 标记正样本
        dn_reg_target = dn_reg_target[dn_pos_mask]
        # 选取正样本回归GT
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        # 构建回归权重
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        # 计算dn正样本数量
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    # 推理阶段也使用FP32
    def post_process(self, model_outs, output_idx=-1):
    # 推理阶段bbox解码
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
        # 调用decoder将预测结果转换为最终bbox
