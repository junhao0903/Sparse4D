import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln

__all__ = [
    "SparseBox3DRefinementModule",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DEncoder",
]


@POSITIONAL_ENCODING.register_module()
class SparseBox3DEncoder(BaseModule):
    def __init__(
        self,
        embed_dims,
        vel_dims=3,
        mode="add",
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        assert mode in ["add", "cat"]
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims
        self.mode = mode

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_relu_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(2, embed_dims[2])
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3])
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        if self.mode == "add":
            output = pos_feat + size_feat + yaw_feat
        elif self.mode == "cat":
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            if self.mode == "add":
                output = output + vel_feat
            elif self.mode == "cat":
                output = torch.cat([output, vel_feat], dim=-1)
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


@PLUGIN_LAYERS.register_module()
class SparseBox3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_quality_estimation=False,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    # SparseBox3DRefinementModule前向传播函数，用于对3D anchor box进行回归细化，并可选输出分类与质量评估
    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        # 将instance feature与anchor embedding相加得到融合特征
        feature = instance_feature + anchor_embed
        # 将融合特征输入MLP网络预测box残差
        output = self.layers(feature)
        # 对需要refine的状态量采用残差方式更新：预测值 + 原anchor值
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        # 如果启用yaw归一化，则对sin(yaw)和cos(yaw)做单位向量归一化
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
        # 如果输出维度大于8，说明包含速度信息
        if self.output_dim > 8:
            # 如果time_interval不是tensor则转换为tensor
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            # 将速度相关输出转置，方便计算
            translation = torch.transpose(output[..., VX:], 0, -1)
            # 根据时间间隔计算速度 = 位移 / 时间
            velocity = torch.transpose(translation / time_interval, 0, -1)
            # 将预测速度与anchor原始速度相加得到最终速度
            output[..., VX:] = velocity + anchor[..., VX:]

        # 如果需要返回分类结果
        if return_cls:
            # 确保模型包含分类分支
            assert self.with_cls_branch, "Without classification layers !!!"
            # 使用instance feature进行分类预测
            cls = self.cls_layers(instance_feature)
        else:
            # 如果不需要分类则返回None
            cls = None
        # 如果需要分类并且启用了质量估计分支
        if return_cls and self.with_quality_estimation:
            # 使用融合特征预测质量评分（例如IoU质量）
            quality = self.quality_layers(feature)
        else:
            # 如果不启用质量估计则返回None
            quality = None
        # 返回refined box、分类结果以及质量评分
        return output, cls, quality


@PLUGIN_LAYERS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = nn.Parameter(
            torch.tensor(fix_scale), requires_grad=False
        )
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    # 前向传播函数，根据anchor生成3D key points，并可根据时间和位姿变换计算历史帧的key points
    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        # 获取batch size和anchor数量
        bs, num_anchor = anchor.shape[:2]
        # 取anchor中的(w,l,h)，并通过exp恢复真实尺寸
        size = anchor[..., None, [W, L, H]].exp()
        # 根据固定比例生成基础key points（相对于box中心）
        key_points = self.fix_scale * size
        # 如果存在可学习的key point并且instance feature存在
        if self.num_learnable_pts > 0 and instance_feature is not None:
            # 通过全连接层预测可学习key points的scale
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            # 将learnable key points与固定key points拼接
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2
            )

        # 初始化旋转矩阵
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])

        # 根据cos(yaw)填充旋转矩阵
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        # 根据-sin(yaw)填充旋转矩阵
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        # 根据sin(yaw)填充旋转矩阵
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        # 根据cos(yaw)填充旋转矩阵
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        # z轴保持不变
        rotation_mat[:, :, 2, 2] = 1

        # 将key points通过旋转矩阵进行旋转
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        # 将key points平移到anchor中心位置
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        # 如果没有提供时间戳或位姿信息，则直接返回当前帧key points
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        # 用于保存各历史帧的key points
        temp_key_points_list = []
        # 从anchor中提取速度(vx,vy,vz)
        velocity = anchor[..., VX:]
        # 遍历每一个历史时间戳
        for i, t_time in enumerate(temp_timestamps):
            # 计算当前帧与历史帧的时间差
            time_interval = cur_timestamp - t_time
            # 根据速度和时间差计算目标的平移量
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )
            # 根据运动模型将key points反推到历史帧位置
            temp_key_points = key_points - translation[:, :, None]
            # 获取当前帧到历史帧的坐标变换矩阵
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            # 将key points转换到历史帧坐标系
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            # 去掉最后一维
            temp_key_points = temp_key_points.squeeze(-1)
            # 保存该历史帧的key points
            temp_key_points_list.append(temp_key_points)
        # 返回当前帧key points和历史帧key points列表
        return key_points, temp_key_points_list

    # 将anchor从源坐标系投影到目标坐标系，同时考虑目标的运动补偿
    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        # 用于存储每个目标坐标系下的anchor结果
        dst_anchors = []
        # 遍历每一个源到目标的变换矩阵
        for i in range(len(T_src2dst_list)):
            # 从anchor中提取速度(vx,vy,vz)
            vel = anchor[..., VX:]
            # 速度向量维度
            vel_dim = vel.shape[-1]
            # 将坐标变换矩阵扩展一维以匹配anchor batch维度
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            # 提取anchor中心点(x,y,z)
            center = anchor[..., [X, Y, Z]]
            # 如果提供了time_intervals，则直接使用
            if time_intervals is not None:
                time_interval = time_intervals[i]
            # 否则根据源时间戳和目标时间戳计算时间差
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            # 如果没有时间信息，则不做时间补偿
            else:
                time_interval = None
            # 如果存在时间差，则根据速度进行运动补偿
            if time_interval is not None:
                # 根据速度和时间差计算平移量
                translation = vel.transpose(0, -1) * time_interval
                # 再转回原来的维度顺序
                translation = translation.transpose(0, -1)
                # 将anchor中心点反向平移到目标时间
                center = center - translation
            # 使用旋转矩阵和平移向量将center从src坐标系转换到dst坐标系
            center = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )
            # 提取anchor尺寸(w,l,h)
            size = anchor[..., [W, L, H]]
            # 使用旋转矩阵更新yaw方向(cos_yaw, sin_yaw)
            yaw = torch.matmul(
                T_src2dst[..., :2, :2],
                anchor[..., [COS_YAW, SIN_YAW], None],
            ).squeeze(-1)
            # 使用旋转矩阵更新速度方向
            vel = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1)
            # 将新的center、size、yaw、velocity拼接成新的anchor
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            # TODO: Fix bug
            # 原本计划根据字段顺序重新排序anchor参数
            # index = [X, Y, Z, W, L, H, COS_YAW, SIN_YAW] + [VX, VY, VZ][:vel_dim]
            # 将index转换为tensor
            # index = torch.tensor(index, device=dst_anchor.device)
            # 对index进行排序
            # index = torch.argsort(index)
            # 根据index重新排列anchor维度
            # dst_anchor = dst_anchor.index_select(dim=-1, index=index)
            # 将转换后的anchor加入列表
            dst_anchors.append(dst_anchor)
        # 返回所有目标坐标系下的anchor列表
        return dst_anchors

    @staticmethod
    def distance(anchor):
        return torch.norm(anchor[..., :2], p=2, dim=-1)
