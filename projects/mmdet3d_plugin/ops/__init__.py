import torch

from .deformable_aggregation import DeformableAggregationFunction


def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def feature_maps_format(feature_maps, inverse=False):
    # 该函数用于在两种特征表示之间转换：一种是多相机多尺度的feature map列表结构，另一种是Sparse4D内部用于高效索引的扁平化列特征结构
    # 之所以需要这种转换，是因为DeformableFeatureAggregation在采样特征时需要连续存储的token形式，方便通过index快速访问
    if inverse:
        # inverse=True表示执行反向操作：将Sparse4D内部使用的列特征结构恢复为原始的多相机多尺度feature map结构
        col_feats, spatial_shape, scale_start_index = feature_maps
        # col_feats: 展平后的特征token，shape通常为(B, CAM*sum(HW), C)
        # spatial_shape: 每个相机每个尺度的空间尺寸(H,W)
        # scale_start_index: 每个尺度在col_feats中的起始位置索引
        num_cams, num_levels = spatial_shape.shape[:2]
        # num_cams表示相机数量，例如nuScenes一般为6
        # num_levels表示FPN输出的尺度数量

        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        # 计算每个尺度feature map的像素数量(H*W)，用于后续拆分token
        split_size = split_size.cpu().numpy().tolist()
        # 转换为python list以便后续split操作

        idx = 0
        cam_split = [1]
        # cam_split记录连续相机组的数量，如果相机特征shape相同则归为同一组
        cam_split_size = [sum(split_size[0])]
        # cam_split_size记录每个相机组对应的token数量
        for i in range(num_cams - 1):
            # 遍历所有相机判断是否需要创建新的相机组
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                # 如果相邻相机的feature shape不同，则需要新建一个相机组
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            # 当前组相机数量+1
            cam_split_size[-1] += sum(split_size[i + 1])
            # 当前组token数量累加
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]
        # 根据cam_split_size拆分col_feats，并用unflatten恢复camera维度
        # 得到每个相机组对应的特征结构

        spatial_shape = spatial_shape.cpu().numpy().tolist()
        # 转为list方便后续访问
        mc_ms_feat = []
        # mc_ms_feat用于存储最终恢复的multi-camera multi-scale特征
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            # 遍历每个相机组
            feat = list(feat.split(split_size[shape_index], dim=2))
            # 按每个scale的token数量拆分
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                # 将token恢复为(H,W)空间结构
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
                # 调整维度顺序为(B, CAM, C, H, W)，符合CNN特征格式
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
            # 更新当前处理到的相机索引
        return mc_ms_feat
        # 返回恢复后的multi-camera multi-scale feature maps

    if isinstance(feature_maps[0], (list, tuple)):
        # 如果输入是嵌套结构（例如多帧特征），则递归处理每一项
        formated = [feature_maps_format(x) for x in feature_maps]
        # 对每个元素执行相同的格式化过程
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        # 将所有列特征拼接
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        # 拼接所有空间shape信息
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        # 拼接scale起始索引
        return [col_feats, spatial_shape, scale_start_index]
        # 返回统一格式

    bs, num_cams = feature_maps[0].shape[:2]
    # 获取batch size和相机数量
    spatial_shape = []
    # 用于记录每个尺度feature map的(H,W)

    col_feats = []
    # 用于存储展平后的特征token
    for i, feat in enumerate(feature_maps):
        # 遍历每个FPN尺度特征
        spatial_shape.append(feat.shape[-2:])
        # 保存当前尺度的(H,W)
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )
        # 将(B,CAM,C,H,W) reshape为(B,CAM,C,H*W)，使每个像素位置变成一个token

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    # 先在最后维度拼接所有scale的token
    # permute调整为(B,CAM,HW,C)
    # flatten合并CAM维度得到(B,CAM*HW,C)
    spatial_shape = [spatial_shape] * num_cams
    # 每个相机共享相同的尺度结构
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    # 转为tensor方便后续GPU运算
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    # 计算每个scale对应的token数量(H*W)
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    # 计算累计token数得到每个scale结束位置
    # [0, 64*176, 64*176+32*88, 64*176+32*88+16*44,
    # 64*176+32*88+16*44+8*22, (64*176+32*88+16*44+8*22)+64*176, ...]
    # size [4*6]
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    # 向右平移得到每个scale的起始位置
    scale_start_index = scale_start_index.reshape(num_cams, -1)
    # reshape为(camera, level)结构

    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    # 返回Sparse4D内部使用的统一特征格式
    # col_feats: 展平token特征
    # spatial_shape: 每个相机每个scale的(H,W)
    # scale_start_index: 每个scale在token序列中的起始索引
    return feature_maps
