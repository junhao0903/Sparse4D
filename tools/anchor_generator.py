import numpy as np
from sklearn.cluster import KMeans
import mmcv

from projects.mmdet3d_plugin.core.box3d import *

# 和传统的grid anchor不一样，kmeans anchor是数据驱动（和数据集相关），模型一开始就知道哪里更可能出现目标
# x、y、z用的是聚类位置，w、l、h用真值平均尺寸
# 传统的grid anchor是人工设计，数量远比kmeans anchor要多
# 之所以采用kmeans anchor是因为Sparse4D思想是少量高质量 query，而不是dense anchor，所以anchor尽量接近真实目标
def get_kmeans_anchor(
    ann_file,
    num_anchor=900,
    detection_range=55,
    output_file_name="nuscenes_kmeans900.npy",
    verbose=False,
):
    # 从ann_file加载标注文件（通常是nuscenes预处理后的pkl文件）
    data = mmcv.load(ann_file, file_format="pkl")
    # 将所有frame中的gt_boxes拼接成一个大的numpy数组
    gt_boxes = np.concatenate([x["gt_boxes"] for x in data["infos"]], axis=0)
    # 计算每个3D box中心点到原点的欧式距离
    distance = np.linalg.norm(gt_boxes[:, :3], axis=-1, ord=2)
    # 只保留在检测范围以内的目标
    mask = distance <= detection_range
    # 根据mask过滤gt_boxes
    gt_boxes = gt_boxes[mask]
    # 初始化KMeans聚类器，聚类中心数量为num_anchor
    clf = KMeans(n_clusters=num_anchor, verbose=verbose)
    # 打印提示信息，开始进行KMeans聚类
    print("===========Starting kmeans, please wait.===========")
    # 使用GT box的(x,y,z)位置进行KMeans聚类
    clf.fit(gt_boxes[:, [X, Y, Z]])
    # 初始化anchor数组，每个anchor包含11个参数
    anchor = np.zeros((num_anchor, 11))
    # 将KMeans聚类得到的中心作为anchor的(x,y,z)
    anchor[:, [X, Y, Z]] = clf.cluster_centers_
    # 将GT box的平均尺寸(w,l,h)取log后作为anchor尺寸
    anchor[:, [W, L, H]] = np.log(gt_boxes[:, [W, L, H]].mean(axis=0))
    # 初始化anchor的yaw方向为cos(yaw)=1（即默认朝向0度）
    anchor[:, COS_YAW] = 1
    # 将生成的anchor保存为npy文件
    np.save(output_file_name, anchor)
    # 打印完成信息和保存路径
    print(f"===========Done! Save results to {output_file_name}.===========")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="anchor kmeans")
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--num_anchor", type=int, default=900)
    parser.add_argument("--detection_range", type=float, default=55)
    parser.add_argument(
        "--output_file_name", type=str, default="_nuscenes_kmeans900.npy"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    get_kmeans_anchor(
        args.ann_file,
        args.num_anchor,
        args.detection_range,
        args.output_file_name,
        args.verbose,
    )
