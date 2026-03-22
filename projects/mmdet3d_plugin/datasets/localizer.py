import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.interpolate as interp


class Localizer:
    def __init__(self, csv_file):
        self.tfs = dict()
        with open(csv_file, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            # 遍历csvreader对象的每一行内容并输出
            is_header = True  # 第一行表头不要
            for row in csvreader:
                if is_header:
                    is_header = False
                    continue
                stamp = int(row[0])
                # position = np.array([float(row[4]), float(row[5]), float(row[6])])
                position = np.array([float(row[4]), float(row[5]), float(0.0)])  # 高度方向上不准
                euler = np.array([float(row[13]), float(row[14]), float(row[15])])
                # !!!此处改成zyx会导致carla数据多帧对不齐，因为carla 的ros解析用的是四元数xyz顺序转欧拉，此处也要用xyz转四元数，可视我们的数据明明是zyx，但是此处用xyz，多帧也对得齐
                quat = R.from_euler('xyz',euler).as_quat()
                self.tfs[stamp] = {'position': position, 'quaternion': quat}

        self.expand(expand_time=100)
        self.time_start = min(self.tfs.keys())
        self.time_end = max(self.tfs.keys())

        # 提取时间戳和对应的值
        timestamps = list(self.tfs.keys())
        positions = [self.tfs[ts]['position'] for ts in timestamps]
        quaternions = [self.tfs[ts]['quaternion'] for ts in timestamps]
        # 对时间和位置进行插值
        self.f_position = interp.interp1d(timestamps, positions, axis=0, kind='linear')
        # 对时间和四元数进行插值（注意：四元数插值需要特殊处理，这里简化为线性插值，但通常应使用球面线性插值）
        self.f_quaternion = interp.interp1d(timestamps, quaternions, axis=0, kind='linear')

    def expand(self, expand_time=100):
        '''
        左右两侧扩展一些，防止时间戳越界
        '''
        # 需要插入的新时间戳
        new_timestamp_left = min(self.tfs.keys()) - expand_time
        new_timestamp_right = max(self.tfs.keys()) + expand_time

        # 对时间戳进行排序并获取列表
        timestamps = sorted(self.tfs.keys())

        # 简单的线性插值函数
        def linear_interpolate(x0, x1, y0, y1, x):
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

            # 插值数据

        interpolated_data = {}

        # 左侧插值

        left_idx = timestamps.index(min(timestamps))
        x0, x1 = timestamps[left_idx], timestamps[left_idx + 1] if left_idx + 1 < len(timestamps) else timestamps[
            left_idx]  # 注意这里可能需要处理边界情况
        y0_pos, y1_pos = self.tfs[x0]['position'], self.tfs[x1]['position'] if left_idx + 1 < len(timestamps) else \
        self.tfs[x0][
            'position']
        y0_quat, y1_quat = self.tfs[x0]['quaternion'], self.tfs[x1]['quaternion'] if left_idx + 1 < len(timestamps) else \
            self.tfs[x0]['quaternion']

        interpolated_data[new_timestamp_left] = {
            'position': np.array(
                [linear_interpolate(x0, x1, yi, yj, new_timestamp_left) for yi, yj in zip(y0_pos, y1_pos)]),
            'quaternion': np.array(
                [linear_interpolate(x0, x1, yi, yj, new_timestamp_left) for yi, yj in zip(y0_quat, y1_quat)])
        }

        # 右侧插值类似，但注意索引和边界条件
        right_idx = timestamps.index(max(timestamps))
        x0, x1 = timestamps[right_idx - 1] if right_idx - 1 >= 0 else timestamps[right_idx], timestamps[right_idx]
        y0_pos, y1_pos = self.tfs[x0]['position'], self.tfs[x1]['position']
        y0_quat, y1_quat = self.tfs[x0]['quaternion'], self.tfs[x1]['quaternion']

        interpolated_data[new_timestamp_right] = {
            'position': np.array(
                [linear_interpolate(x0, x1, yi, yj, new_timestamp_right) for yi, yj in zip(y0_pos, y1_pos)]),
            'quaternion': np.array(
                [linear_interpolate(x0, x1, yi, yj, new_timestamp_right) for yi, yj in zip(y0_quat, y1_quat)])
        }

        # 更新原始数据字典
        self.tfs.update(interpolated_data)
        self.tfs = dict(sorted(self.tfs.items()))
        pass

    def view(self):
        import matplotlib.pyplot as plt
        timestamps = list(self.tfs.keys())
        positions = np.array([self.tfs[ts]['position'] for ts in timestamps])
        quaternions = np.array([self.tfs[ts]['quaternion'] for ts in timestamps])
        # 可视化轨迹
        plt.figure()
        plt.plot(positions[:, 0], positions[:, 1], label='Trajectory', marker='o')
        # 设置图形标签和标题
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Trajectory and Steering Visualization (Top View)')
        plt.legend()
        plt.grid(True)

        # 显示比例尺
        plt.gca().set_aspect('equal', adjustable='box')

        # 显示图形
        plt.show()

    def get_tf(self, stamp):
        if stamp < self.time_start or stamp > self.time_end:
            if self.time_start - 5 < stamp and stamp < self.time_end + 5:
                stamp = self.time_start  # 微调一下，防止报错
            else:
                print('Invalid stamp. Start:{}, End:{}, but input {}'.format(self.time_start, self.time_end, stamp))
                raise ValueError
        # 获取插值结果
        interpolated_position = self.f_position(stamp).reshape(1, -1)  # 确保形状正确
        interpolated_quaternion = self.f_quaternion(stamp).reshape(1, -1)  # 确保形状正确

        rotation_matrix = R.from_quat(interpolated_quaternion).as_matrix().reshape(3, 3)
        rt_mat = np.eye(4)
        rt_mat[:3, :3] = rotation_matrix
        rt_mat[:3, 3] = interpolated_position
        return rt_mat

    def pc_to_local(self, pc, stamp):
        pc = pc.astype(np.float32)
        pc4 = np.concatenate((pc[:, :3], np.ones(shape=(len(pc), 1))), axis=1)
        rt_mat = self.get_tf(stamp)
        # 对点云进行旋转和平移
        pc4 = (pc4 @ rt_mat.T)
        pc[:, :3] = pc4[:, :3]
        return pc


if __name__ == "__main__":
    csv_file = '/home/adt/bags/work_space/datasets/outdoor4_2024-11-14-16-56-14/localization/localization.csv'
    localizer = Localizer(csv_file=csv_file)
    # localizer.view()
    # mat = localizer.get_tf(1685296485760)
    #
    # pc = np.array([[0,0,0,1]])
    # pc2 = localizer.pc_to_local(pc=pc,stamp=1685296485760)
    localizer.view()

