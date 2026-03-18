import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class Grid(object):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


# 周期性地遮挡网格区域 多相机几何一致性 强相关
# 特点：
# 有结构的遮挡
# 保留上下文信息
# 防止模型过拟合局部纹理
# 输入图像
#    ↓
# 生成周期grid mask
#    ↓
# 随机旋转mask
#    ↓
# 裁剪回原尺寸
#    ↓
# 乘到图像上
class GridMask(nn.Module):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        super(GridMask, self).__init__()
        # 是否在高度方向（横向条纹）生成mask
        self.use_h = use_h
        # 是否在宽度方向（纵向条纹）生成mask
        self.use_w = use_w
        # mask旋转的最大角度范围
        self.rotate = rotate
        # 是否在mask区域添加随机偏移（噪声）而不是直接置0
        self.offset = offset
        # 每个grid中被遮挡区域的比例
        self.ratio = ratio
        # mask模式
        # mode=0：遮挡区域为0
        # mode=1：遮挡区域为1（mask取反）
        self.mode = mode
        # 初始概率
        self.st_prob = prob
        # 当前使用概率（可以随训练动态调整）
        self.prob = prob
    # 根据训练epoch动态调整GridMask使用概率
    def set_prob(self, epoch, max_epoch):
        # 训练越往后，使用GridMask的概率越大
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        # 如果随机数大于prob 或 当前是推理模式，则不使用GridMask
        if np.random.rand() > self.prob or not self.training:
            return x
        # 获取输入tensor尺寸
        # x shape = [N, C, H, W]
        n, c, h, w = x.size()
        # reshape为 [N*C, H, W]
        # 方便后面mask扩展
        x = x.view(-1, h, w)
        # 扩大mask尺寸（为了旋转后仍然覆盖原图）
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        # 随机生成grid大小d
        # grid间距
        d = np.random.randint(2, h)
        # 计算遮挡区域宽度
        # ratio * grid_size
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        # 初始化mask为全1
        mask = np.ones((hh, ww), np.float32)
        # 随机生成高度方向起点偏移
        st_h = np.random.randint(d)
        # 随机生成宽度方向起点偏移
        st_w = np.random.randint(d)
        # 在高度方向生成mask（横向条纹）
        if self.use_h:
            for i in range(hh // d):
                # 当前遮挡起点
                s = d * i + st_h
                # 当前遮挡终点
                t = min(s + self.l, hh)
                # 置为0（遮挡）
                mask[s:t, :] *= 0
        # 在宽度方向生成mask（纵向条纹）
        if self.use_w:
            for i in range(ww // d):
                # 当前遮挡起点
                s = d * i + st_w
                # 当前遮挡终点
                t = min(s + self.l, ww)
                # 置为0（遮挡）
                mask[:, s:t] *= 0

        # 随机旋转mask
        r = np.random.randint(self.rotate)
        # 转为PIL图像
        mask = Image.fromarray(np.uint8(mask))
        # 旋转mask
        mask = mask.rotate(r)
        # 再转回numpy
        mask = np.asarray(mask)
        # 从中心裁剪回原始尺寸 (H,W)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        # 转换为torch tensor
        mask = torch.from_numpy(mask.copy()).float().cuda()
        # 如果mode=1，则mask取反
        if self.mode == 1:
            mask = 1 - mask
        # 扩展mask到x相同大小
        mask = mask.expand_as(x)
        # 如果开启offset
        if self.offset:
            # 生成[-1,1]范围随机噪声
            offset = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
                .float()
                .cuda()
            )
            # mask区域保留原值
            # 非mask区域添加随机噪声
            x = x * mask + offset * (1 - mask)
        else:
            # 直接将mask区域置0
            x = x * mask

        # reshape回原始尺寸 [N, C, H, W]
        return x.view(n, c, h, w)
