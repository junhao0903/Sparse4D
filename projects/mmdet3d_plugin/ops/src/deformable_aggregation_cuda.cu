
// 引入ATen基础库（PyTorch底层tensor库）
#include <ATen/ATen.h>
// 引入CUDA上下文
#include <ATen/cuda/CUDAContext.h>
// CUDA基础库
#include <cuda.h>
// CUDA runtime库
#include <cuda_runtime.h>

// 引入CUDA原子操作
#include <THC/THCAtomics.cuh>

// 标准输入输出
#include <iostream>
// 标准库
#include <stdlib.h>


// 双线性插值采样函数（用于在feature map上采样特征）
__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  // 获取采样点左上角像素行
  const int h_low = floorf(h_im);
  // 获取采样点左上角像素列
  const int w_low = floorf(w_im);
  // 获取右下像素行
  const int h_high = h_low + 1;
  // 获取右下像素列
  const int w_high = w_low + 1;

  // 计算采样点与左上像素的垂直距离
  const float lh = h_im - h_low;
  // 计算采样点与左上像素的水平距离
  const float lw = w_im - w_low;
  // 上侧权重
  const float hh = 1 - lh, hw = 1 - lw;

  // width方向stride
  const int w_stride = num_embeds;
  // height方向stride
  const int h_stride = width * w_stride;
  // 左上像素的行offset
  const int h_low_ptr_offset = h_low * h_stride;
  // 右下像素的行offset
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  // 左上像素列offset
  const int w_low_ptr_offset = w_low * w_stride;
  // 右上像素列offset
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  // 左上像素值
  float v1 = 0;
  // 判断是否越界
  if (h_low >= 0 && w_low >= 0) {
    // 计算左上像素地址
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    // 读取像素值
    v1 = bottom_data[ptr1];
  }
  // 右上像素值
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  // 左下像素值
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  // 右下像素值
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  // 双线性插值权重
  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  // 插值结果
  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  // 返回采样值
  return val;
}


// 双线性采样反向传播函数
__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  // 与forward一致计算周围像素
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  // 计算feature梯度
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  // 计算插值结果
  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  // 对权重求梯度
  atomicAdd(grad_weights, grad_output * val);
  // 对采样位置求梯度
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
}


// deformable aggregation前向CUDA kernel
__global__ void deformable_aggregation_kernel(
    const int num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    // 计算线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    // 获取attention权重
    const float weight = *(weights + idx / (num_embeds / num_groups));
    // 当前channel
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    // 当前scale
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    // 当前camera
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    // 当前采样点
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    // 当前anchor
    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    // 当前batch
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    // anchor在batch中的index
    anchor_index = batch_index * num_anchors + anchor_index;
    // sample location offset
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 获取归一化坐标
    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;

    // 计算camera-scale index
    int cam_scale_index = cam_index * num_scale + scale_index;
    // feature offset
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    // 获取feature map尺寸
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    // 转换到feature map坐标
    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    // 进行双线性采样并累加
    atomicAdd(
        output + anchor_index * num_embeds + channel_index,
        bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
}


// deformable aggregation反向传播CUDA kernel
__global__ void deformable_aggregation_grad_kernel(
    const int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    // 计算当前线程处理的数据索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 越界保护
    if (idx >= num_kernels) return;

    // 获取当前权重对应的索引位置
    const int weights_ptr = idx / (num_embeds / num_groups);
    // 当前channel索引
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    // 当前scale层索引
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    // 当前camera索引
    const int cam_index = idx % num_cams;
    idx /= num_cams;
    // 当前采样点索引
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    // 当前anchor索引
    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    // 当前batch索引
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    // 将anchor转换为batch全局索引
    anchor_index = batch_index * num_anchors + anchor_index;
    // 计算采样位置的offset
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 读取采样位置的归一化x坐标
    const float loc_w = sample_location[loc_offset];
    // 如果越界则跳过
    if (loc_w <= 0 || loc_w >= 1) return;
    // 读取归一化y坐标
    const float loc_h = sample_location[loc_offset + 1];
    // 如果越界则跳过
    if (loc_h <= 0 || loc_h >= 1) return;

    // 获取当前输出梯度
    const float grad = grad_output[anchor_index*num_embeds + channel_index];

    // 计算camera-scale联合索引
    int cam_scale_index = cam_index * num_scale + scale_index;
    // 计算feature map数据的偏移位置
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    // 乘2用于读取h,w
    cam_scale_index = cam_scale_index << 1;
    // 获取当前feature map高度
    const int h = spatial_shape[cam_scale_index];
    // 获取当前feature map宽度
    const int w = spatial_shape[cam_scale_index + 1];

    // 将归一化坐标转换为feature map坐标
    const float h_im = loc_h * h - 0.5;
    // 将归一化坐标转换为feature map坐标
    const float w_im = loc_w * w - 0.5;

    /* atomicAdd( */
    /*     output + anchor_index * num_embeds + channel_index, */
    /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
    /* ); */
    const float weight = weights[weights_ptr];
    // 获取权重梯度地址
    float *grad_weights_ptr = grad_weights + weights_ptr;
    // 获取采样位置梯度地址
    float *grad_location_ptr = grad_sampling_location + loc_offset;
    // 调用双线性插值反向传播函数
    bilinear_sampling_grad(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset,
        grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
}


// deformable aggregation前向函数（用于launch CUDA kernel）
void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    // 计算kernel总数量（每个线程处理一个采样点）
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    // 启动CUDA kernel
    deformable_aggregation_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
}


// deformable aggregation反向传播函数（用于launch反向kernel）
void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    // 计算kernel数量
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    // 启动反向传播CUDA kernel
    deformable_aggregation_grad_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
}
