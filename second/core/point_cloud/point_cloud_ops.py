import time

import numba
import numpy as np


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j]) # 以后左下角点为基准，计算每个点在3个维度上的体素索引
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c # 0->2,1->1,2->0,实现xyz->zyx,3个维度上,点云所处体素每个维度的索引
        if failed: # 如果超出范围，则不生成点的体素索引
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]] # zyx,(1,496,432),初始值均为-1
        if voxelidx == -1: # 未遍历过该体素内的点
            voxelidx = voxel_num # 已遍历过的体素添加索引
            if voxel_num >= max_voxels: # 控制体素数量
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx # 更新体素索引(可使用体素坐标访问体素索引)
            coors[voxelidx] = coor #　使用体素索引访问体素坐标
        num = num_points_per_voxel[voxelidx] # 根据体素索引返回每个体素内部的点数
        if num < max_points:
            voxels[voxelidx, num] = points[i] # 将这个点的特征值放入体素数组中(初始为0)
            num_points_per_voxel[voxelidx] += 1 # 该体素内点数加1
    return voxel_num # 体素数量

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0] # 点的个数
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32) # 体素坐标
    voxel_num = 0 # 体素数量
    failed = False
    for i in range(N): # 逐点循环
        failed = False
        for j in range(ndim): # 逐三个方向维度循环
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break # 索引超出范围,结束维度的循环
            coor[j] = c # 找到三个方向维度体素的坐标
        if failed:
            continue # 索引超出范围,直接进入下一个点的循环
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]] # 建立体素坐标与体素索引的映射
        if voxelidx == -1: # -1为初始值
            voxelidx = voxel_num # 默认从0开始
            if voxel_num >= max_voxels: # 限制体素个数,对于新体素的点不再采样
                break
            voxel_num += 1 # 遇到初始为-1,也即新体素,体素数量加1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx # 建立体素坐标到索引的映射
            coors[voxelidx] = coor # 建立体素索引到坐标的映射
        num = num_points_per_voxel[voxelidx] # 体素内点的数量，初始默认为0
        if num < max_points:
            voxels[voxelidx, num] = points[i] # 整体体素索引以及体素点与原始点建立联系
            num_points_per_voxel[voxelidx] += 1 # 体素内点的数量加１
    return voxel_num # 返回体素数量


def points_to_voxel(points,
                     voxel_size,
                     coors_range,
                     max_points=35, # 100
                     reverse_index=True,
                     max_voxels=20000): # 12000
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size # 将体素坐标化
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist()) # 四舍五入求整,然后将数组转换成列表,最后变成元组
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1] # 倒序,z,y,x
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32) # [12000]
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32) # 根据坐标设置体素坐标索引，初始为-1,(1, 496, 432)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype) # 构建最大体素特征数据结果,初始为0(12000, 100, 4).
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32) # 体素坐标,初始为0(12000, 3)
    if reverse_index: # 以下部分不仅返回了体素数量，还生成了每个体素的坐标以及点的数量
        voxel_num = _points_to_voxel_reverse_kernel( # 体素的数量
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num] # 体素索引到体素坐标的映射
    voxels = voxels[:voxel_num] # 体素索引到体素内点特征之间的映射
    num_points_per_voxel = num_points_per_voxel[:voxel_num] # 以体素索引输出各体素内点的数量
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel # 输出体素内点特征，体素坐标，每个体素点的数量


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N, ), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices
