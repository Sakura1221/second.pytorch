import pathlib
import pickle
import time
from collections import defaultdict

import numpy as np
from skimage import io as imgio

from second.core import box_np_ops
from second.core import preprocess as prep
from second.core.geometry import points_in_convex_polygon_3d_jit
from second.core.point_cloud.bev_ops import points_to_bev
from second.data import kitti_common as kitti


def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list) # 特殊的dict,如果key不存在，返回list对应的默认值[]
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v) # 将普通的dict包装成特殊dict
    ret = {}
    example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'gt_boxes', 'voxel_labels',
                'match_indices'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


def prep_pointcloud(input_dict,
                    root_path,
                    voxel_generator,
                    target_assigner,
                    db_sampler=None,
                    max_voxels=20000,
                    class_names=['Car'],
                    remove_outside_points=False,
                    training=True,
                    create_targets=True,
                    shuffle_points=False,
                    reduce_valid_area=False,
                    remove_unknown=False,
                    gt_rotation_noise=[-np.pi / 3, np.pi / 3],
                    gt_loc_noise_std=[1.0, 1.0, 1.0],
                    global_rotation_noise=[-np.pi / 4, np.pi / 4],
                    global_scaling_noise=[0.95, 1.05],
                    global_loc_noise_std=(0.2, 0.2, 0.2),
                    global_random_rot_range=[0.78, 2.35],
                    generate_bev=False,
                    without_reflectivity=False,
                    num_point_features=4,
                    anchor_area_threshold=1,
                    gt_points_drop=0.0,
                    gt_drop_max_keep=10,
                    remove_points_after_sample=True,
                    anchor_cache=None,
                    remove_environment=False,
                    random_crop=False,
                    reference_detections=None,
                    add_rgb_to_points=False,
                    lidar_input=False,
                    unlabeled_db_sampler=None,
                    out_size_factor=2,
                    min_gt_point_dict=None,
                    bev_only=False,
                    use_group_id=False,
                    out_dtype=np.float32):
    """convert point cloud to voxels, create targets if ground truths 
    exists.
    """
    # 这部分用来读取某一帧数据
    points = input_dict["points"] # velodyne_reduced, array(N*4)
    if training:
        gt_boxes = input_dict["gt_boxes"] # 真值框，位置，尺寸，绝对转角，N*1，一个真值框一行
        gt_names = input_dict["gt_names"]
        difficulty = input_dict["difficulty"]
        group_ids = None
        if use_group_id and "group_ids" in input_dict: # False
            group_ids = input_dict["group_ids"]
    rect = input_dict["rect"]
    Trv2c = input_dict["Trv2c"]
    P2 = input_dict["P2"]
    unlabeled_training = unlabeled_db_sampler is not None
    image_idx = input_dict["image_idx"]

    if reference_detections is not None: # None
        C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        frustums = box_np_ops.get_frustum_v2(reference_detections, C)
        frustums -= T
        # frustums = np.linalg.inv(R) @ frustums.T
        frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
        masks = points_in_convex_polygon_3d_jit(points, surfaces)
        points = points[masks.any(-1)]

    if remove_outside_points and not lidar_input: # False
        image_shape = input_dict["image_shape"]
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2,
                                                  image_shape)
    if remove_environment is True and training: # False
        selected = kitti.keep_arrays_by_name(gt_names, class_names)
        gt_boxes = gt_boxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        if group_ids is not None: # None
            group_ids = group_ids[selected]
        points = prep.remove_points_outside_boxes(points, gt_boxes)

    if training:
        # 先去掉真值内的DontCare
        selected = kitti.drop_arrays_by_name(gt_names, ["DontCare"]) # 去掉DontCare
        gt_boxes = gt_boxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        if group_ids is not None: # None
            group_ids = group_ids[selected]

        gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c) # 相机坐标下的真值框转换成激光雷达坐标下[xyz_lidar,w,l,h,r],一个对象一行
        if remove_unknown: # False
            remove_mask = difficulty == -1
            """
            gt_boxes_remove = gt_boxes[remove_mask]
            gt_boxes_remove[:, 3:6] += 0.25
            points = prep.remove_points_in_boxes(points, gt_boxes_remove)
            """
            keep_mask = np.logical_not(remove_mask)
            gt_boxes = gt_boxes[keep_mask]
            gt_names = gt_names[keep_mask]
            difficulty = difficulty[keep_mask]
            if group_ids is not None: # None
                group_ids = group_ids[keep_mask]
        gt_boxes_mask = np.array( # 目标类别的对象标签，布尔类型，同样一行一个对象
            [n in class_names for n in gt_names], dtype=np.bool_)
        # 下面用来对去掉DontCare的真值进行采样补充
        if db_sampler is not None: # not None
            # 数据库预处理类里的方法，返回的是该帧数据内各类别用来补充的采样真值数据，包括真值内出现的非目标类别
            sampled_dict = db_sampler.sample_all(
                root_path,
                gt_boxes,
                gt_names,
                num_point_features,
                random_crop,
                gt_group_ids=group_ids, # None
                rect=rect,
                Trv2c=Trv2c,
                P2=P2)

            if sampled_dict is not None: # 下面将原始数据与补充采样数据合并
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                # gt_names = gt_names[gt_boxes_mask].tolist()
                gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                # gt_names += [s["name"] for s in sampled]
                gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0) # 真值框是目标类别的标志
                if group_ids is not None: # None
                    sampled_group_ids = sampled_dict["group_ids"]
                    group_ids = np.concatenate([group_ids, sampled_group_ids])

                if remove_points_after_sample: # False,将采样框所占位置点云去除
                    points = prep.remove_points_in_boxes(
                        points, sampled_gt_boxes)

                points = np.concatenate([sampled_points, points], axis=0) # 合并原始点云与采样点云
        # unlabeled_mask = np.zeros((gt_boxes.shape[0], ), dtype=np.bool_)
        if without_reflectivity: # False
            used_point_axes = list(range(num_point_features))
            used_point_axes.pop(3)
            points = points[:, used_point_axes]
        pc_range = voxel_generator.point_cloud_range
        if bev_only:  # set z and h to limits, False
            gt_boxes[:, 2] = pc_range[2]
            gt_boxes[:, 5] = pc_range[5] - pc_range[2]
        prep.noise_per_object_v3_( # 每一个对象添加扰动，实际上就是修改每个对象真值框坐标与内部点云坐标
            gt_boxes, # 补充过采样数据的真值框
            points, # 补充过采样数据的点云
            gt_boxes_mask, # 真值框标志（不区分原始数据与采样数据）
            rotation_perturb=gt_rotation_noise, # 旋转角度范围，平均分布
            center_noise_std=gt_loc_noise_std, # 位置正态分布方差
            global_random_rot_range=global_random_rot_range, # 全局旋转[0.0,0.0]
            group_ids=group_ids, # None
            num_try=100)
        # should remove unrelated objects after noise per object
        gt_boxes = gt_boxes[gt_boxes_mask] # 添加扰动后筛选目标类别真值框
        gt_names = gt_names[gt_boxes_mask]
        if group_ids is not None: # Fasle
            group_ids = group_ids[gt_boxes_mask]
        gt_classes = np.array( # (gt_num)
            [class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

        gt_boxes, points = prep.random_flip(gt_boxes, points) # 随机翻转
        gt_boxes, points = prep.global_rotation( # 全局旋转
            gt_boxes, points, rotation=global_rotation_noise)
        gt_boxes, points = prep.global_scaling_v2(gt_boxes, points, # 全局缩放
                                                  *global_scaling_noise)

        # Global translation
        # 全局定位扰动
        gt_boxes, points = prep.global_translate(gt_boxes, points, global_loc_noise_std)

        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        mask = prep.filter_gt_box_outside_range(gt_boxes, bv_range) # 过滤超出鸟瞰图范围的真值框
        gt_boxes = gt_boxes[mask]
        gt_classes = gt_classes[mask]
        if group_ids is not None: # None
            group_ids = group_ids[mask]

        # limit rad to [-pi, pi]
        gt_boxes[:, 6] = box_np_ops.limit_period( # 调整真值转角至目标范围
            gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

    if shuffle_points: # True，打乱全局点云数据
        # shuffle is a little slow.
        np.random.shuffle(points)

    voxel_size = voxel_generator.voxel_size # [0.16, 0.16, 4]
    pc_range = voxel_generator.point_cloud_range # [0, -39.68, -3, 69.12, 39.68, 1]
    grid_size = voxel_generator.grid_size # [432, 496, 1] x,y,z

    # 生成体素
    """
    Returns:
        voxels:[num_voxels, 100, 4] 体素索引映射全局体素特征
        coordinates:[num_voxels, 3]　体素索引映射体素坐标
        num_points:(num_voxels,)　体素索引映射体素内的点数
    """
    voxels, coordinates, num_points = voxel_generator.generate(
        points, max_voxels)

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64)
    }
    example.update({
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
    })
    # if not lidar_input:
    feature_map_size = grid_size[:2] // out_size_factor # [216 248]
    feature_map_size = [*feature_map_size, 1][::-1] # [1,248,216]
    if anchor_cache is not None:
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]
    else:
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
    example["anchors"] = anchors
    # print("debug", anchors.shape, matched_thresholds.shape)
    # anchors_bv = anchors_bv.reshape([-1, 4])
    anchors_mask = None
    if anchor_area_threshold >= 0: # True
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask( # 某处体素是否被采样,否为0,是为1.array[496,432] y,x
            coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0) # 元素值为按列累加到元素的和
        dense_voxel_map = dense_voxel_map.cumsum(1) # 元素值为按行累加到元素的和,相当于统计了元素两个维度上之前元素的和
        anchors_area = box_np_ops.fused_get_anchors_area( # 统计每个锚框内的体素数量
            dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold # 标记超过一个体素的锚框
        # example['anchors_mask'] = anchors_mask.astype(np.uint8)
        example['anchors_mask'] = anchors_mask
    if generate_bev: # False
        bev_vxsize = voxel_size.copy()
        bev_vxsize[:2] /= 2
        bev_vxsize[2] *= 2
        bev_map = points_to_bev(points, bev_vxsize, pc_range,
                                without_reflectivity)
        example["bev_map"] = bev_map
    if not training:
        return example # 测试数据集不需要创建训练目标
    if create_targets: # True
        targets_dict = target_assigner.assign(
            anchors, # (248*216*2,7)
            gt_boxes, # (gt_num, 7)
            anchors_mask, # (248*216*2,)
            gt_classes=gt_classes, # (gt_num,)
            matched_thresholds=matched_thresholds, # (248*216*2,)
            unmatched_thresholds=unmatched_thresholds) # (248*216*2,)
        example.update({
            'labels': targets_dict['labels'], # (total_anchors,),所有锚框对应真值类别(1,2...),无对应真值设为0,dontcare设为-1
            'reg_targets': targets_dict['bbox_targets'], # (total_anchors, 7),所有锚框对应真值相对锚框的偏差编码,无对应真值设为[0,0,0,0,0,0,0]
            'reg_weights': targets_dict['bbox_outside_weights'], # (total_anchors,),所有锚框的外部权重,有对应真值设为1，无对应真值设为0
        })
    return example


def _read_and_prep_v9(info, root_path, num_point_features, prep_func):
    # info: # kitti_infos_train.pkl内的某一个列表元素（某一帧数据）
    """read data from KITTI-format infos, then call prep function.
    """
    # velodyne_path = str(pathlib.Path(root_path) / info['velodyne_path'])
    # velodyne_path += '_reduced'
    v_path = pathlib.Path(root_path) / info['velodyne_path'] # e.g. root_path/training/velodyne/000010.bin
    v_path = v_path.parent.parent / (
        v_path.parent.stem + "_reduced") / v_path.name # e.g. root_path/training/velodyne_reduced/000010.bin

    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_point_features]) #　将点云数据转化为N*4的数组
    image_idx = info['image_idx']
    rect = info['calib/R0_rect'].astype(np.float32)
    Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = info['calib/P2'].astype(np.float32)

    input_dict = {
        'points': points, # velodyne_reduced
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
        'image_shape': np.array(info["img_shape"], dtype=np.int32), # 图像尺寸
        'image_idx': image_idx, # 图像编号
        'image_path': info['img_path'], # 图像路径
        # 'pointcloud_num_features': num_point_features,
    }

    if 'annos' in info: # True
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = kitti.remove_dontcare(annos)
        loc = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        # print(gt_names, len(loc))
        gt_boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32) # 真值框，位置，尺寸，绝对转角，N*1，一个真值框一行
        # gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
        difficulty = annos["difficulty"]
        input_dict.update({
            'gt_boxes': gt_boxes, # 真值框
            'gt_names': gt_names, # 真值类别
            'difficulty': difficulty, # 难度
        })
        if 'group_ids' in annos: # True
            input_dict['group_ids'] = annos["group_ids"]
    example = prep_func(input_dict=input_dict) # 预处理传参
    example["image_idx"] = image_idx # 数据帧索引
    example["image_shape"] = input_dict["image_shape"] # 图像尺寸
    if "anchors_mask" in example: # True
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8) # 标记超过一个体素的锚框
    return example

