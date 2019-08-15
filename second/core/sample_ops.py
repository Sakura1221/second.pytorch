import pathlib
import pickle
import time
from functools import partial, reduce

import numpy as np
from skimage import io as imgio

from second.core import preprocess as prep
from second.core import box_np_ops
from second.data import kitti_common as kitti
import copy

from second.utils.check import shape_mergeable

class DataBaseSamplerV2:
    def __init__(self, db_infos, groups, db_prepor=None,
                 rate=1.0, global_rot_range=None):
        for k, v in db_infos.items():
            print(f"load {len(v)} {k} database infos")

        if db_prepor is not None:
            db_infos = db_prepor(db_infos) # 数据库信息预处理
            print("After filter database:")
            for k, v in db_infos.items():
                print(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos # 预处理之后的数据库信息
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._use_group_sampling = False  # slower
        if any([len(g) > 1 for g in groups]):
            self._use_group_sampling = True
        if not self._use_group_sampling:
            self._group_db_infos = self.db_infos  # 未使用组内采样，保持原数据库信息
            for group_info in groups:
                group_names = list(group_info.keys())
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
        else:
            for group_info in groups:
                group_dict = {}
                group_names = list(group_info.keys())
                group_name = ", ".join(group_names)
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
                self._group_name_to_names.append((group_name, group_names))
                # self._group_name_to_names[group_name] = group_names
                for name in group_names:
                    for item in db_infos[name]:
                        gid = item["group_id"]
                        if gid not in group_dict:
                            group_dict[gid] = [item]
                        else:
                            group_dict[gid] += [item]
                if group_name in self._group_db_infos:
                    raise ValueError("group must be unique")
                group_data = list(group_dict.values())
                self._group_db_infos[group_name] = group_data
                info_dict = {}
                if len(group_info) > 1:
                    for group in group_data:
                        names = [item["name"] for item in group]
                        names = sorted(names)
                        group_name = ", ".join(names)
                        if group_name in info_dict:
                            info_dict[group_name] += 1
                        else:
                            info_dict[group_name] = 1
                print(info_dict)


        self._sampler_dict = {}
        for k, v in self._group_db_infos.items(): # dbinfos内，k:不同类别,v:每个类别内的对象及信息
            self._sampler_dict[k] = prep.BatchSampler(v, k) # 每个类别与对应批采样类组成字典
        self._enable_global_rot = False
        if global_rot_range is not None: # [0.0,0.0]
            if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
                global_rot_range = [-global_rot_range, global_rot_range]
            else:
                assert shape_mergeable(global_rot_range, [2])
            if np.abs(global_rot_range[0] -
                        global_rot_range[1]) >= 1e-3: # False
                self._enable_global_rot = True
        self._global_rot_range = global_rot_range # [0.0, 0.0]

    @property
    def use_group_sampling(self):
        return self._use_group_sampling

    def sample_all(self,
                   root_path,
                   gt_boxes,
                   gt_names,
                   num_point_features,
                   random_crop=False,
                   gt_group_ids=None,
                   rect=None,
                   Trv2c=None,
                   P2=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self._sample_classes, # 组内采样类别与添加的个数,Car,15
                                              self._sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names])) # 真值有n个训练对象，需要采样15-n个
            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num # {类别：需要采样的数量}
            sample_num_per_class.append(sampled_num) # 每个类别需要采样的数量

        sampled_groups = self._sample_classes # Car
        if self._use_group_sampling: # False
            assert gt_group_ids is not None
            sampled_groups = []
            sample_num_per_class = []
            for group_name, class_names in self._group_name_to_names:
                sampled_nums_group = [sampled_num_dict[n] for n in class_names]
                sampled_num = np.max(sampled_nums_group)
                sample_num_per_class.append(sampled_num)
                sampled_groups.append(group_name)
            total_group_ids = gt_group_ids
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes # 避免采样碰撞，这里是所有类别的真值
        
        for class_name, sampled_num in zip(sampled_groups, # 目标类别，采样数量
                                           sample_num_per_class):
            if sampled_num > 0:
                if self._use_group_sampling: # False
                    sampled_cls = self.sample_group(class_name, sampled_num,
                                                       avoid_coll_boxes, total_group_ids)
                else:
                    sampled_cls = self.sample_class_v2(class_name, sampled_num, # 根据需求采样，返回不冲突的采样框的信息
                                                       avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0: # 逐行存储采样对象的3d真值框信息
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0)

                    sampled_gt_boxes += [sampled_gt_box] # 所有新增采样框
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0) # 这是增加采样后所有的真值框
                    if self._use_group_sampling: # False
                        if len(sampled_cls) == 1:
                            sampled_group_ids = np.array(sampled_cls[0]["group_id"])[np.newaxis, ...]
                        else:
                            sampled_group_ids = np.stack(
                                [s["group_id"] for s in sampled_cls], axis=0)
                        total_group_ids = np.concatenate(
                            [total_group_ids, sampled_group_ids], axis=0)

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0) # 所有采样框信息按行排列
            num_sampled = len(sampled)
            s_points_list = []
            for info in sampled: # 根据采样对象读取对应点云信息，并加进当前帧数据中
                s_points = np.fromfile(
                    str(pathlib.Path(root_path) / info["path"]),
                    dtype=np.float32)
                s_points = s_points.reshape([-1, num_point_features])
                # if not add_rgb_to_points:
                #     s_points = s_points[:, :4]
                if "rot_transform" in info: # False
                    rot = info["rot_transform"]
                    s_points[:, :3] = box_np_ops.rotation_points_single_angle(
                        s_points[:, :3], rot, axis=2)
                s_points[:, :3] += info["box3d_lidar"][:3] # 将归一化的点云坐标恢复为原始点云坐标
                s_points_list.append(s_points) # 采样的点云数据（numpy数组）组织成列表
                # print(pathlib.Path(info["path"]).stem)
            # gt_bboxes = np.stack([s["bbox"] for s in sampled], axis=0)
            # if np.random.choice([False, True], replace=False, p=[0.3, 0.7]):
            # do random crop.
            if random_crop: # False
                s_points_list_new = []
                gt_bboxes = box_np_ops.box3d_to_bbox(sampled_gt_boxes, rect,
                                                     Trv2c, P2)
                crop_frustums = prep.random_crop_frustum(
                    gt_bboxes, rect, Trv2c, P2)
                for i in range(crop_frustums.shape[0]):
                    s_points = s_points_list[i]
                    mask = prep.mask_points_in_corners(
                        s_points, crop_frustums[i:i + 1]).reshape(-1)
                    num_remove = np.sum(mask)
                    if num_remove > 0 and (
                            s_points.shape[0] - num_remove) > 15:
                        s_points = s_points[np.logical_not(mask)]
                    s_points_list_new.append(s_points)
                s_points_list = s_points_list_new
            ret = {
                "gt_names": np.array([s["name"] for s in sampled]), # 采样对象类别，一维数组
                "difficulty": np.array([s["difficulty"] for s in sampled]), # 采样对象困难度，一维数组
                "gt_boxes": sampled_gt_boxes, # 按行排列的采样真值框信息
                "points": np.concatenate(s_points_list, axis=0), # 点云是无序的，可以将不同对象的点云数据放在一起
                "gt_masks": np.ones((num_sampled, ), dtype=np.bool_) # 采样框的标志
            }
            if self._use_group_sampling: # False
                ret["group_ids"] = np.array([s["group_id"] for s in sampled])
            else:
                ret["group_ids"] = np.arange(gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled)) # 采样对象序号
        else:
            ret = None
        return ret

    def sample(self, name, num):
        if self._use_group_sampling:
            group_name = name
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret), ), dtype=np.int64)

    def sample_v1(self, name, num):
        if isinstance(name, (list, tuple)):
            group_name = ", ".join(name)
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(l) for l in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret), ), dtype=np.int64)


    def sample_class_v2(self, name, num, gt_boxes):
        sampled = self._sampler_dict[name].sample(num) # 调用类别对应的BatchSampler内的sample方法，返回用来采样对象的信息
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0] # 该帧数据下原始真值数量
        num_sampled = len(sampled) # 该帧数据下采样真值数量
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6]) # 生成原始真值鸟瞰图

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0) # 读取采样真值框，横向排列

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_) #　原始真值框标为0
        valid_mask = np.concatenate(
            [valid_mask,
             np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0) # 将原始真值框标志0与采样真值框标志1拼起来
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy() # 拼接原始真值框与采样真值框
        if self._enable_global_rot: # False
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                num_try=100)
        sp_boxes_new = boxes[gt_boxes.shape[0]:] # 添加扰动后的新采样框
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]) # 新采样框的鸟瞰图

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0) # 总鸟瞰图
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = prep.box_collision_test(total_bv, total_bv) # 根据鸟瞰图判断相互之间是否碰撞，(采样，原始)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False # 对角线上，自己和自己不发生碰撞

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self._enable_global_rot: # False
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][-1] = boxes[i, -1]
                    sampled[i - num_gt]["rot_transform"] = (
                        boxes[i, -1] - sp_boxes[i - num_gt, -1])
                valid_samples.append(sampled[i - num_gt]) # 保留不冲突的采样对象的信息
        return valid_samples

    def sample_group(self, name, num, gt_boxes, gt_group_ids):
        sampled, group_num = self.sample(name, num)
        sampled = copy.deepcopy(sampled)
        # rewrite sampled group id to avoid duplicated with gt group ids
        gid_map = {}
        max_gt_gid = np.max(gt_group_ids)
        sampled_gid = max_gt_gid + 1
        for s in sampled:
            gid = s["group_id"]
            if gid in gid_map:
                s["group_id"] = gid_map[gid]
            else:
                gid_map[gid] = sampled_gid
                s["group_id"] = sampled_gid
                sampled_gid += 1
        
        num_gt = gt_boxes.shape[0]
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        sp_group_ids = np.stack([i["group_id"] for i in sampled], axis=0)
        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask,
             np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0)
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        group_ids = np.concatenate([gt_group_ids, sp_group_ids], axis=0)
        if self._enable_global_rot:
            # place samples to any place in a circle.
            prep.noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                group_ids=group_ids,
                num_try=100)
        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])
        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = prep.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False
        valid_samples = []
        idx = num_gt
        for num in group_num:
            if coll_mat[idx:idx + num].any():
                coll_mat[idx:idx + num] = False
                coll_mat[:, idx:idx + num] = False
            else:
                for i in range(num):
                    if self._enable_global_rot:
                        sampled[idx - num_gt + i]["box3d_lidar"][:2] = boxes[idx + i, :2]
                        sampled[idx - num_gt + i]["box3d_lidar"][-1] = boxes[idx + i, -1]
                        sampled[idx - num_gt + i]["rot_transform"] = (
                            boxes[idx + i, -1] - sp_boxes[idx + i - num_gt, -1])

                    valid_samples.append(sampled[idx - num_gt + i])
            idx += num
        return valid_samples
