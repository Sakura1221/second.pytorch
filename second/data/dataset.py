import pathlib
import pickle
import time
from functools import partial

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.preprocess import _read_and_prep_v9


class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class KittiDataset(Dataset):
    def __init__(self, info_path, root_path, num_point_features,
                 target_assigner, feature_map_size, prep_func):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        #self._kitti_infos = kitti.filter_infos_by_used_classes(infos, class_names)
        self._root_path = root_path
        self._kitti_infos = infos # kitti_infos_train.pkl
        self._num_point_features = num_point_features
        print("remain number of infos:", len(self._kitti_infos))
        # generate anchors cache
        # feature_map_size:(1,248,216)
        ret = target_assigner.generate_anchors(feature_map_size) # 生成训练用锚框，包括匹配的锚框与未匹配的锚框
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            # [z_nums*y_nums*x_nums*rotation_nums,[xyz_center+sizes+rad]] (xyz,wlh,rad,前后左右上下)
            "anchors": anchors, # array[107136, 7]
            "anchors_bv": anchors_bv, # array[107136, 4] [x_min,y_min,x_max,y_max]
            "matched_thresholds": matched_thresholds, # array[107136]
            "unmatched_thresholds": unmatched_thresholds, # array[107136]
        }
        self._prep_func = partial(prep_func, anchor_cache=anchor_cache)

    def __len__(self):
        return len(self._kitti_infos)

    @property
    def kitti_infos(self):
        return self._kitti_infos

    def __getitem__(self, idx):
        return _read_and_prep_v9( # 读取并对点云预处理，返回example
            info=self._kitti_infos[idx], # kitti_infos_train.pkl
            root_path=self._root_path,
            num_point_features=self._num_point_features,
            prep_func=self._prep_func)
