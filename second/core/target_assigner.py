from second.core import box_np_ops
from second.core.target_ops import create_target_np
from second.core import region_similarity
import numpy as np

class TargetAssigner:
    def __init__(self,
                 box_coder,
                 anchor_generators,
                 region_similarity_calculator=None,
                 positive_fraction=None,
                 sample_size=512):
        self._region_similarity_calculator = region_similarity_calculator
        self._box_coder = box_coder # GroundBox3dCoder
        self._anchor_generators = anchor_generators
        self._positive_fraction = positive_fraction
        self._sample_size = sample_size

    @property
    def box_coder(self):
        return self._box_coder

    def assign(self,
               anchors,
               gt_boxes,
               anchors_mask=None,
               gt_classes=None,
               matched_thresholds=None,
               unmatched_thresholds=None):
        if anchors_mask is not None:
            prune_anchor_fn = lambda _: np.where(anchors_mask)[0] # 返回非0元素的坐标,这里的lambda很多余,实际上不需要任何参数
        else:
            prune_anchor_fn = None

        def similarity_fn(anchors, gt_boxes): # 根据鸟瞰图计算IOU,嵌套函数第一时间不执行
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return self._region_similarity_calculator.compare(
                anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors): # 锚框编码,嵌套函数可以第一时间不传参数
            return self._box_coder.encode(boxes, anchors)

        return create_target_np( # 创建训练目标的核心函数
            anchors, # (248*216*2,7)
            gt_boxes, # (gt_num, 7)
            similarity_fn, # func
            box_encoding_fn, # func
            prune_anchor_fn=prune_anchor_fn, # func
            gt_classes=gt_classes, # (gt_num,)从1开始标志不同真值的类别
            matched_threshold=matched_thresholds, # 0.6,(248*216*2,)
            unmatched_threshold=unmatched_thresholds, # 0.45,(248*216*2,)
            positive_fraction=self._positive_fraction, # None
            rpn_batch_size=self._sample_size, # 512
            norm_by_num_examples=False,
            box_code_size=self.box_coder.code_size) # 7

    def generate_anchors(self, feature_map_size):
        anchors_list = []
        matched_thresholds = [
            a.match_threshold for a in self._anchor_generators # 0.6
        ]
        unmatched_thresholds = [
            a.unmatch_threshold for a in self._anchor_generators # 0.45
        ]
        match_list, unmatch_list = [], []
        for anchor_generator, match_thresh, unmatch_thresh in zip(
                self._anchor_generators, matched_thresholds,
                unmatched_thresholds):
            anchors = anchor_generator.generate(feature_map_size) # array[1,248,216,1,2,7]
            anchors = anchors.reshape([*anchors.shape[:3], -1, 7]) # array[1,248,216,2,7],[zyx_idx,r_idx,(xyz_lidar,w,l,h,r)]
            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1]) # 1*248*216*2 = 107136
            match_list.append(
                np.full([num_anchors], match_thresh, anchors.dtype)) # list[array[248*216*2]]
            unmatch_list.append(
                np.full([num_anchors], unmatch_thresh, anchors.dtype))
        anchors = np.concatenate(anchors_list, axis=-2)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        return {
            "anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds
        }

    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num