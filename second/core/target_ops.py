import logging

import numba
import numpy as np
import numpy.random as npr

import second.core.box_np_ops as box_np_ops

logger = logging.getLogger(__name__)


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret




def create_target_np(all_anchors, # (248*216*2,7)
                     gt_boxes, # (gt_num, 7)
                     similarity_fn, # 计算iou的函数
                     box_encoding_fn, # 根据锚框编码真值的函数
                     prune_anchor_fn=None, # func
                     gt_classes=None, # (gt_num,)
                     matched_threshold=0.6, # 0.6,(248*216*2,)
                     unmatched_threshold=0.45, # 0.45,(248*216*2,)
                     bbox_inside_weight=None,
                     positive_fraction=None,
                     rpn_batch_size=300,
                     norm_by_num_examples=False,
                     box_code_size=7):
    """Modified from FAIR detectron.
    Args:
        all_anchors: [num_of_anchors, box_ndim] float tensor.
        gt_boxes: [num_gt_boxes, box_ndim] float tensor.
        similarity_fn: a function, accept anchors and gt_boxes, return
            similarity matrix(such as IoU).真值框们与锚框们的IoU
        box_encoding_fn: a function, accept gt_boxes and anchors, return
            box encodings(offsets).真值框相对锚框的偏移
        prune_anchor_fn: a function, accept anchors, return indices that
            indicate valid anchors.返回有效锚框的索引
        gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must
            start with 1.真值类别标识,从1开始
        matched_threshold: float, iou greater than matched_threshold will
            be treated as positives.
        unmatched_threshold: float, iou smaller than unmatched_threshold will
            be treated as negatives.
        bbox_inside_weight: unused
        positive_fraction: [0-1] float or None. if not None, we will try to
            keep ratio of pos/neg equal to positive_fraction when sample.
            if there is not enough positives, it fills the rest with negatives
        rpn_batch_size: int. sample size
        norm_by_num_examples: bool. norm box_weight by number of examples, but
            I recommend to do this outside.
    Returns:
        labels, bbox_targets, bbox_outside_weights
    """
    total_anchors = all_anchors.shape[0] # 248*216*2
    # 由于点云的稀疏性,过滤点云过少的锚框可以提高效率
    if prune_anchor_fn is not None:
        inds_inside = prune_anchor_fn(all_anchors) # 有效锚框索引
        anchors = all_anchors[inds_inside, :] # 有效锚框数据
        if not isinstance(matched_threshold, float):
            matched_threshold = matched_threshold[inds_inside]
        if not isinstance(unmatched_threshold, float):
            unmatched_threshold = unmatched_threshold[inds_inside]
    else:
        anchors = all_anchors
        inds_inside = None
    num_inside = len(inds_inside) if inds_inside is not None else total_anchors # 有效锚框数量
    box_ndim = all_anchors.shape[1] # 7
    logger.debug('total_anchors: {}'.format(total_anchors))
    logger.debug('inds_inside: {}'.format(num_inside))
    logger.debug('anchors.shape: {}'.format(anchors.shape))
    if gt_classes is None:
        gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)
    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside, ), dtype=np.int32)
    gt_ids = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)
    gt_ids.fill(-1)
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes) # 计算真值与剩余锚框的iou,(res_anchor,gt_num)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1) # 与每个锚框有最大iou的真值索引(res_anchor,),0 to gt_num,注意如果没有重叠也为0
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                anchor_to_gt_argmax]  # 每个锚框与真值最大的iou(res_anchor,)
        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0) # 与每个真值有最大iou的锚框索引(gt_num,)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[ # 每个真值与锚框最大的iou(gt_num,)
            gt_to_anchor_argmax,
            np.arange(anchor_by_gt_overlap.shape[1])]
        # must remove gt which doesn't match any anchor.
        empty_gt_mask = gt_to_anchor_max == 0 # 去掉未和任何锚框匹配的真值
        gt_to_anchor_max[empty_gt_mask] = -1 # 相应真值的iou置为-1,与全部真值和全部锚框的iou_map原来的0区分,方便下面筛选最大iou锚框
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max)[0] # 筛选与真值iou最大的锚框索引,可能存在一个真值对应两个锚框
        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] # 筛选iou最大锚框对应的真值索引,一个真值可以出现多次
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] # iou最大锚框对应真值的类别向labels对应元素赋值,其他锚框对应真值标签默认为-1
        gt_ids[anchors_with_max_overlap] = gt_inds_force # 所有iou最大锚框对应真值的索引向所有锚框真值索引对应元素赋值,其他锚框对应真值索引默认为-1
        # Fg label: above threshold IOU
        pos_inds = anchor_to_gt_max >= matched_threshold # 每个锚框最大iou大于等于阈值为正样本,bool,(res_anchor,)
        gt_inds = anchor_to_gt_argmax[pos_inds] # 返回iou超过阈值的锚框对应的真值索引
        labels[pos_inds] = gt_classes[gt_inds] # 最大iou超过阈值的前景锚框对应的真值类别标签
        gt_ids[pos_inds] = gt_inds # 最大iou超过阈值的前景锚框对应的真值索引
        bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0] # 背景锚框索引
    else:
        # labels[:] = 0
        bg_inds = np.arange(num_inside)
    fg_inds = np.where(labels > 0)[0] # 前景锚框索引（每个真值最大iou锚框与iou达到阈值的锚框）
    fg_max_overlap = None
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        fg_max_overlap = anchor_to_gt_max[fg_inds] # 前景锚框iou
    gt_pos_ids = gt_ids[fg_inds] # 前景锚框对应真值索引
    # bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    # bg_inds = np.where(labels == 0)[0]
    # subsample positive labels if we have too many
    if positive_fraction is not None: # False
        num_fg = int(positive_fraction * rpn_batch_size)
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
            fg_inds = np.where(labels > 0)[0]

        # subsample negative labels if we have too many
        # (samples with replacement, but since the set of bg inds is large most
        # samples will not have repeats)
        num_bg = rpn_batch_size - np.sum(labels > 0)
        # print(num_fg, num_bg, len(bg_inds) )
        if len(bg_inds) > num_bg:
            enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
            labels[enable_inds] = 0
        bg_inds = np.where(labels == 0)[0]
    else:
        if len(gt_boxes) == 0 or anchors.shape[0] == 0: # 如果此帧数据没有真值
            labels[:] = 0
        else:
            labels[bg_inds] = 0 # 背景锚框标签置0
            # re-enable anchors_with_max_overlap
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] # 最大iou锚框对应真值类别标签（未达到0.6但与真值有最大iou的前景锚框）
    bbox_targets = np.zeros(
        (num_inside, box_code_size), dtype=all_anchors.dtype) # 设置方盒训练目标
    if len(gt_boxes) > 0 and anchors.shape[0] > 0: #　有真值且有匹配的锚框
        # print(anchors[fg_inds, :].shape, gt_boxes[anchor_to_gt_argmax[fg_inds], :].shape)
        bbox_targets[fg_inds, :] = box_encoding_fn(
            gt_boxes[anchor_to_gt_argmax[fg_inds], :], anchors[fg_inds, :]) # 编码前景锚框对应真值,也即编码锚框与真值的偏移,得到方盒回归目标
    # Bbox regression loss has the form:
    #   loss(x) = weight_outside * L(weight_inside * x)
    # Inside weights allow us to set zero loss on an element-wise basis.(内部权重可以逐元素置0,忽略不重要对象的loss)
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise
    # NOTE: we don't need bbox_inside_weights, remove it.
    # bbox_inside_weights = np.zeros((num_inside, box_ndim), dtype=np.float32)
    # bbox_inside_weights[labels == 1, :] = [1.0] * box_ndim

    # The bbox regression loss only averages by the number of images in the
    # mini-batch, whereas we need to average by the total number of example
    # anchors selected
    # Outside weights are used to scale each element-wise loss so the final(外部权重可以逐元素放缩loss,比如求平均)
    # average over the mini-batch is correct
    # bbox_outside_weights = np.zeros((num_inside, box_ndim), dtype=np.float32)
    bbox_outside_weights = np.zeros((num_inside, ), dtype=all_anchors.dtype)
    # uniform weighting of examples (given non-uniform sampling)
    if norm_by_num_examples: # False
        num_examples = np.sum(labels >= 0)  # neg + pos
        num_examples = np.maximum(1.0, num_examples)
        bbox_outside_weights[labels > 0] = 1.0 / num_examples
    else:
        bbox_outside_weights[labels > 0] = 1.0 # 这里没有选择按照前景锚框数量求均值归一化,前景锚框权重置1,其他0
    # bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

    # Map up to original set of anchors
    if inds_inside is not None: # 有效锚框数量
        labels = unmap(labels, total_anchors, inds_inside, fill=-1) # (total_anchors,),所有锚框对应真值类别(1,2...),无对应真值设为0,dontcare设为-1
        bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0) # (total_anchors, 7),所有锚框对应真值锚框,非前景锚框设为[0,0,0,0,0,0,0]
        # bbox_inside_weights = unmap(
        #     bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = unmap(
            bbox_outside_weights, total_anchors, inds_inside, fill=0) # (total_anchors,),所有锚框的外部权重,有对应真值设为1，无对应真值设为0
    # return labels, bbox_targets, bbox_outside_weights
    ret = {
        "labels": labels, # (total_anchors,),所有锚框对应真值类别,前景锚框(1,2...),背景锚框设为0,dontcare设为-1
        "bbox_targets": bbox_targets, # (total_anchors, 7),所有锚框与对应真值锚框偏移值编码,无对应真值设为[0,0,0,0,0,0,0]
        "bbox_outside_weights": bbox_outside_weights, # (total_anchors,),所有锚框的外部权重,前景锚框为1,其他为0
        "assigned_anchors_overlap": fg_max_overlap, # (fg_anchors,),前景锚框对应iou
        "positive_gt_id": gt_pos_ids, # (fg_anchors,),前景锚框对应真值标签
    }
    if inds_inside is not None: # True
        ret["assigned_anchors_inds"] = inds_inside[fg_inds] # 所有前景锚框索引
    else:
        ret["assigned_anchors_inds"] = fg_inds
    return ret
