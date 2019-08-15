import torch


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]):每个体素内实际点数(num_voxels,)
        max_num ([type]): 每个体素内最大点数

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1) # (num_voxels,1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape) # [1,1]
    max_num_shape[axis + 1] = -1 # [1,-1]
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape) # arange[1,max_num]
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num # [num_voxels,max_num],bool
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator
