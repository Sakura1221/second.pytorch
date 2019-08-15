"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.utils import get_paddings_indicator
from torchplus.nn import Empty
from torchplus.tools import change_default_args


class PFNLayer(nn.Module): # 体素特征统计后进一步提取
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe: # False
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        """
        体素特征提取层的前向传递
        :param inputs: (num_voxels,max_points,4+3+2),体素的VFE特征,其中num_voxels相当于batch_size
        :return:
        """

        x = self.linear(inputs) # Linear(in_features=9, out_features=64, bias=False)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous() # 维度变换然后再变换回来,批规范化需要将特征数放在前面,注意看英文文档！
        x = F.relu(x) # (num_voxels,max_points,64)

        x_max = torch.max(x, dim=1, keepdim=True)[0] # 返回每个特征值下的最大值,(num_voxels,1,64)

        if self.last_vfe: # True
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module): # 每个体素内的特征统计与提取
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors): # PFN网络前向传递
        """
        :param features: 体素特征,(num_voxels,max_points,4)
        :param num_voxels: 这里可能是打错了,应该是体素内的点数(num_voxels,)
        :param coors: 体素的坐标索引,第一位是用来区别2帧数据的(num_voxels,4)
        :return:每一个特征最大的特征值(num_voxels,64)
        """

        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1) # 求每个体素内所有点的中心坐标,(num_voxels,1,3)
        f_cluster = features[:, :, :3] - points_mean # 体素内的点减去中心点归一化,(num_voxels,100,3)

        # Find distance of x, y, and z from pillar center
        # 根据体素索引，以及体素偏移确定每个体素的水平中心（xy方向）
        f_center = features[:, :, :2] # (num_voxels,max_points,2)
        f_center[:, :, 0] = f_center[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance: # False
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1) # 将最后一维的特征相加,(num_voxels,max_points,4+3+2)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1] # max_points
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0) # 每个体素内用来填充的标签,每个体素内有几个点每行就有几个1(num_voxels,max_points)
        mask = torch.unsqueeze(mask, -1).type_as(features) # (num_voxels,max_points,1)
        features *= mask # 对应元素相乘，空元素确保置0(num_voxels,max_points,4+3+2)

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers: # 调用上面的pfn层进行体素特征提取
            features = pfn(features)

        return features.squeeze() # 去掉值为1的维度


class PointPillarsScatter(nn.Module): # 中间特征提取层
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape # [1,1,496,432,64]
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size): # 2个batch单独处理
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype, # 先建立整个场景的体素特征容器,默认置0[64,432*496]
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt # 区分batch的标志
            this_coords = coords[batch_mask, :] #　筛选每个batch的体素坐标,[num_voxels_per_batch,4]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3] # 由于z方向为1,对所有体素排序只需要考虑x,y方向即可,整个体素沿着x轴(左右)展开成1维，并建立索引
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :] # 筛选每个batch的体素特征,[num_voxels_per_batch,64]
            voxels = voxels.t() # 转置,[64,num_voxels_per_batch]

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels # 将体素特征按照索引放入容器中

            # Append to a list for later stacking.
            batch_canvas.append(canvas) # 2帧数据产生2个容器组成列表

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0) # 将列表的两个元素堆叠成3维数据,[2,64,432*496]

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx) # 2帧数据内所有体素的64通道特征,y前后索引,x左右索引[2, 64, 496, 432]

        return batch_canvas # 2帧数据内所有体素的64通道特征,y前后索引,x左右索引[2, 64, 496, 432]
