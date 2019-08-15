#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/6/20 下午2:46

import numpy as np
path = '/mnt/share/users/zzc/kitti_second/training/velodyne/000000.bin'
points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
# path:待打开的文件对象, dtype: 返回的数据类型, count: int,要读取的项目数.
# reshape([a, b]),原来a*b个一维数组，每b个为一行,a=-1可表示任意长度为b倍数的数组，重组为b列
np.savetxt('/home/zzc/second.pytorch/test/000000_origin.txt', points)
