#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/5/25 下午3:26

import pickle
fp = open('/mnt/share/users/zzc/kitti_second/kitti_dbinfos_train.pkl','rb')
info = pickle.load(fp)
info = str(info)
ft = open('/mnt/share/users/zzc/kitti_second/kitti_dbinfos_train.txt', 'w')
ft.write(info)
fp.close()
ft.close()