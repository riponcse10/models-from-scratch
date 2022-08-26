import pytorch3d
import torch
import os
import numpy as np

from pytorch3d.structures import Pointclouds
filename = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti/training/velodyne/000000.bin"
file2 = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti/training/velodyne/000001.bin"


def get_pcd(path=filename):
    pcd = np.fromfile(file=filename, dtype=np.float32).reshape(-1,4)
    pcd1 = np.fromfile(file=file2, dtype=np.float32).reshape(-1,4)
    pcd = pcd[:,:3] #drop the intensity channel, because pytorch3d Pointcloud object accepts 3 values per point
    pcd2 = pcd1[:, :3]
    pcd_tensor1 = torch.from_numpy(pcd)
    pcd_tensor2 = torch.from_numpy(pcd2)
    # pcd_tensor = torch.unsqueeze(pcd_tensor, 0)
    pcl = Pointclouds([pcd_tensor1, pcd_tensor2])
    # IO().save_pointcloud(pcl, "saved.obj")

    return pcl

# flist = os.listdir("/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti/training/velodyne")
# print(flist[0])