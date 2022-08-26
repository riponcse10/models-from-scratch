from inpute_data import get_pcd

pcd = get_pcd()
pcd = pcd.points_list()
print(type(pcd))
# pcd = pcd.points_padded()
print(pcd.shape)
# print(pcd.num_points_per_cloud())
splits = pcd.split([1,3,4])
print(splits[0].num_points_per_cloud())