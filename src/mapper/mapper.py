import open3d as o3d

class Mapper:
    def __init__(self):
        self.global_pcd = o3d.geometry.PointCloud()
        self.downsampled = o3d.geometry.PointCloud()

    def update_color_pcd(self,global_pcd:o3d.geometry.PointCloud, point_cloud_o3d:o3d.geometry.PointCloud):
        
        # accumulate color pcd
        self.global_pcd = global_pcd
        self.global_pcd += point_cloud_o3d
        self.downsampled = self.global_pcd.voxel_down_sample(voxel_size=0.1)

        # 创建一个KDTree来查找最近的点
        kdtree = o3d.geometry.KDTreeFlann(self.global_pcd)

        # 对于新的点云中的每个点，找到原始点云中最近的点，并使用该点的颜色
        new_colors = []
        for point in self.downsampled.points:
            _, index, _ = kdtree.search_knn_vector_3d(point, 1)
            new_colors.append(self.global_pcd.colors[index[0]])

        self.downsampled.colors = o3d.utility.Vector3dVector(new_colors)
        return self.downsampled
        # print("downsampled in rbg_pcd_callback",len(self.downsampled.points))
            
        # accumulate bbox


class Bbox:
    def __init__(self, id, position_x, position_y, position_z, scale_x, scale_y, scale_z, ns, color=(0, 0, 0, 0)):
        self.id = id
        self.position_x = position_x
        self.position_y = position_y
        self.position_z = position_z
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.ns = ns
        self.color = color        