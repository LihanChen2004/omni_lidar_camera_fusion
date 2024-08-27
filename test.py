# import open3d as o3d
# import numpy as np
# import threading
# import time

# # 创建一个全局的PointCloud对象
# pcd = o3d.geometry.PointCloud()

# # 创建一个全局的可视化对象
# vis = o3d.visualization.Visualizer()

# # 创建一个全局的锁对象，用于同步线程
# lock = threading.Lock()

# def update_point_cloud():
#     global pcd, lock
#     while True:
#         # 生成一些随机的点云数据
#         points = np.random.rand(100, 3)
        
#         # 获取锁，更新点云数据
#         lock.acquire()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         lock.release()

#         # 每秒更新一次
#         time.sleep(1)

# def visualize_point_cloud():
#     global pcd, vis, lock

#     # 初始化可视化对象
#     vis.create_window()
#     vis.add_geometry(pcd)

#     while True:
#         # 获取锁，更新可视化
#         lock.acquire()
#         vis.update_geometry(pcd)
#         vis.poll_events()
#         vis.update_renderer()
#         lock.release()

#         # 每秒刷新一次
#         time.sleep(1)

# if __name__ == "__main__":
#     # 创建并启动更新点云数据的线程
#     update_thread = threading.Thread(target=update_point_cloud)
#     update_thread.start()

#     # 在主线程中进行可视化
#     visualize_point_cloud()

#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from  nav_msgs.msg  import Odometry
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import threading
import time
import tf.transformations as tf_trans
import numpy as np

# 创建一个全局的PointCloud对象
pcd = o3d.geometry.PointCloud()
global_pcd = o3d.geometry.PointCloud()

# 创建一个全局的可视化对象
vis = o3d.visualization.Visualizer()


# 创建一个全局的锁对象，用于同步线程
lock = threading.Lock()

pose_list = []

def pose_callback(data):
    global pose_list
    
    pose_list.append(data)

    # 如果pose_list太大，删除最旧的pose
    if len(pose_list) > 1000:
        pose_list.pop(0)

def rbg_pcd_callback(data):
    global pcd, lock, pose_list, global_pcd

    # print("pcd.header.stamp",data.header.stamp)

    # 在pose_list中找到最近的pose
    closest_pose = min(pose_list, key=lambda pose: abs(pose.header.stamp.to_nsec() - data.header.stamp.to_nsec()))
    
    # print("pose.header.stamp",closest_pose.header.stamp)
    
    # 获取平移向量和旋转矩阵
    translation = [closest_pose.pose.pose.position.x, closest_pose.pose.pose.position.y, closest_pose.pose.pose.position.z]
    rotation = [closest_pose.pose.pose.orientation.x, closest_pose.pose.pose.orientation.y, closest_pose.pose.pose.orientation.z, closest_pose.pose.pose.orientation.w]

    # 将四元数转换为旋转矩阵
    rotation_matrix = tf_trans.quaternion_matrix(rotation)

    # 创建变换矩阵
    transform_matrix = tf_trans.compose_matrix(translate=translation)
    transform_matrix[:3, :3] = rotation_matrix[:3, :3]

    # 获取逆变换矩阵
    inverse_transform_matrix = np.linalg.inv(transform_matrix)

    # 从ROS坐标系转换到OpenCV坐标系
    transform_matrix_ros_to_cv = np.array([[0, 0, 1, 0],
                                        [-1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, 0, 1]])

    # 将PointCloud2消息转换为numpy数组
    pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
    pc_arr = np.array(list(pc))

    # 将点云矩阵的形状从(2644,3)变为(2644,4)，并将最后一列设置为1
    pc_arr = np.hstack((pc_arr, np.ones((pc_arr.shape[0], 1))))

    # 获取锁，更新点云数据
    lock.acquire()
    try:
        pcd.points = o3d.utility.Vector3dVector(pc_arr[:,:3])
        
        # 对点云进行变换
        transformed_points = np.dot(pc_arr, np.dot(inverse_transform_matrix, transform_matrix_ros_to_cv))[:,:3]
        
        # 将numpy数组转换为o3d.utility.Vector3dVector
        pcd.points = o3d.utility.Vector3dVector(transformed_points)
        global_pcd += pcd
        # global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)
    except Exception as e:
        print(f"Error during point cloud transformation: {e}")
    finally:
        lock.release()

def update_point_cloud():
    # rospy.Subscriber("/sensor_scan_rgb", PointCloud2, rbg_pcd_callback)
    rospy.Subscriber("/sensor_scan", PointCloud2, rbg_pcd_callback)
    rospy.Subscriber("/state_estimation", Odometry, pose_callback)
    rospy.spin()

def visualize_point_cloud():
    global pcd, vis, lock, global_pcd
    
    # 初始化可视化对象
    vis.create_window()
    
    # cube = o3d.geometry.TriangleMesh.create_box()
    # vis.add_geometry(cube)
    # # 添加立方体到可视化窗口
    # vis.add_geometry(cube)
    
    # 创建一个坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # 添加坐标系到可视化窗口
    vis.add_geometry(coordinate_frame)
    

    # 添加点云到可视化窗口
    vis.add_geometry(global_pcd)

    while not rospy.is_shutdown():
        # 获取锁，更新可视化
        lock.acquire()
        # global_pcd.scale(0.1, center=global_pcd.get_center())
        print("global_pcd",len(global_pcd.points))
        vis.update_geometry(global_pcd)
        vis.poll_events()
        vis.update_renderer()
        lock.release()

        # 每秒刷新一次
        time.sleep(1)

if __name__ == "__main__":
    rospy.init_node('point_cloud_listener', anonymous=True)

    # 创建并启动更新点云数据的线程
    update_thread = threading.Thread(target=update_point_cloud)
    update_thread.start()

    # 在主线程中进行可视化
    visualize_point_cloud()