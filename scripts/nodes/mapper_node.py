#!/usr/bin/env python
import faulthandler
import threading
import time
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np

# # 单帧点云版
# def callback(data):
#     pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
#     points = np.array(list(pc))
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.visualization.draw_geometries([pcd])

# def listener():
#     rospy.init_node('mapper_node_lidar', anonymous=True)
#     rospy.Subscriber("sensor_scan_rgb", PointCloud2, callback)
#     rospy.spin()

# if __name__ == '__main__':
#     listener()
    
import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2


# # 全局变量，用于存储累加的点云数据
# accumulated_pcd = o3d.geometry.PointCloud()

# def callback(data):
#     global accumulated_pcd
#     # 读取点云数据
#     pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
#     points = np.array(list(pc), dtype=np.float64)  # 确保数据类型匹配
#     new_pcd = o3d.geometry.PointCloud()
#     new_pcd.points = o3d.utility.Vector3dVector(points)  # 使用 Vector3dVector 存储点云
    
#     # 将新的点云累加到全局变量中
#     accumulated_pcd.points.extend(new_pcd.points)  # 使用 extend 来累加点云
    
#     # 可视化累加后的点云
#     o3d.visualization.draw_geometries([accumulated_pcd])

# def listener():
#     rospy.init_node('mapper_node_lidar', anonymous=True)
#     rospy.Subscriber("sensor_scan_rgb", PointCloud2, callback)
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down")

# if __name__ == '__main__':
#     listener()
    
    

    
import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import threading
import time

# # 全局变量，用于存储累加的点云数据
# accumulated_pcd = o3d.geometry.PointCloud()
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(accumulated_pcd)
# lock = threading.Lock()

# def update_visualization():
#     global vis, accumulated_pcd
#     while not rospy.is_shutdown():
#         with lock:
#             vis.update_geometry(accumulated_pcd)
#         vis.poll_events()
#         vis.update_renderer()
#         time.sleep(0.01)

# def callback(data):
#     print("data",data)
#     global accumulated_pcd
#     # 读取点云数据
#     pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
#     points = np.array(list(pc), dtype=np.float64)  # 确保数据类型匹配
#     new_pcd = o3d.geometry.PointCloud()
#     new_pcd.points = o3d.utility.Vector3dVector(points)  # 使用 Vector3dVector 存储点云

#     # 将新的点云累加到全局变量中
#     with lock:
#         accumulated_pcd.points.extend(new_pcd.points)  # 使用 extend 来累加点云

# def listener():
#     rospy.init_node('mapper_node_lidar', anonymous=True)
#     rospy.Subscriber("sensor_scan_rgb", PointCloud2, callback)
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down")

# if __name__ == '__main__':
#     vis_thread = threading.Thread(target=update_visualization)
#     vis_thread.start()
#     listener()
#     vis.destroy_window()
    
    
    
    
    
    
    

# 点云融合还不好使版
# current_pcd = o3d.geometry.PointCloud()

# def callback(data):
#     global current_pcd  # 声明使用全局变量
#     pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
#     points = np.array(list(pc))
#     new_pcd = o3d.geometry.PointCloud()
#     new_pcd.points = o3d.utility.Vector3dVector(points)
    
#     # 将新的点云数据融合到现有的点云中
#     current_pcd += new_pcd
    
#     # 绘制更新后的点云
#     o3d.visualization.draw_geometries([current_pcd])

# def listener():
#     rospy.init_node('mapper_node', anonymous=True)
#     rospy.Subscriber("sensor_scan_rgb", PointCloud2, callback)
#     rospy.spin()

# if __name__ == '__main__':
#     listener()



# # activeGS版
import torch
from open3d.visualization import gui
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
from visualizer.visualizer import Visualizer

if __name__ == '__main__':
    faulthandler.enable()
    
    rospy.init_node('mapper_node_lidar', anonymous=True, log_level=rospy.INFO)
    
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        rospy.logwarn('No GPU available.')
        device = torch.device('cpu')
    hide_windows = False
    if not hide_windows:
        app = gui.Application.instance
        app.initialize()
    w = Visualizer(device,hide_windows)
    if hide_windows:
        rospy.spin()
    else:
        app.run()
    
    rospy.loginfo(f'lidar omni mapper node finished.')