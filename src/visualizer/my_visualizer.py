import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
from enum import Enum
import numpy as np
import json
from queue import Queue
import torch
import threading
import time
from typing import Dict, List, Tuple, Union
import quaternion
from imgviz import depth2rgb
import cv2
import matplotlib.pyplot as plt

import open3d as o3d
from open3d.visualization import rendering, gui

import rospy

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


# class Visualizer:
#     def __init__(self):
#         print("Visualizer init")
#         self.pcd = o3d.geometry.PointCloud()
#         self.vis = o3d.visualization.Visualizer()
#         self.vis.create_window()
#         self.vis.add_geometry(self.pcd)
#         self.__update_main_thread = threading.Thread(
#             target=self.__update_main,
#             name='UpdateMain',
#             daemon=True)
#         self.__update_main_thread.start()
#         rospy.Subscriber("sensor_scan_rgb", PointCloud2, self.callback)

#     def callback(self, data):
#         pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
#         points = np.array(list(pc))
#         print("points",points)
#         self.pcd.points = o3d.utility.Vector3dVector(points)

#     def __update_main(self):
#         while True:
#             self.vis.update_geometry(self.pcd)
#             self.vis.poll_events()
#             self.vis.update_renderer()
#             time.sleep(0.05)
        
        
    
# class Visualizer:
#     def __init__(self):
#         print("Visualizer init")
#         self.pcd = o3d.geometry.PointCloud()
#         self.vis = o3d.visualization.VisualizerWithKeyCallback()
#         self.vis.create_window()
#         self.vis.add_geometry(self.pcd)
#         self.__update_main_thread = threading.Thread(
#             target=self.__update_main,
#             name='UpdateMain',
#             daemon=True)
#         self.__update_main_thread.start()
#         self.new_points = None  # 用于在线程间传递新的点云数据
#         self.lock = threading.Lock()  # 用于线程同步
#         rospy.Subscriber("sensor_scan_rgb", PointCloud2, self.callback)

#     def callback(self, data):
#         with self.lock:  # 确保线程安全
#             pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
#             points = np.array(list(pc))
#             print("points", points)
#             self.new_points = points

#     def __update_main(self):
#         while True:
#             with self.lock:  # 确保线程安全
#                 if self.new_points is not None:
#                     self.pcd.points = o3d.utility.Vector3dVector(self.new_points)
#                     self.pcd.colors = o3d.utility.Vector3dVector([  # 假设颜色为白色
#                         [0, 1, 1] for _ in range(len(self.new_points))
#                     ])
#                     self.new_points = None
#             self.vis.update_geometry(self.pcd)
#             self.vis.poll_events()
#             self.vis.update_renderer()
#             time.sleep(0.01)  # 减少睡眠时间，提高更新频率

# # 确保在退出时释放资源
# def shutdown_hook():
#     vis = Visualizer()  # 假设你的Visualizer实例化对象是vis
#     vis.vis.destroy_window()
#     rospy.signal_shutdown("Shutting down")

# rospy.on_shutdown(shutdown_hook)


class Visualizer:
    def __init__(self):
        print("Visualizer init")
        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)
        self.__update_main_thread = threading.Thread(
            target=self.__update_main,
            name='UpdateMain',
            daemon=True)
        self.__update_main_thread.start()
        self.lock = threading.Lock()  # 用于线程同步
        rospy.Subscriber("sensor_scan_rgb", PointCloud2, self.callback)

    def callback(self, data):
        with self.lock:  # 确保线程安全
            pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
            new_points = np.array(list(pc))
            print("New points received:", new_points.shape)
            # 累加新的点云数据
            if not self.pcd.points.empty():
                # 将新点云转换为Vector3dVector
                new_points_v3d = o3d.utility.Vector3dVector(new_points)
                # 更新现有的点云
                self.pcd.points.resize(self.pcd.points.size() + new_points_v3d.size())
                self.pcd.points += new_points_v3d
            else:
                # 如果是第一次接收点云，直接设置
                self.pcd.points = o3d.utility.Vector3dVector(new_points)

    def __update_main(self):
        while True:
            with self.lock:  # 确保线程安全
                self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)  # 减少睡眠时间，提高更新频率

# 确保在退出时释放资源
def shutdown_hook():
    vis = Visualizer()  # 假设你的Visualizer实例化对象是vis
    vis.vis.destroy_window()
    rospy.signal_shutdown("Shutting down")

rospy.on_shutdown(shutdown_hook)