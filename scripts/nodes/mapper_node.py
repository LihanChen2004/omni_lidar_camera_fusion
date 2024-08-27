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

# 全局变量，用于存储累加的点云数据
accumulated_pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(accumulated_pcd)
lock = threading.Lock()

def update_visualization():
    global vis, accumulated_pcd
    while not rospy.is_shutdown():
        with lock:
            vis.update_geometry(accumulated_pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

def callback(data):
    print("data",data)
    global accumulated_pcd
    # 读取点云数据
    pc = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(pc), dtype=np.float64)  # 确保数据类型匹配
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points)  # 使用 Vector3dVector 存储点云

    # 将新的点云累加到全局变量中
    with lock:
        accumulated_pcd.points.extend(new_pcd.points)  # 使用 extend 来累加点云

def listener():
    rospy.init_node('mapper_node_lidar', anonymous=True)
    rospy.Subscriber("sensor_scan_rgb", PointCloud2, callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    vis_thread = threading.Thread(target=update_visualization)
    vis_thread.start()
    listener()
    vis.destroy_window()
    
    
    
    
    
    
    

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
# from open3d.visualization import gui
# import os
# PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
# SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
# import sys
# sys.path.append(PACKAGE_PATH)
# sys.path.append(SRC_PATH)
# from visualizer.my_visualizer import Visualizer

# if __name__ == '__main__':
#     faulthandler.enable()
    
#     rospy.init_node('mapper_node_lidar', anonymous=True, log_level=rospy.INFO)
    
    
#     hide_windows = False
#     if not hide_windows:
#         app = gui.Application.instance
#         app.initialize()
#     w = Visualizer()
#     if hide_windows:
#         rospy.spin()
#     else:
#         app.run()
    
#     rospy.loginfo(f'lidar omni mapper node finished.')

# #!/usr/bin/env python
# import os
# PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
# SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
# import sys
# sys.path.append(PACKAGE_PATH)
# sys.path.append(SRC_PATH)
# import json
# import argparse
# from typing import Union

# import faulthandler

# import torch
# import numpy as np
# from open3d.visualization import gui
# from PIL import ImageFile, Image
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None

# import rospy

# from utils import PROJECT_NAME, GlobalState
# from dataloader.dataloader import get_dataset, HabitatDataset
# from visualizer.visualizer import Visualizer

# if __name__ == '__main__':
#     faulthandler.enable()
#     seed = 1
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} mapper node.')
#     parser.add_argument('--config',
#                         type=str,
#                         required=False,
#                         default="/home/user/gaussianProject/ros_ws/src/omni_lidar_camera_fusion/config_activegs/datasets/gibson.json",
#                         help='Input config url (*.json).')
#     parser.add_argument('--scene_id',
#                         type=str,
#                         required=False,
#                         default="Denmark",
#                         help='Specify test scene id.')
#     parser.add_argument('--user_config',
#                         type=str,
#                         required=False,
#                         default="/home/user/gaussianProject/ros_ws/src/omni_lidar_camera_fusion/config_activegs/user_config.json",
#                         help='User config url (*.json).')
#     parser.add_argument('--gpu_id',
#                         type=int,
#                         required=False,
#                         default=0,
#                         help='Specify gpu id.')
#     parser.add_argument('--mode',
#                         type=str,
#                         choices=list(GlobalState.__members__)[:-1],
#                         required=False,
#                         default="AUTO_PLANNING",
#                         help='Specify the mode to start with.')
#     parser.add_argument('--actions',
#                         type=str,
#                         required=False,
#                         default="None",
#                         help='Specify the actions to replay.')
#     parser.add_argument('--ros_dataloader',
#                         type=int,
#                         required=False,
#                         default=0,
#                         help='Tell the mapper node to use ROS dataloader.')
#     parser.add_argument('--parallelized',
#                         type=int,
#                         required=False,
#                         default=0,
#                         help='Tell the mapper node to be parallelized.')
#     parser.add_argument('--hide_windows',
#                         type=int,
#                         required=False,
#                         default=0,
#                         help='Disable windows.')
#     parser.add_argument('--debug',
#                         type=int,
#                         required=False,
#                         default=0,
#                         help='Debug mode, save evaluation results.')
    
#     args, ros_args = parser.parse_known_args()
    
#     # ros_args = dict([arg.split(':=') for arg in ros_args])
    
#     rospy.init_node("lidar_mapper_node", anonymous=True, log_level=rospy.DEBUG if bool(args.debug) else rospy.INFO)
    
#     if args.mode == 'REPLAY' and args.actions is None:
#         parser.error('Replay mode requires actions to replay.')
    
#     if torch.cuda.is_available():
#         device = torch.device('cuda', args.gpu_id)
#     else:
#         rospy.logwarn('No GPU available.')
#         device = torch.device('cpu')
        
#     if not args.ros_dataloader:
#         os.chdir(PACKAGE_PATH)
#         rospy.loginfo(f'Current working directory: {os.getcwd()}')
#         with open(args.config) as f:
#             config = json.load(f)
#             if 'env' in config:
#                 config['env']['config'] = os.path.abspath(
#                     os.path.join(os.path.dirname(args.config), os.pardir, os.pardir, config['env']['config']))
#             if 'sensor' in config:
#                 config['sensor']['config'] = os.path.abspath(
#                     os.path.join(os.path.dirname(args.config), os.pardir, os.pardir, config['sensor']['config']))
        
#         with open(args.user_config) as f:
#             user_config = json.load(f)
        
#         dataset:Union[HabitatDataset] = get_dataset(config, user_config, args.scene_id)
#     else:
#         dataset = None

#     hide_windows = bool(args.hide_windows)
#     if not hide_windows:
#         app = gui.Application.instance
#         app.initialize()
#     w = Visualizer(
#         args.config,
#         GlobalState(args.mode),
#         1 if hide_windows else app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE)),
#         device,
#         args.actions,
#         dataset,
#         bool(args.parallelized),
#         hide_windows,
#         bool(args.debug))
#     if hide_windows:
#         rospy.spin()
#     else:
#         app.run()
    
#     rospy.loginfo(f'{PROJECT_NAME} mapper node finished.')