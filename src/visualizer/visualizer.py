from collections import OrderedDict
import os

from mapper.mapper import Bbox, Mapper
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
from enum import Enum
import numpy as np
import json
from queue import Queue
import threading
import time
from typing import Dict, List, Tuple, Union
# import quaternion
# from imgviz import depth2rgb
import cv2
import matplotlib.pyplot as plt

import open3d as o3d
from open3d.visualization import rendering, gui

import rospy

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import threading
import time
import rospy
import numpy as np
import open3d as o3d
# 临时修复 numpy 的弃用问题
np.float = np.float64
import ros_numpy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from open3d_ros_helper import open3d_ros_helper as orh    

import rospy
from  nav_msgs.msg  import Odometry
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import threading
import time
import tf.transformations as tf_trans
import numpy as np
import torch


class Visualizer:
    
    def __init__(self,device:torch.device,hide_windows:bool,):
        print("Visualizer init")
        
        self.__device = device
        self.__hide_windows = hide_windows
        
        self.global_pcd = o3d.geometry.PointCloud()
        self.downsampled =  o3d.geometry.PointCloud()
        self.color_pcd_lock = threading.Lock()
        self.bbox_lock = threading.Lock()
        
        self.global_bbox_list = []
        self.bbox_dict = {}
        self.marker_cache = OrderedDict()
        
        self.pose_list = []
        self.pose_list_for_bbox = []
        
        self.labels = []
        
        self.__update_color_pcd_thread = threading.Thread(
            target=self.__update_color_pcd,
            name='__update_color_pcd',
            daemon=True)
        self.__update_color_pcd_thread.start()
        
        # self.__update_bbox_thread = threading.Thread(
        #     target=self.__update_bbox,
        #     name='__update_bbox',
        #     daemon=True)
        # self.__update_bbox_thread.start()
        
        self.__update_main_thread = threading.Thread(
            target=self.__update_main,
            name='__update_main',
            daemon=True)
        
        self.__init_o3d_elements()
        
        
        
        self.__mapper = Mapper()
        
        self.__init_window()
        
        
        self.__update_main_thread.start()
    
    def __init_o3d_elements(self):
        self.__device_o3c = o3d.core.Device(self.__device.type, self.__device.index)
        
        # NOTE: Independent Open3D elements
        self.__o3d_meshes:Dict[str, o3d.geometry.TriangleMesh] = {
            'scene_mesh': None
        }
        self.__o3d_pcd:Dict[str, o3d.t.geometry.PointCloud] = {
            'current_pcd': None,
        }
        
        
        # 渲染材质
        self.__o3d_materials:Dict[str, rendering.MaterialRecord] = {
            'lit_mat': None,
            'lit_mat_transparency': None,
            'unlit_mat': None,
            'unlit_line_mat': None,
            'unlit_line_mat_slim': None,
        }
            
        
        self.__o3d_materials['lit_mat'] = rendering.MaterialRecord()
        self.__o3d_materials['lit_mat'].shader = 'defaultLit'
        self.__o3d_materials['lit_mat_transparency'] = rendering.MaterialRecord()
        self.__o3d_materials['lit_mat_transparency'].shader = 'defaultLitTransparency'
        self.__o3d_materials['lit_mat_transparency'].has_alpha = True
        self.__o3d_materials['lit_mat_transparency'].base_color = [1.0, 1.0, 1.0, 0.9]
        self.__o3d_materials['unlit_mat'] = rendering.MaterialRecord()
        self.__o3d_materials['unlit_mat'].shader = 'defaultUnlit'
        self.__o3d_materials['unlit_mat'].sRGB_color = True
        self.__o3d_materials['unlit_line_mat'] = rendering.MaterialRecord()
        self.__o3d_materials['unlit_line_mat'].shader = 'unlitLine'
        self.__o3d_materials['unlit_line_mat'].line_width = 1.0
        self.__o3d_materials['unlit_line_mat_slim'] = rendering.MaterialRecord()
        self.__o3d_materials['unlit_line_mat_slim'].shader = 'unlitLine'
        self.__o3d_materials['unlit_line_mat_slim'].line_width = 2.0
        self.__o3d_materials['gaussian_mat'] = rendering.MaterialRecord()
        self.__o3d_materials['gaussian_mat'].shader = 'defaultUnlit'
        # self.__o3d_materials['gaussian_mat'].sRGB_color = True
        # NOTE: Default point size is 3.0
        # self.__o3d_materials['gaussian_mat'].point_size = 6.0
    
    def __init_window(self):
        self.__window:gui.Window = gui.Application.instance.create_window("lidar omni", 1920, 1080)
        self.__window.show(False)
        
        em = self.__window.theme.font_size
        margin = 0.5 * em
        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))
        
        margins = gui.Margins(vspacing)
        self.__panel_control = gui.Vert(spacing, margins)
        
        self.__widget_3d = gui.SceneWidget()
        self.__widget_3d.scene = rendering.Open3DScene(self.__window.renderer)
        self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.__widget_3d.scene.scene.set_sun_light([-0.2, 1.0 ,0.2], [1.0, 1.0, 1.0], 70000)
        self.__widget_3d.scene.scene.enable_sun_light(True)
        
        #####
        # NOTE: Widgets for control panel
        
        self.__panel_control.add_fixed(vspacing)
        
        self.__panel_control.add_child(gui.Label('3D Visualization Settings'))
        
        gt_mesh_cb = gui.Checkbox('Show gt mesh')
        gt_mesh_cb.set_on_checked(self.__on_gt_mesh_cb_changed)
        self.__panel_control.add_child(gt_mesh_cb)
    
        ########
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.__widget_3d.scene.add_geometry('coordinate_frame', coordinate_frame, self.__o3d_materials['lit_mat'])
        self.__widget_3d.scene.add_geometry('downsampled', self.downsampled, self.__o3d_materials['lit_mat'])
        
        
        
        
        self.__window.add_child(self.__widget_3d)
        self.__window.add_child(self.__panel_control)
        
        self.__window.set_on_layout(self.__window_on_layout)
        self.__window.set_on_close(self.__window_on_close)
        
        self.__window.show(True)
        
    # NOTE: callback functions for GUI

    def __window_on_layout(self, ctx:gui.LayoutContext):
        em = ctx.theme.font_size

        panel_width = 15 * em
        rect:gui.Rect = self.__window.content_rect

        self.__panel_control.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        x = self.__panel_control.frame.get_right()
        
        # 3D widget width
        self.__widget_3d.frame = gui.Rect(x, rect.y, rect.get_right(), rect.height)

        return
        
    def __window_on_close(self) -> bool:
        self.__update_main_thread.join()
        gui.Application.instance.quit()
        return True
    
    def __on_gt_mesh_cb_changed(self, checked):
        if checked:
            gt_mesh = o3d.io.read_triangle_mesh("/media/user/Passport/dataset/matterport/v1/tasks/17DRP5sb8fy/17DRP5sb8fy_semantic.ply")
            vertices = np.asarray(gt_mesh.vertices)
            z_threshold = 2.0
            indices = np.where(vertices[:, 2] < z_threshold)[0]
            new_mesh = gt_mesh.select_by_index(indices)
            self.__widget_3d.scene.add_geometry('gt_mesh', new_mesh, self.__o3d_materials['lit_mat'])
        else:
            self.__widget_3d.scene.remove_geometry('gt_mesh')
    
    # update dataset
    def __update_color_pcd(self):
        rospy.Subscriber("/state_estimation", Odometry, self.__pose_callback) # recieve pose, maitain a cache
        rospy.Subscriber("/object_markers",MarkerArray, self.__bbox_callback) # recieve bbox, maitain a cache
        rospy.Subscriber("/sensor_scan_rgb", PointCloud2, self.__rbg_pcd_callback) # recieve color_pcd, sync color_pcd and pose
    
    
    # ros callback functions
    def __pose_callback(self,data):
        
        self.pose_list.append(data)

        # 如果pose_list太大，删除最旧的pose
        if len(self.pose_list) > 500:
            self.pose_list.pop(0)
            
            
    def __rbg_pcd_callback(self,msg):
        # sync color_pcd and pose
        closest_pose = min(self.pose_list, key=lambda pose: abs(pose.header.stamp.to_nsec() - msg.header.stamp.to_nsec()))
        print("close_pose time",closest_pose.header.stamp,"rgb pcd time",msg.header.stamp)
        translation = [closest_pose.pose.pose.position.x, closest_pose.pose.pose.position.y, closest_pose.pose.pose.position.z]
        rotation = [closest_pose.pose.pose.orientation.x, closest_pose.pose.pose.orientation.y, closest_pose.pose.pose.orientation.z, closest_pose.pose.pose.orientation.w]
        rotation_matrix = tf_trans.quaternion_matrix(rotation)
        transform_matrix = tf_trans.compose_matrix(translate=translation)
        transform_matrix[:3, :3] = rotation_matrix[:3, :3]
        point_cloud_o3d = orh.rospc_to_o3dpc(msg)
        points_np = np.asarray(point_cloud_o3d.points)
        points_homogeneous = np.hstack((points_np, np.ones((points_np.shape[0], 1))))
        transformed_points_homogeneous = transform_matrix @ points_homogeneous.T
        transformed_points = transformed_points_homogeneous[:3, :].T
        point_cloud_o3d.points = o3d.utility.Vector3dVector(transformed_points)
        
        # acquire the lock, update the color pcd
        self.color_pcd_lock.acquire()
        try:
            self.downsampled = self.__mapper.update_color_pcd(self.global_pcd ,point_cloud_o3d)
        except Exception as e:
            print(f"Error during point cloud transformation: {e}")
        finally:
            self.color_pcd_lock.release()
            
        # sync color_pcd and bbox
        syn_bbox = self.get_marker_array(msg.header.stamp)
        # print("syn_bbox",syn_bbox.markers[0].header.stamp)
        
        # acquire the lock, update the global bbox list
        
        self.bbox_lock.acquire()
        try:
            if syn_bbox is not None:
                boxes = []
                # print("新的bbox来了")
                for bbox in syn_bbox.markers:
                    if bbox.id not in self.bbox_dict:
                        bbox_obj = Bbox(bbox.id, bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z, bbox.scale.x, bbox.scale.y, bbox.scale.z, bbox.ns)
                        self.bbox_dict[bbox.id] = bbox_obj
                    
                # print("bbox_dict",len(self.bbox_dict))
        except Exception as e:
            print(f"Error during bbox transformation: {e}")
        finally:
            self.bbox_lock.release()
                
                # quat = (bbox.pose.orientation.x, bbox.pose.orientation.y, bbox.pose.orientation.z, bbox.pose.orientation.w)
                # euler = tf_trans.euler_from_quaternion(quat)
                # heading = euler[2]  # Z轴旋转角度
                # # print("(x,y,z),(dx,dy,dz),heading,ns",(x,y,z),(dx,dy,dz),heading,bbox.ns)
                # box = detector.get_3d_box((x,y,z),(dx,dy,dz),heading)
                # box = box.transpose(1,0).ravel()
                # boxes.append(box)
                # # detector.display(boxes)
            
       
    def __bbox_callback(self, msg):
        # print("bbox_callback",msg)
        # global_bbox = data
        # print("Marker timestamp:", msg.markers[0].header.stamp) # they have the same time stamp, so we only need to use the first one
        timestamp = msg.markers[0].header.stamp
        self.marker_cache[timestamp] = msg
        while len(self.marker_cache) > 1000:
            self.marker_cache.popitem(last=False)
            
    def get_marker_array(self, timestamp):
        closest_timestamp = min(self.marker_cache.keys(), key=lambda t: abs(t - timestamp))
        return self.marker_cache.get(closest_timestamp)    
        
    def draw_bbox(self, bbox):
        # 创建一个表示bbox的立方体的顶点
        vertices = np.array([[bbox.position_x - bbox.scale_x / 2, bbox.position_y - bbox.scale_y / 2, bbox.position_z - bbox.scale_z / 2],
                            [bbox.position_x + bbox.scale_x / 2, bbox.position_y - bbox.scale_y / 2, bbox.position_z - bbox.scale_z / 2],
                            [bbox.position_x - bbox.scale_x / 2, bbox.position_y + bbox.scale_y / 2, bbox.position_z - bbox.scale_z / 2],
                            [bbox.position_x + bbox.scale_x / 2, bbox.position_y + bbox.scale_y / 2, bbox.position_z - bbox.scale_z / 2],
                            [bbox.position_x - bbox.scale_x / 2, bbox.position_y - bbox.scale_y / 2, bbox.position_z + bbox.scale_z / 2],
                            [bbox.position_x + bbox.scale_x / 2, bbox.position_y - bbox.scale_y / 2, bbox.position_z + bbox.scale_z / 2],
                            [bbox.position_x - bbox.scale_x / 2, bbox.position_y + bbox.scale_y / 2, bbox.position_z + bbox.scale_z / 2],
                            [bbox.position_x + bbox.scale_x / 2, bbox.position_y + bbox.scale_y / 2, bbox.position_z + bbox.scale_z / 2]])
        # 创建一个表示bbox的立方体的边
        lines = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]])
        # 创建一个LineSet对象
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([0, 0, 0] for _ in range(len(lines)))
        # 删除旧的bbox
        self.__widget_3d.scene.remove_geometry('bbox_' + str(bbox.id))
        # 将新的LineSet对象添加到场景中
        self.__widget_3d.scene.add_geometry('bbox_' + str(bbox.id), line_set, self.__o3d_materials['unlit_line_mat'])
        
        
        # 在bbox的右上角添加一个文本标签
        label = self.__widget_3d.add_3d_label([bbox.position_x + bbox.scale_x / 2, bbox.position_y + bbox.scale_y / 2, bbox.position_z + bbox.scale_z / 2], str(bbox.ns))
        self.labels.append(label)
        label_count = len(self.labels)
        # print("The number of 3D labels is:", label_count)
    
    
    
    
    def __update_ui_frame(self):
        self.downsampled.scale(1, center=self.downsampled.get_center())
        print("downsampled",len(self.downsampled.points))
        gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_frame())
        
    def __update_main_thread_ui_frame(self):        
        if self.downsampled is not None:
            self.__widget_3d.scene.remove_geometry('downsampled')
            self.__widget_3d.scene.add_geometry('downsampled', self.downsampled, self.__o3d_materials['lit_mat'])
            self.__widget_3d.scene.show_geometry('downsampled', True)
            
        self.bbox_lock.acquire()
        
        for label in self.labels:
            self.__widget_3d.remove_3d_label(label)
            self.labels.remove(label)
        
        for bbox_id, bbox in self.bbox_dict.items():
            self.draw_bbox(bbox)
        self.bbox_lock.release()

        return

        
    def __update_main(self):

        while not rospy.is_shutdown():
            
            # 获取锁，更新可视化
            self.color_pcd_lock.acquire()
            self.__update_ui_frame()
            self.color_pcd_lock.release()

            # 每秒刷新一次
            time.sleep(1)