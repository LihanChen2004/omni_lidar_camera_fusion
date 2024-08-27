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
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist, PoseStamped, Point, Pose

from dataloader import RGBDSensor, PoseDataType,  load_scene_mesh, dataset_config_to_ros, convert_to_c2w_opencv
from mapper.mapper import Mapper, MapperState, GaussianColorType
from utils.gui_utils import OPENCV_TO_OPENGL, SmallGaussianPacket, PoseChangeType, create_frustum, rgbd_to_pointcloud, pose_to_matrix, matrix_to_pose, rotation_matrix_from_vectors, c2w_topdown_to_world, c2w_world_to_topdown, is_pose_changed, get_horizon_bound_topdown, update_traj, config_topdown_info, visualize_agent
from utils.graphics_utils import fov2focal
from utils.camera_utils import Camera
from utils.logging_utils import Log
from utils import start_timing, end_timing, PROJECT_NAME, GlobalState
from dataloader.dataloader import HabitatDataset

from activegauss.msg import frame
from activegauss.srv import\
    GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
        ResetEnv, ResetEnvResponse, ResetEnvRequest,\
            GetTopdown, GetTopdownRequest, GetTopdownResponse,\
                GetTopdownConfig, GetTopdownConfigRequest, GetTopdownConfigResponse,\
                    SetPlannerState, SetPlannerStateRequest, SetPlannerStateResponse,\
                        SetMapper, SetMapperRequest, SetMapperResponse,\
                            GetOpacity, GetOpacityRequest, GetOpacityResponse

CURRENT_FRUSTUM = {
    'color': [0.961, 0.475, 0.000],
    'scale': 0.2,
    'material': 'unlit_line_mat',
}
KEYFRAME_FRUSTUM = {
    'color': [0.0, 0.0, 1.0],
    'scale': 0.05,
    'material': 'unlit_line_mat',
}
CURRENT_AGENT = {
    'color': [0.961, 0.475, 0.000],
    'material': 'lit_mat_transparency',
}
CURRENT_HORIZON = {
    'color': [0.0, 1.0, 0.0],
}
VORONOI_GRAPH = {
    'nodes_color': [0.180,0.8,0.443],
    'ridges_color': [0.204,0.596,0.859],
    'nodes_radius': 0.05,
    'material': 'unlit_mat',
}
    
class Visualizer:
        
    class LocalDatasetState(Enum):
        INITIALIZING = 0
        INITIALIZED = 1
        RUNNING = 2
        
    class QueryTopdownFlag(Enum):
        NONE = 0
        ARRIVED = 1
        RUNNING = 2
        MANUAL = 3
        
    class QueryUncertaintyFlag(Enum):
        NONE = 0
        GLOBAL = 1
        LOCAL = 2
        RUNNING = 3
        MANUAL = 4
        
    def __init__(self,
                 config_url:str,
                 init_state:GlobalState,
                 font_id:int,
                 device:torch.device,
                 actions_url:str,
                 local_dataset:Union[HabitatDataset],
                 parallelized:bool,
                 hide_windows:bool,
                 debug:bool):
        # load local dataset
        if self.__local_dataset is not None:
            self.__local_dataset_state = self.LocalDatasetState.INITIALIZING
            self.__local_dataset_condition = threading.Condition()
            self.__local_dataset_pose_pub = rospy.Publisher('orb_slam3/camera_pose', PoseStamped, queue_size=1)
            self.__local_dataset_pose_ros = None
            self.__local_dataset_thread = threading.Thread(
                target=self.__update_dataset,
                name='UpdateDataset',
                daemon=True)
            self.__local_dataset_thread.start()
            self.__local_dataset_label = gui.Label('')
            self.__local_dataset_label.font_id = font_id
            _, step_num = self.__local_dataset.get_step_info() # update step_num
            
        
        self.__update_main_thread = threading.Thread(
            target=self.__update_main,
            name='UpdateMain',
            daemon=True)
        
        scene_mesh = self.__init_dataset()
        
        self.__init_o3d_elements()
        
        bbox = self.__bbox_visualize.copy()
        frame_first = self.__frames_cache.get()
        c2w = frame_first['c2w'].detach().cpu().numpy()
        agent_sensor = c2w[self.__height_direction[0], 3]
        agent_height_start = agent_sensor - self.__rgbd_sensor.position[self.__height_direction[0]]
        agent_height_end = agent_height_start + self.__dataset_config.agent_height
        
        

        current_pcd:o3d.geometry.PointCloud = rgbd_to_pointcloud(
            frame_first['rgb'].detach().cpu().numpy(),
            np.ones_like(frame_first['depth'].detach().cpu().numpy(), dtype=np.float32) * self.__rgbd_sensor.depth_max,
            c2w,
            self.__o3d_const_camera_intrinsics_o3c,
            1000,
            self.__rgbd_sensor.depth_max * 2,
            self.__device_o3c).to_legacy()
        current_pcd_bbox:o3d.geometry.AxisAlignedBoundingBox = current_pcd.get_axis_aligned_bounding_box()
        if self.__height_direction[0] in [1, 2]:
            single_floor_height_start = agent_height_start - config['mapper']['single_floor']['expansion']['head']
            single_floor_height_end = agent_height_end + config['mapper']['single_floor']['expansion']['foot']
        elif self.__height_direction[0] == 0:
            single_floor_height_start = agent_height_start - config['mapper']['single_floor']['expansion']['foot']
            single_floor_height_end = agent_height_end + config['mapper']['single_floor']['expansion']['head']
        else:
            raise ValueError(f'Invalid height direction: {self.__height_direction}')
        bbox[self.__height_direction[0]][0] = max(
            single_floor_height_start,
            bbox[self.__height_direction[0]][0],
            current_pcd_bbox.get_min_bound()[self.__height_direction[0]])
        bbox[self.__height_direction[0]][1] = min(
            single_floor_height_end,
            bbox[self.__height_direction[0]][1],
            current_pcd_bbox.get_max_bound()[self.__height_direction[0]])
        assert bbox[self.__height_direction[0]][0] < bbox[self.__height_direction[0]][1], 'Invalid height dimension'
        if self.__frames_cache.empty(): self.__frames_cache.put(frame_first)
        
        bbox:np.ndarray = bbox + self.__bbox_padding *\
            np.reshape(np.ptp(bbox, axis=1), (3, 1)) *\
                np.array([-1, 1])
                
        
        
        self.__mapper = Mapper(
            config,
            self.__rgbd_sensor,
            self.__device,
            self.q_main2vis,
            self.__results_dir,
            step_num)
        
        
        rospy.Service('set_mapper', SetMapper, self.__set_mapper)
        
        self.__init_window(
            config['mapper']['interval_max_ratio'],
            bbox_o3d[self.__height_direction[0]][(1 - self.__height_direction[1]) // 2],
            (agent_height_start * (1 - self.__height_direction[1]) + agent_height_end * (1 + self.__height_direction[1])) / 2,
            font_id,
            scene_mesh)
        
        self.__update_main_thread.start()
    
    # NOTE: initialization functions
    
    def __init_dataset(self) -> o3d.geometry.TriangleMesh:
        self.__frames_cache:Queue[Dict[str, Union[int, torch.Tensor]]] = Queue(maxsize=1)
        self.__frame_c2w_last = None
        
        if self.__local_dataset is None:
            reset_env_service = rospy.ServiceProxy('reset_env', ResetEnv)
            rospy.wait_for_service('reset_env')
            
            reset_env_success:ResetEnvResponse = reset_env_service(ResetEnvRequest())
            
            self.__cmd_vel_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
            get_dataset_config_service = rospy.ServiceProxy('get_agent_config', GetDatasetConfig)
            rospy.wait_for_service('get_agent_config')
            
            self.__dataset_config:GetDatasetConfigResponse = get_dataset_config_service(GetDatasetConfigRequest())
        else:
            self.__local_dataset_condition.acquire()
            if self.__local_dataset_state == self.LocalDatasetState.INITIALIZING:
                self.__local_dataset_condition.wait()
            

        
        self.__rgbd_sensor = RGBDSensor(
            height=self.__dataset_config.rgbd_height,
            width=self.__dataset_config.rgbd_width,
            fx=self.__dataset_config.rgbd_fx,
            fy=self.__dataset_config.rgbd_fy,
            cx=self.__dataset_config.rgbd_cx,
            cy=self.__dataset_config.rgbd_cy,
            depth_min=self.__dataset_config.rgbd_depth_min,
            depth_max=self.__dataset_config.rgbd_depth_max,
            depth_scale=self.__dataset_config.rgbd_depth_scale,
            position=np.array([
                self.__dataset_config.rgbd_position.x,
                self.__dataset_config.rgbd_position.y,
                self.__dataset_config.rgbd_position.z]),
            downsample_factor=self.__dataset_config.rgbd_downsample_factor)

        if self.__local_dataset is None:
            rospy.Subscriber('sensor_scan_image', frame, self.__frame_callback)
            rospy.wait_for_message('frames', frame)
        else:
            if self.__local_dataset_state == self.LocalDatasetState.INITIALIZED:
                self.__local_dataset_condition.wait()
            self.__local_dataset_condition.notify_all()
            self.__local_dataset_condition.release()
        
        if os.path.exists(self.__dataset_config.scene_mesh_url):
            self.__scene_mesh_transform = pose_to_matrix(self.__dataset_config.scene_mesh_transform)
            scene_mesh, self.__bbox_visualize = load_scene_mesh(
                self.__dataset_config.scene_mesh_url,
                self.__scene_mesh_transform)
        else:
            scene_mesh = None
            self.__bbox_visualize = np.array([
                [self.__dataset_config.scene_bound_min.x, self.__dataset_config.scene_bound_max.x],
                [self.__dataset_config.scene_bound_min.y, self.__dataset_config.scene_bound_max.y],
                [self.__dataset_config.scene_bound_min.z, self.__dataset_config.scene_bound_max.z]])
        
        self.__agent_cylinder_mesh:o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.__dataset_config.agent_radius, height=self.__dataset_config.agent_height)
        self.__agent_cylinder_mesh.compute_vertex_normals()
        vector_end = np.zeros(3)
        vector_end[self.__height_direction[0]] = self.__height_direction[1]
        self.__agent_cylinder_mesh.rotate(
            rotation_matrix_from_vectors(np.array([0, 0, 1]), vector_end),
            np.zeros(3))
        self.__agent_cylinder_mesh.translate(
            (self.__dataset_config.agent_height / 2 - self.__rgbd_sensor.position[self.__height_direction[0]]) * vector_end)
        self.__agent_cylinder_mesh.paint_uniform_color(CURRENT_AGENT['color'])
        return scene_mesh
    
    def __init_o3d_elements(self):
        self.__device_o3c = o3d.core.Device(self.__device.type, self.__device.index)
        
        # NOTE: Independent Open3D elements
        self.__o3d_meshes:Dict[str, o3d.geometry.TriangleMesh] = {
            'scene_mesh': None
        }
        self.__o3d_pcd:Dict[str, o3d.t.geometry.PointCloud] = {
            'current_pcd': None,
        }
        
        self.__o3d_const_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.__rgbd_sensor.width,
            self.__rgbd_sensor.height,
            self.__rgbd_sensor.fx,
            self.__rgbd_sensor.fy,
            self.__rgbd_sensor.cx,
            self.__rgbd_sensor.cy)
        self.__o3d_const_camera_intrinsics_o3c = o3d.core.Tensor(self.__o3d_const_camera_intrinsics.intrinsic_matrix, device=self.__device_o3c)
        
        # 渲染材质
        self.__o3d_materials:Dict[str, rendering.MaterialRecord] = {
            'lit_mat': None,
            'lit_mat_transparency': None,
            'unlit_mat': None,
            'unlit_line_mat': None,
            'unlit_line_mat_slim': None,
        }
            
        if not self.__hide_windows:
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
            self.__o3d_materials['unlit_line_mat'].line_width = 5.0
            self.__o3d_materials['unlit_line_mat_slim'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat_slim'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat_slim'].line_width = 2.0
            self.__o3d_materials['gaussian_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['gaussian_mat'].shader = 'defaultUnlit'
            # self.__o3d_materials['gaussian_mat'].sRGB_color = True
            # NOTE: Default point size is 3.0
            # self.__o3d_materials['gaussian_mat'].point_size = 6.0
        
    def __init_window(self, interval_max_ratio:float, foot_value:float, head_value:float, font_id:int, scene_mesh:o3d.geometry.TriangleMesh=None):
        kf_every = self.__mapper.get_kf_every()
        assert 0 < kf_every, f'Invalid keyframe every: {kf_every}'
        map_every = self.__mapper.get_map_every()
        mapping_iters = self.__mapper.get_mapping_iters()
        assert 0 < map_every, f'Invalid map every: {map_every}'
        update_interval_max = int(max(interval_max_ratio * max(kf_every, map_every), mapping_iters))
        assert interval_max_ratio >= 1.0, f'Invalid interval ratio: {interval_max_ratio}'
        
        self.__open3d_gui_widget_last_state = dict()

        # NOTE: 裁剪mesh场景 相机坐标系下的 y轴方向 [-1.5, 0]
        self.__open3d_gui_widget_last_state['height_direction_bound_slider'] = [
            (foot_value * (1 + self.__height_direction[1]) + head_value * (1 - self.__height_direction[1])) / 2,
            (foot_value * (1 - self.__height_direction[1]) + head_value * (1 + self.__height_direction[1])) / 2]
        
        # self.__set_planner_state_service = rospy.ServiceProxy('set_planner_state', SetPlannerState)
        # rospy.wait_for_service('set_planner_state')
        
        if self.__hide_windows:
            self.__o3d_meshes['scene_mesh'] = o3d.geometry.TriangleMesh(scene_mesh)
            return
        
        else:
            # NOTE: Initialize GUI
            self.__window:gui.Window = gui.Application.instance.create_window(PROJECT_NAME, 1920, 1080)
            self.__window.show(False)
            
            em = self.__window.theme.font_size
            margin = 0.5 * em
            spacing = int(np.round(0.25 * em))
            vspacing = int(np.round(0.5 * em))
            
            margins = gui.Margins(vspacing)
            self.__panel_control = gui.Vert(spacing, margins)
            self.__panel_visualize = gui.Vert(spacing, margins)
            
            self.__widget_3d = gui.SceneWidget()
            self.__widget_3d.scene = rendering.Open3DScene(self.__window.renderer)
            self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
            self.__widget_3d.scene.scene.set_sun_light([-0.2, 1.0 ,0.2], [1.0, 1.0, 1.0], 70000)
            self.__widget_3d.scene.scene.enable_sun_light(True)
            self.__widget_3d.set_on_key(self.__widget_3d_on_key)
            
            # TODO: Add log information label
            
            self.__panel_control.add_fixed(vspacing)
            if self.__local_dataset is not None:
                self.__panel_control.add_child(gui.Label('Local Dataset Info'))
                self.__panel_control.add_child(self.__local_dataset_label)
            
            self.__panel_control.add_child(gui.Label('3D Visualization Settings'))
            
            panel_control_vgrid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))
            
            panel_control_vgrid.add_child(gui.Label('    Mapper Configurations'))
            # TODO: use checkbox to close all mapper configurations
            panel_control_vgrid.add_child(gui.Label(''))

            panel_control_vgrid.add_child(gui.Label('        Map Every'))
            self.__map_every_slider = gui.Slider(gui.Slider.INT)
            self.__map_every_slider.set_limits(1, update_interval_max)
            self.__map_every_slider.int_value = map_every
            self.__map_every_slider.set_on_value_changed(lambda value: self.__mapper.set_map_every(value))
            panel_control_vgrid.add_child(self.__map_every_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Keyframe Every'))
            self.__kf_every_slider = gui.Slider(gui.Slider.INT)
            self.__kf_every_slider.set_limits(1, update_interval_max)
            self.__kf_every_slider.int_value = kf_every
            self.__kf_every_slider.set_on_value_changed(lambda value: self.__mapper.set_kf_every(value))
            panel_control_vgrid.add_child(self.__kf_every_slider)
            

            panel_control_vgrid.add_child(gui.Label('    Global Status'))
            # TODO: use checkbox to close all global states
            panel_control_vgrid.add_child(gui.Label(''))
            
            view_gs_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
            view_gs_grid.add_child(gui.Label('        View Gaussians'))
            self.__view_gaussians_box = gui.Checkbox('')
            self.__view_gaussians_box.checked = True
            def view_gaussians_callback(checked:bool):
                self.followcam_chbox.enabled = checked
                self.staybehind_chbox.enabled = checked
                if checked is False:
                    # 取消查看高斯 则刷新白色背景
                    self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
                return gui.Checkbox.HANDLED
            self.__view_gaussians_box.set_on_checked(view_gaussians_callback)
            view_gs_grid.add_child(self.__view_gaussians_box)
            
            panel_control_vgrid.add_child(view_gs_grid)
            panel_control_vgrid.add_child(gui.Label(''))
            
            panel_control_vgrid.add_child(gui.Label("        Viewing options"))
            chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            self.followcam_chbox = gui.Checkbox("Follow Camera")
            self.followcam_chbox.checked = False
            chbox_tile.add_child(self.followcam_chbox)
            panel_control_vgrid.add_child(chbox_tile)
            
            panel_control_vgrid.add_child(gui.Label(''))
            
            # NOTE: 观察者视角：相机后面 or 相机本体
            self.staybehind_chbox = gui.Checkbox("From Behind")
            self.staybehind_chbox.checked = False
            chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            chbox_tile.add_child(self.staybehind_chbox)
            panel_control_vgrid.add_child(chbox_tile)
                
            panel_control_vgrid.add_child(gui.Label('        H Lower Bound'))
            self.__height_direction_lower_bound_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__height_direction_lower_bound_slider.set_limits(
                self.__bbox_visualize[self.__height_direction[0]][0],
                self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1])
            self.__height_direction_lower_bound_slider.double_value = self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0]
            self.__height_direction_lower_bound_slider.set_on_value_changed(
                lambda value: self.__height_direction_bound_slider_callback(value, 0))
            panel_control_vgrid.add_child(self.__height_direction_lower_bound_slider)
            
            panel_control_vgrid.add_child(gui.Label('        H Upper Bound'))
            self.__height_direction_upper_bound_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__height_direction_upper_bound_slider.set_limits(
                self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0],
                self.__bbox_visualize[self.__height_direction[0]][1] + 0.1)
            self.__height_direction_upper_bound_slider.double_value = self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1]
            self.__height_direction_upper_bound_slider.set_on_value_changed(
                lambda value: self.__height_direction_bound_slider_callback(value, 1))
            panel_control_vgrid.add_child(self.__height_direction_upper_bound_slider)
            
            # NOTE: Gaussian渲染模式
            panel_control_vgrid.add_child(gui.Label("        Rendering options"))
            self.__gaussian_color_combobox = gui.Combobox()
            self.__gaussian_color_type = None
            for gaussian_color_type in GaussianColorType:
                self.__gaussian_color_combobox.add_item(gaussian_color_type.value)
            self.__gaussian_color_combobox.selected_text = GaussianColorType.Color.value
            def gaussian_color_combobox_callback(color_type_name:str, color_type_index:int):
                self.__gaussian_color_type = GaussianColorType(color_type_name)
                return gui.Combobox.HANDLED
            self.__gaussian_color_combobox.set_on_selection_changed(gaussian_color_combobox_callback)
            gaussian_color_combobox_callback(self.__gaussian_color_combobox.selected_text, None)
            panel_control_vgrid.add_child(self.__gaussian_color_combobox)
            
            # NOTE: Gaussian渲染尺度 scale
            panel_control_vgrid.add_child(gui.Label('        Gaussian Scale (0-1)'))
            self.__gaussian_scale_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__gaussian_scale_slider.set_limits(0.001, 1.0)
            self.__gaussian_scale_slider.double_value = 1.0
            panel_control_vgrid.add_child(self.__gaussian_scale_slider)
            
            panel_control_vgrid.add_child(gui.Label(''))
            panel_control_vgrid.add_child(gui.Label(''))
            
            origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            self.__widget_3d.scene.add_geometry('origin_mesh', origin_mesh, self.__o3d_materials['lit_mat'])
            panel_control_vgrid.add_child(gui.Label('        Origin Mesh'))
            origin_mesh_box = gui.Checkbox('')
            origin_mesh_box.checked = True
            origin_mesh_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('origin_mesh', checked))
            panel_control_vgrid.add_child(origin_mesh_box)
            
            if scene_mesh is not None:
                self.__o3d_meshes['scene_mesh'] = o3d.geometry.TriangleMesh(scene_mesh)
                panel_control_vgrid.add_child(gui.Label('        Ground Truth Mesh'))
                self.__scene_mesh_box = gui.Checkbox('')
                self.__scene_mesh_box.checked = False
                self.__scene_mesh_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('scene_mesh', checked))
                panel_control_vgrid.add_child(self.__scene_mesh_box)
            else:
                self.__scene_mesh_box = None
            
            if scene_mesh is not None:
                self.__update_mesh('scene_mesh', self.__scene_mesh_box.checked, self.__o3d_materials['lit_mat'])
            
            panel_control_vgrid.add_child(gui.Label('    Local Status'))
            # TODO: use checkbox to close all local states
            panel_control_vgrid.add_child(gui.Label(''))
            
            panel_control_vgrid.add_child(gui.Label('        Current Frustum'))
            self.__current_frustum_box = gui.Checkbox('')
            self.__current_frustum_box.checked = True
            self.__current_frustum_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_frustum', checked))
            panel_control_vgrid.add_child(self.__current_frustum_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current Agent'))
            self.__current_agent_box = gui.Checkbox('')
            self.__current_agent_box.checked = True
            self.__current_agent_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_agent', checked))
            panel_control_vgrid.add_child(self.__current_agent_box)
            
            # NOTE: 当前观察区域的边界框bbox
            panel_control_vgrid.add_child(gui.Label('        Current Horizon'))
            self.__current_horizon_box = gui.Checkbox('')
            self.__current_horizon_box.checked = True
            self.__current_horizon_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_horizon', checked))
            panel_control_vgrid.add_child(self.__current_horizon_box)
            
            # NOTE: 当前观察区域的点云
            panel_control_vgrid.add_child(gui.Label('        Current PCD'))
            self.__current_pcd_box = gui.Checkbox('')
            self.__current_pcd_box.checked = False
            self.__current_pcd_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_pcd', checked))
            panel_control_vgrid.add_child(self.__current_pcd_box)
            
            # NOTE: 相机的轨迹
            panel_control_vgrid.add_child(gui.Label('        Camera Trajectory'))
            self.__cam_traj_box = gui.Checkbox('')
            self.__cam_traj_box.checked = False
            self.__cam_traj_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('cam_traj', checked))
            panel_control_vgrid.add_child(self.__cam_traj_box)
            
            # NOTE: 维诺图节点可视化
            panel_control_vgrid.add_child(gui.Label('        Voronoi 3D'))
            self.__voronoi_3d_box = gui.Checkbox('')
            self.__voronoi_3d_box.checked = False
            self.__voronoi_3d_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('voronoi_nodes', checked))
            panel_control_vgrid.add_child(self.__voronoi_3d_box)
            
            # NOTE: Keyframe可视化
            panel_control_vgrid.add_child(gui.Label('        Keyframes'))
            self.__keyframe_box = gui.Checkbox('')
            self.__keyframe_box.checked = False
            panel_control_vgrid.add_child(self.__keyframe_box)
                
            self.__panel_control.add_child(panel_control_vgrid)
            
            # NOTE: Widgets for visualize panel
            panel_visualize_tabs = gui.TabControl()
            panel_visualize_tab_margin = gui.Margins(0, int(np.round(0.5 * em)), em, em)
            
            tab_live_view = gui.ScrollableVert(0, panel_visualize_tab_margin)
            
            image_placeholder_numpy = np.zeros((self.__rgbd_sensor.height, self.__rgbd_sensor.width * 2, 3), dtype=np.uint8)
            image_placeholder = o3d.geometry.Image(image_placeholder_numpy)
            
            if self.__is_debug:
                save_current_data_button = gui.Button('Save Current Data')
                self.__save_current_data = lambda: self.__save_current_data_callback()
                save_current_data_button.set_on_clicked(self.__save_current_data)
                tab_live_view.add_child(save_current_data_button)
                tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('  RGBD Live Image'))
            self.__rgbd_live_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_live_image)
            tab_live_view.add_fixed(vspacing)
            
            render_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
            render_grid.add_child(gui.Label('  Rendered RGBD Image'))
            self.__render_box = gui.Checkbox('')
            self.__render_box.checked = True
            def render_box_callback(checked:bool):
                self.__render_every_slider.enabled = checked
                return gui.Checkbox.HANDLED
            self.__render_box.set_on_checked(render_box_callback)
            render_grid.add_child(self.__render_box)
            
            render_grid.add_child(gui.Label('    Render Every'))
            self.__render_every_slider = gui.Slider(gui.Slider.INT)
            self.__render_every_slider.set_limits(1, update_interval_max)
            self.__render_every_slider.int_value = 1
            self.__render_every_slider.enabled = self.__render_box.checked
            render_grid.add_child(self.__render_every_slider)
            
            tab_live_view.add_child(render_grid)
            
            self.__rgbd_render_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_render_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('RGB-Depth Loss'))
            self.__rgbd_loss_vis_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_loss_vis_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Local Uncertainty'))
            self.__local_uncertainty_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__local_uncertainty_map_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Visible Map (RGB)'))
            self.__topdown_visible_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_visible_map_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('  Free Space Map'))
            self.__topdown_free_map_image = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_free_map_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Visible Map (binary)'))
            self.__topdown_visible_map_image_binary = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_visible_map_image_binary)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('Free Space Map (binary)'))
            self.__topdown_free_map_image_binary = gui.ImageWidget()
            tab_live_view.add_child(self.__topdown_free_map_image_binary)
            tab_live_view.add_fixed(vspacing)
            
            # NOTE: Information
            tab_live_info = gui.ScrollableVert(0, panel_visualize_tab_margin)
            self.num_gaussians_info = gui.Label("Number of Gaussians: ")
            tab_live_info.add_child(self.num_gaussians_info)
            self.cam_pose_info = gui.Label("Camera Pose: ")
            tab_live_info.add_child(self.cam_pose_info)
            self.render_use_time_info = gui.Label("Render Use Time: ")
            tab_live_info.add_child(self.render_use_time_info)
            
            panel_visualize_tabs.add_tab('Live View', tab_live_view)
            panel_visualize_tabs.add_tab('Live Info', tab_live_info)
            
            self.__panel_visualize.add_child(panel_visualize_tabs)
            
            self.__window.add_child(self.__panel_control)
            self.__window.add_child(self.__widget_3d)
            self.__window.add_child(self.__panel_visualize)
            
            self.__window.set_on_layout(self.__window_on_layout)
            self.__window.set_on_close(self.__window_on_close)
            
            # NOTE: Setup camera
            center = np.average(self.__bbox_visualize, axis=1)
            center[self.__height_direction[0]] = 0
            bbox = o3d.geometry.AxisAlignedBoundingBox(self.__bbox_visualize[:, 0], self.__bbox_visualize[:, 1])
            self.__widget_3d.setup_camera(60.0, bbox, center)
            height_location = np.max(np.ptp(self.__bbox_visualize, axis=1))
            center_bias = np.zeros(3)
            center_bias[self.__height_direction[0]] = self.__height_direction[1] * (height_location - 1)
            center_bias[(self.__height_direction[0] + 1) % 3] = 5 * self.__height_direction[1]
            up_vector = np.zeros(3)
            up_vector[(self.__height_direction[0] + 1) % 3] = -self.__height_direction[1]
            self.__widget_3d.look_at(center, center + center_bias, up_vector)
            self.__window.show(True)
    
    def __init_debug_info(self):
        self.__get_debug_data_flag = False
    
        opacity_dir = os.path.join(self.runtime_data_dir, 'opacity')
        if not os.path.exists(opacity_dir): os.makedirs(opacity_dir)
        current_vis_data_dir = os.path.join(self.runtime_data_dir, 'current_vis_data')
        if not os.path.exists(current_vis_data_dir): os.makedirs(current_vis_data_dir)
        
        # Results data
        
        self.__debug_info['opacity_dir'] = opacity_dir
        self.__debug_info['current_vis_data_dir'] = current_vis_data_dir
        self.__debug_info['f_num_gaussians'] = os.path.join(self.__results_dir, 'num_gaussians.txt')
        self.__debug_info['current_vis_data'] = dict()

    
    # NOTE: main function
    
    def __update_main(self):
        # TODO: Initialize GUI
        frame_id = None
        frame_last_received = None
        rendering_rgbd_last_frame_id = -np.inf
        self.__gaussian_for_render = None
        self.__gaussian_packet = None # 当前的高斯场景
        self.__render_topdown_free_map = None
        self.__render_topdown_visible_map = None
        self.voronoi_3d_ridges = None
        self.gaussians_num = 0
        
        while True:
            timing_logical_operations = start_timing()
            
            # NOTE: Get observation
            if self.__frames_cache.empty():
                frame_current = None
            else:
                frame_current = self.__frames_cache.get()
                if self.__is_debug:
                    self.__get_debug_data_flag = True
                if frame_id is None:
                    frame_id = 0
                    self.__update_ui_frame(frame_current)
                    self.__get_topdown_flag = self.QueryTopdownFlag.MANUAL
                    self.__get_uncertainty_flag = self.QueryUncertaintyFlag.MANUAL
                else:
                    frame_id += 1
                frame_current['frame_id'] = frame_id
                frame_last_received = frame_current.copy()
            
                
            assert frame_id is not None, 'Initialize failed'


            mapper_state = self.__mapper.run(frame_current)
                
            # render topdown view
            rerender_topdown_flag = False
            if (self.__gaussian_packet is not None and\
                        self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]): #TODO: 这里应该是确保高斯场景, 而且不是一直刷新
                # render topdown view
                topdown_cam = self.get_topdown_cam()
                with self.__use_gaussian_condition:
                    self.__use_gaussian_condition.acquire()
                    gaussian_free_map = {k: v.clone() for k, v in self.__gaussian_packet.params.items()}
                    gaussian_free_map = self.__cut_gaussian_by_height(
                                                    gaussian_free_map, 
                                                    self.__agent_head, 
                                                    self.__agent_foot - self.__agent_foot_adjust)
                    self.__render_topdown_free_map = self.__mapper.render_o3d_image(gaussian_free_map, topdown_cam, scale_modifier=0.01, gaussian_color_type=GaussianColorType.Opacity)
                    self.__render_topdown_visible_map = self.__mapper.render_o3d_image(self.__gaussian_for_render, topdown_cam, scale_modifier=0.01)
                    self.__use_gaussian_condition.release()
                rerender_topdown_flag = True
                
            # NOTE: Update topdown map
            if (self.__get_topdown_flag in [self.QueryTopdownFlag.ARRIVED, self.QueryTopdownFlag.RUNNING]) or \
                rerender_topdown_flag == True:
                topdown_gray_free_map = cv2.cvtColor(
                    self.__render_topdown_free_map,
                    cv2.COLOR_RGB2GRAY)
                topdown_gray_visible_map = cv2.cvtColor(
                    self.__render_topdown_visible_map,
                    cv2.COLOR_RGB2GRAY)
                self.__topdown_info['free_map_binary'] = (topdown_gray_free_map <= 15).astype(np.uint8)
                self.__topdown_info['visible_map_binary'] = (topdown_gray_visible_map == 255).astype(np.uint8)
                
                self.__topdown_info['free_map_cv2'] = self.__render_topdown_free_map
                self.__topdown_info['free_map_binary_cv2'] = cv2.cvtColor(
                    self.__topdown_info['free_map_binary'] * 255,
                    cv2.COLOR_GRAY2RGB)
                self.__topdown_info['visible_map_cv2'] = self.__render_topdown_visible_map
                self.__topdown_info['visible_map_binary_cv2'] = cv2.cvtColor(
                    self.__topdown_info['visible_map_binary'] * 255,
                    cv2.COLOR_GRAY2RGB)
                self.__update_ui_topdown()
                
                if (self.__get_topdown_flag == self.QueryTopdownFlag.RUNNING) or\
                    (self.__get_topdown_flag == self.QueryTopdownFlag.ARRIVED):
                    self.__get_topdown_flag = self.QueryTopdownFlag.NONE
                    with self.__get_topdown_condition:
                        self.__get_topdown_condition.notify_all()
                        self.__get_topdown_condition.wait()
                elif self.__get_topdown_flag == self.QueryTopdownFlag.MANUAL:
                    self.__get_topdown_flag = self.QueryTopdownFlag.NONE
                    with self.__get_topdown_condition:
                        self.__get_topdown_condition.notify_all()
                        
            # NOTE: Update uncertainty map
            rerender_voronoi_3d_flag = False
            if self.__get_uncertainty_flag in [self.QueryUncertaintyFlag.GLOBAL, self.QueryUncertaintyFlag.LOCAL] and frame_last_received is not None:
                if self.__get_uncertainty_flag == self.QueryUncertaintyFlag.GLOBAL:
                    self.voronoi_3d_vertices = []
                    frame_last_received_c2w = frame_last_received['c2w'].detach().cpu().numpy()
                    height = frame_last_received_c2w[self.__height_direction[0], 3]
                    # 删除self.__debug_info['opacity_dir']文件夹中的所有png文件
                    if self.__debug_info is not None:
                        if os.path.exists(self.__debug_info['opacity_dir']):
                            for file in os.listdir(self.__debug_info['opacity_dir']):
                                if file.endswith('.png'):
                                    os.remove(os.path.join(self.__debug_info['opacity_dir'], file))
                    for idx, node in enumerate(self.__mapper.voronoi_nodes):
                        timing_get_uncertainty = start_timing()
                        cur_id = node['id']
                        node_position = node['position']
                        if self.__is_debug:
                            rgb_opacity_vis, node['uncertainty'], node['volume'] = self.__mapper.get_global_uncertainty(
                                frame_last_received, cur_id, node_position, scale_modifier=1.0, is_debug=self.__is_debug
                            )
                            if rgb_opacity_vis is not None:
                                cv2.imwrite(os.path.join(self.__debug_info['opacity_dir'], f'{cur_id}.png'), rgb_opacity_vis)
                        else:
                            _, node['uncertainty'], node['volume'] = self.__mapper.get_global_uncertainty(
                                frame_last_received, cur_id, node_position, scale_modifier=1.0
                            )
                        Log(f'Get global uncertainty used {end_timing(*timing_get_uncertainty):.2f} ms', tag='activeGauss')
                        
                        if not self.__hide_windows:
                            pose = np.eye(4)
                            pose[:3, 3] = node_position
                            pose[:3, 3][self.__height_direction[0]] = height
                            pose = OPENCV_TO_OPENGL @ pose @ OPENCV_TO_OPENGL
                            self.voronoi_3d_vertices.append(pose[:3, 3].tolist())
                    if not self.__hide_windows:
                        # Load voronoi graph points and lines
                        points = np.load(os.path.join(self.__results_dir, 'runtime_data', 'voronoi_graph_3d_points.npy')) # in OpenCV coordinate
                        points[:, self.__height_direction[0]] = height
                        
                        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
                        points_transformed = (OPENCV_TO_OPENGL @ points_homogeneous.T)[:3, :].T
                        
                        lines = np.load(os.path.join(self.__results_dir, 'runtime_data', 'voronoi_graph_3d_lines.npy'))
                        
                        # Create and color LineSet
                        self.voronoi_3d_ridges = o3d.geometry.LineSet()
                        self.voronoi_3d_ridges.points = o3d.utility.Vector3dVector(points_transformed)
                        self.voronoi_3d_ridges.lines = o3d.utility.Vector2iVector(lines)
                        self.voronoi_3d_ridges.paint_uniform_color(VORONOI_GRAPH['ridges_color'])
                        rerender_voronoi_3d_flag = True
                        
                elif self.__get_uncertainty_flag == self.QueryUncertaintyFlag.LOCAL:
                    # NOTE: 对当前agent位置的周围进行uncertainty计算
                    timing_get_uncertainty = start_timing()
                    self.local_node = dict()
                    rgb_opacity_vis, self.local_node['uncertainty'], self.local_node['best_pose'] = self.__mapper.get_local_uncertainty(frame_last_received, scale_modifier=1.0, is_debug=True)
                    if rgb_opacity_vis is not None:
                        # show local uncertainty image
                        self.__uncertainty_info['local_uncertainty_cv2'] = cv2.cvtColor(
                            rgb_opacity_vis,
                            cv2.COLOR_RGB2BGR)
                        self.__update_ui_uncertainty()
                    Log(f'Get local uncertainty used {end_timing(*timing_get_uncertainty):.2f} ms', tag='activeGauss')
                    
                    
                if (self.__get_uncertainty_flag == self.QueryUncertaintyFlag.RUNNING) or\
                    (self.__get_uncertainty_flag == self.QueryUncertaintyFlag.GLOBAL) or\
                        (self.__get_uncertainty_flag == self.QueryUncertaintyFlag.LOCAL):
                    self.__get_uncertainty_flag = self.QueryUncertaintyFlag.NONE
                    with self.__get_opacity_condition:
                        self.__get_opacity_condition.notify_all()
                        self.__get_opacity_condition.wait()
                elif self.__get_uncertainty_flag == self.QueryUncertaintyFlag.MANUAL:
                    self.__get_uncertainty_flag = self.QueryUncertaintyFlag.NONE
                    with self.__get_opacity_condition:
                        self.__get_opacity_condition.notify_all()
            rospy.logdebug(f'Logical operation used {end_timing(*timing_logical_operations):.2f} ms')
            
            # NOTE: Render RGBD image
            rerender_rgbd_flag = False
            if (frame_last_received is not None and\
                frame_id > rendering_rgbd_last_frame_id):
                color_vis, depth_vis, rgb_diff_colored, depth_diff_colored, rgbd_combine_loss_vis = self.__mapper.render_rgbd(frame_last_received, scale_modifier=1.0)
                rgbd = np.hstack((color_vis, depth_vis))
                rgbd_diff = np.hstack((rgb_diff_colored, depth_diff_colored, rgbd_combine_loss_vis))
                
                cv2.imwrite(str(self.render_rgbd_dir) + f'/{frame_id}.png', cv2.cvtColor(rgbd, cv2.COLOR_BGR2RGB))
                if self.__is_debug:
                    self.__debug_info['current_vis_data']['rgb_render'] = color_vis
                    self.__debug_info['current_vis_data']['depth_render'] = depth_vis
                if (not self.__hide_windows and\
                        self.__render_box.checked and\
                        frame_id % self.__render_every_slider.int_value == 0):
                    self.__o3d_cache_render_rgbd = o3d.geometry.Image(rgbd)
                    self.__o3d_cache_rgbd_loss_vis = o3d.geometry.Image(rgbd_diff)
                rendering_rgbd_last_frame_id = frame_id
                rerender_rgbd_flag = True
                
            # NOTE: output some information
            if frame_id % 100 == 0 and self.gaussians_num != 0:
                Log(f'frame_id: {frame_id}, gaussians_num: {self.gaussians_num}')
                
            # TODO: Update GUI
            self.__update_ui_mapper(
                frame_current,
                rerender_rgbd_flag,
                rerender_topdown_flag,
                rerender_voronoi_3d_flag)
            
            if self.__local_dataset is not None:
                if self.__local_dataset_pose_ros is not None:
                    self.__local_dataset_pose_pub.publish(self.__local_dataset_pose_ros)
            time.sleep(0.05)
            
        # NOTE: save constructed scene data
        # Write manifest as json
        manifest_json = json.dumps(self.__mapper.manifest, indent=4)
        with open(self.__mapper.save_path.joinpath("transforms.json"), "w") as f:
            f.write(manifest_json)
        self.traing_finished = True
        self.__mapper.post_processing()
        Log('Saving scene data finished')
        
        if os.path.exists(self.__dataset_config.scene_mesh_url):
            with open(os.path.join(self.__results_dir, 'gt_mesh.json'), 'w') as f:
                gt_mesh_config = {
                    'mesh_url': self.__dataset_config.scene_mesh_url,
                    'mesh_transform': self.__scene_mesh_transform.tolist()}
                json.dump(gt_mesh_config, f, indent=4)
        # set_planner_state_response:SetPlannerStateResponse = self.__set_planner_state_service(SetPlannerStateRequest(GlobalState.QUIT.value))
        self.__close_all()

    def __update_ui_mapper(self,
                        frame_current:Union[None, Dict[str, Union[torch.Tensor, int]]],
                        rerender_rgbd_flag:bool,
                        rerender_topdown_flag:bool,
                        rerender_voronoi_3d_flag:bool
                        ):
        if self.__global_state in [GlobalState.REPLAY, GlobalState.AUTO_PLANNING, GlobalState.MANUAL_PLANNING, GlobalState.MANUAL_CONTROL]:
            self.receive_data(self.q_main2vis)
            if not self.__hide_windows:
                if self.__view_gaussians_box.checked == True:
                    self.render_gaussian()
        
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_mapper(
                    rerender_rgbd_flag,
                    rerender_topdown_flag,
                    rerender_voronoi_3d_flag))
            Log(f'Update ui of mapper used {end_timing(*timing_update_render):.2f} ms')
        
        return
    
    def __update_main_thread_ui_mapper(self,
                        rerender_rgbd_flag:bool,
                        rerender_topdown_flag:bool,
                        rerender_voronoi_3d_flag:bool):
        
        # NOTE: Delete all keyframe frustum
        if not self.__keyframe_box.checked:
            i = 0
            # 删去所有的kf_frustum
            while True:
                if (not self.__widget_3d.scene.has_geometry(f"kf_frustum_{i}")):
                    break
                else:
                    if self.__widget_3d.scene.has_geometry(f"kf_frustum_{i}"):
                        self.__widget_3d.scene.remove_geometry(f"kf_frustum_{i}")
                i += 1
                
        if not self.__voronoi_3d_box.checked or rerender_voronoi_3d_flag:
            if self.__widget_3d.scene.has_geometry("voronoi_ridge"):
                self.__widget_3d.scene.remove_geometry("voronoi_ridge")
            i = 0
            while True:
                if (not self.__widget_3d.scene.has_geometry(f"voronoi_vertices_{i}")):
                    break
                else:
                    if self.__widget_3d.scene.has_geometry(f"voronoi_vertices_{i}"):
                        self.__widget_3d.scene.remove_geometry(f"voronoi_vertices_{i}")
                i += 1
                
        if self.__voronoi_3d_box.checked and rerender_voronoi_3d_flag:
            if not self.voronoi_3d_ridges is None:
                self.__widget_3d.scene.add_geometry("voronoi_ridge", self.voronoi_3d_ridges, self.__o3d_materials[VORONOI_GRAPH['material']])
            voronoi_3d_vertices = self.voronoi_3d_vertices
            
            for vertex_id, vertex in enumerate(voronoi_3d_vertices):
                vertex_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=VORONOI_GRAPH['nodes_radius'])
                vertex_sphere.translate(np.array(vertex))
                vertex_sphere.paint_uniform_color(VORONOI_GRAPH['nodes_color'])
                self.__widget_3d.scene.add_geometry(f"voronoi_vertices_{vertex_id}", vertex_sphere, self.__o3d_materials[VORONOI_GRAPH['material']])
                
        self.num_gaussians_info.text = "Number of Gaussians: {}".format(self.gaussians_num)
            
        if rerender_rgbd_flag:
            self.__rgbd_render_image.update_image(self.__o3d_cache_render_rgbd)
            self.__rgbd_loss_vis_image.update_image(self.__o3d_cache_rgbd_loss_vis)
        return

    def __update_ui_frame(self,
                        frame_current:Union[None, Dict[str, Union[torch.Tensor, int]]]):
        if not self.__update_main_thread.is_alive():
            return
        
        current_frustum = None
        current_agent = None
        rgbd_image = None
        current_pcd = None
        current_horizon = None
        cam_traj = None
        
        if frame_current is not None:
            # TODO: do not add element every time, if the checkbox is false, save them to cache, when the checkbox is true, add them to the scene
            rgb_data:np.ndarray = frame_current['rgb'].detach().cpu().numpy()
            depth_data:np.ndarray = frame_current['depth'].detach().cpu().numpy()
            pose_data:np.ndarray = frame_current['c2w'].detach().cpu().numpy()
            
            self.__topdown_info['rotation_vector'], self.__topdown_info['translation'] = c2w_world_to_topdown(pose_data, self.__topdown_info, self.__height_direction, np.int32)
            self.__update_ui_topdown()
            self.__update_ui_uncertainty()
        
            pose_data_o3d = OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL
            current_frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.__o3d_const_camera_intrinsics,
                np.linalg.inv(pose_data_o3d),
                CURRENT_FRUSTUM['scale'])
            current_frustum.paint_uniform_color(CURRENT_FRUSTUM['color'])
            current_agent = o3d.geometry.TriangleMesh(self.__agent_cylinder_mesh)
            current_agent.translate(pose_data_o3d[:3, 3])
        
            rgb_vis = np.uint8(rgb_data * 255)
            depth_vis = depth2rgb(depth_data, min_value=self.__rgbd_sensor.depth_min, max_value=self.__rgbd_sensor.depth_max)
            rgbd_vis = np.hstack((rgb_vis, depth_vis))
            
            rgbd_image = o3d.geometry.Image(rgbd_vis)
            
            if self.__is_debug:
                self.__debug_info['current_vis_data']['rgb'] = rgb_vis
                self.__debug_info['current_vis_data']['depth'] = depth_vis
            
            if np.any(np.isnan(depth_data)) or np.any(np.isinf(depth_data)) or np.all(depth_data == 0):
                rospy.logwarn('Depth contains NaN, Inf or all 0')
                self.__valid_depth_flag = False
            else:
                self.__o3d_pcd['current_pcd'] = rgbd_to_pointcloud(
                    rgb_vis,
                    depth_data,
                    pose_data,
                    self.__o3d_const_camera_intrinsics_o3c,
                    1000,
                    self.__rgbd_sensor.depth_max,
                    self.__device_o3c
                )
                current_pcd:o3d.t.geometry.PointCloud = self.__update_pcd(
                    'current_pcd',
                    False if self.__hide_windows else self.__current_pcd_box.checked,
                    self.__o3d_materials['unlit_mat'],
                    False)
                current_pcd_legacy:o3d.geometry.PointCloud = current_pcd.to_legacy()
                current_horizon:o3d.geometry.AxisAlignedBoundingBox = current_pcd_legacy.get_axis_aligned_bounding_box()
                current_horizon.color = CURRENT_HORIZON['color']
                self.__topdown_info['horizon_bbox'] = (
                    OPENCV_TO_OPENGL[:3, :3] @ current_horizon.get_min_bound(),
                    OPENCV_TO_OPENGL[:3, :3] @ current_horizon.get_max_bound())
                self.__current_horizon = get_horizon_bound_topdown(
                    self.__topdown_info['horizon_bbox'][0],
                    self.__topdown_info['horizon_bbox'][1],
                    self.__topdown_info,
                    self.__height_direction)
            
            if not self.__hide_windows:
                # NOTE: show information
                if frame_current is not None:
                    c2w = pose_data.copy()
                    c2w = c2w @ OPENCV_TO_OPENGL # habitat发出的OpenCV下的坐标有问题，需变换一下自身姿态
                    log_info = 'Current cam pose(in opencv): \n{}'.format(c2w.round(3))
                    self.cam_pose_info.text = log_info
                    
                # NOTE: show the trajectory
                latest_location = pose_data_o3d[:3, 3].copy()
                self.__traj_info['cam_centers'].append(latest_location)
                if (self.__cam_traj_box.checked) and len(self.__traj_info['cam_centers']) > 1 and self.__traj_info['step_times'] > 0:
                    cam_traj = update_traj(self.__traj_info['cam_centers'], self.__traj_info['step_times'])

                # NOTE: show keyframe frustums
                kf_frustums = None
                if (self.__keyframe_box.checked and len(self.__mapper.keyframe_list) > 0):
                    kf_frustums = self.__update_kf_frustums(self.__mapper.keyframe_list)
        
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_frame(
                    current_frustum,
                    current_agent,
                    rgbd_image,
                    current_pcd,
                    current_horizon,
                    cam_traj,
                    kf_frustums))
            Log(f'Update ui of frame used {end_timing(*timing_update_render):.2f} ms', tag='GUI')
        
        return
    
    def __update_main_thread_ui_frame(self,
                        current_frustum:o3d.geometry.LineSet,
                        current_agent:o3d.geometry.TriangleMesh,
                        rgbd_image:o3d.geometry.Image,
                        current_pcd:o3d.geometry.PointCloud,
                        current_horizon:o3d.geometry.AxisAlignedBoundingBox,
                        cam_traj:o3d.geometry.LineSet,
                        kf_frustums:Union[None, List[o3d.geometry.LineSet]]):
        if current_frustum is not None:
            self.__widget_3d.scene.remove_geometry('current_frustum')
            self.__widget_3d.scene.add_geometry('current_frustum', current_frustum, self.__o3d_materials[CURRENT_FRUSTUM['material']])
            self.__widget_3d.scene.show_geometry('current_frustum', self.__current_frustum_box.checked)
            
        if current_agent is not None:
            self.__widget_3d.scene.remove_geometry('current_agent')
            self.__widget_3d.scene.add_geometry('current_agent', current_agent, self.__o3d_materials[CURRENT_AGENT['material']])
            self.__widget_3d.scene.show_geometry('current_agent', self.__current_agent_box.checked)
            
        if rgbd_image is not None:
            self.__rgbd_live_image.update_image(rgbd_image)
            
        if current_pcd is not None:
            self.__widget_3d.scene.remove_geometry('current_pcd')
            self.__widget_3d.scene.add_geometry(
                'current_pcd',
                current_pcd,
                self.__o3d_materials['unlit_mat'])
            self.__widget_3d.scene.show_geometry('current_pcd', self.__current_pcd_box.checked)
            
        if current_horizon is not None:
            self.__widget_3d.scene.remove_geometry('current_horizon')
            self.__widget_3d.scene.add_geometry('current_horizon', current_horizon, self.__o3d_materials['unlit_line_mat'])
            self.__widget_3d.scene.show_geometry('current_horizon', self.__current_horizon_box.checked)
        
        if cam_traj is not None:
            self.__widget_3d.scene.remove_geometry("cam_traj")
            self.__widget_3d.scene.add_geometry("cam_traj", cam_traj, self.__o3d_materials['unlit_line_mat_slim'])
            
        if kf_frustums is not None:
            for i, frustum in enumerate(kf_frustums):
                if self.__widget_3d.scene.has_geometry(f'kf_frustum_{i}'):
                    self.__widget_3d.scene.remove_geometry(f"kf_frustum_{i}")
                if frustum is not None:
                    self.__widget_3d.scene.add_geometry(f"kf_frustum_{i}", frustum, self.__o3d_materials[KEYFRAME_FRUSTUM['material']])
            
        return
    
    def __update_ui_uncertainty(self):
        local_uncertainty_map_o3d = None
        if self.__uncertainty_info['local_uncertainty_cv2'] is not None:
            local_uncertainty_map = self.__uncertainty_info['local_uncertainty_cv2'].copy()
            local_uncertainty_map_o3d = o3d.geometry.Image(local_uncertainty_map)
            
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_uncertainty(
                    local_uncertainty_map_o3d))
            Log(f'Update ui of uncertainty used {end_timing(*timing_update_render):.2f} ms', tag='GUI')
            
    def __update_main_thread_ui_uncertainty(self,
            local_uncertainty_map_o3d:o3d.geometry.Image):
        if local_uncertainty_map_o3d is not None:
            self.__local_uncertainty_map_image.update_image(local_uncertainty_map_o3d)
                
    def render_gaussian(self):
        if self.__gaussian_for_render is None:
            return
        
        current_cam = self.__get_current_cam()
        
        timing_render_gaussian = start_timing()
        if self.__gaussian_color_type == GaussianColorType.RGBD:
            
            o3d_window_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                current_cam.image_width,
                current_cam.image_height,
                current_cam.fx,
                current_cam.fy,
                current_cam.cx,
                current_cam.cy)
            o3d_window_camera_intrinsics_o3c = o3d.core.Tensor(o3d_window_camera_intrinsics.intrinsic_matrix, dtype=o3d.core.Dtype.Float32, device=self.__device_o3c)
            
            rgb_np, depth_np, view_w2c_gl = self.__mapper.render_o3d_image(
                self.__gaussian_for_render,
                current_cam,
                self.__gaussian_scale_slider.double_value,
                self.__gaussian_color_type)
                
            guassian_pcd = rgbd_to_pointcloud(
                rgb_np,
                depth_np,
                np.linalg.inv(OPENCV_TO_OPENGL @ view_w2c_gl @ OPENCV_TO_OPENGL),
                o3d_window_camera_intrinsics_o3c,
                1000,
                np.inf,
                self.__device_o3c)
            
            
            if not self.__widget_3d.scene.scene.has_geometry('gaussian'):
                guassian_pcd_max_num = 1920 * 1080
                guassian_pcd_init_positions = np.zeros((guassian_pcd_max_num, 3), dtype=np.float32)
                guassian_pcd_init_positions_o3c = o3d.core.Tensor(guassian_pcd_init_positions, dtype=o3d.core.Dtype.Float32, device=self.__device_o3c)
                guassian_pcd_init = o3d.t.geometry.PointCloud(guassian_pcd_init_positions_o3c)
                guassian_pcd_init.paint_uniform_color([0, 0, 0])
                self.__widget_3d.scene.scene.add_geometry(
                    'gaussian',
                    guassian_pcd_init,
                    self.__o3d_materials['gaussian_mat'])
            self.__widget_3d.scene.scene.update_geometry(
                'gaussian',
                guassian_pcd,
                self.__widget_3d.scene.scene.UPDATE_COLORS_FLAG |\
                    self.__widget_3d.scene.scene.UPDATE_POINTS_FLAG |\
                        self.__widget_3d.scene.scene.UPDATE_NORMALS_FLAG |\
                            self.__widget_3d.scene.scene.UPDATE_UV0_FLAG)
            self.__widget_3d.scene.scene.show_geometry('gaussian', True)
        else:
            if self.__widget_3d.scene.scene.has_geometry('gaussian'):
                self.__widget_3d.scene.scene.show_geometry('gaussian', False)
            render_img = self.__mapper.render_o3d_image(self.__gaussian_for_render, current_cam, self.__gaussian_scale_slider.double_value, self.__gaussian_color_type)
            
            self.__widget_3d.scene.set_background([0, 0, 0, 1], o3d.geometry.Image(render_img))
        self.render_use_time_info.text = f'Render gaussians used {end_timing(*timing_render_gaussian):.2f} ms'
    
    # NOTE: Render topdown view 可视化高斯砸向地面的投影
    def get_topdown_cam(self):
        
        camera_pose_h = 1000.0
        
        # TODO: 相机姿态的xy取场景的中心点
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ]
        )
        c2w[:3, 3] = [self.__topdown_info['world_center'][0], -camera_pose_h, self.__topdown_info['world_center'][1]]
        w2c = np.linalg.inv(c2w)
        
        # scene info 
        scene_width = self.__topdown_info['topdown_world_shape'][0]
        scene_height = self.__topdown_info['topdown_world_shape'][1]
        
        height = self.__topdown_info['grid_map_shape'][1]
        width = self.__topdown_info['grid_map_shape'][0]
        image_topdown = torch.zeros(
            (1, int(height), int(width))
        )
        
        # vfov_deg = 0.01 # 垂直视场角
        vfov_deg = np.rad2deg(2 * np.arctan(scene_height / (2 * camera_pose_h)))
        # hfov_deg = self.vfov_to_hfov(vfov_deg, image_topdown.shape[1], image_topdown.shape[2])
        hfov_deg = np.rad2deg(2 * np.arctan(scene_width / (2 * camera_pose_h)))
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_topdown.shape[2])
        fy = fov2focal(FoVy, image_topdown.shape[1])
        cx = image_topdown.shape[2] // 2
        cy = image_topdown.shape[1] // 2
        T = torch.from_numpy(w2c)
        current_cam = Camera.init_from_gui(
            uid=-1,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=image_topdown.shape[1],
            W=image_topdown.shape[2],
        )
        current_cam.update_RT(T[0:3, 0:3], T[0:3, 3])
        return current_cam
                
    def add_camera(self, camera:torch.Tensor, name, color=[0, 1, 0], gt=False, size=0.01):
        # camera 是 OpenCV下的T_wc也就是c2w
        pose_data = camera.detach().cpu().numpy()
        C2W = OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color)
            # self.combo_kf.add_item(name) # TODO: Add keyframe to combo box
            self.frustum_dict[name] = frustum
            # self.__widget_3d.scene.add_geometry(name, frustum.line_set, self.__o3d_materials[KEYFRAME_FRUSTUM['material']])
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        # self.__widget_3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        # self.__widget_3d.scene.show_geometry(name, self.__current_frustum_box.checked)
        return frustum
                
    def receive_data(self, q:Queue):
        if q is None or q.empty():
            # 没有更新高斯数据 继续显示上一帧的高斯数据
            pass
        else:
            # 更新高斯数据
            self.__gaussian_packet:SmallGaussianPacket = q.get()
        
        if self.__gaussian_packet is None:
            return None

        if self.__gaussian_packet.has_gaussians:
            with self.__use_gaussian_condition:
                self.__use_gaussian_condition.acquire()
                self.__gaussian_for_render = {k: v.clone() for k, v in self.__gaussian_packet.params.items()}
                self.__gaussian_for_render = self.__cut_gaussian_by_height(
                                                    self.__gaussian_for_render, 
                                                    self.__height_direction_lower_bound_slider.double_value if not self.__hide_windows else self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0], 
                                                    self.__height_direction_upper_bound_slider.double_value if not self.__hide_windows else self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1])
                
                self.gaussians_num = self.__gaussian_packet.params['means3D'].shape[0]
                if self.__is_debug and self.__get_debug_data_flag == True:
                    with open(self.__debug_info['f_num_gaussians'], 'a') as f:
                        f.write(f'{self.gaussians_num}\n')
                    self.__get_debug_data_flag = False
                self.__use_gaussian_condition.release()

        if not self.__hide_windows and self.__gaussian_packet.current_frame is not None:
            frustum = self.add_camera(
                self.__gaussian_packet.current_frame, name="current", color=[0, 1, 0]
            )
            if self.__view_gaussians_box.checked and self.followcam_chbox.checked:
                self.__current_agent_box.checked = False
                self.__current_horizon_box.checked = False
                self.staybehind_chbox.checked = True
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.__widget_3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])
    
    def __close_all(self):
        while not self.q_main2vis.empty():
            self.q_main2vis.get()
        self.q_main2vis = None
        if self.__local_dataset is not None:
            with self.__local_dataset_condition:
                self.__local_dataset_condition.notify_all()
            self.__local_dataset_thread.join()
        self.__get_topdown_service.shutdown()
        with self.__get_topdown_condition:
            self.__get_topdown_condition.notify_all()
        self.__get_opacity_service.shutdown()
        with self.__get_opacity_condition:
            self.__get_opacity_condition.notify_all()
        if self.__hide_windows:
            rospy.signal_shutdown('Quit')
        else:
            gui.Application.instance.quit()
        Log(f'Exit main update thread', tag='activeGauss')   

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )
        
    def __get_current_cam(self):
        w2c = OPENCV_TO_OPENGL @ self.__widget_3d.scene.camera.get_view_matrix()

        image_gui = torch.zeros(
            (1, int(self.__window.size.height), int(self.__widget_3d_width))
        )
        vfov_deg = self.__widget_3d.scene.camera.get_field_of_view()
        hfov_deg = self.vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        cx = image_gui.shape[2] // 2
        cy = image_gui.shape[1] // 2
        T = torch.from_numpy(w2c)
        current_cam = Camera.init_from_gui(
            uid=-1,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=image_gui.shape[1],
            W=image_gui.shape[2],
        )
        current_cam.update_RT(T[0:3, 0:3], T[0:3, 3])
        return current_cam
        
    
    def __update_dataset(self):
        with self.__local_dataset_condition:
            dataset_config = self.__local_dataset.setup()
            self.__dataset_config:GetDatasetConfigResponse = dataset_config_to_ros(dataset_config)
            rospy.Service('get_dataset_config', GetDatasetConfig, self.__get_dataset_config)
            self.__local_dataset_twist:Twist = None
            rospy.Subscriber('cmd_vel', Twist, self.__cmd_vel_callback, queue_size=1)
            movement_fail_times = 0
            movement_fail_times_pub = rospy.Publisher('movement_fail_times', Int32, queue_size=1)
            self.__local_dataset_state = self.LocalDatasetState.INITIALIZED
            self.__local_dataset_condition.notify_all()
            while True:
                if self.__local_dataset_state == self.LocalDatasetState.INITIALIZED:
                    self.__local_dataset_state = self.LocalDatasetState.RUNNING
                    self.__local_dataset_condition.notify_all()

                apply_movement_flag = self.__local_dataset_twist is not None
                apply_movement_result = False
                if apply_movement_flag:
                    apply_movement_result = self.__local_dataset.apply_movement(self.__local_dataset_twist)
                    self.__local_dataset_twist = None
                step_times, step_num = self.__local_dataset.get_step_info()
                scene_id = self.__local_dataset.get_scene_id()
                self.__traj_info['step_times'] = step_times
                self.__local_dataset_label.text = f'Scene: {scene_id}\nStep: {step_times}/{step_num}'
                Log(f"Scene: {scene_id} Step:{step_times}/{step_num}")
                frame_numpy = self.__local_dataset.get_frame()
                frame_c2w = frame_numpy['c2w']
                pose_change_type = self.__is_pose_changed(frame_c2w)
                if pose_change_type != PoseChangeType.NONE:
                    if pose_change_type in [PoseChangeType.TRANSLATION, PoseChangeType.BOTH]:
                        movement_fail_times = 0
                    frame_quaternion = quaternion.from_rotation_matrix(frame_c2w[:3, :3])
                    frame_quaternion = quaternion.as_float_array(frame_quaternion)
                    pose_ros = PoseStamped()
                    pose_ros.header.stamp = rospy.Time.now()
                    pose_ros.header.frame_id = 'world'
                    pose_ros.pose.position.x = frame_c2w[0, 3]
                    pose_ros.pose.position.y = frame_c2w[1, 3]
                    pose_ros.pose.position.z = frame_c2w[2, 3]
                    pose_ros.pose.orientation.w = frame_quaternion[0]
                    pose_ros.pose.orientation.x = frame_quaternion[1]
                    pose_ros.pose.orientation.y = frame_quaternion[2]
                    pose_ros.pose.orientation.z = frame_quaternion[3]
                    self.__frame_c2w_last = frame_c2w
                    frame_torch = {
                        'rgb': torch.from_numpy(frame_numpy['rgb']),
                        'depth': torch.from_numpy(frame_numpy['depth']),
                        'c2w': torch.from_numpy(frame_c2w)}
                    self.__valid_depth_flag = True
                    self.__update_ui_frame(frame_torch)
                    if self.__frames_cache.empty() and self.__valid_depth_flag:
                        self.__frames_cache.put(frame_torch)
                    self.__local_dataset_pose_ros = pose_ros
                    self.__local_dataset_pose_pub.publish(self.__local_dataset_pose_ros)
                    movement_fail_times_pub.publish(Int32(movement_fail_times))
                elif apply_movement_flag:
                    if apply_movement_result:
                        movement_fail_times += 1
                    movement_fail_times_pub.publish(Int32(movement_fail_times))
                self.__local_dataset_condition.notify_all()
            self.__local_dataset.close()
        
    def __update_ui_topdown(self):
        topdown_free_map_o3d = None
        if self.__topdown_info['free_map_cv2'] is not None:
            topdown_free_map = self.__topdown_info['free_map_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation'] is not None:
                topdown_free_map = visualize_agent(
                    topdown_map=topdown_free_map,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_free_map_o3d = o3d.geometry.Image(topdown_free_map)
            if self.__is_debug:
                self.__debug_info['current_vis_data']['topdown_free_map'] = topdown_free_map
            
        topdown_free_map_binary_o3d = None
        if self.__topdown_info['free_map_binary_cv2'] is not None:
            topdown_free_map_binary = self.__topdown_info['free_map_binary_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation'] is not None:
                topdown_free_map_binary = visualize_agent(
                    topdown_map=topdown_free_map_binary,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_free_map_binary_o3d = o3d.geometry.Image(topdown_free_map_binary)
            if self.__is_debug:
                self.__debug_info['current_vis_data']['topdown_free_map_binary'] = topdown_free_map_binary
            
        topdown_visible_map_o3d = None
        if self.__topdown_info['visible_map_cv2'] is not None:
            topdown_visible_map = self.__topdown_info['visible_map_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation'] is not None:
                topdown_visible_map = visualize_agent(
                    topdown_map=topdown_visible_map,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            if self.__current_horizon is not None:
                cv2.rectangle(
                    topdown_visible_map,
                    np.int32(self.__current_horizon[0]),
                    np.int32(self.__current_horizon[1]),
                    (255, 0, 0),
                    1)
            topdown_visible_map_o3d = o3d.geometry.Image(topdown_visible_map)
            if self.__is_debug:
                self.__debug_info['current_vis_data']['topdown_visible_map'] = topdown_visible_map
            
        topdown_visible_map_binary_o3d = None
        if self.__topdown_info['visible_map_binary_cv2'] is not None:
            topdown_visible_map_binary = self.__topdown_info['visible_map_binary_cv2'].copy()
            if self.__topdown_info['rotation_vector'] is not None and self.__topdown_info['translation'] is not None:
                topdown_visible_map_binary = visualize_agent(
                    topdown_map=topdown_visible_map_binary,
                    meter_per_pixel=self.__topdown_info['meter_per_pixel'],
                    agent_translation=self.__topdown_info['translation'],
                    agent_rotation_vector=self.__topdown_info['rotation_vector'],
                    agent_color=(128, 255, 128),
                    agent_radius=self.__dataset_config.agent_radius,
                    rotation_vector_color=(0, 255, 0),
                    rotation_vector_thickness=2,
                    rotation_vector_length=20)
            topdown_visible_map_binary_o3d = o3d.geometry.Image(topdown_visible_map_binary)
            
        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_topdown(
                    topdown_free_map_o3d,
                    topdown_free_map_binary_o3d,
                    topdown_visible_map_o3d,
                    topdown_visible_map_binary_o3d))
            Log(f'Update ui of topdown used {end_timing(*timing_update_render):.2f} ms', tag='GUI')
        
    def __update_main_thread_ui_topdown(
        self,
        topdown_free_map_o3d:o3d.geometry.Image,
        topdown_free_map_binary_o3d:o3d.geometry.Image,
        topdown_visible_map_o3d:o3d.geometry.Image,
        topdown_visible_map_binary_o3d:o3d.geometry.Image):
        if topdown_free_map_o3d is not None:
            self.__topdown_free_map_image.update_image(topdown_free_map_o3d)
            
        if topdown_free_map_binary_o3d is not None:
            self.__topdown_free_map_image_binary.update_image(topdown_free_map_binary_o3d)
            
        if topdown_visible_map_o3d is not None:
            self.__topdown_visible_map_image.update_image(topdown_visible_map_o3d)
            
        if topdown_visible_map_binary_o3d is not None:
            self.__topdown_visible_map_image_binary.update_image(topdown_visible_map_binary_o3d)
            
    def __update_kf_frustums(self, keyframe_list:List[Dict[str, Union[int, torch.Tensor]]]):
        kf_frustums = []
        for keyframe in keyframe_list:
            # curr_keyframe = {'id': cur_frame_id, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
            pose_data = np.linalg.inv(keyframe['est_w2c'].detach().cpu().numpy()) @ OPENCV_TO_OPENGL # c2w
            pose_data_o3d = OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL
            keyframe_frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.__o3d_const_camera_intrinsics,
                np.linalg.inv(pose_data_o3d),
                KEYFRAME_FRUSTUM['scale'])
            keyframe_frustum.paint_uniform_color(KEYFRAME_FRUSTUM['color'])
            kf_frustums.append(keyframe_frustum)
        return kf_frustums
        
    # NOTE: callback functions for GUI

    def __window_on_layout(self, ctx:gui.LayoutContext):
        em = ctx.theme.font_size

        panel_width = 23 * em
        rect:gui.Rect = self.__window.content_rect

        self.__panel_control.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        x = self.__panel_control.frame.get_right()
        
        # 3D widget width
        self.__widget_3d_width = rect.width - 2*panel_width

        self.__widget_3d.frame = gui.Rect(x, rect.y, rect.get_right() - 2*panel_width, rect.height)
        self.__panel_visualize.frame = gui.Rect(self.__widget_3d.frame.get_right(), rect.y, panel_width, rect.height)

        return
        
    def __window_on_close(self) -> bool:
        if self.__local_dataset is not None:
            with self.__local_dataset_condition:
                self.__local_dataset_condition.notify_all()
            self.__local_dataset_thread.join()
        # Log(f'Waiting for threads to close', tag='GUI')
        # while len(threading.enumerate()) > 2: # 2的原因是mapping和主线程
        #     # Log(f'Threads: {len(threading.enumerate())}', tag='GUI')
        #     time.sleep(0.1) # Wait for the threads to close
        self.__update_main_thread.join()
        gui.Application.instance.quit()
        return True

    def __widget_3d_on_key(self, event:gui.KeyEvent):
        if event.key == gui.KeyName.UP:
            twist_current = {
                'linear': np.array([0.1, 0.0, 0.0]),
                'angular': np.zeros(3)
            }
        elif event.key == gui.KeyName.LEFT:
            twist_current = {
                'linear': np.zeros(3),
                'angular': np.array([0.0, 0.0, 0.1])
            }
        elif event.key == gui.KeyName.RIGHT:
            twist_current = {
                'linear': np.zeros(3),
                'angular': np.array([0.0, 0.0, -0.1])
            }
        elif event.key == gui.KeyName.PAGE_UP:
            twist_current = {
                'linear': np.zeros(3),
                'angular': np.array([0.0, -0.1, 0.0])
            }
        elif event.key == gui.KeyName.PAGE_DOWN:
            twist_current = {
                'linear': np.zeros(3),
                'angular': np.array([0.0, 0.1, 0.0])
            }
        else:
            return gui.Widget.IGNORED
        self.__apply_movement(twist_current)
        return gui.Widget.HANDLED

    
    def __update_mesh(self, mesh_name:str, show:bool, material:Union[str, o3d.visualization.rendering.MaterialRecord]=None, update:bool=True) -> o3d.geometry.TriangleMesh:
        if isinstance(material, str):
            material = self.__o3d_materials[material]
        elif isinstance(material, o3d.visualization.rendering.MaterialRecord):
            pass
        else:
            material = self.__o3d_materials['lit_mat']
        mesh = self.__cut_mesh_by_height(o3d.geometry.TriangleMesh(self.__o3d_meshes[mesh_name]))
        if update:
            self.__widget_3d.scene.remove_geometry(mesh_name)
            self.__widget_3d.scene.add_geometry(mesh_name, mesh, material)
            self.__widget_3d.scene.show_geometry(mesh_name, show)
        return mesh
    
    def __update_pcd(self, pointcloud_name:str, show:bool, material:Union[str, o3d.visualization.rendering.MaterialRecord]=None, update:bool=True) -> o3d.t.geometry.PointCloud:
        if isinstance(material, str):
            material = self.__o3d_materials[material]
        elif isinstance(material, o3d.visualization.rendering.MaterialRecord):
            pass
        else:
            material = self.__o3d_materials['unlit_mat']
        pcd = o3d.t.geometry.PointCloud(self.__o3d_pcd[pointcloud_name])
        points = pcd.point.positions.numpy()
        points_condition = np.logical_or(
            points[:, self.__height_direction[0]] < (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0] if self.__hide_windows else self.__height_direction_lower_bound_slider.double_value),
            points[:, self.__height_direction[0]] > (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1] if self.__hide_windows else self.__height_direction_upper_bound_slider.double_value))
        pcd = pcd.select_by_index(np.where(~points_condition)[0])
        if update:
            if self.__widget_3d.scene.has_geometry(pointcloud_name):
                self.__widget_3d.scene.scene.update_geometry(pointcloud_name,
                                                            pcd,
                                                            rendering.Scene.UPDATE_POINTS_FLAG +\
                                                                rendering.Scene.UPDATE_COLORS_FLAG +\
                                                                    rendering.Scene.UPDATE_NORMALS_FLAG +\
                                                                        rendering.Scene.UPDATE_UV0_FLAG)
            else:
                self.__widget_3d.scene.add_geometry(pointcloud_name, pcd, material)
            self.__widget_3d.scene.show_geometry(pointcloud_name, show)
        return pcd
    
    def __height_direction_bound_slider_callback(self, value:float, is_upper:int):
        if value == self.__open3d_gui_widget_last_state['height_direction_bound_slider'][is_upper]:
            return gui.Slider.HANDLED
        else:
            if self.__scene_mesh_box is not None:
                self.__update_mesh('scene_mesh', self.__scene_mesh_box.checked, self.__o3d_materials['lit_mat'])
            if self.__o3d_pcd['current_pcd'] is not None:
                current_pcd:o3d.t.geometry.PointCloud = self.__update_pcd(
                    'current_pcd',
                    self.__current_pcd_box.checked,
                    self.__o3d_materials['unlit_mat'])
                current_pcd_legacy:o3d.geometry.PointCloud = current_pcd.to_legacy()
                current_horizon:o3d.geometry.AxisAlignedBoundingBox = current_pcd_legacy.get_axis_aligned_bounding_box()
                current_horizon.color = CURRENT_HORIZON['color']
                self.__widget_3d.scene.remove_geometry('current_horizon')
                self.__widget_3d.scene.add_geometry('current_horizon', current_horizon, self.__o3d_materials['unlit_line_mat'])
                self.__widget_3d.scene.show_geometry('current_horizon', self.__current_horizon_box.checked)
            # TODO: Crop other elements here
            self.__open3d_gui_widget_last_state['height_direction_bound_slider'][is_upper] = value
        if is_upper:
            self.__height_direction_lower_bound_slider.set_limits(self.__bbox_visualize[self.__height_direction[0]][0], value)
        else:
            self.__height_direction_upper_bound_slider.set_limits(value, self.__bbox_visualize[self.__height_direction[0]][1] + 0.1)
        return gui.Slider.HANDLED
    
    # NOTE: ros functions
    
    def __frame_callback(self, msg:frame):
        print("frame",frame)
        # frame_quaternion = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        # frame_quaternion = quaternion.from_float_array(frame_quaternion)
        # frame_translation = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # frame_c2w = np.eye(4)
        # frame_c2w[:3, :3] = quaternion.as_rotation_matrix(frame_quaternion)
        # frame_c2w[:3, 3] = frame_translation
        # frame_c2w = convert_to_c2w_opencv(frame_c2w, self.__pose_data_type)
        # if self.__is_pose_changed(frame_c2w) == PoseChangeType.NONE:
        #     return
        
        # frame_rotation_vector = np.degrees(quaternion.as_rotation_vector(frame_quaternion))
        # rospy.loginfo(f'Agent:\n\tX: {frame_translation[0]:.2f}, Y: {frame_translation[1]:.2f}, Z: {frame_translation[2]:.2f}\n\tX_angle: {frame_rotation_vector[0]:.2f}, Y_angle: {frame_rotation_vector[1]:.2f}, Z_angle: {frame_rotation_vector[2]:.2f}')
        
        # if msg.rgb.encoding in ['rgb8', 'bgr8', 'rgba8', 'bgra8']:
        #     if msg.rgb.encoding in ['rgb8', 'bgr8']:
        #         channel_number = 3
        #     elif msg.rgb.encoding in ['rgba8', 'bgra8']:
        #         channel_number = 4
        #     else:
        #         raise NotImplementedError(f'Unsupported RGB encoding: {msg.rgb.encoding}')
        #     frame_rgb = np.frombuffer(msg.rgb.data, dtype=np.uint8).reshape(msg.rgb.height, msg.rgb.width, channel_number)
        #     if msg.rgb.encoding == 'bgr8':
        #         frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        #     elif msg.rgb.encoding == 'rgba8':
        #         frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
        #     elif msg.rgb.encoding == 'bgra8':
        #         frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGRA2RGB)
        #     frame_rgb = frame_rgb.astype(np.float32) / 255.0
        # else:
        #     raise NotImplementedError(f'Unsupported RGB encoding: {msg.rgb.encoding}')
        # if msg.depth.encoding == '32FC1':
        #     frame_depth = np.frombuffer(msg.depth.data, dtype=np.float32).reshape(msg.depth.height, msg.depth.width)
        #     assert self.__rgbd_sensor.depth_scale == 1, 'Depth scale is not 1'
        # elif msg.depth.encoding == '16UC1':
        #     frame_depth = np.frombuffer(msg.depth.data, dtype=np.uint16).reshape(msg.depth.height, msg.depth.width).astype(np.float32)
        #     assert self.__rgbd_sensor.depth_scale == 1000, 'Depth scale is not 1000'
        # else:
        #     raise NotImplementedError(f'Unsupported Depth encoding: {msg.depth.encoding}')
        # frame_depth = frame_depth / self.__rgbd_sensor.depth_scale
        # frame_depth = self.__preprocess_frame(frame_rgb, frame_depth, frame_c2w)
        # if np.any(np.isnan(frame_depth)) or np.any(np.isinf(frame_depth)) or np.all(frame_depth == 0):
        #     rospy.logwarn('Depth contains NaN, Inf or all 0')
        #     return
        # if self.__frames_cache.empty():
        #     self.__frame_c2w_last = frame_c2w
        #     frame_current = {
        #         'rgb': torch.from_numpy(frame_rgb),
        #         'depth': torch.from_numpy(frame_depth.copy()),
        #         'c2w': torch.from_numpy(frame_c2w).float()}
        #     self.__update_ui_frame(frame_current)
        #     self.__frames_cache.put(frame_current)
        return
    
    def __preprocess_frame(self, frame_rgb:np.ndarray, frame_depth:np.ndarray, frame_c2w:np.ndarray):
        # NOTE: 滤除depth深度值>self.__depth_limit[1] and <self.__depth_limit[0]的点
        frame_depth = np.where((frame_depth > self.__depth_limit[1]) | (frame_depth < self.__depth_limit[0]), 0, frame_depth)
        return frame_depth
    
    def __apply_movement(self, twist:Dict[str, np.ndarray]):
        if self.__local_dataset is None:
            twist_msg = Twist()
            twist_msg.linear.x = twist['linear'][0]
            twist_msg.linear.y = twist['linear'][1]
            twist_msg.linear.z = twist['linear'][2]
            twist_msg.angular.x = twist['angular'][0]
            twist_msg.angular.y = twist['angular'][1]
            twist_msg.angular.z = twist['angular'][2]
            self.__cmd_vel_publisher.publish(twist_msg)
        else:
            if not self.__local_dataset_parallelized and not self.__frames_cache.empty():
                return
            with self.__local_dataset_condition:
                self.__local_dataset_twist = twist
                self.__local_dataset_condition.notify_all()
                self.__local_dataset_condition.wait()
        return
        
    def __cmd_vel_callback(self, twist:Twist):
        twist_current = {
            'linear': np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z]),
            'angular': np.array([
                twist.angular.x,
                twist.angular.y,
                twist.angular.z])}
        self.__apply_movement(twist_current)
        
    def __get_dataset_config(self, req:GetDatasetConfigRequest) -> GetDatasetConfigResponse:
        return self.__dataset_config
    
    # def __get_topdown(self, req:GetTopdownRequest) -> GetTopdownResponse:
        # with self.__get_topdown_condition:
        #     if req.arrived_flag:
        #         self.__get_topdown_flag = self.QueryTopdownFlag.ARRIVED
        #     elif self.__get_topdown_flag == self.QueryTopdownFlag.NONE:
        #         self.__get_topdown_flag = self.QueryTopdownFlag.RUNNING
        #     self.__get_topdown_condition.wait()
        #     if self.__global_state == GlobalState.QUIT:
        #         self.__get_topdown_condition.notify_all()
        #         return None
        #     free_map_binary:np.ndarray = self.__topdown_info['free_map_binary'].copy()
        #     visible_map_binary:np.ndarray = self.__topdown_info['visible_map_binary'].copy()
        #     self.__get_topdown_condition.notify_all()
        # topdown_response = GetTopdownResponse()
        # topdown_response.free_map = free_map_binary.flatten().tolist()
        # topdown_response.visible_map = visible_map_binary.flatten().tolist()
        # if req.arrived_flag:
        #     topdown_response.targets_frustums = []
        #     topdown_response.horizon_bound_min.x = self.__topdown_info['horizon_bbox'][0][0]
        #     topdown_response.horizon_bound_min.y = self.__topdown_info['horizon_bbox'][0][1]
        #     topdown_response.horizon_bound_min.z = self.__topdown_info['horizon_bbox'][0][2]
        #     topdown_response.horizon_bound_max.x = self.__topdown_info['horizon_bbox'][1][0]
        #     topdown_response.horizon_bound_max.y = self.__topdown_info['horizon_bbox'][1][1]
        #     topdown_response.horizon_bound_max.z = self.__topdown_info['horizon_bbox'][1][2]
        # return topdown_response
    
    # def __get_opacity(self, req:GetOpacityRequest) -> GetOpacityResponse:
    #     with self.__get_opacity_condition:
    #         if req.arrived_flag:
    #             # Global
    #             # 将voronoi节点的位置信息传递给mapper
    #             node_points = [{'id':req.nodes_id[idx],'position': np.array([node_point.x, node_point.y, node_point.z])} for idx, node_point in enumerate(req.nodes)]
    #             self.__mapper.set_voronoi_nodes(node_points)
    #             self.__get_uncertainty_flag = self.QueryUncertaintyFlag.GLOBAL
    #             self.__get_opacity_condition.wait()
    #             if self.__global_state == GlobalState.QUIT:
    #                 self.__get_opacity_condition.notify_all()
    #                 return None
    #             uncertainties = [node['uncertainty'] for node in self.__mapper.voronoi_nodes]
    #             uncertainties = (np.array(uncertainties) / np.max(uncertainties) * 10).tolist() # 归一化到0-10分数
                
    #             volumes = [node['volume'] for node in self.__mapper.voronoi_nodes]
    #             volumes = (np.array(volumes) / np.max(volumes) * 10).tolist() # 归一化到0-10分数
                
    #             opacity_response = GetOpacityResponse()
    #         else:
    #             # Local
    #             self.__get_uncertainty_flag = self.QueryUncertaintyFlag.LOCAL
    #             self.__get_opacity_condition.wait()
    #             if self.__global_state == GlobalState.QUIT:
    #                 self.__get_opacity_condition.notify_all()
    #                 return None
    #             uncertainties = [self.local_node['uncertainty']] # just one node
    #             volumes = [0,]
    #             opacity_response = GetOpacityResponse()
    #             if self.local_node['best_pose'] is None:
    #                 opacity_response.targets_frustums.append(Pose())
    #             else:
    #                 self.local_node['best_pose'] = self.local_node['best_pose'] @ OPENCV_TO_OPENGL # 还原成habitat输出的c2w格式
    #                 pose = matrix_to_pose(self.local_node['best_pose'])
    #                 opacity_response.targets_frustums.append(pose)
                
            
    #         self.__get_opacity_condition.notify_all()
        
    #     opacity_response.targets_frustums_uncertainty = uncertainties
    #     opacity_response.targets_frustums_volume = volumes
        

    #     return opacity_response
    
    # def __get_topdown_config(self, req:GetTopdownConfigRequest) -> GetTopdownConfigResponse:
    #     topdown_config_response = GetTopdownConfigResponse()
    #     topdown_config_response.topdown_x_world_dim_index = self.__topdown_info['world_dim_index'][0]
    #     topdown_config_response.topdown_y_world_dim_index = self.__topdown_info['world_dim_index'][1]
    #     topdown_config_response.topdown_x_world_lower_bound = self.__topdown_info['world_2d_bbox'][0][0]
    #     topdown_config_response.topdown_x_world_upper_bound = self.__topdown_info['world_2d_bbox'][0][1]
    #     topdown_config_response.topdown_y_world_lower_bound = self.__topdown_info['world_2d_bbox'][1][0]
    #     topdown_config_response.topdown_y_world_upper_bound = self.__topdown_info['world_2d_bbox'][1][1]
    #     topdown_config_response.topdown_x_length = self.__topdown_info['grid_map_shape'][0]
    #     topdown_config_response.topdown_y_length = self.__topdown_info['grid_map_shape'][1]
    #     topdown_config_response.meter_per_pixel = self.__topdown_info['meter_per_pixel']
    #     return topdown_config_response
    
    def __set_mapper(self, req:SetMapperRequest) -> SetMapperResponse:
        
        kf_every_old = self.__mapper.get_kf_every()
        map_every_old = self.__mapper.get_map_every()
        if req.map_every != 0:
            map_every = req.map_every
            self.__mapper.set_map_every(map_every)
            if not self.__hide_windows:
                self.__map_every_slider.int_value = map_every
        if req.kf_every != 0:
            kf_every = req.kf_every
            self.__mapper.set_kf_every(kf_every)
            if not self.__hide_windows:
                self.__kf_every_slider.int_value = kf_every
        
        response = SetMapperResponse()
        response.kf_every_old = kf_every_old
        response.map_every_old = map_every_old
        return response
        
    # NOTE: Common Funtions
    
    def __is_pose_changed(self, frame_c2w:np.ndarray) -> PoseChangeType:
        if self.__frame_c2w_last is None:
            self.__frame_c2w_last = frame_c2w
            return PoseChangeType.BOTH
        else:
            return is_pose_changed(
                self.__frame_c2w_last,
                frame_c2w,
                self.__frame_update_translation_threshold,
                self.__frame_update_rotation_threshold)
            
    def __cut_mesh_by_height(self, mesh:o3d.geometry.TriangleMesh) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        vertices = np.array(mesh.vertices)
        vertices_condition = np.logical_or(
            vertices[:, self.__height_direction[0]] < (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][0] if self.__hide_windows else self.__height_direction_lower_bound_slider.double_value),
            vertices[:, self.__height_direction[0]] > (self.__open3d_gui_widget_last_state['height_direction_bound_slider'][1] if self.__hide_windows else self.__height_direction_upper_bound_slider.double_value))
        mesh.remove_vertices_by_mask(vertices_condition)
        return mesh
    
    def __cut_gaussian_by_height(self, gaussian_params:Dict[str, torch.Tensor], upper_limit, lower_limit) -> Dict[str, torch.Tensor]:
        gauss_condition = torch.logical_or(
            -gaussian_params['means3D'][:,1] < upper_limit,
            -gaussian_params['means3D'][:,1] > lower_limit)
        gaussian_params['means3D'] = gaussian_params['means3D'][~gauss_condition]
        gaussian_params['rgb_colors'] = gaussian_params['rgb_colors'][~gauss_condition]
        gaussian_params['unnorm_rotations'] = gaussian_params['unnorm_rotations'][~gauss_condition]
        gaussian_params['logit_opacities'] = gaussian_params['logit_opacities'][~gauss_condition]
        gaussian_params['log_scales'] = gaussian_params['log_scales'][~gauss_condition]
        return gaussian_params
    
    def __save_current_data_callback(self):
        # 保持当前可视化的数据
        current_vis_data_dir = self.__debug_info['current_vis_data_dir']
        if self.__debug_info['current_vis_data'] is not None:
            for key, value in self.__debug_info['current_vis_data'].items():
                # RGB->BGR
                if value.shape[2] == 3:
                    value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{str(current_vis_data_dir)}/{key}.png', value)
        Log(f'Save current data done', tag='GUI')
        return