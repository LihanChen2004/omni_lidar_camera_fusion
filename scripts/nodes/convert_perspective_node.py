#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from cv_bridge import CvBridge
from equilib import Equi2Pers
from concurrent.futures import ThreadPoolExecutor
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler


class PanoramaToPerspectiveNode:
    def __init__(self):
        rospy.init_node('panorama_to_perspective', anonymous=True)
        self.bridge = CvBridge()

        # 初始化Equi2Pers，控制生成的透视图大小和视场角
        self.equi2pers = Equi2Pers(
            height=320,
            width=640,
            fov_x=120.0,
            mode="bilinear",
        )

        # 创建发布者和订阅者
        self.image_publishers = [rospy.Publisher(f'/camera/perspective_{i}', ROSImage, queue_size=10) for i in range(6)]
        self.camera_info_publisher = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=10)
        rospy.Subscriber('/camera/image', ROSImage, self.image_callback)

        # 使用线程池来并行处理透视图生成
        self.executor = ThreadPoolExecutor(max_workers=6)

        # TF 广播器，用于发布透视图相机的坐标系变换
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # 定义六个视角的旋转角度
        self.rotations = [
            {'roll': 0., 'pitch': np.radians(-15), 'yaw': np.radians(120)},  # upper left view
            {'roll': 0., 'pitch': np.radians(-15), 'yaw': 0.},               # upper front view
            {'roll': 0., 'pitch': np.radians(-15), 'yaw': np.radians(-120)}, # upper right view
            {'roll': 0., 'pitch': np.radians(15), 'yaw': np.radians(120)},   # lower left view
            {'roll': 0., 'pitch': np.radians(15), 'yaw': 0.},                # lower front view
            {'roll': 0., 'pitch': np.radians(15), 'yaw': np.radians(-120)}   # lower right view
        ]

        # BUG: 内参矩阵，这里的 vfov=82 是手调出来有误差的，理论计算值应该是60°，但实测60°的内参矫正后明显不对...
        K = self.compute_intrinsic_matrix(np.radians(self.equi2pers.fov_x), np.radians(82), self.equi2pers.width, self.equi2pers.height)
        self.camera_info_msg = self.generate_camera_info(self.equi2pers.width, self.equi2pers.height, K)

        self.publish_transforms()

    def compute_intrinsic_matrix(self, h_fov, v_fov, width, height):
        """
        计算给定视场角和图像尺寸的相机内参矩阵。
        """
        fx = (0.5 * width) / (np.tan(h_fov / 2))
        fy = (0.5 * height) / (np.tan(v_fov / 2))

        cx, cy = width / 2, height / 2

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        return K

    def generate_camera_info(self, width, height, K):
        """
        生成CameraInfo消息，填充相机参数。
        """
        camera_info_msg = CameraInfo()
        camera_info_msg.height = height
        camera_info_msg.width = width
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.D = [0, 0, 0, 0, 0]  # 无畸变
        camera_info_msg.K = K.flatten().tolist()
        camera_info_msg.R = np.eye(3).flatten().tolist()
        camera_info_msg.P = np.hstack((K, np.zeros((3, 1)))).flatten().tolist()
        camera_info_msg.binning_x = 0
        camera_info_msg.binning_y = 0
        camera_info_msg.roi.x_offset = 0
        camera_info_msg.roi.y_offset = 0
        camera_info_msg.roi.height = 0
        camera_info_msg.roi.width = 0
        camera_info_msg.roi.do_rectify = False

        return camera_info_msg

    def publish_transforms(self):
        """
        发布六个相机的静态TF坐标变换
        """
        for i, rotation in enumerate(self.rotations):
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "camera"
            t.child_frame_id = f"camera/perspective_{i}"
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (0, 0, 0)

            q = quaternion_from_euler(-rotation['pitch'], -rotation['yaw'], rotation['roll'], axes='sxyz')
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q

            self.tf_broadcaster.sendTransform(t)

    def image_callback(self, msg):
        """
        处理接收到的全景图像，转换为六个透视图，并发布。
        """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # 将360°x120°全景图转换为360°x180°全景图
            equi_img_180 = self.extend_pano_image(cv_img)
            equi_img_np = equi_img_180.transpose(2, 0, 1)

            # 并行处理每个视角的透视图
            futures = [self.executor.submit(self.convert_perspective_image, equi_img_np, self.equi2pers, rotation) for rotation in self.rotations]

            for i, future in enumerate(futures):
                perspective_image = future.result()
                ros_image = self.bridge.cv2_to_imgmsg(perspective_image, encoding="rgb8")
                ros_image.header.frame_id = f'camera/perspective_{i}'
                ros_image.header.stamp = msg.header.stamp

                # 发布透视图
                self.image_publishers[i].publish(ros_image)

                # 发布CameraInfo消息
                self.camera_info_msg.header = ros_image.header
                self.camera_info_publisher.publish(self.camera_info_msg)

            self.publish_transforms()

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def extend_pano_image(self, equi_img_np):
        """
        将360°x120°全景图转换为360°x180°全景图，通过在上下添加黑色条带来扩展图片。
        """
        height, width, _ = equi_img_np.shape
        new_height = width // 2
        extended_img = np.zeros((new_height, width, 3), dtype=np.uint8)
        paste_position = (new_height - height) // 2
        extended_img[paste_position:paste_position + height, :, :] = equi_img_np
        return extended_img

    def convert_perspective_image(self, equi_img_np, equi2pers, rotation):
        """
        将360°x180°的全景图转换为指定视角的透视图。
        """
        return equi2pers(equi=equi_img_np, rots=rotation).transpose(1, 2, 0)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = PanoramaToPerspectiveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
