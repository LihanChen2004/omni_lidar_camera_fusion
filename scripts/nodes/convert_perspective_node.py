#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from equilib import Equi2Pers
from concurrent.futures import ThreadPoolExecutor
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler

def extend_pano_image(equi_img_np: np.ndarray) -> np.ndarray:
    """
    将360°x120°全景图转换为360°x180°全景图，通过在上下添加黑色条带来扩展图片。
    """
    height, width, _ = equi_img_np.shape
    new_height = width // 2
    
    # 创建一个新的黑色图像，并将原始图像居中粘贴
    extended_img = np.zeros((new_height, width, 3), dtype=np.uint8)
    paste_position = (new_height - height) // 2
    extended_img[paste_position:paste_position + height, :, :] = equi_img_np
    
    return extended_img

def convert_perspective_image(equi_img_np: np.ndarray, equi2pers: Equi2Pers, rotation: dict) -> np.ndarray:
    """
    将360°x180°的全景图转换为指定视角的透视图。
    """
    perspective_image = equi2pers(equi=equi_img_np, rots=rotation).transpose(1, 2, 0)
    return perspective_image

class PanoramaToPerspectiveNode:
    def __init__(self):
        rospy.init_node('panorama_to_perspective', anonymous=True)
        self.bridge = CvBridge()

        self.equi2pers = Equi2Pers(
            height=320,
            width=640,
            fov_x=120.0,
            mode="bilinear",
        )

        self.publishers = [
            rospy.Publisher(f'/camera/perspective_{i}', ROSImage, queue_size=10) for i in range(6)
        ]

        rospy.Subscriber('/camera/image', ROSImage, self.image_callback)

        self.executor = ThreadPoolExecutor(max_workers=6)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # 定义六个视角的旋转角度
        self.rotations = [
            {'roll': 0., 'pitch': np.radians(-15), 'yaw': np.radians(120)},  # upper left view
            {'roll': 0., 'pitch': np.radians(-15), 'yaw': 0.},             # upper front view
            {'roll': 0., 'pitch': np.radians(-15), 'yaw': np.radians(-120)}, # upper right view
            {'roll': 0., 'pitch': np.radians(15), 'yaw': np.radians(120)}, # lower left view
            {'roll': 0., 'pitch': np.radians(15), 'yaw': 0.},            # lower front view
            {'roll': 0., 'pitch': np.radians(15), 'yaw': np.radians(-120)} # lower right view
        ]

        self.camera_translations = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0) 
        ]
        
        self.publish_transforms()

    def publish_transforms(self):
        for i, rotation in enumerate(self.rotations):
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "camera"
            t.child_frame_id = f"camera/perspective_{i}"
            t.transform.translation.x = self.camera_translations[i][0]
            t.transform.translation.y = self.camera_translations[i][1]
            t.transform.translation.z = self.camera_translations[i][2]

            q = quaternion_from_euler(-rotation['pitch'], rotation['yaw'], rotation['roll'], axes='sxyz')
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.tf_broadcaster.sendTransform(t)

    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # 将360°x120°全景图转换为360°x180°全景图
        equi_img_180 = extend_pano_image(cv_img)
        equi_img_np = equi_img_180.transpose(2, 0, 1)
        
        # 使用线程池并行处理透视图转换
        futures = [
            self.executor.submit(convert_perspective_image, equi_img_np, self.equi2pers, rotation)
            for rotation in self.rotations
        ]
        
        # 获取并发布透视图
        for i, future in enumerate(futures):
            perspective_image = future.result()
            ros_image = self.bridge.cv2_to_imgmsg(perspective_image, encoding="rgb8")
            self.publishers[i].publish(ros_image)

        self.publish_transforms()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PanoramaToPerspectiveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
