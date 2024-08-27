import rospy
import time
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
from collections import OrderedDict
lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

global_bbox = None
class Detector:
    def __init__(self) -> None:
        rospy.init_node('livox_detector', anonymous=True)
        self.marker_pub = rospy.Publisher(
            '/detect_box3d', MarkerArray, queue_size=1) # semantic pcd with bbox
        self.marker_array = MarkerArray()
        self.pred_boxes_sub = rospy.Subscriber('object_markers', MarkerArray, self.bbox_callback)
        self.semantic_pcd_sub = rospy.Subscriber('sensor_scan_semantic_pcd', PointCloud2, self.semantic_pcd_callback) # pcd for syn time
        self.marker_cache = OrderedDict()
    def bbox_callback(self, data):
        
        # global_bbox = data
        # print("Marker timestamp:", data.markers[0].header.stamp) # they have the same time stamp, so we only need to use the first one
        timestamp = data.markers[0].header.stamp
        self.marker_cache[timestamp] = data
        while len(self.marker_cache) > 1000:
            self.marker_cache.popitem(last=False)
            
    def get_marker_array(self, timestamp):
        closest_timestamp = min(self.marker_cache.keys(), key=lambda t: abs(t - timestamp))
        return self.marker_cache.get(closest_timestamp)
        
    def semantic_pcd_callback(self, data):
        global global_bbox
        # semantic_pcd = data
        print("semantic_pcd.head.stamp",data.header.stamp)
        syn_bbox = self.get_marker_array(data.header.stamp)
        print("syn_bbox",syn_bbox.markers[0].header.stamp)
        global_bbox = syn_bbox
        
        
    def rotx(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1,  0,  0],
                        [0,  c,  -s],
                        [0, s,  c]])
    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                        [0,  1,  0],
                        [-s, 0,  c]])
    def rotz(self,t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  -s,  0],
                        [s,  c,  0],
                        [0, 0,  1]])
    def get_3d_box(self, center, box_size, heading_angle):
        ''' Calculate 3D bounding box corners from its parameterization.
        Input:heading_angle
            box_size: tuple of (l,w,h)
            : rad scalar, clockwise from pos z axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        R = self.rotz(heading_angle)
        l, w, h = box_size
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d
    def display(self, boxes):
        self.marker_array.markers.clear()
        for obid in range(len(boxes)):
            ob = boxes[obid]
            tid = 0
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append(Point(ob[i], ob[i+8], ob[i+16]))
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.Time.now()
            marker.id = obid*2
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST
            marker.lifetime = rospy.Duration(0)
            marker.color.r = 1
            marker.color.g = 0
            marker.color.b = 0
            marker.color.a = 1
            marker.scale.x = 0.01
            marker.points = []
            for line in lines:
                marker.points.append(detect_points_set[line[0]])
                marker.points.append(detect_points_set[line[1]])
                self.marker_array.markers.append(marker)
        self.marker_pub.publish(self.marker_array)

import tf.transformations as tf_trans
if __name__ == '__main__':
    
    detector = Detector()
    # rospy.spin()

    while not rospy.is_shutdown():
        # print("global global_bbox",global_bbox)
        if global_bbox is not None:
            boxes = []
            for bbox in global_bbox.markers:
                # print(bbox)
                x, y, z = bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z
                dx, dy, dz = bbox.scale.x, bbox.scale.y, bbox.scale.z
                quat = (bbox.pose.orientation.x, bbox.pose.orientation.y, bbox.pose.orientation.z, bbox.pose.orientation.w)
                euler = tf_trans.euler_from_quaternion(quat)
                heading = euler[2]  # Z轴旋转角度
                # print("(x,y,z),(dx,dy,dz),heading",(x,y,z),(dx,dy,dz),heading)
                box = detector.get_3d_box((x,y,z),(dx,dy,dz),heading)
                box = box.transpose(1,0).ravel()
                boxes.append(box)
                # detector.display(boxes)
            for i in range(1):
                # print(i)
                detector.display(boxes)
                time.sleep(1)