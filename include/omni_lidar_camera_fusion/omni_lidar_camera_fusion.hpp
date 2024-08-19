#ifndef POINT_CLOUD_ON_IMAGE_HPP
#define POINT_CLOUD_ON_IMAGE_HPP

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/core/core.hpp>

using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

class OmniLidarCameraFusion
{
public:
  OmniLidarCameraFusion();

private:
  void callback(
    const sensor_msgs::PointCloud2ConstPtr & input_cloud_msg,
    const sensor_msgs::ImageConstPtr & input_image_msg);

  void filterPointCloud(PointCloud::Ptr & cloud, float min_dist, float max_dist);

  using MySyncPolicy =
    message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image>;

  ros::Publisher img_pub_;
  ros::Publisher pcd_pub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> pcd_sub_;
  message_filters::Subscriber<sensor_msgs::Image> img_sub_;
  std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

  std::string camera_frame_id_;
  std::string lidar_frame_id_;
  std::string imgTopic_;
  std::string pcTopic_;
  float cam_hfov_;
  float cam_vfov_;
  float lidar_min_range_;
  float lidar_max_range_;
  Eigen::Matrix4f lidar2camera_;
  boost::shared_ptr<pcl::RangeImageSpherical> range_image_;
};

#endif  // POINT_CLOUD_ON_IMAGE_HPP
