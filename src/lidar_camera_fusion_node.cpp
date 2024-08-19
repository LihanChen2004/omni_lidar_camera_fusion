#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include "omni_lidar_camera_fusion/omni_lidar_camera_fusion.hpp"

OmniLidarCameraFusion::OmniLidarCameraFusion(ros::NodeHandle & nh)
{
  nh.getParam("camera_frame_id", camera_frame_id_);
  nh.getParam("lidar_frame_id", lidar_frame_id_);
  nh.getParam("/pcd_topic", pcTopic_);
  nh.getParam("/img_topic", imgTopic_);

  nh.getParam("/cam_hfov", cam_hfov_);
  nh.getParam("/cam_vfov", cam_vfov_);

  // Get the transform from lidar frame to camera frame
  tf::TransformListener listener;
  tf::StampedTransform transform;
  try {
    listener.waitForTransform(camera_frame_id_, lidar_frame_id_, ros::Time(0), ros::Duration(20.0));
    listener.lookupTransform(camera_frame_id_, lidar_frame_id_, ros::Time(0), transform);
  } catch (tf::TransformException & ex) {
    ROS_ERROR("%s", ex.what());
    return;
  }
  Eigen::Affine3d lidar2camera_eigen;
  tf::transformTFToEigen(transform, lidar2camera_eigen);
  lidar2camera_ = lidar2camera_eigen.matrix().cast<float>();

  range_image_ = boost::make_shared<pcl::RangeImageSpherical>();

  // Initialize publishers
  pcd_pub_ = nh.advertise<PointCloud>("/sensor_scan_rgb", 1);
  img_pub_ = nh.advertise<sensor_msgs::Image>("/sensor_scan_image", 1);

  // Initialize message filters and synchronizer
  pcd_sub.subscribe(nh, pcTopic_, 1);
  img_sub.subscribe(nh, imgTopic_, 1);

  sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(10);
  sync_->connectInput(pcd_sub, img_sub);
  sync_->registerCallback(boost::bind(&OmniLidarCameraFusion::callback, this, _1, _2));
}

void OmniLidarCameraFusion::filterPointCloud(
  PointCloud::Ptr & cloud, float min_dist, float max_dist)
{
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  PointCloud::Ptr filtered_cloud(new PointCloud);
  for (const auto & point : cloud->points) {
    double distance = std::hypot(point.x, point.y);
    if (distance >= min_dist && distance <= max_dist) {
      filtered_cloud->points.push_back(point);
    }
  }
  cloud.swap(filtered_cloud);
}

void OmniLidarCameraFusion::callback(
  const sensor_msgs::PointCloud2ConstPtr & input_cloud_msg,
  const sensor_msgs::ImageConstPtr & input_image_msg)
{
  // Convert image data
  auto cv_image_ptr = cv_bridge::toCvShare(input_image_msg, sensor_msgs::image_encodings::BGR8);
  auto cv_color_ptr = cv_bridge::toCvCopy(input_image_msg, sensor_msgs::image_encodings::BGR8);

  // Convert point cloud data
  PointCloud::Ptr original_cloud(new PointCloud);
  pcl::fromROSMsg(*input_cloud_msg, *original_cloud);

  // Transform point cloud to camera frame
  pcl::transformPointCloud(*original_cloud, *original_cloud, lidar2camera_);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  colored_point_cloud->header.frame_id = input_image_msg->header.frame_id;
  colored_point_cloud->header.stamp = original_cloud->header.stamp;

  for (const auto & point : original_cloud->points) {
    float r = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    float phi = std::asin(point.y / r);
    float theta = std::atan2(point.x, point.z);

    unsigned int u = ((theta / (cam_hfov_ * M_PI / 180) + 0.5) * input_image_msg->width);
    unsigned int v = ((phi / (cam_vfov_ * M_PI / 180) + 0.5) * input_image_msg->height);

    cv::Vec3b color;
    if (v >= 0 && v < (input_image_msg->height) && u >= 0 && u < (input_image_msg->width)) {
      color = cv_image_ptr->image.at<cv::Vec3b>(v, u);
    } else {
      ROS_WARN("Invalid pixel coordinates (%d, %d)", u, v);
      ROS_DEBUG("point_x: %f, point_y: %f, point_z: %f", point.x, point.y, point.z);
      ROS_DEBUG("r: %f, phi: %f, theta: %f", r, phi, theta);
      continue;
    }

    // Point cloud coloring
    pcl::PointXYZRGB colored_point;
    colored_point.x = point.x;
    colored_point.y = point.y;
    colored_point.z = point.z;
    colored_point.r = color[2];
    colored_point.g = color[1];
    colored_point.b = color[0];
    colored_point_cloud->points.push_back(colored_point);

    // Draw point on image
    cv::circle(
      cv_color_ptr->image, cv::Point(u, v), 1, CV_RGB(color[2] + 40, color[1] + 40, color[0] + 40),
      -1);
  }

  // Transform point cloud back to lidar frame
  pcl::transformPointCloud(*colored_point_cloud, *colored_point_cloud, lidar2camera_.inverse());
  colored_point_cloud->header.frame_id = input_cloud_msg->header.frame_id;
  pcd_pub_.publish(colored_point_cloud);

  // Publish the image with points
  sensor_msgs::ImagePtr output_image_msg =
    cv_bridge::CvImage(input_image_msg->header, "bgr8", cv_color_ptr->image).toImageMsg();
  img_pub_.publish(output_image_msg);
}

int main(int argc, char ** argv)
{
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  ros::init(argc, argv, "point_cloud_on_image");
  ros::NodeHandle nh;

  OmniLidarCameraFusion pc_on_img(nh);

  ros::spin();
  return 0;
}