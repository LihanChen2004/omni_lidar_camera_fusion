#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include "omni_lidar_camera_fusion/omni_lidar_camera_fusion.hpp"

OmniLidarCameraFusion::OmniLidarCameraFusion()
{
  ros::NodeHandle nh("~");
  nh.getParam("camera_frame_id", camera_frame_id_);
  nh.getParam("lidar_frame_id", lidar_frame_id_);
  nh.getParam("pcd_topic", pcTopic_);
  nh.getParam("img_topic", imgTopic_);
  nh.getParam("semantic_pcd_topic", semantic_pcTopic_);
  nh.getParam("semantic_topic", semanticTopic_);
  nh.getParam("cam_hfov", cam_hfov_);
  nh.getParam("cam_vfov", cam_vfov_);
  nh.getParam("lidar_min_range", lidar_min_range_);
  nh.getParam("lidar_max_range", lidar_max_range_);

  // Get the transform from lidar frame to camera frame
  tf::TransformListener listener;
  tf::StampedTransform transform;
  while (true) {
    try {
      listener.lookupTransform(camera_frame_id_, lidar_frame_id_, ros::Time(0), transform);
      break;
    } catch (tf::TransformException & ex) {
      ROS_WARN("%s", ex.what());
      ros::Duration(1.0).sleep();
    }
  }
  Eigen::Affine3d lidar2camera_eigen;
  tf::transformTFToEigen(transform, lidar2camera_eigen);
  lidar2camera_ = lidar2camera_eigen.matrix().cast<float>();

  range_image_ = boost::make_shared<pcl::RangeImageSpherical>();

  // Initialize publishers
  pcd_pub_ = nh.advertise<PointCloud>("/sensor_scan_rgb", 1);
  semantic_pcd_pub_ = nh.advertise<PointCloud>("/sensor_scan_semantic_pcd", 1);
  img_pub_ = nh.advertise<sensor_msgs::Image>("/sensor_scan_image", 1);
  semantic_pub_ = nh.advertise<sensor_msgs::Image>("/sensor_scan_semantic_img", 1);

  // Initialize message filters and synchronizer
  pcd_sub_.subscribe(nh, pcTopic_, 1);
  semantic_pcd_sub_.subscribe(nh, semantic_pcTopic_, 1);
  img_sub_.subscribe(nh, imgTopic_, 1);
  semantic_sub_.subscribe(nh, semanticTopic_, 1);

  sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(10);
  sync_->connectInput(pcd_sub_, img_sub_, semantic_sub_);
  sync_->registerCallback(boost::bind(&OmniLidarCameraFusion::callback, this, _1, _2, _3));
}

void OmniLidarCameraFusion::filterPointCloud(
  PointCloud::Ptr & cloud, float min_dist, float max_dist)
{
  for (auto it = cloud->begin(); it != cloud->end(); ++it) {
    float distance = std::sqrt(it->x * it->x + it->y * it->y + it->z * it->z);
    if (distance < min_dist || distance > max_dist) {
      it = cloud->erase(it);
      --it;
    }
  }
}

void OmniLidarCameraFusion::callback(
  const sensor_msgs::PointCloud2ConstPtr & input_cloud_msg,
  const sensor_msgs::ImageConstPtr & input_image_msg,
  const sensor_msgs::ImageConstPtr & input_semantic_image_msg)
{
  // Convert image data
  auto cv_image_ptr = cv_bridge::toCvShare(input_image_msg, sensor_msgs::image_encodings::BGR8);
  auto cv_color_ptr = cv_bridge::toCvCopy(input_image_msg, sensor_msgs::image_encodings::BGR8);
  auto cv_semantic_ptr = cv_bridge::toCvShare(input_semantic_image_msg, sensor_msgs::image_encodings::BGR8);

  // Convert point cloud data
  PointCloud::Ptr original_cloud(new PointCloud);
  pcl::fromROSMsg(*input_cloud_msg, *original_cloud);

  filterPointCloud(original_cloud, lidar_min_range_, lidar_max_range_);

  // Transform point cloud to camera frame
  pcl::transformPointCloud(*original_cloud, *original_cloud, lidar2camera_);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  colored_point_cloud->header.frame_id = input_image_msg->header.frame_id;
  colored_point_cloud->header.stamp = original_cloud->header.stamp;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr semantic_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  semantic_point_cloud->header.frame_id = input_image_msg->header.frame_id;
  semantic_point_cloud->header.stamp = original_cloud->header.stamp;

  for (const auto & point : original_cloud->points) {
    float r = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    float phi = std::asin(point.y / r);
    float theta = std::atan2(point.x, point.z);

    unsigned int u = ((theta / (cam_hfov_ * M_PI / 180) + 0.5) * input_image_msg->width);
    unsigned int v = ((phi / (cam_vfov_ * M_PI / 180) + 0.5) * input_image_msg->height);

    cv::Vec3b color;
    cv::Vec3b semantic_color;
    if (v >= 0 && v < (input_image_msg->height) && u >= 0 && u < (input_image_msg->width)) {
      color = cv_image_ptr->image.at<cv::Vec3b>(v, u);
      semantic_color = cv_semantic_ptr->image.at<cv::Vec3b>(v, u);
    } else {
      ROS_WARN(
        "Invalid pixel coordinates (%d, %d) \n point: (%f, %f, %f) \n r: %f, phi: %f, theta: %f", u,
        v, point.x, point.y, point.z, r, phi, theta);
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

    //Point cloud semanticing
    pcl::PointXYZRGB semantic_point;
    semantic_point.x = point.x;
    semantic_point.y = point.y;
    semantic_point.z = point.z;
    semantic_point.r = semantic_color[2];
    semantic_point.g = semantic_color[1];
    semantic_point.b = semantic_color[0];
    semantic_point_cloud->points.push_back(semantic_point);

    // Draw point on image
    cv::circle(
      cv_color_ptr->image, cv::Point(u, v), 1, CV_RGB(color[2] + 40, color[1] + 40, color[0] + 40),
      -1);

    // Draw point on semantic image
    cv::circle(
      cv_semantic_ptr->image, cv::Point(u, v), 1, CV_RGB(semantic_color[2] + 40, semantic_color[1] + 40, semantic_color[0] + 40),
      -1);
  }

  // Transform point cloud back to lidar frame
  pcl::transformPointCloud(*colored_point_cloud, *colored_point_cloud, lidar2camera_.inverse());
  pcl::transformPointCloud(*semantic_point_cloud, *semantic_point_cloud, lidar2camera_.inverse());
  colored_point_cloud->header.frame_id = input_cloud_msg->header.frame_id;
  semantic_point_cloud->header.frame_id = input_cloud_msg->header.frame_id;
  pcd_pub_.publish(colored_point_cloud);
  std::cout << "colored_point_cloud->points.size()" << colored_point_cloud->points.size() << std::endl;
  std::cout << "semantic_point_cloud->points.size()" << semantic_point_cloud->points.size() << std::endl;
  semantic_pcd_pub_.publish(semantic_point_cloud);

  // Publish the image with points
  sensor_msgs::ImagePtr output_image_msg =
    cv_bridge::CvImage(input_image_msg->header, "bgr8", cv_color_ptr->image).toImageMsg();
  img_pub_.publish(output_image_msg);
  sensor_msgs::ImagePtr output_semantic_msg =
    cv_bridge::CvImage(input_image_msg->header, "bgr8", cv_semantic_ptr->image).toImageMsg();
  semantic_pub_.publish(output_semantic_msg);
}

int main(int argc, char ** argv)
{
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  ros::init(argc, argv, "omni_lidar_camera_fusion");
  OmniLidarCameraFusion omnilidarcamerafusion;
  ros::spin();
  return 0;
}