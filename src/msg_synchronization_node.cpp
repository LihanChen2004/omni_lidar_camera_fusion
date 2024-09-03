#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

class SyncNode
{
public:
  SyncNode()
  {
    pointcloud_sub_.subscribe(nh_, "/registered_scan", 1);
    image_sub_.subscribe(nh_, "/camera/image", 1);
    semantic_sub_.subscribe(nh_, "/camera/semantic_image", 1);

    sync_.reset(new Sync(MySyncPolicy(10), pointcloud_sub_, image_sub_, semantic_sub_));
    sync_->registerCallback(boost::bind(&SyncNode::callback, this, _1, _2, _3));

    pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("synced/registered_scan", 1);
    image_pub_ = nh_.advertise<sensor_msgs::Image>("synced/camera/image", 1);
    semantic_pub_ = nh_.advertise<sensor_msgs::Image>("synced/camera/semantic", 1);
  }

private:
  // 回调函数
  void callback(
    const sensor_msgs::PointCloud2ConstPtr & pointcloud, 
    const sensor_msgs::ImageConstPtr & image,
    const sensor_msgs::ImageConstPtr & semantic)
  {
    // 打印接收到的同步消息的日志信息
    ROS_INFO("Received synchronized messages");

    // 发布同步后的消息
    pointcloud_pub_.publish(pointcloud);
    image_pub_.publish(image);
    semantic_pub_.publish(semantic);
  }

  ros::NodeHandle nh_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud_sub_;
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> semantic_sub_;

  // 定义同步策略
  using MySyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Image>;
  using Sync = message_filters::Synchronizer<MySyncPolicy>;
  boost::shared_ptr<Sync> sync_;

  // 发布器
  ros::Publisher pointcloud_pub_;
  ros::Publisher image_pub_;
  ros::Publisher semantic_pub_;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "sync_node");
  SyncNode sync_node;
  ros::spin();
  return 0;
}
