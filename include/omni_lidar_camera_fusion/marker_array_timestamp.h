#ifndef MARKER_ARRAY_TIMESTAMP_H
#define MARKER_ARRAY_TIMESTAMP_H

#include <ros/message_traits.h>
#include <visualization_msgs/MarkerArray.h>

namespace ros
{
namespace message_traits
{

template<>
struct TimeStamp<visualization_msgs::MarkerArray>
{
  static ros::Time value(const visualization_msgs::MarkerArray& m)
  {
    return m.markers.empty() ? ros::Time() : m.markers[0].header.stamp;
  }
};

} // namespace message_traits
} // namespace ros

#endif // MARKER_ARRAY_TIMESTAMP_H