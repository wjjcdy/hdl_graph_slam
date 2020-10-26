#include <deque>
#include <chrono>
#include <iostream>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosgraph_msgs/Clock.h>
#include <pcl_conversions/pcl_conversions.h>

#include <apps/prefiltering_nodelet.hpp>
#include <apps/scan_matching_odometry_nodelet.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "offline_scan_matching_odometry");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  ros::NodeHandle prefiltering_nh("/prefiltering_nodelet");
  ros::NodeHandle scan_matching_nh("/scan_matching_odometry_nodelet");

  hdl_graph_slam::PrefilteringNodelet prefiltering;
  prefiltering.init(nh, prefiltering_nh);

  hdl_graph_slam::ScanMatchingOdometryNodelet scan_matching;
  scan_matching.init(nh, scan_matching_nh);

  std::string bag_filename = private_nh.param<std::string>("rosbag", "");
  ROS_INFO_STREAM("open " << bag_filename);

  rosbag::Bag bag(bag_filename, rosbag::bagmode::Read);
  if(!bag.isOpen()) {
    ROS_ERROR("failed to open input bag!!");
    return 1;
  }

  std::vector<std::string> topics = {"/velodyne_points"};
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  std::deque<std::chrono::high_resolution_clock::time_point> times = { std::chrono::high_resolution_clock::now() };
  ros::Publisher clock_pub = nh.advertise<rosgraph_msgs::Clock>("/clock", 1);
  ros::Publisher points_pub = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points_", 1);

  for(const auto& m : view) {
    if(!ros::ok()) {
      break;
    }

    times.push_back(std::chrono::high_resolution_clock::now());
    while(times.size() > 15) {
      times.pop_front();
    }
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(times.back() - times.front()).count() / 1e9;
    std::cout << "[" << m.getTime().toSec() <<"] " << times.size() / elapsed << "fps" << std::endl;

    rosgraph_msgs::Clock clock;
    clock.clock = m.getTime();
    clock_pub.publish(clock);

    sensor_msgs::PointCloud2::Ptr points_msg = m.instantiate<sensor_msgs::PointCloud2>();
    points_msg->header.frame_id = "velodyne";

    if(points_pub.getNumSubscribers()) {
      points_pub.publish(points_msg);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*points_msg, *cloud);

    pcl::PointCloud<pcl::PointXYZI>::ConstPtr filtered = prefiltering.filter(cloud);

    sensor_msgs::PointCloud2::Ptr filtered_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*filtered, *filtered_msg);

    scan_matching.cloud_callback(filtered_msg);

    ros::spinOnce();
  }

  return 0;
}