// SPDX-License-Identifier: BSD-2-Clause

#include <string>

#include <ros/ros.h>
#include <ros/time.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_listener.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace hdl_graph_slam {

class PrefilteringNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZI PointT;

  PrefilteringNodelet() {}
  virtual ~PrefilteringNodelet() {}

  virtual void onInit() {
    nh = getNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    if(private_nh.param<bool>("deskewing", false)) {
      imu_sub = nh.subscribe("/imu/data", 1, &PrefilteringNodelet::imu_callback, this);
    }

    points_sub = nh.subscribe("/velodyne_points", 64, &PrefilteringNodelet::cloud_callback, this);
    points_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 32);
    colored_pub = nh.advertise<sensor_msgs::PointCloud2>("/colored_points", 32);
  }

private:
  void initialize_params() {
    std::string downsample_method = private_nh.param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);

    // 两种滤波方式，体素滤波和近似体素滤波，均采用pcl库
    // 体素滤波，即采用一个体素中所有点的均值替代
    // 近似体素滤波，即采用一个体素的中心点替代
    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = voxelgrid;
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      boost::shared_ptr<pcl::ApproximateVoxelGrid<PointT>> approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
    }

    // 两种孤立噪点滤波功能，pcl库
    // StatisticalOutlierRemoval: 临近点采用高斯分布，距离超出标准差的剔除
    // RadiusOutlierRemoval: 当前点阈值距离范围内其他点个数较少，则剔除
    std::string outlier_removal_method = private_nh.param<std::string>("outlier_removal_method", "STATISTICAL");
    if(outlier_removal_method == "STATISTICAL") {
      int mean_k = private_nh.param<int>("statistical_mean_k", 20);
      double stddev_mul_thresh = private_nh.param<double>("statistical_stddev", 1.0);
      std::cout << "outlier_removal: STATISTICAL " << mean_k << " - " << stddev_mul_thresh << std::endl;

      pcl::StatisticalOutlierRemoval<PointT>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointT>());
      sor->setMeanK(mean_k);
      sor->setStddevMulThresh(stddev_mul_thresh);
      outlier_removal_filter = sor;
    } else if(outlier_removal_method == "RADIUS") {
      double radius = private_nh.param<double>("radius_radius", 0.8);
      int min_neighbors = private_nh.param<int>("radius_min_neighbors", 2);
      std::cout << "outlier_removal: RADIUS " << radius << " - " << min_neighbors << std::endl;

      pcl::RadiusOutlierRemoval<PointT>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT>());
      rad->setRadiusSearch(radius);
      rad->setMinNeighborsInRadius(min_neighbors);
      outlier_removal_filter = rad;
    } else {
      std::cout << "outlier_removal: NONE" << std::endl;
    }

    use_distance_filter = private_nh.param<bool>("use_distance_filter", true);
    distance_near_thresh = private_nh.param<double>("distance_near_thresh", 1.0);
    distance_far_thresh = private_nh.param<double>("distance_far_thresh", 100.0);

    base_link_frame = private_nh.param<std::string>("base_link_frame", "");
  }

  // 高速接收IMU数据，并放入队列中
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    imu_queue.push_back(imu_msg);
  }

  void cloud_callback(pcl::PointCloud<PointT>::ConstPtr src_cloud) {
    if(src_cloud->empty()) {
      return;
    }

    // 运动畸变矫正
    src_cloud = deskewing(src_cloud);

    // if base_link_frame is defined, transform the input cloud to the frame
    if(!base_link_frame.empty()) {
      if(!tf_listener.canTransform(base_link_frame, src_cloud->header.frame_id, ros::Time(0))) {
        std::cerr << "failed to find transform between " << base_link_frame << " and " << src_cloud->header.frame_id << std::endl;
      }

      tf::StampedTransform transform;
      tf_listener.waitForTransform(base_link_frame, src_cloud->header.frame_id, ros::Time(0), ros::Duration(2.0));
      tf_listener.lookupTransform(base_link_frame, src_cloud->header.frame_id, ros::Time(0), transform);

      pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
      pcl_ros::transformPointCloud(*src_cloud, *transformed, transform);
      transformed->header.frame_id = base_link_frame;
      transformed->header.stamp = src_cloud->header.stamp;
      src_cloud = transformed;
    }
    // 有效距离滤波
    pcl::PointCloud<PointT>::ConstPtr filtered = distance_filter(src_cloud);
    // 降采样
    filtered = downsample(filtered);
    // 噪点滤波
    filtered = outlier_removal(filtered);

    points_pub.publish(filtered);
  }

  // 原始点云降采样
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  // 孤立等噪点滤波
  pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!outlier_removal_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    outlier_removal_filter->setInputCloud(cloud);
    outlier_removal_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  // 滤除较远和较近的点云
  pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    filtered->reserve(cloud->size());

    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
      double d = p.getVector3fMap().norm();
      return d > distance_near_thresh && d < distance_far_thresh;
    });

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    filtered->header = cloud->header;

    return filtered;
  }

  // 点云去运动畸变，？？？？？？？
  // 多线激光雷达，理论上应该是垂直同时，旋转扫描
  // 矫正的时间间隔，不知为何，却是以点云的顺序进行全部计算时刻，进行校准
  pcl::PointCloud<PointT>::ConstPtr deskewing(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    // 获取点云接收的时间戳
    ros::Time stamp = pcl_conversions::fromPCL(cloud->header.stamp);
    // 无imu新数据，不畸变矫正
    if(imu_queue.empty()) {
      return cloud;
    }

    // the color encodes the point number in the point sequence
    if(colored_pub.getNumSubscribers()) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
      colored->header = cloud->header;
      colored->is_dense = cloud->is_dense;
      colored->width = cloud->width;
      colored->height = cloud->height;
      colored->resize(cloud->size());

      // 每个点云根据扫描顺序显示颜色，方便观察
      for(int i = 0; i < cloud->size(); i++) {
        double t = static_cast<double>(i) / cloud->size();
        colored->at(i).getVector4fMap() = cloud->at(i).getVector4fMap();
        colored->at(i).r = 255 * t;
        colored->at(i).g = 128;
        colored->at(i).b = 255 * (1 - t);
      }
      colored_pub.publish(colored);
    }

    // 获取imu数据最老数据
    sensor_msgs::ImuConstPtr imu_msg = imu_queue.front();

    // 遍历imu队列中的，根据时间戳，找到时间戳一致的imu数据
    auto loc = imu_queue.begin();
    for(; loc != imu_queue.end(); loc++) {
      imu_msg = (*loc);
      if((*loc)->header.stamp > stamp) {
        break;
      }
    }

    // 剔除已遍历后的imu数据
    imu_queue.erase(imu_queue.begin(), loc);

    // 获取当前角速度
    Eigen::Vector3f ang_v(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    // 反向
    ang_v *= -1;

    // 初始化畸变矫正后的点云数据
    pcl::PointCloud<PointT>::Ptr deskewed(new pcl::PointCloud<PointT>());
    deskewed->header = cloud->header;
    deskewed->is_dense = cloud->is_dense;
    deskewed->width = cloud->width;
    deskewed->height = cloud->height;
    deskewed->resize(cloud->size());

    double scan_period = private_nh.param<double>("scan_period", 0.1);
    for(int i = 0; i < cloud->size(); i++) {
      const auto& pt = cloud->at(i);

      // TODO: transform IMU data into the LIDAR frame
      // 计算每个点云时刻相对于起始点的时刻
      double delta_t = scan_period * static_cast<double>(i) / cloud->size();
      // 根据匀角速度变换，计算每个点的角度姿态角
      Eigen::Quaternionf delta_q(1, delta_t / 2.0 * ang_v[0], delta_t / 2.0 * ang_v[1], delta_t / 2.0 * ang_v[2]);
      // 矫正畸变
      Eigen::Vector3f pt_ = delta_q.inverse() * pt.getVector3fMap();

      deskewed->at(i) = cloud->at(i);
      deskewed->at(i).getVector3fMap() = pt_;
    }

    return deskewed;
  }

private:
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  ros::Subscriber imu_sub;
  std::vector<sensor_msgs::ImuConstPtr> imu_queue;

  ros::Subscriber points_sub;
  ros::Publisher points_pub;

  ros::Publisher colored_pub;

  tf::TransformListener tf_listener;

  std::string base_link_frame;

  bool use_distance_filter;
  double distance_near_thresh;
  double distance_far_thresh;

  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Filter<PointT>::Ptr outlier_removal_filter;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::PrefilteringNodelet, nodelet::Nodelet)
