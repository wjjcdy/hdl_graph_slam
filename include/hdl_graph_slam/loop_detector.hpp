// SPDX-License-Identifier: BSD-2-Clause

#ifndef LOOP_DETECTOR_HPP
#define LOOP_DETECTOR_HPP

#include <boost/format.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/graph_slam.hpp>

#include <g2o/types/slam3d/vertex_se3.h>

namespace hdl_graph_slam {

struct Loop {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Loop>;

  Loop(const KeyFrame::Ptr& key1, const KeyFrame::Ptr& key2, const Eigen::Matrix4f& relpose) : key1(key1), key2(key2), relative_pose(relpose) {}

public:
  KeyFrame::Ptr key1;
  KeyFrame::Ptr key2;
  Eigen::Matrix4f relative_pose;
};

/**
 * @brief this class finds loops by scam matching and adds them to the pose graph
 */
class LoopDetector {
public:
  typedef pcl::PointXYZI PointT;

  /**
   * @brief constructor
   * @param pnh
   */
  LoopDetector(ros::NodeHandle& pnh) {
    distance_thresh = pnh.param<double>("distance_thresh", 5.0);
    accum_distance_thresh = pnh.param<double>("accum_distance_thresh", 8.0);
    distance_from_last_edge_thresh = pnh.param<double>("min_edge_interval", 5.0);

    fitness_score_max_range = pnh.param<double>("fitness_score_max_range", std::numeric_limits<double>::max());
    fitness_score_thresh = pnh.param<double>("fitness_score_thresh", 0.5);

    registration = select_registration_method(pnh);
    last_edge_accum_distance = 0.0;
  }

  /**
   * @brief detect loops and add them to the pose graph
   * @param keyframes       keyframes
   * @param new_keyframes   newly registered keyframes
   * @param graph_slam      pose graph
   */
  std::vector<Loop::Ptr> detect(const std::vector<KeyFrame::Ptr>& keyframes, const std::deque<KeyFrame::Ptr>& new_keyframes, hdl_graph_slam::GraphSLAM& graph_slam) {
    std::vector<Loop::Ptr> detected_loops;
    // 遍历搜索新的keyframe 与 历史keyframe一帧一帧匹配是否出现闭环
    for(const auto& new_keyframe : new_keyframes) {
      // 所有可能的配对
      auto candidates = find_candidates(keyframes, new_keyframe);
      // 将新的关键帧配对的历史帧一一匹配，看是否匹配
      auto loop = matching(candidates, new_keyframe, graph_slam);
      // 存在闭环，则放入闭环队列中
      if(loop) {
        detected_loops.push_back(loop);
      }
    }

    return detected_loops;
  }

  double get_distance_thresh() const {
    return distance_thresh;
  }

private:
  /**
   * @brief find loop candidates. A detected loop begins at one of #keyframes and ends at #new_keyframe
   * @param keyframes      candidate keyframes of loop start
   * @param new_keyframe   loop end keyframe
   * @return loop candidates
   * 根据new key在历史中查找对应keyframe
   */
  std::vector<KeyFrame::Ptr> find_candidates(const std::vector<KeyFrame::Ptr>& keyframes, const KeyFrame::Ptr& new_keyframe) const {
    // too close to the last registered loop edge
    // 累计距离过近，则无需进行闭环检测，无意义
    // 即闭环需要经过一定距离间隔
    if(new_keyframe->accum_distance - last_edge_accum_distance < distance_from_last_edge_thresh) {
      return std::vector<KeyFrame::Ptr>();
    }

    std::vector<KeyFrame::Ptr> candidates;
    candidates.reserve(32);

    for(const auto& k : keyframes) {
      // traveled distance between keyframes is too small
      // 闭环匹配的两帧累计距离间隔须有一定范围
      if(new_keyframe->accum_distance - k->accum_distance < accum_distance_thresh) {
        continue;
      }

      // 两帧关键帧的坐标间隔小于一定范围则认为是匹配对
      const auto& pos1 = k->node->estimate().translation();
      const auto& pos2 = new_keyframe->node->estimate().translation();

      // estimated distance between keyframes is too small
      double dist = (pos1.head<2>() - pos2.head<2>()).norm();
      if(dist > distance_thresh) {
        continue;
      }
      // 
      candidates.push_back(k);
    }

    return candidates;
  }

  /**
   * @brief To validate a loop candidate this function applies a scan matching between keyframes consisting the loop. If they are matched well, the loop is added to the pose graph
   * @param candidate_keyframes  candidate keyframes of loop start
   * @param new_keyframe         loop end keyframe
   * @param graph_slam           graph slam
   */
  Loop::Ptr matching(const std::vector<KeyFrame::Ptr>& candidate_keyframes, const KeyFrame::Ptr& new_keyframe, hdl_graph_slam::GraphSLAM& graph_slam) {
    if(candidate_keyframes.empty()) {
      return nullptr;
    }

    // 新的关键帧作为参考点云
    registration->setInputTarget(new_keyframe->cloud);

    double best_score = std::numeric_limits<double>::max();
    KeyFrame::Ptr best_matched;
    Eigen::Matrix4f relative_pose;

    std::cout << std::endl;
    std::cout << "--- loop detection ---" << std::endl;
    std::cout << "num_candidates: " << candidate_keyframes.size() << std::endl;
    std::cout << "matching" << std::flush;
    auto t1 = ros::Time::now();

    // 遍历每一个keyframe，求出对应中最佳的关键帧
    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    for(const auto& candidate : candidate_keyframes) {
      registration->setInputSource(candidate->cloud);
      Eigen::Isometry3d new_keyframe_estimate = new_keyframe->node->estimate();
      new_keyframe_estimate.linear() = Eigen::Quaterniond(new_keyframe_estimate.linear()).normalized().toRotationMatrix();
      Eigen::Isometry3d candidate_estimate = candidate->node->estimate();
      candidate_estimate.linear() = Eigen::Quaterniond(candidate_estimate.linear()).normalized().toRotationMatrix();
      Eigen::Matrix4f guess = (new_keyframe_estimate.inverse() * candidate_estimate).matrix().cast<float>();
      guess(2, 3) = 0.0;
      registration->align(*aligned, guess);
      std::cout << "." << std::flush;

      // 根据匹配结果，找到最佳匹配的一帧，score越小越好
      double score = registration->getFitnessScore(fitness_score_max_range);
      if(!registration->hasConverged() || score > best_score) {
        continue;
      }

      best_score = score;
      best_matched = candidate;
      relative_pose = registration->getFinalTransformation();
    }

    auto t2 = ros::Time::now();
    std::cout << " done" << std::endl;
    std::cout << "best_score: " << boost::format("%.3f") % best_score << "    time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

    if(best_score > fitness_score_thresh) {
      std::cout << "loop not found..." << std::endl;
      return nullptr;
    }

    std::cout << "loop found!!" << std::endl;
    std::cout << "relpose: " << relative_pose.block<3, 1>(0, 3) << " - " << Eigen::Quaternionf(relative_pose.block<3, 3>(0, 0)).coeffs().transpose() << std::endl;

    // 找到闭环，并记录此次闭环时的累计距离
    last_edge_accum_distance = new_keyframe->accum_distance;

    // 返回闭环结果
    return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose);
  }

private:
  double distance_thresh;                 // estimated distance between keyframes consisting a loop must be less than this distance
  double accum_distance_thresh;           // traveled distance between ...
  double distance_from_last_edge_thresh;  // a new loop edge must far from the last one at least this distance

  double fitness_score_max_range;  // maximum allowable distance between corresponding points
  double fitness_score_thresh;     // threshold for scan matching

  double last_edge_accum_distance;

  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace hdl_graph_slam

#endif  // LOOP_DETECTOR_HPP
