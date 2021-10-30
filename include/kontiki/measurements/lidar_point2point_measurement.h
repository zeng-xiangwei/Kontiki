#ifndef LIDAR_POINT2POINT_MEASUREMENT_H
#define LIDAR_POINT2POINT_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>
#include "../sensors/lidar.h"
#include "../trajectories/trajectory.h"
#include "../trajectory_estimator.h"
#include "ceres/rotation.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDARPoint2PointMeasurement {
public:
	LiDARPoint2PointMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
					             double curr_timestamp, double last_timestamp, double huber_loss, double weight)
			: lidar_(lidar), curr_point_(curr_point), last_point_a_(last_point_a), curr_timestamp_(curr_timestamp),
				last_timestamp_(last_timestamp), loss_function_(huber_loss), weight_(weight) {}

  LiDARPoint2PointMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
					             double curr_timestamp, double last_timestamp, double huber_loss)
      : LiDARPoint2PointMeasurement(lidar, curr_point, last_point_a, curr_timestamp, last_timestamp, huber_loss, 1.0) {}

  LiDARPoint2PointMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
					             double curr_timestamp, double last_timestamp)
      : LiDARPoint2PointMeasurement(lidar, curr_point, last_point_a, curr_timestamp, last_timestamp, 5.) {}

	template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> calculatePoint2Point(const type::Trajectory<TrajectoryModel, T>& trajectory,
                                            const type::LiDAR<LiDARModel, T>& lidar) const {
    int flags = trajectories::EvaluationFlags::EvalPosition;
    auto pos_cur = trajectory.Evaluate(T(curr_timestamp_) + lidar.time_offset(), flags);
    auto pos_last = trajectory.Evaluate(T(last_timestamp_) + lidar.time_offset(), flags);
    T angle_cur = -pos_cur->position.x();
    T angle_last = -pos_last->position.x();
    Eigen::Matrix<T, 3, 1> trans = lidar.relative_position();
    Eigen::Quaternion<T> orien = lidar.relative_orientation();

    Eigen::Matrix<T, 3, 1> p_cur = curr_point_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_last = last_point_a_.cast<T>();
    Eigen::Matrix<T, 3, 3> orien2init_cur, orien2init_last;
    orien2init_cur << ceres::cos(angle_cur), -ceres::sin(angle_cur), T(0),
                      ceres::sin(angle_cur),  ceres::cos(angle_cur), T(0),
                      T(0),        T(0),       T(1);

    orien2init_last << ceres::cos(angle_last), -ceres::sin(angle_last), T(0),
                       ceres::sin(angle_last),  ceres::cos(angle_last), T(0),
                       T(0),        T(0),       T(1);

    p_cur = orien * p_cur + trans;
    p_cur = orien2init_cur * p_cur;

    p_last = orien * p_last + trans;
    p_last = orien2init_last * p_last;

		Eigen::Matrix<T, 3, 1> error = p_cur - p_last;

    return error;
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    Eigen::Matrix<T, 3, 1> dist = calculatePoint2Point<TrajectoryModel, T>(trajectory, lidar);
    return T(weight_) * (dist);
  }


std::shared_ptr<LiDARModel> lidar_;
Eigen::Vector3d curr_point_, last_point_a_;
double curr_timestamp_, last_timestamp_, weight_;


protected:

	template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDARPoint2PointMeasurement<LiDARModel> &m) : measurement(m){}

    /// 每次计算residual,都会根据优化后的参数重新构造traj和lidar sensor
    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      size_t offset = 0;
      auto trajectory = entity::Map<TrajectoryModel, T>(&params[offset], trajectory_meta);

      offset += trajectory_meta.NumParameters();
      auto lidar = entity::Map<LiDARModel, T>(&params[offset], lidar_meta);

      Eigen::Map<Eigen::Matrix<T,3,1>> r(residual);
      r = measurement.Error<TrajectoryModel, T>(trajectory, lidar);
      return true;
    }

    const LiDARPoint2PointMeasurement &measurement;
    typename TrajectoryModel::Meta trajectory_meta;
    typename LiDARModel::Meta lidar_meta;
  }; // Residual;

	template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {

      using ResidualImpl = Residual<TrajectoryModel>;
      auto residual = new ResidualImpl(*this);
      auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);
      std::vector<entity::ParameterInfo<double>> parameters;

      // Find timespans for the two observations
      double tmin_cur, tmax_cur, tmin_last, tmax_last;
      if(this->lidar_->TimeOffsetIsLocked()) {
          tmin_cur = curr_timestamp_;
          tmax_cur = curr_timestamp_;

          tmin_last = last_timestamp_;
          tmax_last = last_timestamp_;
      }
      else {
          tmin_cur = curr_timestamp_ - this->lidar_->max_time_offset();
          tmax_cur = curr_timestamp_ + this->lidar_->max_time_offset();
          
          tmin_last = last_timestamp_ - this->lidar_->max_time_offset();
          tmax_last = last_timestamp_ + this->lidar_->max_time_offset();
      }

      /// A.1 先将轨迹参数添加到parameters中 (control points)
      estimator.AddTrajectoryForTimes({
                                        {tmin_last, tmax_last},
                                        {tmin_cur, tmax_cur}
                                      },
                                      residual->trajectory_meta,
                                      parameters);

      /// A.2 再将lidar传感器的参数添加到parameters中 (relative pose and timeoffset and so on ..(if have))
      lidar_->AddToProblem(estimator.problem(),
                           {
                              {tmin_last, tmax_last},
                              {tmin_cur, tmax_cur}
                           },
                           residual->lidar_meta, parameters);

      /// B.1 先往cost_function中添加待优化参数
      // Add parameters to cost function
      for (auto& pi : parameters) {
        cost_function->AddParameterBlock(pi.size);
      }
      // Add measurement info
      cost_function->SetNumResiduals(3);

      /// B.2 再添加residual
      // Give residual block to Problem
      std::vector<double*> params = entity::ParameterInfo<double>::ToParameterBlocks(parameters);

      estimator.problem().AddResidualBlock(cost_function,
                                           &loss_function_,
                                           params);
//      estimator.problem().AddResidualBlock(cost_function,
//                                           nullptr,
//                                           params);

//      estimator.problem().AddResidualBlock(cost_function,
//                                           nullptr,
//                                           entity::ParameterInfo<double>::ToParameterBlocks(parameters));

  }

	// The loss function is not a pointer since the Problem does not take ownership.
  ceres::HuberLoss loss_function_;

  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;


};


} // end namespace measurements
} // end namespace kontiki

#endif // LIDAR_POINT2POINT_MEASUREMENT_H