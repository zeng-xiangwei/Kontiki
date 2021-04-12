#ifndef LIDAR_EDGE_MEASUREMENT_H
#define LIDAR_EDGE_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>
#include "../sensors/lidar.h"
#include "../trajectories/trajectory.h"
#include "../trajectory_estimator.h"
#include "ceres/rotation.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDAREdgeMeasurement {
public:
	LiDAREdgeMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
					             Eigen::Vector3d &last_point_b, double timestamp, double huber_loss, double weight)
			: lidar_(lidar), curr_point_(curr_point), last_point_a_(last_point_a), last_point_b_(last_point_b), 
				timestamp_(timestamp), loss_function_(huber_loss), weight_(weight) {}

  LiDAREdgeMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
					             Eigen::Vector3d &last_point_b, double timestamp, double huber_loss)
      : LiDAREdgeMeasurement(lidar, curr_point, last_point_a, last_point_b, timestamp, huber_loss, 1.0) {}

  LiDAREdgeMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
					             Eigen::Vector3d &last_point_b, double timestamp)
      : LiDAREdgeMeasurement(lidar, curr_point, last_point_a, last_point_b, timestamp, 5.) {}

	template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> calculatePoint2Line(const type::Trajectory<TrajectoryModel, T>& trajectory,
                                            const type::LiDAR<LiDARModel, T>& lidar) const {
    int flags = trajectories::EvaluationFlags::EvalPosition | trajectories::EvaluationFlags::EvalOrientation;

    auto T_IktoG = trajectory.Evaluate(T(timestamp_), flags);

		// 将当前激光系的点云转换到世界系
    Eigen::Matrix<T, 3, 1> p_Lk = curr_point_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_G = T_IktoG->orientation * p_Lk + T_IktoG->position;

		Eigen::Matrix<T, 3, 1> lpa = last_point_a_.cast<T>();
		Eigen::Matrix<T, 3, 1> lpb = last_point_b_.cast<T>();
		// Eigen::Matrix<T, 3, 1> lpa{T(last_point_a_.x()), T(last_point_a_.y()), T(last_point_a_.z())};
		// Eigen::Matrix<T, 3, 1> lpb{T(last_point_b_.x()), T(last_point_b_.y()), T(last_point_b_.z())};

		Eigen::Matrix<T, 3, 1> nu = (p_G - lpa).cross(p_G - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		Eigen::Matrix<T, 3, 1> error = nu / de.norm();

    return error;
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    Eigen::Matrix<T, 3, 1> dist = calculatePoint2Line<TrajectoryModel, T>(trajectory, lidar);
    return T(weight_) * (dist);
  }


std::shared_ptr<LiDARModel> lidar_;
Eigen::Vector3d curr_point_, last_point_a_, last_point_b_;
double timestamp_, weight_;


protected:

	template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDAREdgeMeasurement<LiDARModel> &m) : measurement(m){}

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

    const LiDAREdgeMeasurement &measurement;
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
      double tmin, tmax;
      if(this->lidar_->TimeOffsetIsLocked()) {
          tmin = timestamp_;
          tmax = timestamp_;
      }
      else {
          tmin = timestamp_ - this->lidar_->max_time_offset();
          tmax = timestamp_ + this->lidar_->max_time_offset();
      }

      /// A.1 先将轨迹参数添加到parameters中 (control points)
      estimator.AddTrajectoryForTimes({{tmin, tmax}},
                                      residual->trajectory_meta,
                                      parameters);

      /// A.2 再将lidar传感器的参数添加到parameters中 (relative pose and timeoffset and so on ..(if have))
      lidar_->AddToProblem(estimator.problem(), {{tmin, tmax}},
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

#endif // LIDAR_EDGE_MEASUREMENT_H