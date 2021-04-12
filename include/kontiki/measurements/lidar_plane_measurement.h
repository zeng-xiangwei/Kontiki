#ifndef LIDAR_PLANE_MEASUREMENT_H
#define LIDAR_PLANE_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>
#include "../sensors/lidar.h"
#include "../trajectories/trajectory.h"
#include "../trajectory_estimator.h"
#include "ceres/rotation.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDARPlaneMeasurement {
public:
	LiDARPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &plane_unit_norm,
						               double d, double timestamp, double huber_loss, double weight)
			: lidar_(lidar), curr_point_(curr_point), plane_unit_norm_(plane_unit_norm), d_(d), 
				timestamp_(timestamp), loss_function_(huber_loss), weight_(weight) {}

  LiDARPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &plane_unit_norm,
						               double d, double timestamp, double huber_loss)
      : LiDARPlaneMeasurement(lidar, curr_point, plane_unit_norm, d, timestamp, huber_loss, 1.0) {}

  LiDARPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &plane_unit_norm,
						               double d, double timestamp)
      : LiDARPlaneMeasurement(lidar, curr_point, plane_unit_norm, d, timestamp, 5.) {}

  // ~LiDARPlaneMeasurement() {std::cout << "decontruct LiDARPlaneMeasurement\n\n";}

	template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 1, 1> calculatePoint2Plane(const type::Trajectory<TrajectoryModel, T>& trajectory,
                                            const type::LiDAR<LiDARModel, T>& lidar) const {
    int flags = trajectories::EvaluationFlags::EvalPosition | trajectories::EvaluationFlags::EvalOrientation;

    auto T_IktoG = trajectory.Evaluate(T(timestamp_), flags);

		// 将当前激光系的点云转换到世界系
    Eigen::Matrix<T, 3, 1> p_Lk = curr_point_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_G = T_IktoG->orientation * p_Lk + T_IktoG->position;

		Eigen::Matrix<T, 3, 1> norm = plane_unit_norm_.cast<T>();

		Eigen::Matrix<T, 1, 1> error(norm.dot(p_G) + T(d_));

    return error;
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 1, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    Eigen::Matrix<T, 1, 1> dist = calculatePoint2Plane<TrajectoryModel, T>(trajectory, lidar);
    return T(weight_) * (dist);
  }


std::shared_ptr<LiDARModel> lidar_;
Eigen::Vector3d curr_point_, plane_unit_norm_;
double d_, timestamp_, weight_;
static int index;
int selfIndex;


protected:

	template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDARPlaneMeasurement<LiDARModel> &m) : measurement(m){}

    /// 每次计算residual,都会根据优化后的参数重新构造traj和lidar sensor
    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      size_t offset = 0;
      auto trajectory = entity::Map<TrajectoryModel, T>(&params[offset], trajectory_meta);
      // std::cout << "in operate() index = " << measurement.selfIndex << std::endl;
      // std::cout << "in operate() r3 mintime: " << trajectory.r3_view_->MinTime()
      //           << "  so3 mintime: " << trajectory.so3_view_->MinTime() << std::endl;
      // std::cout << "in operate() r3 maxtime: " << trajectory.r3_view_->MaxTime()
      //           << "  so3 maxtime: " << trajectory.so3_view_->MaxTime() << std::endl;
      
      // std::cout << " timestamp_ = " << measurement.timestamp_ << "\n\n";
      offset += trajectory_meta.NumParameters();
      auto lidar = entity::Map<LiDARModel, T>(&params[offset], lidar_meta);

      Eigen::Map<Eigen::Matrix<T,1,1>> r(residual);
      r = measurement.Error<TrajectoryModel, T>(trajectory, lidar);
      return true;
    }

    const LiDARPlaneMeasurement &measurement;
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
      // std::cout << "index = " << selfIndex << std::endl;
      // std::cout << "residual->trajectory_meta.r3_meta.MinTime = " << residual->trajectory_meta.r3_meta.MinTime()
      //           << "  MaxTime = " << residual->trajectory_meta.r3_meta.MaxTime() << std::endl;

      // std::cout << "residual->trajectory_meta.so3_meta.MinTime = " << residual->trajectory_meta.so3_meta.MinTime()
      //           << "  MaxTime = " << residual->trajectory_meta.so3_meta.MaxTime() << std::endl;
      
      // std::cout << "r3 NumParameters = " << residual->trajectory_meta.r3_meta.NumParameters()
      //           << "  so3 NumParameters = " << residual->trajectory_meta.so3_meta.NumParameters() << "\n";
      // std:: cout << " timestamp_ = " << timestamp_ << "\n\n";

      /// A.2 再将lidar传感器的参数添加到parameters中 (relative pose and timeoffset and so on ..(if have))
      lidar_->AddToProblem(estimator.problem(), {{tmin, tmax}},
                           residual->lidar_meta, parameters);

      /// B.1 先往cost_function中添加待优化参数
      // Add parameters to cost function
      for (auto& pi : parameters) {
        cost_function->AddParameterBlock(pi.size);
      }
      // Add measurement info
      cost_function->SetNumResiduals(1);

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

#endif // LIDAR_PLANE_MEASUREMENT_H