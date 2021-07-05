#ifndef LIDAR_PIVOTPOINTPLANE_MEASUREMENT_H
#define LIDAR_PIVOTPOINTPLANE_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>
#include "../sensors/lidar.h"
#include "../trajectories/trajectory.h"
#include "../trajectory_estimator.h"
#include "ceres/rotation.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDARPivotPointPlaneMeasurement {
public:
	// LiDARPivotPointPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector4d &coeff,
	// 	                            	double pivot_time, double timestamp, double huber_loss, double weight)
	// 		: lidar_(lidar), curr_point_(curr_point), coeff_(coeff), pivot_time_(pivot_time), 
	// 			timestamp_(timestamp), loss_function_(huber_loss), weight_(weight) {}

  // LiDARPivotPointPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector4d &coeff,
	// 					          						double pivot_time, double timestamp, double huber_loss)
  //     : LiDARPivotPointPlaneMeasurement(lidar, curr_point, coeff, pivot_time, timestamp, huber_loss, 1.0) {}

  // LiDARPivotPointPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector4d &coeff,
	// 					          						double pivot_time, double timestamp)
  //     : LiDARPivotPointPlaneMeasurement(lidar, curr_point, coeff, pivot_time, timestamp, 5.) {}


  LiDARPivotPointPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector4d &coeff,
		                            	Eigen::Vector3d &pivot_pos, Eigen::Quaterniond &pivot_rot, double timestamp, double huber_loss, double weight)
			: lidar_(lidar), curr_point_(curr_point), coeff_(coeff), pivot_pos_(pivot_pos), pivot_rot_(pivot_rot),
				timestamp_(timestamp), loss_function_(huber_loss), weight_(weight) {}

  LiDARPivotPointPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector4d &coeff,
						          						Eigen::Vector3d &pivot_pos, Eigen::Quaterniond &pivot_rot, double timestamp, double huber_loss)
      : LiDARPivotPointPlaneMeasurement(lidar, curr_point, coeff, pivot_pos, pivot_rot, timestamp, huber_loss, 1.0) {}

  LiDARPivotPointPlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector4d &coeff,
						          						Eigen::Vector3d &pivot_pos, Eigen::Quaterniond &pivot_rot, double timestamp)
      : LiDARPivotPointPlaneMeasurement(lidar, curr_point, coeff, pivot_pos, pivot_rot, timestamp, 5.) {}

  // ~LiDARPivotPointPlaneMeasurement() {std::cout << "decontruct LiDARPivotPointPlaneMeasurement\n\n";}

	template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 1, 1> calculatePoint2Plane(const type::Trajectory<TrajectoryModel, T>& trajectory,
                                            const type::LiDAR<LiDARModel, T>& lidar) const {
    int flags = trajectories::EvaluationFlags::EvalPosition | trajectories::EvaluationFlags::EvalOrientation;

		// 将当前激光系的点云转换到世界系
		// auto T_IptoG = trajectory.Evaluate(T(pivot_time_), flags);
    const Eigen::Quaternion<T> q_IptoG = pivot_rot_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_IpinG = pivot_pos_.cast<T>();
    auto T_IktoG = trajectory.Evaluate(T(timestamp_), flags);

    const Eigen::Matrix<T, 3, 1> p_LinI = lidar.relative_position();
    const Eigen::Quaternion<T> q_LtoI = lidar.relative_orientation();

    Eigen::Matrix<T, 3, 1> p_Lk = curr_point_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_I = q_LtoI * p_Lk + p_LinI;

    // Eigen::Matrix<T, 3, 1> p_temp = T_IptoG->orientation.conjugate()*(T_IktoG->orientation * p_I + T_IktoG->position - T_IptoG->position);
    Eigen::Matrix<T, 3, 1> p_temp = q_IptoG.conjugate()*(T_IktoG->orientation * p_I + T_IktoG->position - p_IpinG);
    Eigen::Matrix<T, 3, 1> p_M = q_LtoI.conjugate() * (p_temp - p_LinI);

		Eigen::Matrix<T, 3, 1> norm = coeff_.cast<T>().head(3);

		Eigen::Matrix<T, 1, 1> error(norm.dot(p_M) + T(coeff_(3, 0)));

    return error;
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 1, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    Eigen::Matrix<T, 1, 1> dist = calculatePoint2Plane<TrajectoryModel, T>(trajectory, lidar);
    return T(weight_) * (dist);
  }


std::shared_ptr<LiDARModel> lidar_;
Eigen::Vector3d curr_point_, pivot_pos_;
Eigen::Vector4d coeff_;
Eigen::Quaterniond pivot_rot_;  // 注意是 pivot 时刻下 imu 的位姿
double pivot_time_, timestamp_, weight_;
static int index;
int selfIndex;


protected:

	template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDARPivotPointPlaneMeasurement<LiDARModel> &m) : measurement(m){}

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

    const LiDARPivotPointPlaneMeasurement &measurement;
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

			// double pivot_min_time, pivot_max_time;
      // if(this->lidar_->TimeOffsetIsLocked()) {
      //     pivot_min_time = pivot_time_;
      //     pivot_max_time = pivot_time_;
      // }
      // else {
      //     pivot_min_time = pivot_time_ - this->lidar_->max_time_offset();
      //     pivot_max_time = pivot_time_ + this->lidar_->max_time_offset();
      // }

      /// A.1 先将轨迹参数添加到parameters中 (control points)
      estimator.AddTrajectoryForTimes({
																				// {pivot_min_time,pivot_max_time},
																				{tmin, tmax}
																			},
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
      lidar_->AddToProblem(estimator.problem(), 
													{
														// {pivot_min_time,pivot_max_time},
														{tmin, tmax}
													},
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

#endif // LIDAR_PIVOTPOINTPLANE_MEASUREMENT_H