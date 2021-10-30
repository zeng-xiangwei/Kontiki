#ifndef LIDAR_POINT2PLANE_MEASUREMENT_H
#define LIDAR_POINT2PLANE_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>
#include "../sensors/lidar.h"
#include "../trajectories/trajectory.h"
#include "../trajectory_estimator.h"
#include "ceres/rotation.h"

namespace kontiki {
namespace measurements {

template<typename LiDARModel>
class LiDARPoint2PlaneMeasurement {
public:
	LiDARPoint2PlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
                       Eigen::Vector3d &last_point_b, Eigen::Vector3d &last_point_c, double t_cur, double t_a, double t_b,
					             double t_c, double huber_loss, double weight)
			: lidar_(lidar), curr_point_(curr_point), last_point_a_(last_point_a), last_point_b_(last_point_b), last_point_c_(last_point_c),
        t_cur_(t_cur), t_a_(t_a), t_b_(t_b), t_c_(t_c), loss_function_(huber_loss), weight_(weight) {}

  LiDARPoint2PlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
					             Eigen::Vector3d &last_point_b, Eigen::Vector3d &last_point_c, double t_cur, double t_a, double t_b,
					             double t_c, double huber_loss)
      : LiDARPoint2PlaneMeasurement(lidar, curr_point, last_point_a, last_point_b, last_point_c, t_cur, t_a, t_b, t_c, huber_loss, 1.0) {}

  LiDARPoint2PlaneMeasurement(std::shared_ptr<LiDARModel> lidar, Eigen::Vector3d &curr_point, Eigen::Vector3d &last_point_a,
					             Eigen::Vector3d &last_point_b, Eigen::Vector3d &last_point_c, double t_cur, double t_a, double t_b,
					             double t_c)
      : LiDARPoint2PlaneMeasurement(lidar, curr_point, last_point_a, last_point_b, last_point_c, t_cur, t_a, t_b, t_c, 5.) {}

	template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 1, 1> calculatePoint2Point(const type::Trajectory<TrajectoryModel, T>& trajectory,
                                            const type::LiDAR<LiDARModel, T>& lidar) const {
    int flags = trajectories::EvaluationFlags::EvalPosition;
    auto pos_cur = trajectory.Evaluate(T(t_cur_) + lidar.time_offset(), flags);
    auto pos_a = trajectory.Evaluate(T(t_a_) + lidar.time_offset(), flags);
    auto pos_b = trajectory.Evaluate(T(t_b_) + lidar.time_offset(), flags);
    auto pos_c = trajectory.Evaluate(T(t_c_) + lidar.time_offset(), flags);
    T angle_cur = -pos_cur->position.x();
    T angle_a = -pos_a->position.x();
    T angle_b = -pos_b->position.x();
    T angle_c = -pos_c->position.x();
    Eigen::Matrix<T, 3, 1> trans = lidar.relative_position();
    Eigen::Quaternion<T> orien = lidar.relative_orientation();

    Eigen::Matrix<T, 3, 1> p_cur = curr_point_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_a = last_point_a_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_b = last_point_b_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_c = last_point_c_.cast<T>();

    Eigen::Matrix<T, 3, 3> orien2init_cur, orien2init_a, orien2init_b, orien2init_c;
    orien2init_cur << ceres::cos(angle_cur), -ceres::sin(angle_cur), T(0),
                      ceres::sin(angle_cur),  ceres::cos(angle_cur), T(0),
                      T(0),        T(0),       T(1);

    orien2init_a << ceres::cos(angle_a), -ceres::sin(angle_a), T(0),
                    ceres::sin(angle_a),  ceres::cos(angle_a), T(0),
                    T(0),        T(0),       T(1);

    orien2init_b << ceres::cos(angle_b), -ceres::sin(angle_b), T(0),
                    ceres::sin(angle_b),  ceres::cos(angle_b), T(0),
                    T(0),        T(0),       T(1);
    
    orien2init_c << ceres::cos(angle_c), -ceres::sin(angle_c), T(0),
                    ceres::sin(angle_c),  ceres::cos(angle_c), T(0),
                    T(0),        T(0),       T(1);

    p_cur = orien * p_cur + trans;
    p_cur = orien2init_cur * p_cur;

    p_a = orien * p_a + trans;
    p_a = orien2init_a * p_a;

    p_b = orien * p_b + trans;
    p_b = orien2init_b * p_b;

    p_c = orien * p_c + trans;
    p_c = orien2init_c * p_c;

    p_b = p_a - p_b;
    p_c = p_a - p_c;
    p_cur = p_cur - p_a;
    p_b = p_b.cross(p_c);
    p_b.normalize();

    T dist = ceres::DotProduct(p_b.data(), p_cur.data());
    // / ceres::sqrt(p_b.x() * p_b.x() + p_b.y() * p_b.y() + p_b.z() * p_b.z())
		Eigen::Matrix<T, 1, 1> error(dist);

    return error;
  }

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 1, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory,
                               const type::LiDAR<LiDARModel, T> &lidar) const {
    Eigen::Matrix<T, 1, 1> dist = calculatePoint2Point<TrajectoryModel, T>(trajectory, lidar);
    return T(weight_) * (dist);
  }


std::shared_ptr<LiDARModel> lidar_;
Eigen::Vector3d curr_point_, last_point_a_, last_point_b_, last_point_c_;
double t_cur_, t_a_, t_b_, t_c_, weight_;


protected:

	template<typename TrajectoryModel>
  struct Residual {
    Residual(const LiDARPoint2PlaneMeasurement<LiDARModel> &m) : measurement(m){}

    /// 每次计算residual,都会根据优化后的参数重新构造traj和lidar sensor
    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      size_t offset = 0;
      auto trajectory = entity::Map<TrajectoryModel, T>(&params[offset], trajectory_meta);

      offset += trajectory_meta.NumParameters();
      auto lidar = entity::Map<LiDARModel, T>(&params[offset], lidar_meta);

      Eigen::Map<Eigen::Matrix<T,1,1>> r(residual);
      r = measurement.Error<TrajectoryModel, T>(trajectory, lidar);
      return true;
    }

    const LiDARPoint2PlaneMeasurement &measurement;
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
      double tmin_cur, tmax_cur, tmin_a, tmax_a, tmin_b, tmax_b, tmin_c, tmax_c;
      if(this->lidar_->TimeOffsetIsLocked()) {
          tmin_cur = t_cur_;
          tmax_cur = t_cur_;

          tmin_a = t_a_;
          tmax_a = t_a_;

          tmin_b = t_b_;
          tmax_b = t_b_;

          tmin_c = t_c_;
          tmax_c = t_c_;
      }
      else {
          tmin_cur = t_cur_ - this->lidar_->max_time_offset();
          tmax_cur = t_cur_ + this->lidar_->max_time_offset();
          
          tmin_a = t_a_ - this->lidar_->max_time_offset();
          tmax_a = t_a_ + this->lidar_->max_time_offset();

          tmin_b = t_b_ - this->lidar_->max_time_offset();
          tmax_b = t_b_ + this->lidar_->max_time_offset();

          tmin_c = t_c_ - this->lidar_->max_time_offset();
          tmax_c = t_c_ + this->lidar_->max_time_offset();
      }

      /// A.1 先将轨迹参数添加到parameters中 (control points)
      estimator.AddTrajectoryForTimes({
                                        {tmin_a, tmax_a},
                                        {tmin_b, tmax_b},
                                        {tmin_c, tmax_c},
                                        {tmin_cur, tmax_cur}
                                      },
                                      residual->trajectory_meta,
                                      parameters);

      /// A.2 再将lidar传感器的参数添加到parameters中 (relative pose and timeoffset and so on ..(if have))
      lidar_->AddToProblem(estimator.problem(),
                           {
                              {tmin_a, tmax_a},
                              {tmin_b, tmax_b},
                              {tmin_c, tmax_c},
                              {tmin_cur, tmax_cur}
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

#endif // LIDAR_POINT2PLANE_MEASUREMENT_H