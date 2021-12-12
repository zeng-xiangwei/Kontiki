//
// Created by hannes on 2017-11-29.
//

#ifndef KONTIKIV2_RELATIVE_VELOCITY_MEASUREMENT_H
#define KONTIKIV2_RELATIVE_VELOCITY_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>
#include <kontiki/trajectories/trajectory.h>
#include <kontiki/trajectory_estimator.h>

namespace kontiki {
namespace measurements {

template<typename ImuModel>
class RelativeVelocityMeasurement {
  using Vector3 = Eigen::Vector3d;
 public:
  RelativeVelocityMeasurement(std::shared_ptr<ImuModel> imu, double ti, double tj, const Vector3 &ai, const Vector3 &aj, double weight) : 
  imu_(imu), ti(ti), tj(tj), ai(ai), aj(aj), weight(weight) {};

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Measure(const type::Imu<ImuModel, T> &imu, const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    int flags = trajectories::EvaluationFlags::EvalOrientation | trajectories::EvaluationFlags::EvalVelocity;
    auto result_i = trajectory.Evaluate(T(ti), flags);
    auto result_j = trajectory.Evaluate(T(tj), flags);

    Eigen::Matrix<T, 3, 1> vi = result_i->velocity;
    Eigen::Matrix<T, 3, 1> vj = result_j->velocity;
    Eigen::Quaternion<T> qi = result_i->orientation;
    Eigen::Quaternion<T> qj = result_j->orientation;

    T dt = T(tj) - T(ti);

    Eigen::Matrix<T, 3, 1> acc = (qj * (aj.cast<T>() - imu.template accelerometer_bias()) 
                                + qi * (ai.cast<T>() - imu.template accelerometer_bias())) / T(2) + imu.template refined_gravity();
    // auto r = vj - vi - acc * dt;
    auto r = vj - vi;
    return r;
  };

  template<typename TrajectoryModel>
  Vector3 Measure(const type::Trajectory<TrajectoryModel, double> &trajectory) const {
    return Measure<TrajectoryModel, double>(*imu_, trajectory);
  };

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Imu<ImuModel, T> &imu, const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return T(weight) * Measure<TrajectoryModel, T>(imu, trajectory);
  }
  
  template<typename TrajectoryModel>
  Vector3 Error(const type::Trajectory<TrajectoryModel, double> &trajectory) const {
    return Error<TrajectoryModel, double>(*imu_, trajectory);
  }

  template<typename TrajectoryModel>
  Vector3 ErrorRaw(const type::Trajectory<TrajectoryModel, double> &trajectory) const {
    return Measure<TrajectoryModel, double>(*imu_, trajectory);
  }

  std::shared_ptr<ImuModel> imu_;
  // Measurement data
  double ti, tj; // ti < tj
  Vector3 ai, aj;
  double weight;

 protected:

  // Residual struct for ceres-solver
  template<typename TrajectoryModel>
  struct Residual {
    Residual(const RelativeVelocityMeasurement &m) : measurement(m) {};

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      size_t offset = 0;
      const auto trajectory = entity::Map<TrajectoryModel, T>(&params[offset], trajectory_meta);
      offset += trajectory_meta.NumParameters();
      const auto imu = entity::Map<ImuModel, T>(&params[offset], imu_meta);

      Eigen::Map<Eigen::Matrix<T,3,1>> r(residual);
      r = measurement.Error<TrajectoryModel, T>(imu, trajectory);
      return true;
    }

    const RelativeVelocityMeasurement& measurement;
    typename ImuModel::Meta imu_meta;
    typename TrajectoryModel::Meta trajectory_meta;
  }; // Residual;

  template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {
    using ResidualImpl = Residual<TrajectoryModel>;
    auto residual = new ResidualImpl(*this);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);
    std::vector<entity::ParameterInfo<double>> parameter_info;

    // Add trajectory to problem
    double tmin_i, tmax_i, tmin_j, tmax_j;
    if (this->imu_->TimeOffsetIsLocked()) {
      tmin_i = ti;
      tmax_i = ti;
      tmin_j = tj;
      tmax_j = tj;
    }
    else {
      tmin_i = ti - this->imu_->max_time_offset();
      tmax_i = ti + this->imu_->max_time_offset();
      tmin_j = tj - this->imu_->max_time_offset();
      tmax_j = tj + this->imu_->max_time_offset();
    }

    estimator.AddTrajectoryForTimes({{tmin_i, tmax_i}, {tmin_j, tmax_j}}, residual->trajectory_meta, parameter_info);

    // Add IMU to problem
    imu_->AddToProblem(estimator.problem(), {{tmin_i, tmax_i}, {tmin_j, tmax_j}}, residual->imu_meta, parameter_info);

    // Let cost function know about the number and sizes of parameters dynamically added
    for (auto& pi : parameter_info) {
      cost_function->AddParameterBlock(pi.size);
    }

    // Add measurement
    cost_function->SetNumResiduals(3);
    ceres::internal::ResidualBlock *res_id = 
    estimator.problem().AddResidualBlock(cost_function, nullptr, entity::ParameterInfo<double>::ToParameterBlocks(parameter_info));

    estimator.res_ids.push_back(res_id);
  }

  // TrajectoryEstimator must be a friend to access protected members
  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;
};

} // namespace measurements
} // namespace kontiki


#endif //KONTIKIV2_RELATIVE_VELOCITY_MEASUREMENT_H
