//
// Created by hannes on 2017-11-29.
//

#ifndef KONTIKIV2_velocity_MEASUREMENT_H
#define KONTIKIV2_velocity_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>
#include <kontiki/trajectories/trajectory.h>
#include <kontiki/trajectory_estimator.h>

namespace kontiki {
namespace measurements {

class VelocityMeasurement {
  using Vector3 = Eigen::Vector3d;
 public:
  VelocityMeasurement(double t, const Vector3 &v, double weight) : t(t), v(v), weight(weight) {};

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Measure(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return trajectory.Velocity(T(t));
  };

  template<typename TrajectoryModel, typename T>
  Eigen::Matrix<T, 3, 1> Error(const type::Trajectory<TrajectoryModel, T> &trajectory) const {
    return T(weight) * (v.cast<T>() - Measure<TrajectoryModel, T>(trajectory));
  }

  // Measurement data
  double t;
  Vector3 v;
  double weight;

 protected:

  // Residual struct for ceres-solver
  template<typename TrajectoryModel>
  struct Residual {
    Residual(const VelocityMeasurement &m) : measurement(m) {};

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      auto trajectory = entity::Map<TrajectoryModel, T>(params, meta);
      Eigen::Map<Eigen::Matrix<T,3,1>> r(residual);
      r = measurement.Error<TrajectoryModel, T>(trajectory);
      return true;
    }

    const VelocityMeasurement& measurement;
    typename TrajectoryModel::Meta meta;
  }; // Residual;

  template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {
    using ResidualImpl = Residual<TrajectoryModel>;
    auto residual = new ResidualImpl(*this);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);
    std::vector<entity::ParameterInfo<double>> parameter_info;

    // Add trajectory to problem
    //estimator.trajectory()->AddToProblem(estimator.problem(), residual->meta, parameter_blocks, parameter_sizes);
    estimator.AddTrajectoryForTimes({{t,t}}, residual->meta, parameter_info);
    for (auto& pi : parameter_info) {
      cost_function->AddParameterBlock(pi.size);
    }

    // Add measurement
    cost_function->SetNumResiduals(3);
    // If we had any measurement parameters to set, this would be the place

    // Give residual block to estimator problem
    estimator.problem().AddResidualBlock(cost_function,
                                         nullptr,
                                         entity::ParameterInfo<double>::ToParameterBlocks(parameter_info));
  }

  // TrajectoryEstimator must be a friend to access protected members
  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;
};

} // namespace measurements
} // namespace kontiki


#endif //KONTIKIV2_velocity_MEASUREMENT_H
