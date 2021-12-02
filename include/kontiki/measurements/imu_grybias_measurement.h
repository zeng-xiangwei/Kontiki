#ifndef KONTIKIV2_IMU_GRYBIAS_MEASUREMENT_H
#define KONTIKIV2_IMU_GRYBIAS_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>

#include "../trajectories/trajectory.h"
#include "../sensors/imu.h"
#include "../trajectory_estimator.h"

namespace kontiki {
namespace measurements {
using ImuModel = kontiki::sensors::ConstantBiasImu;

class IMUGrysBiasMeasurement {
  using Vector3 = Eigen::Vector3d;
 public:
  IMUGrysBiasMeasurement(std::shared_ptr<ImuModel> imu, const Vector3& bg, double weight) :
    imu_(imu), bg(bg), weight(weight) { };

  Vector3 ErrorRaw() const {
    return bg - imu_->gyroscope_bias();
  }

  // Data
  std::shared_ptr<ImuModel> imu_;

  // Measurement data
  Vector3 bg; // imu gryscope and accelerater bias
  double weight;

 protected:
  struct Residual {
    Residual(const IMUGrysBiasMeasurement &m) : measurement(m) {};

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      Eigen::Matrix<T, 3, 1> bg_last = measurement.bg.cast<T>();
      residual[0] = T(measurement.weight) * (bg_last.x() - params[0][0]);
      residual[1] = T(measurement.weight) * (bg_last.y() - params[0][1]);
      residual[2] = T(measurement.weight) * (bg_last.z() - params[0][2]);
      return true;
    }

    const IMUGrysBiasMeasurement& measurement;
    typename ImuModel::Meta imu_meta;
  }; // Residual;

  template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {
    using ResidualImpl = Residual;
    auto residual = new ResidualImpl(*this);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);

    cost_function->AddParameterBlock(3);

    // Add measurement
    cost_function->SetNumResiduals(3);
    ceres::internal::ResidualBlock *res_id = 
    estimator.problem().AddResidualBlock(cost_function, nullptr, residual->measurement.imu_->gyroscope_bias(0));
    
    estimator.res_ids.push_back(res_id);
  }

  // TrajectoryEstimator must be a friend to access protected members
  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;
};

} // namespace measurement
} // namespace kontiki

#endif //KONTIKIV2_IMU_ACCBIAS_MEASUREMENT_H
