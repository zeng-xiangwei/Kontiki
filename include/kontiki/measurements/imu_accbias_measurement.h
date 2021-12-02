#ifndef KONTIKIV2_IMU_ACCBIAS_MEASUREMENT_H
#define KONTIKIV2_IMU_ACCBIAS_MEASUREMENT_H

#include <Eigen/Dense>

#include <iostream>

#include "../trajectories/trajectory.h"
#include "../sensors/imu.h"
#include "../trajectory_estimator.h"

namespace kontiki {
namespace measurements {
using ImuModel = kontiki::sensors::ConstantBiasImu;

class IMUAcceBiasMeasurement {
  using Vector3 = Eigen::Vector3d;
 public:
  IMUAcceBiasMeasurement(std::shared_ptr<ImuModel> imu, const Vector3& ba, double weight) :
    imu_(imu), ba(ba), weight(weight) { };

  Vector3 ErrorRaw() const {
    return ba - imu_->accelerometer_bias();
  }

  // Data
  std::shared_ptr<ImuModel> imu_;

  // Measurement data
  Vector3 ba; // imu gryscope and accelerater bias
  double weight;

 protected:
  struct Residual {
    Residual(const IMUAcceBiasMeasurement &m) : measurement(m) {};

    template <typename T>
    bool operator()(T const* const* params, T* residual) const {
      Eigen::Matrix<T, 3, 1> ba_last = measurement.ba.cast<T>();
      residual[0] = T(measurement.weight) * (ba_last.x() - params[0][0]);
      residual[1] = T(measurement.weight) * (ba_last.y() - params[0][1]);
      residual[2] = T(measurement.weight) * (ba_last.z() - params[0][2]);
      return true;
    }

    const IMUAcceBiasMeasurement& measurement;
    typename ImuModel::Meta imu_meta;
  }; // Residual;

  template<typename TrajectoryModel>
  void AddToEstimator(kontiki::TrajectoryEstimator<TrajectoryModel>& estimator) {
    using ResidualImpl = Residual;
    auto residual = new ResidualImpl(*this);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<ResidualImpl>(residual);

    // Let cost function know about the number and sizes of parameters dynamically added
    cost_function->AddParameterBlock(3);

    // Add measurement
    cost_function->SetNumResiduals(3);
    ceres::internal::ResidualBlock *res_id = 
    estimator.problem().AddResidualBlock(cost_function, nullptr, residual->measurement.imu_->accelerometer_bias(0));
    
    estimator.res_ids.push_back(res_id);
  }

  // TrajectoryEstimator must be a friend to access protected members
  template<template<typename> typename TrajectoryModel>
  friend class kontiki::TrajectoryEstimator;
};

} // namespace measurement
} // namespace kontiki

#endif //KONTIKIV2_IMU_ACCBIAS_MEASUREMENT_H
