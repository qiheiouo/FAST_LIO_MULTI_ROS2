#ifndef FAST_LIO_POSE_COVARIANCE_HPP
#define FAST_LIO_POSE_COVARIANCE_HPP

#include <array>
#include <Eigen/Core>

namespace fast_lio
{
// Native state error: [dp_world, dphi_body], with p' = p + dp_world
// and R_WB' = R_WB * Exp(dphi_body). ROS output uses additive position
// and infinitesimal fixed-parent-axis rotation, ordered xyz then XYZ.
// Both rotations refer to the same current pose and selected parent basis.
// A zero-start transform is held fixed here: this re-expresses a marginal P,
// not covariance of a relative pose against an uncertain startup state.
inline std::array<double, 36> native_pose_covariance(
    const Eigen::Matrix<double, 6, 6> &native_covariance,
    const Eigen::Matrix3d &parent_from_world_rotation,
    const Eigen::Matrix3d &parent_from_body_rotation)
{
    Eigen::Matrix<double, 6, 6> jacobian = Eigen::Matrix<double, 6, 6>::Zero();
    jacobian.block<3, 3>(0, 0) = parent_from_world_rotation;
    jacobian.block<3, 3>(3, 3) = parent_from_body_rotation;
    const Eigen::Matrix<double, 6, 6> covariance =
        jacobian * native_covariance * jacobian.transpose();
    std::array<double, 36> output;
    for (int row = 0; row < 6; ++row)
        for (int column = 0; column < 6; ++column)
            output[row * 6 + column] = covariance(row, column);
    return output;
}
} // namespace fast_lio

#endif
