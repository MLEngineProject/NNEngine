#pragma once
#include <Eigen/Core>

namespace mlengine {

using Scalar = float;
using MatrixRM =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace mlengine