#pragma once
#include <Eigen/Core>

namespace mlengine {

/**
 * @brief Canonical scalar type used throughout the engine.
 */
using Scalar = float;
/**
 * @brief Row-major dense matrix type used for all tensor storage.
 */
using MatrixRM =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace mlengine