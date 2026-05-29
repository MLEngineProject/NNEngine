#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/Random.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

class DataLoader {
 private:
  MatrixRM X_;
  MatrixRM y_;
  MatrixRM X_shuffled_;
  MatrixRM y_shuffled_;
  size_t batch_size_;
  bool shuffle_;
  bool drop_last_;
  std::vector<int> indices_;
  size_t current_idx_;

 public:
  DataLoader(const MatrixRM& X, const MatrixRM& y, size_t batch_size,
             bool shuffle = true, bool drop_last = false)
      : X_(X),
        y_(y),
        batch_size_(batch_size),
        shuffle_(shuffle),
        drop_last_(drop_last),
        current_idx_(0) {
    if (X_.rows() != y_.rows()) {
      throw std::runtime_error(
          "Features and Targets must have same row count.");
    }
    indices_.resize(X_.rows());
    std::iota(indices_.begin(), indices_.end(), 0);
    reset();
  }

  void reset() {
    current_idx_ = 0;
    if (shuffle_ && !indices_.empty()) {
      std::shuffle(indices_.begin(), indices_.end(), core::rng());

      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X_.rows());
      perm.indices() = Eigen::Map<Eigen::VectorXi>(indices_.data(), X_.rows());

      X_shuffled_.noalias() = perm * X_;
      y_shuffled_.noalias() = perm * y_;
    } else {
      X_shuffled_ = X_;
      y_shuffled_ = y_;
    }
  }

  bool has_next() const {
    if (drop_last_) {
      return current_idx_ + batch_size_ <= indices_.size();
    }
    return current_idx_ < indices_.size();
  }

  void next_batch(MatrixRM& X_batch, MatrixRM& y_batch) {
    if (!has_next()) return;

    size_t remaining = indices_.size() - current_idx_;
    size_t actual_batch_size = std::min(batch_size_, remaining);

    if (X_batch.rows() != actual_batch_size ||
        X_batch.cols() != X_shuffled_.cols()) {
      X_batch.resize(actual_batch_size, X_shuffled_.cols());
    }
    if (y_batch.rows() != actual_batch_size ||
        y_batch.cols() != y_shuffled_.cols()) {
      y_batch.resize(actual_batch_size, y_shuffled_.cols());
    }

    X_batch.noalias() = X_shuffled_.middleRows(current_idx_, actual_batch_size);
    y_batch.noalias() = y_shuffled_.middleRows(current_idx_, actual_batch_size);

    current_idx_ += actual_batch_size;
  }
};

}  // namespace mlengine::core