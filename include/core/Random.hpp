#pragma once

#include <cstdint>
#include <random>

namespace mlengine::core {

inline std::mt19937& rng() {
  static std::mt19937 engine{std::random_device{}()};
  return engine;
}

inline void set_seed(std::uint32_t seed) { rng().seed(seed); }

}  // namespace mlengine::core