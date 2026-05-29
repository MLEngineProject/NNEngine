#pragma once

#include <cstdint>
#include <random>

namespace mlengine::core {

/**
 * @brief Return the process-wide Mersenne Twister used for initialization.
 */
inline std::mt19937& rng() {
  static std::mt19937 engine{std::random_device{}()};
  return engine;
}

/**
 * @brief Seed the shared RNG to make initialization and shuffling repeatable.
 * @param seed Seed value to apply to the global generator.
 */
inline void set_seed(std::uint32_t seed) { rng().seed(seed); }

}  // namespace mlengine::core