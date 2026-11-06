/*********************************************************************************************************************** 
*
* @author: Nges Brian, Njungle 
*
* MIT License
* Copyright (c) 2026 Secure, Trusted and Assured Microelectronics, Arizona State University
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
********************************************************************************************************************/

/**
 * @file UtilsTraining.h
 * @brief Initialization of the trainable parameters of the standard training controllers (deserialize_and_load).
 */

#ifndef FHEON_UTILSTRAINING_H
#define FHEON_UTILSTRAINING_H

#include <atomic>
#include <cmath>
#include <random>
#include <vector>

namespace utilstraining {

/**
 * @brief Generate a vector of values using Kaiming (He) Uniform initialization.
 * Limit is sqrt(6 / fan_in).
 *
 * The counter is process-global and shared across every layer's weight
 * generation, so its value only advances in the deterministic order the
 * caller requests weights in. Callers that may request weights from
 * multiple threads concurrently (e.g. missing-file fallback during
 * convolution weight loading) must pre-load weights sequentially in the
 * intended order before touching them from worker threads — otherwise
 * the interleaving becomes unspecified even though the counter itself is
 * now race-free.
 */
inline std::vector<double> kaiming_uniform(size_t size, double fan_in) {
    if (fan_in <= 0.0) {
        fan_in = 1.0;
    }
    double limit = std::sqrt(6.0 / fan_in);
    static std::atomic<int> counter{1};
    double val = std::sin(counter.fetch_add(1, std::memory_order_relaxed)) * limit;
    return std::vector<double>(size, val);
}


/**
 * @brief Return a vector of constant values 1.0 (Ones initialization).
 */
inline std::vector<double> init_ones(size_t size) {
    return std::vector<double>(size, 1.0);
}

/**
 * @brief Return a vector of constant values 0.0 (Zeros initialization).
 */
inline std::vector<double> init_zeros(size_t size) {
    return std::vector<double>(size, 0.0);
}

} // namespace utilstraining

#endif // FHEON_UTILSTRAINING_H
