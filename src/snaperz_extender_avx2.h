#pragma once

#if __AVX2__
// The detailed guide on instructions in the AVX2 (and other) instruction
// set can be found on the Intel reference:
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
#include <immintrin.h>
#include <type_traits>
#include <cstring>
#include <cassert>

#include "constants.h"

namespace snaperz
{
  // Hard limitation, since we only have implementations for <=16-bit elements.
  static_assert(std::numeric_limits<len_t>::max() <= std::numeric_limits<uint16_t>::max(),
                "Extender length must fit into a 16-bit uint");

  template<typename T>
  static constexpr T to_even(T value)
  {
    return value + (value & 0x1);
  }

  static constexpr uint32_t kElemCount = sizeof(__m256i) / sizeof(len_t);
  static constexpr uint32_t kSegCount =
    (kLength + 1 > 2 * kElemCount) ? kLength + 1 : to_even(kLength + 1);
  static constexpr uint32_t kSaturationCount =
    std::min(kSegCount, 2 * kElemCount);
  
  struct Extender
  {
    len_t* segments;
    // The odd and even active windows of the segments that are currently
    // being simulated.
    __m256i _windows[2];
    // Counters keeping track of how many blocks we have seen at that index
    // of the window. This is used to check if we are in the last segment.
    __m256i _counter;
    // A cached value of the _last_seg_mask values used during simulation of
    // the pulses.
    __m256i _last_seg_masks[2];
    // The parity bit defining which window is active
    uint32_t parity_bit = 0b0;
    // The position of the sequence which is first in the active window.
    size_t p;
    // The total number of pulses that have been simulated.
    uint64_t steps;
  };

  namespace avx2
  {
    template<typename T>
    void _reverse(const __m256i& _value, __m256i& _dst);
    
    template<typename T>
    void _right_shift(const __m256i& _value, __m256i& _dst);
    
    template<typename T>
    void _simulate_step(Extender& extender);

    template<typename T>
    inline bool _finished(const Extender& extender);

    template<typename T>
    inline bool _equals(const Extender& lhs, const Extender& rhs);

    // Note: debugging perposes only!
    template<class T>
    inline void TEST_log(const __m256i & value)
    {
      const size_t n = sizeof(__m256i) / sizeof(T);
      T buffer[n];
      _mm256_storeu_si256((__m256i*)buffer, value);
      for (uint32_t i = n; i-- != 0; )
      {
        std::cout << +buffer[i] << " ";
      }
      std::cout << std::endl;
    }

    /* uint8_t implementation for AVX2 */

    template<>
    inline void _reverse<uint8_t>(const __m256i& _value, __m256i& _dst)
    {
      // Reverse bytes in value, i.e. compute:
      //   V'[i] = V'[n - i - 1], forall n < i <= 0
      //
      // This operation is done with two instructions. The first instruction
      // will reverse within the 128-bit lanes individually by using shuffle,
      // and the other instruction will then permute 128-bit langes such that
      // they are swapped. Below is a demonstration of this procedure:
      //
      //   V:
      //     V[7], V[6], V[5], V[4], V[3], V[2], V[1], V[0].
      //
      //   Shuffle(V):
      //     V[4], V[5], V[6], V[7], V[0], V[1], V[2], V[3].
      //   Permute(Shuffle(V)):
      //      V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7].

      // Perform shuffle instruction
      const __m256i _shuffle_control = _mm256_set_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, // 1st 128-bit lane
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15  // 2nd 128-bit lane
      );
      __m256i _tmp = _mm256_shuffle_epi8(_value, _shuffle_control);
      // Perform the permutation, swapping the 128-bit lanes. The lower 4 bits
      // of the control determine the lower half of the result, and upper 4 bits
      // determine the upper half. In this case, we just select the 1st (upper
      // half of first argument), and 0th (lower half of first argument) as the
      // respective results.
      _dst = _mm256_permute2x128_si256(_tmp, _tmp, 0x01);
    }

    template<>
    inline void _right_shift<uint8_t>(const __m256i& _value, __m256i& _dst)
    {
      // Shift the value right by 1 byte, i.e. compute:
      //   V' = V >> 8
      //     <==>
      //   V'[i] = V[i + 1], forall n > i > 0
      //   V'[n - 1] = 0
      //
      // This will be done in two operations. One operation will perform a
      // logical shift-right on V. Since this operation is only performed
      // within the 128-bit lanes, we will have some values that are zeroed
      // out. In particular, V[n/2 - 1] and V[n - 1] will both be zeroed out.
      // Therefore, we also need to restore V[n/2 - 1]. This can be done in
      // several ways. We will restore it by reversing V, and transferring
      // V[n/2] to V[n/2 - 1] by blending. Below is a demonstration:
      //
      //   V:
      //     V[7], V[6], V[5], V[4], V[3], V[2], V[1], V[0].
      //
      //   RightShift(V, 1):
      //        0, V[7], V[6], V[5],    0, V[3], V[2], V[1].
      //   Reverse(V):
      //     V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7].
      //   Blend(RightShift(V, 1), Reverse(V)):
      //        0, V[7], V[6], V[5], V[4], V[3], V[2], V[1].
      //
      // We can also perform a right rotation this way by blending
      // back in V[0] as the most significant element.
      
      // Perform the right shift on _value first. This should allow the below
      // reverse operation to be performed in parallel.
      __m256i _tmp = _mm256_srli_si256(_value, 1);
      // Reverse the value. This is done in multiple steps. Refer to _reverse
      // for more info.
      __m256i _rev;
      _reverse<uint8_t>(_value, _rev);
      // Blend the value with index 15 from the reverse into the value, i.e.
      // only the 15th value should have the 7th bit set. Just use 0xFF, since
      // the remaining bits are ignored.
      const __m256i _mask = _mm256_set_epi8(
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
      );
      _dst = _mm256_blendv_epi8(_tmp, _rev, _mask);
    }

    template<>
    inline void _simulate_step<uint8_t>(Extender& extender)
    {
      // Constants
      const __m256i _zeros = _mm256_setzero_si256();
      const __m256i _ones = _mm256_set1_epi8(1);
      
      const __m256i _push_limit = _mm256_set1_epi8(kPushLimit);
      const __m256i _last_push_limit = _mm256_set1_epi8(kLastPushLimit);
      const __m256i _len_plus_one = _mm256_set1_epi8(kLength + 1);

      // Compute reference to current (C) and next (N) segment(s). Also flip
      // the parity bit to prepare for next iteration.
      __m256i& _curr = extender._windows[extender.parity_bit];
      __m256i& _next = extender._windows[extender.parity_bit ^= 0b1];
      __m256i& _last_seg_mask = extender._last_seg_masks[extender.parity_bit];
      // Store the result in the extender segments, so we can use it the next
      // time the window passes this value (since it will be gone after the
      // right shift below). Only do this once we have saturated the windows.
      if (extender.steps >= kSaturationCount)
      {
        // Compute the sequence index of the first element in the window.
        const auto i = (extender.p + (kSegCount - kSaturationCount)) % kSegCount;
        extender.segments[i] = static_cast<uint8_t>(_mm256_cvtsi256_si32(_next));
      }
      // Shift the next segment one to the right. This will have the effect
      // of actually making it the next segment (it is the previous segment
      // at the start of this iteration).
      _right_shift<uint8_t>(_next, _next);
      // Insert the next segment (after the last current element) into the
      // window, as the last element.
      const auto next_length = extender.segments[extender.p];
      _next = _mm256_insert_epi8(_next, next_length, kSaturationCount / 2 - 1);

      // Figure out if we are in the last segment.
      __m256i& _counter = extender._counter;
      // Increase the counter by the number of blocks in the current segment
      _counter = _mm256_add_epi8(_counter, _curr);
      // Check if the counter is kLength + 1, i.e. we are the last segment
      _last_seg_mask = _mm256_cmpeq_epi8(_counter, _len_plus_one);

      // Handle pushing case:

      // Compute: C' = C - 1, i.e. C'[i] = C[i] - 1, forall i.
      __m256i _curr_minus_one = _mm256_sub_epi8(_curr, _ones);
      // Compute push limit based on whether the current one is the last segment
      // or not. In the case where we are the last segment, the virtual push limit
      // no longer applies directly, and we can actually push an extra block.
      //     _curr_push_limit = _last_segment_mask ? _last_push_limit : _push_limit
      __m256i _curr_push_limit = _mm256_blendv_epi8(_push_limit, _last_push_limit, _last_seg_mask);
      // Compute: PD = min(push_limit, C - 1).
      __m256i _push_delta = _mm256_min_epu8(_curr_push_limit, _curr_minus_one);
      
      // Mask out the push delta for every case that equals 1.
      __m256i _equal_one_mask = _mm256_cmpeq_epi8(_curr, _ones);
      // Compute: _push_delta = _push_delta & !(_equal_one_mask)
      _push_delta = _mm256_andnot_si256(_equal_one_mask, _push_delta);
      // Mask out the push delta for every case that equals 0. This has the
      // effect that the pushing only applies for segment lengths greater
      // than 1.
      __m256i _equal_zero_mask = _mm256_cmpeq_epi8(_curr, _zeros);
      // Compute: _push_delta = _push_delta & !(_equal_zero_mask)
      _push_delta = _mm256_andnot_si256(_equal_zero_mask, _push_delta);

      // Handle pulling case:
      
      // We simply pull everything from the next segment, unless it is the last
      // segment, in which case we have to pull nothing.
      __m256i _pull_delta = _mm256_andnot_si256(_last_seg_mask, _next);
      // Mask out the pull delta for every case that is not equal to 1.
      // Compute: _pull_delta = _pull_delta & _equal_one_mask
      _pull_delta = _mm256_and_si256(_equal_one_mask, _pull_delta);

      // Compute the total delta to add to the current segments, and subtract
      // from the next segments. This is simply the push delta subtracted from
      // the pull delta.
      // Compute: D = _pull_delta - _push_delta
      __m256i _delta = _mm256_sub_epi8(_pull_delta, _push_delta);
      // Finally, add and subtract the result from the segments.
      _curr = _mm256_add_epi8(_curr, _delta);
      _next = _mm256_sub_epi8(_next, _delta);

      // Add the blocks that moved to this segment to the counter, and check
      // again if we are the last segment. This generally only occurs when we
      // are pulling, but this also ensures that we keep the counter up-to-date
      // when we consider the next segment (which might now be the last segment).
      _counter = _mm256_add_epi8(_counter, _delta);
      // Check if the counter is kLength + 1, i.e. we are the last segment
      _last_seg_mask = _mm256_cmpeq_epi8(_counter, _len_plus_one);
      // Reset counter if we are still at the last segment. This ensures that
      // it remains zero until we loop back ground to the first sequence, since
      // we will never have any blocks in the following segments (essentially
      // allows for an efficient reset of the counter).
      _counter = _mm256_andnot_si256(_last_seg_mask, _counter);

      extender.p = (extender.p + 1) % kSegCount;
      extender.steps++;
    }

    template<>
    inline bool _finished<uint8_t>(const Extender& extender)
    {
      // Compute the index of the first segment in the currently active window.
      assert(0 <= extender.p && extender.p <= kSaturationCount);
      // Special case where extender.p might wrap to zero, in which case the
      // result should also be zero. This is also relevant if this is called
      // before the extender has simulated the first pulse.
      uint32_t first_seg_index = (extender.p > 0) * (kSaturationCount - extender.p) / 2;
      // Compute the parity, i.e. the window that contains the first segment.
      const uint32_t parity = extender.parity_bit ^ (extender.p & 0x1);
      const __m256i& _last_seg_mask = extender._last_seg_masks[parity];
      // We are done once the first segment is also the last segment.
      return _mm256_movemask_epi8(_last_seg_mask) & (1 << first_seg_index);
    }

    template<>
    inline bool _equals<uint8_t>(const Extender& lhs, const Extender& rhs)
    {
      // The extenders are equal if (1) their currently active pulses are at
      // the same segments, and that the segments (2) inside (in the registers)
      // the active window, and (3) outside the window are equal.
      if (lhs.p != rhs.p)
      {
        // (1) Currently simulating the same pulses
        return false;
      }
      // (2) Active windows are equal
      for (uint32_t i = 0; i < 2; i++)
      {
        __m256i _window_equal = _mm256_cmpeq_epi8(lhs._windows[i], rhs._windows[i]);
        if (~_mm256_movemask_epi8(_window_equal))
        {
          // At least one of the values are not equal (i.e. not 1 before the
          // above negation).
          return false;
        }
      }
      // (3) Segments outside windows are equal
      static constexpr size_t cnt = kSegCount - kSaturationCount;
      if constexpr (cnt != 0)
      {
        assert(0 <= lhs.p && lhs.p <= kSaturationCount);
        const auto lhs_start = lhs.segments + lhs.p;
        const auto rhs_start = rhs.segments + rhs.p;
        return std::memcmp(lhs_start, rhs_start, cnt * sizeof(uint8_t)) == 0;
      }
      return true;
    }
    
    /* uint16_t implementation for AVX2 */

    template<>
    inline void _reverse<uint16_t>(const __m256i& _value, __m256i& _dst)
    {
      // See uint8_t version for implementation details.
      const __m256i _shuffle_control = _mm256_set_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // 1st 128-bit lane
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14  // 2nd 128-bit lane
      );
      __m256i _tmp = _mm256_shuffle_epi8(_value, _shuffle_control);
      _dst = _mm256_permute2x128_si256(_tmp, _tmp, 0x01);
    }

    template<>
    inline void _right_shift<uint16_t>(const __m256i& _value, __m256i& _dst)
    {
      // See uint8_t version for implementation details.
      __m256i _tmp = _mm256_srli_si256(_value, sizeof(uint16_t));
      __m256i _rev;
      _reverse<uint16_t>(_value, _rev);
      const __m256i _mask = _mm256_set_epi8(
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
      );
      _dst = _mm256_blendv_epi8(_tmp, _rev, _mask);
    }

    template<>
    inline void _simulate_step<uint16_t>(Extender& extender)
    {
      // See uint8_t version for implementation details.
      const __m256i _zeros = _mm256_setzero_si256();
      const __m256i _ones = _mm256_set1_epi16(1);
      
      const __m256i _push_limit = _mm256_set1_epi16(kPushLimit);
      const __m256i _last_push_limit = _mm256_set1_epi16(kLastPushLimit);
      const __m256i _len_plus_one = _mm256_set1_epi16(kLength + 1);

      __m256i& _curr = extender._windows[extender.parity_bit];
      __m256i& _next = extender._windows[extender.parity_bit ^= 0b1];
      __m256i& _last_seg_mask = extender._last_seg_masks[extender.parity_bit];

      if (extender.steps >= kSaturationCount)
      {
        const auto i = (extender.p + (kSegCount - kSaturationCount)) % kSegCount;
        extender.segments[i] = static_cast<uint16_t>(_mm256_cvtsi256_si32(_next));
      }
      
      _right_shift<uint16_t>(_next, _next);
      
      const auto next_length = extender.segments[extender.p];
      _next = _mm256_insert_epi16(_next, next_length, kSaturationCount / 2 - 1);
      
      __m256i& _counter = extender._counter;
      _counter = _mm256_add_epi16(_counter, _curr);
      _last_seg_mask = _mm256_cmpeq_epi16(_counter, _len_plus_one);

      // Handle pushing case:

      __m256i _curr_minus_one = _mm256_sub_epi16(_curr, _ones);
      __m256i _curr_push_limit = _mm256_blendv_epi8(_push_limit, _last_push_limit, _last_seg_mask);
      __m256i _push_delta = _mm256_min_epu16(_curr_push_limit, _curr_minus_one);
      
      __m256i _equal_one_mask = _mm256_cmpeq_epi16(_curr, _ones);
      _push_delta = _mm256_andnot_si256(_equal_one_mask, _push_delta);
      __m256i _equal_zero_mask = _mm256_cmpeq_epi16(_curr, _zeros);
      _push_delta = _mm256_andnot_si256(_equal_zero_mask, _push_delta);

      // Handle pulling case:
      
      __m256i _pull_delta = _mm256_andnot_si256(_last_seg_mask, _next);
      _pull_delta = _mm256_and_si256(_equal_one_mask, _pull_delta);

      __m256i _delta = _mm256_sub_epi16(_pull_delta, _push_delta);
      _curr = _mm256_add_epi16(_curr, _delta);
      _next = _mm256_sub_epi16(_next, _delta);

      _counter = _mm256_add_epi16(_counter, _delta);
      _last_seg_mask = _mm256_cmpeq_epi16(_counter, _len_plus_one);
      _counter = _mm256_andnot_si256(_last_seg_mask, _counter);

      extender.p = (extender.p + 1) % kSegCount;
      extender.steps++;
    }

    template<>
    inline bool _finished<uint16_t>(const Extender& extender)
    {
      // See uint8_t version for implementation details.
      assert(0 <= extender.p && extender.p <= kSaturationCount);
      uint32_t first_seg_index = (extender.p > 0) * (kSaturationCount - extender.p) / 2;
      const uint32_t parity = extender.parity_bit ^ (extender.p & 0x1);
      const __m256i& _last_seg_mask = extender._last_seg_masks[parity];
      // Note: should be shifted twice as far over due to 16-bit versus 8-bit.
      return _mm256_movemask_epi8(_last_seg_mask) & (1 << (2 * first_seg_index));
    }

    template<>
    inline bool _equals<uint16_t>(const Extender& lhs, const Extender& rhs)
    {
      // See uint8_t version for implementation details.
      if (lhs.p != rhs.p)
      {
        return false;
      }
      for (uint32_t i = 0; i < 2; i++)
      {
        __m256i _window_equal = _mm256_cmpeq_epi16(lhs._windows[i], rhs._windows[i]);
        if (~_mm256_movemask_epi8(_window_equal))
        {
          return false;
        }
      }
      static constexpr size_t cnt = kSegCount - kSaturationCount;
      if constexpr (cnt != 0)
      {
        assert(0 <= lhs.p && lhs.p <= kSaturationCount);
        const auto lhs_start = lhs.segments + lhs.p;
        const auto rhs_start = rhs.segments + rhs.p;
        return std::memcmp(lhs_start, rhs_start, cnt * sizeof(uint16_t)) == 0;
      }
      return true;
    }
  } // namespace avx2

  Extender create()
  {
    Extender extender;
    extender.segments = new len_t[kSegCount];
    for (uint32_t i = 0; i < kSegCount; i++)
    {
      // Note: there are sometimes trailing segments with zeros.
      extender.segments[i] = (i <= kLength) ? 1 : 0;
    }
    // Reset the two windows, and the parity bit:
    for (uint32_t i = 0; i < 2; i++)
    {
      extender._windows[i] = _mm256_setzero_si256();
      extender._last_seg_masks[i] = _mm256_setzero_si256();
    }
    extender._counter = _mm256_setzero_si256();
    extender.parity_bit = 0b0;
    extender.p = 0;
    extender.steps = 0;
    return std::move(extender);
  }

  void destroy(Extender& extender)
  {
    delete[] extender.segments;
    extender.segments = nullptr;
  }

  void simulate_pulse(Extender& extender)
  {
    // Make sure that we fit another pulse in the currently active window.
    // Otherwise, simulate until we have finished the current pulses (or
    // at least the oldest one of the ones in the active window).
    while (extender.p >= kSaturationCount)
    {
      // Simulate the rest of the extender.
      avx2::_simulate_step<len_t>(extender);
    }
    // Actually simulate the next pulse.
    avx2::_simulate_step<len_t>(extender);
    avx2::_simulate_step<len_t>(extender);
  }

  bool equals(const Extender& lhs, const Extender& rhs)
  {
    return avx2::_equals<len_t>(lhs, rhs);
  }

  bool finished(const Extender& extender)
  {
    return avx2::_finished<len_t>(extender);
  }
} // namespace snaperz
#else // __AVX2__
// There is a bug in snaperz_extender.h if this happens.
#error "Requires AVX2 support."
#endif // !__AVX2__
