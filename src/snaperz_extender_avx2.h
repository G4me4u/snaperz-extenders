#pragma once

#if __AVX2__ || 1
// The detailed guide on instructions in the AVX2 (and other) instruction
// set can be found on the Intel reference:
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
#include <immintrin.h>
#include <type_traits>
#include <cassert>

#include "constants.h"

static_assert(std::is_same<len_t, uint8_t>::value,
              "Extender length must fit into an 8-bit uint");

constexpr size_t kElemCount = sizeof(__m256i) / sizeof(len_t);
static_assert(kLength + 1 >= 2 * kElemCount,
              "Extender must have length at least 63");

struct SnaperzExtender
{
    len_t* segments;
    // The odd and even active windows of the segments that are currently
    // being simulated.
    __m256i _windows[2];
    // Counters keeping track of how many blocks we have seen at that index
    // of the window. This is used to check if we are in the last segment.
    __m256i _counter;
    // A cached value of the _last_seg_mask value used during simulation of
    // the pulses.
    __m256i _last_seg_mask;
    // The parity bit defining which window is active
    uint32_t parity_bit = 0b0;
    // The position of the sequence which is first in the active window.
    size_t p;
    // The total number of pulses that have been simulated.
    uint64_t steps;
};

inline void _reverse(const __m256i& _value, __m256i& _dst)
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

inline void _right_shift(const __m256i& _value, __m256i& _dst)
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
     _reverse(_value, _rev);
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

inline void _simulate_step(SnaperzExtender& extender)
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
    // Store the result in the extender segments, so we can use it the next
    // time the window passes this value (since it will be gone after the
    // right shift below). Only do this once we have saturated the windows.
    if (extender.steps > 2 * kElemCount)
    {
        // Compute the sequence index of the first element in the window.
        const auto i = (extender.p + (kLength - 2 * kElemCount + 1)) % (kLength + 1);
        extender.segments[i] = static_cast<len_t>(_mm256_cvtsi256_si32(_next));
    }
    // Shift the next segment one to the right. This will have the effect
    // of actually making it the next segment (it is the previous segment
    // at the start of this iteration).
    _right_shift(_next, _next);
    // Insert the next segment (after the last current element) into the
    // window, as the last element.
    const auto next_length = extender.segments[extender.p];
    _next = _mm256_insert_epi8(_next, next_length, kElemCount - 1);

    // Figure out if we are in the last segment.
    __m256i& _counter = extender._counter;
    __m256i& _last_seg_mask = extender._last_seg_mask;
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

    extender.p = (extender.p + 1) % (kLength + 1);
    extender.steps++;
}

SnaperzExtender create_snaperz_extender()
{
    SnaperzExtender extender;
    // Note: the lengths require an alignment of 
    extender.segments = new len_t[kLength + 1];
    for (uint32_t i = 0; i < kLength + 1; i++)
    {
        extender.segments[i] = 1;
    }
    // Reset the two windows, and the parity bit:
    for (uint32_t i = 0; i < 2; i++)
    {
        extender._windows[i] = _mm256_setzero_si256();
    }
    extender._counter = _mm256_setzero_si256();
    extender._last_seg_mask = _mm256_setzero_si256();
    extender.parity_bit = 0b0;
    extender.p = 0;
    extender.steps = 0;
    return std::move(extender);
}

void destroy_snaperz_extender(SnaperzExtender& extender)
{
    delete[] extender.segments;
    extender.segments = nullptr;
}

void snaperz_extender_simulate_pulse(SnaperzExtender& extender)
{
    // Make sure that we fit another pulse in the currently active window.
    // Otherwise, simulate until we have finished the current pulses (or
    // at least the oldest one of the ones in the active window).
    while (extender.p >= 2 * kElemCount)
    {
        // Simulate the rest of the extender.
        _simulate_step(extender);
    }
    // Actually simulate the next pulse.
    _simulate_step(extender);
    _simulate_step(extender);
}

bool snaperz_extender_equal(const SnaperzExtender& lhs, const SnaperzExtender& rhs)
{
    // TODO: actually perform check if extenders are equal
    return false;
}

bool snaperz_extender_finished(const SnaperzExtender& extender)
{
    // Compute the index of the first segment in the currently active
    // window.
    assert(1 <= extender.p && extender.p <= 2 * kElemCount);
    const uint32_t first_seg_index = (2 * kElemCount - extender.p) / 2;
    // We are done once the first segment is also the last segment.
    return _mm256_movemask_epi8(extender._last_seg_mask) & (1 << first_seg_index);
}
#else // __AVX2__
// There is a bug in snaperz_extender.h if this happens.
#error "Requires AVX2 support."
#endif // !__AVX2__
