#pragma once

// To learn more about Snaperz extenders, take a look at this document:
// https://docs.google.com/document/d/1KCM7lk-GBn_-RIhuuUiZdNNiBWDc6Zm7g88cdIFOeQg/edit
//
// A Snaperz extender is an extender that relies on a certain mechanic to extend
// and retract. A Snaperz extender of length L is constructed by placing a row of
// 2L - 1 observers, output facing down, above the extender, with repeaters set to
// 4 ticks of delay (the second setting) placed on top of the observers, with their
// output facing in the same direction as the pistons of the extender. The extender
// is then operated by sending short pulses through this repeater line at regular
// intervals.
//
// The most straightforward algorithm would mimic the game engine, keeping track
// of each piston's position over time, updating that based on which pistons are
// powered at each point in time. However, this is quite inefficient, as each piston
// acts independently, which makes the calculations quite complex. We will introduce
// two new concepts to construct a more efficient algorithm.
//
// First, we represent the extender as a series of piston segments. This allows us
// to simplify the behavior of groups of pistons into a single action. If we allow
// segments to have length 0, and we stipulate that segments are separated by single
// air blocks, then an extender of length L will have L + 1 segments. This is easy
// to see when the extender is fully extended: one segment for each piston, plus a
// segment for the extended block. Segments will grow and shrink as pistons pull and
// push blocks between segments. With each push and pull there is a transaction:
// blocks move from one segment to another. Thus, as one segment loses x blocks,
// another will gain x blocks.
//
// Second, we introduce the concept of the virtual push limit. It is well known that
// pistons have a push limit of 12. We call this the hard push limit, as no matter
// what, pistons cannot move more than that number of blocks. However, another type
// of push limit emerges, as a piston is not able to push due to blocks moving in
// front of it. This virtual push limit depends entirely on the period of Snaperz
// extender. For an extender with period P the virtual push limit turns out to be
// (P / 4) - 2. Within a Snaperz extender a piston's behavior can be described
// entirely in terms of this virtual push limit.
//
// Armed with these two concepts, we construct a new algorithm. Rather than
// simulating the extender one game tick at a time, we simulate it one pulse through
// the repeater line at a time. While in reality multiple pulses are happening
// simultaneously, the virtual push limit captures the effects of that perfectly.
// Combined with the segmentation approach the behavior of the extender is simplified
// a lot. The only special case is the segment with the extended block. This block
// does not contribute to the virtual push limit, but it does contribute to the hard
// push limit.
//
// As each pulse moves across the extender, it causes each segment to grow or shrink
// depending on its length and the length of the segment ahead of it. If we number
// each segment 0 to L from back to front, the algorithm works according to the
// following pseudo code for each pulse.
// for Segment s_k in extender:
//     if len(s_k) == 0: # empty segment, no pistons to push or pull any blocks
//         continue
//     if len(s_k) == 1: # single piston, it will pull the next segment
//         set_len(s_k, len(s_k) + len(s_k+1)) # grow current segment
//         set_len(s_k+1, 0)                   # shrink next segment to 0
//     if len(s_k) > 1: # multiple pistons, blocks will be pushed into next segment
//         blocks_to_push = min(push_limit, len(s_k)) # push at most push_limit number of blocks
//         set_len(s_k, len(s_k) - blocks_to_push)     # shrink current element
//         set_len(s_k+1, len(s_k+1) + blocks_to_push) # grow next element
//
// This algorithm can be repeated until the extender is fully retracted. To check if
// the extender is retracted, we can query the length of the first segment. If that
// segment has length L + 1, then it must contain the extended block, and thus the
// extender is fully retracted.

struct SnaperzExtender;

// Initializes the given extender to the extended state, i.e. one where every
// segment has length one (indicating that every piston is in its own segment
// with one air block in between).
//
// Note: the extender returned by this method must be destroyed using the
//       complementary destroy_snaperz_extender(...) function.
SnaperzExtender create_snaperz_extender();

// Frees up memory used by a snaperz extender created after an invocation of
// the create_snaperz_extender() function.
//
// Note: the extender is no longer usable after an invocation of this method.
void destroy_snaperz_extender(SnaperzExtender& extender);

// Simulate a single extender pulse. Note that while in-game multiple pulses
// occur simultaneously, this function captures that context in the virtual
// push limit, which is dependent on the period of the extender.
void snaperz_extender_simulate_pulse(SnaperzExtender& extender);

// Checks if two extenders contain the same segments
bool snaperz_extender_equal(const SnaperzExtender& lhs, const SnaperzExtender& rhs);

// Checks whether the given extender is finished, i.e. whether the extender
// reached the goal state, where every block is retracted into a single
// segment.
bool snaperz_extender_finished(const SnaperzExtender& extender);

// Specialized implementations of the snaperz extender.
#if __AVX2__
// Use the faster AVX2 implementation
#include "snaperz_extender_avx2.h"
#else // __AVX2__
#warning "AVX2 not supported. Using fallback implementation."
#include "snaperz_extender_fallback.h"
#endif // !__AVX2__
