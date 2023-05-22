#include <iostream>
#include <chrono>
#include <cstdint>
#include <algorithm>

// Defines the extender itself
static constexpr uint32_t kLength = 53;
static constexpr uint32_t kPeriod = 12;

// Constants
static constexpr uint32_t kHardPushLimit = 12;
static constexpr uint32_t kVirtualPushLimit = (kPeriod / 4 - 2);
static constexpr uint32_t kPushLimit = std::min(kHardPushLimit, kVirtualPushLimit);

// Definitions for checking loops. Use 1 for on, 0 for off.
#define CHECK_LOOP 1
// Can be up to 2 times faster at finding loops, but slows down simulation slightly.
#define FAST_LOOP_DETECTION 1

// Definitions for logging status updates
#define LOG_STATUS_UPDATES 1
// interval in number of pulses
#define LOGGING_INTERVAL 1000000000


// To learn more about Snaperz extenders, take a look at this document:
// https://docs.google.com/document/d/1KCM7lk-GBn_-RIhuuUiZdNNiBWDc6Zm7g88cdIFOeQg/edit?usp=sharing
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


struct BlockSegment
{
    uint32_t len;
    BlockSegment* next;
};

// Simulate a single extender pulse. Note that while in-game multiple pulses
// occur simultaneously, this function captures that context in the virtual
// push limit, which is dependent on the period of the extender.
void simulate_pulse(BlockSegment* root)
{
    auto curr = root;
    while (true)
    {
        if (curr->len > 1)
        {
            // Handle pushing case:
            //   This case is quite trivial, since we simulate the extender in
            //   the direction of the push. Therefore, we only need to consider
            //   how many pistons are pushed at the end of this segment, to form
            //   a new segment (or merge with the next sequential segment). How
            //   the pistons push into further small segments is then handled by
            //   simulating the next sequential segment in the next iteration of
            //   the loop.
            uint32_t blocks_to_push;
            if (kPushLimit == kHardPushLimit || curr->next != nullptr)
            {
                blocks_to_push = std::min(kPushLimit, curr->len - 1);
            }
            else
            {
                // Note: We are at the last segment. Since the last block in
                //       this segment is not a piston, the virtual push limit
                //       no longer applies, and thus we push an extra block.
                blocks_to_push = std::min(kPushLimit + 1, curr->len - 1);
            }
            // The blocks should be moved to the next sequential segment.
            auto seq_next = curr + 1;
            if (seq_next->len == 0)
            {
                // Insert the segment into the linked list (since it was zero,
                // and therefore not present previously).
                seq_next->next = curr->next;
                curr->next = seq_next;
            }
            // Actually perform the move
            curr->len -= blocks_to_push;
            seq_next->len += blocks_to_push;
        }
        else
        {
            // Handle pulling case:
            //   This case is not as simple, since it requires some knowledge of
            //   how the extender behaves. In particular, whenever a piston is
            //   pulled, it will immediately pull the next piston in front of it.
            //   This has the consequence of essentially merging the next segment
            //   into the current segment.
            if (curr->next == nullptr)
            {
                // The segment only consists of the last block,
                // i.e. not a piston. Therefore, we can not pull
                // anything.
                break;
            }
            auto seq_next = curr + 1;
            if (seq_next->len != 0)
            {
                // Completely merge the segment into the current segment.
                curr->len += seq_next->len;
                seq_next->len = 0;
                // Remove the segment from the linked list.
                curr->next = seq_next->next;
                if (curr->next == nullptr)
                {
                    // We merged with the last segment in the extender.
                    break;
                }
            }
        }
        // Go to next segment.
        curr = curr->next;
    }
}

// Initializes the given segments to the extended state, i.e. one where every
// segment has length one (indicating that every piston is in its own segment
// with one air block in between).
void init_segments(BlockSegment* segments, uint32_t count)
{
    for (uint32_t i = 0; i < count; i++)
    {
        auto curr = segments + i;
        curr->len = 1;
        // Extra check if we are the last segment.
        curr->next = (i + 1 != count) ? curr + 1 : nullptr;
    }
}

// Checks if two extenders contain the same segments
bool extenders_equal(BlockSegment* lhs, BlockSegment* rhs)
{
    auto lhs_curr = lhs, rhs_curr = rhs;
    while (lhs_curr != nullptr && rhs_curr != nullptr)
    {
        // Check if the segment length is the same
        if (lhs_curr->len != rhs_curr->len)
        {
            return false;
        }
        // Check if the relative indices are the same
        if ((lhs_curr - lhs) != (rhs_curr - rhs))
        {
            return false;
        }
        // Go to next segment
        lhs_curr = lhs_curr->next;
        rhs_curr = rhs_curr->next;
    }
    // Check if they are both equal (i.e. nullptr).
    return (lhs_curr == rhs_curr);
}

void simulate_extender()
{
    // Note: leave an extra segment for the last block.
    BlockSegment segments[kLength + 1];
    BlockSegment slow_segments[kLength + 1];

    init_segments(segments, kLength + 1);
    init_segments(slow_segments, kLength + 1);

    uint64_t pulses = 0;
    uint64_t pulses_since_last_status_update = 0;
    while (segments->next != nullptr)
    {
        pulses++;

#if LOG_STATUS_UPDATES
        pulses_since_last_status_update++;

        if (pulses_since_last_status_update == LOGGING_INTERVAL)
        {
            pulses_since_last_status_update = 0;

            std::cout
                << pulses
                << " pulses so far..."
                << std::endl;
        }
#endif // LOG_STATUS_UPDATES

        simulate_pulse(segments);
#if CHECK_LOOP
        // Skip every other pulse for the slower segments.
        if ((pulses & 0x1) == 0)
        {
            simulate_pulse(slow_segments);
        }
#if FAST_LOOP_DETECTION
        if (extenders_equal(segments, slow_segments)) // loop check
        {
#else // FAST_LOOP_DETECTION
        // Slow loop detection: skip every other check.
        if ((pulses & 0x1) == 0 && extenders_equal(segments, slow_segments)) // loop check
        {
#endif // !FAST_LOOP_DETECTION
            std::cout
                << "Loop at: "
                << pulses << " pulses."
                << std::endl;
            return;
        }
#endif // CHECK_LOOP
    }
    std::cout
        << pulses << " pulses in total."
        << std::endl;
}

int main()
{
    std::cout
        << "Running "
        << kLength << " extender, "
        << kPeriod << " tick period."
        << std::endl;
    
    auto start_time = std::chrono::steady_clock::now();
    simulate_extender();
    auto delta = std::chrono::steady_clock::now() - start_time;
    
    // Print status message at the end.
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(delta).count();
    std::cout
        << std::endl
        << "Done! The operation took "
        << elapsed / 1000.0
        << " seconds."
        << std::endl;
    return 0;
}
