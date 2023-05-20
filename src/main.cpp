#include <iostream>
#include <chrono>
#include <cstdint>
#include <algorithm>

// Defines the extender itself
static constexpr uint32_t kLength = 53;
static constexpr uint32_t kPeriod = 12;

// Constants
static constexpr uint32_t kHardPushLimit = 12;
static constexpr uint32_t kVirtualPushLimit (kPeriod / 4 - 2);
static constexpr uint32_t kPushLimit = std::min(kHardPushLimit, kVirtualPushLimit);

// Definitions for checking loops. Use 1 for on, 0 for off.
#define CHECK_LOOP 1
// Can be up to 2 times faster at finding loops, but slows down simulation slightly.
#define FAST_LOOP_DETECTION 0

struct BlockSegment
{
    uint32_t len;
    BlockSegment* next;
};

// Simulate a single extender pulse. Note that when performing the pulses in-game,
// there might be multiple pulses happening at the same time. This function will
// simulate how the pulse will affect the piston after going through every piston.
// It assumes that the pulses are equally spaced according to kPeriod.
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
            //   This has the consequence of essentially merging the current and
            //   next segment into a single segment.
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
    while (segments->next != nullptr)
    {
        pulses++;

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
        << pulses << " pulses."
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
