#pragma once

#include "constants.h"

namespace snaperz
{
  // Paranoid sanity check; not a hard limitation. Just a slight
  // optimization over using bytes or similar for the block segments.
  // Very unlikely that this will fail anyway.
  static_assert(std::numeric_limits<len_t>::max() <= std::numeric_limits<uint32_t>::max(),
                "Extender length must fit into a 32-bit uint");

  struct BlockSegment
  {
    uint32_t len;
    BlockSegment* next;
  };

  struct Extender
  {
    BlockSegment* segments;
  };

  Extender create()
  {
    // Note: leave an extra segment for the last block.
    Extender extender;
    extender.segments = new BlockSegment[kLength + 1];
    for (uint32_t i = 0; i < kLength + 1; i++)
    {
      auto curr = extender.segments + i;
      curr->len = 1;
      // Extra check if we are the last segment.
      curr->next = (i != kLength) ? curr + 1 : nullptr;
    }
    return std::move(extender);
  }

  void destroy(Extender& extender)
  {
    delete[] extender.segments;
    extender.segments = nullptr;
  }

  void simulate_pulse(Extender& extender)
  {
    auto curr = extender.segments;
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

        // Note: If we are at the last segment. Since the last block in this
        //       segment is not a piston, the virtual push limit no longer
        //       applies, and thus we push an extra block.
        uint32_t push_limit = curr->next ? kPushLimit : kLastPushLimit;
        uint32_t blocks_to_push = std::min(push_limit, curr->len - 1);
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

  bool equals(const Extender& lhs, const Extender& rhs)
  {
    auto lhs_curr = lhs.segments;
    auto rhs_curr = rhs.segments;
    while (lhs_curr && rhs_curr)
    {
      // Check if the segment length is the same
      if (lhs_curr->len != rhs_curr->len)
      {
        return false;
      }
      // Check if the relative indices are the same
      if ((lhs_curr - lhs.segments) != (rhs_curr - rhs.segments))
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

  bool finished(const Extender& extender)
  {
    // Check if extender itself is the last segment, i.e. that
    // every block is in the first segment.
    return extender.segments->next == nullptr;
  }
}
