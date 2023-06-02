#pragma once

#include <cstdint>
#include <algorithm>

#include "smallest_fit.h"

// Defines the extender itself
static constexpr uint32_t kLength = 65;
static constexpr uint32_t kPeriod = 12;
static constexpr uint32_t kHardPushLimit = 12;

// Constants
static constexpr uint32_t kVirtualPushLimit = (kPeriod / 4 - 2);
static constexpr uint32_t kPushLimit = std::min(kHardPushLimit, kVirtualPushLimit);
static constexpr uint32_t kLastPushLimit = std::min(kPushLimit + 1, kHardPushLimit);

typedef smallest_fit<kLength + 1>::type len_t;

// Definitions for checking loops. Use 1 for on, 0 for off.
#define CHECK_LOOP 1
// Can be up to 2 times faster at finding loops, but slows down simulation slightly.
#define FAST_LOOP_DETECTION 1

// Definitions for logging status updates
#define LOG_STATUS_UPDATES 1
// Interval in number of pulses
#define LOGGING_INTERVAL UINT64_C(100000000)
