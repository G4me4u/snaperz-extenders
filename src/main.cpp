#include <iostream>
#include <chrono>
#include <cstdint>
#include <algorithm>

#include "snapers_extender.h"
#include "constants.h"

void simulate_extender()
{
    SnaperzExtender extender = create_snaperz_extender();
    uint64_t pulses = 0;

#if LOG_STATUS_UPDATES
    uint64_t pulses_since_last_status_update = 0;
#endif // LOG_STATUS_UPDATES

#if CHECK_LOOP
    SnaperzExtender slow_extender = create_snaperz_extender();
#endif // CHECK_LOOP

    while (!snaperz_extender_finished(extender))
    {
        snaperz_extender_simulate_pulse(extender);
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

#if CHECK_LOOP
        // Skip every other pulse for the slower segments.
        if ((pulses & 0x1) == 0)
        {
            snaperz_extender_simulate_pulse(slow_extender);
        }
#if FAST_LOOP_DETECTION
        if (snaperz_extender_equal(extender, slow_extender)) // loop check
        {
#else // FAST_LOOP_DETECTION
        // Slow loop detection: skip every other check.
        if ((pulses & 0x1) == 0 && snaperz_extender_equal(extender, slow_extender)) // loop check
        {
#endif // !FAST_LOOP_DETECTION
            std::cout
                << "Loop at "
                << pulses
                << " pulses."
                << std::endl;
            return;
        }
#endif // CHECK_LOOP
    }
    // Print final status message.
    std::cout
        << "Done! "
        << pulses
        << " pulses in total."
        << std::endl;

    // Perform Cleanup
    destroy_snaperz_extender(extender);
#if CHECK_LOOP
    destroy_snaperz_extender(slow_extender);
#endif // CHECK_LOOP
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
        << "The operation took "
        << elapsed / 1000.0
        << " seconds."
        << std::endl;
    return 0;
}
