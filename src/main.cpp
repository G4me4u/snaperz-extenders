#include <iostream>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <algorithm>

#include "snaperz_extender.h"
#include "constants.h"

std::ostream& print_time(std::ostream& os, std::chrono::nanoseconds ns)
{
    typedef std::chrono::duration<uint64_t, std::ratio<86400>> days;
    char fill = os.fill();
    os.fill('0');
    auto d = std::chrono::duration_cast<days>(ns);
    ns -= d;
    auto h = std::chrono::duration_cast<std::chrono::hours>(ns);
    ns -= h;
    auto m = std::chrono::duration_cast<std::chrono::minutes>(ns);
    ns -= m;
    auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
    os << std::setw(2) << d.count() << "d:"
       << std::setw(2) << h.count() << "h:"
       << std::setw(2) << m.count() << "m:"
       << std::setw(2) << s.count() << 's';
    os.fill(fill);
    return os;
}

void simulate_extender()
{
  snaperz::Extender extender = snaperz::create();
  uint64_t pulses = 0;

#if LOG_STATUS_UPDATES
  uint64_t pulses_since_last_status_update = 0;
#endif // LOG_STATUS_UPDATES

#if CHECK_LOOP
  snaperz::Extender slow_extender = snaperz::create();
#endif // CHECK_LOOP

  while (!snaperz::finished(extender))
  {
    snaperz::simulate_pulse(extender);
    pulses++;

#if LOG_STATUS_UPDATES
    pulses_since_last_status_update++;
    if (pulses_since_last_status_update == LOGGING_INTERVAL)
    {
      pulses_since_last_status_update = 0;
      std::cout
        << '\r'
        << pulses
        << " pulses so far..."
        << std::flush;
    }
#endif // LOG_STATUS_UPDATES

#if CHECK_LOOP
    // Skip every other pulse for the slower segments.
    if ((pulses & 0x1) == 0)
    {
      snaperz::simulate_pulse(slow_extender);
    }
#if FAST_LOOP_DETECTION
    if (snaperz::equals(extender, slow_extender)) // loop check
    {
#else // FAST_LOOP_DETECTION
    // Slow loop detection: skip every other check.
    if ((pulses & 0x1) == 0 && snaperz::equals(extender, slow_extender)) // loop check
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
  snaperz::destroy(extender);
#if CHECK_LOOP
  snaperz::destroy(slow_extender);
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
  std::cout
    << std::endl
    << "The operation took: ";
  print_time(std::cout, delta);
  std::cout << std::endl;
  return 0;
}
