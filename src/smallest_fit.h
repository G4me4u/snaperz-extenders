#pragma once

#include <limits>
#include <type_traits>

template<auto N, typename T1, typename... Args>
struct choose_smallest_fit
{
    typedef typename std::conditional<
        N <= std::numeric_limits<T1>::max(),
        T1,
        typename choose_smallest_fit<N, Args...>::type
    >::type type;
};

template<auto N, typename T>
struct choose_smallest_fit<N, T>
{
    static_assert(N <= std::numeric_limits<T>::max(),
                  "Value does not fit into any unsigned integral");
    typedef T type;
};

template<auto N>
struct smallest_fit
{
    typedef typename choose_smallest_fit<
        N,
        uint8_t,
        uint16_t,
        uint32_t,
        uint64_t
    >::type type;
};
