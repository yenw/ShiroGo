#pragma once
#include <climits>
#include <cstdint>
#include <ctime>
#include <limits>

#define BIG_UINT (UINT_MAX > 0xffffffffU)
class TT800
{
public:
    TT800();
    TT800(uint32_t seed);
    static TT800& Get();

    void Init(uint32_t seed);
     int32_t FastR31(int32_t range);
     int32_t RangeR31(int min, int max);
     int32_t FastRange31(int min, int max);
     int32_t R31();
    uint32_t R32();
    uint64_t R64();
       float RFloat();

    // UniformRandomBitGenerator interface
    using result_type = std::uint32_t;
    constexpr static result_type min() { return std::numeric_limits<result_type>::min(); }
    constexpr static result_type max() { return std::numeric_limits<result_type>::max(); }
    result_type operator()() { return R32(); }

private:
    uint32_t Next();

private:
    const int m;
    const int s;
    const int t;
    const uint32_t a;
    const uint32_t b;
    const uint32_t c;
    static const int N = 25;

    uint32_t x[N];
    int k;
};

typedef TT800 GoRandom;
