#include "Random.h"
#include <thread>

TT800& TT800::Get()
{
    static thread_local TT800 s_rng{0};
    return s_rng;
}

TT800::TT800()
    :m(7), s(7), t(15), a(0x8ebfd028U), b(0x2b5b2500U), c(0xdb8b0000U)
{
    Init(time(nullptr));
}

TT800::TT800(uint32_t seed)
    :m(7), s(7), t(15), a(0x8ebfd028U), b(0x2b5b2500U), c(0xdb8b0000U)
{
    if (seed)
        Init(seed);
    else
        Init(time(nullptr));// TODO
}

void TT800::Init(uint32_t seed)
{
    for(int i = 0; i < N; i++)
    {
#if BIG_UINT
        seed &= 0xffffffffU;
#endif
        x[i] = seed;
        seed *= 1313;
        seed += 88897;
    }
    k = N - 1; /* Force an immediate iteration of the TGFSR. */
}

uint32_t TT800::Next()
{
    int y;
    if(++k == N)
    {
        //TGFSR
        for(int i = 0; i < N - m; i++)
            x[i] = x[i + m] ^ (x[i] >> 1) ^ ((x[i] & 1) ? a : 0);
        for(int i = N - m; i < N; i++)
            x[i] = x[i + m - N] ^ (x[i] >> 1) ^ ((x[i] & 1) ? a : 0);
        k = 0;
    }
    y  = x[k];
    y ^= ((y << s) & b);
    y ^= ((y << t) & c);
#if BIG_UINT
    y &= 0xffffffffU;
#endif
    return y;
}

int32_t TT800::R31()
{
    return (int32_t)(Next() & 0x7fffffff);
}

uint32_t TT800::R32()
{
    return Next();
}

uint64_t TT800::R64()
{
    return ((uint64_t) Next() << 32) | Next();
}

float TT800::RFloat()
{
    return Next() * 2.328306436538696e-10;/* 2^-32 */
}

int32_t TT800::FastR31(int32_t range)
{
    return ((R31() & 0xffff) * range) >> 16;
}

int32_t TT800::RangeR31(int min, int max)
{
    return min + (R31() % (max - min));
}

int32_t TT800::FastRange31(int min, int max)
{
    return min + (((R31() & 0xffff) * (max - min)) >> 16);
}
