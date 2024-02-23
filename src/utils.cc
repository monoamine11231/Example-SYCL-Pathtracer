#include "include/utils.h"

int miscutils::ShadowFactor(float a) {
    return a > 0 ? 1 : 0; 
}

double miscutils::XorShiftPRNG::operator()() {
    uint64_t x = this->state_;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;

    this->state_ = x;
    return (double)(x) / (double)(0xFFFFFFFFFFFFFFFF);
}

/* Big creds to:
https://lemire.me/blog/2018/08/15/fast-strongly-universal-64-bit-hashing-everywhere/
*/
uint64_t miscutils::Hash64(uint64_t i) {
    uint64_t a1 = 0x65d200ce55b19ad8;
    uint64_t b1 = 0x4f2162926e40c299;
    uint64_t c1 = 0x162dd799029970f8;
    uint64_t a2 = 0x68b665e6872bd1f4;
    uint64_t b2 = 0xb6cfcf9d79b51db2;
    uint64_t c2 = 0x7a2b92ae912898c2;

    uint32_t low = (uint32_t)i;
    uint32_t high = i >> 32;

    return ((a1 * low + b1 * high + c1) >> 32)
        | ((a2 * low + b2 * high + c2) & 0xFFFFFFFF00000000L);
}