#pragma once

#include <inttypes.h>

/// Example:
/// ```C
/// uint8_t digest[16]; // also a uint64_t[2];
/// md5(&ze_struct, sizeof(the_struct), digest);
/// ```
__device__ void md5(uint8_t *const, uint32_t, uint8_t *);
