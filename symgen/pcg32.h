#ifndef SYMGEN_PCG_32_H
#define SYMGEN_PCG_32_H

#include <stdint.h>

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

static inline uint32_t pcg32_random_r(pcg32_random_t * rng);

// Initialize the generator with seed and sequence
static inline void pcg32_srandom(pcg32_random_t * rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}

static inline uint32_t pcg32_random_r(pcg32_random_t * rng) {
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline double pcg32_uniform(pcg32_random_t * rng) {
  const long u = pcg32_random_r(rng);
  return ((double) u) / 4294967295;
}

static inline number_t pcg32_uniform_value(pcg32_random_t * rng) {
  return (number_t) pcg32_uniform(rng);
}

static inline number_t pcg32_normal(pcg32_random_t * rng) {
  double u = 0.0;
  const int n = 10;
  const double mean = n / 2;
  const double norm = sqrt(n / 12);
  for (int i = 0; i < n; ++i) {
    u += pcg32_uniform(rng);
  }

  return (u - mean) / norm;
}

#endif