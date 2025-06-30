#ifndef SYMGEN_BITSET_H
#define SYMGEN_BITSET_H

/* need to be defined in the main file */
// #define BIT_SET_SIZE ${BIT_SET_SIZE}

typedef unsigned char bitmask_t;
const bitmask_t FULL_MASK = ~((bitmask_t)0u);

typedef struct {
  bitmask_t bits[BIT_SET_SIZE];
  unsigned int size;
} BitSet;

static inline void print_bitset(const BitSet bitset) {
  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    for (unsigned int j = 0; j < sizeof(bitmask_t) * BITS; ++j) {
      int value = (bitset.bits[i] & (1u << j)) ? 1 : 0;
      printf("%d ", value);
    }
    printf(" ");
  }
  printf("(size = %u)", bitset.size);
}

static inline unsigned int bitset_size(const bitmask_t * bits) {
  unsigned int size = 0;

  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    for (unsigned int j = 0; j < sizeof(bitmask_t) * BITS; ++j) {
      if (bits[i] & (1u << j)) {
        ++size;
      }
    }
  }

  return size;
}

static inline unsigned int bitset_contains(const BitSet bitset, const unsigned i) {
  const unsigned int block = i / (sizeof(bitmask_t) * BITS);
  const unsigned int offset = i % (sizeof(bitmask_t) * BITS);
  return (bitset.bits[block] & (1u << offset));
}

static inline BitSet bitset_empty() {
  BitSet bitset;

  int i;
  for (i = 0; i < BIT_SET_SIZE; ++i) {
    bitset.bits[i] = 0u;
  }
  bitset.size = 0;
  return bitset;
}

static inline BitSet bitset_full(unsigned int n) {
  BitSet bitset;

  unsigned int i;
  for (i = 0; i < n / (sizeof(bitmask_t) * BITS); ++i) {
    bitset.bits[i] = FULL_MASK;
  }
  const unsigned int offset = n % (sizeof(bitmask_t) * BITS);
  bitset.bits[i] = ~(FULL_MASK << offset);

  for (++i; i < BIT_SET_SIZE; ++i) {
    bitset.bits[i] = 0;
  }
  bitset.size = n;

  return bitset;
}

static inline unsigned int bitset_random_element(pcg32_random_t * rng, BitSet bitset) {
  uint32_t u = pcg32_random_r(rng);
  uint32_t n = u % bitset.size;

  #ifdef SYMGEN_DEBUG
    printf("selecting %u (%u) from: ", n, u);
    print_bitset(bitset);
    printf("\n");
  #endif

  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    for (unsigned int j = 0; j < sizeof(bitmask_t) * BITS; ++j) {
      if (bitset.bits[i] & (1u << j)) {
        if (n == 0) {

          #ifdef SYMGEN_DEBUG
            printf("selected %lu\n", i * sizeof(bitmask_t) * BITS + j);
          #endif

          return i * sizeof(bitmask_t) * BITS + j;
        } else {
          --n;
        }
      }
    }
  }

  return -1;
}

static inline BitSet bitset_random_subset(pcg32_random_t * rng, BitSet bitset) {
  if (bitset.size == 0) {
    return bitset;
  }

  BitSet result;

  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    const bitmask_t selection_mask = (bitmask_t) (pcg32_random_r(rng) % FULL_MASK);
    result.bits[i] = bitset.bits[i] & selection_mask;
  }

  result.size = bitset_size(result.bits);

  return result;
}

static inline BitSet bitset_random_subset_non_empty(pcg32_random_t * rng, BitSet bitset) {
  if (bitset.size == 0) {
    return bitset;
  }

  BitSet result;

  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    const bitmask_t selection_mask = (bitmask_t) (pcg32_random_r(rng) % FULL_MASK);
    result.bits[i] = bitset.bits[i] & selection_mask;
  }

  result.size = bitset_size(result.bits);

  if (result.size == 0) {
    // rejecting sampling: there is 0.5 chance it succeeds
    // setting a random element to 1 artificially lifts the prob of size 1 sets
    return bitset_random_subset_non_empty(rng, bitset);
  }

  return result;
}

static inline BitSet bitset_random_subset_p(pcg32_random_t * rng, BitSet bitset, double p) {
  if (bitset.size == 0) {
    return bitset;
  }

  BitSet result;

  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    result.bits[i] = 0u;

    for (unsigned int j = 0; j < sizeof(bitmask_t) * BITS; ++j) {
    const bitmask_t offset = (1u << j);
      /// saving rng calls
      if (bitset.bits[i] & offset) {
        const double u = pcg32_uniform(rng);
        result.bits[i] |= (u < p) ? offset : 0u;
      }
    }
  }

  result.size = bitset_size(result.bits);

  return result;
}

static inline BitSet bitset_random_subset_p_non_empty(pcg32_random_t * rng, BitSet bitset, double p) {
  if (bitset.size == 0) {
    return bitset;
  }

  BitSet result;

  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    result.bits[i] = 0u;

    for (unsigned int j = 0; j < sizeof(bitmask_t) * BITS; ++j) {
    const bitmask_t offset = (1u << j);
      /// saving rng calls
      if (bitset.bits[i] & offset) {
        const double u = pcg32_uniform(rng);
        result.bits[i] |= (u < p) ? offset : 0u;
      }
    }
  }

  result.size = bitset_size(result.bits);

  if (result.size > 0) {
    return result;
  }

  if (p > 0.25) {
    return bitset_random_subset_p_non_empty(rng, bitset, p);
  }

  /// forcing one bit because rejection sampling might take a lot of iterations
  uint32_t n = pcg32_random_r(rng) % bitset.size;
  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    for (unsigned int j = 0; j < sizeof(bitmask_t) * BITS; ++j) {
      const bitmask_t offset = (1u << j);
      if (bitset.bits[i] & offset) {
        if (n == 0) {
          result.bits[i] |= offset;
          result.size = 1;
          return result;
        } else {
          --n;
        }
      }
    }
  }

  return result;
}

static inline void shuffle(pcg32_random_t * rng, unsigned int * array, unsigned int size, unsigned int limit) {
  unsigned int tmp;

  limit = limit < size ? limit : size;

  for (unsigned int i = 0; i < limit; ++i) {
    const unsigned int j = i + pcg32_random_r(rng) % (size - i);
    // swapping i and j, they might be in the same location!
    tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
  }
}

static inline BitSet bitset_random_choice_rejection_sampling(pcg32_random_t * rng, BitSet bitset, unsigned int n) {
  unsigned int k = 0;
  BitSet result = bitset_empty();

  while (k < n) {
    const unsigned i = pcg32_random_r(rng) % (BIT_SET_SIZE * BITS);
    const unsigned int block = i / (sizeof(bitmask_t) * BITS);
    const unsigned int offset = i % (sizeof(bitmask_t) * BITS);
    const bitmask_t mask = (bitmask_t) (1u << offset);

    if ((bitset.bits[block] & mask) && !(result.bits[block] & mask)) {
      result.bits[block] |= mask;
      k++;
    }
  }
  result.size = n;
  return result;
}

static inline BitSet bitset_random_choice_shuffle(pcg32_random_t * rng, BitSet bitset, unsigned int n) {
  if (n >= bitset.size) {
    return bitset;
  }

  unsigned int elements[bitset.size];
  unsigned int k = 0;

  for (unsigned int i = 0; (i < BIT_SET_SIZE) && (k < bitset.size); ++i) {
    for (unsigned int j = 0; j < sizeof(bitmask_t) * BITS; ++j) {
      const bitmask_t offset = (1u << j);
      if (bitset.bits[i] & offset) {
        elements[k] = i * sizeof(bitmask_t) * BITS + j;
        k++;
      }
    }
  }

  shuffle(rng, elements, bitset.size, n);

  BitSet result = bitset_empty();

  for (unsigned int i = 0; i < n; ++i) {
    const unsigned int block = elements[i] / (sizeof(bitmask_t) * BITS);
    const unsigned int offset = elements[i] % (sizeof(bitmask_t) * BITS);
    result.bits[block] |= (1u << offset);
  }
  result.size = n;
  return result;
}

static inline BitSet bitset_random_choice_threshold(
  pcg32_random_t * rng, BitSet bitset,
  const unsigned int n, const double sampling_rate_threshold
) {
  if (n >= bitset.size) {
    return bitset;
  }

  const double sampling_rate = n / (bitset.size - n + 1);

  if (sampling_rate < sampling_rate_threshold) {
    return bitset_random_choice_rejection_sampling(rng, bitset, n);
  } else {
    return bitset_random_choice_shuffle(rng, bitset, n);
  }
}

// was determined by tests
// should be around 1 by the order of magnitude.
// the complexity of rejection sampling n from N with total M on average is
// sum_{i =0}^{k} 1 / (probability of success)
// sum_{i = 0}^{k} 1 / (n / N * (n - i) / n) = sum_i N / (n - i) ~ N log (n / (n - k + 1))
// the complexity of shuffle on average
// O(N / 2) + O(k)
// i.e., if k << n or k ~= n but n is small rejection sampling might be more efficient
// the combined random_choice does not seem to be very sensitive to the constant
#define MAGICAL_THRESHOLD_RATE 0.45

static inline BitSet bitset_random_choice(pcg32_random_t * rng, BitSet bitset, const unsigned int n) {
  return bitset_random_choice_threshold(rng, bitset, n, MAGICAL_THRESHOLD_RATE);
}

static inline BitSet bitset_random_choice_fraction(pcg32_random_t * rng, BitSet bitset, const double fraction) {
  int n = (int) ceil(fraction * bitset.size);
  if (n <= 0) {
    return bitset_empty();
  }
  if (n >= bitset.size) {
    return bitset;
  }
  return bitset_random_choice(rng, bitset, (unsigned int) n);
}

static inline BitSet bitset_range(const unsigned int start, const unsigned int end) {
  BitSet result;

  for (unsigned int i = 0; i < BIT_SET_SIZE; ++i) {
    result.bits[i] = 0u;
  }
  for (unsigned int i = start; i < end; ++i) {
    const unsigned int block = i / (sizeof(bitmask_t) * BITS);
    const unsigned int offset = i % (sizeof(bitmask_t) * BITS);
    const bitmask_t mask = 1u << offset;
    result.bits[i] |= mask;
  }
  result.size = end - start;
  result.size = result.size < 0 ? 0 : result.size;
  return result;
}

#endif // SYMGEN_BITSET_H