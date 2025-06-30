#ifndef SYMGEN_STABLE_H
#define SYMGEN_STABLE_H

#include <math.h>
#include "symgen.h"

const number_t MEAN_LOG_NORMAL = 1.6487212707001282;
const number_t STD_LOG_NORMAL = 2.1611974158950877;
const number_t VAR_LOG_NORMAL = 4.670774270471604;

const number_t SQRT_2 = sqrt(2.0);
const number_t SQRT_3 = sqrt(3.0);
const number_t INV_SQRT_2 = sqrt(0.5);
const number_t INV_SQRT_3 = 1.0 / sqrt(3.0);
const number_t INV_SQRT_12 = 1.0 / sqrt(12.0);
const number_t LOG_2 = log(2.0);

static inline number_t pinv_p_c(number_t x, number_t c) {
  const double magic_beta = -9.86342959e-01;
  const double magic_alpha = -4.40520003e+00;

  // log1p instead of log to make it stable around 0.0 at the expense of some bias
  const double asymptotic = log1p(c);
  const double local = magic_beta * exp(magic_alpha * c);
  return exp(asymptotic + local) * MEAN_LOG_NORMAL / (x + c);
}

#endif // SYMGEN_STABLE_H