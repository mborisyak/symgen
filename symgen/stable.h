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

static inline number_t ndiv_n_p_c(number_t x1, number_t x2, number_t c) {
  const double w1 = 0.99056203;
  const double w2 = 4.35480133;

  const double inv_std = exp(log1p(c) - w1 * exp(-w2 * c));
  return x1 / (x2 + c) * inv_std;
}

static inline number_t psqrt_p_c(number_t x, number_t c) {
  const double w0 = 0.23253458;
  const double w2 = -0.342227;
  const double w3 = 5.0;
  const double norm = exp(
    w0 + 0.5 * log1p(c) + w2 * exp(-w3 * c)
  );
  return (sqrt(x + c) - sqrt(c)) * MEAN_LOG_NORMAL * norm;
}

static inline number_t ntanh_n_c(number_t x, number_t c) {
  const double w0 = 0.638976995;
  const double w1 = 0.52642439;
  const double w2 = 0.6372085;

  const double std = w0 * (tanh(w1 * c) + 1) * (tanh(-w1 * c) + 1);
  const double mean = tanh(w2 * c);

  return (tanh(x + c) - mean) / std;
}

static inline number_t nlog_p_c(number_t x, number_t c) {
  const double w0 =-1.26318484;
  const double w1 = 1.49033208;

  const double mean = log(c + SQRT_2) - 0.5 * LOG_2 * exp(w0 * c);
  const double std = w1 / (w1 + c);

  return (log(x + c) - mean) / std;
}

#endif // SYMGEN_STABLE_H