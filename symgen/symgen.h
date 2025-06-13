#ifndef SYMGEN
#define SYMGEN

#define PY_SSIZE_T_CLEAN

#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define INT_T NPY_INT32
#define NUMBER_T NPY_FLOAT32

typedef npy_int32 int_t;
typedef npy_float32 number_t;

typedef unsigned char byte;
typedef unsigned int uint;

typedef union {
  int_t integer;
  number_t number;
} arg_t;

typedef struct {
  int_t command;
  arg_t argument;
} instruction_t;

#endif // define SYMGEN