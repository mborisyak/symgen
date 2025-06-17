#include "symgen.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <Python.h>
#include "numpy/arrayobject.h"

// debug on / off
${DEBUG}

#ifdef SYMGEN_DEBUG
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

#define STATUS_OK 0
#define ERROR_STACK 1
#define ERROR_MAX_DEPTH 2
#define ERROR_UNPROCESSED_CONDITION 3

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

static inline double uniform(pcg32_random_t * rng) {
  const long u = pcg32_random_r(rng);
  return ((double) u) / 4294967295.0;
}

static inline number_t uniform_value(pcg32_random_t * rng) {
  return (number_t) uniform(rng);
}

typedef struct {
  instruction_t * stack;
  instruction_t * empty;
  instruction_t * full;
} InstructionStack;

static inline int push_instruction(InstructionStack * instruction_stack, instruction_t element) {
  if (instruction_stack->stack < instruction_stack->full) {
    DEBUG_PRINT("PUSH! %d\n", element.command);
    *(instruction_stack->stack) = element;
    ++(instruction_stack->stack);
    return STATUS_OK;
  } else {
    return ERROR_STACK;
  }
}

static inline instruction_t pop_instruction(InstructionStack * instruction_stack) {
  --(instruction_stack->stack);
  const instruction_t element = *(instruction_stack->stack);
  return element;
}

#define BIT_SET_SIZE 2
#define BYTES 8

typedef unsigned char bitmask_t;

#define FULL_MASK (~((bitmask_t)0u))

typedef struct {
  bitmask_t bitset[BIT_SET_SIZE];
  int size;
} VariableSet;

static inline void print_set(VariableSet var_set) {
  for (int i = 0; i < BIT_SET_SIZE; ++i) {
    for (int j = 0; j < sizeof(bitmask_t) * BYTES; ++j) {
      int value = (var_set.bitset[i] & (1u << j)) ? 1 : 0;
      printf("%d ", value);
    }
    printf(" ");
  }
  printf("(size = %d)", var_set.size);
}

static inline int random_element(pcg32_random_t * rng, VariableSet var_set) {
  uint32_t u = pcg32_random_r(rng);
  uint32_t n = u % var_set.size;

  #ifdef SYMGEN_DEBUG
    printf("selecting %d (%d) from: ", n, u);
    print_set(var_set);
    printf("\n");
  #endif

  for (int i = 0; i < BIT_SET_SIZE; ++i) {
    for (int j = 0; j < sizeof(bitmask_t) * BYTES; ++j) {
      if (var_set.bitset[i] & (1u << j)) {
        if (n == 0) {

          #ifdef SYMGEN_DEBUG
            printf("selected %d\n", i * sizeof(bitmask_t) * BYTES + j);
          #endif

          return i * sizeof(bitmask_t) * BYTES + j;
        } else {
          --n;
        }
      }
    }
  }

  return -1;
}

static inline unsigned int set_size(bitmask_t * bitset) {
  int size = 0;

  for (int i = 0; i < BIT_SET_SIZE; ++i) {
    for (int j = 0; j < sizeof(bitmask_t) * BYTES; ++j) {
      if (bitset[i] & (1u << j)) {
        ++size;
      }
    }
  }

  return size;
}

static inline VariableSet random_subset(pcg32_random_t * rng, VariableSet var_set) {
  if (var_set.size == 0) {
    return var_set;
  }

  VariableSet result;

  for (int i = 0; i < BIT_SET_SIZE; ++i) {
    const bitmask_t selection_mask = (bitmask_t) (pcg32_random_r(rng) % FULL_MASK);
    result.bitset[i] = var_set.bitset[i] & selection_mask;
  }

  result.size = set_size(result.bitset);

  return result;
}

static inline VariableSet random_subset_non_empty(pcg32_random_t * rng, VariableSet var_set) {
  if (var_set.size == 0) {
    return var_set;
  }

  VariableSet result;

  for (int i = 0; i < BIT_SET_SIZE; ++i) {
    const bitmask_t selection_mask = (bitmask_t) (pcg32_random_r(rng) % FULL_MASK);
    result.bitset[i] = var_set.bitset[i] & selection_mask;
  }

  result.size = set_size(result.bitset);

  if (result.size == 0) {
    // rejecting sampling: there is 0.5 chance it succeeds
    // setting a random element to 1 artificially lifts the prob of size 1 sets
    return random_subset_non_empty(rng, var_set);
  }

  return result;
}

static inline VariableSet random_subset_p(pcg32_random_t * rng, VariableSet var_set, double p) {
  if (var_set.size == 0) {
    return var_set;
  }

  VariableSet result;

  for (int i = 0; i < BIT_SET_SIZE; ++i) {
    result.bitset[i] = 0u;

    for (int j = 0; j < sizeof(bitmask_t) * BYTES; ++j) {
    const bitmask_t offset = (1u << j);
      /// saving rng calls
      if (var_set.bitset[i] & offset) {
        const double u = uniform(rng);
        result.bitset[i] |= (u < p) ? offset : 0u;
      }
    }
  }

  result.size = set_size(result.bitset);

  return result;
}

static inline VariableSet random_subset_p_non_empty(pcg32_random_t * rng, VariableSet var_set, double p) {
  if (var_set.size == 0) {
    return var_set;
  }

  VariableSet result;

  for (int i = 0; i < BIT_SET_SIZE; ++i) {
    result.bitset[i] = 0u;

    for (int j = 0; j < sizeof(bitmask_t) * BYTES; ++j) {
    const bitmask_t offset = (1u << j);
      /// saving rng calls
      if (var_set.bitset[i] & offset) {
        const double u = uniform(rng);
        result.bitset[i] |= (u < p) ? offset : 0u;
      }
    }
  }

  result.size = set_size(result.bitset);

  if (result.size == 0) {
    if (p > 0.25) {
      return random_subset_p_non_empty(rng, var_set, p);
    }

    /// forcing one bit because rejection sampling might take a lot of iterations
    uint32_t n = pcg32_random_r(rng) % var_set.size;
    for (int i = 0; i < BIT_SET_SIZE; ++i) {
      for (int j = 0; j < sizeof(bitmask_t) * BYTES; ++j) {
        const bitmask_t offset = (1u << j);
        if (var_set.bitset[i] & offset) {
          if (n == 0) {
            result.bitset[i] |= offset;
            result.size = 1;
            return result;
          } else {
            --n;
          }
        }
      }
    }
  }

  return result;
}

static inline VariableSet variable_set_full(int n) {
  VariableSet var_set;

  int i;
  for (i = 0; i < n / (sizeof(bitmask_t) * BYTES); ++i) {
    var_set.bitset[i] = FULL_MASK;
  }
  const int offset = n % (sizeof(bitmask_t) * BYTES);
  var_set.bitset[i] = ~(FULL_MASK << offset);

  for (++i; i < BIT_SET_SIZE; ++i) {
    var_set.bitset[i] = 0;
  }
  var_set.size = n;

  return var_set;
}

// defines
${DEFINES}

#define RANDOM_SUBSET_P(prob) (random_subset_p(rng, input_set, prob))
#define RANDOM_SUBSET (random_subset(rng, input_set))
#define RANDOM_INPUT (random_element(rng, input_set))

${DECLARATIONS}

${DEFINITIONS}

static PyObject * expr_gen(PyObject *self, PyObject *args) {
  PyObject *py_seed_1 = NULL;
  PyObject *py_seed_2 = NULL;
  PyObject *py_instructions = NULL;
  PyObject *py_instruction_sizes = NULL;
  PyObject *py_max_inputs = NULL;
  PyObject *py_max_depth = NULL;

  if (!PyArg_UnpackTuple(
    args, "expr_gen", 6, 6,
    &py_seed_1, &py_seed_2, &py_instructions, &py_instruction_sizes,
    &py_max_inputs, &py_max_depth
  )) {
    return NULL;
  }

  if (!PyLong_Check(py_seed_1)) {
    PyErr_SetString(PyExc_TypeError, "seed_1 must be an integer.");
    return NULL;
  }
  if (!PyLong_Check(py_seed_2)) {
    PyErr_SetString(PyExc_TypeError, "seed_2 must be an integer.");
    return NULL;
  }
  const unsigned long seed_1 = PyLong_AsLong(py_seed_1);
  const unsigned long seed_2 = PyLong_AsLong(py_seed_2);

  pcg32_random_t rng;
  pcg32_srandom(&rng, seed_1, seed_2);

  if (!PyArray_Check(py_instructions)) {
    PyErr_SetString(PyExc_TypeError, "instructions must be a numpy array.");
    return NULL;
  }
  if (!PyArray_Check(py_instruction_sizes)) {
    PyErr_SetString(PyExc_TypeError, "instruction_sizes must be a numpy array.");
    return NULL;
  }
  if (!PyLong_Check(py_max_inputs)) {
    PyErr_SetString(PyExc_TypeError, "max_inputs must be an integer.");
    return NULL;
  }

  if (!PyLong_Check(py_max_depth)) {
    PyErr_SetString(PyExc_TypeError, "max_depth must be an integer.");
    return NULL;
  }

  const PyArrayObject * instructions_array = (PyArrayObject *) py_instructions;
  const PyArrayObject * instruction_sizes_array = (PyArrayObject *) py_instruction_sizes;
  const long max_inputs = PyLong_AsLong(py_max_inputs);
  const long max_depth =  PyLong_AsLong(py_max_depth);

  if (max_inputs > sizeof(bitmask_t) * BYTES * BIT_SET_SIZE) {
    PyErr_SetString(
      PyExc_TypeError, "Max inputs exceeds maximal number of inputs. Recompile the machine with a larger limit."
    );
    return NULL;
  }

  if (!(
    PyArray_IS_C_CONTIGUOUS(instructions_array) &&
    PyArray_TYPE(instructions_array) == INT_T &&
    PyArray_NDIM(instructions_array) == 2 &&
    PyArray_DIM(instructions_array, 1) == 2
  )) {
    PyErr_SetString(PyExc_TypeError, "Instructions must be a (total_instructions, 2) int32 array with C order.");
    return NULL;
  }
  const npy_intp max_instructions = PyArray_DIM(instructions_array, 0);

  if (!(
    PyArray_IS_C_CONTIGUOUS(instruction_sizes_array) &&
    (PyArray_TYPE(instruction_sizes_array) == INT_T) &&
    (PyArray_NDIM(instruction_sizes_array) == 1)
  )) {
    PyErr_SetString(PyExc_TypeError, "Instruction sizes must be a (n_batch, ) int32 array.");
    return NULL;
  }

  const npy_intp max_expressions = PyArray_DIM(instruction_sizes_array, 0);

  instruction_t * instructions = (instruction_t *) PyArray_DATA(instructions_array);
  instruction_t * instructions_end = instructions + 2 * max_instructions;

  int_t * instruction_sizes = (int_t *) PyArray_DATA(instruction_sizes_array);
  int_t * instruction_sizes_end = instruction_sizes + 2 * max_expressions;

  const npy_intp instructions_stride_0 = PyArray_STRIDE(instructions_array, 0) / sizeof(int_t);

  //Py_BEGIN_ALLOW_THREADS

  InstructionStack instruction_stack = (InstructionStack) {
    .stack=instructions, .empty=instructions, .full=instructions_end
  };
  VariableSet input_set = variable_set_full(max_inputs);

  int expression_index = 0;
  while (instruction_stack.stack < instruction_stack.full && expression_index < max_expressions) {
    const instruction_t * beginning = instruction_stack.stack;
    const int status = ${SEED_SYMBOL};

    //Py_END_ALLOW_THREADS

    if (status != STATUS_OK) {
      switch (status) {
        case ERROR_STACK:
          DEBUG_PRINT("Maximal stack size reached\n");
          instruction_stack.stack = beginning;
          break;

        case ERROR_MAX_DEPTH:
          DEBUG_PRINT("Maximal expression depth reached\n");
          instruction_stack.stack = beginning;
          continue;

        case ERROR_UNPROCESSED_CONDITION:
          PyErr_SetString(PyExc_TypeError, "The state didn't satisfy any conditions.");
          return NULL;

        default:
          PyErr_SetString(PyExc_TypeError, "Generation failed: unknown reason.");
          return NULL;
      }
    } else {
      instruction_sizes[expression_index] = (int) (instruction_stack.stack - beginning);
      ++expression_index;
    }
  }

  const int generated = expression_index;

  for(; expression_index < max_expressions; ++expression_index) {
    instruction_sizes[expression_index] = 0;
  }

  return PyLong_FromLong(generated);
}

static PyObject * hash(PyObject *self, PyObject *args) {
  if (!PyArg_UnpackTuple(args, "hash", 0, 0)) {
    return NULL;
  }

  PyObject *py_str = PyUnicode_FromString(${HASH});
  return py_str;
}

static PyObject * debug_on(PyObject *self, PyObject *args) {
  if (!PyArg_UnpackTuple(args, "debug_on", 0, 0)) {
    return NULL;
  }

#ifdef DEBUG
  return PyBool_FromLong(1);
#else
  return PyBool_FromLong(0);
#endif
}

static PyMethodDef SymGenMethods[] = {
    {"expr_gen",  expr_gen, METH_VARARGS, "generates random expression."},
    {"hash",  hash, METH_VARARGS, "returns the machine hash."},
    {"debug_on",  debug_on, METH_VARARGS, "returns True if debug is on."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef sym_gen_module = {
    PyModuleDef_HEAD_INIT,
    "sym_gen",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SymGenMethods
};

PyMODINIT_FUNC
PyInit_sym_gen(void)
{
    PyObject *m;
    m = PyModule_Create(&sym_gen_module);
    if (m == NULL) return NULL;

    import_array();  // Initialize the NumPy API

    return m;
}