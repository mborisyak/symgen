#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// debug on / off
${DEBUG}

#ifdef SYMGEN_DEBUG
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

#define BIT_SET_SIZE ${BIT_SET_SIZE}
#define BITS 8

#define MEMORY_LIMIT ${MEMORY_LIMIT}

#include "symgen.h"
#include "pcg32.h"
#include "bitset.h"

#include <Python.h>
#include "numpy/arrayobject.h"

#define STATUS_OK 0
#define ERROR_STACK 1
#define ERROR_MAX_DEPTH 2
#define ERROR_UNPROCESSED_CONDITION 3

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

// defines
${DEFINES}

#define RANDOM_INPUT() (bitset_random_element(rng, domain))

#define DOMAIN_RANDOM_SUBSET() (bitset_random_subset(rng, domain))
#define DOMAIN_RANDOM_SUBSET_P(prob) (bitset_random_subset_p(rng, domain, prob))
#define DOMAIN_RANDOM_CHOICE(n) (bitset_random_choice(rng, domain, n))
#define DOMAIN_RANDOM_FRACTION(fraction) (bitset_random_choice_fraction(rng, domain, fraction))

#define RANGE(start, end) (bitset_range(start, end))

#define RANDOM_ELEMENT(bitset) (bitset_random_element(rng, bitset))

#define RANDOM_SUBSET(bitset) (bitset_random_subset(rng, bitset))
#define RANDOM_SUBSET_P(bitset, prob) (bitset_random_subset_p(rng, bitset, prob))
#define RANDOM_CHOICE(bitset, n) (bitset_random_choice(rng, bitset, n))
#define RANDOM_FRACTION(bitset, fraction) (bitset_random_choice_fraction(rng, bitset, fraction))

#define RANDOM_NORMAL() (pcg32_normal(rng))

static inline unsigned int random_memory(pcg32_random_t * rng, const unsigned int * allocated) {
  const unsigned int limit = *allocated;
  if (limit == 0) {
    DEBUG_PRINT("ERROR: accessing empty memory");
    return 0;
  }

  return pcg32_random_r(rng) % limit;
}

static inline unsigned int random_memory_range(
  pcg32_random_t * rng, const unsigned int start, const unsigned int end
) {
  const unsigned int limit = end - start;
  if (limit <= 0) {
    DEBUG_PRINT("ERROR: empty range for random memory");
    return start;
  }

  return start + (pcg32_random_r(rng) % limit);
}

#define ALLOCATE() (allocate(allocated))
#define RANDOM_MEMORY() (random_memory(rng, allocated))
#define RANDOM_MEMORY_RANGE(start, end) (random_memory_range(rng, start, end))

${DECLARATIONS}

${DEFINITIONS}

static PyObject * expr_gen(PyObject *self, PyObject *args) {
  PyObject *py_seed_1 = NULL;
  PyObject *py_seed_2 = NULL;
  PyObject *py_instructions = NULL;
  PyObject *py_instruction_sizes = NULL;
  PyObject *py_max_inputs = NULL;
  PyObject *py_max_depth = NULL;
  PyObject *py_max_expression_length = NULL;

  if (!PyArg_UnpackTuple(
    args, "expr_gen", 7, 7,
    &py_seed_1, &py_seed_2, &py_instructions, &py_instruction_sizes,
    &py_max_inputs, &py_max_depth, &py_max_expression_length
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

  if (!PyLong_Check(py_max_expression_length)) {
    PyErr_SetString(PyExc_TypeError, "max_expression_length must be an integer.");
    return NULL;
  }

  const PyArrayObject * instructions_array = (PyArrayObject *) py_instructions;
  const PyArrayObject * instruction_sizes_array = (PyArrayObject *) py_instruction_sizes;
  const long max_inputs = PyLong_AsLong(py_max_inputs);
  const long max_depth =  PyLong_AsLong(py_max_depth);
  const long max_expression_length = PyLong_AsLong(py_max_expression_length);

  if (max_inputs > sizeof(bitmask_t) * BITS * BIT_SET_SIZE) {
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

  int expression_index = 0;
  int instruction_index = 0;
  const BitSet all_variables = bitset_full(max_inputs);

  while ((instruction_index + max_expression_length <= max_instructions) && (expression_index < max_expressions)) {
    InstructionStack instruction_stack = (InstructionStack) {
      .stack=instructions + instruction_index,
      .empty=instructions + instruction_index,
      .full=instructions + instruction_index + max_expression_length
    };

    const int status = ${SEED_SYMBOL};

    //Py_END_ALLOW_THREADS

    if (status != STATUS_OK) {
      switch (status) {
        case ERROR_STACK:
          PyErr_SetString(PyExc_ValueError, "Maximal stack size.");
          return NULL;

        case ERROR_MAX_DEPTH:
          PyErr_SetString(PyExc_ValueError, "Maximal depth is reached.");
          return NULL;

        case ERROR_UNPROCESSED_CONDITION:
          PyErr_SetString(PyExc_ValueError, "The state didn't satisfy any conditions.");
          return NULL;

        default:
          PyErr_SetString(PyExc_ValueError, "Generation failed: unknown reason.");
          return NULL;
      }
    }

    const unsigned int generated_instructions = (unsigned int) (instruction_stack.stack - instruction_stack.empty);
    DEBUG_PRINT("Generated %d instructions (expression %d)\n", generated_instructions, expression_index);

    instruction_sizes[expression_index] = generated_instructions;
    instruction_index += generated_instructions;
    ++expression_index;
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