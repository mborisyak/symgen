#define PY_SSIZE_T_CLEAN

#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#ifdef DEBUG
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

#define MAXIMAL_STACK_SIZE 1024

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

#define NONE 0

static inline number_t pop(number_t ** stack) {
  --(*stack);
  const number_t element = **stack;
  DEBUG_PRINT("[DEBUG] pop %.1f\n", element);
  return element;
}

static inline void push(number_t ** stack, number_t element) {
  **stack = element;
  DEBUG_PRINT("[DEBUG] push %.1f\n", element);
  (*stack)++;
}

// ${DEFINES}

const char * COMMAND_NAMES[] = {${COMMAND_NAMES}};

// ${COMMANDS}


static PyObject * stack_eval(PyObject *self, PyObject *args) {
  PyObject *py_instructions = NULL;
  PyObject *py_instruction_sizes = NULL;
  PyObject *py_inputs = NULL;
  PyObject *py_outputs = NULL;

  if (!PyArg_UnpackTuple(
    args, "stack_eval", 4, 4,
    &py_instructions, &py_instruction_sizes, &py_inputs, &py_outputs
  )) {
    return NULL;
  }

  if (!PyArray_Check(py_instructions)) {
    PyErr_SetString(PyExc_TypeError, "instructions must be a numpy array.");
    return NULL;
  }
  if (!PyArray_Check(py_instruction_sizes)) {
    PyErr_SetString(PyExc_TypeError, "instruction_sizes must be a numpy array.");
    return NULL;
  }
  if (!PyArray_Check(py_inputs)) {
    PyErr_SetString(PyExc_TypeError, "instructions must be a numpy array.");
    return NULL;
  }
  if (!PyArray_Check(py_outputs)) {
    PyErr_SetString(PyExc_TypeError, "outputs must be a numpy array.");
    return NULL;
  }

  const PyArrayObject * instructions_array = (PyArrayObject *) py_instructions;
  const PyArrayObject * instruction_sizes_array = (PyArrayObject *) py_instruction_sizes;
  const PyArrayObject * inputs_array = (PyArrayObject *) py_inputs;
  const PyArrayObject * outputs_array = (PyArrayObject *) py_outputs;

  if (!(
    PyArray_IS_C_CONTIGUOUS(instructions_array) &&
    PyArray_TYPE(instructions_array) == INT_T &&
    PyArray_NDIM(instructions_array) == 2 &&
    PyArray_DIM(instructions_array, 1) == 2
  )) {
    PyErr_SetString(PyExc_TypeError, "Instructions must be a (total_instructions, 2) int32 array with C order.");
    return NULL;
  }
  const npy_intp n_instructions = PyArray_DIM(instructions_array, 0);

  if (!(
    PyArray_IS_C_CONTIGUOUS(instruction_sizes_array) &&
    PyArray_TYPE(instruction_sizes_array) == INT_T &&
    PyArray_NDIM(instruction_sizes_array) == 1
  )) {
    PyErr_SetString(PyExc_TypeError, "Instruction sizes must be a (n_batch, ) int32 array.");
    return NULL;
  }

  const npy_intp n_batch = PyArray_DIM(instruction_sizes_array, 0);

  if (!(
    PyArray_IS_C_CONTIGUOUS(inputs_array) &&
    PyArray_TYPE(inputs_array) == NUMBER_T &&
    PyArray_NDIM(inputs_array) == 3 &&
    PyArray_DIM(inputs_array, 0) == n_batch
  )) {
    PyErr_SetString(PyExc_TypeError, "Inputs must be a (n_batch, n_samples, input dim) float32 array.");
    return NULL;
  }
  const npy_intp n_samples = PyArray_DIM(inputs_array, 1);

  if (!(
    PyArray_IS_C_CONTIGUOUS(outputs_array) &&
    PyArray_TYPE(outputs_array) == NUMBER_T &&
    PyArray_NDIM(outputs_array) == 3 &&
    PyArray_DIM(outputs_array, 0) == n_batch &&
    PyArray_DIM(outputs_array, 1) == n_samples
  )) {
    PyErr_SetString(PyExc_TypeError, "Outputs must be a (n_batch, n_samples, output dim) float32 array.");
    return NULL;
  }
  const npy_intp output_dim = PyArray_DIM(outputs_array, 2);

  const instruction_t * instructions = (instruction_t *) PyArray_DATA(instructions_array);
  const int_t * instruction_sizes = (int_t *) PyArray_DATA(instruction_sizes_array);
  const number_t * inputs = (number_t *) PyArray_DATA(inputs_array);
  number_t * outputs = (number_t *) PyArray_DATA(outputs_array);

  const npy_intp instructions_stride_0 = PyArray_STRIDE(instructions_array, 0) / sizeof(int_t);

  const npy_intp inputs_stride_0 = PyArray_STRIDE(instructions_array, 0) / sizeof(number_t);
  const npy_intp inputs_stride_1 = PyArray_STRIDE(instructions_array, 1) / sizeof(number_t);
  const npy_intp inputs_stride_2 = PyArray_STRIDE(instructions_array, 2) / sizeof(number_t);

  const npy_intp outputs_stride_0 = PyArray_STRIDE(outputs_array, 0) / sizeof(number_t);
  const npy_intp outputs_stride_1 = PyArray_STRIDE(outputs_array, 1) / sizeof(number_t);
  const npy_intp outputs_stride_2 = PyArray_STRIDE(outputs_array, 2) / sizeof(number_t);

  Py_BEGIN_ALLOW_THREADS

  unsigned long long instruction_index = 0;

  number_t empty_stack[MAXIMAL_STACK_SIZE];
  number_t * stack;
  number_t memory[MAXIMAL_STACK_SIZE];

  for (unsigned int i = 0; i < n_batch; ++i) {
    const unsigned int program_end = instruction_index + instruction_sizes[i];

    for (unsigned int j = 0; j < n_samples; ++j) {
      stack = empty_stack;
      const number_t * expression_input = inputs + i * inputs_stride_0 + j * inputs_stride_1;

      for (
        unsigned int current_instruction_index = instruction_index;
        current_instruction_index < program_end;
        ++current_instruction_index
      ) {
        const instruction_t instruction = instructions[current_instruction_index];
        DEBUG_PRINT(
          "[DEBUG] executing %s[%d] (%d[%d]);\n",
          COMMAND_NAMES[instruction.command], instruction.argument.integer,
          instruction.command, instruction.argument.integer
        );

        switch (instruction.command) {
          // ${COMMAND_SWITCH}

          default:
            PyErr_SetString(PyExc_TypeError, "Unknown command!");
            return NULL;
        }

        #ifdef DEBUG
        DEBUG_PRINT("[DEBUG] stack: [");
        for (number_t * stack_tr = empty_stack; stack_tr < stack; ++stack_tr) {
          DEBUG_PRINT("%.1f ", *stack_tr);
        }
        DEBUG_PRINT("]\n");
        #endif
      }
      number_t * stack_tr = empty_stack;
      unsigned int k = 0;
      while (stack_tr < stack && k < output_dim) {
        outputs[i * outputs_stride_0 + j * outputs_stride_1 + k * outputs_stride_2] = *stack_tr;
        stack_tr++;
        k++;
      }
    }
    instruction_index = program_end;
  }

  Py_END_ALLOW_THREADS

  return PyLong_FromLong(0);
}

static PyMethodDef SymGenMethods[] = {
    {"stack_eval",  stack_eval, METH_VARARGS, "evaluates expressions."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef sym_eval_module = {
    PyModuleDef_HEAD_INIT,
    "sym_eval",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SymGenMethods
};

PyMODINIT_FUNC
PyInit_sym_eval(void)
{
    PyObject *m;
    m = PyModule_Create(&sym_eval_module);
    if (m == NULL) return NULL;

    import_array();  // Initialize the NumPy API

    return m;
}