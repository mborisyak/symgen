import numpy as np

from symgen.dag import RandomTopology
from symgen import lower

def test_random_topology_acyclic():
  rng = np.random.default_rng(0)
  gen = RandomTopology(n_nodes=12, max_inputs=4)

  for _ in range(50):
    n_in, n_out = 3, 2
    graph = gen(rng, n_in, n_out)

    assert len(graph) == n_in + 12 + n_out

    for i in range(n_in):
      assert graph[i] is None

    for cell in range(n_in, len(graph)):
      links, context = graph[cell]
      assert context == {}                                # RandomTopology passes no context
      assert 1 <= len(links) <= 4
      assert len(set(links)) == len(links)               # distinct

      if cell < n_in + 12:                                # internal node
        assert all(g < cell for g in links)               # only earlier cells
      else:                                               # output node (trailing n_out cells)
        assert all(g < n_in + 12 for g in links)          # no output->output edges

def test_lower_rewires_and_stores():
  """Local loads are rewired to global cells and each node stores into its own cell."""
  graph = [
    None, None,                                            # inputs (cells 0, 1)
    ((1, 0), {}),                                          # cell 2
    ((2,), {}),                                            # cell 3 (output)
  ]
  exprs = {
    2: [('load', 0), ('load', 1), ('add',)],
    3: [('load', 0)],
  }
  program = lower(graph, exprs, n_out=1)

  assert program == [
    ('load', 1), ('load', 0), ('add',), ('store', 2),   # node 2 + store cell 2
    ('load', 2), ('store', 3),                          # node 3 + store cell 3
    ('load', 3),                                        # output
  ]

def test_lower_executes_on_machine():
  """A single add node lowers to a program the machine runs correctly."""
  import symgen
  machine = symgen.StackMachine(symgen.lib.core, symgen.lib.std)

  graph = [None, None, ((0, 1), {})]
  exprs = {2: [('load', 0), ('load', 1), ('add',)]}
  program = lower(graph, exprs, n_out=1)

  result = machine(program, np.array([3.0, 4.0]))
  assert abs(float(result[0]) - 7.0) < 1.0e-6

def test_lower_compacts_and_rewires():
  """Compaction renumbers live nodes to a contiguous range, skipping pruned dead cells."""
  import symgen
  graph = [
    None, None,                 # inputs 0, 1
    ((0, 1), {}),               # cell 2: in0 + in1
    ((2,), {}),                 # cell 3: DEAD (absent from expressions)
    ((1,), {}),                 # cell 4: DEAD
    ((0, 2), {}),               # cell 5: output, references inputs/earlier nodes (in0, cell 2)
  ]
  exprs = {
    2: [('load', 0), ('load', 1), ('add',)],   # local 0->cell0, local 1->cell1
    5: [('load', 0), ('load', 1), ('mul',)],   # local 0->links[0]=cell0, local 1->links[1]=cell2
  }
  program = lower(graph, exprs, n_out=1)

  assert program == [
    ('load', 0), ('load', 1), ('add',), ('store', 2),   # node 2 -> cell 2
    ('load', 0), ('load', 2), ('mul',), ('store', 3),   # node 5 -> cell 3; its load of cell 2 stays 2
    ('load', 3),                                         # output (cell 5 -> 3)
  ]

  used = sorted({args[0] for op, *args in program if op in ('load', 'store')})
  assert used == [0, 1, 2, 3]

  machine = symgen.StackMachine(symgen.lib.core, symgen.lib.std)
  out = machine(program, np.array([[2.0], [3.0]]))
  assert abs(float(out[0, 0]) - (2.0 * (2.0 + 3.0))) < 1.0e-6

def _eval_dag(graph, exprs, inputs, machine):
  """Independent oracle: evaluate the DAG directly (each node on its links' values), bypassing `lower`."""
  n_in = next((i for i, node in enumerate(graph) if node is not None), len(graph))
  values = {i: np.asarray(inputs[i]) for i in range(n_in)}
  for cell in sorted(exprs):
    links, _ = graph[cell]
    local = np.stack([values[g] for g in links]) if len(links) > 0 else np.empty((0, inputs.shape[1]))
    values[cell] = np.asarray(machine(exprs[cell], local))[0]
  return values

def test_lower_matches_dag_semantics():
  """The lowered, compacted program computes exactly what the DAG means (checked against the oracle)."""
  import symgen
  machine = symgen.StackMachine(symgen.lib.core, symgen.lib.std)
  rng = np.random.default_rng(0)

  graph = [
    None, None,
    ((0, 1), {}),        # cell 2
    ((0,), {}),          # cell 3: DEAD
    ((2, 0), {}),        # cell 4
    ((4, 2), {}),        # cell 5 (output)
  ]
  exprs = {
    2: [('load', 0), ('load', 1), ('add',)],
    4: [('load', 0), ('load', 1), ('mul',)],
    5: [('load', 0), ('load', 1), ('sub',)],
  }
  n_out = 1
  program = lower(graph, exprs, n_out)

  for _ in range(20):
    inputs = rng.normal(size=(2, 32))
    expected = _eval_dag(graph, exprs, inputs, machine)            # oracle (no lower)
    got = np.asarray(machine(program, inputs))                    # lowered + compacted program
    output_cell = len(graph) - n_out                             # = 5
    assert np.allclose(got[0], expected[output_cell], atol=1.0e-5)

def test_lower_prunes_unused_input_only_gap():
  """An unreferenced input cell stays allocated (seeded positionally) and is the only memory gap."""
  import symgen
  graph = [None, None, ((0,), {})]
  exprs = {2: [('load', 0), ('load', 0), ('mul',)]}   # in0 * in0
  program = lower(graph, exprs, n_out=1)

  assert program == [
    ('load', 0), ('load', 0), ('mul',), ('store', 2),
    ('load', 2),
  ]
  machine = symgen.StackMachine(symgen.lib.core, symgen.lib.std)
  out = machine(program, np.array([[3.0], [99.0]]))   # input 1 (=99) is unused
  assert abs(float(out[0, 0]) - 9.0) < 1.0e-6
