# Example output

A complete run of the reference instance
(`KnapsackProblem.random(15, seed=38)`, capacity 68, DP optimum 272) on
`ibm_kingston`, with `prune_keep=0.5` and `STEPS=3`.

The code that produces it:

```python
from qiskit_qaoa_knapsack_domi import KnapsackProblem, RunConfig, solve

problem = KnapsackProblem.random(15, seed=38)
session = solve(problem, config=RunConfig(dry_run=False, prune_keep=0.5))
```

Or step by step — note that `build_result` is what prints the summary block;
without it nothing does.

```python
counts = run_circuit(tr.circuit, backend, cfg, pen)
result = build_result(counts, problem, backend=backend, transpiled=tr)
```

---

## Job monitoring

Printed by `run_circuit` while the job is on the device. This only appears on
real hardware — in `dry_run` there is no job to follow.

```
Job ID: daasu31l216s739onhrg
 Job ID   : daasu31l216s739onhrg
 Backend  : ibm_kingston
 Started  : 20:59:22
----------------------------------------------------------------
  [→] QUEUED          : 20:59:22
  [←] QUEUED          :    2.3s
  [→] RUNNING         : 20:59:24
  [←] RUNNING         :  112.3s
  [→] DONE            : 21:01:17
  [←] DONE            :    0.0s
----------------------------------------------------------------
 TIME BREAKDOWN:
   QUEUED         :    2.3s (  2.0%)
   RUNNING        :  112.3s ( 97.5%)
   DONE           :    0.0s (  0.0%)

   Billed usage   :   94.0s
   TOTAL WALL     :  115.2s
================================================================
```

`Billed usage` is only reported on Pay-As-You-Go plans; on other plans that line
is silently skipped. The job ID appears twice because `run_circuit` prints it as
soon as the job is submitted, and `monitor_job` repeats it in its header.

---

## Result summary

Printed by `build_result` (and by `solve`, which calls it internally).

```
================================================================
                    KNAPSACK  -  RESULT SUMMARY
================================================================
 Backend : ibm_kingston             Logical qubits : 15
 Method  : digitized quantum annealing (no optimizer)
 Layout  : seed-scan   |   2q gates : 766
----------------------------------------------------------------
 Selected items
----------------------------------------------------------------
     Item    Value   Weight
   ------ -------- --------
        1       31       12
        2       32       11
        3       53        9
        4       51       11
        9       34       13
       10       28        8
       12       43        3
   ------ -------- --------
    TOTAL      272       67    (capacity 68)
----------------------------------------------------------------
 Solution quality
----------------------------------------------------------------
   Method                   Value   % of optimum
   ---------------------- ------- --------------
   Quantum (hardware)         272         100.0%
   Exact (dynamic prog.)      272              -
================================================================
```

The hardware found the exact optimum: 272 at weight 67, one unit under the
capacity.

`Logical qubits : 15` is the number of problem qubits, not the size of the chip
— `ibm_kingston` has considerably more. `2q gates : 766` is the count after
transpilation: 55 kept logical RZZ × 3 steps, expanded to roughly 4.6 physical
two-qubit gates per logical RZZ by routing onto heavy-hex.

---

## Reading the result

Everything in the table is available on the returned object:

```python
result.value        # 272
result.optimum      # 272
result.is_optimal   # True
result.ratio        # 1.0
result.selection    # [1, 2, 3, 4, 9, 10, 12]
result.weight       # 67
result.twoq         # 766
```

Sampling statistics come from `build_distribution`:

```python
dist = build_distribution(counts, problem, good=0.90)
# OPT = 272   optimal bitstrings: 1
# P(feasible) = 0.1260   P(optimum)  = ...   P(within 10%) = ...
```

`P(feasible)` of 12.6 % is below the 30 % that would be comfortable — a
consequence of the pruning and the shallow schedule. The optimum was still
recovered because the selection step is classical: among all sampled bitstrings,
the best feasible one is kept. See [theory.md](theory.md) for what the penalty
coefficients do to that fraction.
