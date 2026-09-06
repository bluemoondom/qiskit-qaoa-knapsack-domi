# Theory

Background for [qiskit-qaoa-knapsack-domi](../README.md): how the knapsack is
encoded as a QUBO, converted to an Ising Hamiltonian and solved by digitized
quantum annealing, and what the Lagrange multipliers `LAM1` and `LAM2` actually
do.

The maths renders on GitHub. All numbers quoted here come from the reference
instance `KnapsackProblem.random(15, seed=38)` run on `ibm_kingston` — see
[example_output.md](example_output.md) for that run's full log.

---

# Knapsack problem as a QUBO → Ising → quantum annealing

## 1. The optimization problem

With item values $v_i$ (`values`), weights $w_i$ (`weights`) and capacity $C$ (`CAPACITY`):

$$
\max_{x\in\{0,1\}^N}\ \sum_{i=0}^{N-1} v_i\,x_i
\qquad\text{subject to}\qquad
\sum_{i=0}^{N-1} w_i\,x_i \;\le\; C .
$$

A quantum annealer minimizes energy, so we flip the sign of the objective and minimize $-\sum_i v_i x_i$.

The reference instance (`KnapsackProblem.random(15, seed=38)`) has $C=68$ and DP optimum $272$.

---

## 2. Two ways to encode the inequality — and which one runs

A QUBO can only encode **equalities** or **soft penalties**. Both encodings are described below; **only the second one is implemented in this package**.

### (a) Slack variables — present but commented out

Turn "$\le$" into "$=$" by adding a slack $S\in\{0,\dots,C\}$ stored in a bounded log-encoding,

$$
\sum_i w_i x_i + S = C,\qquad
S=\sum_{k=0}^{K-1} a_k\,s_k,\quad
a_k = 2^{k}\ (k<K-1),\quad
a_{K-1}=C-\bigl(2^{K-1}-1\bigr),\quad K=\lfloor\log_2 C\rfloor+1,
$$

with a single penalty weight $P=\max_i v_i+1$. Correctness is then **guaranteed by construction** — no tuning needed — at the cost of $K\approx\log_2 C$ extra qubits ($K=7$ here, so $n=22$).

This route is not implemented in the package. It is the right one for large $N$, where the full $2^N$ verification used below is not available.

### (b) Soft penalty — **this is what runs**

No slack, no extra qubits: $n=N=15$. The constraint is folded into the objective as an augmented Lagrangian,

$$
E(x) \;=\; -\alpha\,V(x) \;+\; \lambda_1\,W(x) \;+\; \lambda_2\,\bigl(W(x)-C\bigr)^2,
\qquad V=\sum_i v_i x_i,\quad W=\sum_i w_i x_i,
$$

with `ALPHA`, `LAM1`, `LAM2` chosen by `tune_penalties()`. Feasibility is **not** guaranteed here — it depends entirely on those three coefficients, which is why `check_ground_state()` verifies that the minimum of $E$ is feasible before the circuit is built.

Values used in the run: `ALPHA = 2.0`, `LAM1 = 6.20`, `LAM2 = 0.179`.

---

## 3. The QUBO coefficients

Using $x_i^2=x_i$ and dropping the constant $\lambda_2 C^2$:

$$
\text{lin}_i = -\alpha v_i + \lambda_1 w_i + \lambda_2\bigl(w_i^2 - 2C w_i\bigr),
\qquad
\text{quad}_{ij} = 2\lambda_2\,w_i w_j \quad (i<j).
$$

The couplings are **all-to-all**: $\text{quad}_{ij}\propto w_iw_j$ is nonzero for every pair, so a full layer costs $N(N-1)/2 = 105$ RZZ gates.

---

## 4. Coupler pruning (`PRUNE_KEEP`) — a deliberate depth/exactness trade

The couplings are all-to-all, so the two-qubit gate count grows as
$\mathcal{O}(N^2\cdot\texttt{STEPS})$ and quickly exceeds what the device can hold
coherently. `PRUNE_KEEP` caps it by keeping only the largest-magnitude couplings:

```python
prune_keep = 0.5      # keeps 55 of 105 RZZ  ->  766 physical 2q gates
```

**The target is a gate budget, not a fraction.** With $\approx 4.6$ physical 2q gates
per logical RZZ on heavy-hex, a budget $G$ implies

$$
\texttt{PRUNE\_KEEP} \;\approx\; \min\!\left(1,\;
\frac{G}{4.6\cdot \tfrac{N(N-1)}{2}\cdot \texttt{STEPS}}\right)
$$

For $G=1200$, `STEPS = 3`: $N=12 \to 1.0$, $N=15 \to 0.83$, $N=18 \to 0.57$,
$N=20 \to 0.46$, $N=25 \to 0.29$.

### What pruning actually does to the sampled distribution

Pruning changes the cost function, so the argmin of the **pruned** model is generally not
the knapsack optimum. That turns out to be the wrong thing to measure. A shallow anneal
at $\Delta t = 2$ is nowhere near converged onto its ground state, so where the minimum
sits says little about where the samples land. What matters is the **rank of the optimum
in the sampled distribution**, because the selection step is classical — out of 300 000
shots, the best *feasible* bitstring is kept and verified against the constraint.

Ranks from a noiseless statevector simulation of the circuit that actually runs
(`seed 38`, `STEPS = 3`, `T = 6.0`, `ALPHA = 2.0`, `LAM1 = 6.20`, `LAM2 = 0.179`):

| `prune_keep` | RZZ kept | rank of optimum | $P(\text{opt})$ | vs uniform |
|---|---|---|---|---|
| 1.00 | 105 | 152 | 0.0770 % | 25× |
| **0.50** | **55** | **39** | **0.3089 %** | **101×** |
| 0.25 | 27 | 291 | 0.0432 % | 14× |
| 0.20 | 22 | 231 | 0.0535 % | 18× |
| 0.10 | 14 | 318 | 0.0212 % | 7× |

At `prune_keep = 0.5` the pruned circuit concentrates on the optimum **four times more
strongly** than the complete one — while using half the RZZ, and therefore decohering
less on hardware. Pruning here is not a compromise; on this instance it is an
improvement, and the measured hardware run recovered the exact optimum.

### But lower it with care — the good values are not a smooth range

The ranks above are not monotone in `prune_keep`, and the bad regions **move with the
penalties**. Rank of the optimum around the 0.5–0.8 band:

| `LAM1` | `LAM2` | 0.80 | 0.75 | 0.70 | 0.65 | 0.60 | 0.55 | 0.50 |
|---|---|---|---|---|---|---|---|---|
| 6.20 | 0.179 | 223 | 2620 | 2761 | 190 | 28 | 24 | 39 |
| 5.60 | 0.150 | 252 | 4011 | 1277 | 134 | 23 | 25 | 41 |
| 6.80 | 0.220 | 254 | 3370 | 2571 | 187 | 26 | 24 | 41 |
| 6.20 | 0.100 | 44 | 51 | 80 | 162 | **805** | 64 | 61 |
| 6.20 | 0.300 | 766 | 1089 | 178 | 60 | 16 | 54 | 101 |

No single value of `prune_keep` is inherently broken. At `LAM2 = 0.179` the collapse is at
0.70–0.75; at `LAM2 = 0.100` those are fine and 0.60 collapses instead. `LAM1` and `LAM2`
reshape the landscape, so they and `prune_keep` have to be judged together, never in
isolation.

### What a fine sweep shows

Sweeping `prune_keep` from 1.00 down to 0.10 in steps of 0.02, with penalties tuned
separately for each instance ($N=15$, `STEPS = 3`, `T = 6.0`):

| instance | behaviour | in top 300 |
|---|---|---|
| `seed 38` | flat ~150 down to 0.80, **collapse across 0.78–0.68** (ranks 987–9542), best 16–39 over 0.58–0.48, degrades below 0.44 | 29 / 46 |
| `seed 7` | flat 5–9 from 1.00 to 0.50, gradual decay, unusable below 0.26 | 38 / 46 |
| `seed 21` | rank 1 from 1.00 to 0.46, mild decay after | 46 / 46 |

Three things follow, none of them obvious in advance:

**The bad region, when it exists, is wide and contiguous** — on `seed 38` it spans 0.10 in
`prune_keep`, not a narrow spike. **Its location is instance-specific**, and on two of the
three instances it does not exist at all. And **quality degrades below roughly 0.3–0.4** on
every instance tested, which is the one robust boundary.

`prune_keep` is also quantized: it selects an integer number of couplings, so on this
instance 0.12 and 0.10 both leave 14 couplings and compile to the same circuit. Sweeping
more finely than about 1/105 buys nothing.

### A halving ladder does not help

The ladder 1.0 → 0.5 → 0.25 → 0.125 is a natural thing to reach for, but the data does not
support it:

| instance | 1.0 | 0.5 | 0.25 | 0.125 |
|---|---|---|---|---|
| `seed 38` | 158 | **39** | 284 | 309 |
| `seed 7` | 7 | **9** | 392 | 3775 |
| `seed 21` | 1 | **1** | 27 | 4 |

The third rung is already unusable on two of the three instances. Halving neither avoids
the collapse bands — it steps over `seed 38`'s by luck, not by design — nor tracks the
degradation boundary, which sits between the second and third rung and is not a power of
two. What the ladder does reliably is halve the gate count, and it happens that the
second rung, `prune_keep = 0.5`, is good on all three instances.

**So: 1.0 and 0.5 are both safe starting points, and 0.5 is the better one — it is at
least as good on all three instances and uses half the RZZ. Below 0.5, verify the
specific value; do not extrapolate.** The verification worth doing, when $N$ allows the
$2^N$ enumeration, is the rank of the optimum in the simulated distribution — not the
argmin of the pruned cost.

### One thing pruning is *not*

It is not merely deleting redundant gates. In `qubo_to_ising` every coupling contributes
to two linear biases as well,

```python
h[j] += -q / 4
h[l] += -q / 4
J[(j, l)] = q / 4
```

so dropping it removes those contributions too and changes the normalizer $s$. Measured
at `prune_keep = 0.5` on this instance: $\max_j|\Delta h_j| = 0.86$ in normalized units
where the maximum is 1.0 by construction, and $s$ moves from 176.7 to 157.0. The pruned
circuit implements a genuinely different Ising model — which is precisely why it can
sample *better* than the complete one, not merely cheaper.

`check_ground_state()` operates on the **unpruned** cost, so it validates the *tuning*,
not the *circuit*. That is still worth checking; it is just a different question.

### Two ways to buy back the gates without pruning

**(a) Mean-field absorption.** Instead of discarding a dropped coupling, fold its
first-order effect into the linear terms ($\langle x_j\rangle \approx \tfrac12$):

```python
else:
    lin[i] += q * 0.5
    lin[j] += q * 0.5          # instead of dropping q entirely
```

Free — no extra gates. Measured on this instance at `PRUNE_KEEP = 0.5`, this pulls the
pruned ground state from *value 328 / weight 94* back to *value 237 / weight 67*, i.e.
back **inside the capacity**. Still not the optimum, but the sampler is now concentrated
in the feasible region instead of outside it, which is exactly what post-selection wants.

**(b) A linear SWAP network instead of generic routing.** The odd–even transposition
network realizes *all* $N(N-1)/2$ couplings on a qubit path in $N$ layers, at ~3 two-qubit
gates per pair (the ZZ merges into the SWAP), versus the ~4.6 the transpiler currently
spends per *kept* pair:

| $N$ | pairs | SWAP-network 2q gates (`STEPS = 3`, **no pruning**) |
|---|---|---|
| 12 | 66 | 594 |
| 15 | 105 | **945** |
| 18 | 153 | 1377 |
| 20 | 190 | 1710 |

At $N=15$ that is the **complete** model for 945 gates — under the 1200 budget and cheaper
than today's pruned 766-gate circuit is per coupling. This is the option worth trying first;
pruning then only becomes necessary from $N\approx 18$ up.

---

## 5. QUBO → Ising Hamiltonian

Quantum hardware works with spins $Z_j\in\{+1,-1\}$, so substitute

$$
x_j=\frac{1-Z_j}{2}
\qquad\bigl(x_j=0\leftrightarrow Z_j=+1,\quad x_j=1\leftrightarrow Z_j=-1\bigr).
$$

Collecting terms gives

$$
H=\sum_j h_j\,Z_j+\sum_{j<l} J_{jl}\,Z_jZ_l+\text{const},
\qquad
h_j=-\tfrac12\,\text{lin}_j\;-\;\tfrac14\!\!\sum_{l\,:\,(j,l)\in\text{quad}}\!\!\text{quad}_{jl},
\qquad
J_{jl}=\tfrac14\,\text{quad}_{jl}.
$$

The constant only shifts all energies equally, so it is dropped.

**Normalization.** Every $h_j,J_{jl}$ is divided by $\text{scale}=\max(|h_j|,|J_{jl}|)$ so the problem energy scale is $\mathcal{O}(1)$, comparable to the mixer. Without this the penalty would dominate the mixer and the anneal would not be adiabatic. Decoding reads bits directly, so rescaling is harmless — but note it also means `ALPHA` is purely a scale factor: $(\alpha,\lambda_1,\lambda_2)$ and $(k\alpha,k\lambda_1,k\lambda_2)$ give a bit-identical circuit.

---

## 6. Digitized (Trotterized) quantum annealing — no optimizer

Two Hamiltonians:

$$
H_M=-\sum_j X_j \quad(\text{mixer, ground state }|+\rangle^{\otimes n}),
\qquad
H_P=\sum_j h_j Z_j+\sum_{j<l} J_{jl} Z_jZ_l \quad(\text{problem}).
$$

Start in $|+\rangle^{\otimes n}$ and interpolate linearly from mixer to problem, $H(s)=(1-s)H_M+sH_P$, discretized into `STEPS` Trotter steps with $\Delta t=T/\texttt{STEPS}$ and a **midpoint** schedule

$$
s_p=\frac{p+\tfrac12}{\texttt{STEPS}},\qquad p=0,\dots,\texttt{STEPS}-1,
$$

$$
U_p=\underbrace{e^{-i\,\Delta t\,s_p\,H_P}}_{\text{problem layer}}\;
     \underbrace{e^{-i\,\Delta t\,(1-s_p)\,H_M}}_{\text{mixer layer}} .
$$

The midpoint is deliberate. With the naive $s_p=p/\texttt{STEPS}$ the last mixer angle is zero, the final problem layer is diagonal and therefore invisible to a Z-basis measurement — at `STEPS = 1` the output distribution would be exactly uniform regardless of the coefficients.

There is **no classical optimizer** — the schedule is fixed, exactly like a D-Wave anneal. `STEPS` / `T` play the role of `num_sweeps` / anneal time. The run uses `STEPS = 3`, `T = 6.0`, i.e. $\Delta t = 2.0$.

---

## 7. Why the $R_{ZZ}$ (and $R_Z$) gates commute — and why that matters

Every term in $H_P$ is **diagonal** (built only from $Z$ and $Z\!\otimes\! Z$). Diagonal operators all commute:

$$
[\,Z_a,\,Z_bZ_c\,]=0,\qquad [\,Z_aZ_b,\,Z_cZ_d\,]=0 .
$$

Therefore the exponential of the *whole* problem layer factorizes **without any Trotter error**:

$$
e^{-i t H_P}=\prod_j e^{-i t\,h_j Z_j}\;\prod_{j<l} e^{-i t\,J_{jl} Z_jZ_l}
\qquad(\text{exact}).
$$

Two useful consequences:

- **Order-independence.** Because the $R_{ZZ}$ factors commute, they can be applied in *any* order (and reordered/parallelized freely by the transpiler) — the resulting unitary is identical.
- **The only approximation is between layers.** The mixer terms $-X_j$ likewise all commute with each other, so $e^{-i t H_M}=\prod_j e^{+itX_j}$ is also exact. The *sole* Trotter error comes from $[\,H_M,H_P\,]\neq 0$. That discretization error is the annealing granularity, controlled by `STEPS` — and it grows with $\Delta t$, which is why the tuner docs recommend keeping $\Delta t \lesssim 1.3$.

### Gate translations (what the code emits)

Using $R_Z(\theta)=e^{-i\theta Z/2}$, $R_{ZZ}(\theta)=e^{-i\theta\,Z\otimes Z/2}$, $R_X(\theta)=e^{-i\theta X/2}$:

$$
e^{-i t\,h_j Z_j}=R_Z(2 t\,h_j),\qquad
e^{-i t\,J_{jl} Z_jZ_l}=R_{ZZ}(2 t\,J_{jl}),\qquad
e^{+i t\,X_j}=R_X(-2 t).
$$

which is exactly

```python
qc.rz (2 * t_prob * h[j],   j)       # problem: single-Z
qc.rzz(2 * t_prob * J[j,l], j, l)    # problem: ZZ coupling (all mutually commuting)
qc.rx (-2 * t_mix,          j)       # mixer:  -X_j
```

with `t_prob = dt * s`, `t_mix = dt * (1 - s)` and `s = (p + 0.5) / STEPS`.

`ibm_kingston` has `rzz` in its native basis, but after routing an all-to-all layer onto heavy-hex the transpiler still emits SWAP-equivalent `cz`: 55 logical RZZ × 3 steps expanded to **766** two-qubit gates, i.e. ≈ 4.6 physical 2q gates per logical RZZ.

---

# Lagrange multipliers `LAM1` and `LAM2`

## 1. Where they come from

The original problem is **constrained**:

$$\max_{x\in\{0,1\}^N} \sum_i v_i x_i
\qquad\text{subject to}\qquad \sum_i w_i x_i \le C$$

A quantum circuit cannot do constrained optimization. The phase layer applies
$e^{-i\gamma H_C}$, where $H_C$ is a diagonal operator — it can assign a number
to every bitstring, but it cannot forbid any bitstring. The constraint therefore
has to be moved **into the objective**. That is what Lagrangian relaxation is for:

$$\mathcal{L}(x,\lambda) = -\alpha\underbrace{\sum_i v_i x_i}_{V(x)}
\;+\;\lambda_1\Bigl(\underbrace{\sum_i w_i x_i}_{W(x)} - C\Bigr)$$

The sign in front of $V$ is negative because the circuit looks for a **minimum**,
whereas the knapsack is a maximization problem.

The linear term alone is not enough — its minimum is always at a corner of the
cube (every item is either obviously worth taking or obviously not), and the
capacity is hit only by accident. A quadratic term is therefore added, giving the
**augmented Lagrangian**:

$$E(x) = -\alpha V(x) + \lambda_1 W(x) + \lambda_2\bigl(W(x)-C\bigr)^2$$

Strictly speaking only $\lambda_1$ is a Lagrange multiplier; $\lambda_2$ is a
penalty coefficient. In practice they are tuned together and behave as a pair.

---

## 2. What $\lambda_1$ does — the shadow price of capacity

$\lambda_1/\alpha$ is the **exchange rate between value and weight**. Item $i$ is
favoured by the linear term exactly when

$$-\alpha v_i + \lambda_1 w_i < 0
\qquad\Longleftrightarrow\qquad
\frac{v_i}{w_i} > \frac{\lambda_1}{\alpha}$$

So $\lambda_1/\alpha$ is a **threshold on value density**. Items above the
threshold get taken, items below it get left behind.

This is not an arbitrary number. Duality theory says that the optimal multiplier
for the LP relaxation of the knapsack is precisely the density of the **break
item** — the one on which the capacity runs out when filling in order of
decreasing density.

### For the reference N=15 instance (`seed 38`, $C = 68$)

Filling by density $v_i/w_i$:

| rank | item | $v_i$ | $w_i$ | $v_i/w_i$ | cumul. weight |
|---|---|---|---|---|---|
| 1 | 12 | 43 | 3 | 14.33 | 3 |
| 2 | 3 | 53 | 9 | 5.89 | 12 |
| 3 | 4 | 51 | 11 | 4.64 | 23 |
| 4 | 10 | 28 | 8 | 3.50 | 31 |
| 5 | 2 | 32 | 11 | 2.91 | 42 |
| 6 | 13 | 47 | 17 | 2.76 | **59** |
| — | 9 | 34 | 13 | **2.615** | ← break item (13 > 68 − 59) |
| 8 | 1 | 31 | 12 | 2.58 | |
| 9 | 8 | 49 | 19 | 2.58 | |
| 10 | 0 | 45 | 19 | 2.37 | |

$$\lambda^\star = 2.615$$

The six items above the break have value 254 at weight 59; the LP bound is
$254 + 9\cdot 2.615 = 277.5$, while the DP optimum is **272**. Unlike the earlier
instance, the LP relaxation here is **not** integral — the gap of 5.5 means the
linear threshold alone does not hit the optimum and the quadratic term has real
work to do.

Compared with what `tune_penalties()` found:

| quantity | value |
|---|---|
| $\lambda_1/\alpha$ (`LAM1 = 6.20`, `ALPHA = 2.0`) | 3.100 |
| ratio to $\lambda^\star$ | **1.19×** |

So the tuner, independently and purely by maximizing $P(\text{opt})$ in
simulation, arrived at the shadow price from LP duality — and landed exactly in
the 1.09–1.38× window on which the automatic `auto_window=(0.7, 1.7)` is built.
That is a strong check that it is searching for a meaningful quantity, and at the
same time a practical tip: a sensible starting guess for $\lambda_1/\alpha$ is
the density of the break item, not a random number.

---

## 3. What $\lambda_2$ does — curvature and the target weight

Completing the square:

$$-\alpha V + \lambda_1 W + \lambda_2 (W-C)^2
= \lambda_2\bigl(W - C_{\text{eff}}\bigr)^2 - \alpha V + \text{const},
\qquad
\boxed{\,C_{\text{eff}} = C - \frac{\lambda_1}{2\lambda_2}\,}$$

So the pair $(\lambda_1,\lambda_2)$ does not say "stay under the capacity", it
says **"aim at weight $C_{\text{eff}}$ with stiffness $\lambda_2$"**. $\lambda_1$
moves the target, $\lambda_2$ decides how hard the target is enforced.

Two important properties:

**The penalty is symmetric.** It punishes underfilling exactly as much as
overfilling. A state with $W = C+1$ pays $\lambda_2 \cdot 1$, whereas a feasible
state with $W = C-2$ pays $\lambda_2 \cdot 4$. Without $\lambda_1$ the global
minimum can therefore be an infeasible state, and **no value of $\lambda_2$ will
fix that** — the only things that help are a large enough $\lambda_1$, or slack
variables.

**$C_{\text{eff}}$ tends to sit below $C$.** For this instance:

| $\lambda_1/\alpha$ | $\lambda_2/\alpha$ | $C_{\text{eff}}$ | capacity |
|---|---|---|---|
| 3.100 | 0.0895 | **50.7** | 68 |

The target weight lies 17 below the capacity. The quadratic term here therefore
does not act as a hard constraint but as a convex pull towards a slightly
underfilled knapsack; the selection is largely decided by the linear threshold.
The optimum has weight 67, so the penalty does not pull towards it on its own —
it is the linear term that holds it there. Compared with the earlier instance
(where $C_{\text{eff}}$ even came out negative) this is a considerably healthier
regime.

---

## 4. What the multipliers do inside QAOA

They define the whole of $H_C$, and through it affect four things at once:

**a) The ordering of states.** The global minimum of $E(x)$ must be the optimum
we are looking for. Otherwise the circuit converges perfectly onto the wrong
state. `tune_penalties()` checks this condition hard (via the Pareto reduction)
and discards most of the grid on it. **Careful:** it checks it on the
**unpruned** model. With `PRUNE_KEEP < 1` it no longer holds — see the warning in
the pruning section above.

**b) The scale after normalization.** What goes into the circuit is
$\tilde h = h/s$, $\tilde J = J/s$, where $s = \max(\max|h|, \max|J|)$. A stronger
penalty increases $s$; after the division all couplings shrink and **the same
$\gamma$ performs a smaller part of the evolution**. The multipliers therefore act
partly as an inverse time knob — which is why they cannot be tuned independently
of $T$ and `STEPS`.

**c) The ratio of couplings to biases.** $h_i$ contains both $-\alpha v_i$ and
penalty terms, whereas $J_{ij} = \tfrac{\lambda_2}{2}w_i w_j$ comes **exclusively**
from the quadratic penalty. As $\lambda_2 \to 0$ the two-qubit gates lose their
meaning and the problem falls apart into independent bits. For this instance:

$$\frac{\max|J|}{\max|h|} = 0.203$$

which is above the floor (≈ 0.1), but not by a wide margin — the couplings here
carry a real, though not dominant, part of the information.

**d) The fraction of feasible samples.** A stronger penalty raises `P(feasible)`,
but it is paid for through point (b). Measured on hardware:
`P(feasible) = 12.6 %`.

---

## 5. Three failure modes

| mode | symptom | consequence |
|---|---|---|
| $\lambda_1$ too small | the minimum of $E$ is an infeasible state | the circuit converges on an invalid solution |
| $\lambda_1, \lambda_2$ too large | inflated $s$, small $\tilde J$ | level differences below resolution, `P(opt)` drops |
| $\lambda_2$ too small | $\tilde J \ll \tilde h$ | the problem is nearly separable, a product state, classically samplable |

The third mode is treacherous, because in simulation it looks **best** — high
$P(\text{opt})$, low rank. But a product state can be sampled classically in
closed form, so the comparison baseline is then not the uniform distribution but
that classical sampler.

---

## 6. Diagnostics

```python
import numpy as np

v, w = np.array(values, float), np.array(weights, float)
l1, l2 = LAM1 / ALPHA, LAM2 / ALPHA

# a) shadow price from the LP relaxation
order = np.argsort(-(v / w)); cum = 0.0
for i in order:
    if cum + w[i] <= CAPACITY: cum += w[i]
    else: break
print(f"LP dual = {v[i]/w[i]:.3f}   LAM1/ALPHA = {l1:.3f}"
      f"   ratio = {l1/(v[i]/w[i]):.2f}x")

# b) target weight
print(f"C_eff = {CAPACITY - l1/(2*l2):.1f}   (capacity {CAPACITY})")

# c) ratio of couplings to biases
lin = -v + l1*w + l2*(w**2 - 2*CAPACITY*w)
h   = -lin/2 - (l2/2) * w * (w.sum() - w)
J   = np.triu(np.outer(w, w) * (l2/2), 1)
print(f"max|J| / max|h| = {np.abs(J).max()/np.abs(h).max():.4f}"
      f"   (below ~0.1 the problem is nearly separable)")
```

The package exposes the first of these directly:

```python
from qiskit_qaoa_knapsack_domi import lp_dual
print(lp_dual(problem))          # 2.615 for this instance
```

Measured for this run: `LP dual = 2.615`, `LAM1/ALPHA = 3.100` (1.19×),
`C_eff = 50.7`, `max|J|/max|h| = 0.2026`, `P(feasible) = 0.126`.

Target values: $\lambda_1/\alpha$ on the order of the LP dual,
$\max|J|/\max|h|$ ideally above 0.3, and `P(feasible)` at least 30 % in the
measured data. This run **does not meet** the last two — both are a consequence
of the pruning and the shallow schedule.

---

# What `tune_penalties()` computes

A purely mathematical account of the preprocessing, with no reference to the
implementation. Everything below is what the function evaluates; the algorithmic
tricks only change the cost of evaluating it, never the value.

## 1. The optimization problem being solved

Given an instance $(v, w, C)$ with DP optimum
$\text{OPT} = \max\{V(x) : W(x)\le C\}$, let

$$\mathcal{X}^\star = \{x : W(x)\le C,\ V(x)=\text{OPT}\}$$

be the set of optimal bitstrings. The tuner searches over
$(\lambda_1,\lambda_2,P,T)$ — penalty pair, Trotter step count, total time — and
returns

$$
(\lambda_1^\star,\lambda_2^\star,P^\star,T^\star)
=\arg\max_{(\lambda_1,\lambda_2)\in\mathcal{F}, \ (P,T)\in\mathcal{S}}
\ \mathrm{score}(\lambda_1,\lambda_2,P,T)
$$

where $\mathcal{S}$ is the schedule grid, $\mathcal{F}$ is the feasible set of
§2, and the score is defined in §5. The objective is a **statevector simulation
of the circuit that will actually run** — not a surrogate.

### Scale invariance

Internally the search runs at $\alpha=1$ over
$\ell_1 = \lambda_1/\alpha$, $\ell_2 = \lambda_2/\alpha$, and the result is
rescaled as $\lambda_i = \ell_i\,\alpha_{\text{out}}$. This is legitimate because
the Ising coefficients are normalized by their own maximum (§3): the triples
$(\alpha,\lambda_1,\lambda_2)$ and $(k\alpha,k\lambda_1,k\lambda_2)$ produce a
bit-identical circuit. `ALPHA` is therefore a reporting convention, not a
degree of freedom.

## 2. The hard condition and its Pareto reduction

Write the classical cost at $\alpha=1$ as

$$E_{\text{cls}}(x) = -V(x) + \ell_1 W(x) + \ell_2\bigl(W(x)-C\bigr)^2 .$$

A pair $(\ell_1,\ell_2)$ is admissible iff the global minimum of
$E_{\text{cls}}$ is attained on $\mathcal{X}^\star$:

$$\mathcal{F} = \Bigl\{(\ell_1,\ell_2) \ :\
\min_{x\notin\mathcal{X}^\star} E_{\text{cls}}(x)
\;>\;\max_{x\in\mathcal{X}^\star} E_{\text{cls}}(x)\Bigr\}.$$

$E_{\text{cls}}$ depends on $x$ **only through the pair** $(V,W)$, so the
$2^N$ states collapse onto the reachable set

$$\mathcal{R} = \bigl\{(V,W) : \exists x,\ V(x)=V,\ W(x)=W\bigr\}
\ \subseteq\ \{0,\dots,\textstyle\sum_i v_i\}\times\{0,\dots,\sum_i w_i\},$$

computed by the 0/1 recursion
$\mathcal{R}_0=\{(0,0)\}$, $\mathcal{R}_i=\mathcal{R}_{i-1}\cup(\mathcal{R}_{i-1}+(v_i,w_i))$.

For fixed $W$, $E_{\text{cls}}$ is strictly decreasing in $V$, so only the
largest attainable value matters. With
$V_{\max}(W)=\max\{V:(V,W)\in\mathcal{R}\}$ and $V_{2\text{nd}}(W)$ the second
largest, the minimum over non-optimal states is attained on

$$
\mathcal{C}=\Bigl\{\bigl(V_{\max}(W),W\bigr)\ :\ W>C \ \text{ or } \ V_{\max}(W)<\text{OPT}\Bigr\}
\ \cup\
\Bigl\{\bigl(V_{2\text{nd}}(W),W\bigr)\ :\ W\le C,\ V_{\max}(W)=\text{OPT}\Bigr\}
$$

and the maximum over optimal states on
$\mathcal{O}=\{(\text{OPT},W) : W\le C,\ V_{\max}(W)=\text{OPT}\}$. The test
becomes

$$\min_{(V,W)\in\mathcal{C}} E_{\text{cls}} \;>\; \max_{(V,W)\in\mathcal{O}} E_{\text{cls}},$$

two reductions over sets of size $O(\sum_i w_i)$ instead of $2^N$. The
classification is exhaustive: for $W\le C$ the definition of the optimum gives
$V_{\max}(W)\le\text{OPT}$, and for $W>C$ no optimal state exists.

## 3. Ising energies as an affine map of $E_{\text{cls}}$

The QUBO coefficients are

$$\text{lin}_i = -v_i + \ell_1 w_i + \ell_2\bigl(w_i^2-2Cw_i\bigr),
\qquad \text{quad}_{ij} = 2\ell_2 w_i w_j\quad(i<j),$$

so that $E_{\text{QUBO}} = E_{\text{cls}} - \ell_2 C^2$. Substituting
$x_i=(1-Z_i)/2$ splits off a constant,

$$
h_j = -\tfrac12\text{lin}_j - \tfrac{\ell_2}{2}\,w_j\bigl(W_{\text{sum}}-w_j\bigr),
\qquad
J_{jl} = \tfrac{\ell_2}{2}\,w_j w_l ,
$$

$$
\kappa = \tfrac12\sum_i \text{lin}_i + \tfrac{\ell_2}{4}\Bigl(W_{\text{sum}}^2-\sum_i w_i^2\Bigr),
\qquad W_{\text{sum}}=\sum_i w_i .
$$

Because $J$ is a rank-one outer product scaled by $\ell_2/2$, its largest entry
is fixed by the two heaviest items $w_{(1)}\ge w_{(2)}$, giving the normalizer in
closed form:

$$s = \max\Bigl(\max_j |h_j|,\ \tfrac{\ell_2}{2}\,w_{(1)}w_{(2)}\Bigr).$$

The normalized diagonal energy of every basis state then follows **without ever
touching the $N(N-1)/2$ couplings individually**:

$$\boxed{\ \tilde E(x) \;=\; \frac{E_{\text{cls}}(x) - \ell_2 C^2 - \kappa}{s}\ }$$

An affine map in $E_{\text{cls}}$, evaluated in $O(2^N)$ additions rather than
$O(2^N \cdot N^2)$. Both $h$ and $\kappa$ cost $O(N)$.

## 4. The simulated state

With $P$ steps, $\Delta t = T/P$ and the midpoint schedule
$s_p = (p+\tfrac12)/P$, the simulation applies

$$|\psi\rangle = \Bigl[\textstyle\prod_{p=P-1}^{0}
e^{\,i\,\Delta t\,(1-s_p)\sum_j X_j}\;
e^{-i\,\Delta t\,s_p\,\tilde E}\Bigr]\;|+\rangle^{\otimes N},$$

the diagonal factor acting as $e^{-i\Delta t\,s_p \tilde E(x)}$ on each basis
state, and the mixer factoring into single-qubit rotations
$\cos\beta_p\,I + i\sin\beta_p X_j$ with $\beta_p=\Delta t(1-s_p)$. This is
exactly the circuit `build_circuit()` emits, so the reported probabilities
describe the real circuit, in the noiseless limit.

The quantities extracted from $p(x)=|\langle x|\psi\rangle|^2$ are

$$
P_{\text{opt}} = \!\!\sum_{x\in\mathcal{X}^\star}\!\! p(x),
\qquad
P_{\text{feas}} = \!\!\sum_{x:\,W(x)\le C}\!\! p(x),
\qquad
\text{rank} = \bigl|\{y : p(y) > p(x^\star)\}\bigr| + 1,
$$

with $x^\star$ the most probable optimal state, and $U=2^{-N}$ the uniform
baseline against which $P_{\text{opt}}/U$ is quoted.

## 5. Scoring against hardware noise

Let $R = P\cdot\frac{N(N-1)}{2}$ be the number of logical RZZ. Under a global
depolarizing model with per-two-qubit-gate error $\varepsilon$ and routing
overhead $\eta$ (default $2.9$), the surviving coherent fraction is

$$F = \exp\bigl(-\varepsilon\,\eta\,R\bigr),$$

and the state is $F\rho_{\text{ideal}} + (1-F)\,\mathbb{1}/2^N$, giving

$$\mathrm{score} = F\cdot P_{\text{opt}} + (1-F)\cdot U\cdot|\mathcal{X}^\star| .$$

With $\varepsilon$ unset, $F=1$ and the score reduces to $P_{\text{opt}}$. This
is what stops the search from selecting a deep schedule whose ideal
$P_{\text{opt}}$ is high but which the device cannot sustain: $P_{\text{opt}}$
grows with $P$ while $F$ decays exponentially in $P$.

## 6. Where the search looks

The LAM1 window, when not given explicitly, is centred on the LP dual
$\lambda^\star$ of §2 above:

$$\ell_1 \in \bigl[0.7\,\lambda^\star,\ 1.7\,\lambda^\star\bigr],$$

traversed on an arithmetic grid of spacing $h$. LAM2 is traversed
**geometrically**,

$$\ell_2 \in \Bigl\{\ell_2^{\min} r^{k}\Bigr\}_{k=0}^{n_{\ell_2}-1},
\qquad r = \Bigl(\ell_2^{\max}/\ell_2^{\min}\Bigr)^{1/(n_{\ell_2}-1)},$$

because the admissible region is bounded below by a validity floor above which
the optimum typically sits within a small multiplicative factor — a region an
arithmetic grid resolves poorly. Refinement then re-searches a neighbourhood of
each of the best coarse points, arithmetically in $\ell_1$ with spacing $h/5$ and
geometrically in $\ell_2$ over $[\ell_2/r,\ \ell_2 r]$.

Only phases 1–2 are run at the cheapest schedule; the full grid $\mathcal{S}$ is
evaluated on the best candidates only. The reported plateau is

$$\bigl\{(\lambda_1,\lambda_2) : \mathrm{score} \ge 0.9\cdot\mathrm{score}^\star\bigr\},$$

reported as its bounding box, and the shot recommendation follows from requiring
ten expected hits on the optimum:

$$\text{SHOTS} \ \ge\ \max\Bigl(10\cdot 2^N,\ \bigl\lceil 10/P_{\text{opt}}\bigr\rceil\Bigr).$$
