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

### What pruning costs, and why the run is still valid

Pruning changes the cost function, so the global minimum of the **pruned** model is no
longer the knapsack optimum:

| `PRUNE_KEEP` | RZZ kept | ground state of pruned QUBO | feasible? |
|---|---|---|---|
| 1.00 | 105 | value 272, weight 67 | yes — the true optimum |
| 0.70 | 74 | value 281, weight 77 | no (cap 68) |
| 0.50 | 55 | value 328, weight 94 | no (cap 68) |

This is **not** a failure of the method as used here. The circuit is a *sampler*, and the
selection step is classical: keep the best feasible bitstring. Pruning shifts the
distribution's centre of mass, so the price is paid in shots, not in correctness — the
answer is still verified against the constraint before it is reported. What it does mean:

- `check_ground_state()` checks the **unpruned** cost
  (`-ALPHA*_V + LAM1*_W + LAM2*(_W-CAPACITY)**2`), so it verifies the *tuning*, not the
  *circuit*. That is a reasonable thing to check; it is just not a check on the pruned model.
- `tune_penalties()` scores the unpruned circuit, so its predicted `P(opt)` is an upper
  bound on what the pruned circuit delivers.

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

