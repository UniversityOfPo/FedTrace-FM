# FedTrace v3 — LaTeX Equation Library

**Status:** Reference document only, produced as part of Phase P3 (Theoretical Formalization). No code, config, test, result, checkpoint, README, or Research Database file was modified in producing it. Every snippet below is copy-paste ready for an Overleaf `amsmath`/`amsthm` document (`\usepackage{amsmath, amssymb, amsthm}` assumed). No paper prose is written here — equations and their defining context only, per this task's explicit restriction. Full derivations/justification live in `analysis/FEDTRACE_THEORY_FORMALIZATION.md`.

---

## 1. Federated Update Definition

```latex
w_r \in \mathbb{R}^d \quad \text{(global model at round } r\text{)}
```
```latex
\Delta_{k,r} \;=\; w_{k,r} - w_r \in \mathbb{R}^d
```
```latex
K_r \subseteq \{1,\dots,N\}, \qquad n_r = |K_r|
```

## 2. Magnitude Signal

```latex
m_{k,r} \;=\; \left\lVert \Delta_{k,r} \right\rVert_2
```
```latex
\mu^m_k \;=\; \operatorname{mean}\bigl\{ m_{k,r} : r < T_{\mathrm{ref}} \bigr\}, \qquad
s^m_k \;=\; \operatorname{stdev}\bigl\{ m_{k,r} : r < T_{\mathrm{ref}} \bigr\}
```
```latex
z^m_{k,r} \;=\; \frac{\left| m_{k,r} - \mu^m_k \right|}{s^m_k}
```

## 3. Temporal Cosine

```latex
t_{k,r} \;=\; \cos\!\left(\Delta_{k,r}, \Delta_{k,r-1}\right)
\;=\; \frac{\left\langle \Delta_{k,r}, \Delta_{k,r-1} \right\rangle}
           {\left\lVert \Delta_{k,r} \right\rVert_2 \left\lVert \Delta_{k,r-1} \right\rVert_2},
\qquad r \geq 1
```

## 4. Common Round Reference (No Leave-One-Out)

```latex
c_r \;=\; \operatorname*{median}_{j \in K_r} t_{j,r}
```

## 5. Temporal Residual

```latex
d_{k,r} \;=\; c_r - t_{k,r}
```

## 6. Historical Residual Calibration

```latex
D_k \;=\; \bigl\{ d_{k,r} : 1 \leq r < T_{\mathrm{ref}} \bigr\}
```
```latex
\mu^d_k \;=\; \operatorname{mean}(D_k), \qquad
s^d_k \;=\; \operatorname{stdev}(D_k)
```

## 7. One-Sided Temporal Residual Anomaly Score

```latex
z^{t}_{k,r} \;=\; \max\!\left( 0,\; \frac{d_{k,r} - \mu^d_k}{s^d_k} \right)
```

## 8. Final FedTrace v3 Score

```latex
S_{k,r} \;=\; \max\!\left( z^{m}_{k,r},\; z^{t}_{k,r} \right)
```
```latex
% Undefined-component exclusion (prose form, for a Method-section footnote):
% if only one of z^m_{k,r}, z^{t}_{k,r} is defined, S_{k,r} equals that one
% value; the missing component is excluded from the max, never treated as 0.
```

## 9. Persistence Rule

```latex
\text{raw anomaly at } (k,r): \quad S_{k,r} \text{ defined and } S_{k,r} > \tau, \qquad \tau = 3.0
```
```latex
\text{confirmed streak of length } P \text{ starting at } r^{*}: \quad
S_{k,r} > \tau \ \ \forall\, r \in \{r^{*}, r^{*}+1, \dots, r^{*}+P-1\}
```
```latex
P = 3
```

## 10. Onset Estimate and Confirmation Round

```latex
\hat{r}_{\mathrm{onset}}(k) \;=\; r^{*}
\qquad\qquad
r_{\mathrm{confirm}}(k) \;=\; r^{*} + P - 1
```

## 11. Attribution Rule

```latex
\hat{k}_{\mathrm{adv}} \;=\; \operatorname*{arg\,min}_{k \,:\, \text{confirmed}}
\Bigl( r_{\mathrm{confirm}}(k),\; -\overline{S}_k,\; k \Bigr)
```
```latex
% \overline{S}_k = mean score during k's confirmed run; the tuple ordering
% encodes: earliest confirmation wins, ties -> higher mean score, then
% lexical client_id -- exactly the three-key sort in the implementation.
```

## 12. Persistence-Delay Lemma (L1)

```latex
\text{If } S_{k,r} > \tau \ \forall\, r \in [r^{*}, r^{*}+P-1], \text{ then}
```
```latex
r_{\mathrm{confirm}}(k) = r^{*} + P - 1, \qquad
\text{detection\_delay} = P - 1
```
```latex
% For P = 3: minimum confirmation delay = 2 rounds.
```

## 13. Gaussian Attack: Norm Decomposition (Proposition P1)

```latex
\Delta' \;=\; \Delta + \varepsilon, \qquad
\left\lVert \Delta' \right\rVert^2 \;=\;
\left\lVert \Delta \right\rVert^2 + 2\left\langle \Delta, \varepsilon \right\rangle + \left\lVert \varepsilon \right\rVert^2
```
```latex
\left\lVert \varepsilon \right\rVert = \rho \left\lVert \Delta \right\rVert,
\qquad
\varepsilon = \left\lVert \varepsilon \right\rVert \cdot U,\;\; U \sim \mathrm{Uniform}\bigl(S^{d-1}\bigr) \perp \Delta
```
```latex
\mathbb{E}\bigl[\left\langle \Delta, \varepsilon \right\rangle\bigr] = 0
\qquad\Longrightarrow\qquad
\mathbb{E}\bigl[\left\lVert \Delta' \right\rVert^2\bigr] \;=\; (1+\rho^2)\left\lVert \Delta \right\rVert^2
```
```latex
% Per-realization caveat (do not omit in prose): ||\Delta'|| < ||\Delta||
% occurs exactly when \cos\theta < -\rho/2, \theta = \angle(\Delta,\varepsilon).
```

## 14. Directional Attack: Exact Construction (Proposition P2)

```latex
\hat{\Delta} \;=\; \frac{\Delta}{\left\lVert \Delta \right\rVert}, \qquad
u_{\perp} \perp \hat{\Delta}, \quad \left\lVert u_{\perp} \right\rVert = 1
```
```latex
a \;=\; c\,\hat{\Delta} + \sqrt{1-c^2}\; u_{\perp}, \qquad c \in [-1,1]
```
```latex
\Delta_{\mathrm{attack}} \;=\; \left\lVert \Delta \right\rVert \cdot a
```
```latex
\left\lVert \Delta_{\mathrm{attack}} \right\rVert = \left\lVert \Delta \right\rVert,
\qquad
\cos\!\left(\Delta_{\mathrm{attack}}, \Delta\right) = c
```
```latex
% Special case c = 0 (used throughout this project):
% \Delta_{\mathrm{attack}} = \left\lVert \Delta \right\rVert \cdot u_{\perp},
% \quad \cos\!\left(\Delta_{\mathrm{attack}}, \Delta\right) = 0 \text{ exactly.}
```

## 15. Norm-Only Indistinguishability (Corollary C1)

```latex
\forall\, f : \mathbb{R}_{\geq 0} \to \{\text{honest}, \text{attack}\}, \qquad
\left\lVert \Delta_{\mathrm{attack}} \right\rVert = \left\lVert \Delta \right\rVert
\;\Longrightarrow\;
f\bigl(\left\lVert \Delta_{\mathrm{attack}} \right\rVert\bigr) = f\bigl(\left\lVert \Delta \right\rVert\bigr)
```

## 16. Common-Mode Drift Decomposition (Proposition P3)

```latex
t_{k,r} \;=\; g_r + b_k + \eta_{k,r}, \qquad \mathbb{E}_r[\eta_{k,r}] = 0 \quad \text{(honest clients)}
```
```latex
c_r = g_r + \beta_r, \quad \beta_r := \operatorname*{median}_j \bigl(b_j + \eta_{j,r}\bigr)
\quad\Longrightarrow\quad
d_{k,r} = \left(\beta_r - b_k\right) - \eta_{k,r}
```
```latex
% Median breakdown point at n_r = 5: tolerates up to \lfloor (n_r-1)/2 \rfloor = 2
% arbitrary values before c_r becomes unbounded -- a single attacker is below
% this bound, which is why no leave-one-out is required to compute c_r.
```

## 17. Krum / Multi-Krum Score

```latex
d(i,j) \;=\; \left\lVert u_i - u_j \right\rVert_2^2
\;=\; \left\lVert u_i \right\rVert^2 + \left\lVert u_j \right\rVert^2 - 2\left\lVert u_i \right\rVert \left\lVert u_j \right\rVert \cos\theta_{ij}
```
```latex
\mathrm{score}(i) \;=\; \sum_{j \in \mathcal{N}_i} d(i,j),
\qquad \mathcal{N}_i = \text{the } (n-f-2) \text{ nearest peers of } i \text{ by } d(i,\cdot)
```
```latex
m = n - f \qquad \text{(number of clients selected each round)}
```
```latex
% This project: n = 5, f = 1 \Rightarrow neighbors\_used = n-f-2 = 2,\; m = 4.
```

---

## Notes for Overleaf Integration

- Recommended theorem-environment names, matching `analysis/FEDTRACE_THEOREM_LEMMA_INVENTORY.csv`'s `ID` column exactly: `\newtheorem{proposition}{Proposition}`, `\newtheorem{lemma}{Lemma}`, `\newtheorem{corollary}{Corollary}`, `\newtheorem{observation}{Observation}`, `\newtheorem{remark}{Remark}` — use `Proposition` for P1/P2/P3, `Lemma` for L1, `Corollary` for C1, `Observation` for O1, `Remark` for R1/R2/R3 (matching the Type column of the inventory CSV, which distinguishes formally-proved results from conditional/mechanistic ones).
- Every symbol above (`\Delta`, `m`, `t`, `c_r`, `d`, `z^m`, `z^t`, `S`, `\tau`, `P`) is used consistently with `analysis/FEDTRACE_THEORY_FORMALIZATION.md` — copy definitions in the order given (1→17) so each equation's symbols are already introduced when it appears.
- No numeric value in this file was altered from the frozen implementation (`\tau=3.0`, `P=3`, `T_{\mathrm{ref}}=20`, `n=5`, `f=1`) — these are literal, copy-forward constants, not placeholders to be filled in later.
