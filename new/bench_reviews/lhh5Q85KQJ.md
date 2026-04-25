Now let me search for calibration anchors.Now I have enough information to write the final synthesized review.

---

## Summary
SubDisMO proposes a resource-aware federated learning algorithm that trains adaptively-sized submodels with SAM-style perturbations, framed as a "distributed minimax optimization" problem. The method introduces a minimum covering number C* to characterize convergence, unifying FedAvg, FedSAM, RAM-Fed, and OAP as special cases, and extends PAC-Bayesian generalization bounds to the partial-network setting via per-layer remaining rates s_j. Experiments on CIFAR-10/100 show consistent improvements over resource-constrained baselines.

---

## Strengths

- **Unified convergence result via minimum covering number C* (Corollary 1, Remark 1)**: The O(1/√(QTC*)) rate cleanly recovers four prior algorithms as special cases (FedAvg at C*=N, δ=0; FedSAM at C*=N; RAM-Fed at δ=0; OAP at C*=1, δ=0). This is a concrete, specific theoretical bridge between resource-limited FL and perturbation-based FL that is properly derived and explicitly verified.

- **PAC-Bayesian generalization bound for the submodel setting (Theorem 2)**: Incorporating the per-layer remaining rate s_j into the generalization bound is a genuine extension beyond prior work. When s_j=1 it correctly reduces to the FedSAM-style bound, and the tightening for s_j<1 (fewer active parameters) is analytically justified via Lemma 5.

- **Empirical ablation of key theoretical quantities (Table 2, Figure 4)**: The experiments show that larger C* yields faster convergence and higher accuracy, and that δ has an optimal range—both directly validating the theoretical predictions in Corollary 1 and Remark 3.

- **Consistent improvement and reduced variance over the closest baseline**: In Table 1, SubDisMO outperforms RAM-Fed by 1.52–2.97% accuracy on CIFAR-10 and 0.55–1.26% on CIFAR-100 across IID and non-IID settings, while also reducing standard deviation in most comparisons (e.g., 8.47 vs. 11.49 at μ=0.5 on CIFAR-10).

---

## Weaknesses

### Fatal
None.

### Major

- **Missing critical ablation: existing submodel method + SAM as baseline.** The paper's entire algorithmic contribution over RAM-Fed is the addition of a SAM-style gradient ascent step on the submodel. Yet Table 1 includes no baseline of the form "RAM-Fed + SAM," "OAP + FedSAM," or any analogous combination. Without this, it is impossible to determine whether the performance gain (1–3% over RAM-Fed) stems from SubDisMO's specific design or simply from applying an existing technique (SAM) naively to an existing framework (RAM-Fed). This is not a peripheral experiment—it is the central question about whether the paper makes an algorithmic contribution beyond a trivial combination, and its absence prevents the core empirical claim from being properly attributed.

- **Framing as "distributed minimax optimization" overclaims the scope.** The paper introduces the problem with reference to AUC maximization (SGDAM-PEF, LocalSCGDAM), DRO (Sinha et al.), and general GDA-based methods (FedGDA-GT, FedSGDA+), framing SubDisMO as solving "distributed minimax optimization" generally. However, Eq. (2) restricts the inner maximization to a norm-ball perturbation ||ε_i|| ≤ δ, which is precisely the SAM objective—a specific structured instance of minimax, not a generalization. No AUC maximization, DRO, or any other genuine minimax task is evaluated. The method is correctly described as resource-aware FedSAM for submodels internally, but the introduction and title position it more broadly than the experiments and formulation support.

### Minor

- **Convergence criterion covers only trained parameters K_q, a non-standard choice.** Theorem 1 and Corollary 1 bound (1/Q)Σ_q Σ_{i∈K_q} E[||∇f^i(θ_q)||²], summing gradient norms only over parameters trained in each round and excluding untrained parameters. The paper presents this as a feature in Remark 2 ("we innovatively analyze… trained parameters K_q and untrained parameters S − K_q"), but does not discuss the implication that parameters never or rarely trained are entirely excluded from the convergence guarantee. Standard non-convex FL convergence results cover the full gradient. The paper should either justify why the partial-gradient criterion is the appropriate measure for the resource-constrained setting, or acknowledge this as a limitation.

- **Residual non-vanishing terms undercut the "asymptotically optimal" claim.** Corollary 1 includes terms O(l²B₀/C*) and O(σ_g²/C*) that remain constant as Q → ∞, meaning the algorithm converges to a neighborhood of a stationary point, not to a stationary point itself. This is acceptable and standard for non-convex FL, but the claim of "asymptotically optimal convergence rate" refers only to the dominant decaying term. The paper should clarify that the non-vanishing residuals bound the size of the neighborhood, and the "asymptotically optimal" label applies to the rate of the decaying component.

- **Small experimental scale (10 clients only in main results).** All Table 1 results use exactly 10 clients. For a federated learning paper, this is a small regime. Scalability to 50–100 clients is acknowledged as deferred to the appendix, which is acceptable but means the main results cover a narrow regime where C* ∈ {1,…,10}.

### Trivial

- **Same mini-batch used for perturbation step and gradient step (Algorithm 1, lines 122–127).** The gradient g_{q,n,t-1} and the update gradient g̃_{q,n,t-1} are computed with the same sample ξ_{n,t-1}, which is common in SAM implementations but introduces a correlation. This is an implementation detail widely accepted in the SAM literature and does not threaten the main claims.

---

## Nice-to-Haves

- Including RAM-Fed+SAM or OAP+FedSAM as baselines in Table 1 would definitively clarify the contribution. This could transform a major weakness into a strength.
- Evaluating on at least one genuine minimax task (e.g., AUC maximization with imbalanced federated data, or federated DRO) would validate the broader "minimax" framing. A CIFAR-10 with long-tailed class imbalance setup (where AUC-based objectives are natural) would be a low-overhead addition.
- A plot of the theoretical convergence quantity (1/Q)Σ_q Σ_{i∈K_q} E[||∇f^i||²] over training, comparing it to the standard full-gradient norm, would empirically validate Theorem 1 and expose any gap between the partial criterion and standard convergence.
- An experiment with larger client counts (N=50, 100) in the main text to show that the C*-dependent rate generalizes beyond the narrow 10-client regime.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic's claim that Lemma 1 and 2 cannot be "borrowed" from Qu et al. (2022) because Qu et al. analyzes FedSAM on the full model.** The paper applies these lemmas as bounding tools, adapted for the masked setting. The mathematical content of the bounds (variance bounds for perturbed gradients) is derived independently of whether the model is full or masked. The Harsh Critic's concern is overstated and does not invalidate the theoretical structure.

- **Harsh Critic's claim about learning rate conditions creating "implicit constraints" never verified experimentally.** The learning rate conditions in Theorem 1 constrain η_l and η_g given C*, T, N. These are existence conditions for the convergence theorem, not requirements that must be reported for each experiment. It is standard practice in FL theory papers to state these conditions and set learning rates that satisfy them by construction (e.g., η_g = √C*/√T, η_l = 1/√Q in Corollary 1). This is a normal-level concern for theory papers, not a flaw.

- **Harsh Critic's claim that Assumption 4 is unjustified for random masking (vs. importance-based pruning).** The paper does briefly note in Remark 2 that "although clients in our algorithm are adaptive generate submodel according to the resource, the assumption is also established." This acknowledgment is imprecise but the assumption itself is mathematically standard (bounded relative deviation from the full model). Demanding full justification for this is a scope-creep concern.

- **Remark 4's "tighter bound" interpretation.** The Harsh Critic argues this conflates bound tightness with practical improvement. Mathematically, the claim (that s_j < 1 gives a tighter bound than s_j = 1) is correct as stated. Whether this translates to practical improvement is a different question, and the paper is careful to say "tighter bound," not "better practical performance." This is a minor precision issue that does not rise to a real weakness.

- **Strength Finder's generic claim about "Algorithm is straightforwardly implementable":** Removed as too generic.

- **Strength Finder's claim about "Tighter stochastic gradient assumption":** Removed as a minor mathematical observation insufficient to constitute a standalone paper strength.

---

## Novel Insights

The minimum covering number C* is a clean and practically interpretable parameter that interpolates between all-or-nothing training regimes and reveals how parameter coverage frequency governs convergence in heterogeneous-resource FL. The extension of PAC-Bayesian bounds to per-layer remaining rates is a technically non-trivial contribution that bridges model pruning and generalization theory in the federated setting. However, the central open question the paper raises but does not answer—whether adding perturbations to submodel training contributes beyond simply adding SAM to any existing submodel framework—remains empirically unresolved and constitutes the most actionable direction for future work.

---

## Suggestions

1. **Add RAM-Fed+SAM, OAP+FedSAM as baselines in Table 1.** This single experiment would resolve the most significant weakness in the paper and is straightforward to run.
2. **Narrow the "minimax" framing in the introduction.** Reframe the contribution as "resource-aware FedSAM via submodel training" rather than positioning against general GDA-style minimax algorithms. Alternatively, add one AUC or DRO experiment to justify the broader framing.
3. **Add a clear statement in Section 4.1** that Theorem 1 provides a partial-gradient convergence bound (covering only trained parameters), explaining the practical interpretation and why this is meaningful in the resource-constrained setting.
4. **Clarify "asymptotically optimal"** in Corollary 1 to explicitly note that this refers to the rate of the dominant decaying term, with the algorithm converging to a neighborhood whose size is controlled by the l and σ_g residual terms.

---

## Score and Decision

**Calibration Anchors:**

| Paper | Avg Score | Relevance |
|---|---|---|
| `/human_reviews/jj5ZjZsWJe.md` | 8.0 | High: Strong theoretical FL contribution with communication compression; tighter than this paper in scope/experiment match |
| `/human_reviews/e0rQRMUhs7.md` | 6.6 | Medium-high: Resource-constrained FL with LLM, accepted as poster; experiments more directly supporting claims |
| `/human_reviews/kWsJkH1tNi.md` | 5.0 | Most comparable: PAC-Bayes bounds for FL with some unrealistic assumptions, decent theory but scope/experiment gaps; rejected |
| `/human_reviews/Ob0UafH2YI.md` | 4.67 | Medium-low: Federated compositional optimization with actual factual errors in related work table; withdrawn |
| `/human_reviews/WoJzHQIIUk.md` | 1.5 | Low anchor: Minimax + neural networks paper; fundamentally poor experimental design — clearly weaker than this paper |
| `/human_reviews/GOt2kP383R.md` | 5.25 | Medium: Overclaim in framing + decent empirical results — structurally similar to this paper's situation |

**Reasoning:** SubDisMO is closest to the kWsJkH1tNi (5.0) and GOt2kP383R (5.25) anchors: real theoretical contributions with PAC-Bayes flavor in an FL setting, but with meaningful gaps between claimed scope and actual evaluation. Compared to kWsJkH1tNi, this paper has a more complete algorithm+theory+experiment package, but the missing critical ablation (RAM-Fed+SAM) is more damaging than that paper's concerns. The framing overclaim is real but not fatal. The paper is clearly above the low anchors (1.5–2.0), and somewhat below the high anchors (6.6+) which have stronger experiment-claim alignment. The center of the relevant anchor cluster is ~5.0. The missing ablation—which prevents clean attribution of the main empirical claim—pulls it slightly below this center.

**Score: 4.5 — Reject**

The paper has genuine contributions in convergence unification and generalization bounds, and the empirical results are positive, but the missing ablation (existing submodel method + SAM) is a fundamental gap for a paper whose core claim is precisely the benefit of combining submodel training with perturbation, and the "distributed minimax optimization" framing is broader than what the formulation and experiments justify.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>