Now let me search for calibration papers to compare against for scoring.Now I have enough information to write the final consolidated review. Let me synthesize everything.

---

## Summary

This paper studies the *conditional causal bandit* problem, where the arms are single-node conditional interventions on a causal graph and the goal is to identify which nodes are worth intervening on. The main contribution is a graphical characterization of the minimal globally interventionally superior set (mGISS)—the smallest set guaranteed to contain the optimal intervention node—as the LSCA (Lowest Strict Common Ancestor) closure of the parents of the target variable Y. A key intermediate result shows that conditional-intervention superiority is equivalent to deterministic atomic-intervention superiority (Proposition 4), enabling clean reasoning. The authors also propose the C4 algorithm that computes this set in O(|V|+|E|) time, and provide empirical illustrations of search-space reduction and regret improvement.

---

## Claims and Support

**Claim 1: mGISS = L∞(Pa(Y)) — graphical characterization of the minimal set**
*Well-supported.* The definitions (Defs. 7–9), Theorem 12 (Λ-structure characterization), and Theorem 13 (equality with mGISS) form a coherent chain. Minimality is argued via witness SCMs. No unsupported jumps are apparent.

**Claim 2: Equivalence of conditional-intervention superiority and deterministic atomic-intervention superiority (Proposition 4)**
*Well-supported.* The equivalence relies on the observable-conditioning-set assumptions, which are carefully stated in Section 2. The paper appropriately notes in Appendix/Example 26 where analogous equivalence would fail (general probabilistic SCMs), which strengthens credibility.

**Claim 3: C4 algorithm is correct and runs in O(|V|+|E|)**
*Well-supported.* The algorithm is straightforward; correctness reduces to Lemma 15, which is argued carefully. Runtime is standard.

**Claim 4: Pruning to mGISS significantly reduces search space in random and real-world graphs**
*Partially supported.* Results are shown only when Y is selected as the node with the most ancestors (Section 6, footnote 8). This biases the picture toward large ancestor sets. The claim that "we can expect our method to be useful in real-world causal models" is broader than the evidence supports.

**Claim 5: mGISS integration substantially accelerates bandit convergence**
*Partially supported.* Section 6 explicitly states in footnote 11: *"we use the estimated best arm, defined as the arm that most runs concluded to be the best at the end of training."* This is not true cumulative regret against the actual optimal arm — it is a post hoc proxy derived from the same runs being evaluated. Additionally, only node-selection regret is measured, not end-to-end reward under the full conditional-intervention problem.

---

## Strengths

- **Novel and complete theoretical contribution:** The LSCA closure characterization is genuinely new — this is the first complete minimal-search-space result for conditional single-node interventions. Both sufficiency and minimality are proven rigorously, which is the correct form of such a result.
- **Elegant reduction via Proposition 4:** Establishing that conditional-intervention superiority coincides with deterministic atomic-intervention superiority is a substantive conceptual result. It simplifies subsequent analysis and has independent interest.
- **Efficient algorithm with clean structure:** C4 is remarkably simple (11 lines), graph-only, and linear-time. The connector concept gives it strong intuitive grounding (Lemma 15).
- **Well-scoped problem:** The paper correctly identifies that single-node conditional interventions are *harder*, not easier, than multi-node hard interventions when characterizing the minimal search space, and uses this to motivate a non-trivial contribution.
- **Honest about scope:** The paper explicitly acknowledges that no-latent-confounder is a limitation and motivates it as a necessary step. The appendix honestly reveals the computational fragility of the experimental setup.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Bandit experiments use a non-standard, self-referential regret proxy** — The regret curves in Figure 3 are computed relative to "the arm that most runs concluded to be the best at the end of training" (footnote 11), not against the true optimal arm. This is a significant mismatch with the cumulative regret objective defined in Section 2. The proxy can favor whichever variant more consistently locks onto a self-confirming arm, which does not constitute validation of the formal regret objective. This weakens the abstract's claim of "substantially accelerates convergence rates" in the standard bandit sense.

- **Bandit experiments measure only node-selection regret, not the actual conditional-intervention problem** — The method's stated value is reducing the search space for conditional interventions, yet the evaluation decomposes this into node selection (measured) plus per-context UCB for policy selection (not evaluated end-to-end). The performance gain shown is consistent with the trivial expectation that a smaller candidate set speeds up node selection — it does not demonstrate improvement on the actual arm space or total reward in the full conditional-intervention problem.

- **No heuristic baselines in empirical evaluation** — The only comparator is "brute-force" (all ancestors). Comparing against simpler alternatives (e.g., restricting to Pa(Y) only, or to one hop of common ancestors) is essential to demonstrate that the full mGISS characterization provides practical benefit over naive graph-based pruning. Without this, it is unclear whether the specific structure of the LSCA closure matters or whether any reasonable pruning would yield similar regret reduction.

### Minor

- **Search-space reduction experiments use a cherry-picked target protocol** — In all experiments, Y is chosen as the node with the most ancestors (footnote 8). This maximizes the size of An(Y), giving the most room for pruning. This does not represent the distribution of targets in general causal bandit tasks. The broad empirical claim — that the method will be "especially effective" in real-world graphs — is not supported across the distribution of possible targets.

- **Observable conditioning set assumption needs sharper framing** — The assumption An(X)\{X} ⊆ Z_X is stated as natural (Section 2), and arguably is in the described problem setting. However, the paper does not discuss how results change when Z_X is restricted to a strict subset. This matters practically: in some settings, not all ancestors of X may be observable or cost-efficient to condition on.

### Trivial

- The abstract's final claim ("substantially accelerates convergence rates") would more accurately read "can substantially reduce node-selection regret" given what the experiments actually show.

---

## Nice-to-Haves

- **Formal regret bounds** connecting mGISS size reduction to improved regret bounds (e.g., showing the regret scales with |mGISS| rather than |An(Y)|). This would be the natural theoretical companion to the empirical regret curves and would substantially strengthen the paper.
- **Worst-case characterization:** Identify when mGISS = An(Y)\{Y} (no pruning). Dense graphs approach this, but a formal characterization of the "no benefit" regime would help practitioners decide a priori whether to apply C4.
- **Analysis under partial graph knowledge:** Even an informal discussion of how a single added/removed edge can affect mGISS would address practical concerns about graph misspecification.
- **Evaluation with varied targets:** Reporting mGISS size fractions across all non-trivial targets per graph (rather than only the most-ancestral one) would validate the broad empirical generalization claims.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **"No latent confounders limits applicability"** (Neutral Reviewer, Human Finder): The paper explicitly acknowledges this as a limitation and future work ("a natural next step for future work," Section 2). The scoping is honest and the single-node problem under no latent confounding is already highly non-trivial. This does not deserve inclusion as a weakness — it is part of the declared scope.

- **"Single-node restriction is too restrictive"** (Human Finder): The paper argues in Section 2 that restricting to single-node interventions *makes the problem more challenging*. This framing is correct and the contribution is precisely solving this harder case. Criticizing the scope is inappropriate here.

- **"g has no restrictions on the function is misleading"** (Human Finder): The paper states "We do not impose any restrictions on the function g" in the context of the conditional intervention, and footnote 3 clarifies that the conditioning set Z_X is pre-specified. The statement is precise and the comparison to "soft interventions" is a misapplication of a different paper's criticism.

- **Reproducibility / hyperparameter disclosure concerns**: The paper provides a code repository. No implementation details are unreasonably withheld.

- **"No comparison with Lee & Bareinboim's search space experimentally"**: The paper clearly explains why these are non-comparable settings (multi-node hard vs. single-node conditional). Requiring an experimental comparison across incompatible problem settings would be scope creep; the paper's theoretical discussion of the distinction is sufficient.

---

## Novel Insights

The most genuinely novel insight in this paper is the reduction of conditional-intervention superiority to deterministic atomic-intervention superiority (Proposition 4), which is *not* the obvious move. At first glance, conditional interventions (which allow policy g to depend on an observed context Z_X) seem strictly harder to reason about than simple atomic operations. The authors show they are equivalent — meaning that to determine whether node X is globally superior for conditional interventions, one needs only check the deterministic single-value setting. This is a conceptual simplification that likely has broader applicability beyond this paper's specific setting, and the authors appropriately call it out as a "surprising" supplementary result. The further characterization via Λ-structures (as opposed to the more obvious LCA characterization, which the authors show is insufficient) is also genuinely insightful and geometrically transparent.

---

## Suggestions

1. Replace the estimated-best-arm regret proxy with true regret against an oracle (known SCM) or independently computed benchmark; report at least one such experiment even at small scale.
2. Add at least one heuristic baseline (e.g., Pa(Y)-only pruning) in both the search-space fraction and regret experiments to demonstrate the necessity of the full LSCA closure.
3. Report mGISS fractions across all non-trivial targets per graph, not just the most-ancestral node, to support broad empirical claims.
4. Clarify in the abstract that the experimental validation addresses node-selection regret rather than end-to-end conditional-intervention regret.

---

## Score and Decision

**Calibration references:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| `uuriavczkL` | Counterfactual Realizability — strong causal theory + algorithm | 6, 10, 6, 8 | Accept (Spotlight) |
| `YcW8i9VCf5` | Adversarial Causal BO — causal bandit with formal results | 6, 6, 6, 6 | Accept (Poster) |
| `IPayPEGwdE` | Causal Contextual Bandits — weaker theory, poor experiments | 5, 5, 5, 5 | Reject |
| `MVpvyeVeyI` | Causal BO — split reviews, mixed theory/empirics | 10, 3, 8, 5 | Reject |

**Positioning:** The paper under review sits between `IPayPEGwdE` (5s, Reject) and `uuriavczkL` (avg ~7.5, Spotlight). Specifically:
- The theoretical core is meaningfully stronger and more novel than `IPayPEGwdE`, which drove that rejection.
- `YcW8i9VCf5` (6,6,6,6 Accept Poster) offers a useful anchor: it is a causal bandit paper with formal guarantees and clean experiments, similar in spirit. The paper under review has stronger theoretical novelty but worse experimental rigor.
- `uuriavczkL` features *complete* characterization results with broader scope; the paper here has a comparable degree of theoretical rigor but is narrower in scope (no confounders, single node).

The experimental weaknesses (non-standard regret, no heuristic baselines, cherry-picked targets) are real but do not undermine the theoretical core, which is the main contribution. Evaluated primarily as a theory paper with illustrative experiments — which is the fair characterization — this is a solid, complete, and publishable contribution.

**Overall assessment:** Strong on originality (novel problem formulation + complete characterization), strong on soundness of theory and algorithm, moderate on experimental depth, moderate on breadth (no confounders, single node). Appropriate as a poster-track contribution.

**Score: 6.5**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>