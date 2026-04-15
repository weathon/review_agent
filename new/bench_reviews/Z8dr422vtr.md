Now let me search for calibration papers to anchor the score.Now I have all the information I need. Let me produce the final consolidated review.

---

## Summary

This paper introduces the problem of *Cross-Domain Off-Policy Evaluation and Learning (OPE/L)* for contextual bandits, wherein logged data from multiple source domains supplements target-domain data to handle scenarios (deterministic logging, new actions, few-shot data) that conventional methods cannot address. The authors propose COPE, a new estimator that applies *multiple importance weighting* across a target-domain cluster for the "domain-cluster effect" and domain-specific reward regression for residual differences. They extend COPE to a policy gradient method (COPE-PG) and provide theoretical bias characterization (Theorem 3.1) alongside empirical evaluation on a synthetic multi-domain setup derived from the KuaïRec recommendation dataset.

---

## Claims and Support

**Claim 1: Cross-domain OPE/L is a useful new problem formulation that addresses scenarios conventional single-domain OPE/L cannot.**
*Partially supported.* The motivation is sound: source-domain support can cover target-domain gaps. However, the empirical evidence rests on a single synthetic multi-domain construction on one dataset, which does not test the full generality of the formulation (see Weakness 1 below).

**Claim 2: COPE effectively leverages target and source data with lower bias/variance than existing estimators under new actions, deterministic logging, and few-shot target data.**
*Partially supported.* Figures 2–4 do show COPE outperforming the chosen baselines. However, a critical structural limitation undermines this claim: in the experiments, rewards are sampled with the same underlying mean `q(x_u, a)` for all domains (Section 4, Eq. 13–14 and the sentence "We sample the reward r^k from a normal distribution with mean q(x_u, a) and standard deviation σ = 1"). There is no domain-specific reward component `q^k(x_u, a) ≠ q^T(x_u, a)`. This means `h(x, a, k) = 0` for all k in Eq. (8) — the CPC condition (Condition 3.2) is trivially satisfied throughout, and the bias from Theorem 3.1 is identically zero regardless of regression model quality. The evaluation thus tests data-sharing under support deficiency but not the paper's core reward-decomposition mechanism.

**Claim 3: COPE can be unbiased even under deterministic logging and new actions.**
*Supported as a conditional theoretical statement.* Corollary 3.1 is formally correct under Conditions 3.1 and 3.2. The paper itself appropriately disclaims that CPC is not expected to hold in practice ("It is important to clarify that we do not expect Condition 3.2 to hold in practice"). However, the abstract/introduction still leans on the unbiasedness framing.

**Claim 4: COPE is a strict generalization of DR and DR-ALL and "never performs worse" with good tuning.**
*Supported by construction for the formal reduction; the "never performs worse" statement is tautologically correct if oracle tuning is granted (since DR and DR-ALL are special cases at |φ(T)|=1 and |φ(T)|=K), but the paper does not provide evidence that practical data-driven tuning reliably achieves this.*

**Claim 5: COPE-PG outperforms existing OPL methods under new actions and deterministic logging.**
*Partially supported.* Figures 5–6 do show COPE-PG winning across the three test axes, subject to the same shared-reward-function caveat.

---

## Strengths

- **Novel and practically motivated problem formulation.** Cross-domain OPE/L is a genuinely new setup that addresses the union of three distinct failure modes (new actions, deterministic logging, few-shot data). The hospital/country/segment motivations are concrete and realistic.

- **Principled estimator design.** The reward decomposition into cluster and domain-specific effects, the use of multiple importance weighting for the shared component, and regression for domain differences form a coherent, well-motivated structure. The connection to DR and DR-ALL through |φ(T)| as a spectrum lever is insightful.

- **Useful theoretical analysis.** Theorem 3.1's explicit bias expression is more informative than a vague robustness claim. The Common Cluster Support condition (Condition 3.1) is a meaningful generalization of standard common support, and the paper correctly identifies when unbiasedness holds.

- **Broad empirical scope.** The experiments stress-test three practically relevant axes (new-action ratio, deterministic-logging prevalence, target data size) and decompose MSE into bias² and variance, providing a detailed picture of each method's failure mode.

- **Complete OPE and OPL coverage.** Unlike the related OffCEM estimator (OPE only), this paper also extends to policy learning, providing a complete solution for the problem.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Experimental setup does not instantiate cross-domain reward heterogeneity — the core mechanism the paper claims to solve.** In Section 4, rewards in every domain are sampled from N(q(x_u, a), σ=1) — the **same underlying function** for all k. Domain differences are entirely through user distributions p^k(u) and logging policies π_0^k, not through reward functions. This means q^k(x, a) = q^T(x, a) for all k, so the domain-specific component h(x, a, k) = 0 identically and CPC is trivially satisfied throughout. The experiments therefore validate that COPE can borrow action-coverage from source domains, but they do not test COPE's bias-correction mechanism for cross-domain reward differences — which is the novel methodological contribution distinguishing COPE from simpler pooling strategies. Results may not generalize to settings with genuine reward heterogeneity across domains.

- **Clustering is heuristic and unvalidated.** Clustering is a critical component: the bias of COPE (Theorem 3.1) depends on how well domains in the same cluster satisfy CPC, and the practical performance depends on how well the heuristic (empirical average reward similarity) identifies meaningful clusters. The main paper fixes |φ(T)| = 4 without any ablation on clustering quality (e.g., random clusters, oracle clusters, adversarial clusters). Figure 6(right) only varies cluster *size* for OPL and does not test misspecification. There is no evidence that the heuristic recovers meaningful clusters from logged data under realistic distribution shifts.

### Minor

- **Single dataset, synthetic domain construction.** All experiments use one dataset (KuaïRec) with artificial domain construction. Domains that naturally differ in reward structure (e.g., hospital settings, different geographic regions) are not evaluated. This limits the confidence in generalizability.

- **Incomplete baselines.** The comparison includes only naive single-domain (IPS/DR/DM with target-only data) and naive pooling (IPS-ALL/DR-ALL). Structured domain adaptation baselines (e.g., reward-reweighted transfer, representation alignment) are absent. A simple ablation comparing COPE to a cluster-restricted DR that uses source-cluster data but without the multiple importance weighting decomposition would isolate the contribution of the proposed structure.

- **No theoretical analysis for COPE-PG.** Despite OPL being a major contribution, the policy-gradient extension lacks a corresponding bias/variance theorem analogous to Theorem 3.1. The paper provides only the OPE-level analysis.

- **"Never performs worse" requires oracle tuning.** The claim in Section 3.1 that "COPE never performs worse than DR or DR-ALL with a good (if not perfect) tuning" is correct by construction (since both are special cases) but practically requires reliable data-driven hyperparameter selection. The paper cites relevant tuning literature but does not demonstrate it works in the cross-domain setting.

### Trivial

- The expected value expression in Theorem 3.1 uses ambiguous notation (p^T(x) vs. π^T(x)) that appears to be a formatting artifact but slightly impedes readability.

---

## Nice-to-Haves

- Add a second dataset with naturally occurring domain shifts (e.g., different hospitals, geographic regions, or user segments with distinct reward functions) to substantiate the generality claims.
- Ablate clustering quality: compare oracle clusters (ground-truth reward-function similarity), the proposed heuristic, and random clusters to measure sensitivity.
- Provide an experiment with non-trivial domain-specific reward differences (e.g., construct domains where q^k(x,a) are explicitly different fractions of the underlying matrix) to test the core mechanism under CPC violation.
- Provide bias bounds in terms of degree of CPC violation to give practitioners guidance on when COPE is reliable.
- Analyze sensitivity of COPE to density-ratio estimation quality, since errors in p^{φ(T)}(a|x) propagate into importance weights and affect both bias and variance.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Computational overhead criticism (Human Finder reviewer):** Removed as a pure nitpick — standard density-ratio estimation and reward regression are not unusual for this literature and the paper is not a systems paper obligated to benchmark runtime.
- **Concern about "cannot be independently verified" or doubts about the KuaïRec dataset's existence/availability:** No reviewer raises this explicitly, but any such concern would be removed per hard rules.
- **Asymmetric comparison (using target-only methods as baselines):** Removed — the paper compares against naive pooling (IPS-ALL, DR-ALL), which is already a harder comparison for the authors' method since it risks naive-pooling bias. The asymmetry does not favor the authors.
- **Reproducibility/hyperparameter nitpicks:** Removed — the paper gives sufficient configuration details for the experimental setup (|φ(T)|=4, Random Forest, 3-fold cross-fitting, ε=0.2, etc.).

---

## Novel Insights

The paper's most genuinely novel observation is the *spectrum interpretation* of COPE: by varying the cluster size |φ(T)| from 1 (recovering DR) to K (recovering DR-ALL), COPE unifies two previously orthogonal estimator design choices — target-only estimation and naive cross-domain pooling — into a single family. The insight that the only structural requirement on the regression model for unbiasedness is that it preserve *relative* reward differences across domains (CPC), rather than absolute reward values, is a meaningful relaxation relative to conventional DR. Connecting this to the multiple importance weighting literature is natural but well-executed.

---

## Suggestions

1. **Redesign the simulation to introduce domain-specific reward functions.** Concretely, use a different subset or weighted combination of the KuaïRec reward matrix per domain to induce q^k(x,a) ≠ q^T(x,a). This directly tests the bias-correction mechanism.
2. **Provide a principled clustering ablation.** Test at least three cluster configurations (oracle, heuristic, random) to measure how much clustering quality matters.
3. **Add a theorem for COPE-PG.** A parallel bias/consistency result for the policy gradient estimator is needed given OPL is a major claim.
4. **Tone down the abstract's unbiasedness message.** Distinguish clearly between "formally unbiased under CPC (which doesn't hold in practice)" and the practical robustness argument based on Theorem 3.1.
5. **Evaluate data-driven cluster size tuning.** Apply one of the cited OPE tuning methods (Su et al., 2020b; Felicioni et al., 2024) to select |φ(T)| from logged data and report performance, since the "never worse" claim depends on it.

---

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Decision | Scores |
|---|---|---|---|
| POTEC (LXftdR11io) | OPL reward decomposition + clustering, contextual bandits | Accept (Spotlight) | 8, 6, 8, 8 (avg ~7.5) |
| LNkMWCEssX | Domain adaptation for contextual bandits | Accept (Poster) | 6, 6, 8 (avg ~6.7) |
| mUbYof5MKp | OPL framework, novel setting | Accept (Poster) | 6, 6, 6, 6 (avg 6.0) |
| 89EjtiGWVS | Novel OPE/L estimator, single-domain | Withdrawn | 6, 5, 6, 6 (avg ~5.75) |

The paper under review is closest in spirit to POTEC: reward decomposition, clustering, both OPE and OPL, contextual bandit setting. POTEC earned Spotlight (avg 7.5) with stronger evaluation (tested larger action spaces with meaningful decomposition structure, multi-setting analysis, and theory for the policy gradient). The paper under review is below POTEC because:
- The core reward-decomposition mechanism is not stress-tested (shared reward table issue)
- Heuristic clustering without ablation
- No theoretical analysis for the policy gradient

It is comparable to or slightly above LNkMWCEssX (avg 6.7): the problem formulation is more novel, the method is more carefully designed with bias analysis, and empirical coverage is broader. The key limiting factor is the evaluation gap.

It is clearly above the 89EjtiGWVS withdrawn paper (avg 5.75): the problem is more novel, the theory is more informative, and the method is more principled.

**Assessment:** Good problem identification, principled method, and sound theory; weakened primarily by an evaluation design that does not test the hardest aspect of the paper's own mechanism. The result is a paper that makes a larger claim (handling cross-domain reward heterogeneity) than what is demonstrated (handling support deficiency through cross-domain data pooling). The contribution is real but narrower than advertised.

**Final score: 5.5 — marginally below the accept borderline**, positioned between the withdrawn LSE paper and the accepted domain adaptation poster, reflecting the genuine novelty of problem formulation and the principled method against the significant evaluation gap.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>