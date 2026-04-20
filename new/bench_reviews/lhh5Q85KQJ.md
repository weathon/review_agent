## Summary
The paper proposes SubDisMO, a resource-aware distributed optimization algorithm that trains adaptively-sized submodels with SAM-style perturbations to mitigate "arbitrary submodel sharpness" in federated learning. It provides a convergence rate of O(1/√(QTC*)) that unifies several prior algorithms as special cases, a generalization bound incorporating per-layer parameter remaining rates, and experiments on CIFAR-10/100 showing consistent improvements over submodel baselines on most (not all) settings.

## Strengths
- **Unifying convergence rate via minimum covering number C***: Theorem 1 and Corollary 1 derive an O(1/√(QTC*)) convergence rate that recovers four prior algorithms as special cases (RAM-Fed, FedSAM, FedAvg, OAP) under specific settings. The explicit incorporation of C* — the minimum number of clients training any given parameter — into a convergence bound is a meaningful theoretical contribution for resource-aware FL.
- **Generalization bound with per-layer remaining rates**: Theorem 2 introduces the per-layer parameter remaining rate s_j into a PAC-Bayesian generalization bound (Eq. 17). The product term ∏(s_j + 1/r)² being strictly smaller than the full-model bound when s_j < 1 directly links submodel pruning to tighter generalization — this has not appeared in prior federated pruning literature.
- **Empirical validation of C* and perturbation impact**: Figure 4 provides clean empirical evidence that higher C* yields faster convergence and higher accuracy, consistent with Corollary 1. Figure 3's loss landscape visualization effectively demonstrates the perturbation's flattening effect on the submodel-trained model.
- **Consistent gains across most settings**: Table 1 shows SubDisMO outperforms submodel baselines on 11 of 12 settings, with the remaining setting (CIFAR-100 Dir(μ=1.0)) being a modest regression.

## Weaknesses

### Fatal
None.

### Major

- **False claim of universal superiority: SubDisMO underperforms a submodel baseline on CIFAR-100 Dir(μ=1.0)**: The paper states "our proposed SubDisMO outperforms other baselines in terms of average accuracy" (§5.2, line 298), but Table 1 shows SubDisMO at 25.43% vs. OAP.O at 26.09% on CIFAR-100 Dir(μ=1.0) — a 0.66% deficit. The paper also claims gains of "0.55%–1.26% on CIFAR-100," which cherry-picks the IID case (0.55% gap) and ignores the Dir(0.5) gap of 1.26%. This misrepresentation of results undermines confidence in the reported empirical contributions. This matters because it suggests the benefits of perturbation may be more conditional than claimed, and overclaiming on an easy-to-verify metric is a red flag for the rest of the evaluation.

- **Convergence bound excludes untrained parameters (S \ K_q), limiting the meaning of the global guarantee**: Theorem 1 and Corollary 1 bound (1/Q)∑_q∑_{l∈K_q} E‖∇f^l(θ_q)‖² — the squared gradient norm only over parameters trained in round q. In a submodel setting where parameters dynamically change membership in K_q, untrained parameters S\K_q may accumulate drift or stagnate between updates. Remark 2 (§4.1, line 203 acknowledges this: "we innovative analyze... so we give a rigorous bound of the averaged gradient of the trained parameters." However, scoping the result to trained parameters only means the proof does not guarantee global convergence — the overall loss could still be dominated by un-updated parameter drift. This matters because the paper's framing ("asymptotically optimal global convergence rate") implies a guarantee on the full model, which the mathematics does not provide.

- **No SAM+submodel control isolates the perturbation contribution**: The experiments include many submodel baselines (IST, PruneFL, OAP, FedRolex) with various aggregation schemes, and RAM-Fed as a submodel+non-iid baseline. However, none of these baselines apply perturbation. SubDisMO uniquely adds a perturbation step. Without a SAM+OAP or SAM+PruneFL baseline, it is impossible to determine whether the improvements come from the specific aggregation/minimax formulation or simply from applying SAM-style perturbation to any submodel pipeline. This matters because it weakens the claim that SubDisMO's specific design is necessary for the observed gains.

- **Mask policy is theoretically adaptive but empirically random**: Algorithm 1 uses a "resource-aware adaptive mask policy P(θ_q; R_n)" (§3, line 66). However, Section 5.1 (§5.1, "Submodel setting," line 292) states: "we randomly split the full model into four submodels... without overlap." The experiments use hardcoded random non-overlapping splits, not adaptive masking based on resources. This disconnect between the theoretical framing (adaptive, resource-aware masking) and the experimental implementation (random fixed splits) makes it unclear whether the convergence analysis applies to the experiment or vice versa.

### Minor

- **The "minimax" framing overstates the relationship to distributed minimax optimization**: The paper's central objective (Eq. 2) and perturbation step (Eq. 5) are structurally identical to SAM's perturbation framework — a one-step first-order Taylor ascent on the input-space perturbation ε. While SAM can be formulated as minimax, the related work section (§1–2) discusses distributed minimax methods (FedGDA-GT, FedSGDA+, LocalSCGDAM) that involve distinct adversarial variables, dual coordination, and multiple inner ascent steps. By placing the SAM-style perturbation under this umbrella without clearly differentiating the two problem classes, the paper conflates distinct areas of optimization. This could mislead readers about the scope of the theoretical contribution.

- **Non-standard gradient variance assumption (Assumption 3)**: Assumption 3 bounds the variance of *normalized* stochastic gradients: E∥(∇f_n/‖∇f_n‖) − (∇f_n/‖∇f_n‖)∥² ≤ σ_l². This is not standard in FL convergence literature. The normalizer introduces nonlinear coupling that is difficult to justify in stochastic settings — specifically, the expectation of normalized stochastic gradients is not the normalized expected gradient. While the resulting bound may be tighter (σ_l² ≤ π² as noted), it is unclear that this assumption holds in practice for FL with submodel masking and perturbation.

- **High variance in submodel baselines limits statistical reliability**: Table 1 reports standard deviations up to 16.67% (IST.O on CIFAR-10 Dir(μ=0.5)) and 14.90% (PruneFL.O). The gaps between SubDisMO and second-best methods (e.g., 2.97% on CIFAR-10 Dir(μ=0.5)) are often smaller than the within-baseline variance, making it unclear whether improvements are statistically significant. No statistical tests (e.g., t-tests, paired bootstrap) are provided.

### Trivial
- The paper's claim of "1.52%-2.97% on CIFAR-10" (line 298) is correct for CIFAR-10 but the analogous "0.55%-1.26% on CIFAR-100" is misleading because it excludes the setting where SubDisMO underperforms.
- The convergence rate notation alternates between C* and 𝒞* (Definition 1 uses 𝒞*, Corollary 1 uses C*), creating minor notational inconsistency.

## Nice-to-Haves
- A compute-overhead analysis (FLOPs, memory) comparing SubDisMO to submodel baselines would help justify the resource-efficiency claim given the doubled per-iteration gradient computation.
- Empirical tracking of how parameters outside K_q evolve over rounds would strengthen the global convergence argument.
- Confidence intervals or statistical significance tests on Table 1 results would improve the reliability of accuracy comparisons given the high variance in submodel methods.
- Testing an adaptive (rather than random) mask policy would better align with the theoretical framework.

## Removed Points
- **REMOVED: "SAM perturbation disguised as minimax is a categorical error"**: The minimax framing of SAM is standard in the FL literature (FedSAM, the paper's own comparison baseline, also frames SAM as minimax optimization). The formulation is a valid minimax objective — it differs from adversarial training or GAN-style minimax in that the adversarial variable is an input-space perturbation rather than a model parameter, but this is a matter of problem class, not an error. The harsh critic overstates this.
- **REMOVED: "Step-size condition η_g ≤ 2√N/√C* contradicts stability requirements"**: This condition is an *upper bound* on η_g for the proof to hold; Corollary 1's actual step-size choice η_g = √C*/√T *decreases* with C*, which is consistent with standard partial-participation stability. The proof condition and the practical choice go in opposite directions, but this is a proof conservatism issue, not a contradiction.
- **REMOVED: "Doubles per-step gradient computations, contradicting resource premise"**: The paper acknowledges the additional computation in Remark 3: "we improve generalization through a little computation but without slowing down the convergence rate." Whether this is acceptable in the "resource-limited" setting is a design choice, not a contradiction — SAM has always had this overhead.
- **REMOVED: "Arbitrary submodel splitting destroys Transformer representational capacity"**: This is an inherent challenge of any submodel/parameter-pruning approach for Transformers, not unique to this paper. The paper compares against other submodel methods using the same splitting strategy, making the comparison fair within its category.

## Novel Insights
The paper's most valuable contribution is not the algorithmic design (which is essentially SAM applied to federated submodel training), but the explicit identification and theoretical quantification of C* — the minimum per-round parameter coverage — as the dominant factor in submodel FL convergence. This reframes the submodel problem from "how to prune/aggregate" to "how to ensure sufficient parameter coverage," which could influence future work on resource-aware FL. The per-layer s_j generalization bound similarly provides a lens for understanding why some pruning strategies may generalize better than others.

## Suggestions
1. **Correct the performance claim**: Acknowledge explicitly that SubDisMO underperforms OAP.O on CIFAR-100 Dir(μ=1.0). Report the full range of results including regressions.
2. **Add a SAM+pruning baseline**: Add at least one baseline combining SAM perturbation with an existing submodel method (e.g., OAP+SAM) to isolate the contribution of the specific aggregation/minimax logic.
3. **Clarify scope of convergence bound**: Add a discussion of the limitation that the bound only covers trained parameters K_q, and characterize under what conditions (e.g., C* ≥ 1 with random assignment) untrained parameter drift is negligible.
4. **Clarify the mask policy**: Distinguish between the general framework (which allows adaptive masking) and the experimental instantiation (which uses random non-overlapping splits).

## Score and Decision
After comparing against calibration anchors:
- **Accepted papers with similar patterns**: FedP3 (scores 6,8,5,6, accepted) — practical contribution, decent experiments, but incomplete theory and missing ablations was still accepted due to clear practical value.
- **Papers with proof concerns accepted**: The Momentum FL paper (scores 5,5,8,5, accepted) had reviewer-flagged proof errors but was accepted based on empirical validation and the insight.
- **Papers withdrawn for weak experiments/theory mismatches**: FedMAP (3,5,5,3, withdrawn) had convergence analysis without matching empirical convergence.

This paper sits somewhere between these. It has real empirical results (Table 1, Figures 3-4) and a unifying convergence rate, but also has a demonstrated result misrepresentation, a partial convergence guarantee, and missing baselines. It is better than FedMAP (the experiments work) but weaker than FedP3 (the proof gaps are more central to the claims). It is roughly comparable to the Momentum FL paper: solid empirical contribution, but theoretical limitations that reviewers would weigh against acceptance.

The misrepresentation of CIFAR-100 Dir(1.0) results is particularly concerning because it appears intentional — the authors had access to the table and chose not to report the regression. This damages overall confidence in the evaluation.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>