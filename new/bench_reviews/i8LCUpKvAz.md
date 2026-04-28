## Summary
This paper introduces EQO (Exploration via Quasi-Optimism), a tabular RL algorithm that achieves minimax optimal regret without requiring empirical variance estimates. The key theoretical innovation is the "quasi-optimism" framework, which relaxes the conventional requirement for full optimism while maintaining tight regret bounds under weaker boundedness assumptions. Experiments on RiverSwim demonstrate superior regret performance against five baselines.

## Strengths
- **Genuine theoretical novelty (quasi-optimism framework)**: The paper introduces a provably valid relaxation of full optimism (Lemma 2: V_h^k(s) + (3/2)λ_k H ≥ V_h^*(s)), enabling a simple c/N bonus without empirical variance while achieving minimax optimal regret. This challenges the prevailing assumption that variance estimates are necessary for tight bounds.

- **Tightest known regret bound under weakest assumptions**: Table 1 clearly shows EQO matches the best-known bound (H√(SAK) + HS²A) while requiring only bounded value (0 ≤ V_h^*(s) ≤ H) rather than bounded reward or return. Section 4.1 explains why this assumption is strictly weaker than prior work.

- **Algorithmic simplicity with empirical validation**: Algorithm 1 eliminates the need to track second moments, reducing implementation complexity. Figure 1 demonstrates EQO outperforms UCRL2, UCBVI-BF, EULER, ORLC, and MVP on RiverSwim (S=30, H=120 and S=40, H=160), a challenging exploration benchmark.

- **Sound proof technique using Freedman's inequality**: The analysis (Section 4.4) decouples variance and 1/N terms via Lemma 1, avoiding the need to alternate between expected and sampled trajectories when bounding √(Var/n) terms. This is a legitimate technical contribution.

## Weaknesses

### Fatal
None

### Major
- **Limited empirical scope for "practical effectiveness" claims**: The paper claims EQO offers "the best of both theoretical soundness and practical effectiveness" (Abstract) and "superior empirical performance" (Contribution 5). However, evaluation is restricted to RiverSwim (Section 5), an environment specifically designed to test deep exploration and known to favor aggressive exploration bonuses. EQO's bonus structure (c/N with c ∝ √K) is significantly more aggressive than standard UCB bonuses (√(1/N)) for low visit counts. Without evaluation on environments where over-exploration is costly (e.g., low-variance MDPs, domains with negative rewards for unnecessary exploration, or standard benchmarks like GridWorld), the claim of general "practical effectiveness" is not well-supported. This is a significant gap between the paper's positioning and its empirical validation.

### Minor
- **"Computational efficiency" claim is somewhat misleading**: The Abstract and Introduction repeatedly highlight "computational efficiency" as a key contribution. However, in the tabular RL setting, maintaining and updating empirical variances is an O(1) operation per state-action visit, identical in complexity class to updating visit counts. The dominant cost in Algorithm 1 is the Value Iteration sweep (Lines 7-11), which is O(HS²A) or O(HSA) per episode regardless of the bonus term. The actual benefit is **implementation simplicity** (no second-moment tracking) rather than asymptotic computational efficiency. This distinction should be clarified to avoid overstating the advantage.

- **No parameter sensitivity analysis for c**: The theoretical value of c depends on logarithmic factors (ℓ₁, ℓ₂,K) and confidence δ, which are often conservative in practice. Theorem 1 requires knowledge of total episode count K to set c optimally; Theorem 2 provides an anytime version via doubling trick. The experiments do not clarify which setting was used, and there is no analysis showing how regret varies with c. If tuning was required to achieve the plotted performance, the claim of "convenient control through a single parameter" (Contribution 1) is weakened. If the theoretical c was used directly, its magnitude (scaling with √K) could lead to excessive exploration in large-horizon tasks—a risk not discussed.

### Trivial
- **Experimental setup details missing**: Figure 1 does not specify error bars, number of seeds, or whether the fixed-horizon (Theorem 1) or anytime (Theorem 2) parameter setting was used. While the appendix reportedly contains execution time data (Table 4), the main text should clarify these methodological details for reproducibility.

## Nice-to-Haves
- Add experiments on at least one additional domain (e.g., Random MDP or GridWorld) to demonstrate EQO's performance does not degrade in environments where RiverSwim-style aggressive exploration is not required.
- Include a bonus comparison plot visualizing EQO's c/N bonus versus UCBVI's √(1/N) bonus over time for a representative state-action pair to illustrate the difference in exploration aggression.
- Clarify whether the bounded value assumption (Assumption 1) enables EQO to handle any practically relevant MDPs that violate bounded reward/return assumptions but satisfy bounded value.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Bonus magnitude causing numerical instability**: The harsh critic raised concerns that "with large K, this could lead to overflow or numerical instability in V_h^k propagation." However, the paper's quasi-optimism analysis (Lemma 2) explicitly controls underestimation by a bounded amount ((3/2)λ_k H), and Line 10 caps Q_h^k at H when N=0. This is a speculative concern not grounded in evidence from the experiments.

- **Clipping mechanism for Q-values**: The critic noted "there is no clipping mechanism for Q_h^k when N>0." However, the algorithm does cap Q at H for unvisited states (Line 10), and the theoretical analysis bounds the overestimation (Lemma 3: V_h^k(s) - V_h^π^k(s) ≤ (5/2)λ_k H + 2U_h^k(s)). The concern partially misreads the algorithm's safeguards.

- **Dependency on horizon K making experiments unclear**: While the critic noted experiments don't clarify which K-setting was used, Theorem 2 explicitly provides an anytime version, and the paper states c_k can be set as "a constant independent of k" (footnote 4). This is a minor presentation issue, not a methodological gap.

- **Claim about transferability to function approximation being speculative**: The Conclusion states the idea is "transferable to... general function approximation." While this is forward-looking, the paper appropriately frames it as anticipated future work ("we anticipate that the underlying idea will be transferable"). This is standard positioning, not overclaiming.

## Novel Insights
The quasi-optimism framework represents a genuine conceptual shift in optimism-based analysis. Prior work universally required V_h^k(s) ≥ V_h^*(s) (full optimism), which necessitated variance-dependent bonuses to tightly bound estimation error. By relaxing this to V_h^k(s) + (3/2)λ_k H ≥ V_h^*(s) (quasi-optimism), the paper demonstrates that variance estimates are not strictly necessary for minimax optimality. The key technical insight is allowing controlled underestimation of the transition error term (I₁ in Eq. 1) while bounding the resulting propagation error (I₂) through a novel variance-sum bound (2HV_h^*(s) - (V_h^*)²(s) ≤ H²). This decoupling via Freedman's inequality (Lemma 1) rather than Bernstein-type bounds is the mechanism enabling variance-free bonuses.

## Suggestions
1. **Temper empirical claims**: Revise the Abstract and Contributions to frame EQO as demonstrating strong performance on exploration-challenging benchmarks (RiverSwim) rather than claiming general "practical effectiveness." Add language acknowledging that evaluation on additional domains would further validate practical utility.

2. **Clarify efficiency framing**: Replace "computational efficiency" with "implementation simplicity" or "reduced implementation complexity" throughout, since the asymptotic complexity class is unchanged. If runtime savings exist, provide a per-episode breakdown distinguishing bonus computation from planning cost.

3. **Add parameter sensitivity plot**: Include a figure showing regret versus c to verify whether the theoretical value is practical or requires tuning. Clarify which parameter setting (Theorem 1 or 2) was used in experiments.

4. **Discuss bonus magnitude**: Add a brief discussion of the practical magnitude of c/N for large K (e.g., K=10⁶ implies c ≈ 1000H) and why this does not cause numerical issues or excessive early regret compared to √(1/N) bonuses.

## Calibration and Score
I compared this paper against the following anchors:

| Anchor | Avg Score | Comparison |
|--------|-----------|------------|
| **fE0RJto3Na** (6.5) | 6.50 | Fine-grained gap-dependent regret for model-free RL with experiments on synthetic MDPs. Stronger empirical coverage than EQO, but EQO has more novel theoretical framing (quasi-optimism vs. analytical framework extension). |
| **lbLAgGF8OO** (7.5) | 7.50 | Improved DEC framework removing optimism mechanism. Similar theoretical novelty to EQO, but with broader applications (adversarial MDPs, hybrid settings). EQO is more narrowly scoped. |
| **2XSP20jV0T** (6.0) | 6.00 | First approach for general-utility MDPs in single-trial regime. Authors upfront about limited experiments. Very similar profile to EQO—strong theory, limited but honest empirical validation. |
| **a4z7OlgSxC** (4.5) | 4.50 | Posterior-sampling Q-learning with theory-experiment mismatch (analyzed algorithm differs from evaluated). EQO is stronger because it evaluates the actual analyzed algorithm. |
| **tgcbMml49n** (4.0) | 4.00 | List-replicable RL with prohibitive sample complexities (H²⁴S¹¹) and disconnected experiments. EQO's theory is cleaner and experiments are more aligned. |

EQO sits between **2XSP20jV0T (6.0)** and **fE0RJto3Na (6.5)**. The theoretical contribution is genuinely novel (quasi-optimism is a new concept, not just an extension), but the empirical validation is limited to one domain. Unlike **a4z7OlgSxC (4.5)**, there is no theory-experiment mismatch—EQO evaluates the actual algorithm. The overclaiming on "practical effectiveness" is a concern but not fatal, as the RiverSwim results do demonstrate the algorithm works. Following the pattern of **2XSP20jV0T**, where limited experiments were accepted because the authors were upfront and the theory was solid, EQO warrants a similar score.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>