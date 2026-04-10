## Summary
This paper proposes Constraint-aware Reward Relabeling (CARL), a wrapper method for offline safe reinforcement learning (OSRL). CARL iteratively updates a cost Q-function and relabels rewards with a large penalty for state-action pairs predicted to violate a safety constraint. The method is minimalist, introduces no new hyperparameters, and can be combined with standard offline RL algorithms. Empirical evaluation on the DSRL benchmark shows CARL consistently produces safe policies under tight cost budgets while maintaining competitive reward performance.

## Strengths
- **Effective and Simple Wrapper Design:** The core algorithmic idea—alternating cost critic updates with reward relabeling—is conceptually straightforward and practically implemented as a lightweight wrapper around existing offline RL backbones (e.g., TD3-BC, IQL). This design achieves strong safety performance without introducing task-specific hyperparameters or complex constrained optimization, enhancing accessibility and ease of use.
- **Consistent Safety Under Tight Budgets:** CARL reliably satisfies cost constraints (normalized cost ≤ 1) across all 8 Bullet tasks and on 8 of 11 SafetyGym tasks under stringent budgets (κ=5, 10), as shown in Table 1. This safety consistency outperforms prior methods, which often violate constraints on several tasks.
- **Robustness and Flexibility:** The method works effectively with different offline RL backbones (TD3-BC and IQL, Table 2) and demonstrates a notable ability to learn safe policies even when trained only on unsafe trajectories (Figure 3), highlighting its constraint-enforcement capability.

## Weaknesses
### Major:
- **Decoupling of Theoretical Formulation and Implementation Undermines Foundational Claims:** The paper's central theoretical contribution (Theorem 1) proves that solving the unconstrained problem with penalty `-V_max` (the maximum infinite-horizon return) yields a policy satisfying the state-action-wise safety constraint (Equation 2). However, the main empirical results in Table 1 use `-R_max` (the maximum immediate reward from the dataset) as the penalty. The authors note this is a practical choice and include an ablation with `V_max` in an appendix (Table 5), but they do not justify why the theoretically required penalty is abandoned or analyze the conditions under which the weaker penalty still yields safe policies. This creates a significant gap: the strong theoretical guarantee does not apply to the primary algorithm being evaluated, weakening the paper's conceptual foundation.
- **Insufficient Evaluation of Safety in a "One-Shot" Critical Context:** The paper motivates the need for pointwise safety constraints (Equation 2) by appealing to "one-shot" safety-critical deployments where any violation is unacceptable. However, the evaluation only reports the *average* normalized cost over 20 episodes. A policy with an average cost below the threshold could still violate the constraint in a substantial fraction of episodes, which would be catastrophic for the stated application. The paper does not report worst-case costs, violation rates, or other distributional metrics necessary to validate the claim of robust, per-deployment safety.
- **Empirical Claims of Superiority Lack Statistical Rigor:** While Table 1 shows CARL's strong performance, the paper makes comparative claims (e.g., "outperforms prior methods on a greater number of tasks") without statistical significance testing. Standard deviations are often large, and for many tasks, the performance difference between CARL and other safe baselines (e.g., FISOR, CAPS) appears small. Claims of superiority should be supported by appropriate statistical tests across tasks and seeds to be convincing.

### Minor:
- **Incomplete Analysis of Algorithmic Stability and Convergence:** The paper identifies oscillation as a failure mode for large `K` and `M` (Figure 1) and heuristically selects `K=M=1` for stability. While empirically effective, this choice lacks formal analysis. The paper acknowledges convergence guarantees are "an open problem," but a deeper empirical analysis (e.g., tracking policy and critic drift) or discussion of sufficient conditions for stability would strengthen the methodological contribution.
- **Limited Investigation of Failure Modes:** CARL fails to satisfy constraints on several SafetyGym tasks (e.g., CarCircle1, CarCircle2 in Table 1). The paper does not analyze why—whether due to cost function over/under-estimation, poor generalization, or dataset coverage issues. Understanding these limitations is crucial for practitioners and for future improvements.

### Trivial:
- **Minor Presentation Issue:** The choice of `R_max` over `V_max` for the main results is mentioned only briefly in Section 6.2 and relegated to an appendix ablation. This is a major design choice that should be motivated and discussed more prominently in the main text, given its departure from the theoretical setup.

## Nice-to-Haves
- A runtime/computational efficiency comparison with more complex baselines (e.g., diffusion-based FISOR) would highlight CARL's practical advantage as a simple wrapper.
- An ablation studying the sensitivity of performance to the penalty magnitude (between `R_max` and `V_max`) could provide practical guidance and better connect the implementation to the theory.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

**Strengths that were removed:**
- "The paper is well-written." (Generic strength)
- "The topic is important." (Generic strength)
- "The experiments are extensive." (Generic; replaced with specific praise for breadth on DSRL benchmark)

**Weaknesses that were removed:**
- **"Theoretical analysis is incomplete." (Too vague; replaced with the specific, major point about the theory-implementation mismatch.)**
- **"Performance trade-off in some tasks." (Weakness partially valid but rephrased; the paper already acknowledges that safe policies may have lower reward than unsafe baselines, which is inherent to the problem, not a specific flaw of CARL.)**
- **"Missing comparison to a tuned Lagrangian baseline." (Scope creep; the paper adequately compares to published Lagrangian methods (CPQ, COptiDICE) and its contribution is a Lagrangian-free alternative. Demanding a re-implementation and exhaustive tuning of all Lagrangian variants is beyond standard evaluation.)**
- **"Missing comparison to simple reward shaping (`r' = r - λc`)." (Scope creep; the penalty-based baseline is a well-known and often insufficient approach, and the paper's contribution is a more sophisticated iterative relabeling scheme. Its omission is not a critical flaw.)**
- **"Needs uncertainty estimation for the cost critic." (Nice-to-have or next step; while valuable, it is not a standard requirement for the current contribution.)**
- **"Needs analysis of how often relabeling occurs." (Interesting but not a core weakness; it is an analysis depth suggestion, not a flaw in the method's validation.)**
- **"Concerns about cost function underestimation." (Strawman; the paper does not claim its OPE is perfect, and it uses standard FQE. This criticism is a general challenge for all offline RL, not a specific shortcoming of CARL's design.)**
- **"Scalability concerns." (Unsubstantiated; no evidence is provided that CARL would fail in higher dimensions, and it builds on scalable offline RL backbones.)**

## Suggestions
- **Address the Theory-Implementation Gap:** Revise the paper to either: (1) theoretically justify why the `R_max` penalty can be sufficient under certain conditions (e.g., with a well-covered dataset), formally linking the practical implementation back to the safety guarantees; or (2) pivot the narrative to present the `V_max` penalty as the canonical method, with `R_max` presented as a effective, empirically-derived heuristic, and move the main results to use `V_max`.
- **Strengthen the Safety Evaluation:** Report additional metrics such as the per-episode violation rate (percentage of test episodes where cost > κ) and the maximum observed cost. This is essential to evaluate the claim of "one-shot" safety and would make the empirical validation much more compelling for the stated problem setting.
- **Add Statistical Significance Testing:** Perform statistical significance tests (e.g., pairwise t-tests with multiple-testing correction) to substantiate claims that CARL outperforms other safe methods. Present these results succinctly in the main text or appendix.