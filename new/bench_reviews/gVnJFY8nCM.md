The paper content was provided inline. Let me verify key claims:

1. The abstract says "customizing continuous-control policies at execution time, given access to the prior action distribution alone" — this omits the dynamics model requirement. Confirmed criticism.

2. The paper says "approximates the optimal customized policy online" (Sec. 3.1 after Eq. 6). Confirmed the theoretical overclaim.

3. In Table 1, HalfCheetah: Residual-MPPI = 1936.2 ± 109.3, Greedy-MPPI = 1939.9 ± 134.7. Swimmer: Residual-MPPI = -60.0 ± 5.2, Greedy-MPPI = -58.9 ± 5.4. Hopper: Residual-MPPI = 7363.0 ± 254.9, Greedy-MPPI = 7367.0 ± 199.4. These are indeed very close within standard deviations. Confirmed.

4. In GTS (Table 2), Greedy-MPPI and Guided-MPPI both fail (cannot finish a lap), so the log π term is clearly critical there. Confirmed.

Now let me write the final review.

## Summary

Residual-MPPI proposes an online planning algorithm for customizing continuous-control policies at execution time by integrating the Residual Q-Learning (RQL) framework into Model Predictive Path Integral (MPPI) control. The method modifies the MPPI scoring function to include both an add-on reward and the log-likelihood of a prior policy, enabling zero-shot and few-shot policy customization without access to the original reward function. Experiments on MuJoCo benchmarks and the Gran Turismo Sport (GTS) environment demonstrate effective customization, including adapting the champion-level GT Sophy 1.0 racing agent to reduce off-course behavior.

## Strengths

- **Well-motivated and practically important problem.** Online policy customization—adapting a pre-trained policy to new requirements at execution time without retraining—is a highly relevant problem for real-world robotics and autonomy, where new constraints emerge after deployment. The paper clearly articulates why retraining/fine-tuning is undesirable.

- **Conceptually clean and simple algorithmic idea.** The core contribution—substituting the RQL augmented reward (ω′ log π + r_R) into the MPPI scoring function while sampling around the prior policy—is elegantly straightforward and easy to implement. It naturally leverages the prior policy's action distribution for both initialization and regularization.

- **Impressive GTS application with GT Sophy 1.0.** Customizing a champion-level racing agent in the complex GTS environment, reducing off-course steps from 93.13 ± 1.98 (prior) to 4.43 ± 2.39 (few-shot), is a compelling scalability demonstration. The failure of Greedy-MPPI and Guided-MPPI to even complete a lap dramatically illustrates the necessity of the log π term in complex domains.

- **Comprehensive and well-designed baseline set.** Seven baselines systematically isolate the contribution of each component: Greedy-MPPI (no log π), Full-MPPI (no prior sampling), Guided-MPPI (prior sampling + full reward), Valued-MPPI (prior value as terminal estimator), Residual-SAC (RL-based competitor), and Fulltask-SAC (upper bound). This systematic ablation design is a strong point.

- **Clear sample efficiency advantage in GTS.** Residual-MPPI uses ~2K laps for dynamics training + ~100 laps for fine-tuning versus 80K+ laps for Residual-SAC, producing a less conservative and faster policy. This is a meaningful practical advantage.

## Weaknesses

### Major:

- **Theoretical framing overclaims the connection to RQL.** The paper repeatedly states that Residual-MPPI "solves" the augmented MDP M^aug and "approximates the optimal customized policy." However, Proposition 1 establishes equivalence between MPPI's distribution and the maximum-entropy optimal policy only under conditions that are not met in practice: γ=1 and infinite-variance sampling. The actual implementation uses γ<1, finite noise variance Σ, no terminal value estimator, and a finite planning horizon. The paper acknowledges this informally ("γ close to 1 and large noise variance") but provides no analysis of how much the approximation degrades when these conditions are violated. This is not a fatal flaw for an applications paper—the method works empirically—but the theoretical claims should be reframed as heuristic inspiration from RQL rather than principled optimality. (See Sec. 3.1 and the discussion following Eq. 5-6.)

- **Marginal improvements over Greedy-MPPI in MuJoCo.** In three of four MuJoCo environments (HalfCheetah, Swimmer, Hopper), Residual-MPPI and Greedy-MPPI achieve nearly identical total rewards within standard deviations: HalfCheetah 1936.2 vs 1939.9, Swimmer -60.0 vs -58.9, Hopper 7363.0 vs 7367.0. The paper's narrative of consistent superiority is not supported by these numbers. Only in Ant (6846.7 vs 6104.2) and especially GTS does Residual-MPPI show a clear advantage. The MuJoCo results suggest the log π term provides modest benefit when add-on and basic rewards are not strongly conflicting—a condition the paper should acknowledge more prominently.

- **No computational cost or timing analysis.** The paper claims "online" customization at "execution time," but never reports wall-clock time, number of samples K, or planning horizon T. MPPI requires K forward rollouts through the dynamics model per planning step, and the practical feasibility of online control—especially in GTS—cannot be assessed without this information. This omission undermines the "online" claim. (Algorithm 1 lists K and T as parameters but their values are not in the main text.)

- **Hyperparameter ω′ sensitivity is undocumented.** The weight ω′ balancing prior log-likelihood and add-on reward is the core knob controlling the customization trade-off. Yet no sensitivity analysis or principled selection guidance is provided. The reader cannot assess whether the method works robustly or requires per-environment tuning. This is a significant practical gap.

### Minor:

- **Abstract claim of "given access to the prior action distribution alone" is misleading.** The method also requires a dynamics model and an evaluable add-on reward function. These are not minor assumptions—a learned dynamics model is central to the approach and non-trivial to obtain.

- **Discount factor gap between theory and practice.** Proposition 1 assumes γ=1, but Algorithm 1 uses γ^t in the scoring function. The paper provides no analysis of the approximation error introduced by γ<1, leaving the theory-practice gap unaddressed beyond an informal note.

- **Dynamics model quality and out-of-distribution behavior are under-characterized.** The zero-shot→few-shot improvement in GTS (off-course: 9.03→4.43) suggests sensitivity to model accuracy, but no systematic analysis of how model error propagates under policy shift is provided. This limits confidence in the method's robustness.

- **Residual-SAC comparison in GTS is a "from scratch" baseline, not a true policy-customization competitor.** Residual-SAC trains from scratch on the combined reward without leveraging prior policy knowledge in the same way. The dramatic efficiency gap (2K vs 80K laps) partly reflects model-based vs. model-free advantage rather than a direct comparison of policy customization approaches.

## Nice-to-Haves

- ω′ trade-off curves showing basic reward vs. add-on reward as ω′ varies, providing practical guidance and demonstrating controllability of the customization.
- Wall-clock timing per planning step for both MuJoCo and GTS to substantiate the online feasibility claim.
- A dynamics model quality ablation (varying training data from 200 to 2K to 20K episodes) to show zero-shot performance scaling with model quality.
- Comparison with a modern planning-with-prior method like TD-MPC2 or MBOP-style approaches in MuJoCo.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Full-MPPI is a strawman baseline because it catastrophically fails."** Full-MPPI does not use prior-guided sampling and lacks the log π term—the paper's point is precisely that both contributions matter. Its failure demonstrates the importance of these components and is not an unfair baseline. Removed because criticizing baselines that the paper's method outperforms with less information is not a valid weakness.

- **"Missing comparisons with TD-MPC2, MBOP, or diffusion planners."** These methods address different problems (same-task planning with full reward access), not the policy customization setting where the prior reward is unknown. They would require modifications that the paper explicitly discusses in Sec. 6 as non-applicable. Removed as scope creep—the paper evaluates against the baselines appropriate for its stated problem.

- **"Abstract says 'prior action distribution alone' which is misleading because it omits the dynamics model requirement."** While this is technically imprecise, it's a minor framing issue rather than a substantive weakness—moved to Minor weaknesses since the paper clearly discusses dynamics model requirements in Sec. 3.2.

- **"Few-shot Residual-MPPI is underspecified—number of fine-tuning iterations and convergence criteria not reported."** This is a reproducibility detail that belongs in an appendix (which the paper references). Not a core methodological weakness.

- **"GTS experiments only use one track."** This is a scope limitation but the paper's contribution is about the algorithm, not track-specific verification. Moved to Nice-to-Haves.

- **"IL prior policy results are relegated to the appendix."** The paper includes these results (Table 10 in Appendix F.1); the main text focuses on RL priors for clarity. This is a presentation choice, not a weakness.

- **"Valued-MPPI is unavailable in GTS due to lack of critic access—no alternative terminal value estimator is tried."** This is inherent to the problem setting (the prior provides only the policy network). The paper explicitly acknowledges this limitation. Removed as the baseline infeasibility reflects the real-world constraint, not an experimental gap.

## Novel Insights

The key insight validated by this work is that the log π term from a prior policy serves not merely as regularization but as a proxy for long-horizon reward information beyond the MPPI planning horizon. The Ant environment (orthogonal x/y rewards) and especially GTS (where Greedy-MPPI collapses to "stay still") demonstrate this vividly: without log π, the planner cannot reason about the long-term consequences of actions beyond its finite horizon, even when the add-on reward is locally well-defined. This provides concrete evidence for a phenomenon previously argued only theoretically in the RQL framework.

## Suggestions

- Explicitly reframe the theoretical contribution as RQL-inspired rather than RQL-solving. State that Residual-MPPI is a practical algorithm that heuristically extends RQL's continuous-space formulation through MPPI, and that the theoretical guarantees are approximate under realistic conditions.
- Add ω′ sensitivity sweeps and ω′ trade-off curves (basic vs. add-on reward) as a core experiment. This is the single most important missing analysis for practical deployment.
- Report wall-clock planning times per step and the planning parameters (K, T, Σ) used in each experiment in the main text or a clearly referenced table.
- Provide a dynamics model quality ablation showing how customization performance scales with model accuracy/data volume.

## Score and Decision Calibration

**Calibration papers:**
- **BMPC (i7jAYFYDcM)**: Accepted poster (scores 6,8,8,6). Strong empirical results combining MPC with learned components, moderate novelty, clean presentation. Similar profile to Residual-MPPI in combining existing frameworks effectively with empirical validation.
- **Diffusion-Based Planning for AD (wM2sfVgMDH)**: Accepted oral (scores 8,8,8,6). Very strong empirical results on a challenging domain, but reviewers note incremental technical novelty. Higher bar than Residual-MPPI due to more comprehensive evaluation and clearer contribution.
- **Reflect-then-Plan (6jr94SCjH6)**: Rejected (scores 5,5,3,5,5). Model-based planning with Bayesian integration. Mixed performance, fair baselines concern, and theoretical/practical gap issues. Similar weaknesses to this paper but weaker overall results.
- **Off-Road Driving via MPPI+RL (uaKBM9sGEm)**: Rejected (scores 6,3,1,6). MPPI + RL combination, but major concerns about baseline selection and complexity analysis. This paper is somewhat analogous but has clearer problem formulation and more convincing results.

Residual-MPPI sits between BMPC (accepted, ~7) and Reflect-then-Plan (rejected, ~4.6). It has a clean problem formulation, convincing GTS results, and comprehensive ablations. However, the theoretical overclaim, marginal MuJoCo improvements, missing computational analysis, and absent ω′ sensitivity prevent a higher score. The paper makes a solid systems-level contribution to an important problem, even if individual components are not novel.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>