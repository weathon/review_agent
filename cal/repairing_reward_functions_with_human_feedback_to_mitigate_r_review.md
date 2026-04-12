=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary
This paper proposes **Preference-Based Reward Repair (PBRR)**, an iterative method for correcting a human-specified proxy reward by learning an additive transition-level correction from trajectory preferences. The main idea is to exploit the fact that many reward-hacking problems may be fixable by modifying only a small subset of over-rewarded behaviors, and to do so more sample-efficiently than learning a reward model from scratch. The paper combines a tailored preference-learning objective with a reference-policy-based exploration strategy, gives regret guarantees for a tabular/linear variant, and shows strong empirical results on several reward-hacking benchmarks.

## Strengths
- **The reward-repair formulation is genuinely well-motivated and distinct from standard RLHF.** Instead of discarding an existing proxy reward and relearning from scratch, the paper models the true reward as `proxy + correction` (Eq. 2). This is a concrete and practically meaningful reframing of reward alignment that matches how many systems are actually developed: humans already provide proxies, and the problem is often to repair them rather than replace them wholesale.

- **The proposed objective in Eq. 3 is a specific and insightful contribution, not just a generic regularizer.** The loss uses the proxy reward to partition pairs into cases where the proxy agrees/disagrees with preferences, then regularizes toward preserving already-correct rankings and preferentially *decreasing* reward on dispreferred trajectories. This is a sharp design choice aimed at the specific failure mode of over-optimistic proxy rewards. The paper supports this with ablations in Section 6.4 / Appendix G.4 showing that replacing Eq. 3 with standard cross-entropy materially hurts stability and performance.

- **The empirical study is stronger than a simple headline figure.** Beyond the main comparisons, the paper includes targeted analyses of: optimistic-assumption violations (Glucose, Appendix G.6/K.7 discussion), random reference policies (Appendix G.8), retraining fragility under longer optimization (Appendix G.2), and qualitative failure analyses for competing methods (Appendix H). This makes the empirical narrative more convincing than a standard “wins on benchmarks” presentation.

- **The paper is unusually explicit about where the method’s assumptions break and what still works empirically.** For example, Section 6 openly states that the undominated-set machinery is intractable in the deep RL benchmarks and therefore the empirical implementation uses `C1 = 0`, i.e., the simpler reference-vs-current-policy exploration. Likewise, Section 4 explicitly notes that the optimism assumption is leveraged by Eq. 3 but “our algorithm does not require this assumption,” with the regularization decayed over time.

- **The qualitative analyses provide real mechanistic insight.** Appendix H is especially useful: it explains why standard Online-RLHF conflates instrumental subgoals with terminal objectives in the gridworld, why residual reward modeling can get trapped updating only local high-proxy-reward regions, and why the proposed loss avoids over-valuing reference-policy behaviors. These are concrete observations that help explain *why* PBRR works.

## Weaknesses

### Fatal
None.

### Major:
- **There is a real theory/practice disconnect between the regret analysis and the empirical method actually run.**  
  The paper’s theoretical results in Section 5 analyze Algorithm 1 in the tabular/linear setting, including the undominated policy set `Π_t` and a fallback to uncertainty-maximizing policy pairs when the reference-policy comparison is insufficient. But in the experiments, Section 6 explicitly states that this machinery is intractable in the high-dimensional nonlinear domains and therefore sets `C1 = 0`, meaning exploration always compares the repaired-reward policy against the reference policy. This is not a reviewer misread; the paper says:
  > “in our empirical results we set \( C_1 = 0 \), which implies ... PBRR always uses the reference policy and the policy that optimizes for the corrected proxy reward function.”
  
  So the regret theorems do **not** justify the main empirical algorithm used in the paper’s flagship experiments. This does not invalidate the empirical results, but it does weaken the paper’s integrated “theory + practice” claim substantially. The paper partly acknowledges this, but the abstract/introduction still present the theory and experiments too seamlessly.

- **The empirical claims about preference efficiency are established only under synthetic preference labels, not realistic human feedback conditions.**  
  All experiments use preferences sampled from the ground-truth reward via a Boltzmann/Bradley-Terry style labeling procedure (Section 6.1, Appendix E.5), rather than real humans or even richer simulated human noise models. The paper is candid about this and even discusses why it uses full-trajectory preferences to avoid another form of mismatch, but the central practical motivation is reducing the cost of human feedback. As a result, the evidence supports “preference efficiency under simulated labels derived from the ground-truth reward,” not yet “reduced real human labeling burden” in a strong external-validity sense. This is a substantive limitation because the paper’s target problem is explicitly human feedback efficiency.

- **Dependence on the reference policy is important and not fully characterized.**  
  PBRR’s exploration strategy fundamentally relies on comparing trajectories from the current repaired-reward policy to a supplied `π_ref`. The paper does include a helpful random-initialized-reference experiment in Appendix G.8, which weakens the strongest form of this criticism. However, what is still missing is a systematic analysis of *which properties* of the reference policy matter: performance, state-space coverage, contrast with the proxy-induced policy, or safety. Since this is central to the method’s sample efficiency story, a more controlled sensitivity study would materially strengthen the work.

- **The optimism-biased repair objective is plausible and effective, but its behavior outside the optimistic-proxy regime remains only partially characterized.**  
  Section 4 explicitly designs Eq. 3 around the expectation that human proxies are “aligned or overly optimistic,” and the third term is constructed to favor reducing reward on dispreferred behavior rather than increasing reward on preferred behavior. The paper does provide mitigating evidence: Glucose violates the optimism assumption and PBRR still performs well; Appendix G.6 also studies a pessimistic gridworld proxy. Still, this remains a genuine limitation of the method’s conceptual framing: outside optimistic-proxy settings, the inductive bias is no longer obviously well matched, and the paper’s fix is primarily a decaying regularization schedule rather than a deeper treatment of when the bias helps or hurts.

### Minor
- **The regularization schedule for \(\lambda_1,\lambda_2\) is heuristic and under-analyzed.**  
  Appendix E.6 states that in practice the coefficients are set to 10 and effectively decayed as `10 / |D^+|`. This is a reasonable practical choice, but it is not theoretically tied to the optimization dynamics of Eq. 3, and the paper does not provide a sensitivity analysis over these values. Given that the objective is a main contribution, understanding how brittle or robust this schedule is would improve confidence.

- **Only three random seeds are used in the main plots.**  
  The paper does more than many submissions by including an additional 10-seed analysis for part of the Pandemic setting (Appendix G.9), but the main benchmark figures still average over 3 seeds. Since stability is one of the paper’s headline empirical claims, broader seed coverage or stronger significance reporting would make that claim more compelling.

- **The main text could more prominently surface the reward-fragility caveat from Appendix G.2.**  
  The appendix shows that in Glucose, performance can deteriorate under substantially longer optimization on the repaired reward. This is an important and honest result: PBRR improves robustness but does not fully eliminate reward-model fragility under stronger optimization. That nuance deserves more emphasis in the main discussion.

### Trivial
- **The scaled/clipped presentation in Figure 2 makes cross-environment trends easy to read but obscures absolute effect sizes.**  
  Since the unscaled plots are available in Appendix G.7, the paper could point readers there more aggressively in the main experimental discussion.

## Nice-to-Haves
- A systematic ablation over reference policy quality/coverage, rather than only the default and random-initialized cases.
- A sensitivity analysis for the Eq. 3 regularization coefficients and decay schedule.
- Additional experiments with more realistic noisy preference models or limited-consistency labelers.
- Stronger significance reporting across all environments/seeds.
- A clearer statement in the abstract/introduction that the regret guarantees apply to a tabular/linear variant, not the deep nonlinear empirical implementation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper does not ablate the key components of PBRR.”**  
  Removed because this is factually incorrect. Section 6.4 and Figure 3 directly ablate the preference-learning objective versus standard cross-entropy, and compare PBRR’s objective inside alternative exploration strategies. Appendix G.4 further ablates `L+` and `L-` individually.

- **Claims that the experiments use a random or uniform reference policy by default, or that the paper fails to discuss suboptimal references at all.**  
  Removed as inaccurate. The main experiments use concrete reference policies described in Table 1 / Appendix B, and Appendix G.8 explicitly studies a randomly initialized reference policy.

- **Criticism that comparisons are unfair because baselines lack access to the reference policy.**  
  Removed because the paper explicitly gives reference-policy assistance to several baselines where relevant; e.g., Online-RLHF’s candidate batch includes reference-policy trajectories, and state-constrained baselines are built around the same reference policy. This is not a valid asymmetry complaint.

- **Reproducibility complaints about code release or artifact availability.**  
  Removed under the review rules. The paper states code is attached and will be released upon publication; questioning existence/availability is not a valid criticism here.

- **Pure formatting/style/parser complaints.**  
  Removed as instructed.

## Novel Insights
The most interesting synthesis across the paper is that PBRR’s empirical advantage seems to come less from “better reward learning in the abstract” and more from a **useful asymmetry in how reward repair should behave**: when starting from a hand-designed proxy, it is often safer and more data-efficient to identify and *downgrade over-rewarded loopholes* than to relearn the whole reward landscape or aggressively upweight preferred behavior. The qualitative analyses and ablations jointly support this interpretation. A second notable insight is that the reference policy’s value may not be its return quality per se, but the **contrastive coverage it provides relative to the proxy-induced policy**—an idea the paper hints at, but could elevate more explicitly.

## Suggestions
- Reframe the theory more carefully: clearly present Section 5 as guarantees for a tabular/linear variant, and avoid implying those theorems support the deep empirical implementation.
- Add a controlled study varying reference-policy quality and coverage to identify what properties of `π_ref` drive PBRR’s gains.
- Add sensitivity experiments for `λ1, λ2` and their decay schedule, since Eq. 3 is a central contribution.
- Strengthen the empirical validation of “human-feedback efficiency” with noisier simulated preference models or a small-scale human study if feasible.
- Surface the retraining-fragility result from Appendix G.2 in the main paper, framing PBRR as mitigating rather than eliminating reward-model overoptimization.
- If space allows, add a visualization of the learned correction term \(g(s,a,s')\) in interpretable environments to show that the method is truly targeting reward-hacking transitions rather than making diffuse global reward shifts.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 8.0, 2.0]
Average score: 5.0
Binary outcome: Reject
