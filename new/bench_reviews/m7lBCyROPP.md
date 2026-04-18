## Summary
The paper proposes GCQS, a goal-conditioned RL framework that uses achieved goals from hindsight experience replay (HER) as subgoals within a phasic policy structure. A Q-BC objective (Q-learning regularized by behavior cloning) learns a policy for reaching achieved goals, and a prior policy defined over these achieved subgoals provides a KL-regularized initialization for learning to reach desired goals. The central motivation is that GCAC and GCWSL methods exhibit a "short-horizon bias" in their relabeling, and subgoals from longer-horizon achieved goals can address this.

## Strengths

- **Empirical improvements on standard benchmarks**: GCQS achieves noticeably higher success rates and faster learning than DDPG+HER, MHER, and several GCWSL methods on Fetch and Hand manipulation tasks (Fig. 5), with particularly strong performance on FetchPick, FetchSlide, and HandRotateXYZ.

- **Simple and practical core idea**: Using relabeled achieved goals as subgoals without a separate subgoal-discovery module is a natural and elegant design choice, avoiding the complexity of typical hierarchical RL approaches.

- **Clear conceptual framing**: The phasic goal structure — first learning to reach achieved goals, then using them as a prior for desired goals — is intuitively motivated and provides a clean two-phase training paradigm.

- **Ablation evidence on easy tasks**: Fig. 8 shows that both subgoals and BC-regularization contribute to GCQS's performance, with subgoals appearing more important than BC-regularization on Fetch/Hand tasks.

## Weaknesses

### Major

- **Overclaimed AntMaze results contradict the paper's central motivation**: The abstract claims GCQS achieves "results comparable to such state-of-the-art subgoal-based methods" on AntMaze, and the introduction frames the method as addressing long-horizon tasks. However, Fig. 7 clearly shows GCQS achieves near-0% success on S-AntMaze and π-AntMaze, while BEAG reaches ~80%. Even on U-AntMaze, GCQS lags BEAG substantially. This directly undermines the paper's core claim that relabeled subgoals effectively handle long-horizon tasks.

- **The Q-BC derivation (Sec. 5.1, Eqs. 10–12) contains mathematical errors**: The transition from Eq. 10 to Eq. 12 has a critical flaw. Eq. 11 claims that minimizing D_KL(π || π_relabel) equals minimizing E[log π(a|s,g')], but this is the reverse KL direction — maximizing E_{B_r}[log π] minimizes D_KL(π_relabel || π), not D_KL(π || π_relabel). The KL direction matters and fundamentally changes the interpretation. Additionally, the "Dirac-Delta" justification for a stochastic policy is internally inconsistent (a Dirac delta is deterministic). Since Q-BC is presented as a core contribution with theoretical justification, these errors undermine confidence in the method's theoretical grounding, even if the objective works empirically.

- **Theorem 4.1 is trivially true and does not establish a problematic bias**: The theorem states that cumulative probabilities over increasing horizons are monotonically decreasing — S(p(I+1)) ≤ S(p(I)). This holds for any probability distribution over positive integers and is a basic property of cumulative sums, not a structural "bias" of GCAC/GCWSL. While Fig. 2 empirically shows concentration at short horizons (which is expected under future-sampling in finite episodes), the paper never demonstrates that this distribution actually harms performance or that specifically correcting it is what drives GCQS's gains.

- **The prior policy and subgoal mechanism are underspecified**: Eq. 14 defines π^prior(a|s,g) = E_{s_g ~ τ^{g'}}[π(a|s,s_g)], but it is unclear how s_g relates to the desired goal g. Without conditioning on g, this expectation averages over all achieved goals in a trajectory, many of which may be irrelevant or even counterproductive as subgoals for a specific desired goal. Implementation details are relegated to an unavailable appendix (B.1), and the paper does not specify how subgoals are selected, how many are sampled, or how gradient backpropagation works through the Monte Carlo approximation.

### Minor

- **Missing SAC+HER baseline despite building on SAC**: The paper states that GCQS "integrates the SAC following GCAC," yet the strongest baseline compared is DDPG+HER. Since SAC+HER is a stronger and more direct comparison (same backbone algorithm), its absence makes it difficult to assess whether GCQS's gains come from the proposed subgoal/Q-BC mechanism or simply from using SAC rather than DDPG.

- **Offline-method baselines likely under-tuned**: DWSL and GoFar are compared as GCWSL methods, but the paper itself acknowledges they "perform poorly, likely due to their configurations being more suited for offline goal-conditioned RL." Comparing an online method against offline methods without proper online re-tuning inflates GCQS's apparent advantage.

- **Ablations limited to easy tasks**: The ablations in Fig. 8 only cover FetchReach, FetchPick, FetchPush, and HandReach — precisely the tasks where "No Subgoals" and "No BC-Regularized Q" already perform reasonably well. Ablations on the harder AntMaze environments would be far more informative given the paper's long-horizon motivation.

- **Theorem 5.1 provides limited insight**: The bound depends on an undefined quantity |I| and a generic KL constraint η. It is a standard KL-regularized policy iteration result that does not meaningfully distinguish the phasic goal structure from any other KL-regularized approach, and no empirical or theoretical analysis connects the bound to observed performance.

### Trivial

- The DDPG citation is mis-typed as "Lillierap et al., 2015" in Section 6.

## Nice-to-Haves

- Analysis of subgoal quality: visualizing sampled subgoals on AntMaze would reveal whether they form meaningful waypoints or are mostly uninformative positions — directly testing the core hypothesis.
- Ablation varying the number or selection strategy of subgoals (e.g., filtering for longer-horizon subgoals only); the current approach uses all achieved goals with no filtering.
- Comparison against SAC+HER to isolate the contribution of the subgoal/Q-BC mechanism from the backbone algorithm choice.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"DWSL/GoFar comparison is unfair because they're offline methods"** — While the comparison may be unfair, the *direction* of unfairness favors GCQS (the author's method), making GCQS look good. This is kept above as a minor weakness.

- **"Theorem 5.1 has undefined quantities"** — This is kept in minor weaknesses but was considered for removal since it's a minor notational issue; however, the deeper problem that the theorem is generic and uninformative justifies keeping it.

- **Demands for larger benchmark diversity (e.g., Franka Kitchen, image-based tasks)** — This is scope creep; the paper evaluates on standard benchmarks used throughout the field.

- **Formatting/style nitpicks** — e.g., figure labeling inconsistencies are trivial.

- **Reproducibility concerns about hyperparameters** — Standard for this type of work; β=0.2 is stated.

- **"The claim that GCWSL underperforms GCAC is speculative"** — The paper states this as an empirical observation from their own experiments; while they don't show the data, this is a reasonable design choice justification, not a core claim.

- **"No analysis of computational overhead"** — Nice-to-have but not a core flaw.

- **Missing related works (e.g., specific HRL methods)** — Outside the paper's stated scope; cannot verify existence of cited methods.

## Novel Insights

The key insight — that achieved goals from HER's future relabeling strategy naturally provide subgoals that can improve policy learning for longer-horizon goals — is genuinely useful. However, the evidence suggests this insight's practical benefit is currently limited to relatively easy manipulation tasks; it does not appear sufficient for truly long-horizon navigation problems where subgoals must lie at structural bottlenecks rather than arbitrary achieved states. The disconnect between the claimed motivation (addressing short-horizon bias in long-horizon tasks) and the empirical reality (strong on short-horizon tasks, weak on long-horizon AntMaze) is the paper's most important unresolved tension.

## Suggestions

1. Moderate AntMaze claims to reflect actual results — GCQS is competitive with some subgoal methods on L-AntMaze and U-AntMaze but clearly fails on S-AntMaze and π-AntMaze.
2. Fix the Q-BC derivation: either correct the KL direction or acknowledge that Q-BC is an empirical objective inspired by AWAC/conservative RL rather than derived from a principled constrained optimization.
3. Add SAC+HER as a baseline and run ablations on AntMaze tasks to disentangle algorithmic components where they matter most.

## Evaluation Summary

**Originality**: Moderate. The idea of using relabeled goals as subgoals is natural and has tangential precedent (Chane-Sane et al., 2021), but the specific phasic structure with Q-BC and KL-regularized prior is new.

**Importance of research question**: High. Improving goal-conditioned RL under sparse rewards is an important and actively studied problem.

**Whether claims are well supported**: Partially. Strong empirical results on Fetch/Hand; overclaimed on AntMaze; theoretical derivations contain errors.

**Soundness of experiments**: Mixed. Positive on manipulation benchmarks; lacking key baselines (SAC+HER) and adequate ablations on hard tasks; offline-method baselines not properly controlled.

**Clarity of writing**: Moderate. The core idea is communicated, but the mathematical presentations are imprecise and the appendix-reliant descriptions hamper reproducibility.

**Value to community**: Moderate. If the subgoal mechanism were better isolated and the theory corrected, this could be a useful contribution to the goal-conditioned RL toolkit.

## Calibration

Compared to calibrated papers at similar quality levels:
- **OjCWG58ZyY** (Goal-Conditioned RL with Virtual Experiences; scores 6,5,6,5; rejected): Similar domain (goal-conditioned RL + subgoals), similar empirical improvements on Fetch/Hand, similar overclaiming issues. GCQS has slightly weaker theory but slightly stronger empirical gains.
- **BsQTw0uPDX** (HPO; scores 3,6,8,5; rejected): HPO had similar issues with underspecified formulation and limited experimental support. GCQS has more thorough experiments but similar theoretical issues.
- **Xkf2EBj4w3** (Stabilizing Contrastive RL; scores 5,8,8,8; accepted spotlight): Strong empirical work with clear, actionable findings. GCQS is not at this level due to theoretical flaws and weak AntMaze results.

GCQS sits in the range of 4–5: it proposes a reasonable idea with empirical gains on standard manipulation tasks, but the theoretical grounding is flawed, the main motivational claim is based on a trivial observation, AntMaze results are overclaimed, and key design details remain underspecified.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>