Now I have a good picture of the calibration landscape. Let me now write the final consolidated review.

Key observations from calibration:
- Skipper (prior work this builds on): scores 5, 6, 6 → accept (poster)
- Goal-conditioned RL with Virtual Experiences (similar HER relabeling topic): scores 5, 6, 5, 6 → reject
- Null Counterfactual for GCRL (HER improvement with evaluation on robotics tasks): scores 5, 6, 8, 8 → accept (poster)
- Bidirectional-Reachable HRL: scores 5, 5, 6 → reject
- Horizon Generalization in RL (conceptual + limited experiments): scores 3, 3, 5, 6 → accept (poster) but weak

This paper has a similar profile to papers in the 4-5 range: conceptual contribution with limited experimental scope, incremental algorithmic novelty, and some overclaiming of generality.

Let me now write the consolidated review.</think>## Summary

This paper introduces the concept of "delusions" in target-directed reinforcement learning—systematic failures where agents hold false beliefs about self-generated targets—and proposes a taxonomy of delusion types (G.1/G.2 for generator failures, E.0/E.1/E.2 for estimator failures). The authors propose two new hindsight relabeling strategies ("generate" and "pertask") and hybrid combinations that decouple generator and estimator training data to mitigate delusions, showing improved OOD generalization in MiniGrid-based environments.

## Strengths

- **Useful diagnostic taxonomy**: The decomposition of failure modes into generator vs. estimator contributions (G.1, G.2, E.0, E.1, E.2) provides a structured vocabulary for discussing problems that have been under-analyzed in the goal-conditioned RL literature. The distinction between G.2 (temporarily unreachable targets due to irreversibility/segregation) and G.1 (nonexistent targets) is particularly insightful, as G.2 effects are often overlooked.

- **Principled mapping from diagnosis to solution**: The strategies "generate" and "pertask" are directly motivated by the diagnosed delusion types—"generate" addresses E.1 and "pertask" addresses E.2—which makes the experimental design feel coherent rather than ad hoc. The 2-slotted hybrid insight that generators and estimators have conflicting training data needs (Section 4.3) is a practical and non-obvious design principle.

- **Thorough internal evaluation**: Within its narrow experimental scope, the paper evaluates fine-grained metrics (estimation errors per delusion type, delusional behavior frequencies, OOD success rates) rather than relying solely on aggregate performance. The SSM environment provides ground-truth access to these metrics, enabling meaningful decomposition of results.

- **Consistent results across 4 experiment sets**: As summarized in Section 5.6, results across 2 environments and 2 methods (Skipper and LEAP) align in their conclusions about the relative effectiveness of different strategies.

## Weaknesses

### Fatal
None.

### Major

- **The "delusion" framing adds terminology but limited formal substance beyond existing concepts.** The core technical insight—that HER's training data distribution can leave estimators blind to unreachable or temporarily unreachable goals proposed at decision time—is important and well-diagnosed. However, the psychiatric analogy and "delusion" terminology do not produce new formal constraints, metrics, or theoretical predictions beyond what is already captured by standard RL notions of reachability mismatch and estimator coverage. G.1 is essentially hallucination/invalid goals, G.2 is irreversibility in non-ergodic MDPs, and E.0/E.1/E.2 are value estimation errors over different state-goal subsets. The paper's primary contribution is conceptual framing rather than methodological, so its significance depends on whether the taxonomy yields genuinely new insights—which the paper partially delivers (particularly via G.2) but does not fully formalize.

- **Experimental scope is too narrow to support the claimed generality.** The abstract claims the strategies "make agents address delusions preemptively and autonomously" and Section 4.1 states they "should be expected to be applicable generally." However, the in-main-text evidence consists of one algorithm (Skipper) on one custom discrete gridworld (SSM), with three other setups in the appendix using similarly structured environments. There are no experiments on standard continuous-control or robotics benchmarks where HER is most commonly applied, no results on environments with partial observability or learned state encoders, and no comparison to non-target-directed baselines that would help isolate whether the identified failures are specific to the target-directed architecture. The claims of general applicability are not substantiated by the evidence.

- **Mixing proportions for hybrid strategies lack principled selection and sensitivity analysis.** The hybrid strategies (e.g., F-(E+P+G) with 50%-25%-25%) use hand-selected mixing proportions without justification or ablation. Given that the relative improvements depend on these choices, and the proportions likely depend on the environment and delusion-type distribution, this is a meaningful gap. Without showing robustness to these hyperparameters, it is unclear whether the improvements reflect genuine principled design or careful tuning to SSM's specific delusion profile.

### Minor

- **Delusion metrics are not rigorously specified.** The paper relies on L1 distance errors with clipping for unreachable targets as its primary delusion metric, and "delusional behavior frequencies" are described qualitatively. How G.1 and G.2 targets are detected at decision time, and whether reachability is defined under any policy or the current/optimal policy, is not fully specified. While this is understandable given the controlled SSM environment, it limits the reproducibility and generality of the measurement framework.

- **The psychiatric analogy is primarily motivational rather than constructive.** The paper references Corlett (2019) and Kiran & Chaudhury (2009) for the generator-estimator incoordination framing, but does not leverage formal models from that literature to derive algorithmic constraints. The analogy could be trimmed without loss of technical content.

- **No analysis of computational overhead.** The paper acknowledges that "generate" incurs additional computation but does not quantify this cost, making it difficult for practitioners to assess tradeoffs.

### Trivial
None.

## Nice-to-Haves

- Evaluation on at least one continuous-control benchmark (e.g., AntMaze, FetchReach) where HER is standard, to demonstrate that the proposed strategies transfer beyond discrete gridworlds.
- Ablation study varying mixing proportions to assess sensitivity and provide practical guidance.
- Comparison with a non-target-directed baseline (e.g., HAC, HIRO) on the same environments to isolate whether the identified failures are delusion-specific or general OOD generalization challenges.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that proposed strategies are "incremental" variants of existing HER work**: The harsh critic argues that "generate" and "pertask" are close to prior ideas (Yang et al. 2021a; Nasiriany et al. 2019). However, the paper explicitly acknowledges connections (Section 4.1.1 and Table 1), and "pertask" (cross-episode, within-task goal sampling) is a distinct relabeling strategy not directly proposed in those works. The 2-slotted hybrid approach is also a specific contribution. The strategies are modest but not merely rebranded existing work.

- **Claim that "safety catastrophes" framing is speculative/unsubstantiated**: The paper uses this language in the introduction for motivation and does provide an environment with lava traps (episode-terminating states), which is a safety-relevant setup. This is reasonable motivational framing for an RL paper.

- **Demand for comparison with non-target-directed baselines (PPO, SAC) on SSM**: While this would be informative, the paper's stated scope is target-directed agents. Whether delusions are specific to that class is a valid question for future work, not a core flaw.

- **Demand for formal necessity proofs for the three conditions**: The paper frames these as intuitive/empirical conditions rather than formal theorems. Requesting formal proofs for what is essentially an empirical/design paper would be scope creep.

- **Criticism that the paper doesn't test on environments without temporary unreachability**: The paper's contribution is specifically about environments with G.1/G.2 structure. Testing on trivial cases where delusions don't exist would not substantively affect the claims.

- **Nitpick about not reporting wall-clock time as separate weakness from the acknowledged computational cost**: Already covered in minor weaknesses above; the paper does acknowledge the cost qualitatively.

## Novel Insights

The distinction between G.2 (temporarily unreachable targets) and G.1 (nonexistent targets) is an underappreciated insight: in non-ergodic MDPs with irreversible state transitions (like picking up objects), goals that were once reachable become structurally inaccessible, and training relabeling strategies like "episode" and "future" can systematically create exposure to these now-unreachable states for the generator while failing to expose the estimator to their unreachability from the current state. This asymmetry between what the generator learns to propose and what the estimator learns to evaluate is the paper's most valuable diagnostic observation.

## Suggestions

- Scale back generality claims (e.g., change "should be expected to be applicable generally" to conditional statements about when the strategies are expected to help) and acknowledge the narrow experimental scope explicitly.
- Add a sensitivity analysis for the mixing proportions, even if just on SSM, to show that improvements are not brittle to hyperparameter choices.
- Provide a clearer formal specification of how G.1/G.2 categories and E.1/E.2 errors are computed at evaluation time, so the measurement framework can be ported to other environments.

## Score and Decision

**Calibration comparison:**

- *Skipper* (prior work, similar gridworld-only scope, scores 5/6/6): The current paper builds directly on Skipper but has a narrower and more incremental algorithmic contribution (HER relabeling variants) compared to Skipper's novel planning architecture.
- *Horizon Generalization in RL* (conceptual + limited experiments, scores 3/5/6): Similar profile—a conceptual reframing with limited empirical scope. This paper has more concrete algorithmic proposals but less theoretical grounding.
- *Goal-Conditioned RL with Virtual Experiences* (HER-related, scores 5/6/5/6, rejected): Similar level of novelty in HER variants, but evaluated on broader environments. This paper has cleaner diagnostic framing but narrower experiments.
- *Null Counterfactual for GCRL* (HER improvement with robotics evaluation, scores 5/6/8/8, accepted): Significantly stronger empirical validation, including continuous control tasks.

The paper makes a meaningful diagnostic contribution (the taxonomy, especially G.2) and provides practical mitigations, but falls short on experimental breadth and formal depth. The algorithmic contributions are modest (relabeling variants + mixing), and the generality claims outpace the evidence. Relative to calibration, this sits below accepted papers with broader empirical validation and below Skipper's novelty, but above rejected papers with broken theory or no clear contribution. A score around 5 is appropriate—borderline, leaning reject given the overclaim and narrow evaluation.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>