# DT-Pro: Proactive Decision Transformers with Implicit Latent Space Planning

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Decision Transformers (DTs) address decision making problems through sequence modeling and have achieved surprisingly strong results. However, DTs still struggle in long-horizon tasks due to their poor planning ability. Existing works have demonstrated that subgoal prediction helps to guide DTs' decision making in complex and long-horizon tasks. However, explicit planning via subgoal prediction suffers from suboptimality, inefficiency and instability. In this paper, we present DT-Pro, a variant of DT that enhances its planning ability by integrating a natural implicit planning step into sequence modeling. Compared with explicit planning via subgoal prediction, the implicit planning works by inferring a latent plan from a structured plan space. Through this way, DT-Pro enables high-quality adaptive plan generation and efficient stepwise replanning with only a marginal increase in the computational cost.  Extensive experimental results show that DT-Pro achieves strong performance on a variety of widely used control and navigation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DT-Pro, a Decision-Transformer variant that inserts an implicit latent-plan step. 
Pipeline: 
(i) mine "critical" subgoals per trajectory using a decayed-RTG heuristic (Alg. 1), 
(ii) encode the subgoal sequence into a compact latent with a contrastive regularizer, and 
(iii) condition a DT-style action model on that latent (encoder frozen in stage-2) with step-wise replanning at test time. 
The paper reports a higher average normalized return than DT/ADT/WT/CQL/IQL on D4RL Gym-MuJoCo and Maze2d, shows ablations (subgoal strategy, contrastive term, #subgoals), a sparse-reward variant using a DT-predicted RTG signal, and provides decoded-plan visualizations.

### Strengths
- Simple, targeted idea: Implicit latent plan sidesteps brittle explicit waypoints, integrates cleanly with DT.
- Empirical signal on long horizons: Consistent gains on D4RL Gym-MuJoCo and Maze2d, subgoal-mining ablation is convincing.
- Interpretability: Decoded subgoal visualizations suggest some learned temporal structure.
- Scope & ablations: Vary #subgoals and (claimed) contrastive regularization, explore a sparse-reward setting.
- Potential impact: If baselines are re-established under a common protocol with stronger statistics, DT-Pro could be a go-to DT variant for longer-horizon offline RL.

### Weaknesses
- Objective inconsistency (paper vs. appendix vs. code): Main text describes CE/log-likelihood (plus contrastive) for subgoal decoding/policy, whereas algorithms use squared-error. The supplementary code appears to optimize MSE at both stages, with no contrastive term. This must be reconciled and reflected in the results.
- Baseline provenance/fairness: Many baseline scores are imported (only a subset rerun), risking potential protocol drift (normalization/eval differences). Stronger conclusions require rerunning baselines under a unified pipeline.
- Evaluation protocol clarity: Missing or underspecified train/val/test splits, checkpoint selection, exact D4RL normalization, and number of eval rollouts.
- "Optional plan decoding" ambiguity: Abstract hints at optional decoding/search, but evaluation appears not to use any test-time plan decoding, clarify and, if applicable, report results with/without it.
- Minor polish: Fix ADT average (73.7 vs. text 74.9), make captions self-contained (units, seeds), and add raw returns in the appendix.
- Reproducibility: Supplementary code indicates default training uses MSE objectives and omits contrastive loss. Evaluation relies on encoder outputs, with no plan-sequence decoding at test time.

### Questions
1. Losses: Are subgoal decoding and policy trained with CE/log-likelihood (main text) or L2 (Appendix Algs 3–4)? What does the released code actually optimize? If both were tried, please report a comparison.
2. Baseline protocol: For Table 1, which baselines were re-run under your pipeline and which were imported? How did you ensure identical target returns, normalization, number of eval episodes, and scoring for imported numbers?
3. Evaluation splits: What are your train/val/test and checkpoint-selection rules in offline replay? Please state D4RL normalization and eval rollout counts precisely.
4. Capacity/budgets: Sensitivity to plan dimension, N subgoals (partially reported), and any plan-search budget. A small grid on Hopper-MR and Maze2d-medium would help.
5. Sparse-reward: Using a pretrained DT to densify RTG - did you check for train/test contamination? How sensitive are results to the critic’s quality? Please add stds for all methods in Table 2.
6. Test-time decoding/search: The abstract implies an optional decoding/search, sections.4.3/4.4 suggests it’s not needed. Did you try multiple latent-plan samples or a beam over subgoals at inference? If yes, report it. If not, clarify.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new decision transformer (DT) architecture variant, showing state-of-the-art performance on range of offline RL control benchmarks. In difference to the standard DT architecture, the authors learn and use a sub-goal representation as an additional DT input to guide the auto-regressive next action prediction. The sub-goal representation is learned prior to the DT. The sub-goal training data is obtained from the traces in the training data, by splitting for each time step along the trace the remainder of the trace into n fragments, which are equally spaced in terms of the reward-the-go; using the staring point of each fragment as a sub-goal for the considered time step. An auto-encoder is trained to predict for each of those augmented training samples the associated sub-goals. The encoder part is then used to obtain the additional DT input. An experimental study demonstrates superior performance on the Mujoco and Maze2d benchmarks.

### Strengths
The paper introduces a small but impactful optimization to the DT architecture. It is conceptually relatively simple, seems to introduce only a marginal overhead in training (although this should be evaluated more thoroughly), and leads to state-of-the-art performance in the considered control benchmarks. The text is overall well-written, clearly structured, and easy to follow. Code and benchmarks are available, which should suffice to reproduce the results.

### Weaknesses
The authors however oversell their contributions. In particular, there are certain claims in the abstract and the introductions which are not in line with the proposed method or backed up by the experiments. Specifically, there are the following points:
- "Enhancing planning ability": The proposed method improve the performance of DT, but there is no clear evidence that it would improve its "planning ability". First, it is actually not clear what "planning ability" should be in this context precisely. None of the components of the architecture performs any explicit planning step, e.g., search over multiple alternatives. Secondly, the benchmarks focus entirely on control benchmarks (and Maze2d), none of which require any strong "planning ability". To show this claim, the authors need to consider other benchmarks, like puzzles where strategic decisions are essential.
- "RTG-based plan search algorithm": The method for finding the sub-goals is simple. The remaining trace at the step for which the sub-goals should be computed is simply split according to equally distributed intervals of the reward-to-go. Simplicity is not a bad thing, but clearly there is no "planning" or even real "search" involved. Don't oversell this method.
- Improves "Optimality of future plans": It is not clear at all what this is supposed to mean. What are future plans? What is "optimality", and what does it mean to improve optimality? Bottom line is that the proposed method improves the DT performance in some (not even all benchmarks) by a moderate and sometime a considerable portion. And that is about it.
- Improves "interpretability and utility of the plans": Again, how exactly should the proposed method achieve this? Without compelling explanation, I would argue that this claim is plain wrong.

I also find the wording "plan representation" misleading. What is presented in sections 4.1 and 4.2 is a model to predict for state reward pairs a set of sub-goals (in terms of state landmarks at which a certain reward-to-goal fraction is reached) to may guide the DT in its next action predictions. At no point does it learn to predict a "plan".

Some small clarity issues: The explantation for Algorithm 1 needs to be extended to cover the corner cases. It is not clear what is being done for those time steps where less than n (the number of to be chosen sub-goals) steps are remaining. In Section 4.2, it should be explained how the similarity between the traces is computed. In Section 4.4, it is not clear how a DT can be pre-trained to predict imaginary reward signals, i.e., how the reward function is reshaped to tackle the problem with the sparse reward signals.

### Questions
1. How do you handle corner cases in algorithm 1? Can the same t' be selected for multple \lambda_i?
2. Could you provide some more justifications for the claimed contributions (cf. review)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an enhanced variant of Decision Transformer (DT) designed to handle long-horizon tasks, where traditional DTs often struggle due to limited planning capability. The method introduces a two-stage training procedure involving three modules: a Plan Search module that identifies critical subgoals based on decaying returns-to-go, a Plan Coding module learns a compact latent space to represent the plans, and an Action Prediction module that executes these subgoals through generated actions. The approach demonstrates improved performance across selected benchmarks compared to standard DTs and related baselines.

### Strengths
1. The paper is well-written and easy to follow, with clear motivation and logical structure.

2. The proposed framework improves performance even under sparse or limited data conditions, showing robustness beyond ideal settings.

3. The ablation studies are well-designed, supporting the validity of the method’s components.

### Weaknesses
1. Increased training complexity and computational cost.

    - The method requires two training stages and three modules, compared to the single-stage DT baseline.

    - Although the authors report only 8–12% additional training time per added module, the first-stage cost is not clearly accounted for. The claim that it "runs entirely offline before training" is unclear, since all components are trained offline and should still contribute to total compute time.

    - Clarifying the total wall-clock cost or presenting a fair compute comparison with DT would strengthen the paper.

    - The authors mention "pretraining a DT as a critic" to provide granular returns-to-go (RTG) signals for Plan Search in sparse environments.

2. Ambiguities in implementation details.

    - The parameter N (number of subgoals) is said to vary by environment, but the rationale or selection criterion is not described.

    - The definition of small-scale datasets used in the first-stage Plan Search module ("runs entirely offline before training”) is vague, please specify what qualifies as "small-scale" and how it was chosen.

3. Questionable experimental coverage for the stated objective.

    - The paper claims to address long-horizon decision-making, yet all tested environments are relatively short-horizon MuJoCo tasks (e.g., UMaze and medium).

    - More suitable benchmarks such as Maze2D-large, AntMaze, or FrankaKitchen (as used in OGBench) would better represent the intended objective.

    - Additionally, recently proposed long-horizon baselines (e.g., TAP [1], diffusion-based planners [3]) are not included, despite being mentioned in the related works section. Omitting these comparisons weakens the evaluation’s credibility.

4. Incremental contribution.

    - While the method improves upon DT, it does so by adding extra components rather than addressing the core limitation of planning horizon in transformer-based RL.

    - The improvement (~18–24% increase in performance with comparable extra training time) is promising but might not constitute a significant conceptual advance beyond existing DT variants.

    - The authors could strengthen their contribution by positioning their approach relative to diffusion-based planning methods [3] or latent-action planners [1,2].

**Minor Suggestions (Not affecting the score)**

1. To facilitate comparison, align the order of tasks in Table 1 with that in Table 2 of the original DT paper, maintaining consistency with prior work.

** References**

[1] Zhang, Tianjun, et al. "Efficient Planning in a Compact Latent Action Space." *The Eleventh International Conference on Learning Representations*.

[2] Park, Seohong, et al. "OGBench: Benchmarking Offline Goal-Conditioned RL." *The Thirteenth International Conference on Learning Representations*.

[3] Janner, Michael, et al. "Planning with Diffusion for Flexible Behavior Synthesis." *International Conference on Machine Learning*. PMLR, 2022.

### Questions
1. What qualifies as the “small-scale benchmark dataset” used for the first module? How do you select it in practice?

2. Why was TAP [1] not included as a baseline, given that it targets the same long-horizon problem space and is cited in the related works?

3. How does your method compare conceptually and empirically to diffusion-based planners [3], which also emphasize flexible long-horizon planning?

### Soundness
2

### Presentation
4

### Contribution
1
