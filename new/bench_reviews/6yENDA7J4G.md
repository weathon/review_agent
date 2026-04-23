## Summary

This paper proposes a "foundation model approach" for Mixed Integer Linear Programming (MILP), training a single deep learning model per task on diverse MILP problem classes to generalize across unseen classes. To address the lack of diverse training data, the authors introduce MILP-Evolve, an LLM-based evolutionary pipeline that generates over a thousand diverse MILP classes from 8 seed classes. The approach is evaluated on three tasks: integrality gap prediction, learning to branch, and a novel Language-MILP contrastive learning task, demonstrating improvements over baselines on held-out MILP-Evolve classes and MIPLIB transfer learning.

## Strengths

- **MILP-Evolve is a creative and well-designed LLM-based evolutionary pipeline for generating diverse MILP classes.** The modular class representation (Data/Modeling/Parameters), the 10 prompt operators (add, crossover, mutate, new, delete), and the filtering/parameter search steps form a principled framework. The t-SNE visualization in Figure 1b shows the evolved classes cover a much broader region of the embedding space than seed classes, and the quantitative scale (>1000 classes) far exceeds existing datasets.

- **The core finding — that training on diverse MILP classes substantially improves cross-class generalization — is well-supported by scaling experiments.** Figure 4 provides clean evidence that class diversity matters more than instance quantity. In Figures 4b and 4c, controlling for total training instances, increasing the number of classes consistently improves performance across all three tasks. This is a useful and actionable finding for the ML-for-MILP community.

- **Comprehensive three-task evaluation framework** testing different facets of MILP understanding (prediction, solving, interpretation), making the diversity argument more convincing than a single-task evaluation would. The Language-MILP contrastive learning task is a novel and interesting addition that bridges MILP with natural language.

- **Transfer learning to MIPLIB** (Table 3) shows that pretraining on MILP-Evolve data improves fine-tuning on a genuinely external benchmark, with faster convergence (Figure 5) and better final performance. This provides some evidence that the synthetic data captures useful structure beyond the generation distribution.

- **The "Ours - Attn." ablation** (Table 1) shows that the main gains come from data diversity rather than the attention module, which strengthens the paper's core claim about the importance of diverse training data.

## Weaknesses

### Fatal
None.

### Major

- **The "foundation model" framing is overclaimed relative to what is delivered.** The paper repeatedly uses "foundation model" language (abstract: "we take a foundation model training approach"; Section 1: "the first to propose a foundation model training approach for MILP learning"; contribution title: "A Foundation Model Approach for Efficient Multi-Class MILP Learning"). By the standard Bommasani et al. (2021) definition, a foundation model is a single model adapted to multiple downstream tasks. What this paper delivers is three separate task-specific models trained on shared diverse data — which is multi-class training, not a foundation model. The "Towards" in the title does moderate this somewhat, and the conclusion acknowledges the limitation ("we acknowledge that this work still trains separate models for each learning task"), but the body of the paper consistently inflates the framing. This matters because it defines how the contribution is understood: the paper's real contribution is demonstrating that diverse training data improves multi-class generalization, not building a foundation model.

- **MILP-Evolve's specific value over alternative class-generation methods is unvalidated.** All baselines (Seed, Seed+Param, Seed+VAE) operate within the same small set of seed classes. The comparison therefore shows that *having more diverse training classes helps*, which is important but does not establish that the LLM-based evolutionary approach is necessary or better than simpler alternatives (e.g., random constraint generation with varying sparsity/objective types, systematic combinatorial construction of new MILP classes, or human-designed novel classes). Without this comparison, the paper proves the importance of diversity but not the superiority of its diversity-generation mechanism.

### Minor

- **The MIPLIB transfer learning evaluation is incomplete, covering only 2 of 3 tasks.** The branching task is omitted because only 13 MIPLIB instances have suitable solve times (Section 5.3). While the explanation is understandable, branching is arguably the most practically important task, and its absence from the strongest OOD evaluation weakens the generalization claim. The IG prediction improvement on MIPLIB (correlation 0.44→0.59) is modest in absolute terms.

- **The IG prediction evaluation filters out hard instances.** Section 5.1 excludes instances not solved within 200s, systematically removing the cases where integrality gap prediction would be most valuable (large-gap, hard-to-solve instances). This creates an evaluation bias toward easier cases.

- **Multiplicative improvement claims over weak baselines are misleading.** The abstract's "5.8x correlation improvement" stems from a Seed baseline with correlation 0.10 (essentially no predictive power). Multiplicative improvements over near-random baselines are not meaningful indicators of model quality. The absolute correlation of 0.58 is decent but the framing inflates perceived gains.

- **No experimental comparison with Huang et al. (2024), the most directly comparable prior work.** This concurrent work trains a joint model on five MILP classes and is mentioned in the introduction as "limited," but never experimentally compared against. If the contribution is scaling beyond 5 classes, demonstrating gains over this existing multi-class approach would strengthen the paper.

## Nice-to-Haves

- A true multi-task model trained on all three tasks simultaneously, even if performance degrades, would illuminate whether tasks share representations and strengthen the foundation model trajectory.
- Qualitative examples of MILP-Evolve-generated classes (full formulations, not just evolution chains) would help readers assess the diversity and realism of generated problems.
- Analysis of what fraction of LLM-generated classes pass the filtering step, and how many are infeasible or degenerate, would strengthen the MILP-Evolve evaluation.
- Failure analysis on held-out classes: which types of unseen MILP classes does the model still struggle with?

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Reproducibility concerns about GPT-4o dependency"**: The harsh critic raised concerns about MILP-Evolve depending on GPT-4o and different LLMs producing different datasets. This is a nitpick about reproducibility of an artifact that is released publicly — the generated data is available. Removed per rules against nitpicks about reproducibility of large artifacts.

- **"Architecture confound between baselines and Ours"**: The harsh critic questioned whether baselines use the same architecture as the proposed model. For branching (Table 1c), all methods are labeled "(GCN)" confirming identical architecture. For IG and Language-MILP, the "Ours - Attn." ablation (which uses the same GCN-only architecture as baselines) still outperforms Seed, demonstrating the gains come from data diversity. The concern is partially addressed by the ablation.

- **"Figure 4a confounded by total data increasing with more classes"**: While the purple vs. gray comparison in Figure 4a is indeed confounded, Figures 4b and 4c properly control for total training instances, and the paper's main scaling claim is supported by the better-controlled experiments. This is a minor presentation issue, not a substantive weakness.

- **"Practical utility of Language-MILP contrastive learning is unclear"**: The paper explicitly frames this as a stepping-stone task (analogous to CLIP→DALL-E), which is a reasonable research motivation. Questioning downstream applications is scope creep for a paper that introduces the task.

- **Strength finder's claim about "5.8x correlation improvement" as a core strength**: Removed as it conflicts with the verified weakness that multiplicative improvements over near-random baselines are misleading.

## Novel Insights

The scaling analysis (Figure 4) reveals a clear and actionable insight: for MILP learning, *class diversity dominates instance quantity*. This mirrors findings in NLP and vision about data diversity, but establishing it for the combinatorial optimization domain — where the structure of problems is fundamentally different from images or text — is non-trivial and significant. The finding that even constant total training volume, spreading instances across more classes yields better generalization, has practical implications for how the community should invest data collection and model training effort.

## Suggestions

- Retitle or reframe the contribution as "multi-class MILP learning with diverse data" rather than "foundation model approach" — this honestly represents what is achieved while still being significant.
- Add one non-LLM class-generation baseline (e.g., random constraint structures, or human-designed novel classes) to isolate MILP-Evolve's specific contribution from the general value of diversity.
- Report IG prediction results on hard instances (e.g., using LP relaxation gap as a proxy when the optimal is unknown) to address the evaluation bias.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| GOAL | /home/wg25r/review_agent/human_reviews/z2z9suDRjw.md | 6.25 | True multi-task CO foundation model with single backbone + adapters; more honest framing, actually delivers a unified model. This paper is weaker because it doesn't deliver a single multi-task model. |
| RouteFinder | /home/wg25r/review_agent/human_reviews/du9reSRIo1.md | 5.50 | Foundation model for VRP with single model for multiple variants; comparable aspiration, actually delivers a single model but was rejected. This paper is similar in having real contributions with framing issues. |
| MIPGen | /home/wg25r/review_agent/human_reviews/YhwDw31DGI.md | 4.40 | MILP instance generation; narrower scope than this paper, which has broader ambition and more comprehensive evaluation. |
| Multi-task CO bandit | /home/wg25r/review_agent/human_reviews/Dgc5RWZwTR.md | 4.75 | Multi-task training for CO; less novel than this paper's data generation contribution. |
| CPLLM | /home/wg25r/review_agent/human_reviews/fnBYPL5Ged.md | 2.00 | Claimed foundation model but per-task models with marginal improvements; this paper is significantly better with real data generation contributions and more convincing results. |
| Narrow Transformer | /home/wg25r/review_agent/human_reviews/ech9J3xl9X.md | 2.50 | Per-language fine-tuning with inflated framing; this paper has much more substance. |
| WizardMath | /home/wg25r/review_agent/human_reviews/mMPMHWOdOy.md | 8.00 | Evolutionary data generation for LLMs; much stronger validation and cleaner framing. This paper is well below this standard. |

The paper sits between RouteFinder (5.50, rejected) and GOAL (6.25, accepted poster). It has a real and creative data generation contribution (MILP-Evolve) with useful scaling insights, but the "foundation model" framing overclaims what is actually multi-class training, and MILP-Evolve's specific value is under-validated against alternative class-generation methods. The MIPLIB transfer results are encouraging but incomplete. Below RouteFinder because RouteFinder actually delivers a single model; below GOAL because GOAL delivers a true multi-task architecture. Above MIPGen because the scope and ambition are greater.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>