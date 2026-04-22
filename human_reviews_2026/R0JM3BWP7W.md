# Tricks or Traps? A Deep Dive into RL for LLM Reasoning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Reinforcement learning (RL) for LLM reasoning has rapidly emerged as a prominent research area, marked by a significant surge in related studies on both algorithmic innovations and practical applications. Despite this progress, several critical challenges remain, including the absence of standardized guidelines for applying RL techniques and a fragmented understanding of their underlying mechanisms. In addition, inconsistent experimental settings, variations in training data, and differences in model initialization have led to conflicting conclusions, obscuring the key characteristics of these techniques and creating confusion among practitioners when selecting appropriate techniques. This paper systematically reviews widely adopted RL techniques through rigorous reproductions and isolated evaluations within a unified open-source framework. We analyze the internal mechanisms, applicable scenarios, and core principles of each technique through fine-grained experiments, including datasets of varying difficulty, model sizes, and architectures. Based on these insights, we present clear guidelines for selecting RL techniques tailored to specific setups and provide a reliable roadmap for practitioners navigating the RL for the LLM domain. Finally, we show that a minimalist combination of two techniques can unlock the learning capability of critic-free policies with a vanilla PPO loss. The results demonstrate that our simple combination consistently improves performance, surpassing strategies such as GRPO and DAPO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shares a systematic review of variations of experimental settings in LLM-RL training to provide a unified guideline for practitioners. The review includes analysis on advantage normalization, PPO clipping and loss aggregation. In each category, the paper provides empirical results with different model sizes and dataset difficulty and a practical recommendation based on the result. Finally, the paper proposes Lite PPO, a combination of two techniques (group-mean/batch-std advantage normalization and token-level loss aggregation), improves performance of non-aligned LLM models.

### Strengths
The paper is well written and comprehensive. Contributions are clearly stated in introduction and supported by experiments. Addressing the lack of a standard guideline in the LLM-RL field is important to allow practitioners to understand choice of techniques. From this point of view, this paper tackles an important problem in this domain. Overall, the experimental results are well conducted and the proposed Lite PPO algorithm is an natural extension to GRPO based on the findings provided in this paper.

### Weaknesses
Although there isn't obvious flaws in the paper, there are comments to improve the quality of the analysis further.
- In Section 4.2.3, explanation of why 8B model doesn't have "scaling law" of the upper bound clipping parameter strengthens this analysis. Analysis on trends of LLM's outputs might help explain this.
- In Section 4.4.1, there is a lack of explanation around what leads to the experimental results shown in Figure 10 and 11. What learning dynamics influences these result? This analysis seems vital to understand the choice of overlong filtering.


Minor grammatical errors:
- Line 388, `As illustrated in As shown in` -> `As illustrated in`

### Questions
- Computation cost is also another important dimension when people select techniques. Do authors have any insight around this? I imagine that the most of techniques analyzed in this paper won't impact the computing performance though.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic and rigorous evaluation of reinforcement learning (RL) techniques for improving reasoning capabilities in large language models (LLMs). The authors address the current fragmentation in RL4LLM methodologies by reproducing and analyzing popular RL "tricks"—such as normalization, clipping, loss aggregation, and filtering—under a unified framework. Through extensive experiments across diverse model sizes (Qwen3-4B/8B), architectures (base vs. aligned), and dataset difficulties (easy/medium/hard), the study offers actionable insights into the mechanisms and applicability of each technique. A key contribution is the proposal of Lite PPO, a minimalist combination of two techniques (group-mean + batch-std normalization and token-level loss aggregation), which outperforms more complex methods like GRPO and DAPO in critic-free policy optimization.

### Strengths
1. Comprehensive and Reproducible Evaluation: The paper leverages a unified open-source framework (ROLL) and over 160 independent experiments, ensuring robust and statistically meaningful conclusions. The ablation studies (e.g., standard deviation removal in normalization, clip-bound scaling laws) are particularly insightful.

2. Practical Guidelines: The authors translate empirical findings into clear, scenario-specific recommendations (e.g., token-level loss for base models, sequence-level for aligned models), addressing a critical need for standardization in RL4LLM.

3. Minimalist Innovation: Lite PPO demonstrates that simplicity can outperform heavily engineered methods, challenging the trend of over-complication and offering an efficient baseline for future work.

### Weaknesses
1. Experiments are confined to Qwen-family models and mathematical reasoning tasks. While math is a common benchmark, broader validation on diverse domains (e.g., code generation, commonsense reasoning) would strengthen claims of generalizability.

2. While Lite PPO is promising, more ablation is needed to disentangle the contributions of its two components (normalization vs. loss aggregation) across all settings.

### Questions
How might your guidelines adapt to non-mathematical tasks? Are there techniques whose effectiveness is highly domain-dependent?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper conducts a systematic evaluation of implementation choices for RL with LLMs, covering advantage-normalization variants, clipping with a higher upper bound, loss aggregation at the token vs. sequence level, and overlong-response filtering, within a unified PPO-style framework. Experiments are run on math-reasoning benchmarks using Qwen3-4B/8B (both base and aligned variants). From these studies, the authors distill seven empirical takeaways and introduce a minimalist recipe, Lite PPO, which pairs group-mean with batch-std advantage normalization and token-level loss under a vanilla PPO objective without a critic. On base models, Lite PPO achieves consistent gains over GRPO and DAPO across several math datasets.

### Strengths
- Organizes scattered RL techniques into a coherent, condition-aware evaluation (base vs. aligned; easy vs. hard), yielding concrete and readable takeaways.
- Provides careful ablations and useful diagnostics (e.g., entropy and ratio behavior; token-level analyses) that improve interpretability of clipping and aggregation choices.
- Delivers a simple, reproducible recipe (Lite PPO) that reduces complexity yet performs strongly on base models, offering immediate practical value.

### Weaknesses
- The contribution of this paper is mainly an empirical synthesis of known implementation choices rather than an algorithmic or theoretical advance; no new RL objective or formal analysis is introduced, which constrains the paper’s originality compared to prior work
- If the practical objective is the strongest final model, the focus on base models leaves uncertain whether the proposed recipe yields meaningful gains for aligned/instruction-tuned models that start from stronger baselines.
- Evidence is confined to math reasoning; presenting the work as a general “roadmap” for RL with LLMs risks over-generalization without results on other reasoning modalities (e.g., logical, strategic, open-ended).

### Questions
- What motivated restricting experiments to Qwen3? Could you report at least a final-recipe run on another family (e.g., Llama or Mistral) to assess portability?
- Can you apply the Lite PPO recipe to aligned models and compare against strong aligned baselines to clarify its utility when the goal is the best final system?
- Several techniques and experimental patterns resemble DAPO. Could you delineate, in methodological terms, how your approach differs (objective, normalization, clipping, loss aggregation), and provide ablations isolating the incremental contribution beyond DAPO?
- The higher-clipping “scaling law” appears to rest on a single family with limited sweep points. Do you view this as a size-dependent trend rather than a law? If not, what additional evidence (broader sizes, denser sweeps, statistical tests) supports the stronger terminology?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an empirical analysis of components of efficient RL pipelines for LLMs (in particular GRPO- and DAPO-style techniques) on both non-aligned and aligned Qwen3 models. The authors study the impact of data difficulty, advantage calculation and normalization, clipping strategies, loss aggregation granularity, and reward shaping via overlong filtering, all under a unified PPO-based setup. Based on these observations, they propose LitePPO for non-aligned models, which uses group-level mean and batch-level standard deviation for advantage normalization, together with token-level loss aggregation. Experiments on Qwen3 4B and 8B non-aligned models evaluated on Math500, OlympiadBench, AMC23, Minerva Math, AIME24, and AIME25 show that LitePPO outperforms GRPO and DAPO.

### Strengths
- Systematic and fairly extensive analysis of the effect of data difficulty, advantage normalization (including std/no-std variants), clipping strategies, loss aggregation, and overlong filtering on RL performance for LLMs.

- Evaluation covers both non-aligned and aligned model variants and two parameter scales (4B, 8B), which makes the conclusions more convincing than single-model studies.

- The paper is clearly written and well structured; the individual “takeaways” for each component are easy to follow and practically useful.

- The proposed LitePPO recipe is simple yet effective.

### Weaknesses
- Statistical robustness / variability:

    - It is not clear how many random seeds are used; Appendix C suggests a single seed (seed=42). If this is the case, the results are potentially sensitive to randomness.

    - Showing mean and variance over multiple runs (e.g., multiple seeds) would make the empirical conclusions stronger, especially in Figure 12. Even if re-running all experiments is too expensive, clearly stating the number of runs and acknowledging this limitation would help.

- Minor typo:
    - Line 388: “As illustrated in As shown in Figure 9...”, remove one of “As illustrated in / As shown in”.

### Questions
1. For Figures 3, 4, 6, 8, 9, 10, and 12, could you clarify precisely what the y-axis “accuracy” refers to? Is it the average accuracy across all six evaluation benchmarks, or a subset, or a particular held-out split from the Easy / Medium / Hard training datasets themselves?

### Soundness
3

### Presentation
3

### Contribution
3
