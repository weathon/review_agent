# TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Multimodal Large Language Models (MLLMs), particularly smaller, deployable variants, exhibit a critical deficiency in understanding temporal and procedural visual data, a bottleneck hindering their application in real-world embodied AI. This gap is largely caused by a systemic failure in training paradigms, which lack large-scale, procedurally coherent data. To address this problem, we introduce TPRU, a large-scale dataset sourced from diverse embodied scenarios such as robotic manipulation and GUI navigation. TPRU is systematically designed to cultivate temporal reasoning through three complementary tasks: Temporal Reordering, Next-Frame Prediction, and Previous-Frame Review. A key feature is the inclusion of challenging negative samples, compelling models to transition from passive observation to active, cross-modal validation. We leverage TPRU with a reinforcement learning (RL) fine-tuning methodology, specifically targeting the enhancement of resource-efficient models. Experiments show our approach yields dramatic gains: on our manually curated TPRU-Test, the accuracy of TPRU-7B soars from 50.33\% to 75.70\%, a state-of-the-art result that significantly outperforms vastly larger baselines, including GPT-4o. Crucially, these capabilities generalize effectively, demonstrating substantial improvements on established benchmarks. The codebase is available at \url{https://github.com/Stephen-gzk/TPRU/}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the critical failure of smaller, deployable MLLMs in understanding temporal and procedural visual data, a bottleneck for real-world embodied AI applications. To solve this, the authors introduce TPRU, a large-scale dataset sourced from diverse embodied scenarios like robotic manipulation and GUI navigation. The dataset is systematically structured into three complementary tasks: Temporal Reordering, Next-Frame Prediction, and Previous-Frame Review, and includes challenging negative samples to force models to perform active validation. Using a reinforcement learning (RL) fine-tuning strategy with TPRU, the authors achieved dramatic gains: the TPRU-7B model's accuracy soared from 50.33% to 75.70% on their manually curated TPRU-Test.

### Strengths
+ The paper effectively addresses a novel and well-framed problem regarding the deficiency in temporal understanding in MLLMs, substantiated by the introduction of a large-scale and valuable dataset, TPRU.

+ Despite being vague on the task scope, the methodology and data generation pipeline are presented with great clarity, supported by well-designed and informative figures that make the complex processes easy to understand.

+ The application of an RL fine-tuning strategy provides a novel insight for this task.

+ The benchmark evaluates a breadth of public benchmarks.

### Weaknesses
+ There is a notable discrepancy between the paper's stated motivation of advancing embodied AI and the composition of the TPRU dataset. A significant portion of the data, GUI navigation, has limited relevance to physical agent-environment interaction. Furthermore, the embodied scenarios included are largely constrained to static (pure navigation) or tabletop settings, which may not fully represent the complexity and dynamism of real-world embodied tasks.

+ The data pipeline's heavy reliance on a single upstream model (Qwen2.5-VL-72B) for both quality filtering and text description generation risks inheriting and amplifying that model's intrinsic biases.

+ The TPRU-Test is small, with only 461 instances, and its tasks directly mirror the training setup. While this design is effective for measuring direct improvement on the learned skills, it concludes "significant gains" more easily achievable within its own data distribution. The evidence for the generalizability of these improvements to a broader range of temporal understanding tasks remains somewhat limited.

+ Although the paper claims the performance gain is notable, from Table 2, the fine-tuning process did not lead to universal improvements across all evaluated sub-tasks. Notably, there were performance regressions on certain benchmarks, such as a significant drop on the "Multiview" sub-task of LEGO-Puzzles for TPRU-7B and negative gains on other sub-tasks for TPRU-32B.

+ The paper relies solely on point estimates for accuracy (e.g., in Tables 1-3)  and lacks statistical robustness checks like confidence intervals or significance testing, weakening the conclusions drawn from small test sets.

+ A potential ethics statement concern is the authors' claim to use videos from YouTube (e.g., Arvin Bricks) without specifying the exact license used.

I will consider raising my rating if the authors could address my concerns above.

### Questions
1. How does your dataset, with its static or tabletop scenarios and GUI tasks, generalize to the dynamic, real-world challenges of embodied AI?
2. Your data pipeline heavily relies on Qwen2.5-VL-72B. How did you prevent this single model's biases from being amplified in your dataset?
3. Given the small 461-instance TPRU-Test mirrors your training tasks, how do you prove the "significant gains" are true generalization and not just overfitting?
4. Could you please provide the confidence interval or significant results on the TPRU test set?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces *TPRU* (Temporal and Procedural Understanding), a real-world, multi-image dataset and training recipe to improve temporal and procedural understanding in small-to-medium Large Multimodal Models (MLLMs). TPRU contains 24,750 QA pairs over 126k images sourced from four embodied scenarios (robotic manipulation, embodied navigation, mobile GUI interaction, LEGO assembly) and is organized into three complementary tasks: Temporal Reordering, Next-Frame Prediction, and Previous-Frame Review. The authors also curate TPRU-Test (461 expert-verified items) for evaluation. Using rule-based RL (GRPO) on Qwen2.5-VL backbones, the method's TPRU-7B improves from 50.33% to 75.70% on TPRU-Test and shows transfer to MuirBench and LEGO-Puzzles without degrading general multi-image performance.

### Strengths
1. The paper is well written and easy to follow.
2. By pairing a training dataset explicitly structured for temporal reasoning with a matched held-out test set, this paper tries to address a known gap where multi-image sequences are often treated as unordered sets. The negative-sample design explicitly trains rejection of inconsistent options, pushing models toward cross-modal verification rather than text-prior heuristics.
3. The ablations show all three tasks are synergistic, negative samples materially help, and scaling provides diminishing returns.
4. The reported results demonstrate that small/efficient models can close much of the gap to very large proprietary systems on temporal, procedural understanding via data+RL, which is important for edge deployments.

### Weaknesses
1. Quality control and description generation rely on Qwen2.5-VL-72B. This can induce latent bias or style leakage into both data and targets. While pragmatic, the paper does not quantify inter-annotator agreement on machine-generated descriptions nor analyze failure modes from automated filtering. A small human-validated subset analysis for precision/recall of filter acceptance and error taxonomy would strengthen robustness claims.
2. The three tasks are framed around 3–4-frame sequences with MCQ/permutation outputs. This is a strong first step but may not capture long-horizon dependencies, branching plans, open-ended multi-turn grounding, or action-conditioned predictions typical in robotics.
3. Most comparisons are to multi-image MLLM benchmarks. Strong video-temporal baselines (e.g., models trained on video or long-context visual streams) are not reported. Even if adapters are needed, a discussion or a pilot comparison would clarify how much of the reported gains are specific to the quiz-style multi-image setup vs. general temporal understanding.
4. The RL setup does not provide training stability metrics (reward curves, variance across seeds), nor does it probe whether the format reward induces prompt-format overfitting rather than genuine reasoning upgrades.

### Questions
1. How accurate is the automated filtering? A human audit of a random sample (≥500 sequences) would quantify dataset reliability better.
2. Have you tried >4-frame sequences or variable-length ordering?
3. Have you tried free-form forecasting (describe the next state), counterfactuals (what step would prevent reaching frame 4?), or procedure repair (select the minimal fix to an erroneous sequence)? These would test deeper causal modeling than recognition-style choices.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces TPRU, a large-scale dataset designed to enhance MLLMs’ temporal and procedural understanding. The dataset includes three complementary tasks—Temporal Reordering, Next-Frame Prediction, and Previous-Frame Review—and a held-out evaluation set. The authors also finetune Qwen-VL using RL to demonstrate the dataset’s effectiveness.

### Strengths
Valuable dataset contribution. The creation of a large-scale dataset focused on temporal and procedural reasoning is a meaningful engineering effort. If released publicly, it could be beneficial for the community and future research.

Clear visual presentation. Figures are well-designed and make the method and dataset structure easy to understand.

Comprehensive related work and supplementary content. The paper provides detailed literature review and supplemental materials that help contextualize the contribution.

### Weaknesses
Motivation needs stronger justification. The core motivation—that existing datasets treat images as unordered—is not fully convincing. For example, LLaVA-Next-Interleave and other multimodal corpora already include temporal and sequential interactions (e.g., embodied tasks, spatial sequences). The paper should more clearly articulate what specific gaps remain and how TPRU uniquely addresses them.

Dataset source selection lacks coherence. The four data domains—robotic manipulation, LEGO assembly, GUI operation, and embodied navigation—appear loosely connected. It is unclear why these domains were chosen, whether they are complementary, or whether their combination leads to emergent abilities. As written, it feels as if these datasets were chosen opportunistically rather than driven by a principled rationale.

Modest performance gains. The improvements over the baseline are relatively small. More discussion is needed to interpret the results and understand the practical significance of the dataset.

Evaluation scope is limited. Relying on a single self-introduced benchmark is not sufficiently convincing. Additional established benchmarks related to temporal or sequential understanding would provide more robust validation.

### Questions
none

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces TPRU (Temporal-Procedural Understanding dataset): a large training and evaluation datasets for temporal coherence in embodied multimodal foundation models.

The paper notes that current training paradigms for embodied multimodal foundation models do not train explicitly for temporal understanding, which leads to fundamental limitations on their ability to perform in real world applications. In contrast, the TPRU dataset offers an alternative that bakes temporal understanding tasks into the model.

TPRU dataset includes three temporal reasoning tasks:
- Temporal Reordering: photos from a sequential stream are given in a random order and the MLLM is required to order it.
- Next-Frame Prediction: photos from a sequential stream are given in order with one in the middle missing, the MLLM must choose which image is missing
-  Previous-Frame Review: photos from a sequential stream are given in order with one in the beginning missing, the MLLM must choose which image starts the sequence

The dataset is collected from multiple sources with different embodiments (GUI, robotic manipulation, navigation) and base MLLMs are used to filter and label them. Furthermore, each task type has negative samples (i.e. samples where the correct answer is to reject all given options).

A 7B model trained on TPRU shows significant uplift compared to the base model and other strong baselines, achieving SOTA results on the TPRU-test set.

### Strengths
- The dataset is moderately large (~25k examples, ~126k images), which is useful for training.
- The dataset covers a variety of embodiments (e.g. GUI, robotic manipulation, navigation...), which is useful for multiple applications.
- The test set is manually curated/validated and held-out from training, which means it's likely to be high quality and a good measure of generalization.
- The TPRU-trained models and the data generation pipeline have comprehensive evaluations and ablation studies, which show the importance of the different tasks and negative answers, as well as various model sizes.
- The TPRU-trained models retain the same performance on general benchmarks like MMStar and MMMU-Dev.
- I found the point about training the MLLMs to explicitly to reject invalid logic very interesting, and I think it can be useful in other applications with these models.

### Weaknesses
- It's unclear to me why only GRPO was used on the data as a learning algorithm/paradigm, but not simpler training regimes like simple SFT, which is usually the first thing people try.

- The dataset poses all examples as multiple-choice-questions (MCQs) where one of the answer is "reject all answers", it's unclear why this format was chosen instead of free generation of answers.

- It's unclear to me why the 7B model ends up performing better on TPRU-test than the 32B model. 

- Section 3 is called "TRPU" in the paper, it should be "TPRU" for consistency.

### Questions
- How do you know that the TPRU-test set (461 examples) is sufficiently deduplicated from the training data (25k examples). How did you confirm there's no overlap or too much semantic similarity between held out set and training set to avoid information leakage?

### Soundness
4

### Presentation
4

### Contribution
4
