# ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Efficient processing of high-resolution images is crucial for real-world vision–language applications. However, existing Large Vision-Language Models (LVLMs) incur substantial computational overhead due to the large number of vision tokens. With the advent of "thinking with images" models, reasoning now extends beyond text to the visual domain. This capability motivates our two-stage "coarse-to-fine" reasoning pipeline: first, a downsampled image is analyzed to identify task-relevant regions; then, only these regions are cropped at full resolution and processed in a subsequent reasoning stage. This approach reduces computational cost while preserving fine-grained visual details where necessary. A major challenge lies in inferring which regions are truly relevant to a given query. Recent related methods often fail in the first stage after input-image downsampling, due to perception-driven reasoning, where clear visual information is required for effective reasoning. To address this issue, we propose ERGO (Efficient Reasoning & Guided Observation) that performs reasoning-driven perception—leveraging multimodal context to determine where to focus. Our model can account for perceptual uncertainty, expanding the cropped region to cover visually ambiguous areas for answering questions. To this end, we develop simple yet effective reward components in a reinforcement learning framework for coarse-to-fine perception. Across multiple datasets, our approach delivers higher accuracy than the original model and competitive methods, with greater efficiency. For instance, ERGO surpasses Qwen2.5-VL-7B on the V* benchmark by 4.7 points while using only 23% of the vision tokens, achieving a 3× inference speedup.  The code is available at https://github.com/nota-github/ERGO.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Directly processing high-resolution images increases computational overhead, as many processed tokens are not task-relevant. To address this, the paper proposes a two-stage, coarse-to-fine approach. In the first stage, the model processes a low-resolution image to identify task-relevant regions; in the second stage, cropped high-resolution images of these regions are used for further processing. To enable this mechanism, the authors combine standard accuracy and format rewards with a box adjustment reward and a region-verification reward. As a result, the model achieves better performance than the base model while processing significantly fewer image tokens.

### Strengths
1. The paper is well motivated, and the two-stage coarse-to-fine approach makes sense and appears effective.

2. The model performs better than the base model while processing fewer image tokens.

3. The motivation for the additional rewards, such as the region-verification and box adjustment rewards, is clear.

### Weaknesses
The system appears complex due to the use of multiple rewards and reward models.

There is no ablation study on the weights of the different rewards. For example, in Line 216, the paper directly specifies the weights without any explanation or ablation analysis.

Some experimental details seem unclear. For instance, the paper lists DeepEyes [1] as a baseline for comparison, but the reported evaluation metrics differ significantly from those in the original DeepEyes paper.

Reference:
[1] DeepEyes: Incentivizing “Thinking with Images” via Reinforcement Learning

### Questions
What causes the significant drop in DeepEyes’ performance as reported in the table 2?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ERGO (Efficient Reasoning & Guided Observation), a two-stage "coarse-to-fine" framework aimed at improving the efficiency of high-resolution image understanding in Large Vision-Language Models (LVLMs). While existing two-stage methods often fail when target objects are indiscernible in low-resolution inputs, ERGO solves this by transitioning from "perception-driven reasoning" to "reasoning-driven perception," which uses contextual information to locate relevant image regions.
To achieve this, the authors employ a reinforcement learning (RL) scheme and propose the Task-driven Contextual Exploration (TCE) reward. This reward includes a component that ensures the cropped region is self-sufficient for answering the query, and another that actively encourages smaller, more efficient crops. Experiments on high-resolution benchmarks, including V* and HR-Bench, show that ERGO achieves superior accuracy while drastically reducing the number of vision tokens used, leading to faster inference compared to the base model and competing efficient methods.

### Strengths
1. Solves a Real and Critical Bottleneck: The paper tackles a major pain point: the prohibitive cost of processing high-resolution images for LVLMs in real-world applications. The authors sharply identify the failure case of existing coarse-to-fine methods—where tiny objects disappear after downsampling, and offer an effective, targeted solution.

2. Strong Experimental Results and Efficiency Gains: ERGO not only achieves sota performance on several high-resolution benchmarks like V* but also dramatically cuts down the computational cost (fewer Tokens, less latency).

### Weaknesses
1. The complexity of the coarse-to-fine pipeline is increased by multiple tool calls. The authors do not provide the average number of turns per sample, which is a critical missing metric for a complete evaluation of the true inference efficiency.

2. The Box Adjustment Reward uses a hard, dataset-derived threshold ($\gamma$) to prioritize small crops. This rigid prior risks confining the model's exploratory behavior and severely limiting its generalization ability on scenarios where a large, global context is genuinely required.

### Questions
Q1. Why is the Performance Gain So Small on HR-Bench 4K? According to Table 2, under the 1280×28×28 constraint, ERGO's gain on HR-Bench 4K is much lower than its gain on HR-Bench 8K and V*. Could the authors provide a fundamental analysis to explain why ERGO's advantage is significantly suppressed in the HR-Bench 4K scenario?

Q2. Performance and Efficiency Comparison without Resolution Constraint. If the initial input resolution were relaxed to be close to or equal to the original resolution, how would ERGO's performance and efficiency be affected?

Q3. Testing Generalization on Complex High-Resolution Tasks. The high-resolution benchmarks evaluated in this paper, such as V* and HR-Bench, are relatively simple. Given that the coarse-to-fine strategy might be more effective in simpler scenarios, could the authors provide performance data for ERGO, including a comparison against relevant works (DeepEyes, PixelReasoner, TreeVGR, Mini-O3), on more complex benchmarks that require multi-step reasoning, logical judgment, and finer targets (e.g., Visual-Probe[1] or TreeBench[2])?

[1] Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search

[2] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology

Q4. Quantifying the Multi-Turn Inference Cost. ERGO's inference process is essentially a multi-turn tool calling process. Could the authors compile and report the average number of tool calls per sample on the high-resolution benchmarks, along with a comparison to relevant works, to quantify the overhead of the decision-making process?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce ERGO (Efficient Reasoning & Guided Observation), a novel rl-based framework for efficient high-resolution visual question answering in VLMs. The key innovation lies in a reasoning-driven perception pipeline that identifies semantically relevant regions using low-resolution inputs and re-encodes those regions at full resolution for fine-grained reasoning. Unlike traditional perception-driven approaches, ERGO explicitly integrates multimodal reasoning into region selection through a two-stage coarse-to-fine pipeline and a task-driven contextual exploration reward in the RL setup. Experiments across multiple high-resolution VQA benchmarks demonstrate strong performance and efficiency, outperforming state-of-the-art models in both accuracy and computational cost.

### Strengths
1. Introduces a principled RL framework that aligns vision token efficiency with VQA performance using a custom TCE reward.
2. Paper is well-written, figures are illustrative, and reward function formulations are clear .

### Weaknesses
1. While the RL approach is justified, comparison with alternative non-RL cropping or saliency-based models is missing.
2. Qwen2.5 is used as both the base and reward model. It's unclear whether the approach generalizes well to other vision backbones.
3. The paper assumes that an RL formulation is the optimal paradigm for guiding region selection, yet no strong justification is provided for why this should outperform supervised learning approaches (e.g., using ground truth regions, attention supervision, or self-training).
4. The proposed reward structure (especially the region-verification reward) seems to mimic supervised alignment between image regions and answers, raising the question of whether RL introduces unnecessary complexity.
5. The combination of region-verification and box-adjustment rewards appears empirically effective, but the design choices (e.g., fixed $\gamma$ = 0.6, hard step function in Eq. 2) are purely heuristic.
6. The core contribution is the “reasoning-driven perception” strategy, yet the evaluation focuses heavily on end-to-end VQA accuracy, without thoroughly validating whether the selected regions are actually semantically aligned with the question (e.g, via IOU with GT regions or attention overlap with human fixations). Figure 5 attempts to address this, but the metric is poorly defined, lacks baseline comparisons, and only handles binary masking scenarios.
7. The reward model is frozen, but still based on Qwen2.5-VL-72B, which is the same family as the base policy model (Qwen2.5-VL-7B). This introduces a risk of reward leakage, particularly if the reward model and policy share weights or similar training objectives.

### Questions
See Weaknesses Above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents ERGO, a model designed to tackle high-resolution visual understanding for vision-language models without burning through too much computing power. The core idea is a two-stage pipeline: first, it analyzes a low-res version of the image to spot task-relevant regions, then crops only those areas at full resolution for detailed reasoning. Unlike other methods that struggle when objects get blurry after downsampling, ERGO uses reasoning-driven perception to find important regions. Trained with a reinforcement learning framework that includes rewards for good region selection, efficient crop size, and accurate answers, ERGO delivers impressive results. It outperforms models like Qwen2.5-VL-7B on various benchmarks. Even with strict pixel constraints, it maintains high accuracy across multiple high-res and conventional multimodal benchmarks.

### Strengths
1. The reasoning-driven perception approach is super smart. By using context to find relevant regions instead of relying on clear visual cues, it handles blurry low-res inputs way better than existing perception-driven models. 

2. The authors propose a well-designed reward system in the RL framework, combining rewards for region quality, crop size, and answer accuracy, makes sure the model learns to be both effective and efficient. 

3. The results are really practical. The model delivers real latency improvements and doesn’t lose out on performance when applied to other multimodal tasks beyond high-res reasoning.

### Weaknesses
1. The author mentions using a frozen reward model to provide the region reward. Analysis of how the reward model performs is crucial for others to validate the effectiveness of this reward design.

2. I appreciate the authors' effort in providing the predicted area ratio. However, I also want to know the success ratio of the predicted area for benchmark evaluation. This serves as an important metric to analyze how the model perform and the effevctiveness of model training.

3. The authors should provide more baseline method, like using the SFT method to performance area cropping in previous work (Visual-CoT and Chain-of-Spot), also the zero-shot baseline and tool-calling baseline should be included.

### Questions
As stated in weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
