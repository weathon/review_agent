# Unleashing Perception-Time Scaling to Multimodal Reasoning Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Recent advances in inference-time scaling, particularly those leveraging reinforcement learning with verifiable rewards, have substantially enhanced the reasoning capabilities of Large Vision-Language Models (LVLMs). Inspired by this success, similar strategies have been applied to multimodal reasoning, yet their impact on visual perception remains unclear. To investigate this gap, we introduce DisTANCE, a perception-centric benchmark for visual estimation tasks. Evaluation results show that LVLMs exhibit limited estimation precision, and inference-time scaling offers only marginal gains. We attribute this to the fast perception paradigm of current LVLMs, where visual understanding is treated as a one-shot output without modeling the underlying perceptual process. To address this, we propose Perception-Time Scaling (PTS), a novel paradigm that encourages token-rich perception and decomposes complex perception problems into intermediate tractable sub-problems, thereby enabling perception to align with and benefit from inference-time scaling. Combined with reinforcement learning techniques, PTS significantly improves perception accuracy, raising high-precision performance on DisTANCE from 8.0% to 64.7%, and generalizes well to out-of-domain tasks. Surprisingly, even though PTS data are purely synthetic, combining them with math reasoning data yields consistent gains in both reasoning and real-world perception benchmarks. Further analysis reveals that PTS introduces more perception-related tokens and increases the model’s attention to image tokens. Our code and data are released at https://github.com/RUCAIBox/PTS

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a new method for inference-time scaling for perception-heavy reasoning tasks such as visual estimation tasks (i.e. tasks that require estimating the size of geometric elements). The paper introduces a new way for inference-time scaling called perception-time-scaling (PTS). The main idea is that models are encouraged to decompose tasks by using some tokens for "unit estimation" (i.e.  <==========> corresponds to 1 unit). To encourage models to use this method, they introduce a task Distance reasoning Task with Analytical Numeric and Comparative Estimation (DisTANCE), which is automatically generated using programmatically generated images which includes a known ground truth, and PTS (and CoT for ablations) reasoning traces using GPT4o. Combining SFT and RL they train models and show that PTS leads to better performance than CoT. They also show that finetuning on this data, improves performance for other perception heavy tasks such as Geoperception and Lego-Puzzles.

### Strengths
- The method is innovative and addresses a well-recognized challenge in VLLMs.
- The ablation study comparing CoT and PTS (Tables 3 and 4) is convincing: SFT+RL with PTS yields notable gains over SFT+RL with CoT, with improvements exceeding 10 accuracy points on DisTANCE, though the benefits on out-of-distribution tasks are more modest.
- The appendix provides a comprehensive set of ablations that further motivate and support the proposed approach.

### Weaknesses
While the method appears effective for perceptual measurement tasks, its usefulness in out-of-domain settings is less convincing. Beyond measurement tasks, the gains seem limited at best. For example, in Table 5 the improvements are marginal, and the absence of multiple runs or error bounds further reduces confidence in their significance.

### Questions
- In Table 4, why does the method yield larger improvements on GeoLHC for Qwen2.5VL-3B, but greater gains on LegoHeight for the 7B model? Reporting results over multiple runs with deviations would help clarify this.
- In Figure 4b, which tasks are used to evaluate image-focused attention? Is this analysis specific to DisTANCE, or do the observed patterns generalize to out-of-domain tasks as well?

### Soundness
3

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
4

### Summary
This paper introduces DisTANCE, a new perception-centric benchmark for visual estimation tasks. The authors show that most VLMs struggle to accurately assess visual perception capabilities. To address this limitation, they propose the Perception-Time Scaling (PTS) training paradigm, which consists of two stages: (1) a cold-start SFT phase using PTS-annotated data, and (2) a GRPO optimization phase. This approach achieves significant performance gains, improving accuracy by 8.0% to 64.7% over baseline models.

### Strengths
- This paper introduces a new perception-centric benchmark and demonstrates that even state-of-the-art proprietary VLMs exhibit limited performance, highlighting the importance of enhancing not only reasoning but also perception capabilities.

- The paper also presents the Perception-Time Scaling (PTS) paradigm, which enables small VLMs to learn to perceive images in a more human-like manner.

### Weaknesses
- One of the main contributions is the introduction of a new perception-centric benchmark, particularly focusing on visual estimation, which differs from prior perception tasks. To further enhance the value of this benchmark, I recommend evaluating proprietary reasoning models (e.g., Gemini-2.5-Pro). Since the benchmark contains only 300 samples, this additional experiment seems feasible. When I tested one of the examples from the paper using GPT-5, the model invoked tool-calling behavior and solved the task successfully. However, please note that this comment does not affect my rating of the paper.

- The term Inference-Time Scaling usually refers to the phenomenon where allocating more inference-time compute leads to improved performance. However, the proposed Perception-Time Scaling seems somewhat different in nature. From my understanding, it adds perception-related steps into the reasoning process during training—so it appears closer to “perception-enhanced reasoning.” Moreover, the experiments do not show how performance changes when scaling up perception-related compute during inference. As a result, the naming feels slightly forced, as if it were attached to align with the popular term “inference-time scaling.”

- In the perception elaboration and perception decomposition stages, symbolic tokens (e.g., <==========>) are used. I am curious how performance changes if these tokens are simply provided as prompt hints during inference. This could clarify whether the model truly lacked visual estimation capability and whether reinforcement learning genuinely boosted this ability.

- If the benchmark was generated programmatically in Python, scaling up the dataset for training should be relatively straightforward. It would be interesting to see whether simply training on a larger dataset of similar style improves performance. Such a setting could help demonstrate that data scaling alone is insufficient for the proposed visual estimation task, reinforcing the task’s importance. That said, I understand that conducting this experiment within the rebuttal period may be unrealistic—so even a qualitative discussion or expected outcome would be helpful.

- Intuitively, I expected tool-augmented RL-trained VLMs to perform better. However, I suspect that your proposed method offers significant efficiency advantages. Could you provide details on the actual inference speed or runtime efficiency?

- Could you clarify the decoding settings (e.g., top-p, temperature) used during inference?

- GRPO performance can be highly sensitive to reward design. I am curious why the reward was defined in a continuous form. For instance, what would happen if you used a binarized reward—assigning a reward of 1 when the estimated value difference is below 0.1, and 0 otherwise? A detailed ablation analysis on this design choice would be insightful, especially since recent work [1, Fig. 6] reports that binary rewards can yield better results.

- While I recognize the substantial experimental effort behind this work, the methodological novelty appears limited. The proposed approach mainly adapts GRPO to the visual estimation task rather than introducing new RL algorithms. This is not a major issue, but it does make the paper feel more like an application of existing methods rather than a conceptual advance. I would have appreciated more insightful discussions, such as an analysis of the trade-off between perception and reasoning. If possible, could you elaborate on potential directions or reflections in this regard? For example, which part should we consider when we empower VLM's perception capability? should we strengthen VLM's reasoning part for better perception?


Additional Minor Suggestions (Not Affecting the Score)

- The term “Fast Perception” feels somewhat ambiguous. Would “Fast Thinking” be a more appropriate description?

- In line 187, the result shown in Figure 2 seems more like a fine-grained analysis than a decomposition of a complex perception task—so the phrase “decompose complex perception” was a bit confusing.

- In the Appendix section “Prompt for synthesizing PTS instructions”, a newline character (\n) seems to be missing between items 4 and 5.

---

Reference

[1] Unified Reinforcement and Imitation Learning for Vision-Language Models.

### Questions
Detailed questions in the weaknesses part.

### Soundness
2

### Presentation
2

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
This paper investigates whether inference-time scaling techniques can enhance the visual perception capabilities of Large Vision-Language Models (LVLMs). The authors introduce DisTANCE, a perception-centric benchmark focused on visual estimation tasks involving geometric shapes. Their evaluation reveals that current LVLMs exhibit limited estimation precision, with inference-time scaling offering only marginal improvements. To address this limitation, the authors propose Perception-Time Scaling (PTS), a novel paradigm that encourages token-rich perception through two key components: (1) Perception Elaboration, which uses symbolic tokens (e.g., <==========>) to represent abstract visual attributes like distance, and (2) Perception Decomposition, which breaks down complex perception tasks into tractable sub-problems. The method is trained via a two-stage pipeline: supervised fine-tuning followed by GRPO-based reinforcement learning.

### Strengths
I appreciate the core insight of this paper that the observation about "fast perception" in Section 2.3 is quite revealing: reasoning models produce longer chains of thought, but the perception-related content remains sparse. This is an important finding that hasn't been explicitly studied before, and the quantitative analysis of perception ratios in Table 2 makes this concrete.

What I find most interesting is the symbolic tokenization idea. Using <==========> to represent distances feels almost toy-like at first, but it's actually quite elegant. It gives the model a way to "think through" visual measurements step by step, similar to how humans might count or measure incrementally. 

The paper is generally well-written. Figure 1 effectively sets up the problem, and Figure 3 clearly illustrates the method. The examples in the appendix (Figures 7-9) are helpful for understanding how the approach works in practice. I can see the step-by-step decomposition and symbolic representation clearly.

### Weaknesses
1. Narrow task selection: While DisTANCE is well-designed for its purpose, the evaluation is limited to very specific geometric estimation tasks. The paper would benefit from demonstrating the method's effectiveness on a broader range of perception-centric tasks beyond length, perimeter, and area estimation of synthetic shapes.

2. For out-of-domain evaluation (Table 4), the paper only reports results on specific subsets like Comparison subset from Geoperception and the Height subset from LEGO-Puzzles. Both benchmarks have multiple other subsets that test different aspects of spatial reasoning. What is the performance on these other subsets? Selective reporting raises concerns about cherry-picking results.

3. In Table 5, only three subsets of BLINK (Jigsaw, Multi, Local) are reported. BLINK contains many more subsets designed to evaluate different perceptual capabilities. Why were only these three selected? A comprehensive evaluation across all BLINK subsets would strengthen the claims about perception improvement.

4. Table 5 introduces a different training setup by combining PTS data with Geometry3K math reasoning data. While this shows that PTS can be combined with other data, it doesn't directly address whether the core PTS training (SFT + RL on synthetic data) improves general multimodal capabilities. It would be more convincing to see results of the DisTANCE-trained model (from Table 3) evaluated on the general benchmarks in Table 5 to understand whether the perception improvements truly generalize.

5. Many related works misses several recent important works on multimodal reasoning and visual perception enhancement:
- VL-Rethinker (https://arxiv.org/abs/2504.08837): Addresses similar issues in visual reasoning
- ReVPT (https://arxiv.org/abs/2509.01656): Relevant work on vision-language model perception
- R-4B (https://arxiv.org/abs/2508.21113): Related to multimodal reasoning enhancement
- Zero-Thinker (https://arxiv.org/abs/2503.05132): Relevant to inference-time reasoning approaches

### Questions
1. The paper commits to a specific symbolic representation (<==========>) but doesn't explore whether other symbolic forms might work equally well or better. For instance:
- Would using numbers with special tokens (e.g., [DIST:1.0], [DIST:0.1]) be more efficient?
- Could using other symbols or patterns provide similar or better results?
- How sensitive is the approach to the granularity of the symbolic representation beyond the δ=0.1 vs δ=0.05 comparison?

2. While Table 5 shows some transfer to real-world perception tasks, the mechanism of this transfer is not well understood. Why would training on synthetic geometric shapes improve performance on natural images with complex visual content?

3. The symbolic representation significantly increases sequence length. For a distance of 10 units, the model would generate 100+ tokens (<==========> repeated 10 times). What are the computational costs? How does this scale?

4. What are the main failure modes of PTS? Are there specific types of geometric configurations or estimation tasks where the approach struggles? Understanding limitations would be valuable.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the effectiveness of inference time scaling for perception-centric tasks. It first proposes a new visual estimation benchmark for testing LVLMs' perception capability and then proposes the idea of Perception Time Scaling, which encourages the model to produce richer perception-related tokens in perception-heavy tasks. Experiment results show promising performance improvements.

### Strengths
1. I like the idea of introducing the DisTANCE benchmark, which focuses on testing the perception of LVLMs. The idea is simple yet probes the models' capability in perception, in isolation of their textual reasoning capabilities. Experiment results show existing models struggle on this benchmark, even for the proprietary ones, and that reasoning models provide only marginal gains.

2. Analysis on the ratio of perception-related tokens (in Table2 and Figure 2) provides insights and good intuitions on the proposed method on perception elaboration and perception decomposition as the main axes for inference time scaling.

3. Main experiment setup ablates the effect of data (including Direct and CoT) and training objectives (SFT and RL). Evaluations cover perception related tasks, and also general multimodal benchmarks. The results are promising that PTS improves quite significantly on visual estimation tasks and maintain performance on general multimodal tasks.

### Weaknesses
1.  While the high-level idea of perception time scaling is interesting and perhaps general, the proposed method appears to be quite task-specific for visual estimation kind of tasks. For example, in perception elaboration, it is specifically using <==========> to represent length, and the decomposition part that teaches the models to use the operation of tiling both appears to be specific for visual estimation. Have the authors thought about how this perception time scaling paradigm can be used for other perception-centric tasks?

2. In the experiments, it is perhaps less surprising to see improvements on the proposed DisTANCE benchmark (given that the models are trained on this specific dataset). My main concern is how general the proposed method is. For instance, we see relatively much less performance improvements when tested on OOD datasets comparing PTS and the baselines. For Geo and Lego tasks, why are the other subsets not considered?

### Questions
1. In the experiments, why is there no "Direct with SFT" as a baseline?

Please see other questions in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
