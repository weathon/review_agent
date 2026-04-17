# Adaptive Curriculum Strategies: Stabilizing Reinforcement Learning for Large Language Models

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Curriculum learning has shown promise for enhancing Large Language Models (LLMs) through progressive difficulty management, yet existing approaches suffer from instability issues when applied to reinforcement learning paradigms. Existing curriculum-based RL training exhibits catastrophic performance collapse during difficulty transitions, particularly when models encounter samples beyond their current capabilities. This instability stems from rigid curriculum designs that fail to adapt to individual model characteristics and learning trajectories. To address these limitations, we propose Adaptive Curriculum Strategies (ACS), a framework that promotes stable and effective training throughout curriculum progression. Our approach introduces model-specific difficulty calibration that adapts to each model's capabilities, and ``Guided Prompting'' that transforms challenging samples to prevent training instability. Experiments demonstrate that ACS prevents performance collapse in traditional curriculum RL training, achieving substantial improvements across five mathematical reasoning benchmarks while enhancing training stability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Adaptive Curriculum Strategies (ACS) to stabilize reinforcement learning for LLM mathematical reasoning by (a) model-specific difficulty calibration via multi-sample accuracy per item and (b) Guided Prompting that transforms hard problems with partial-solution hints until the model meets a stability threshold, combined with staged GRPO training and a curriculum review data-mixing strategy to mitigate collapse and forgetting.

### Strengths
- Clear identification of instability during curriculum stage transitions in RL fine-tuning, with an operational definition and visualizations illustrating catastrophic drops without stability mechanisms.
- Concrete, implementable pipeline: per-sample multi-draw accuracy calibration, guided hinting with thresholds, and staged GRPO augmented with a curriculum review mixing policy that reduces forgetting.
- Consistent gains across five math benchmarks on two model sizes, plus cross-model results on DeepSeek-Math-7B-Instruct; ablations indicate curriculum review outperforms naive staging.

### Weaknesses
- Narrow domain scope: all tasks are mathematical reasoning; there is no evidence ACS generalizes to code, QA, multi-turn dialogue, or retrieval-augmented regimes, limiting external validity of stability claims.
- Stability attribution is under-isolated: guided prompting, data partitioning, GRPO modifications, and review mixing change simultaneously; ablations do not fully disentangle which component prevents collapse under consistent compute budgets.
- Guided Prompting may leak reference solution structure into training examples, risking distribution shift and overfitting; safeguards and analyses of hint-length sensitivity or label leakage are not provided.
- Difficulty calibration relies on n=16 sampled generations per item with temperature 0.7; the resulting ACC is stochastic and decoding-dependent, yet robustness to n, temperature, and evaluator scripts is not quantified.
- The stability-aware GRPO objective is presented, but theoretical guarantees on stability (e.g., monotone improvement, bounded gradient variance across curriculum transitions) are not established, leaving “stability” as an empirical observation.
- Baseline protocols vary in ways that may advantage ACS (e.g., discarding vs retaining hard samples, or using fixed external assessors) without strong hyperparameter sweeps or compute parity evidence across methods.

### Questions
- How robust are the results to different n and temperature choices in the calibration step, and do deterministic decoding or alternative evaluators (e.g., verifier models) change partitioning outcomes?​
- Does Guided Prompting induce dependency on solution prefixes at inference time, and how does performance change if hints are removed post-training or restricted to schematic advice rather than literal step prefixes?​
- Can the components be ablated under equalized compute to quantify each contribution to stability: calibration only, prompting only, review only, and GRPO stability term only?​
- How does ACS behave on non-math tasks (e.g., GSM8K vs HotpotQA vs code benchmarks) and with retrieval-augmented inputs, where difficulty and instability arise from different factors?​
- What are memory/latency impacts in large-scale settings: calibrating 100k–1M samples, longer max lengths, larger candidate counts, and bigger models; is <5% overhead still valid?​
- Could a verifier-based or cost-sensitive calibration (penalizing formatting or compute) outperform raw ACC, and does mixing by calibrated uncertainty (rather than discrete tertiles) improve stability?​

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies why curriculum learning combined with RL for LLM mathematical reasoning often collapses at stage transitions and proposes Adaptive Curriculum Strategies (ACS) to keep training stable and effective. The core idea is to calibrate difficulty for the current model via multi-sample accuracy estimates, transform over-difficult items with Guided Prompting (prefix hints from the reference solution) so they stay learnable. Experiments across different math benchmarks and model sizes show ACS removes the catastrophic drops seen in naïve curricula and improves average performance

### Strengths
1. The observation and the analysis on the collapses at difficulty stage transitions is interesting.

2. The studied problem is important.

3. Different model families (both Qwen and DeepSeek Math) are involved in the experiments, demonstrating the generalizability of the proposed method across different models.

4. Guided prompting method is reasonable.

### Weaknesses
1. This paper does not discuss and compare against many existing curriculum-learning methods for reinforcement learning, despite a growing body of existing works (see references below). The reported baselines are naive heuristics, which makes it hard to compare the proposed methed and other advanced curriculum learning methods. In addition, the related-work section does not adequately position the method within prior curriculum strategies on RL. A better evaluation should includes more existing RL curriculum learning methods. It is also suggested to expand the related-work discussion to clarify what is new versus known in curriculum design for LLM RL and traditional RL. Some of the existing methods listed below are also adaptive curriculum instead of learning via fixed difficulies orders/phases.

2. All experiments use GRPO; there is no evaluation against other LLM-RL algorithms such as PPO, DAPO, or Reinforcement++. This makes it unclear whether ACS is only applicable to GRPO. Adding PPO/DAPO/Reinforcement++ experiments, would make the experiments more comprehensive and verify the generalizability of the proposed method.

Zhang et al., Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation. EMNLP 2025.

Tzannetos et al., Proximal Curriculum for Reinforcement Learning Agents. TMLR 2023.

Shi et al., Efficient Reinforcement Finetuning via Adaptive Curriculum Learning

Parashar et al., Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning

Wang et al., DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training

Chen et al., Self-Evolving Curriculum for LLM Reasoning

Bae et al., Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning

### Questions
See Weaknesses.

### Soundness
3

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
3

### Summary
This paper proposed to improve reinforcement training using curriculum strategy. Compared to existing literature that studies curriculum assisted RL training, it adopts a more robust difficulty measurer and proposes guided prompting to mitigate instability training.

### Strengths
1. The paper is well written and easy to follow.
2. The authors have conducted extensive experiments on three backbone models, namely Qwen2.5 Math 1.5B, Qwen2.5 Math 7B and Deepseek Math 7B Instruct.

### Weaknesses
Weaknesses and Questions:
1. For section 3.1, I think the curriculum strategy adopted in this paper is basically BabyStep[1], with the step number set to 3. Although the difficulty is calculated based on the current accuracy, it is still pre-defined. Since the authors claim to "adapt sample assessment to each model's evolving capabilities", I believe Self-paced Learning[2] (SPL) will be a much better choice. SPL is a variant of automatic curriculum learning[3] which adopts a dynamic scheduler. For some variant of SPL, samples may be assigned a dynamic weight based on the current capability of the base model, which may avoids the problem of training instability when advancing to the next stage.
2. I believe a large part of the contribution lies in the curriculum strategy applied in reinforcement learning. However, some hyper-parameters for curriculum learning such as step number are not ablated. Meanwhile, only two variants of curriculum learning strategies are experimented, and for Curriculum Review, can the authors specify exactly how many easier samples are incorporated? And is the proportion static across different models? 
3. For guided prompting, the difficulty of training samples is manually reduced by providing “hints.” However, wouldn’t this approach potentially weaken the model’s ability to handle difficult samples, since it only encounters samples with hints? Is there a mechanism to gradually remove these hints during training? Moreover, the hyper-parameters — the hint ratio \alpha and threshold \tau — are not specified or ablated, even though they may play a critical role in the effectiveness of this strategy. Are these parameters kept static across different models?

References:

[1] Spitkovsky, V. I., Alshawi, H., & Jurafsky, D. (2010, June). From baby steps to leapfrog: How “less is more” in unsupervised dependency parsing. In Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics (pp. 751-759).

[2] Wang, X., Chen, Y., & Zhu, W. (2021). A survey on curriculum learning. IEEE transactions on pattern analysis and machine intelligence, 44(9), 4555-4576.

[3] Tullis, J. G., & Benjamin, A. S. (2011). On the effectiveness of self-paced learning. Journal of memory and language, 64(2), 109-118.

### Questions
Please refer to the Weakness.

### Soundness
2

### Presentation
3

### Contribution
2
