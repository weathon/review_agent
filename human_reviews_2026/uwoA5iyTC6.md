# Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
To advance time series forecasting (TSF), various methods have been proposed to improve prediction accuracy, evolving from statistical techniques to data-driven deep learning architectures. Despite their effectiveness, most existing methods still adhere to a fast thinking paradigm - relying on pattern recognition and trend prediction as their core modeling philosophy, lacking an explicit "thinking process" that incorporates intermediate time series reasoning. Meanwhile, emerging slow-thinking LLMs (e.g., ChatGPT-o1) have shown remarkable multi-step reasoning capabilities, offering an alternative way to overcome these issues. However, prompt engineering alone presents several limitations—including high computational cost, privacy risks, and limited capacity for in-depth domain-specific time series reasoning. To address these limitations, a more promising approach is to train LLMs to develop slow-thinking capabilities and acquire strong time series reasoning skills. To acquire such slow thinking reasoning capabilities, we propose Time-R1, a two-stage reinforcement fine-tuning framework designed to enhance multi-step reasoning ability of LLMs for time series forecasting. Specifically, the first stage conducts supervised fine-tuning for warmup adaptation, while the second stage employs reinforcement learning to improve the model's generalization ability. Particularly, we introduce GRIP (group-based relative importance for policy optimization), which utilizes non-uniform sampling along with a fine-grained multi-objective reward specifically designed for time series forecasting to further encourage and optimize the model's exploration of effective reasoning paths. Experiments demonstrate that Time-R1 significantly improves forecast performance across diverse datasets. Source code is available https://anonymous.4open.science/r/Time-R1-NeurIPS-2025/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Time-R1, a two-stage reinforcement fine-tuning framework designed to enhance large language models (LLMs) for time series forecasting (TSF) by developing explicit multi-step reasoning capabilities.

### Strengths
- Several LLM-based models are used as benchmarks. Competitive forecasting benchmarks like PatchTST are also included.
- Presentation and results in Table 3 are clear and well-structured.
- The purposed approach appears to outperform most baselines across datasets.
- The figures are visually effective and contribute to understanding the proposed framework.
- The paper’s proposed framework for fine-tuning dataset construction is a valuable direction.

### Weaknesses
Several claims in the paper motivating use of reasoning for time series models are either vague or overstated:
   - For example, the statement “These models map history to future directly without detecting regime changes or performing step-by-step inference, resembling fast (not deliberate) thinking for time series” lacks precision. The notion of “regime changes” is not well defined. Modeling covariates could, for instance, already capture temporal dynamics associated with external events.
   - Lines 62–63: The phrase “time series often reflect more complex temporal logic, which should not merely be ‘fitted’—they should be understood and reasoned” is conceptually unclear. What is meant by “complex temporal logic” (e.g., domain knowledge, text-based covariates)? The authors should better justify why “reasoning” about such logic is necessary or beneficial compared to existing data-driven approaches.
   - Line 84: The assertion that the model is “fine-tuned for memorization” is a strong and potentially misleading claim. Forecasting fine-tuning does not inherently imply memorization, and the distinction between memorization and generalization is an active topic of research that requires supporting evidence.

Methodology contribution concerns: 
- Reasoning evaluation (Eq. 1): The evaluation procedure for reasoning remains unclear. If the metric is based solely on achieving the lowest MSE, it does not actually assess whether the model engages in reasoning, it merely rewards numerical accuracy.

- Reward design (Eqs. 3 and 4): The proposed reward functions appear conceptually similar to traditional loss formulations such as MSE and trend–seasonality decomposition. This raises questions about how they differ from conventional forecasting objectives. The authors critique prior approaches for lacking explicit reasoning, yet the reward relies on classical statistical constructs. The paper should clarify how this formulation advances the goal of reasoning-based forecasting. Moreover, this seems inconsistent with statements in the introduction (Lines 51–52): “Although effective in benchmarks, their underlying logic is largely based on pattern recognition (Cheng et al., 2025a) and trend prediction, lacking an explicit reasoning process.”

- Model architecture clarity: The underlying architecture on which Time-R1 is built is not clearly described. Rather than presenting Time-R1 as a standalone model, the contributions of the reasoning-oriented fine-tuning and reinforcement learning framework, would be more compelling if shown to consistently improve performance across multiple LLM backbones.

Benchmark concerns:
- The paper lacks comparisons with non-LLM-based Time Series Foundation Models (e.g., TimesFM, Moirai, Moment) and classical statistical baselines (e.g., ARIMA, ETS). Including such comparisons is crucial to validate its claimed advantages and motivate the use of LLMs for forecasting.

### Questions
Other:
- Lines 261-269: formulation of the GRIP Objective could be more clearly articulated to outline the provided equations.
- The purposed multi-trajectory approach in lines 292–300 is interesting. It could be interesting to explore ensembling of top-k sample trajectories. Such approach could reduce forecast variance an important consideration for forecasting in practice.
- In addition to the reward based on returned prediction sequence length, including a reward function based on prediction variance from distributional loss functions could help further improve model performance.
- Normalization is very often used in time series forecasting models. It could be interesting to show an ablation study with and without normalization to see how robust this approach is. 
- Some acronym and functions are not defined:
   - Function ‘g’ (Line 107) not defined.
   - SFT (Lines 095, 151) not defined.
   - Extrema (Eq. 5) and Eq. 6 unclear. Unclear how extrema are determined.

### Soundness
2

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
3

### Summary
This paper propose Time-R1, a novel time series forecasting framework that trains large language model with slow-thinking reasoning capabilities via a two-stage reinforcement fine-tuning pipeline. First, supervised fine-tuning uses synthetic chain-of-thought trajectories to teach the model temporal analysis and output formatting. Second, reinforcement learning with a fine-grained multi-objective reward function enhances generalization. A key innovation is GRIP, a non-uniform sampling and adaptive weighting strategy for optimizing reasoning paths. Experiments on nine diverse datasets show Time-R1 outperforms traditional deep learning and LLM-based baselines in MSE and MAE, improving temporal coherence and out-of-distribution generalization. By integrating explicit reasoning into TSF, Time-R1 addresses the "fast thinking" limitation of existing methods, offering both interpretability and state-of-the-art performance.

### Strengths
1. Introduces slow-thinking reasoning for TSF, replacing direct pattern mapping with explicit step-by-step temporal inference, enhancing interpretability and logical consistency.

2. Designing Two-Stage RFT Framework, SFT warmup ensures proper formatting and basic reasoning, while RL with tailored rewards optimizes for TSF-specific goals.

3. Non-uniform sampling and adaptive weighting balance exploration/exploitation, reducing computational cost while amplifying gradient signals from high-quality reasoning paths.

4. Very strong generalization, which trained on one dataset (ETTh1) but achieves superior performance across nine diverse domains, outperforming domain-specific baselines without task-specific fine-tuning.

### Weaknesses
1. There is significant discrepancy between the MSE losses of each reported model and the error values in their respective original papers, For instance, the MSE of the Exchange dataset is extremely small, while the errors of ETTh1 and ETTm1 are notably large. Additionally, there is a lack of necessary setup details for comparing the baseline models, and different model configurations may lead to unfair comparisons. 

2. Using text modality for input and output of time series may not  inefficient. For example, a numerical value with 4 decimal places like 16.3864 may require 2–4 tokens to represent, especially for larger number of variates and  time series lengths. 

3. It is not very reasonable to achieve better performance on other time series datasets by only training on the ETTh1 dataset, as the inherent characteristics and distributions of different data may vary significantly such as stock or foreign exchange time series. Please explain the underlying principle with experiments or theoretical support. 

4. There are minor writing errors. For example, in Appendix A.3: "offering computational efficiency but suffering from off-policy issues (?)."

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Time-R1, a framework that recasts time series forecasting as a reasoning task for an LLM. Moving away from traditional "fast-thinking" models that directly map historical data to future values, Time-R1 adopts a "slow-thinking" paradigm. In this approach, the LLM first generates an explicit, step-by-step reasoning process about the time series' properties (trends, seasonality, etc.) before outputting the final numerical forecast.

The framework employs a two-stage training process. First, SFT is used to adapt the LLM to the task format and instill basic reasoning patterns, using synthetic reasoning trajectories generated by a more powerful "teacher" model. Second, RL is applied to refine and generalize this reasoning ability. A key part of the RL stage is a novel optimization algorithm called GRIP, which uses non-uniform sampling to focus on high-reward reasoning paths, guided by a fine-grained, multi-objective reward function. Experiments show that a model trained on a single dataset (ETTh1) can generalize effectively to eight other unseen datasets, outperforming traditional TSF models.

### Strengths
1. Framing TSF as a "slow-thinking" reasoning task is a compelling and innovative idea that could open up new avenues for building more intelligent forecasting systems.
2. The model's ability to train on one dataset and perform well on eight others is a significant strength, showcasing that it learns transferable reasoning skills rather than just dataset-specific patterns.
3. The generation of an explicit reasoning chain before the forecast provides a degree of interpretability that is absent in most traditional "black-box" TSF models.
4. The paper includes comprehensive ablations that effectively demonstrate the importance of each part of the complex pipeline, from SFT and RL to the individual reward components.

### Weaknesses
1. The entire framework is exceptionally complex, involving synthetic data generation with a powerful teacher model, SFT, and a sophisticated RL pipeline with a custom optimizer. The reported inference speed (~3 samples/s) and training requirements make it impractical for many real-world applications. The trade-off between accuracy and efficiency is severe. 
2. The SFT stage relies on reasoning paths generated by another LLM that is prompted with the ground-truth answer. This could lead to the model learning to generate "post-hoc justifications" rather than engaging in genuine, from-scratch reasoning. The quality of the final model is heavily dependent on the quality of the teacher model.
3. The multi-objective reward function is a complex combination of several terms, and the paper provides little justification for the specific formulation and weighting of these components. This makes the design feel ad-hoc and potentially difficult to transfer to other tasks.
4. While presented as a key contribution, the GRIP algorithm offers a modest improvement over existing methods like GRPO. The core ideas of focusing on high-reward samples are not entirely new, and the ablation in Figure 4a confirms the improvement is not dramatic.

### Questions
I don't think all TSF tasks require slow thinking. Is it possible to clearly define when we need slow thinking and when fast thinking is sufficient to obtain good results? Even for slow thinking, what specific real-world applications do you envision for this "slow-thinking" paradigm, where the benefits in accuracy and interpretability would justify the extreme costs in latency and compute?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Time-R1, a two-stage reinforcement fine-tuning (RFT) framework designed to enhance LLMs' reasoning capabilities for time series forecasting (TSF). The approach consists of: (1) a warmup supervised fine-tuning (SFT) stage using chain-of-thought (CoT) trajectories generated by DeepSeek-R1, and (2) a reinforcement learning stage employing GRIP, which uses non-uniform sampling and adaptive trajectory weighting with a multi-objective reward function. Experiments on nine datasets demonstrate improvements over baseline methods including traditional deep learning models and LLM-based approaches. The authors argue that slow-thinking reasoning provides better temporal understanding than fast-thinking pattern matching.

### Strengths
(1)  The distinction between "fast-thinking" (pattern matching) and "slow-thinking" (explicit reasoning) paradigms is intuitive and well-articulated. The motivation that time series contain complex temporal logic warranting structured reasoning is compelling.
(2) The combination of SFT (for format stabilization and basic reasoning) followed by RL (for generalization and reasoning refinement) is a principled approach that addresses practical training challenges. The ablation study (Table 3, Figure 4b) effectively demonstrates the necessity of both stages.

### Weaknesses
(1) Using LLMs to generate synthetic CoT trajectories for fine-tuning is not novel (e.g., similar to approaches in reasoning LLMs). The key distinction from existing work is unclear.
(2) While the adaptive weighting is presented as novel, it is essentially a variant of importance sampling with softmax normalization. The core algorithmic novelty is limited. The comparison with GRPO (Figure 4a) shows only marginal improvements.
(3) The paper claims "slow thinking" but this mirrors existing reasoning-based LLM approaches (o1, DeepSeek-R1). What is fundamentally new beyond applying them to TSF?
(4) SFT data is generated using DeepSeek-R1, which itself is a slow-thinking model. This creates a conceptual circularity: improving reasoning for TSF via a model that already has slow-thinking capabilities.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
