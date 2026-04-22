# Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
Preference optimization is crucial for aligning large language models (LLMs) with human values and intentions. A significant challenge in this process is the distribution mismatch between pre-collected offline preference data and the evolving model policy. Existing methods attempt to reduce this gap using static heuristics or decoupled online sampling strategies, but they often fail to adapt to the model's dynamic learning state. To bridge this gap, we propose Meta-Weighted Adaptive Preference Optimization (MetaAPO), a novel framework that dynamically couples data generation with model training. MetaAPO employs a lightweight meta-learner, as an "alignment gap estimator", to evaluate the potential benefits of on-policy sampling in relation to offline data. This guides targeted online generation and assigns sample-wise meta-weights to the optimization objective, dynamically balancing the quality and distribution of online and offline data. Experiments on AlpacaEval 2, Arena-Hard and MT-Bench demonstrate that MetaAPO consistently outperforms existing preference optimization approaches across various settings, while reducing 42% in online annotation costs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the distribution mismatch of the offline alignment dataset and the evolving model state in preference optimization.
The authors propose Meta-Weighted Adaptive Preference Optimization (MetaAPO), a framework that adaptively adjusts the online sampling and training sampling weights based on the model's current state.
To achieve this, they utilize a meta-learner (two-layer MLP) that takes the model's preference score on each sample as input to determine whether to augment the sample with online sampling and to adjust the training sampling weight accordingly.
The training of this meta-learner encourages it to improve the overall preference score of the current model on the training samples (including both online and offline samples).
Experiments on different models and benchmarks demonstrate the effectiveness of MetaAPO in improving preference optimization performance over existing static and online methods.

### Strengths
1. **Clear Motivation and Simple Solution**: Solving the distribution mismatch between offline alignment data and the evolving model state is a well-motivated problem in direct preference optimization scenarios, and MetaAPO provides a straightforward adpative sampling and weighting solution to address this issue.
2. **Writing Quality**: The paper is well-written and easy to follow, with clear explanations of the proposed method and experimental results.

### Weaknesses
1. **Concerns on Reasonability**: The preference score $\ell$ input to the meta-learner measures the agreement between the model and the human preference data, with a higher score indicating a lower DPO loss and alignment with human preferences. 
The objective of the meta-learner is to improve the overall preference score of the model on training samples, so data samples that the model already agrees with (high $\ell$) are less likely to be augmented with online sampling and are assigned higher training weights. This seems counterintuitive, as one would expect the model to focus more on samples it currently fails to align with (low $\ell$) to improve alignment [1,2]. The authors should provide more justification for this design choice.
2. **Limited Ablation Studies**: As shown in Figure 3, the learned meta-learner's mapping from preference score $\ell$ to weight $w$ appears to be similar across different iterations. This raises the question of whether a fixed sampling/loss weighting scheme could achieve similar performance. Although the ablation study "w/ uniform loss weighting" and "w/o meta-learner" are provided, a loss weighting/ sampling scheme based on the final learned meta-learner (fixed after training) should be included for a more direct comparison.
3. **Baseline Comparisons**: It seems like the ablation "w/ all sampling" is equivalent to the "online DPO" baseline? The authors should clarify the difference between these two settings to avoid confusion. Moreover, the authors should consider including more data filtering baselines in the online sampling setting, such as [2,3,4] (which are already cited in the related work section), to provide a more comprehensive comparison since they also take the model state into account when selecting samples for preference optimization.

**References**\
[1] Yang, Sen, et al. "Not All Preference Pairs Are Created Equal: A Recipe for Annotation-Efficient Iterative Preference Learning." Findings of the Association for Computational Linguistics: EMNLP 2024. 2024.\
[2] Huang, Kexin, et al. "Larger or Smaller Reward Margins to Select Preferences for LLM Alignment?." Forty-second International Conference on Machine Learning.\
[3] Deng, Xun, et al. "Less is more: Improving llm alignment via preference data selection." arXiv preprint arXiv:2502.14560 (2025).\
[4] Gao, Chengqian, et al. "Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples." Forty-second International Conference on Machine Learning.

### Questions
Can you provide more ablation studies comparing to fixed sampling/loss weighting schemes based on the learned meta-learner after training, as mentioned in Weakness 2?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes MetaAPO, a meta-learning-based framework for adaptive preference optimization. Instead of statically mixing offline and online preference data, the method introduces a meta-learner that dynamically estimates the alignment gap between the current policy and offline data. The meta-learner assigns sample-specific weights $w_i$ and decides whether to perform online augmentation. This yields an adaptive data selection and loss weighting mechanism that focuses computation on high-gain samples. Experiments on Llama-3.1-8B and Qwen2.5-7B across AlpacaEval 2, Arena-Hard, and MT-Bench show consistent improvements over offline (DPO, IPO, KTO), online (PPO, Online DPO), and hybrid (SELM, ADPO) baselines.

### Strengths
1. The paper introduces a clear and well-motivated framework that integrates meta-learning into preference optimization. By learning sample-specific weights to balance offline and online data, MetaAPO effectively unifies filtering and augmentation within a single adaptive process, avoiding the heuristic thresholds used in prior methods such as ADPO or Selective DPO.
2. The proposed meta-weighted sampling mechanism demonstrates strong empirical advantages. It focuses training resources on high-gain samples, achieves smoother “explore–integrate” dynamics, and significantly reduces annotation costs while maintaining or improving performance across multiple benchmarks.
3. The experimental evaluation is comprehensive. The authors include diverse baselines (offline, online, and hybrid), three major alignment benchmarks and detailed ablations on sampling and loss weighting.

### Weaknesses
One of the limitation is the sensitivity to hyperparameters introduced by the meta-learner, particularly the update interval $T_{meta}$. Achieving optimal results appears to require careful tuning, which may increase the implementation burden. However, this does not substantially detract from the overall contribution given the consistent empirical gains.

### Questions
None

### Soundness
3

### Presentation
4

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
This paper proposes MetaAPO, which introduces a lightweight meta-learner into preference optimization (RLHF/DPO). The meta-learner dynamically learns sample weights from offline preference scores to jointly guide online sample selection and the weighting of offline/online data during training. Across several mainstream benchmarks, MetaAPO outperforms existing alignment methods (e.g., DPO, IPO), achieving better alignment performance and higher data efficiency.

### Strengths
(1) Proposes a clear, intuitive, and novel online–offline joint optimization framework with strong practical value.
(2) Demonstrates consistent effectiveness across multiple models and benchmarks.
(3) Provides solid theoretical derivations with comprehensive ablation and robustness studies.

### Weaknesses
(1) The meta-learner’s input features are overly simplistic, which may limit generalization. Although the appendix briefly explores multi-feature inputs (Table 7), the main experiments rely solely on a single offline score, lacking analysis of why this is sufficient.
(2) The paper lacks qualitative and human evaluation. Results rely solely on GPT-4o as the judge, without validating consistency with human preferences. Case analyses are limited in number and do not cover complex scenarios (e.g., multi-turn dialogue or safety alignment).
(3) The online sampling design (K=8 and stochastic thresholding) is not empirically justified, and its sensitivity to these parameters remains unexplored.
(4) Recent baselines (e.g., DPO-Shift, MAP) and multi-stage alignment frameworks (e.g., RLHF + DPO hybrids) are not included, which may under- or overestimate MetaAPO’s relative advantages.

### Questions
(1) Have you tried adding richer features (e.g., model gradients, response semantic similarity) as inputs to the meta-learner? If not, please discuss potential information loss from relying solely on a single offline score.
(2) Could you report results for different numbers of sampled candidates (K=4/12/16) to analyze the trade-off between performance and computational cost?
(3) Can you include a small-scale human evaluation or justify the consistency between GPT-4o judgments and human preferences?

### Soundness
3

### Presentation
2

### Contribution
2
