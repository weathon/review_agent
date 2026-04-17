# Inference-Time Alignment via Hypothesis Reweighting

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Chat assistants must handle diverse and often conflicting user preferences, requiring adaptability to various user needs.
We propose Hypothesis Reweighting (HyRe), a method that enables real-time personalization by reweighting ensemble members based on just 1-5 labeled examples from the target user or domain.
Our key insight is that uniform ensemble averaging, while effective on the training distribution, often underperforms individual ensemble members under distribution shift.
HyRe trains a single network with multiple prediction heads that capture different valid interpretations of preference data, then performs a simple Bayesian update to upweight heads that best match the target user's preferences.
This requires only a single forward pass with negligible (<1\%) computational overhead, making it practical for inference-time alignment.
We empirically validate HyRe in several target evaluation distributions.
With as few as five preference pairs from each target distribution, adaptation via HyRe surpasses state-of-the-art reward models on RewardBench at both the 2B and 8B parameter scales, and improves reward model accuracy by 20\% across 32 diverse personalization tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In order to maximize the benefits of LLMs in various domains, it is necessary to personalize or adapt their outputs to each user and query they encounter.  While LLMs are trained with RLHF to model average preferences across large populations, at test time we hope the model can adjust to the particular needs and preferences of the current user.  To address this challenge, this paper proposes HyRe.  HyRe is a method for inference-time alignment or adaptation, where a few labelled examples are used to decide how to weight the predictions of various ensemble members.  The authors offer a Bayesian interpretation of the approach.  HyRe’s effectiveness is examined in experiments on, e.g., regression and reward modeling.

### Strengths
This paper addresses a timely challenge that is of significant interest to the community.  While LLMs are quite good at modeling average text and following average preferences, unlocking their full potential requires adapting the models to the particular environment they face in deployment.  Additionally, it should be hoped that this adaptation will be active and continuous.  HyRe attempts to take a step in this important direction.

### Weaknesses
While I appreciate the goals of this paper, ultimately I find that it has significant weaknesses.

First, I find that the algorithm lacks novelty, and the paper does not clearly place it amongst the existing literature.  For example, I think that the comparison made in lines 41-46 should also discuss the connection between this approach and mixture of experts, as opposed to ensembling).  Also, I think all of the experiments should include trained MoE comparisons.  

Ultimately this algorithm is quite simple: train an ensemble, then take i.i.d. target data associated with a query and evaluate your models to decide which to use.  I am not convinced that this is actually practical though.  On the one hand, we can not expect to get labeled examples corresponding to each test example that comes into a chatbot, so HyRe is not useful to adapt to each individual example.  On the other hand, if we’re only applying this every so often, then collecting 100-200 examples and doing LoRa seems feasible.  I am not sure where exactly this algorithm is useful.  Relatedly, in Section 6.3 it says “Results in Table 2 indicate that while targeted ﬁne-tuning models achieve higher performance in their respective target metrics, they significantly reduce performance in the others…” But why should this matter in this setting?  For each test example, we know that it’s from the same target distribution as the support set, so once you’ve done that adaptation you don’t expect that model to have to address data from another distribution.

Also, I would note that this seems very reminiscent of classic few shot learning, e.g. ProtoNets (https://arxiv.org/abs/1703.05175).  That line of work should be discussed.

I am somewhat confused by Section 3.1: isn’t it trivially true that a single model will perform better than the average, unless they all perform the same?  The problem is usually that people don’t assume they can evaluate all of the ensemble members on i.i.d. target data to figure out which is best, and thus they use the ensemble (or MoE) for inference.

Finally, I appreciate the authors including the WILDS experiments where their method did not perform as well.  However, this also detracts from my assessment of the potential impact of this algorithm.

### Questions
Can you describe some practical scenarios where you imagine HyRe could be applied (i.e., you're getting the labels you need to do this), but fine-tuning would not be feasible?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Hypothesis Reweighting (HyRe), an efficient method for aligning models to diverse target distributions at inference time. HyRe addresses the computational cost of traditional adaptation by first training a single shared backbone network with multiple lightweight prediction heads, which capture different valid interpretations of the training data. Then, at inference time, it uses as few as 1-5 labeled examples from a new target distribution to perform a simple Bayesian update. This reweights the heads to favor those that best match the target distribution's characteristics. This approach is motivated by the insight that under distribution shift, a uniform ensemble average is often suboptimal compared to a well-chosen individual member. The method is highly practical, requiring only a single forward pass with negligible computational overhead, yet it achieves state-of-the-art results on RewardBench at 2B and 8B scales and significantly improves accuracy across 32 diverse personalization tasks.

### Strengths
- **Practical and efficient approach:** The method addresses a real-world problem—adapting models to diverse user preferences—with minimal computational overhead. The single forward pass requirement makes it genuinely practical.
- **Solid theoretical grounding:** The connection to generalized Bayesian inference provides a principled framework, and the analysis of when/why ensemble reweighting works (Section 4) adds valuable insights.
- **Clear presentation:** The paper effectively motivates the problem and solution, with good use of figures and examples.

### Weaknesses
- **Analysis depth:** Section 4's toy examples are insufficient to explain the method's success on complex language tasks. Is there any interpretable feature or analysis of each head in language models?
- **Missing comparison with recent personalization methods:** Recently, there has been much research[1] addressing the personalization of large language models. Some of this research could be compared with HyRe in your settings, where there are only a few adaptation samples with a pretrained model (e.g., [2] and their baselines).

[1] Zhang, Zhehao, et al. "Personalization of large language models: A survey." arXiv preprint arXiv:2411.00027 (2024).


[2] Shenfeld, Idan, et al. "Language model personalization via reward factorization." arXiv preprint arXiv:2503.06358 (2025).

### Questions
- Please see the weaknesses above.
- Is there any potential degradation in other domains that are not related to the target distribution? For example, a model aligned for harmlessness using HyRe could degrade in performance on math problem-solving tasks.

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
3

### Summary
**Summary**

This paper proposes Hypothesis Reweighting (HyRe), a method for inference-time alignment and personalization of large models.  HyRe uses multiple prediction heads to a shared backbone, each representing a distinct interpretation of the data. At test time, it evaluates each head on a small adaptation set (1–5 labeled samples) from the target user or domain, and applies a Bayesian reweighting based on accuracy to form a weighted ensemble. HyRe shows state-of-the-art performance on RewardBench at 2B and 8B scales, outperforming GPT-4o and Gemini-1.5-Pro with minimal supervision, and shows **improvement** across 32 personalization tasks. Additionally, It also shows strong performance on regression under mild distribution shifts (UCI datasets).

### Strengths
### **Strengths**

1. **Simple yet effective inference-time adaptation.**
   HyRe introduces a lightweight and elegant alternative to fine-tuning, enabling model personalization through Bayesian reweighting of multiple prediction heads , requiring only a single forward pass and minimal compute overhead.

2. **Strong empirical performance.**
   The method achieves state-of-the-art results on **RewardBench** (2B and 8B models), surpassing larger systems such as **GPT-4o** and **Gemini-1.5-Pro**, and consistently improves performance across **32 diverse personalization tasks** and several regression benchmarks.

3. **Efficiency and scalability.**
   The added parameters (<0.03%) and runtime overhead (<1%) make HyRe practical for deployment in large models, with adaptation possible from as few as 1–5 labeled samples.

4. **Comprehensive experiments and analysis.**
   The paper includes detailed ablations (reweighting criteria, head diversity, non-i.i.d. adaptation) and analyzes both success and failure cases (e.g., limited gains on WILDS).

### Weaknesses
### **Weaknesses**

1. **Limited conceptual novelty.**
   While the formulation is elegant, the idea of inference-time ensemble reweighting extends existing Bayesian ensemble and dynamic weighting techniques rather than introducing a fundamentally new learning paradigm.

2. **Dependence on head diversity.**
   The method’s effectiveness depends on adequate diversity among prediction heads. As shown in the WILDS experiments, limited diversity reduces adaptation effectiveness.

3. **Lack of comparison to self-consistency [1]**
   Since HyRe and self-consistency prompting share the idea of aggregating multiple hypotheses at inference time, a direct experimental comparison would clarify their differences and relative advantages.

References: 
[1] Self-Consistency Improves Chain of Thought Reasoning in Language Models

### Questions
### **Q1. Conceptual connection to self-consistency prompting**

The authors might consider discussing the conceptual similarity between **HyRe** and **self-consistency prompting** [1]. Both methods aggregate multiple hypotheses at inference time—HyRe through **Bayesian accuracy-based reweighting** of learned heads, and self-consistency through **voting over diverse reasoning paths** sampled from the same model.
A brief comparison could help clarify how HyRe generalizes these ideas to **few-shot, labeled adaptation** and provides an explicit **Bayesian formalization** of inference-time hypothesis aggregation.

---

### **Q2. Comparison and label-free setting**

Including **self-consistency prompting as an additional baseline** would make the evaluation more convincing.
For instance, the authors could repeatedly sample outputs (e.g., 100 generations) from the base model and select the most frequent prediction as the final answer, following the standard self-consistency protocol.
Because self-consistency is **label-free**, it would be valuable to discuss how HyRe could operate under a similar unsupervised scenario—either by estimating pseudo-labels or by running HyRe with **1–5 pseudo-labeled adaptation examples** as a demonstration.
This comparison would clarify whether HyRe’s Bayesian weighting brings consistent advantages over stochastic consensus-based aggregation.


References: [1] Self-Consistency Improves Chain of Thought Reasoning in Language Models

### Soundness
3

### Presentation
4

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
This paper proposes Hypothesis Reweighting (HyRe), a method for personalizing large-scale AI models in real time and in a computationally efficient way to match diverse user preferences. The method trains a shared backbone network with multiple prediction heads and dynamically reweights these heads at inference time using a few labeled examples from the target distribution.
HyRe applies generalized Bayesian inference for updating ensemble weights, supporting non-differentiable metrics and robust adaptation.
Experiments on RewardBench, WILDS, and PERSONA datasets demonstrate consistent improvements over uniform ensembles and standard fine-tuning in low-data settings while remaining negligible additional overhead.

### Strengths
- HyRe shows significant gains with as few labeled examples, outperforming fine-tuning and uniform ensembles, which has strong applied significance for LLM alignment and model deployment.
- The method adds negligible parameters and imposes minimal inference-time compute, which is critical for deploying large models in production environments.

### Weaknesses
- Even the paper explicitly frames itself as inference-time reweighting, the underlying mechanism seems to be closely related to existing methods in multi-task learning, multi-objective optimization, and mixture-of-experts. The paper conceptually differentiates itself from MoE by avoiding training-time gating and instead performing inference-time adaptation, but it still lacks a direct empirical comparison to a learned MoE gating baseline or an explicit ablation isolating this distinction.
- According to Figure 4, HyRe performs best in the extreme few-shot regime, while fine-tuning begins to outperform it when ≥ 64 labeled examples are available. If the task is well-specified and labeled data are plentiful, fine-tuning may ultimately achieve higher accuracy. However, the paper does not provide an explicit compute-versus-accuracy comparison; since HyRe uses only a single forward pass for adaptation, quantifying its wall-clock or FLOPs cost relative to fine-tuning would clarify this trade-off.
- The method’s success depends on the ensemble heads capturing useful and relevant functional diversity during training. If a user’s preference lies outside the span of this learned diversity, HyRe may fail to find an appropriate head. The authors acknowledge this limitation, noting weaker gains on certain WILDS datasets.

### Questions
- The number of ensemble heads seems to be a crucial hyperparameter. Was a sensitivity analysis performed to see how performance and computational cost change as the vary of number of heads?
- Since the paper already compares to fine-tuning and few-shot prompting, adding a comparison with parameter-efficient tuning methods (e.g., LoRA or adapters) would help contextualize HyRe among other lightweight adaptation strategies.

### Soundness
3

### Presentation
3

### Contribution
3
