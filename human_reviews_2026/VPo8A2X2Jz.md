# ESLM: Risk-Averse Selective Language Modeling with Hierarchical Batch Selection

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Large language model pretraining is compute-intensive, yet many tokens contribute marginally to learning, resulting in inefficiency.
We introduce Efficient Selective Language Modeling (ESLM), an online, risk-aware batch selection algorithm that improves training efficiency and distributional robustness. ESLM operates in two phases: $(i)$ instance-level selection via a shallow early-exit model pass that computes proxy per-instance statistics (e.g., loss or entropy) and retains data points using value-at-risk thresholding; and $(ii)$ loss shaping with token-level selection via risk-aware thresholding on per-token scores. This data-centric mechanism reshapes the training objective, prioritizing high-risk tokens and eliminating redundant gradient computation. We frame ESLM as a bilevel game: the model competes with a masking adversary that selects worst-case token subsets under a constrained thresholding rule. In the loss-based setting, ESLM recovers conditional value-at-risk loss minimization, linking selective pretraining to distributionally robust optimization. We extend our approach to Ada-ESLM, which adaptively tunes the selection confidence during training. Experiments on GPT-2 pretraining show that ESLM significantly reduces training FLOPs while maintaining or improving perplexity and downstream performance compared to baselines. Our approach also scales across model sizes, pretraining corpora, and integrates naturally with knowledge distillation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel token-level training data selection algorithm that aims to boost training efficiency by focusing exclusively on high-risk tokens, thereby reducing FLOPs. The method further enhances performance through dynamic token selection ratios and by incorporating probabilistic signals for selecting tokens from larger models.

### Strengths
* **Clear Writing.** The paper is readable and well-structured.

* **Strong Motivation.** The motivation—to avoid spending full compute on low-value tokens and instead concentrate updates on high-risk (uncertain/high-loss) tokens—is compelling and well positioned relative to standard CLM and online data selection.

* **Extensive Experiments.** Compared to the previous version, the current experiments and analysis are broader, making the results more credible and the analysis more comprehensive.

### Weaknesses
* **Wall-clock time remains higher than CLM despite lower FLOPs.** While ESLM cuts training FLOPs, its end-to-end training time is still **longer than CLM** in the reported 124M setting (≈9.9–11.7 h vs. 5.3 h). The authors attribute the overhead to mismatches between selective sparsity and current accelerator kernels/tensor fusion, but this nonetheless constrains real-world applicability where time-to-train dominates. The paper acknowledges this in the limitation section.

* ESLM is a solid and practical recipe. However, beyond the engineering, the conceptual insight feels limited: Beyond the tasks discussed in this paper, how does the discussion of “high-risk” tokens provide utility or insight in broader tasks and scenarios? Moreover, given the limited applicability of the task itself, these two factors together limit a higher score.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the algorithm Efficient Selective Language Modeling (ESLM) to reshapes the training objective. It prioritizing high-risk tokens while eliminating redundant gradient computation. The author further conduct experiments to demonstrate the superiority of algorithm.

### Strengths
1. The algorithm introduction is quite detailed.
2. The ablation study in terms of different hyper-parameters are quite detailed.

### Weaknesses
1. The novelty is limited. The core idea $VaR$ is not firstly proposed. I'm curious its difference compared against predictive entropy and perplexity. The intuition why we need risk-aware metric is not clear.
2. The comparison between VaR-entropy and CVaR-loss is absent. In Table 1, it looks like both methods have advantages in some benchmarks but I don't know how to use them in practice.
3. In Table 11, Table 12, Table 9, we can see that the final performance also depends on the training data and model size, which challenges the robustness of the algorithm.
4. The author claim ESLM is a bilevel adversarial game between the model and a masker. I am a little confused how the author optimize the masker. In Algorithm 1 line 11, ESLM only updates the model parameters.

### Questions
Please check the weakness part.

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
4

### Summary
ESLM introduces a hierarchical selection framework that filters training data at both the sequence and token levels, focusing computational resources on high-risk, informative examples. The approach is grounded in risk-sensitive optimization. An adaptive variant dynamically adjusts selection strictness during training, and the method extends to enable efficient knowledge distillation. Empirically, ESLM reduces training compute.

### Strengths
- Improving the sample efficiency during LLM pre-training warrants large potential more effective pipelines 
- The hierarchical nature of ESLM seems to be quite useful to actually reduce the token count during pre-training (in contrast to previously existing approaches where instances or tokens were down-weight without actually getting removed)
- I appreciate the evaluation of the computational cost (FLOPs)

### Weaknesses
- I am wondering how the dynamics over time change when the objective changes towards being risk averse? It is not fully clear to me how ESLM treats extreme instances or tokens. How does the behavior of LLMs change in edge cases when tuning the risk score threshold? I am refering to extreme cases (e.g., instances that carry little information or extremely long token sequences).
- There is an instance re-weighting paper that was presented at ICLR 2025 that also uses batch statistics only and seems to have similar performance characteristics [1]. I am wondering how much benefit each stage (esp. stage 1) of ESLM adds on its own. 
- The models discussed in the paper are small and the pre-training dataset of choice (SlimPajama-6B) is also not large enough to yield sufficient insights into model quality. While 1.5B parameter models are frequently used for ablation studies, these models are typically trained with 30B+ tokens [1]. 
- The paper discusses reproducibility in various points. I am wondering if there is a code implementation that comes with the paper and if there is a plan to open source it? A good way to make it available during the review process is an anonymous git repo.


**Sources**

[1] Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining, Sow. et al., 2025

### Questions
Q1) How many training steps does the model need in order to be able to provide a sufficiently good signal for ESLM stage 1? I would think, there needs to be some kind of a warm up.

Q2) How would ESLM perform when using MoE models? I would assume the stage 1 proxy would not work the same way as it does for dense models.

### Soundness
3

### Presentation
3

### Contribution
2
