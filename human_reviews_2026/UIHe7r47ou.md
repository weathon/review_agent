# Multi-Sample Preference Optimization for Generative Model Alignment

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Recent advancements in generative models, particularly large language models (LLMs) and diffusion models, have been driven by extensive pre-training on large datasets followed by post-training. However, current post-training methods such as reinforcement learning from human feedback (RLHF) and direct alignment from preference methods (DAP) primarily utilize single-sample comparisons. These approaches often fail to capture critical characteristics such as generative diversity and bias, which are more accurately assessed through multiple samples. To address these limitations, we introduce a novel approach that extends post-training to include multi-sample comparisons. To achieve this, we propose Multi-sample Direct Preference Optimization (mDPO) and Multi-sample Identity Preference Optimization (mIPO). These methods improve traditional DAP methods by focusing on group-wise characteristics. Empirically, we demonstrate that multi-sample comparison is more effective in optimizing collective characteristics~(e.g., diversity and bias) for generative models than single-sample comparison. Additionally, our findings suggest that multi-sample comparisons provide a more robust optimization framework, particularly for dataset with label noise.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
RLHF and DPO are popular post-training methodologies for LLM alignment. The standard method uses preference pairs consisting of single samples to align LLMs. This increases the probability of generating preferred samples when the reward is computed at a per-sample level. However, in several cases like increasing the diversity of responses etc. the rewards cannot be computed at a per-sample level.

This paper extend DPO and IPO to such cases. In this work, the authors generate a set of responses for each prompt. Analogous to preferred and dis-preferred samples, the multi-sample formulation has a preferred and dis-preferred set. The authors provide an unbiased estimator for the multi-sample formulation of IPO through theoretical analysis. Through five empirical studies, they show that the multi-sample DPO and IPO performs better than DPO and IPO when the reward is computed at a distribution level.

### Strengths
1. Computing rewards at a sample level is an important limitation of DPO and IPO. The extension of standard preference optimization framework to distributional level rewards addresses this limitation
2. The theoretical analysis which shows that the variance of the multi-sample estimator decreases with group size is novel
3. The range of empirical studies show that the multi-sample DPO/IPO improves upon the performance of standard DPO/IPO when the rewards are formulated at a distribution level across a range of modalities.

### Weaknesses
1. The contribution of this work seems limited. The primary difference between the standard DPO setting and the multi-sample DPO setting is in the way that the samples are separated into preferred and unpreferred groups. Instead of separating them at a sample level, they are first grouped into sets and separated at a set level. Given that the reward is computed at a distributional level, this seems like the natural application of DPO for such a problem.
2. There seems to be a strong overlap between this paper and Li et al [2024]. The authors have not clearly stated the original contributions which differ from Lit et al [lines 118 - 124]
3. This work could benefit from stronger baselines - extending the RLHF framework to distributional reward problems. Furthermore, I would encourage the authors to compare this results with Zhong et al [2023] and Melnky et al [2024].

### Questions
Please see the weaknesses

### Soundness
3

### Presentation
4

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
This paper introduces **multi-sample extensions of preference optimization methods (mDPO and mIPO)** for alignment. Whereas standard approaches such as RLHF and DPO/IPO rely on **single-sample pairwise comparisons**, this work proposes to instead optimize over distributions of responses. This allows the methods to capture distributional characteristics such as **diversity and bias**, and to better handle cases where preferences may not exist between individual samples but emerge clearly when comparing groups. The authors highlight challenges in unbiased estimation and provide empirical evidence that the proposed approaches can improve robustness against label noise and enhance diversity in generated outputs.

### Strengths
- Tackles an **important and underexplored problem**: extending preference optimization from single-sample to multi-sample comparisons.  
- Provides a **novel formulation** (mDPO, mIPO) that enables aligning distributions rather than instances.  
- Shows **promising empirical results**, especially in improving **diversity** and reducing **bias** in outputs.  
- Highlights the robustness of mDPO against label noise, which is valuable in real-world preference datasets.  
- Overall **presentation** is good, with several intuitive illustrations of the advantages of multi-sample formulations.

### Weaknesses
While I liked the paper overall and I believe it tackles an important problem, some key weaknesses should be addressed:
1. **Insufficient experimental comparisons**:  
  - The paper does not compare against **other multi-sample methods** such as DFT (Guo et al., 2025).  
  - No experiments with **naive multi-sample baselines**, e.g., running DPO/IPO over all pairwise comparisons between positive and negative sets in the mini-batch, i.e. 
    $$
    \frac{1}{k^2}\sum_{i=1}^k\sum_{j=1}^k l(x, y_{w,i}, y_{l,j})
    $$
2. **Overstatements in claims**: page 9 line 449, “both mDPO and mIPO significantly outperform the baselines” is too strong looking at the figure for mIPO $k=5$. This should be further argued or weakened.
3. **Experimental detail gaps**:  
  - The paper acknowledges that obtaining an unbiased estimator of mDPO is challenging, but still reports experiments with mDPO. It is unclear whether an estimator or a biased version is used.
  - Figure 5 and Table 3: why does mIPO with $k=3$ outperform $k=5$? One would expect larger $k$ to monotonically improve performance (even if with diminishing returns).  
  - Figure 6: it is not clear whether $k=5$ is an outlier or performance saturates? $k=6$ would help clarify this.
  - Iterative improvement experiments (page 9): baseline not clearly stated, should be iterative DPO/IPO for fairness.

Guo et. al. "Discriminative Finetuning of Generative Large Language Models without Reward Models and Human Preference Data." 2025. arXiv:2502.18679.


**Minor issues that have not affected the rating**
- Page 3 line 121: “foci” → “focus”.  
- Page 7 line 341: Figure 5 caption should have more space from text.

### Questions
- See the weakness section for suggested additional experiments; more baseline comparison would greatly benefit the paper.
- See the weakness section for some discussion and experimental details that the paper would benefit from answering.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper extends the direct preference alignment frameworks DPO and IPO to deal with preferences over groups of items over binary preference over pairs of items. The authors provide intuitive modifications to the DPO/IPO objective to incorporate group-wise comparisons, derive mini-batch estimators for these objective function, and show the validity of the method with various experiments.

### Strengths
The problem of needing to compare groups of items instead of individual items in some cases is natural and of practical relevance. The experiment case studies (random numbers generation, image debiasing, improving quality of creative fiction generation and training with label noise) are all quite interesting and both validates the approach and gives real-world examples where one might want to compare distributions instead of pairs of items. The paper is also well written and easy to follow.

### Weaknesses
1. The proposed methodology is quite straight-forward and the novelty of the proposed solution is moderate.

2. It seems like the objective estimates can have bias/variance and it seems like this would depend on the batch size. However, from the experiments I don't see this angle being explored sufficiently. For someone trying to deploy this method, how would they deal with the bias/variance issue, how does that change with the batch size?

### Questions
See weakness

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes multi-sample variants of direct preference optimization methods (called mDPO and mIPO) that replace single response comparisons with group-wise comparisons aimed at aligning distributional properties (e.g., diversity, bias). In particular, the authors lift DPO/IPO from singletons to sets by using the (geometric-mean) product likelihood of a response group and optimizing the same surrogate with expectations over group samples. Then, they derived an unbiased mini-batch estimator for mIPO (a squared-loss objective over mean implicit rewards) and discussed a biased but lower-variance estimator for mDPO. Finally, they added NLL as an auxiliary term to stabilize finetuning. Experiments cover LLM random number generation, creative fiction, and diffusion debiasing, plus a synthetic-label robustness study where multi-sample wins more often under label noise.

### Strengths
- The paper argues that many properties, such as diversity, are distributional and not captured by single-sample comparisons, and the group-wise formulation is intuitive.
- Extending DPO/IPO by averaging implicit rewards over sets keeps training compatible with existing implementations.

### Weaknesses
- Technically, mDPO/mIPO mostly reuse the same surrogates with a group-average of implicit rewards and a straightforward mini-batch estimator. The constrained-optimization view for adding NLL is already common in practice. Relative to DPO/IPO, the step from singletons to sets reads as expected algebra rather than a new learning principle. 
- There is prior work on distributional difference/alignment that directly targets set-level objectives. The paper cites some of these, but the differences are mainly about different experiments and applications, which are vague to me.

### Questions
- Can you provide a theoretical justification that mDPO/mIPO are proper surrogates for a target distributional objective? Can we provide any consistency under a Bradley–Terry-style group model?

- How does mIPO compare to work in distributional preference alignment in principle (what objective is optimized)?

- Is there any specific reason behind considering the geometric mean for the aggregation over a group?

### Soundness
3

### Presentation
3

### Contribution
2
