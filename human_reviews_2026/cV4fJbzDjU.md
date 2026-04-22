# Differentiable Entropy Regularization: A Complexity-Aware Approach for Neural Optimization

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
We introduce the first differentiable approximation of range-partition entropy, a complexity measure from computational geometry that directly bounds algorithmic runtime. Unlike architectural modifications, our method is a complementary regularizer that provides orthogonal efficiency gains when combined with existing optimizations. We establish theoretical guarantees in computational geometry, achieving 4--5$\times$ provable speedups on convex hull and triangulation with $<$0.2\% error. On ImageNet-1K with ViT-Base, entropy regularization achieves 80.1\% top-1 accuracy at 80\% sparsity (1.60$\times$ standalone speedup), and when combined with FlashAttention yields 2.07$\times$ speedup versus 1.63$\times$ for FlashAttention alone. On large language models (LLaMA-2 7B, Mistral-7B, Phi-2), we achieve 1.48--1.60$\times$ inference speedups at 70--75\% sparsity with minimal quality degradation (ROUGE-L drops of 0.3--0.4 points, perplexity increase of 0.9). Unlike prior regularization methods that target output distributions, we directly minimize representation complexity, yielding both efficiency gains and improved robustness through semantically structured sparsity patterns (IoU 0.73 vs 0.41 for magnitude pruning, CIFAR-100-C mCE 48.7 vs 55.4). Benefits are strongest for geometry and vision transformers, with more modest but measurable gains on LLMs, demonstrating that complexity regularization offers a principled pathway to joint efficiency-robustness optimization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper suggests an entropy regularization factor added to the task loss. The goal is to encourage lower complexity representations in models, and thus improve robustness while reducing computational cost. The authors focus on the idea that both of these issues can be addressed via analyzing "spurious" features, and propose a smooth differentiable surrogate technique to push the model towards learning simple representations (which are more easily partitioned into clusters). The central contribution is the introduction of the regularizer term, with supported theoretical analysis. Two of the main datasets being highlighted are both size 32x32x3 (per sample), with best results stemming from the computational geometry domain.

### Strengths
- The idea of analyzing learning representations through the lens of entropy is valuable and might encourage interpretability (although this was not directly discussed in the paper.). A differentiable regularizer based on range-partition is pleasantly theoretically grounded. Indeed, I find the theoretical contributions of this work to be the most interesting and significant. The experiments with the geometry tasks are the most compelling. 

- The method demonstrates valid empirical improvements on CIFAR-100-C and SVHN datasets, which support the fundamental idea that analyzing sparsity patterns in learning representations can lead to benefits. 

- The proposed method is practically complementary with most other methods, and stacks on top of prior work while being GPU-agnostic. This might be valuable in resource constrained settings.

### Weaknesses
- The main contribution is a regularizer penalty term added to a loss function, in order to encourage learning simpler representations. This is not an incredibly novel idea, and entropy based penalties to improve robustness is not new ([1], [2], [3] are a few examples of many). 

- The main experiments being highlighted are small scale, with CIFAR-100-C and SVHN (similar task to MNIST) both containing samples sizes of 32x32x3. The community is pushing towards scale and speed. Experiments on trivially small datasets are not compelling. This method likely will have difficulty scaling to high dimensions, which might be why larger experiments were not included.  Especially since the authors transparently acknowledge this method will slow down training time.

- Since the codebase is not provided, it is questionable if these experiments can be re-produced. A large number of factors contribute to performance, besides just implementing the correct loss function. The authors seem to compare metrics between experiments ran on Intel CPUs, to a single NVIDIA A100 GPU, to distributed settings with multiple GPUS and DeepSpeed (specific parameters not listed). These are not 1:1 comparisons. 

- Some slight typing and formatting issues are present, but not a huge concern. For example, there is a formatting error on line 357.


[1] Huang C, Lu W, Zhang W. PEAR: Phase Entropy Aware Reward for Efficient Reasoning. arXiv preprint arXiv:2510.08026. 2025 Oct 9.

[2] Fan, Feng-Lei, et al. "On interpretability of artificial neural networks: A survey." IEEE Transactions on Radiation and Plasma Medical Sciences 5.6 (2021): 741-760.

[3] Pfrommer, Samuel Ian. "Safety, Robustness, and Interpretability in Machine Learning." PhD diss., University of California, Berkeley, 2025.

### Questions
- The paper references "Transformers" as a model throughout. Which transformer based model is this referencing? I found this to be unclear.
- The authors are transparent about the significant training overhead their method introduces. This is a critical disadvantage, have the authors analyzed cost-benefit analysis? If the reward is increased robustness, at the cost of lengthier training time, can you demonstrate/quantize this on a currently relevant dataset? 

- The method introduces new hyperparameters and will likely struggle in high-dimension problems. Have you performed ablation studies on hyperparameter sensitivity and attempted high-dimensional data? 

- Once again, if the main advantage in this proposed method is increased robustness, have the authors conducted more meaningful experiments beyond "label smoothing"? How does this compare to SOTA methods like adversarial training in enhancing robustness?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a complexity-aware regularizer grounded in algorithmic entropy, designed to encourage models to learn simpler and more robust representations under corruption and distribution shifts. The authors report 1.5–2× speedups without accuracy loss when integrating the proposed regularizer into architectures such as FlashAttention and RetNet.

### Strengths
The topic is relevant and well-studied–improving robustness to corruption and distribution shifts.

### Weaknesses
**Major Concerns**:

1. **Unclear motivation**: The motivation is weak and poorly articulated. The paper primarily surveys prior work without convincingly identifying unresolved gaps or specific limitations. It remains unclear what core problems the authors aim to solve, why they matter, and why existing approaches are insufficient.

2. **Questionable technical grounding**: The proposed regularizer is claimed to be based on range-partition entropy, but the paper rarely provides background on this concept, which is not widely recognized. As a result, it is difficult to assess the soundness or novelty of the theoretical formulation. Besides, the rationale for the computational efficiency is also not well explained beyond empirical observations.

3. **Limited originality**: The contribution appears incremental relative to existing regularization-based methods for robustness. The paper does not offer substantial new insights into why this particular regularizer is preferable or theoretically justified compared to other forms of penalties.

4. **Poor writing and organization**: The paper’s presentation lacks clarity. The authors frequently refer to the appendix instead of offering an intuitive overview or proof sketch of their theoretical results. The experiments are fragmented into two sections for no reason and not organized into well-structured subsections, making it difficult to follow the setups and findings.

5. **Weak empirical validation**: Experimental evidence does not convincingly support the claimed advantages. Table 2 lacks comparisons with other regularization-based robustness methods, which is essential for contextualizing the proposed approach.

**Minor Issues**:

1. Several terms are vague or under-defined, reducing readability and precision. Examples include but are not limited to “data characteristics,” “complexity of learned representations,” “instance complexity,” and “separator-driven procedures.”

2. The paper lacks a discussion of its limitations.

### Questions
1. The purpose and interpretation of Figure 1 are unclear. What comparisons are being shown? In particular, how does Figure 1(c) substantiate the claim that the method “discovers patterns that align with algorithmic efficiency”? Without detailed textual descriptions, it’s hard to associate the captions with your method.

2. Why investigate computational geometry experiment, a domain rarely explored in mainstream machine learning community, especially regarding the robustness topic? How does this task demonstrate the generality or relevance of your method? If the approach only performs well in such specialized settings, its broader applicability should be justified.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Differentiable Entropy Regularization, a novel regularizer that encourages neural networks to learn simpler, more structured representations by minimizing a differentiable surrogate of algorithmic entropy.

The surrogate measures representation complexity via soft partitions and is trained jointly with the task loss.

Experiments show that DER improves both robustness and efficiency.

### Strengths
1. Introduces the first differentiable surrogate for algorithmic entropy, providing robustness and efficiency.
2. Complements existing efficiency methods (FlashAttention, RetNet) and yields interpretable, robust representations.
3. Presents provable bounds, runtime guarantees.

### Weaknesses
1. While geometric tasks align with theory, the improvements on ViT/BERT/GPT are empirical. There is no clear theoretical connection between range-partition entropy and the dynamics of attention mechanisms.
2. Performance may depend on careful choice of hyperparameters. It would be better to have automatic or theoretically grounded tuning methods.
3. It would be good to compare with other information-theoretic regularizers.

### Questions
1. Could DER be applied to reinforcement learning or diffusion models?
2. How sensitive is the method to anchor initialization?
3. How well does the surrogate behave for non-Euclidean embeddings

### Soundness
3

### Presentation
2

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
The paper proposes a differentiable entropy-based regularizer that encourages neural networks to learn simpler, lower-complexity representations. Inspired by algorithmic entropy (or range-partition entropy from computational geometry), the method directly penalizes representation complexity to improve robustness, efficiency, and interpretability.

### Strengths
- The paper proposes a differentiable surrogate for algorithmic entropy which provide smooth, differentiable approximations to range-partition entropy, allowing gradient-based optimization.

- It provides data-dependent bounds connecting the differentiable surrogate to true entropy.

- It shows how minimizing surrogate entropy correlates with improved runtime efficiency in geometric algorithms.

- Empirically, it improves robustness on CIFAR-100-C and SVHN OOD and 1.47–2.07× inference speedups with minor accuracy loss on transformers.

- Entropy regularization combines effectively with FlashAttention v2 and RetNet, yielding compounded efficiency gains.

### Weaknesses
- It adds 2–12% computational cost during training; and amortization is only beneficial for long-term or production-scale inference (≥450K batches for ViT-Base).

- It requires careful tuning of temperature ($\tau, \alpha$) and regularization weight ($\lambda$) for stability and performance.

- Guarantees are strongest in geometric settings; results for Transformers and LLMs are mostly empirical, with weaker formal grounding. While I don't expect the authors to provide guarantees for different type of models, empirically it seems that the proposed technique is not as beneficial for larger foundation models e.g. LLMs and VLMs.

- For example, while complementary, it doesn’t consistently surpass the absolute state-of-the-art standalone (e.g., FlashAttention still faster alone in some cases).

- In large Transformers or ViT-Base scale models, computing soft assignments over many tokens is heavy. The effect grows with sequence length or feature dimension (since distance computations dominate). While FAISS or subsampling helps, this can still become the bottleneck for larger foundation models.

- In terms of the writeup, I had to check the literature to see what's done before. This should be already reflected in intro and related work, and I didn't find enough discussion there. Also, in terms of scope the papers talks about "modern deep models". Although I understand that the proposal is complementary to FlashAttention etc, it seems to me that the method is mostly beneficial to vision models, not really LLMs/VLMs. If so, the authors can clarify the scope better in the abs/intro.

### Questions
Can the authors discuss the applicability of the proposed methods to large foundation models such as LLMs and VLMs? I see the experiment on Llama2, but the benefit is not really significant there. Do you think the method benefits even larger models or the gains will be even smaller there?

### Soundness
3

### Presentation
3

### Contribution
3
