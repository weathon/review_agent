# CNSP: Consistent Null-Space Projection for Principled Prompt-Based Continual Learning

- Avg Score: 4.80
- Decision: Reject
- Scores: 4, 4, 4, 6, 6

## Abstract
Continual learning aims to acquire new knowledge sequentially without forgetting previous tasks, yet catastrophic forgetting remains a major challenge. Prompt-based continual learning has recently shown competitive empirical progress, yet its theoretical underpinnings remain incomplete. We introduce Consistent Null-Space Projection (CNSP), the first unified and mathematically rigorous framework for representational consistency in prompt-based continual learning. CNSP proves that task-performance preservation reduces to two jointly sufficient requirements—feature preservation and head preservation—while deriving explicit consistency conditions under full Transformer parameterization. These conditions yield a tractable null-space projection rule for stable prompt updates.
Across various benchmarks and backbones, CNSP demonstrates consistent improvements in accuracy and forgetting, with especially clear benefits in high-dimensional and domain-shift scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CNSP (Consistent Null-Space Projection), a principled framework for prompt-based continual learning with Vision Transformers. Building on NSP^2, the authors strengthen its theoretical foundations by providing per-head sufficient conditions for multi-head attention, a matrix-form treatment of LayerNorm, and a relaxed variance-only constraint for prompt updates. They also ensure end-to-end consistency by incorporating classification head preservation. Experiments show consistent gains in accuracy and reduced forgetting compared to prior methods, including NSP^2.

### Strengths
1. This paper provides an extensive analysis of existing null-space projection method in continual learning (especially NSP^2), and propose corresponding improvements.

2. The proposed method is derived from theoretical basis of feature preservation and head preservation, and seems reasonable.

### Weaknesses
1. From my understanding, this work is built on existing null-space projection method NSP^2, which limits its technical novelty.

2. The proposed designs, although very extensive, achieve only marginal improvements over NSP^2.

3. In Table 2 (ablation study), the authors seem to remove major components of null-space projection method. However, as the extension of NSP^2, each individual design should be compared with the counterpart in NSP^2.

4. The authors only consider ViT-B/16 pretrained on ImageNet-21K as the backbone. Does the proposed method also apply to other backbones especially self-supervised checkpoints?

### Questions
My major concerns lie in the effectiveness of individual designs over NSP^2. Please refer to the Weaknesses.

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
5

### Summary
This paper studies the problem of prompt-based continual learning. It analyzes the limitations of a regularization-based method, NSP2, and identifies several key weaknesses, including the lack of rigorous justification for its extension to multi-head attention, the oversimplified treatment of LayerNorm, the unstable invariance assumption on prompts, and the neglect of classification head analysis. To overcome these issues, the paper proposes Consistent Null-Space Projection, which introduces rigorous multi-head analysis, matrix-form LayerNorm modeling, a relaxed distributional constraint, and classification-head preservation to ensure theoretical consistency. Experimental results show that the proposed method consistently outperforms NSP2 in terms of both average accuracy and forgetting across multiple benchmarks.

### Strengths
1. This paper introduces the concepts of feature preservation and head preservation through theoretical analysis, identifying the key factors that maintain the stability of prompt-based continual learning models.

2. In the methodology section, a more rigorous derivation of multi-head attention is presented, leading to a set of sufficient conditions for feature preservation in VPT.

3. The analysis reformulates LayerNorm’s broadcast operations in matrix form, thereby ensuring greater algebraic rigor and theoretical soundness.

4. CNSP achieves performance competitive with state-of-the-art prompt-based continual learning methods, demonstrating both theoretical and empirical advantages.

### Weaknesses
1. Since each attention head operates independently, the per-head analysis conducted in this paper essentially follows the same procedure as in NSP2, except that NSP2 did not explicitly denote each head with the superscript (h). As a result, the derivation ultimately reduces to single-head-level results rather than a unified multi-head formulation. This head-level analysis does not provide substantial new insights.

2. The adoption of a right-side nullification form on the second constraint appears questionable. The purpose of the second condition in Eq. (27) is to ensure that the attention output (i.e., the product of the attention matrix and the value matrix V) remains invariant. However, by removing the attention matrix S from the formulation, this invariance is no longer guaranteed, weakening the theoretical justification for the constraint.

3. The claim of “classification head preservation by design” lacks meaningful novelty. This setup does not require any special design, as most continual learning methods already adopt the same practice (i.e., training task-specific heads and concatenating them at inference).

4. The experimental results are limited, consisting mainly of a single comparison with the current state-of-the-art and one ablation study (in the appendix). Moreover, as shown in Table 1, the improvements over NSP2 are marginal and not particularly significant.

### Questions
1. In Eq. (23), both gamma and beta are defined as element-wise parameters. However, in Eq. (24), after reformulating them into matrix form, they are expressed as if they participate in matrix multiplication. This seems inconsistent with their element-wise nature. The paper should provide a clear explanation for this formulation choice, as it could directly affect the validity of the subsequent derivations and conclusions.

2. Eq. (25) introduces the variance-invariance assumption, but in transitioning from Eq. (23) to (24), the mean term $\mu_{P_t+\Delta P}$ is omitted. It is unclear whether this omission is mathematically equivalent to the original formulation (Eq. 23). If not, then the assumption in Eq. (25), which constrains only the variance while ignoring the mean, may not be strictly valid, potentially weakening the theoretical rigor of the analysis.

### Soundness
2

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
This paper introduces Consistent Null-Space Projection (CNSP), a novel prompt-based continual learning method. Building upon NSP2, the method systematically analyzes multi-head attention and LayerNorm, thereby enhancing the theoretical foundation for knowledge retention through null-space projection. Extensive experimental evaluations demonstrate that CNSP achieves state-of-the-art performance across multiple benchmark datasets.

### Strengths
1.	The motivation behind this paper is both intuitive and clearly articulated. In contrast to previous works, CNSP provides a thorough investigation of multi-head attention, LayerNorm, and the classification head within the context of continual learning. Specifically, the paper introduces a more principled and effective framework for visual prompt tuning with vision transformers.
2.	This paper includes theoretical derivations and proofs.
3.	The paper is written in a clear and well-organized manner.

### Weaknesses
1.	Compared to NSP2, the paper provides a more systematic theoretical analysis of multi-head attention, LayerNorm, and the classification head. However, the experimental results show only modest improvements.
2.	The contribution appears to be somewhat limited and does not fully meet the conference's expectations. The method seems to be an instantiation of orthogonal projection techniques, applied to continual learning in image classification, and is based on visual prompt tuning with vision transformers.
3.	Maintaining a separate classification head for each task leads to an increase in the number of parameters as the number of tasks grows. Furthermore, calibrating the logits obtained from independently trained classification heads across different tasks presents a new challenge.
4.	There is a notation error in lines 248 and 654. The correct index should be A(2) instead of A(1).

### Questions
1.	Please clarify the core contribution of the work, rather than describing the differences in unexplored components from previous works, such as multi-head attention, LayerNorm, and the classification head.
2.	From the ablation study in Table 2, the variance preservation loss has minimal impact on the results. Does this suggest that the assumptions in Equation 25 do not significantly affect the derivation of the final constraints?

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
This paper introduces CNSP (Consistent Null-Space Projection), a theoretically rigorous framework for prompt-based continual learning in Vision Transformers (ViTs). The work addresses limitations in prior methods (e.g., NSP²) by proposing: (1) Per-head sufficient conditions for multi-head self-attention (MHSA) to ensure feature preservation; (2) Matrix-form LayerNorm modeling to replace scalar approximations; (3) Variance-only constraints for prompt updates (relaxing NSP²’s mean-variance invariance); (4) End-to-end task preservation via classification head constraints. Experiments on CIFAR-100, ImageNet-R, and DomainNet show CNSP consistently outperforms NSP².

### Strengths
The paper's theoretical foundation is rigorous and innovative in practice.

### Weaknesses
1.SVD on $RR^\top$ may incur latency for large $D$ (e.g., ViT-L/16). Training time/GPU memory vs. NSP² should be reported.
2.Softmax avoidance (Eq. 20) lacks theoretical justification. The impact of attention-score invariance (Eq. 20) on representational capacity needs analysis.
3.Classification heads grow linearly with tasks (Appendix E.1). For more tasks, parameter explosion may occur—compression techniques (e.g., prompting heads) should be discussed.
4.The pseudocode is too long, so it is recommended to put it in the appendix and put more valuable experiments in the main text.
5.CNSP treats all tasks equally, but catastrophic forgetting varies with task correlation (e.g., CIFAR-100 classes vs. ImageNet-R domains). Experiments use frozen ImageNet-21K ViT-B/16 for all tasks. This bypasses challenges like: Cross-domain pretraining gaps (e.g., medical vs. natural images). Model scaling effects (performance on ViT-L vs. ViT-B).
6.The experimental results are not convincing, and the gains against NSP^2 is trivial.

### Questions
See Section Weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Consistent Null-Space Projection (CNSP), a framework for prompt-based continual learning in Vision Transformers. Prior work NSP² derived sufficient conditions for preserving representations under prompt updates, but relied on simplifying assumptions regarding LayerNorm and multi-head self-attention (MHSA), and enforced a strong mean–variance constraint on prompts. This paper revisits the problem using a matrix-level formulation, deriving per-head feature preservation conditions and providing a matrix-form characterization of LayerNorm. The authors further show that variance preservation alone is sufficient, leading to a more stable optimization strategy. They implement prompt updates via right-side null-space projection, ensuring constraints hold during learning. Experiments on CIFAR-100, ImageNet-R, and DomainNet demonstrate consistent improvements over NSP² and competitive performance among prompt-based continual learning baselines.

### Strengths
- Addresses a clear theoretical limitation in NSP², especially in the treatment of LayerNorm and MHSA.
- Derivations are technically sound and detailed.
- Relaxing to variance-only preservation improves stability in practice.
- Null-space projection is computationally efficient and easy to implement.
- Consistent empirical gains across benchmarks.
- Ablation studies effectively demonstrate the necessity of key components.

### Weaknesses
- The novelty is largely a refinement of NSP² rather than a fundamentally new idea.
- Certain theoretical assumptions are not empirically analyzed (e.g., fixed γ and β in LayerNorm).
- Performance improvements, though consistent, are modest.
- Lack of comparison to strong non–prompt-based CL methods limits understanding of overall competitiveness.
- Training overhead of SVD-based projection is not reported.

### Questions
1. Does null-space projection restrict the expressive capacity of prompts as the number of tasks increases?
2. How sensitive is the variance-preservation alignment loss to domain shift between tasks?
3. What is the computational overhead of null-space projection compared to NSP² in practice?

### Soundness
3

### Presentation
3

### Contribution
3
