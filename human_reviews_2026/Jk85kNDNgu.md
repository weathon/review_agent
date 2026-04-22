# MuRA: Multi-Rank Adaptation for Efficient and Effective Test-Time Vision-Language Generalization

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities, but their performance degrades significantly when encountering distribution shifts. Recently, test-time adaptation (TTA) methods have been introduced to enhance VLMs' generalization ability. Among these methods, knowledge-adaptive approaches that incorporate Low-Rank Adaptation (LoRA) into vision models show relatively limited improvement compared to other TTA strategies. Our investigation reveals that the fundamental limitation stems from LoRA's static rank configuration, as visual inputs with varying information densities inherently require different ranks for optimal adaptation. To address this challenge, we propose Multi-Rank Adaptation (MuRA), a dynamic rank selection mechanism that adapts to varying data distributions. MuRA achieves state-of-the-art performance on domain generalization and cross-dataset benchmarks. By restricting adaptation to only the deepest layer, MuRA shortens the gradient backpropagation path, thereby significantly reducing both computational and memory overhead. Our method represents an efficient and effective approach to test-time vision-language generalization. Our code will be released as soon as possible.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper targets knowledge-adaptive TTA for VLMs and argues that a static LoRA rank is suboptimal because different inputs/datasets require different adaptation capacity. Empirically, the optimal rank varies widely across domains and correlates strongly with image entropy. The proposed MuRA prepares multiple rank-specific LoRA modules via Multi-Rank Orthogonal Decomposition (MROD) and soft, token-level routing (UCF) with Continuous Router Updating (CRU). MuRA delivers state-of-the-art average accuracy on ImageNet OOD and cross-domain suites with attractive accuracy–efficiency trade-offs.

### Strengths
- Clear diagnosis & evidence. The paper convincingly shows rank sensitivity across datasets and its linear relation to image entropy, motivating dynamic rank selection.

- Well-designed, cohesive method. MROD: principled SVD-based init yielding orthogonal residuals; improves stability of one-step TTA. UCF (token-level soft MoE) + CRU: learns token-wise rank preferences over time; soft routing > hard, token-level > instance-level (especially with CRU).

### Weaknesses
- Varying data distributions. Although the paper claims strength under varying data distributions, current experiments use a single test distribution per benchmark. Please evaluate sequential distribution shifts (CL-style streams) to show that CRU adapts appropriately over time and visualize how CRU’s rank routing evolves as the distribution changes. Framing the method explicitly as strong for CL + TTA would also sharpen the contribution.

- Academic formatting quality. The manuscript’s presentation needs polishing (e.g., oversized figures, occasional image blurring/pixelation, inconsistent layout). Please standardize figure sizes/resolution, ensure vector graphics where possible for professional readability.

### Questions
- Please visualize the evolution of CRU’s rank routing as the distribution changes (e.g., rank-utilization entropy over the stream, per-domain routing profiles) --- See Weakness 1.
- Please improve the paper formatting quality --- See Weakness 2.

### Soundness
2

### Presentation
1

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
The paper proposes MuRA, a test-time adaptation method for VLMs that fuses multiple LoRA modules of different ranks via a softmax-weighted router and initializes the rank components through an SVD-based “Multi-Rank Orthogonal Decomposition” (MROD). A “Continuous Router Updating” (CRU) strategy is claimed to retain routing knowledge across samples. The method adapts only the deepest visual layer and uses 63 augmentations per test image with entropy-weighted selection. Experiments report gains on ImageNet variants and cross-domain datasets.

### Strengths
1. Practical, lightweight design: restricting adaptation to the deepest layer is reasonable for efficiency and simplicity.
2. Competitive results on multiple benchmarks, with ablations that partially justify the components (MROD, UCF, CRU).
3. Clear framing of rank sensitivity and correlation with image entropy; the motivation for dynamic rank choice is intuitive.

### Weaknesses
1. Limited technical novelty: the core idea—combining multiple LoRA ranks with a softmax router—is a straightforward mixture-of-experts/gating over adapters and feels incremental relative to existing PEFT/TTA adapter ensembles.
2. SVD vs. random partition: the paper does not convincingly demonstrate why SVD-based rank construction is superior to simpler alternatives (e.g., random splits, fixed-rank LoRA, or PCA variants). Please add controlled comparisons (same parameter budget) and report effect sizes.
3. CRU under-specified: the Continuous Router Updating component (around Line 300) is described in one sentence without algorithmic detail. How exactly is the router state retained/reset across samples/batches? What regularization, optimizer, learning rate schedule, and stability safeguards are used? Provide pseudo-code and a failure/sensitivity analysis.
4. Efficiency concerns: generating 63 augmentations per test image can be costly. The paper should include a clear time and memory complexity analysis (asymptotics and wall-clock),  and throughput vs. accuracy trade-offs (e.g., 8/16/32/63 views). Report results with fewer views to show robustness.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MuRA, a test-time adaptation (TTA) method for CLIP that addresses the limitations of static rank configurations in prior knowledge adaptation paradigms. The core idea is that visual inputs with varying information density may require different adaptation capacities. To this end, the method dynamically selects and fuses multiple low-rank adaptation components to achieve efficient adaptation across diverse image types. Extensive experiments on 15 datasets demonstrate competitive performance.

### Strengths
- The paper is well-written and easy to understand, with clear explanations of the method and experimental results.
- The knowledge adaptation paradigm represents an interesting form of TTA.

### Weaknesses
- Intrinsic limitation of the knowledge adaptation paradigm.   The proposed method appears tightly coupled with a specific architecture and may not generalize well to other baselines such as CLIP with a ResNet-50 backbone.
- Risk of overconfidence from entropy minimization. The use of entropy minimization loss may cause over-confident predictions during test-time adaptation, which could negatively affect model calibration.
- The manuscript lacks comparisons with the state-of-the-art VLM TTA methods, such as MCP[1], GS-Bias[2], and TT-RAA[3].

[1] Multi-Cache enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models. ICCV 2025

[2] GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models. ICML 2025

[3] Test-Time Retrieval-Augmented Adaptation for Vision-Language Models. ICCV 2025

### Questions
- In Table 3, the performance improvement of Unified Component Fusion (UCF) after MROD appears marginal.   Does this suggest that the contribution of UCF is limited?
- What theoretical justification supports the design of the Continuous Router Updating (CRU) strategy? I am concerned that continuously updating the router across test samples may lead to the accumulation of adaptation errors or drift over time.
- The paper lacks visualization or interpretability studies (e.g., t-SNE feature visualizations) that could provide insights into how the proposed adaptation influences the representation space.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes MuRA (Multi-Rank Adaptation), a test-time adaptation (TTA) framework designed for Vision-Language Models (VLMs) such as CLIP. MuRA adapts both visual and textual embeddings across multiple ranks in a unified optimization objective, yielding more flexible and robust test-time updates. Experiments on benchmark datasets show improved accuracy and robustness compared to single-rank or fixed adapter baselines.

### Strengths
S1. **Comprehensive Experiments.** 
The paper adheres to the standard evaluation protocols established in the VLM TTA community and shows consistent performance gains over prior baselines across multiple benchmark datasets.

S2. **Balanced technical depth and clarity.**
The paper provides clear algorithmic exposition, ablation studies on rank configurations, and qualitative analyses supporting the motivation in Appendix.

### Weaknesses
W1. **Missing comparison against multiple same-rank adapters.**
While the idea of employing multiple ranks for adaptation is interesting, the paper does not convincingly show that the improvement comes from rank diversity itself rather than from simply using multiple adapters. The ablation study compares only single-rank vs. multi-rank configurations, but there is **no baseline that uses several adapters of the same rank. Without this comparison, it remains unclear whether MuRA’s advantage originates from heterogeneous rank composition or just increased model capacity. Including such an experiment would make the contribution much more convincing.

W2. **Unfair comparison due to unmatched adaptation capacity.** 
Several reported gains may stem from larger trainable capacity rather than the proposed multi-rank design. In many tables (Table 1&2), MuRA appears to use more total trainable parameters than baselines that use a single adapter of a fixed rank or adapt only one branch. Without capacity-controlled baselines, the comparison is confounded. To ensure the fairness, the paper should report the total trainable parameters, further match total trainable parameters. 

W3. **Inference Overhead.** 
Unlike prior single-LoRA approaches, MuRA introduces a gating module that determines which LoRA branch to activate during inference.
In practice, this requires computing the forward pass of both the base model and a LoRA module selected by gating module, followed by their weighted summation. This design inevitably incurs additional inference overhead compared to a single LoRA model, where the LoRA weights can be merged into the base weights, resulting in identical forward cost. Therefore, the authors should report and analyze the inference-time computational cost of MuRA, including latency, FLOPs, and throughput, to clarify the trade-off between performance gain and efficiency.

W4. **Incremental performance gains.**
As shown in Tables 2–4, the proposed method achieves approximately 1.0–2.5% improvement over single-rank baselines. While the gains are consistent, they are relatively modest given the additional complexity introduced by the multi-rank design. Considering that MuRA requires multiple adapters and a gating mechanism at inference, the improvement-to-overhead ratio appears limited. A more detailed efficiency analysis or scenarios where MuRA provides significantly larger benefits (e.g., under extreme domain shifts) would help justify the practical value of the approach.

### Questions
I wrote all my concerns in Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
