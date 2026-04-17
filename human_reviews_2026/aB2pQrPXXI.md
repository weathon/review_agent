# Learning Compact Vision Tokens for Large Multimodal Models

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large multimodal models (LMMs) suffer significant computational challenges due to the high cost of Large Language Models (LLMs) and the quadratic complexity of processing long vision token sequences. In this paper, we explore the spatial redundancy among vision tokens and shorten the length of vision token sequences for inference acceleration. Specifically, we propose a Spatial Token Fusion (STF) method to learn compact vision tokens for short vision token sequence, where spatial-adjacent tokens are fused into one. Meanwhile, weight-frozen vision encoder can not well adapt to the demand of extensive downstream vision-language tasks. To this end, we further introduce a Multi-Block Token Fusion (MBTF) module to supplement multi-granularity features for the reduced token sequence. Overall, we combine STF and MBTF module to balance token reduction and information preservation, thereby improving inference efficiency without sacrificing multimodal reasoning capabilities. Experimental results demonstrate that our method based on LLaVA-v1.5 achieves comparable or even superior performance to the baseline on 8 popular vision-language benchmarks with only 25% vision tokens of baseline. The source code and trained weights will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method for accelerating large multimodal models (LMMs) by substantially reducing the number of vision tokens supplied to the language model, while striving to maintain or enhance multimodal performance. The core contribution is a dual-stage token compression pipeline: (1) a Multi-Block Token Fusion (MBTF) module aggregates multi-level features from selected vision encoder layers, and (2) a Spatial Token Fusion (STF) module fuses spatially adjacent vision tokens into compact representations via learnable convolutions. Experiments on the LLaVA-1.5-7B backbone across eight popular vision-language benchmarks validate the approach, showing that with only 25% of the original vision tokens, the proposed method achieves comparable or improved performance versus the full-token baseline and several state-of-the-art efficient LMMs.

### Strengths
1. Clear Identification of Redundancy: The motivation—identifying excessive spatial redundancy in standard vision token sequences for LMMs—is empirically substantiated in Figure 1 and corresponding baseline experiments, which demonstrate minor performance losses or even gains when aggressively pooling tokens.

2. Principled Dual-Stage Compression: The architecture integrates MBTF for multi-granularity feature fusion, followed by STF for spatially adaptive reduction, as clearly illustrated in Figure 2 and specified in detail with Table 1. This strategy is logical and well-justified by the observed redundancy in vision tokens.

### Weaknesses
## **1. Missing Coverage of Directly Related Recent Work**
A substantial body of recent, directly relevant work on token compression, adaptive visual tokenization, and information steering in multimodal models is omitted from both the related work (Section 2) and experimental comparisons:
Notable works missing include [Cheng et al., 2024] (survey), [Zhang et al., 2024] (MMTok, SEED, Multi-Stage Vision Token Dropping, ElasticTok), [Zhang et al., 2024] (VoCo-LLaMA), [Li et al., 2025] (Hidden Life of Tokens), etc.
The discussion focuses primarily on token pruning and merging from existing efficient LLaVA variants, but does not adequately position this method with respect to the broader, very recent multimodal token compression literature (see Potentially Missing Related Work section below for specific papers and integration suggestions). This oversight undermines both the novelty claims and the work's perceived significance and must be addressed for publication.

## **2. Limited Theoretical Justification for "Lossless" Claim**
The paper repeatedly claims "lossless" token reduction when $k=2$, based on token/channel dimension matching (Section 3.4). However, the rationale lacks rigorous theoretical support or formal information-theoretic analysis. The claim is based on the matching of hidden sizes, rather than provable metrics regarding preserved semantic content. For instance, Eqn (in Section 3.4) specifies the process, but does not analyze information bottleneck, nor the risk of over-/underfitting due to arbitrarily increasing channels. This may mislead readers regarding the true trade-off between compactness and fidelity.

## **3. Empirical Evidence not Comprehensive Enough**
While Table 2 presents an extensive comparison on eight benchmarks, direct experimental contrasts with many distinct token reduction strategies (e.g., methods selected from the missing related work above) are absent. The ablations (Tables 3–5) primarily compare in-house baselines (MBTF, STF, simple pooling), while more rigorous comparison to recent learning-free or adaptive tokenization methods would strengthen the empirical evidence. In the absence of such, it remains unclear if the proposed approach outperforms the current state-of-the-art across scenarios.

## **4. Component and Parameter Clarity Limitations**
The mathematical specification and parameterization of the fusion kernels and projection modules are described, but details on their initialization, regularization, and design choices (e.g., GeLU activations, precise effect of kernel stride/sizes in practice) are somewhat sparse. For instance, in Table 1 and Section 3.4, a rationale for channel size progression, and potential overfitting in larger-k models (mentioned in Section 4.3.2) is offered, but more systematic analysis or regularization ablation would improve transparency.

## **5. Unclear Generalization Across Modalities/Backbones**
The method is tested only on LLaVA-1.5-7B (with CLIP-ViT and Vicuna-1.5-7B). There is no demonstration of cross-backbone generality or performance on models using different visual architectures or non-Vicuna LLMs—limiting the generalizability of the claims. Additionally, experiments mostly cover high-level vision-language reasoning and VQA; more fine-grained domains (e.g., visual grounding or image captioning) are not evaluated, which would clarify the trade-off between compression and task specificity.

## **6. Ablation Study Depth**
The ablations, while illuminating, would be stronger if (i) out-of-distribution robustness (e.g., to images substantially different from training data) were reported, (ii) FLOP/token reductions were systematically related to accuracy for a broader set of $k/E$ values, and (iii) an explicit analysis of failure cases under aggressive token compression was included.

## **7. Ambiguity in Performance Claims for STF vs. MBTF**
Table 3 suggests MBTF alone slightly outperforms the combined STF+MBTF approach in outright accuracy, yet this is brushed aside as a trade-off for efficiency. More nuanced discussion of why the combination does not enhance average accuracy, and scenarios where one should be prioritized over the other, would enhance scientific rigor.

## **8. Minor Typographical and Expositional Issues**
There are instances of unclear phrasing and notation, e.g., "stride of 2, and only $25 %$ tokens are remained" (Page 1), or ambiguous variable definitions in some formulae—these do not prohibit comprehension, but do reduce polish.

## **9. Assumption of Vision Encoder Freezing**
The method relies on a frozen vision encoder throughout training (Section 3.5, 4.1.3). It is unclear whether similar efficiency/performance gains would be possible if joint tuning with visual features occurred. This assumption may limit applicability in scenarios where full model finetuning is desired.

## **10. Code release**
The authors claim that they will release the code in the future. However, I think the code is available to reviewers during this stage, and authors are suggested to upload the source code as the supplementary material.

### Questions
Please see the weaknesses

### Soundness
2

### Presentation
2

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
The paper proposes a two-stage projector for large multimodal models (LMMs) that reduces the number of vision tokens fed to the large language model (LLM) while preserving task performance. First, a Multi-Block Token Fusion (MBTF) module concatenates intermediate features from evenly sampled visual blocks and compresses them with convolutions. Second, a Spatial Token Fusion (STF) module applies another convolution that fuses adjacent spatial tokens, followed by alignment to the LLM embedding space through convolutions. Experiments show that the proposed approach matches or exceeds some prior baselines on the LLaVA-1.5 model under 1.9 TFLOPs.

### Strengths
* The paper is clearly written and easy to follow.

* Thorough ablation studies are provided, including fusion modules, kernel sizes, and fusion strategies, as well as qualitative case studies, which help readers gain a better understanding of how the proposed approach functions and contributes to overall performance.

### Weaknesses
* End-to-end latency metrics are missing. Since lower TFLOPs don’t necessarily lead to significantly faster wall-clock speed, I’d suggest that the authors report latency to better assess efficiency improvements.

* It seems all experiments are conducted on LLaVA-1.5-7B with only one size and one input resolution. It is important to verify the effectiveness of the proposed method in more diverse settings.

* The additional cost introduced by the fine-tuning stage is not clearly discussed. The paper indicates that both the LLM and the projector are fine-tuned, which may involve notable computational expense.

* In terms of token reduction on LLaVA, some related work, such as [1], is not discussed in the paper. Clarifying how the proposed method relates to or differs from this prior line of work would be helpful for readers. [1] CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers by Shi et al. In ICML24.

### Questions
Please refer to the Weaknesses Section.

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
5

### Summary
The paper try to tackle LMM inference cost by compressing vision tokens. It introduces Spatial Token Fusion (STF) to merge spatially adjacent tokens and Multi-Block Token Fusion (MBTF) to re-inject multi-granularity features after reduction. Built on LLaVA-1.5, the method preserves accuracy on 8 VL benchmarks while using ~25% of baseline vision tokens, yielding speedups. Code and weights are promised for release.

### Strengths
1. The research question is interesting, and the scenario has clear practical relevance.

2. Across multiple benchmarks, the LLaVA-1.5–based approach delivers competitive performance.

### Weaknesses
1. **On Generalization vs. OCR Tasks**

   Papers [1] and [2] have shown that in many scenarios, even simple resizing techniques yield strong performance—consistent with this paper’s claim that AvgPool works well. However, both also highlight that on OCR-related tasks, aggressive token reduction often leads to notable performance degradation. I therefore recommend adding evaluations on OCR-specific benchmarks such as *OCRBench* and *ChartQA* to better assess robustness in those contexts.

2. **Applicability to Other VLMs**

   Can the proposed method be directly applied to models like *Qwen2.5-VL* [3] or *LLaVA-NeXT* [4]? Clarifying compatibility or adaptation requirements would strengthen the paper’s practical relevance.

3. **Outdated Baselines**

   The choice of baselines appears limited. Since the method is training-based, I suggest comparisons with *Q-Former* [5], a widely used architecture. It would also be helpful to include recent training-free compression approaches such as *VisionZip* [6] and *SparseVLM* [7], as the current baseline set appears somewhat outdated.

[1] VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning 

[2] Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods

[3] Qwen2.5-VL Technical Report 

[4] LLaVA-NeXT: Improved reasoning, OCR, and world knowledge 

[5] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models 

[6] VisionZip: Longer is Better but Not Necessary in Vision Language Models 

[7] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

### Questions
See weaknesses

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
5

### Summary
This paper studies the efficient large multi-modal models (LMMs). The author considers reducing the number of vision tokens fed into LLM and proposes the Spatial Token Fusion (STF) and Multi-Layer Token Fusion (MLTF) module to fuse adjacent vision tokens and capture multi-granularity visual features to obtain compact vision tokens. Extensive experiments have been conducted in standard vision-language datasets, where the proposed methods outperform the existing efficient LMM methods.

### Strengths
1. The proposed methods are reasonable and relevant for efficient LMMs. The motivation and design are clear and easy to understand.

2. The proposed method can effectively reduce the visual tokens while achieving comparable results to other methods on average. The ablation study also justifies the effectiveness of the proposed modules.

### Weaknesses
1. The proposed method's improvement is not significant enough. In Table 2, the MME metrics of the proposed method (STC) are significantly worse than those of FastV, but there is no elaboration on the performance loss. This makes it suspicious for the reviewer to be convinced that the proposed method can achieve the state-of-the-art results.

2. More vision-language tasks should be considered. The reduction of the vision token will inevitably lead to information loss on visual content, which may result in a larger performance drop on tasks that heavily rely on visual content, e.g., referral object detection. The author should consider adding a comparison of those tasks to demonstrate whether the proposed method remains effective.

3. Missing ablation study. The author mentioned that "directly fusing of all these features can not obviously improve the performance, but significantly increases the computation cost", while the claim has not been verified in the analysis section. The author should add this comparison to justify the claim.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2
