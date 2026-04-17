# FourierRoFormer: Learned Fourier Attention for Vision Transformers

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Vision Transformers (ViTs) excel at long-range reasoning but lack principled mechanisms for modeling spatial frequencies and controlling how attention decays with distance. We propose FourierRoFormer, a frequency-aware Transformer that augments rotary positional embeddings with learnable Fourier components. This enables explicit modeling of multi-scale visual patterns and adaptive distance-dependent modulation of attention. Our analysis shows that FourierRoFormer produces attention hierarchies aligned with object boundaries (correlation $r=0.85$) and distinct specialization across attention heads. On ImageNet-1K, FourierRoFormer achieves 84.1\% top-1 accuracy (+1.8pp over RoFormer-B) and outperforms non-hierarchical spectral methods, including SpectFormer-B (+1.98pp) and GFNet-B (+3.4pp), while maintaining comparable parameter efficiency. Our hierarchcial variant, FourierRoFormer-H-B, achieves 85.3\% top-1 accuracy, demonstrating compatibility with hierarchical architectures. The method improves transfer to dense prediction tasks, yielding +2.6 mAP on COCO detection and +2.2 mAP on instance segmentation. Ablation studies highlight the complementary roles of frequency modulation (+4.43pp) and adaptive damping (+2.09pp). The approach introduces only 0.04\% additional parameters and $\sim3\%$ computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel vision transformer architecture, FourierRoFormer, which extends rotary positional embeddings (RoPE) with learnable Fourier modulation and an additional exponential damping term. The key idea is to allow the model to learn optimal spatial frequency components—amplitudes, frequencies, and phases—via a differentiable Fourier modulation function, while the damping term enforces spatial locality by exponentially attenuating distant token interactions. The authors apply this post-attention frequency reweighting mechanism to standard ViT and RoFormer architectures, achieving consistent improvements across several image understanding benchmarks. Comprehensive analyses, including theoretical proofs (boundedness, convergence, interpretability) and ablations, support the proposed approach.

### Strengths
S1.  In-depth analytical insights: The paper provides detailed theoretical and empirical analyses. In particular, the multi-head frequency specialization analysis (L302–370) clearly shows how different attention heads specialize to distinct spatial frequency bands, providing interpretability and supporting the authors’ theoretical claims.

S2. Diverse Evaluation Domains: The proposed method is evaluated on several tasks and benchmarks.

### Weaknesses
W1. Inferior Performance to Existing Spectral Methods:
In Table 1, the proposed FourierRoFormer underperforms compared to recent spectral transformer models such as WaveViT-B and SpectFormer-H-B, despite having more parameters and FLOPs. Although Table 7 provides an “efficiency score,” this metric appears somewhat subjective, and it is unclear whether the proposed model demonstrates a statistically significant improvement in efficiency over baselines.

W2. Limited Backbone Generalization:
All experiments are conducted on standard ViT-style backbones. It remains unclear whether the proposed modulation generalizes to hierarchical or multi-scale architectures (e.g., Swin Transformer). Demonstrating effectiveness on such architectures would substantially strengthen the paper’s claims of generality.

W3. Lack of Comparison with Hierarchical Transformers: Recent hierarchical vision transformers (e.g., MViTv2 (CVPR’22)) achieve higher accuracy with fewer FLOPs. The paper should clarify why FourierRoFormer would be preferable to these efficient hierarchical backbones, or ideally, provide comparative experiments.

W4. Effect on Extrapolation Capability:
One of the main motivations behind rotary embeddings is their excellent positional extrapolation behavior. The paper does not analyze how the proposed Fourier modulation and damping affect this property. Experiments with longer sequences or higher-resolution images would help assess this aspect.

W5. Missing Visualization results: Although the authors mention attention visualizations (L442–448), no qualitative figures are provided in the main text.
Including examples—particularly for the multi-head frequency specialization analysis (L302–370)—would greatly enhance the interpretability and credibility of the claims.

### Questions
Overall, I find the proposed method and its analytical insights interesting.
However, my main concern is its limited practical effectiveness. The proposed method does not clearly outperform existing spectral or hierarchical transformers in efficiency or accuracy, leaving its practical advantage insufficiently justified.
It would also strengthen the paper to include analyses on whether RoFormer’s extrapolation capability is preserved and additional visualizations that better support the claimed frequency–semantic alignment.

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
2

### Summary
This work proposes a novel frequency-aware Transformer architecture, named FourierRoFormer, to model spatial frequencies and attention decays with respect to distance. The proposed method is compatible with the rotary positional encoding which is widely used in large language models and vision Transformers. This work augments the vanilla Transformer with the multi-scale features that are particularly suitable for computer vision tasks. Extensive experiments on classification, detection, and instance segmentation tasks demonstrate the effectiveness of the designed modules.

### Strengths
a)	This work effectively identifies the key limitations of Transformer networks, including a lack of spatial inductive bias, frequency blind and absence of attention decays

b)	The designed method effectively improves the baseline performance while bringing only minor inference burden and keeping the architecture simplicity.

c)	This work provides detailed analysis and proof to explain the proposed method.

### Weaknesses
i.	The authors need to compare the proposed method with Swin transformer which uses shifted windows to build multi-scale visual patterns, and PVT which uses spatial reduction attention in Table 1 to Table 3.

ii.	The authors could provide more visualization results to show how spatial frequencies and attention decays work in the proposed FourierRoFormer.

iii.	Is the proposed method sensitive to the image resolution, especially when the training and testing images have different resolutions?

iv.	Section 3 should be organized into more subsections to make the manuscript easier to follow.

v.	How do Fourier modulation and damping maintain interpretability?

vi.	References of the compared methods should be included in Table 1.

### Questions
See weakness.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FourierRoFormer, a novel attention mechanism for Vision Transformers that incorporates learnable Fourier components for positional encoding in the attention. The method modulates attention scores based on token distance using a mixture of sinusoidal functions whose parameters (amplitude, frequency, phase) are learned end-to-end, along with an optional exponential damping term. It is a way to model multi-scale visual patterns and adaptive attention decay. The main contributions are the proposed mechanism, its theoretical analysis guaranteeing stability and expressivity, and extensive empirical results demonstrating performance improvements.

### Strengths
- While spectral methods for transformers exist, the proposed approach is original in its formulation of a *learnable, distance-dependent modulator* based on a Fourier series, directly integrated with the commonly used RoPE.

- Most claims are supported by both theoretical analysis (Theorems 1-3) and a comprehensive empirical evaluation. The ablation studies are thorough, and the interpretability analysis is particularly impressive. It provides strong quantitative evidence (e.g., Pearson correlation r=0.85 between learned frequencies and object boundaries) that the model learns a meaningful, hierarchical division of labor, which goes beyond simply reporting improved accuracy.

- The paper is well-structured, making the core ideas and experimental results easy to follow and understand.

### Weaknesses
I only have 2 minor weaknesses to point out:

1. Performance-Parameter Trade-off: The claim that the method "offers a better performance-parameter trade-off than spectral methods" appears to be an overstatement based on the results in Table 1. For instance, SVT-H-B achieves higher accuracy (85.2%) with significantly fewer parameters (32.8M) and GFLOPs (6.5) than FourierRoFormer-B (84.1% acc, 86.4M params, 17.5 GFLOPs). This discrepancy likely arises from an unfair (to the disavantage of FourierRoFormer) comparison between a model built on a standard ViT backbone and others using more efficient hierarchical backbones. The claim should be more carefully qualified to reflect that the primary benefit is the improvement over its direct baseline (RoFormer).
2. Uncertainty of Benefits at Larger Scale: The paper's core motivation is to introduce a beneficial inductive bias for modeling visual data. However, the experiments are limited to datasets up to the scale of ImageNet-1K. It remains an open question whether this explicit frequency-based bias would continue to provide an advantage, or potentially act as a constraint, when training at a much larger scale (e.g., on web-scale datasets used for training CLIP-like models). A discussion on this limitation is important, especially since the original ViT paper already noted that transformers can learn sinusoidal position patterns implicitly given sufficient data. To be clear I’m not asking for additional large-scale experiments, which would be unreasonable, I’m just asking for a discussion on that matter.

### Questions
1. Could you please clarify the claim of a superior performance-parameter trade-off in light of models like SVT-H-B and SpectFormer-H-B? 
2. Regarding the inductive bias, what is your perspective on how the FourierRoFormer mechanism might behave at a much larger scale of data and compute? Do you hypothesize that the learned frequency structure would become more complex and beneficial, or is it possible that a sufficiently large, unbiased model could learn superior patterns on its own, rendering this explicit structure less impactful?
3. The paper briefly mentions a +0.5pp accuracy gain from using head-specific frequency parameters. This seems like a promising direction. Could you elaborate on this result? Does this variant lead to an even sharper and more diverse frequency specialization across heads, and what is the associated parameter/computational overhead?

### Soundness
4

### Presentation
4

### Contribution
3
