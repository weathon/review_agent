# Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Rotary Position Embedding (RoPE) is a widely adopted technique for encoding relative positional information in large language models (LLMs). However, when extended to vision-language models (VLMs), RoPE and its variants enforce relative positional dependencies separately within text and image tokens, introducing unintended cross-modal positional biases. For example, image tokens depicting semantically consistent content are assigned distinct positional encodings solely due to spatial location variations. As a result, such tokens exhibit entirely different relative positional relationships with their corresponding text tokens, ultimately leading to misaligned cross-modal representations.
To address this, we propose Per-Token Distance, a simple yet effective metric for quantifying the independence of positional encodings across modalities. Informed by this analysis, we introduce Circle-RoPE, a novel encoding scheme designed to eliminate spurious cross-modal biases.
Our key idea is to project image token indices onto a *ring* that is orthogonal to the linear axis of text token indices, thereby forming a cone-like structure in the positional encoding space. In this configuration, each text token (point on the linear text axis) becomes the apex of a cone and maintains an equal distance to all image tokens (points on the circular image *ring*), reducing artificial cross-modal biases while preserving intra-image spatial information.
To further enhance performance, we propose a staggered strategy that applies different RoPE variants across layers. Extensive experiments demonstrate that our method effectively preserves spatial information from images while reducing relative positional bias, offering a more robust and flexible positional encoding framework for VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Circle-RoPE, a novel positional encoding method for vision-language models that addresses cross-modal positional biases in existing RoPE variants. The key idea is to project image token indices onto a circular structure orthogonal to the text token sequence, forming a cone-like geometry where each text token maintains equal distance to all image tokens. The authors also propose an Alternating Geometry Encoding strategy that switches between M-RoPE and Circle-RoPE across layers.

### Strengths
- The Per-Token Distance (PTD) metric quantifies cross-modal positional bias, clearly demonstrating the limitations of existing approaches.
- The cone-like structure is intuitive, achieving PTD=0 while preserving intra-image spatial relationships.
- Experiments show modest gains across multiple benchmarks (e.g., +1.89 on MMMU, +4.07 on MMStar) and generalization to different architectures (LLaVA).

### Weaknesses
- The circular projection introduces significant geometric constraints that may not align with how VLMs naturally learn cross-modal relationships. By forcing all image tokens to be equidistant from text tokens, the method may hinder spatial reasoning tasks. For example, questions like "What is in the top-right corner of the image?" rely on the model associating spatial language ("top-right") with specific image regions. Circle-RoPE's equidistance constraint removes positional cues that could help the model learn these associations, potentially making such spatial grounding more difficult.
- Statistical significance is not reported
- Table 6 shows Circle-RoPE underperforms initially, suggesting the method fights against pretrained representations. More investigation is needed on whether gains justify the retraining requirements.

### Questions
- What about n dimensional input? How would you method scale with modality dimensions?
- Why is PTD=0 the right objective?
- How sensitive is model performance to the radius hyperparameter?
- “For convenience, we set Nimage = 9 and Ntext = 5.” I am confused by this sentence. Did you set the  Nimage = 9 and Ntext = 5.” for all examples? What is the reasoning behind this choice?

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
3

### Summary
This paper identified cross-modal positional bias in RoPE-based VLMs, where text and image token indices induce spurious relative relationships that misalign semantics. Based on the observation, it introduces Circle-RoPE, which projects image token indices onto a circle orthogonal to the text index axis, forming a cone-like geometry that equalizes text-to-image RoPE distances while retaining intra-image structure via mixed-angle mapping. It also includes alternating geometry encoding strategy interleaving Circle-RoPE with M-RoPE across layers to balance decoupling and spatial priors. Experiments on multiple benchmarks and architecture show reasonable improvements.

### Strengths
1. The paper is well-written and easy to follow.
2. The motivation to alleviate cross-modal positional bias makes a lot of sense to me. And the idea to project the image taken indices onto a circle orthogonal to the text index axis also sounds very reasonable.

### Weaknesses
1. The design of the mixed-angle circular mapping feels ad-hoc. The spatial-origin angle groups tokens with similar polar angles, which is reasonable (though it ignores inter-token distance). But adding the grid-index angle breaks this principle and makes positions appear random. For instance, the grid-index angle places patches 2 and 3 together even though their spatial-origin angles differ greatly. The combination of these two angles doesn’t make sense to me. Furthermore, Table 3 shows no dominant weighting scheme or monotonic trend, and the performance differences across strategies are minimal.
2. The auto-k radius calculation also feels ad-hoc to me. Increasing R appears to reduce visual–textual attention. What’s the rationale? If the input image is large, should we actually be lowering attention to the visual tokens?
3. In the experiments, the authors freeze the vision encoder and projection layer and finetune only the LLM. However, in Qwen2.5-VL the vision encoder and LLM are trained jointly, meaning the visual encoder is co-trained with M-PoPE as the positional encoding. This mismatch may complicate the evaluation of Circle-PoPE, since we don’t know whether using Circle-PoPE during the training result in inferior visual representations.

Overall, while I appreciate the orthogonal design between the visual and language indices, I am not fully persuaded by how the angle and radius of the visual-index circle were chosen. I would be glad to reevaluate the work if the authors could elaborate on the rationale for these design choices.

### Questions
1. The experiments only use single-image VQA. Whether Circle-RoPE can be extended to multiple-image scenario?
2. How is Circle-RoPE robust to the different resolution of the input images?
3.  Does the baseline (Qwen2.5-VL) in Table 2 retrained using the same training setting or is the original version of Qwen2.5-VL?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Circle-RoPE, a positional encoding schema designed to decouple text and image positional biases in vision-language models (VLMs). The approach projects image token indices onto a circle orthogonal to the text token axis, forming a cone-like geometry that guarantees all text tokens are equidistant from all image tokens in RoPE space—aiming to minimize artificial cross-modal positional bias without sacrificing intra-image spatial relationships. The authors further introduce Alternating Geometry Encoding (AGE), alternating between traditional and Circle-RoPE layers to blend cross-modal disentanglement and spatial structure preservation. Exhaustive experiments demonstrate that Circle-RoPE yields measurable improvements on a suite of multimodal benchmarks and remains robust across VLM architectures.

### Strengths
1. The paper clearly identifies and articulates a subtle yet critical problem: cross-modal positional biases. The VQA example in Figure 2 is highly persuasive, as it visually demonstrates how existing encodings (like M-RoPE) cause misalignment. For instance, the phrase "high on" (semantically related to the tower's top, index 1) is shown to be positionally closer to index 8, which is incorrect. This provides strong support for the paper's motivation.

2. The core idea of Circle-RoPE—constructing a "cone-like structure" to achieve equal distance by projecting image token indices onto a ring that is orthogonal to the linear axis of text tokens—is a highly novel and mathematically elegant solution. Instead of relying on complex network architectures, it addresses the problem defined by the PTD metric directly from the geometric nature of positional encoding.

3. The experimental section of this paper is very solid.Ablation of Key Components: Table 3 and Table 4 present comprehensive ablation studies on CIP and AGE (different layering strategies), thoroughly validating the design choices. The discussion on "adaptation cost" in Appendix A.1 is excellent. The authors candidly note that the new method's performance lags behind the baseline in early training (3k steps) and only begins to surpass it after approximately 8.5k steps.

### Weaknesses
1. This is the paper's most significant weakness, stemming from a narrative tension between the stated problem and the optimal solution.
**The Problem**: The paper's premise is that existing methods like M-RoPE introduce harmful, spurious cross-modal biases.
**The Goal**: The paper aims to eliminate this bias by using Circle-RoPE to achieve a Per-Token Distance (PTD) of 0.
**The Contradiction**: The best-performing model (AGE, strategy 4) is a hybrid strategy that re-introduces the supposedly "biased" M-RoPE in half of its layers.
**The Evidence**: The ablation study (Table 4) shows that the "pure" Circle-RoPE strategy (strategy 1), which fully achieves the goal, performs (Avg score 63.19) worse than the mixed AGE strategy (Avg score 63.64). This suggests that complete decoupling (PTD=0) may itself be a suboptimal objective, undermining the paper's central argument.

2. While the generalizability experiment on LLaVA (Table 5) is included, the performance improvement is marginal (Avg Score 29.83 vs. 29.48). This minimal gain, achieved on a small-scale 0.5B model, provides insufficient evidence to strongly support the method's universal applicability to non-M-RoPE architectures.

3. Appendix states that multi-image sequences are handled by translating each image's circular encoding center along a fixed global axis g=[1,1,1]. This design appears to re-introduce a linear, hard-coded sequential bias between images, which contradicts the paper's core philosophy of opposing "hard-coded designs" and "artificial biases."

4. The paper's inference that PTD=0 (a metric based on Euclidean distance) equates to a complete "elimination of bias" remains empirical. It lacks a theoretical proof that PTD=0 is a sufficient or necessary condition for removing all positional bias. Furthermore, it is debatable whether the orthogonality used to achieve PTD=0 is a necessary design choice.

5. The paper contains minor typos, such as "Circle-RoPR" in the conclusion for Table 5, which should be "Circle-RoPE."

### Questions
1. The ablation in Table 4 shows that the "pure" Circle-RoPE (strategy 1) performs worse than the hybrid AGE strategy (strategy 4). Why is this the case, and does it imply that your stated goal of PTD=0 is suboptimal?

2. What specific spatial information (which you refer to as the "strong grid-based spatial prior" ) is lost when using pure Circle-RoPE that M-RoPE retains?

3. Can you provide a formal proof that minimizing your PTD metric necessarily leads to better cross-modal alignment, rather than just being an empirical correlation?

4. Is the "orthogonal cone structure" the only geometric configuration that can achieve PTD=0, or did you explore other non-orthogonal transformations?

5. The performance gain on LLaVA in Table 5 is marginal (Avg Score 29.83 vs 29.48). Is this minimal gain due to a lack of specific tuning, or is the cross-modal bias problem inherently less severe in 1D-RoPE models like LLaVA?

6. In Appendix A.3, you re-introduce a hard-coded linear bias to handle multiple images (translating along a fixed axis g=[1,1,1]) . Why is this hard-coded bias acceptable between images when your paper's premise is to eliminate this exact type of bias between text and image tokens?

### Soundness
2

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
4

### Summary
This paper addresses the cross-modal positional biases that occur when standard RoPE is applied to MLLMs. The authors propose Circle-RoPE, a novel encoding scheme that projects image token indices onto a ring. This ring is oriented to be orthogonal to the linear axis of the text token indices , which forms a cone-like structure in the positional encoding space. This design effectively reduces artificial cross-modal biases while simultaneously preserving the original spatial information within the image. To further improve performance, the authors also propose a staggered strategy (Alternating Geometry Encoding) that alternates RoPE variants across different layers.

### Strengths
1. The paper proposes a novel and intriguing concept with the "cone-like structure" for the positional encoding space. This is a creative approach to decoupling cross-modal positional dependencies.
2. The paper is well-written. The authors' summary and comparison of the three common categories of positional encodings in MLLMs (hard, unordered, spatial) are exceptionally clear. This effectively contextualizes the problem and makes the contribution of Circle-RoPE easy to understand.

### Weaknesses
The primary weakness of this paper lies in the insufficient experimental comparisons, which makes the reported results less reliable and the conclusions less convincing.

1. **Missing Baseline in Table 2:** The main comparison in Table 2 is insufficient. To properly benchmark the proposed "Ours" method, a critical baseline is missing: the results of the base Qwen2.5-VL model after being fine-tuned on MAmmoTH-VL-Sub dataset. 
2. **Missing Baseline in Table 4:** In Table 4, which compares the four Alternating Geometry Encoding strategies, the analysis lacks the most important baseline: the performance of a model using the standard M-RoPE (the method used in the original Qwen-VL) in all layers. 
3. **Marginal Gains and Inconsistent Benchmarks:** The performance improvements in the ablation studies (e.g., Table 3) appear very marginal. This concern is compounded by the fact that different ablation experiments are evaluated on different sets of benchmarks (e.g., Table 3 uses MMMU/MMStar, while Table 4 uses AI2D/ChartQA). The authors should provide results across a consistent and more comprehensive set of benchmarks for all ablations to avoid any suspicion of cherry-picking and to more robustly validate their design choices.
4. **Missing PTD=0 Baseline in Table 5:** Table 5, which tests generalizability, is missing a critical baseline: the "unordered positional embedding" (as described in Figure 1b and Table 1). The paper explicitly states that both Circle-RoPE and unordered embedding achieve a $PTD=0$. Therefore, comparing against this baseline is essential to prove that the performance gains come from Circle-RoPE's specific geometric structure (preserving intra-image spatial information) and not merely from achieving a $PTD=0$

### Questions
1. In Table 4, the comparison of AGE strategies is missing the M-RoPE baseline. Could the authors provide this result to clarify the actual benefit of the alternating strategy?
2. The generalizability test in Table 5 is missing the "unordered embedding" baseline, which also achieves PTD=0. Why was this important baseline omitted, and how does Circle-RoPE compare to it?

### Soundness
3

### Presentation
3

### Contribution
2
