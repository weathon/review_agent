# SoftPose: Learning Soft Attention for Interaction-Aware Multi-Person Image Generation

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Pose-guided human image generation aims to synthesize images of individuals performing specific actions based on pose conditions and textual descriptions. While current methods achieve promising results in single-person scenarios, they often struggle to generalize to multi-person settings, particularly under complex spatial interactions. Existing methods typically employ pose guidance in an undifferentiated manner across the image, leading to structural ambiguity and frequent limb entanglement in interaction zones. To tackle this challenge, we propose SoftPose, a novel approach that learns interaction-aware soft attention to adaptively modulate attention flow across pose regions, enabling more fine-grained focus on both self-related and cross-person occlusion areas. By modeling long-range, global spatial dependencies within and across pose regions, SoftPose effectively resolves ambiguities in interactive scenarios while preserving precise single-person pose fidelity. Additionally, we introduce a progressive feature injection strategy that balances global spatial coherence and local pose details across multiple scales. Extensive experiments demonstrate the superiority of SoftPose compared to current methods in generating high-quality multi-person images with complex interactions and varying scenes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes SoftPose, a pose-conditioned T2I generation framework built upon a U-Net foundation model. The method first extracts dilated skeleton masks for the given N objects and processes them to obtain a soft attention prior characterized by three learnable attention weights (α, β, γ). These weights are designed to reduce limb fragmentation within objects and suppress spurious or ambiguous inter-object correlations during image generation. The resulting pose-guided priors are progressively injected into the U-Net to enhance structural coherence and pose fidelity in the generated images. Extensive experimental results demonstrate the effectiveness of the proposed method.

### Strengths
+ Customized pose-conditioned image generation is a practical and meaningful topic, as it addresses a gap in adapting large foundation models for fine-grained, pose-specific image generation. A single text prompt is often insufficient to describe complex actions, especially in multi-person scenarios.

+ The idea of explicitly injecting pose-based attention priors into the U-Net is straightforward and intuitive, yet it proves effective in enhancing pose fidelity in the generated images.

### Weaknesses
- As shown in Table 1, the experimental results of SoftPose achieve optimal performance only on AP and CAP, while showing suboptimal results on the remaining four metrics, which fails to convincingly demonstrate the overall validity of the method.

- Implementation details are missing. The paper does not specify the dilation rate used in the morphological dilation. This parameter may vary between scenes with two persons and those with more individuals; using a larger dilation rate in multi-person scenarios could cause severe occlusions between subjects. How is this parameter determined?

- The skeleton maps do not contain person identity information, and therefore cannot effectively guide identity preservation during generation. This may lead to identity confusion in the results. For example, in Figure 4, the prompt “oil painting, a painting of two angels and a man” produces an image where the “man” appears to be a woman. It is recommended that the authors evaluate their method using prompts that include explicit person identities, rather than relying solely on simple gender terms such as “man” or “woman”. This would help verify the model’s ability to preserve individual identity in multi-person image generation.

- The paper only employs U-Net–based models, which limits both the image resolution of the generated results and the complexity of input prompts. The authors should integrate their method into more advanced and widely used foundation models, such as SD3 or Flux, to better demonstrate its scalability and general applicability.

### Questions
- The paper only evaluates the proposed method on U-Net based Stable Diffusion, which limits its practical impact and offers minimal advancement to the research area, particularly for a top-tier conference such as ICLR. In fact, Flux and SD3 are already widely used in customized image generation, and evaluating the method on these models would make the work significantly more convincing and meaningful.

- Lines 311–318 indicate that the paper follows the experimental settings of GRPose to conduct experiments on Human-Art. However, the AP score reported in Table 1 (49.50) is significantly lower than the 57.01 reported in the original GRPose paper. The authors should clarify the reason for this discrepancy.

- Why is the FID score lower than that of the original Stable Diffusion? Does this imply that introducing additional conditioning degrades image quality? In contrast, HumanSD improves the FID score to 10.03.

### Soundness
3

### Presentation
2

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
This paper aims to synthesize multi-person image given postures as the guidace. It introduces an interaction-aware attention prior to handle the occlusion area between individuals by assigning higher attention value to the overlapped region. The authors also propose a progressive mult-scale feature injection strategy, where they use different dilation kernel sizes under different denoising timesteps. The experiment results show the proposed method achieves better pose accuracy but worse image quality.

### Strengths
1) The motivation is clear. The proposed attention prior specifically handles the occulusion area between individuals in a multi-person image. 

2) Most part of the paper is easy to follow. Figure 3 provides a good visualization to explain the atteniton prior.

3) The authors conducted experiments on three different benchmarks.

### Weaknesses
1) The analysis of Algorithm 1 is insufficient. Is the self-atteniton product of m_i normalized in Algorithm 1? If not, how will this affect visually smaller individuals in the group photo? In addition, how is self-overlapping such as crossed arms addressed in the attention prior?

2) The experiment results are not convincing. In Table 1, the proposed method has better pose accuracy but worse image quality compared to ControlNet, which is insufficient to support the claim that it generates better details in the interaction region. Since it's difficult to compare using only objective metrics, I suggest the authors provide human evaluation. The authors should also include ControlNet as a baseline in Table 2 since it shows better image quality scores in the Human-Art dataset. Image quaity metrics should also be include in Table 2 as they are essential for a fair evaluation of the proposed method.

3) The visualization results in the paper are all anime-style or art-style. It will be better to see how well the proposed method handles the occlusion in real photos. Can the authors provide more photorealistic examples (e.g., from MHP dataset)?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes SoftPose, a pose-guided diffusion framework for multi-person image generation that explicitly models inter-person interactions. Two core ideas drive the method: (1) an Interaction-Aware Soft Attention Module (IA-SAM) that composes a learnable, overlap-gated attention prior from dilated, per-person skeleton masks; this prior reweights attention to emphasize both self regions and cross-person regions only where skeletons overlap, thereby disambiguating occlusions and reducing limb entanglement. (2) a Progressive Feature Injection Strategy (PFIS) that injects coarse-scale pose features in early denoising steps and fine-scale features later, aligning guidance with diffusion’s coarse-to-fine trajectory. Built on SD-1.5/ControlNet-style conditioning, SoftPose achieves improved pose fidelity on Human-Art, MHP-v2, and MPII using AP, CAP, PCE, FID/KID, and CLIP-score; notably, it reports consistent AP/CAP gains over recent baselines and complementary ablations for IA-SAM, the interaction-aware loss, and PFIS.

### Strengths
1. The attention-prior construction is simple but novel in this context: patch-level, overlap-gated self/cross components with learnable weights (α, β, γ) that multiply and renormalize the base attention scores. This offers a concrete, attention-space mechanism to target interaction zones, rather than treating the whole image uniformly, addressing a well-known pain point in multi-person synthesis (occlusions/limb entanglement).
2. PFIS is a clean, schedule-aware control that matches diffusion’s refinement, improving layout first, details later.
3. The method is well specified: algorithmic definition of the attention prior, integration into ViT attention (Eq. (2)), and a clear two-stage mask-dilation schedule. The ablations isolate IA-SAM, the interaction-aware loss, and PFIS, showing each component’s contribution (Table 3; kernel study Table 4).
4. The paper’s architecture figure and Algorithm 1 make the overlap gating idea easy to follow; Eq. (2) makes the multiplicative reweighting explicit. Qualitative figures vividly show reduced limb merging in overlaps.

### Weaknesses
1. The CAP drop with PFIS (Table 3) is discussed as a trade-off; however, a pose-part AP (e.g., wrists/elbows/knees) or OKS by joint type would substantiate gains on occlusion-prone limbs and clarify where PFIS helps or hurts.
2. The attention prior Am is conceptually L×L at patch resolution; although patching bounds cost, the paper does not fully quantify memory/runtime overhead versus ControlNet across resolutions/timesteps. The claim that runtime is nearly constant w.r.t. subject count (due to fixed grid) is intuitive but would benefit from a measured complexity/runtime table and peak memory profiling, especially at higher resolutions.
3. Ablations on α/β/γ (learned weights) and on the overlap gate itself are limited; e.g., what happens if cross-attention is allowed beyond overlaps but down-weighted by distance? A more granular analysis could reveal when cross-links are helpful outside strict overlaps (Table 3).
4. Most main results focus on SD-1.5; while Fig. 5a hints at compatibility across styles, a stronger baseline on SDXL/FLUX/Qwen-Image would better position the contribution in year 2026.

### Questions
1. Can you provide detailed time/memory breakdowns versus ControlNet/StablePose across resolutions (e.g., 512, 768, 1024) and different numbers of people? Please include per-step cost for computing Am and attention reweighting, and peak VRAM during training/inference. This would substantiate the “nearly constant with subject count” claim (Sec. 4.2).
2. Ablations on attention prior design. (a) What are the learned values/trajectories of α, β, γ over training? (b) How sensitive are results to the overlap gate, e.g., permitting cross-attention outside overlaps but decayed by inter-mask distance; replacing binary a with a soft overlap intensity; or using Gaussian proximity around contact regions? (Algorithm 1/Eq. (2)).
3. Beyond the brief compatibility demo, can you provide quantitative results on SDXL or FLUX or Qwen-Image to confirm that gains from IA-SAM/PFIS are backbone-agnostic?
4. You note face/hand artifacts and rare viewpoints as limitations. Could you show results after adding whole-body keypoints (e.g., hands/face) or auxiliary part-specific losses, and quantify improvements on the identified failures (A.4/Fig. 10)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SoftPose for the generation of human body images in multi-person interaction: IA-SAM is used to redistribute the attention of overlapping areas, and PFIS is used to layout first and then refine the joints. The experiment was conducted on multiple datasets, and the indicators improved.

### Strengths
The motivation focuses on the occlusion/limb entanglement problem in multi-person interaction. The design of IA-SAM is intuitive and can be integrated into the existing SD pipeline. 

The coarse-to-fine schedule of PFIS and the injection from large nuclei to small nuclei are in line with the generation process mechanism and are interpretable. The cross-dataset results and stepwise ablation prove that the method is effective.

### Weaknesses
Is it the same as the strongest baseline in hyperparameters? The advantage is not very obvious. 

The automatic dot filling and automatic text description of MHP/MPII may introduce distribution offsets, affecting generalization evaluation.

### Questions
As the number of people increases and the resolution rises, is the actual latency/video memory consistent with the theoretical increase of patch-level decoupling.

### Soundness
3

### Presentation
3

### Contribution
2
