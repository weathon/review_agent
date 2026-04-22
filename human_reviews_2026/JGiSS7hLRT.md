# MagicTryOn: Harnessing Diffusion Transformer for Garment-Preserving Video Virtual Try-on

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Video Virtual Try-On (VVT) aims to synthesize garments that appear natural across consecutive video frames, capturing both their dynamics and interactions with human motion. Despite recent progress, existing VVT methods still suffer from inadequate garment fidelity and limited spatiotemporal consistency. The reasons are (i) under-exploitation of garment information, with limited garment cues being injected, resulting in weaker fine-detail fidelity, and (ii) the lack of spatiotemporal modeling, which hampers cross-frame identity consistency and causes temporal jitter and appearance drift. In this paper, we present MagicTryOn, a diffusion transformer–based framework for garment-preserving video virtual try-on. To preserve fine-grained garment details, we propose a fine-grained garment-preservation strategy that disentangles garment cues and injects these decomposed priors into the denoising process. To improve temporal garment consistency and suppress jitter, we introduce a garment-aware spatiotemporal rotary positional embedding (RoPE) that extends RoPE within full self-attention, using spatiotemporal relative positions to modulate garment tokens. We further impose a mask-aware loss during training to enhance fidelity within garment regions. Moreover, we adopt distribution-matching distillation to compress the sampling trajectory to four steps, enabling real-time inference without degrading garment fidelity. Extensive quantitative and qualitative experiments demonstrate that MagicTryOn outperforms existing methods, delivering superior garment-detail fidelity and temporal stability in unconstrained settings. Code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MagicTryOn, a framework for video virtual try-on (VVT) based on a diffusion transformer architecture. The primary goal is to enhance garment fidelity and spatiotemporal consistency in the generated videos. The authors propose three main contributions: 1) A fine-grained garment preservation strategy that decomposes the garment image into semantic, structural, and appearance cues, which are then injected into the denoising process via dedicated cross-attention mechanisms. 2) A garment-aware spatiotemporal rotary position embedding (GAS RoPE) designed to improve temporal stability by encoding relative positional information for garment tokens across frames. 3) The use of distribution-matching distillation to create a "Turbo" version of the model, enabling real-time inference in just four steps. The method is evaluated on standard VVT datasets like ViViD and VVT, where it reports state-of-the-art performance on several metrics.

### Strengths
*Strong Quantitative Results*: The paper presents impressive quantitative results on the ViViD and VVT datasets (Tables 1, 2, 5), consistently outperforming prior state-of-the-art methods across multiple metrics (VFID, SSIM, LPIPS). This suggests the proposed combination of techniques is highly effective.

*High-Speed Inference*: The development of MagicTryOn-Turbo, a distilled version capable of generating a 64-frame video in ~6.7 seconds, is a significant practical contribution. Achieving such a substantial speed-up (reportedly 50x) while maintaining strong performance makes the method much more viable for real-world applications.

*Well-Written Paper*: The paper is generally well-structured, clearly written, and easy to follow. The methodology is explained in sufficient detail, and the figures (especially Figure 2) are helpful in understanding the overall architecture.

### Weaknesses
*Lack of Video Evidence for Key Claims (Major Concern)*: The paper's primary claims revolve around improving spatiotemporal consistency and reducing temporal jitter. However, all comparisons—both against SOTA methods and in the ablation study—are presented as sequences of still frames. Still frames are insufficient for evaluating temporal dynamics and can be susceptible to cherry-picking. For a video generation task, providing actual video results in the supplementary material (e.g., via an anonymous repository) is crucial for validating claims of temporal superiority. The absence of video comparisons severely weakens the paper's persuasiveness.

*Overstated Novelty of Some Components*:
- Full Self-Attention: The paper mentions employing full self-attention to unify spatial and temporal modeling. However, this is not a novel contribution to the VVT domain, as it was previously utilized in CatV2TON. Moreover, the base model used, Wan2.1, already incorporates this architecture. The authors appear to be leveraging an existing feature of the backbone rather than introducing a new concept.
- Garment-Aware Spatiotemporal RoPE (GAS RoPE): The novelty of GAS RoPE is not clearly distinguished from prior work. The w/o GAS ablation simply removes RoPE for the prepended garment token, which would obviously degrade performance due to the lack of spatial alignment information. Methods like CatV2TON also treat the garment reference as a special token (akin to a first frame) to provide conditioning. A more detailed explanation of how GAS RoPE is fundamentally different and more effective is needed.
- Unfair Comparison with the Base Model: In Table 2, the authors compare MagicTryOn with a fine-tuned version of Wan2.1-I2V to demonstrate the effectiveness of their proposed modules. However, the paper states that the foundational model for MagicTryOn is Wan2.1-Fun-Control, which is an improved version of Wan2.1-I2V fine-tuned on additional data. This comparison is unfair. To isolate the true benefit of the proposed modules, the comparison should be against a fine-tuned Wan2.1-Fun-Control baseline, not the weaker Wan2.1-I2V.

*Questions Regarding the Ablation Study*:
- Magnitude of Impact: The ablation results in Table 3 are suspicious. Every single ablated component leads to a very significant drop in performance across nearly all metrics. It is unusual for each module to have such a large, independent impact. A more convincing approach would be an additive study, starting from a baseline and progressively adding each new component to show its marginal gain.
- Impact of Mask-Aware Loss: The performance drop from removing the mask-aware loss (w/o mask) is surprisingly large, nearly comparable to removing the entire Feature-Guided Cross-Attention (FGCA) module. Typically, such a loss term provides a modest refinement rather than being a cornerstone of performance. This counter-intuitive result requires a thorough explanation. Furthermore, mask-aware loss can risk information leakage from the mask shape, causing the model to generate garments that unnaturally conform to the mask boundaries. The paper does not discuss how this potential issue is mitigated.

*Potential Technical Inaccuracy*: Equation (5) formulates the training objective as a noise prediction loss (||εθ - ε||²). However, the Wan2.1 backbone model, on which this work is based, is known to use velocity prediction. This discrepancy should be clarified.

### Questions
*Component Choices*:
- Could you justify the use of a dedicated line estimation module (Pan, 2025) over a simpler, classic method like Canny edge detection? Was an ablation performed to compare different line extraction methods?
- For text descriptions, you used Qwen2.5-VL-7B. How sensitive is the model's performance to the quality and accuracy of these generated captions? Was a more powerful model, such as the 32B variant, considered to potentially provide more accurate descriptions?

*Interpretability*: To better demonstrate the contribution of the decomposed structural cues, could you provide visualizations of attention maps? This could offer a more direct and intuitive view of how the Line Tokens specifically contribute to the generation of sharp garment details, patterns, and logos.

*Ablation Study Clarifications*:
- Could you provide an explanation for the unexpectedly large performance drop when the mask-aware loss is removed?
- How did you prevent the mask-aware loss from causing the model to overfit to the mask shape, which can lead to unrealistic garment deformations, especially when the agnostic mask is not perfectly accurate? Video results for this specific ablation would be very informative.

*Video Results*: Will you be providing video comparisons to substantiate the claims of superior temporal consistency? (e.g., via an anonymous repository) This would include comparisons against SOTA methods and videos for the key ablation studies (e.g., with and without GAS RoPE, with and without mask-aware loss).

### Soundness
2

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
The paper introduces MagicTryOn, a diffusion transformer-based framework for garment-preserving video virtual try-on (VVT) that aims to improve both garment-detail fidelity and spatiotemporal consistency in generated try-on videos. The approach leverages a fine-grained garment-preservation strategy (decomposing cues into semantic, structure, and appearance), introduces a garment-aware spatiotemporal rotary position embedding (RoPE) for enhanced temporal consistency, and utilizes a mask-aware loss to enforce garment-region fidelity. Furthermore, it integrates a distribution-matching distillation to accelerate inference without compromising quality. The authors present quantitative and qualitative improvements over several state-of-the-art methods on public VVT benchmarks, supported by ablations and user studies.

### Strengths
1. Innovative Spatiotemporal Encoding: The extension of RoPE to “garment-aware spatiotemporal RoPE” is principled and directly addresses temporal instability; the subsuming of garment tokens into the full self-attention with grid extension is mathematically described and justified.
2. Thorough Ablations: Table 3 and Figure 12 provide a detailed ablation of key architectural modules, with analyses that specify the performance and qualitative impact of removing each token stream or loss component, which aids scientific transparency.
3. Clear writting.

### Weaknesses
1. Limited Novelty in Architectural Choices: While the garments’ semantic/structural/appearance decomposition and the cross-attention wiring are interesting, the overall method mainly combines existing mechanisms from prior VVT and diffusion transformer literature, such as patchification, CLIP feature usage, full self-attention, and cross-token fusion. As evident from the Related Work and as per foundational methods like ViViD, CatV2TON, or Hunyuan-DiT (all cited in the main text), many core strategies (semantic cross-attention, diffusion transformer backbone, pose-agnostic inputs, multi-modal fusion) are also used in recent related architectures.
2. In addition, the semantic/structural/appearance sequence tokens are redundant and can be replaced by a limited number of keyframes, thereby further enhancing efficiency and reducing preprocessing requirements.
3. Based on the video materials provided, the clothing replacements demonstrated in the paper appear limited—for instance, only showcasing exchanges between different T-shirts or between various pairs of jeans, while failing to present cross-category swaps such as skirts versus trousers. This limitation may be attributed to constraints inherent in the mask-based approach used.

### Questions
See weakness.

### Soundness
2

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
3

### Summary
This paper introduces **MagicTryOn**, a diffusion transformer-based framework for **Video Virtual Try-On (VVT)** to address issues with inadequate garment fidelity and limited spatiotemporal consistency in existing methods. 

The framework improves fine-detail fidelity by proposing a **fine-grained garment-preservation strategy** that disentangles and injects decomposed garment cues into the denoising process. 

To enhance temporal consistency, the authors introduce a **garment-aware spatiotemporal rotary positional embedding (RoPE)** and further employ a distribution-matching distillation technique to enable real-time, four-step inference, ultimately demonstrating superior garment-detail fidelity and temporal stability over state-of-the-art methods.

### Strengths
1. The paper proposes a well-structured diffusion transformer framework tailored for video virtual try-on with clear motivation and technical contributions.
2. It introduces innovative modules, including fine-grained garment-preservation and garment-aware spatiotemporal RoPE, effectively enhancing detail fidelity and temporal consistency.
3. The method achieves real-time inference through distribution-matching distillation while maintaining strong performance, supported by comprehensive experiments.

### Weaknesses
1. The proposed design appears rather standard, primarily relying on a combination of strong pretrained encoders and DiT blocks with cross-attention. The architectural novelty and unique algorithmic contribution seem limited.
2. The framework integrates numerous large components—VAE, T5 encoder, CLIP encoder, Qwen-7B, and Wan2.1—resulting in a highly complex system. It remains unclear which specific modules in Figure 2 are initialized with Wan2.1 pretrained weights, as mentioned in line 315.
3. Given the scale of the model and the processing of 64 frames over 6.69 seconds, the claim of real-time inference seems questionable. Details regarding the evaluation setup, baseline comparisons, number of runs, and deviation bars are missing, making the performance claim less convincing.
4. The analysis section feels overloaded with too many individual use cases (nine in total), which makes it difficult to assess the true impact of each component. A more focused and compact ablation study combining related modules would make the results clearer and more insightful.

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
This paper proposes MagicTryOn, a DiT-based framework for video virtual try-on. It decomposes garments into semantic/structure/appearance cues, injects them via two cross-attention modules, and extends RoPE to garment-aware spatiotemporal positional encoding. The proposed method is evaluated on the ViViD dataset.

### Strengths
1. Experiments are conducted on both image-based and video-based datasets.
2. Ablation studies are conducted to evaluate the effectiveness of each component.

### Weaknesses
1. The claim of correctly maintaining “compositional relationships” in multi-garment try-on is only supported by qualitative Fig.4, with no quantitative metrics (e.g., VFID-I3D, SSIM for multi-garment sequences) or statistical analysis. This leaves the performance of multi-garment handling unsubstantiated.
2. The main comparison (Table 1) excludes recent state-of-the-art methods like DreamVVT (Zuo et al., 2025) or SwiftTry (Nguyen et al., 2025), which also focus on temporal consistency. This incomplete benchmarking makes it hard to contextualize MagicTryOn’s true standing in the current VVT landscape.
3. Some qualitative examples are not promising. For instance, in Figure 4, the shorts in the generated video underwent deformation to align with the mask shape of the skirt from the original video.
4. The paper lacks analysis of scenarios where the method may underperform.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
