# NGS-Marker: Robust Native Watermarking for 3D Gaussian Splatting

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
With the rapid development and adoption of 3D Gaussian Splatting (3DGS), the need for effective copyright protection has become increasingly critical. Existing watermarking techniques for 3DGS mainly focus on protecting rendered images via pre-trained decoders, leaving the underlying 3D Gaussian primitives vulnerable to misuse. In particular, they are ineffective against **Partial Infringement**, where an adversary extracts and reuses only a subset of Gaussians. In this paper, we propose **NGS-Marker**, a novel native watermarking framework for 3DGS. It integrates a jointly trained watermark injector and message decoder, and employs a gradient-based progressive injection strategy to ensure full-scene coverage. This enables robust ownership decoding from any local region. We further extend NGS-Marker with hybrid protection (combining native and indirect watermarks) and support for multimodal watermarking. Extensive experiments demonstrate that NGS-Marker effectively defends against partial infringement while offering practical flexibility for real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes NGS-Marker, a native watermarking framework for 3D Gaussian Splatting (3DGS) that addresses the problem of partial infringement. Unlike existing indirect methods that watermark rendered images, NGS-Marker embeds watermarks directly into 3D Gaussian primitives, enabling detection even when only subsets of primitives are extracted and reused. The method uses a jointly trained watermark injector and message extractor with a progressive optimization strategy.

### Strengths
## Strengths

1. The paper identifies a critical and underexplored vulnerability in 3DGS assets - partial infringement, where adversaries extract and reuse subsets of Gaussian primitives. The motivation is clearly articulated with concrete examples.

2. The method achieves native watermarking embedding and extraction.

3. The hybrid protection mechanism and personalized watermarking demonstrate practical flexibility.

### Weaknesses
## Weaknesses

1. While the feasibility analysis (Section 3, Figure 2) shows that HiDDeN can embed watermarks in random noise, the theoretical connection to why this should work for structured 3D Gaussian primitives is weak. The paper lacks analysis of what properties of Gaussian primitives make them suitable for watermark embedding.

2. Table 4 shows embedding time increases significantly with scene complexity (4 min for 42K primitives to 35 min for 589K primitives)
   The authors acknowledge in limitations that "designing an efficient partitioning and traversal strategy could further improve scalability" but don't provide solutions.
   How does the method scale to production-level scenes with millions of primitives?

3. No analysis of adversarial attacks specifically designed to remove native watermarks
   What if an adversary fine-tunes or re-optimizes the Gaussian primitives?
   The robustness experiments (Table 3) only test standard distortions, not adversarial removal attempts.

4. Although the method achieves native embedding, the method is built on top of the PointTransformer. What parameters are used in the injection and extraction? Are all parameters used for injection and extraction? There should be an ablation study for different parameters that are selected for injection and decoding.
For example, if we can only use the color and position attributes, can we extend the method to become compatible with point cloud data?

5.  The partial infringement simulation may not fully capture real-world misuse scenarios. More challenging infringements should be discussed such as 3D adversarial attacks, Semantic editing[1] or recoloring[2]

[1] GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting

[2] GeometrySticker: GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields

### Questions
## Questions

1. What is the false positive rate? How often does the method incorrectly identify non-watermarked primitives as watermarked?

2. How does the method achieve indirect message encoding and decoding? 

Do you optimize the injected with the pretrained frozen message decoder just like the approaches in [1][2]?

How different is the optimization process for the indirect, native, and hybrid optimization?

Do you first optimize for native and then for indirection protection? Or do you optimize them simultaneously?

I think the explanation for this port should be clearer.

3. The paper focuses on static 3DGS, but many applications involve dynamic content. 

Can the proposed method be extended to dynamic scenes?

4.  Only 4 test scenes are quite limited for drawing strong conclusions. 
     Are the injector and extractor pretrained on the 24 training scenes and then only validated on the 4 test scenes?
    How are the validation results on the training datasets?
     Beyond the 4 test scenes, how does the method perform on more complex, diverse scenes such as Tanks and Temples Benchmark?

[1] WateRF: Robust Watermarks in Radiance Fields for Protection of Copyrights

[2] 3D-GSW: 3D Gaussian Splatting for Robust Watermarking

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
4

### Summary
To address Partial Infringement of 3D Gaussian Splatting (3DGS), where an adversary extracts and reuses only a subset of Gaussians, this paper presents NGS-Marker, a native watermarking framework for 3DGS by integrating a jointly trained watermark injector and message decoder and adopting a gradient-based progressive injection strategy, to achieve robust ownership decoding from any local region. The NGS-Marker is further extended with hybrid protection (combining native and indirect watermarks) while personalized watermarking is supported to embed image watermark. The experimental results show that NGS-Marker effectively defends against partial infringement while offering practical flexibility for real-world deployment.

### Strengths
A new watermarking framework named NGS-Maker is proposed for 3D Gaussian Splatting (3DGS) by integrating a jointly trained watermark injector and message decoder and adopting a gradient-based progressive injection strategy, to achieve robust ownership decoding from any local region. Hybrid protection (combining native and indirect watermarks) and personalized watermarking are both supported. The experimental results show the efficacy of NGS-Marker.

### Weaknesses
First, the concepts of indirect watermarking and native watermarking have not been clearly explained or referred, making it difficult to compare the proposed watermarking framework named NGS-Maker with the related methods. 

Similarly, the definition of Partial Infringement also needs to be referred and compared with terms used in the literature. 

Third, the performance comparison between the proposed method with three baselines is not enough and the performance analysis in robustness is very limited.

Last, injecting image-based watermarks into 3D scenes using NGS-Marker is okay but not very personalized.

### Questions
1. What mostly make the proposed watermarking framework NGS-Maker different from the existing methods?

2. What are the concrete concepts/definitions of “indirect watermarking” and “native watermarking”?

3. What are the methods in the literature most similar to the proposed one?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
NGS-Marker proposes a native 3DGS watermark that is embedded and decoded directly from local Gaussian neighborhoods via a progressive optimization pass. The method targets partial infringement (copying only subsets of Gaussians) and reports strong per-Gaussian detection and rendering fidelity, plus optional hybrid protection with indirect, image-domain marks and a proof-of-concept for personalized watermarks.

### Strengths
1. Composable design. The proposed hybrid objective indicates native marks don’t conflict with rendering-domain watermarks, and the image-message experiment shows the architecture can accommodate non-binary payloads with small changes to encoders/decoders. This extends the method’s use beyond simple IDs. 

2. Minimal visual impact with qualitative and quantitative evidence. Difference maps show small, spatially diffuse changes, and PSNR/SSIM/LPIPS metrics remain strong after embedding; this is the right mix of perceptual and pixel-wise indicators for vision papers. The visual comparisons make the progressive optimization design choice more convincing. 

3. Clear pipeline and rationale. The progression from local injection to progressive, scene-level embedding is well-motivated: naive block-wise injection yields boundary artifacts, whereas repeated random injection accumulates distortion; the final procedure addresses both issues. This kind of explicit failure-mode discussion is useful for re-implementation.

### Weaknesses
1. White-box resilience not evaluated. The attack suite focuses on stochastic distortions rather than adversarial removal with knowledge of the extractor. With the absence of such results, the robustness to an adaptive attacker remains an open question. 

2. Scaling cost and production fit. Embedding time rises from 4 minutes (42k primitives) to 35 minutes (589k), which is acceptable for small scenes but may be significant for game/film assets with millions of Gaussians; the paper notes scalability opportunities but does not provide an algorithmic path or empirical stress test beyond the four test scenes.

### Questions
1. Verification cost: What is end-to-end verification latency (per million Gaussians), including neighborhood sampling and similarity computation? Any heuristics to prune the search while maintaining recall? 

2. Breadth of validation: Could you add evaluations on larger, diverse benchmarks (e.g., Tanks & Temples) and any dynamic/4DGS content to assess generalization?

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
This paper introduces NGS-Marker, a novel native watermarking framework designed to protect 3D Gaussian Splatting assets against partial infringement. The authors identify a critical limitation in existing indirect watermarking methods, which fail when the rendering distribution changes significantly. NGS-Marker addresses this by embedding watermarks directly into the Gaussian primitives using a jointly trained injector-extractor pair and a progressive gradient-based optimization strategy. The method supports hybrid protection and multimodal watermarking, and demonstrates robustness to common distortions.

### Strengths
1. The paper clearly identifies and formalizes the partial infringement problem in 3DGS, which is both practical and underexplored. This is a meaningful contribution to the field of 3D asset protection.

2. The proposed framework enable native local watermarking for 3DGS. The use of a perturbation-based injector and extractor, combined with progressive optimization, is well-motivated and technically sound.

### Weaknesses
1. Although the method supports large scenes, the embedding time grows with the number of primitives. Is it possible to add a comparison of watermark embedding time and extracting time with previous methods in the experiment?


2. Lack of discussion and comparison on computational complexity. Since the proposed method utilizes Point Transformer as the injector, adding experimental results on this aspect would better reflect the model's characteristics.

### Questions
1. How does this method perform in areas with significant overlap, such as when the embedded Gaussians and the watermarked Gaussians interweave with each other?

2. Could the choice of $\delta$ influence the noise robustness of the proposed method? Especially for densification and dropout.

### Soundness
3

### Presentation
3

### Contribution
3
