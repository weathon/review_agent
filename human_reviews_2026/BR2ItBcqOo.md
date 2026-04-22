# RoRE: Rotary Ray Embedding for Generalised Multi-Modal Scene Understanding

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Transformers have emerged as powerful implicit rendering models, capable of performing geometric reasoning and producing photorealistic novel views in a single feedforward pass. A central challenge in these architectures is how to inject camera parameters into the transformer in a way that generalises across diverse sensing conditions. In this work, we present Rotary Ray Embedding (RoRE), an approach that embeds image patches directly as rays, using a learning based rotary positional embedding (RoPE). This ray-based formulation provides a unified and general representation, improving robustness to unconventional camera geometries and sensing modalities. We evaluate our approach on conventional perspective imagery, fisheye cameras, and multi-modal RGB-thermal setups, showing that a single network can flexibly integrate arbitrary numbers of cameras and modalities into a coherent scene representation. Experiments demonstrate improved generalisation and cross-modal consistency compared to existing methods, highlighting the potential for relative ray-based embeddings to build adaptable, plug-and-play vision systems. Code available at: https://roboticimaging.github.io/RoRE

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents RoRE, a novel positional encoding technique for vision transformers. Built upon RoPE (rotatory encoding) that uses 2D patch index encoding, RoRE enhances each image patch using a Plucker ray with learned rotary positional encoding. The authors claim that doing so adds prior knowledge about the camera type into the model (regular cameras vs fisheye). Two innovations in RoRE are (1) learning the rotation frequencies of the positional embedding rather than fixed frequencies, and (2) applying asymmetric rotation to break symmetric biases in attention. The authors evaluated their model on five datasets, and RoRE shows to handle different camera/modalities with strong generalizability.

### Strengths
-	The method itself appears simple and thus should be easy to reproduce.
-	Learned 6-dimension positional encoding empirically aligns with the intuition that the frequencies should decay and shows more uniform patch-wise attention.
-	Experiments show performance gain compared to previous works (excepting concurrent work PRoRE).
-	The ablation studies cover most new components in the model.
-	The paper is well written and easy to follow. The differences from related works are explained clearly.

### Weaknesses
-	The idea of using Plucker rays as positional encoding has been introduced at least in prior work LVSM, which assigns pixel-wise Plucker ray embedding. (However, LVSM does not use positional encoding but simply passes the rays through a linear layer.)
-	A position value P_p is defined for each image patch. However, the pixels within an image patch could have different position values (rays). If just one value is used for an entire patch, then does the size of each patch matter?
-	In Table 4, the changes in metrics are fairly small. Recommend that authors further analyze why. The caption saying it is a boost is largely overstating.
-	In Table 5, the PSNR from thermal needs to be separated from the metrics for RGB.
-	The multi-modal scene understanding is a little bit distant from the main topic. It is hard to see what the authors want to show. Does the author want to demonstrate RoRE is a good sensor fusion tool? If so, this part needs much more detail and analysis. There isn’t any comparison to other works either.
-	The failure cases in Line 420 should still be shown in the paper.

### Questions
-	Line 161, recommend revising to: “, which has 3 position (or moment in the case of Plucker coordinates) dimensions t, and 3 direction dimensions d.” This improves readability.
-	In the qualitative results, what does each of the two input images represent? Recommend to explain somewhere.
-	Line 303, 381 has a grammatical mistake. “is can be see…”
-	Is the masking ratio a fixed value or does it follow a schedule during training?
-	In Figure 4, what are P1 and P2? Different modalities? Furthermore, what is being evaluated in multi-modal understanding? The predicted images seem to be from the same pose as the input, so are you testing image reconstruction?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel positional encoding method for generalizable multiview vision transformers. The proposed Rotary Ray Embedding (RoRE) combines the benefits of relative positional encodings with ray-based position encoding methods, both of which have shown promising results in vision tasks and multi-view 3D scene perception. RoRE demonstrates intriguing robustness properties against changes in camera intrinsics and across heterogeneous sensor modalities. The experimental results validate the robustness and effectiveness of the proposed positional encoding method across various tasks.

### Strengths
- The method demonstrates robust performance on multiple tasks including novel view synthesis under varying focal lengths, barrel & fish-eye distortion, and multi-modal RGB-thermal fusion. 
- The evaluation spans five datasets covering perspective imagery , fisheye cameras, and multi-modal setups, demonstrating breadth. The zero-shot generalization experiments (varying intrinsics, distorted inputs) are particularly valuable.

### Weaknesses
- **ablation study with marginal improvements**: Table 4 shows very small differences between ablation conditions (e.g., PSNR varies only 0.08-0.09 dB, SSIM differs by ~0.002-0.003). These marginal improvements raise questions about statistical significance of the reported differences and whether the proposed components genuinely contribute to performance.
- **Lack of interpretable analysis for learned frequencies**: Figure 1 shows position dimensions have larger rotation magnitudes than direction dimensions, but what does this mean conceptually? How does this relate to the semantic differences between position and direction?
- **Unclear practical motivation for RGB-thermal fusion**: While technically interesting, the paper doesn't establish compelling real-world applications for the joint RGB-thermal novel view synthesis. It would be beneficial to include real-world scenarios where joint RGB-thermal fusion would be valuable to help readers understand the practical significance of the method.

### Questions
- How is P = [t, d] computed for a patch? Is it the ray through the patch center? Average of all pixel rays?
- Why does standard RoPE produce biased attention that decays as position values diverge? How exactly does P = [t, -t+bias, d, -d+bias] "equalize the distance between two poses"?

### Soundness
3

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
This paper proposes a Rotary Ray Embedding (RoRE) scheme that encodes camera parameters and images directly as ray-based scene tokens. RoRE feeds rays (camera origin + direction derived from intrinsics/extrinsics) into a transformer and applies a ray-conditioned variant of Rotary Position Embedding (RoPE) so that the model can handle perspective, fisheye, and RGB–thermal inputs with the same architecture.

### Strengths
The motivation and the solution—encoding everything into the ray—are clear to me. Most concurrent or prior works either stay with token-level ray maps but only for pinhole cameras, use relative camera encodings like PRoPE but are not multimodal, or handle thermal/RGB but not arbitrary cameras. This paper attempts to do all three at once to justify generalizability via ray-level embedding. Tables 2 and 3 support this by providing tasks under different camera intrinsics and different types of inputs (distorted or fisheye cameras). Table 9 indicates that multi-view rendering across modalities is possible.

### Weaknesses
“Improved generalization and cross-modal consistency” is not fully reflected across all experiments. I would consider raising the score if the authors can make some experimental designs more convincing.

* In Table 4, the PSNR result basically says: “if one keeps APE and adds our RPE, the numbers don’t really move.” That slightly weakens the claim that the ray component alone is doing the work. Are there more contrastive results on challenging tasks (e.g., a fisheye dataset)?
* Table 5 and Figure 5 are not convincing to me regarding why multi-modal training succeeds. The authors admit there are no known prior works (L364) for this particular problem. However, it might be possible to compare with a fusion-based transformer to assess whether transfer from a single modality to a different modality is feasible. Currently, the result has a PSNR around 22, which is not particularly promising.
* Again in Table 9, I wonder if “multiple images” implies rendering multiple views in one pass. If so, what is the difference between rendering 8 images once and rendering 1 image eight times, apart from time? Would rendering multiple RGB views confer any advantage if the backbone is drawn from RoRE/LVSM/GTA/PRoPE?

### Questions
* In L141–L142, is $\mathbf{R}_m^d$ a typo? Should it be $\mathbf{R}_m^n$?
* Any intuition on why initializing $\theta$ with a uniform distribution (L214) leads to an exponentially decaying final frequency in Figure 1?
* In Figure 6, should the RoPE module be replaced with the RoRE module? Can the experimental setup in Table 6—comparing different positional-embedding schemes—be understood as a replacement of this RoPE (or RoRE) module?
* A depth loss is added in L732–L750. How is ground-truth depth at pixel $i$ obtained? Is it required during training? What happens if $\lambda_{\text{depth}} = 0$?

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
3

### Summary
This paper introduces a Rotary Ray Embedding approach that encodes image patches as rays to enhance multi-modal scene understanding. The proposed method demonstrates state-of-the-art performance and exhibits superior robustness under varying focal lengths and distorted input images

### Strengths
- The paper is well written and clearly organized.
- The proposed method achieves state-of-the-art performance and demonstrates superior robustness across challenging conditions.

### Weaknesses
1. Although the proposed method achieves state-of-the-art performance overall, it remains inferior to the concurrent work PRoPE in novel view synthesis results.
2. According to Table 4, all variants exhibit similar performance. In particular, the second variant and the full method achieve identical results across all three metrics, suggesting that the proposed learned frequencies may not contribute significantly.

### Questions
1. Although the proposed method achieves superior performance under varying focal lengths and distorted input settings, its performance on general novel view synthesis tasks is inferior to PRoPE. It is recommended to analyze the reasons behind this discrepancy.

### Soundness
3

### Presentation
3

### Contribution
3
