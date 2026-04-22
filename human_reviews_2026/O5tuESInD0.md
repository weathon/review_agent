# On Equivariance and Fast Sampling in Video Diffusion Models Trained with Warped Noise

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Temporally consistent video-to-video generation is critical for applications such as style transfer and upsampling. In this paper, we provide a theoretical analysis of warped noise—a recently proposed technique for training video diffusion models—and show that pairing it with the standard denoising objective implicitly trains models to be equivariant to spatial transformations of the input noise. We term such models EquiVDM. This equivariance enables motion in the input noise to align naturally with motion in the generated video, yielding coherent, high-fidelity outputs without the need for specialized modules or auxiliary losses. A further advantage is sampling efficiency: EquiVDM achieves comparable or superior quality in far fewer sampling steps. When distilled into one-step student models, EquiVDM preserves equivariance and delivers stronger motion controllability and fidelity than distilled non-equivariant baselines. Across benchmarks, EquiVDM consistently outperforms prior methods in motion alignment, temporal consistency, and perceptual quality, while substantially lowering sampling cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper starts from an interesting theoretical observation that training with warped noise iduces equivariance to spatical transformations of the input, but suffers from significant flaws overall. Through the whole paper, I can not find any detail method description, no pseudo-code for this method. Its core theory is built upon warped noise methos proposed by others (see Line 152), and its main technical contribution (one-step distillation) utilizes a method similar to existing DMD method [1], with weak connection to the core equivariance theory. This paper has restated the highly related work done by Burgert et al. [2], however, there is no comparison with it.

[1] One-step Diffusion with Distribution Matching Distillation: https://arxiv.org/abs/2311.18828

[2] Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise : https://arxiv.org/pdf/2501.08331

### Strengths
1. Theoretical Clarity. Theorem 4.1 seems the most solid part of this paper. It clearly proves that under the use of warped noise and the standard denoising loss, the optimal solution naturally possesses equivariance. But there still exists some puzzles, in Eq.4, the authors utilize $\sum_k$, so why $\mathbf{V_t}$ is not written as $V^{(k)} + n^{(0)}$ ?

2. The visualization results seem good.

### Weaknesses
1. Lack of novelty. The core component, warped noise, is entirely derived from the work from others. This paper merely applies this existing method to the traininng of video diffusion models and proves a related theoretical property. This is a direct application, not a fundamental methodological innovation.

2. Missing comparison with related work. The paper repeatedly mentions the work of Burget et al. (Go-with-the-Flow), which also fine-tunes models using warped noise to improve motion control. This is a highly related and directly competitive work. However, in all experiments, this no comparison with this work. This severely weakens the paper's persuasiveness.

3. Vague method description, lack of reproducibility. This paper provides no pseudo-code, and I can not understand how the authors extracct the optical-flow, and how to use this to construct the wraped noise, and most critically, where is the definition of warping transfromation $\mathcal{T}_k$, how to calculate it? These implementation details are omitted, making the method nearly impossible to reproduce.

4. Is the paper's contribution proposing a new training framework, or merely proving a theoretical property? The content suggests the latter. Hope the authors can clarity the contribution of this paper.

### Questions
See the weakness.

### Soundness
3

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
5

### Summary
This paper proposes training video diffusion models with temporally warped noise to enforce inherent motion consistency and faster sampling. By replacing independent Gaussian noise with flow-warped noise during training, the model (termed EquiVDM) implicitly learns equivariance to spatial transformations of the noise, so that motion in the input noise translates into identical motion in the generated video without extra modules. The authors provide a theoretical proof (Theorem 4.1) that the standard denoising objective on warped noise yields an equivariant denoiser, aligning output frames with input motions. Empirically, EquiVDM achieves comparable or superior video quality with far fewer sampling steps, and a one-step distilled EquiVDM preserves this equivariance while improving motion fidelity over non-equivariant baselines. Comprehensive experiments across benchmarks demonstrate improved motion alignment, temporal coherence, and perceptual quality versus prior methods, supporting the paper’s claims

### Strengths
Theoretical insight: It provides a sound theoretical result – Theorem 4.1 – showing that using warped noise with the standard objective provably induces equivariance in the denoiser. This analysis is original and gives a clear explanation for why motion in input noise yields corresponding motion in outputs, solidifying the approach’s foundation.

Sampling efficiency: The study demonstrates faster sampling without quality loss. EquiVDM’s generation trajectory is much straighter (lower curvature) than the baseline’s, enabling high-fidelity videos in significantly fewer diffusion steps. 

Comprehensive evaluation: The experiments are thorough and insightful. The authors evaluate both text-to-video and video-to-video generation scenarios with rigorous metrics , and include ablations (e.g. varying β in Eq. 5) to justify design choices. These ablations confirm that adding a small independent-noise component (β≈0.9) balances quality and consistency, lending credence to the approach’s robustness.

### Weaknesses
Dependence on optical flow: The approach relies on accurate motion vectors/optical flow from a driving video to warp noise, which limits applicability when such motion cues are unavailable (e.g. pure text-to-video generation). A suggested improvement is to incorporate a learned or inferred motion prior (e.g. a flow predictor from text prompts) so that EquiVDM can extend to scenarios without ground-truth flow.

Scope of equivariance theory: The theoretical equivariance guarantee assumes every frame is a perfect spatial warp of the first frame, an ideal scenario that may not hold for complex motions (e.g. new occlusions or content changes). It would strengthen the work to discuss how equivariance might be approximated when real motion deviates from pure warping, or to introduce training constraints that handle partial equivariance in more general settings.

Background consistency trade-off: In some evaluations, EquiVDM shows a slightly lower background consistency than the best baseline (e.g. minor drops in BgC metric). This suggests a trade-off between enforcing equivariance and preserving background fidelity. Please analyze these cases specifically and provide a targeted analysis in the rebuttal

### Questions
The evaluation relies on CLIP scores for both video-to-video and text-to-video tasks. However, CLIP was primarily standardized for image–text alignment and, even when averaged over frames, does not directly assess temporal coherence or frame-level consistency. To more faithfully measure textual adherence in videos, please complement CLIP with a dedicated video–text alignment metric (e.g., UMTScore [1]).  This addition would provide a stronger, temporally aware assessment beyond image-centric CLIP scores.

[1] Liu Y, Li L, Ren S, et al. Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation[J]. Advances in Neural Information Processing Systems, 2023, 36: 62352-62387.

### Soundness
3

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
4

### Summary
This paper fine-tunes a pretrained video diffusion model using *warped (flow-correlated) noise* instead of i.i.d. Gaussian noise, claiming this induces equivariance and enables faster sampling. Experiments show improved motion alignment and fewer sampling steps.

### Strengths
1. The paper demonstrates that simple warping to the noise distribution yields measurable quality enhancement and speed-up.

2. The experiment results demonstrate improved temporal consistency when optical flow is available.

3. The implementation is compatible with existing video diffusion architectures.

### Weaknesses
1. **The paper overlooks the gap between the theoretical result and the behavior of real diffusion models.**

    Theorem 4.1 characterizes the Bayesian optimal denoiser under highly idealized assumptions (perfect warping, linear transform, noise consistency, no occlusion, etc.). 
    In practice, current video diffusion models do not contain any mechanism that enforces or even approximates $D_\\theta^{(k)}(V_t) = T_k \\circ D_\\theta^{(0)}(V_t)$, either in pixel space or latent space. 
    As a result, the model can trivially minimize the loss by memorizing and locally denoising the warped noise, instead of learning true equivariant behavior. 
    To substantiate the claim of “emergent equivariance,” I suggest that the authors directly test equivariance on synthetic warp operators (i.e., warp the noise, feed it to the model, and check whether the output matches the warped baseline). 
    From my view, evaluating only motion alignment in generated frames may not validate the equivariance of the denoiser itself.

2. **The practical constraints of the method outweigh its claimed benefits.**

    In my opinion, the model is likely to merely restrict the source distribution by injecting external motion information into the noise, rather than improve the denoiser’s intrinsic motion modeling ability. 
    In fact, a strong Gaussian denoiser is fine-tuned into a weaker, conditional denoiser that only works on a specially correlated noise distribution. 
    This intuitively weakens the model's generation capability of diverse/unseen motion. 
    To verify this, the authors should report performance under *OOD flow fields* or randomized flow noise.

3. **The method doesn't provide significant advantages over existing optical-flow-based guidance.**

   Unlike inference-time conditioning or plug-and-play guidance, the proposed modification permanently alters the model: It can no longer sample from standard i.i.d. Gaussian noise, nor function without optical flow. 
    The paper acknowledges this only briefly in the limitations section, but does not quantify the severity (e.g., failure modes when flow is unavailable, inaccurate, or out-of-domain). 
    No evaluation is provided on pure Gaussian noise, erroneous flow, or non-warpable videos, which undermines the generality of the proposed method.

### Questions
Please refer to the weaknesses.

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
5

### Summary
This paper addresses temporal consistency and sampling efficiency in video diffusion models, focusing on the effects of using warped noise during training. The authors theoretically prove (Theorem 4.1) that training with temporally warped noise induces equivariance to spatial transformations without the need for architectural modifications or auxiliary losses. They introduce EquiVDM, an equivariant video diffusion model, and propose a distribution-matching distillation method that trains fast, one-step student models while preserving motion controllability and fidelity. The approach is comprehensively evaluated and achieves compelling improvements in motion alignment, temporal coherence, and sample efficiency compared to baselines and prior state-of-the-art methods.

### Strengths
Theoretical Depth and Clarity: The paper delivers a clear theoretical foundation, most notably Theorem 4.1, which formally establishes that training diffusion models with warped noise enforces equivariance to warping transformations under the standard objective. The proof is sound and leverages concise, well-articulated mathematical reasoning (see Section 4.1).

Practical Applicability: Beyond theory, the paper details a straightforward recipe for integrating warped noise training into off-the-shelf state-of-the-art video diffusion architectures, including both UNet and transformer-based backbones. It is notable that no new architectural components or loss functions are introduced—a rarity for claims of improved temporal coherence.

Significant Empirical Results: Across several large-scale datasets (OpenVideo-1M, VidGen-1M, Youtube-VIS, MSRVTT), the empirical evaluation is thorough and multidimensional, covering FID, FVD, CLIP, cf-PSNR, ImQ, BgC, SubC, and others. In Table 1 and Table 2, EquiVDM consistently and substantially outperforms strong baselines such as VC2, Show-1, and state-of-the-art video control methods on both quality and temporal consistency metrics.

Fast Sampling/Distillation: The proposed distribution-matching distillation (DMD) enables one-step inference while retaining competitive performance (see Table 3 and Figure 8), with substantial quality preservation compared to traditional multi-step sampling. This is a substantial practical advantage.

### Weaknesses
The core approach requires high-quality motion vectors from optical flow estimation for every video, as outlined in Section 4.1 and acknowledged in the conclusion. This dependence on a non-trivial, error-prone subroutine introduces fragility: as observed in the paper (Section 4.2, Figure 2), errors or occlusions in optical flow estimation can break the required equivariance, especially in complex scenes with large motion or occlusions.

The authors themselves note—and Figure 2 demonstrates—that applying warped noise in latent space does not always yield the theoretically desired temporal consistency, due to unpredictable mappings from pixel to latent space and high-frequency encoder artifacts (Section 4.2). Although the proposed solution of injecting a small amount of independent noise (Equation 5) is empirically validated (Table 4), there is no theory provided to justify the optimality or generality of this heuristic. The tradeoff between motion consistency and noise manifold coverage remains a “hack.”

Theorem 4.1, while sound, makes idealized assumptions (all frames perfectly correspond under warping transformations and noise is perfectly warped). The paper acknowledges practical violations due to encoder drift (Section 4.2), occlusions, and imperfect flow, which dilutes the theorem’s generality. There is little quantitative measurement of how far real-world data/processes deviate from the theorem’s assumptions, and the model’s equivariance is not characterized beyond qualitative results.

Potential Drifting in Long Videos: As noted in the conclusion, introducing only warped noise is not sufficient to fully eliminate temporal drift over long sequences. No experiments are reported for video lengths well beyond those seen in training, and the issue of maintaining long-range consistency is largely unexplored. This is a significant omission, given the practical importance of long video synthesis.

Abrupt Handling of Non-Available Motion Priors: The method depends on motion priors extracted from input (“driving”) videos, which are unavailable in text-to-video or generative settings without external cues. The discussion around synthesizing plausible flow from text prompts is surface-level (Section 6), and the approach is currently limited to video-to-video scenarios.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
