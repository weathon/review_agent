# DAWN: Dual Space Regeneration Attack

- Decision: Reject
- Scores: 2, 8, 2, 2

## Abstract
The growing use of generative models has intensified the need for watermarking methods that ensure content attribution and provenance. While recent semantic watermarking schemes improve robustness by embedding signals in latent or frequency representations, we show they remain vulnerable even under resource constrained adversarial settings. We present \textsc{DAWN}, a training-free, single-image attack that removes or weakens watermarks without access to the underlying model. By projecting watermarked images onto natural priors across complementary representations, \textsc{DAWN} suppresses watermark signals while preserving visual fidelity. Experiments across diverse watermarking schemes demonstrate that our approach consistently reduces watermark detectability, revealing fundamental weaknesses in current designs. Our code is available at \url{https://anonymous.4open.science/r/DAWN-567A/}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes DAWN, a training-free, single-image, model-agnostic watermark removal attack that sequentially (1) performs frequency-domain reconstruction, (2) applies diffusion-based semantic refinement, and (3) matches channel-wise mean/variance for color/tone correction. Experiments across pixel/frequency/latent watermark families report high attack success, though the authors acknowledge chroma/hue shifts.

### Strengths
- __Practical adversarial setting__. The no-box, single-image threat model is realistic and clearly stated, and the pipeline is simple, fast (single pass), and model-agnostic.
- __Clear pipeline design with insights motivated__. The paper articulates the role of each stage (spectral suppression → semantic restoration → color alignment), with an explicit color-correction formula.

### Weaknesses
- __Perceptual quality degradation__ (visible color/tone shift).    
Although the paper claims “perceptual and semantic consistency,” the qualitative figures indicate noticeable hue shifts; the authors themselves note that luminance is preserved while chroma deviates. In an attack intended for usable images, this level of color drift is a material drawback. A user seeking to remove a watermark typically still wants to use the resulting image. Current evidence suggests DAWN often achieves success by sacrificing chromatic fidelity (the images appear noticeably “purpler” in multiple visualizations). I recommend reporting color-sensitive metrics in addition to PSNR/SSIM/LPIPS. Further, a user-study on color acceptability (or thresholds) would make the “perceptual fidelity” claim more persuasive.

- __Section §3 hypothesis is not fairly tested.__    
The stated hypothesis is that frequency-based reconstruction is more effective than pixel-only regeneration at weakening frequency-domain watermarks. However, the current experiment demonstrates higher success at the cost of worse perceptual quality (higher LPIPS, lower CLIP similarity), which is expected—stronger distortion can trivially improve removal. To validate the hypothesis fairly, match image quality across methods (e.g., tune the diffusion regeneration strength/steps and the frequency UNet noise/mask until PSNR/LPIPS/ΔE are aligned), then compare detector p-values. Otherwise the conclusion conflates attack strength with tolerated degradation.

- __Novelty is incremental/assembly-style.__    
The approach is largely a sequential combination of a known frequency-space denoising/reconstruction, a standard img2img refinement, and simple channel-wise normalization. The paper’s primary contribution is empirical: showing this particular stacking is effective under the single-image threat model. The methodological novelty is modest.

- __Evaluation design favors DAWN via luminance-heavy reporting.__    
Success is reported alongside PSNR/SSIM/LPIPS and CLIP, but then SSIM_lum and CLIP_lum are emphasized—metrics that explicitly downweight color errors. This can systematically under-report DAWN’s most visible artifact (hue shift). Fairness requires quality-matched comparisons across attacks and inclusion of color-perception metrics

- __Baselines not run at matched quality.__   
Some baselines (e.g., semantic regeneration or “imprint-removal”) could plausibly improve success if allowed to trade perceptual quality for removal. The paper should retune competing attacks to reach comparable PSNR/LPIPS, then compare success rates, exactly as recommended for §3. This will clarify whether DAWN’s advantage persists when the “cost” (quality loss) is controlled. For instance, for SemRefine, in Zodiac Table 15, the authors tune the regeneration steps to controll the attack strength.

### Questions
Please refer to the Weakness part.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces DAWN, a training-free, single-image, model-agnostic attack that removes or weakens generative watermarks by sequentially projecting watermarked images onto natural priors across frequency and semantic domains. DAWN achieves >95% success on classical pixel/frequency watermarking schemes and 70–90% on latent-space methods (TREE-RING, ZODIAC) while maintaining high perceptual fidelity. The paper highlights structural vulnerabilities of current watermarking approaches.

### Strengths
* The paper introduces a novel attack formulation by introducing a practical single-image, no-box adversarial setting rarely explored in watermarking research (although related work and references could be extended, see below)
* The method appears to be simple and generalisable; it's training-free at inference and adaptable across watermark types and domains
* The benchmark is comprehensive, indlucing six waterarking schemes 
* Overall, the paper is clear and raises important concerns for the robustness of watermarking

### Weaknesses
* Related work and references are limited and should be extended to contextualise the work
* The evaluations rely on stable diffusion-based backbones and it's unclear how it generalises to other architectures
* While “training-free,” the method still relies on access to large pretrained generative models
* A bit more depth of publishing watermarking-removal pipelines would be appreciated

### Questions
* How does DAWN perform on more recent watermarking systems that interleave multiple frequency bands or use cryptographic verification?
* Could DAWN be adapted for video or multimodal watermarking schemes?
* What safeguards do the authors suggest for responsible disclosure or controlled release of such attacks?

### Soundness
3

### Presentation
2

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
This paper introduces DAWN, a single-image, training-free attack designed to remove watermarks from images. The method works without access to the generative model or binary messages. It combines three stages: (1) a frequency-domain UNet to suppress spectral artifacts, (2) a diffusion-based semantic refinement (img2img) to restore image structure, and (3) a perceptual color correction step to match the original image's statistics.

### Strengths
S1. Realistic Threat Model: The attack setup is practical and compelling. It operates under a highly constrained, realistic threat model: it is training-free (at inference time), model-agnostic, and only requires a single watermarked image.

### Weaknesses
W1. Missing experimental details: Key experimental details are ambiguous. In Section 3, the paper describes "a single pass of SD-v2 img2img". It is unclear if this refers to using only the VAE autoencoder for reconstruction or applying a full diffusion-purification step. If it's the latter, the noise level and diffusion parameters are not specified.

W2. Motivating experiment: The experiment in Section 3, which motivates the entire approach, is not very convincing. It claims the frequency-domain UNet is more effective at watermark removal than the diffusion model. However, it also reports that the UNet's output has significantly worse perceptual quality (LPIPS 0.53 vs. 0.10). The improved "removal" is likely just a byproduct of greater image degradation. A fair comparison would require evaluating both methods at a fixed level of perceptual quality.

W3. Qualitative Results: The attack severely degrades image quality, rendering the "attacked" images unusable. The qualitative results in Figures 1, 4, and 6 show extreme color artifacts (strong purple and green tints). This is supported by the quantitative metrics in Table 2, which report PSNR values as low as 14.56 for the SDP dataset, which is very low.

W4. Weak WM Baselines: The attack is primarily evaluated against weak or outdated watermarking methods (e.g., DWTDCT, DWTDCTSVD, Riavgan, SSL). More robust, state-of-the-art methods (e.g., TrustMark, Invismark, WAM) are not included.

W5. Unfair baseline for attacks: The comparison to baseline attacks (imprint-removal, regeneration-based) in Section 6.2  is incomplete. The paper reports their attack success (Fig 3)  but omits their corresponding perceptual quality metrics (PSNR, LPIPS, etc.). This makes it impossible to evaluate the trade-off between removal success and image fidelity, which is the key metric for any attack.

Minor. Missing citations:
- https://arxiv.org/abs/2310.07726
- https://proceedings.neurips.cc/paper_files/paper/2024/file/67b2e2e895380fa6acd537c2894e490e-Paper-Conference.pdf

### Questions
Q1: The paper states in Section 3 and Section 5  that it "embed[s] TREE-RING watermarks" or "appl[ies] the target watermarking schemes" to existing clean images. However, Treering is an in-generation watermark that cannot be applied post-hoc. How was this implemented?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes **DAWN**, a training-free, single-image, model-agnostic attack that aims to suppress semantic and frequency-space watermarks by sequentially applying (1) frequency-domain reconstruction, (2) diffusion-based semantic refinement, and (3) tone/color correction. 
Experiments target multiple watermarking schemes and report relatively high attack success rate at the price of low image perceptual quanlity.

### Strengths
- The paper addresses a timely and relevant problem, that is practical single-image attacks on modern semantic/frequency watermarks with a clear threat model. 
- The two design principles (frequency-space projection for suppression; diffusion for semantic recovery) are motivated by an analysis highlighting the limits of pixel-only regeneration.

### Weaknesses
* **Missing trade-off reporting.** Section 3 argues frequency-domain reconstructions suppress watermarks more effectively than pixel-based diffusion but at the cost of **perceptual quality**, which is a widely recognized trade-off in the watermark-attack literature. However, the experiments do not explicitly present **paired** trade-off curves/tables (e.g., **PSNR/LPIPS vs. attack success rate**) across methods to quantify this. This makes it hard to judge whether DAWN’s higher success is achieved at an acceptable perceptual cost. 
* **Perceptual degradation.** Even ignoring the trade-off framing, the reported image-quality are low for semantic watermarks (e.g., PSNR ≈ 14–16 dB on TREE-RING/ZODIAC), and visualizations show notable color-tone shifts relative to the originals. This suggests the method’s higher success rate is obtained by **substantial** degradation, weakening the fairness of comparisons that claim superiority over baselines without controlling quality levels. 
* **Visualization intent unclear.** The **luminance** comparisons in Figure 4 are not clearly tied to conclusions; moreover, DAWN does not appear consistently better than baselines in luminance-based views, and the qualitative benefit of including these panels is ambiguous. 
* **Ambiguity about “single forward pass”** The paper states it uses a *single forward pass* and “takes couple of seconds,” yet Step 2 is **diffusion-based semantic refinement** (img2img), which typically entails multiple denoising steps. The intended definition of “forward pass” is unclear, and end-to-end compute vs. imprint-removal is not reported in directly comparable units. 
* **Limited originality.** The three stages—frequency-domain reconstruction, diffusion regeneration, and simple channel-wise color statistics matching, are each close to existing methods; the novelty is modest in algorithmic terms.

### Questions
1. Can you report **paired** PSNR/LPIPS vs. **attack-success** curves (or Pareto tables) across all compared attacks to enable *quality-controlled* comparisons? This would address the concern that higher success may come from larger perceptual changes.
2. What exactly constitutes one “forward pass” in DAWN? How many diffusion steps are run in Step 2?

### Soundness
2

### Presentation
2

### Contribution
1
