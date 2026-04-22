# SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Visual autoregressive (VAR) models generate images through next-scale prediction, naturally achieving coarse-to-fine, fast, high-fidelity synthesis mirroring human perception. In practice, this hierarchy can drift at inference time, as limited capacity and accumulated error cause the model to deviate from its coarse-to-fine nature. We revisit this limitation from an information-theoretic perspective and deduce that ensuring each scale contributes high-frequency content not explained by earlier scales mitigates the train–inference discrepancy. With this insight, we propose Scaled Spatial Guidance (SSG), training-free, inference-time guidance that steers generation toward the intended hierarchy while maintaining global coherence. SSG emphasizes target high-frequency signals, defined as the semantic residual, isolated from a coarser prior. To obtain this prior, we leverage a principled frequency-domain procedure, Discrete Spatial Enhancement (DSE), which is devised to sharpen and better isolate the semantic residual through frequency-aware construction. SSG applies broadly across VAR models leveraging discrete visual tokens, regardless of tokenization design or conditioning modality. Experiments demonstrate SSG yields consistent gains in fidelity and diversity while preserving low latency, revealing untapped efficiency in coarse-to-fine image generation. Code is available at https://github.com/Youngwoo-git/SSG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a training-free inference guidance method for Visual Autoregressive Generative Models (VAR), called Scaled Spatial Guidance (SSG). This method aims to address the problem of VAR deviating from the coarse-to-fine hierarchical generation pattern during inference. From an information theory perspective, the authors point out that each generation scale should only supplement high-frequency semantic residuals not covered by the previous scale. To this end, SSG guides the model's logits during inference by calculating residuals with the frequency domain prior constructed in the previous step and amplifying high-frequency components. Experiments show that SSG significantly improves image sharpness and structural consistency while maintaining the same inference speed.

### Strengths
1. The observations in this paper are highly relevant to the proposed method and make sense.

2. The proposed Discrete Spatial Enhancement effectively addresses the information loss caused by linear interpolation.

3. Comprehensive experimental data show that SSG can consistently improve the generation performance of various variable models without compromising the generation speed.

4. Ablation experiments validates the effectiveness of the proposed method.

### Weaknesses
1. Most of the experimental results in this paper come from class-conditional image generation. I believe that including more experimental results from text-to-image VAR models as the main experimental content would better demonstrate the practical value of the method. For example, adding more visual comparisons on infinity models to illustrate the visual improvement of the proposed method, or further expanding Table 4.

2. The proposed method seems to be applicable only to next-scale prediction.

### Questions
Please see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a simple, training-free guidance trick for next-scale visual autoregressive (VAR) models called \emph{Scaled Spatial Guidance} (SSG). At sampling step $k$, the method constructs a coarse ``prior'' by upsampling the previous step's logits via a frequency-domain procedure (\emph{Discrete Spatial Enhancement}, DSE), and then applies a closed-form logit update
\[
\ell^{\text{SSG}}_{k} = \ell_{k} + \beta_{k}\bigl(\ell_{k} - \ell_{\text{prior}}\bigr),
\]
intended to emphasize the predicted high-frequency residual while keeping the coarse structure aligned with the autoregressive hierarchy. The approach requires no retraining, adds negligible overhead, and is reported to improve FID/IS across several VAR baselines and tokenizations on ImageNet (256/512) and MJHQ-30K, without increasing the number of sampling steps.


While practical and easy to bolt on, the novelty feels incremental relative to existing guidance/contrastive sharpening ideas (e.g., CFG-style logit shaping, spectral emphasis, residual boosting). The theoretical framing relies on strong heuristics---coarse-state Markov assumptions and a clean low/high-frequency separation in logit space---that may not hold for learned decoders; likewise, DSE implicitly assumes band-independence that is only approximate. The method appears sensitive to the quality of the transported prior and the schedule $\{\beta_k\}$ (and temperature), yet the paper offers limited analysis of failure modes (e.g., misaligned priors), robustness, or hyperparameter stability. Empirical scope is narrow (primarily ImageNet and a single T2I set), comparisons depend on in-house reruns under one protocol (raising fairness/reproducibility concerns), and there is no statistical testing, human preference evaluation, or detailed profiling of latency/memory across resolutions/hardware. The technique is appealing as an inference-time tweak, but stronger evidence (broader datasets, stress tests on compositional/OOD cases, ablations on DSE variants, instability/degeneration analyses, and transparent baseline parity) would be needed to elevate its contribution.

### Strengths
SSG is a one-line logit update with a clear algorithmic recipe (Alg. 1–2) and no retraining or architectural changes; it operates directly on residual logits and drops into a wide range of VAR-style decoders.

### Weaknesses
SSG feels close to existing guidance/contrastive logit-shaping (e.g., CFG-like sharpening, residual boosting), with limited conceptual leap.

The motivation for DSE/SSG leans on orthonormal transforms yielding “independent and non-interfering” bands, enabling “exact, lossless reconstruction” and a clean separation of low/high frequencies for prior construction. That’s a strong assumption in learned logit spaces and likely violated by aliasing, context coupling, and tokenization quirks; the paper doesn’t test how departures from this assumption affect outcomes.

Ablations show nearest/linear spatial priors worsen both FID and IS versus baseline (no SSG). This suggests SSG can hurt quality unless the frequency-domain prior is used, narrowing practical robustness and increasing configuration risk. Yet the paper doesn’t map when a user might inadvertently choose a weaker prior (e.g., for speed).

The method explicitly depends on the transported/upsampled prior. Misalignment (e.g., spatial drift, aliasing, or tokenization artifacts) can produce oversharpening, texture hallucinations, or suppression of semantically important low-frequency structure.

### Questions
In Fig. 4(b), did you sweep identical temperature grids for baseline and +SSG, and report matched points with CIs over multiple seeds? If not, can you replot with a shared grid and error bars to substantiate the “consistently better FID–IS trade-off” claim?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Scaled Spatial Guidance (SSG) — a training-free, inference-time enhancement for visual autoregressive (VAR) models. VAR models generate images via next-scale prediction in a coarse-to-fine hierarchy but often suffer from train–inference discrepancies, where fine scales redundantly reconstruct coarse details instead of adding new high-frequency content. SSG addresses this by guiding each generation step to focus on the semantic residual—the high-frequency information not captured by previous scales. This is achieved through Discrete Spatial Enhancement (DSE), a frequency-domain prior construction that preserves coarse structure while isolating finer details. Applied directly to logits during inference, SSG improves image fidelity and diversity across multiple VAR baselines (VAR-d16 to VAR-d36), achieving up to 11.5% FID improvement at 512×512 resolution with negligible latency.

### Strengths
1. SSG works purely at inference on logits, requiring no retraining, fine-tuning, or architectural changes.

2. The method is derived from an information bottleneck perspective, offering a principled justification for why emphasizing high-frequency residuals improves coarse-to-fine generation.

3. The frequency-domain DSE module is simple and elegant.

4. Performance is good. SSG consistently improves VAR across various model sizes and input resolution, with only negelectable cost.

### Weaknesses
1. The proposed method is closely tied to the coarse-to-fine next-scale structure of VAR models. While this focus is well-motivated, it somewhat limits the generality of the contribution. It remains unclear whether SSG can extend to broader autoregressive or diffusion-based generation frameworks. Discussing how the information-theoretic insights or the frequency-domain prior could inspire guidance mechanisms beyond VAR would strengthen the paper’s broader impact.

2. Most examples highlight success cases where SSG clearly improves detail. Including a few failure or neutral cases could offer a more balanced understanding of when the guidance might be less effective.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a training-free guidance mechanism for Visual Autoregressive (VAR) models by leveraging the inherent scale-structured generation order. Inspired by information bottleneck principles, the authors introduce Scaled Spatial Guidance (SSG), which adjusts token generation dynamics across scales to correct distortions and enforce structure. Experiments across multiple architectures and benchmarks demonstrate consistent improvements in synthesis quality, suggesting that scale-aware inference adjustments can serve as an effective plug-and-play enhancement for VAR models.

### Strengths
- The paper identifies that the generation order of VAR models implicitly forms a scale-guidance structure and introduces a training-free guidance method that exploits this property.
- The proposed approach is grounded in the information bottleneck perspective, and the authors empirically support that the method can correct distorted signals.
- Extensive experiments across various models and datasets demonstrate consistent improvements, highlighting the robustness and generality of the method.

### Weaknesses
- While the method appears generic, the presentation is heavily specialized for VAR, making the broader applicability unclear.
- The approach, although training-free, feels ad-hoc and may not fundamentally address scalability and representation issues in visual tokenization; it would strengthen the contribution to explore how SSG could be incorporated into tokenization or model design directly.
- The abstract emphasizes improving high-frequency details, but the validation for this claim is limited and not rigorously demonstrated.

### Questions
**On generalization of the method**
Could the proposed method extend beyond VAR to other hierarchical generative frameworks such as diffusion guidance or hierarchical autoregressive models in below? If not directly, what modifications might be required?
- Parallel Multiscale Autoregressive Density Estimation, ICML 18
- Generating Diverse High-Fidelity Images with VQ-VAE-2, NeurIPS 19
- Locally Hierarchical Auto-Regressive Modeling for Image Generation, NeurIPS 22

**On frequency-domain evidence**
The paper claims enhanced high-frequency fidelity (e.g., Fig. 7), but the evidence is mostly visual. It would be beneficial to provide spectral analysis:
- Compare the average spectral amplitude of the dataset vs. VAR w/o SSG vs. VAR+SSG.
- Test Gaussian blur sensitivity: if SSG enhances high-frequency components, performance should degrade more under blurring.
- The paper On the Frequency Bias of Generative Models (NeurIPS'21) may serve as a useful analytical framework.
- Adding spectrum visualizations alongside Fig. 8/9/10 could provide direct support.

**On Appendix L temperature settings**
Appendix L reports different temperature ranges for baseline w/ and w/o SSG (0.5–1.2 vs. 0.7–1.5). Would it be fairer to evaluate both methods over the same temperature range (e.g., 0.5–1.5) and report best results or display curves for each? Since temperature sweeps were performed, sharing plots would clarify any performance variance across sampling strengths.

### Soundness
3

### Presentation
3

### Contribution
3
