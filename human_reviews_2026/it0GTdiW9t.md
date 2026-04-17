# Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
Cross-modal image translation remains brittle and inefficient. Standard diffusion approaches often rely on a single, global linear transfer between domains. We find that this shortcut forces the sampler to traverse off-manifold, high-cost regions, inflating the correction burden and inviting semantic drift. We refer to this shared failure mode as fixed-schedule domain transfer. In this paper, we embed domain-shift dynamics directly into the generative process. Our model predicts a spatially varying mixing field at every reverse step and injects an explicit, target-consistent restoration term into the drift. This in-step guidance keeps large updates on-manifold and shifts the model’s role from global alignment to local residual correction. We provide a continuous-time formulation with an exact solution form and derive a practical first-order sampler that preserves marginal consistency. Empirically, across translation tasks in medical imaging, remote sensing, and electroluminescence semantic mapping, our framework improves structural fidelity and semantic consistency while converging in fewer denoising steps. The source code is in https://github.com/LaplaceLab/CDTSDE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new diffusion-based framework (CDTSDE) for cross-modality image translation. The core idea is to embed the domain shift directly into the generative process, instead of relying on a global linear blend between source and target images. The authors introduce a spatial-channel varying weight matrix for combining source and target paired data, which serves as the condition in diffusion models. The forward and reverse sampling processes are developed. Extensive experiments across several applications are conducted.

### Strengths
-  The paper gives a convincing geometric argument for why globally linear / fixed schedules send trajectories off-manifold, creating large correction burdens and semantic drift, motivating the method. 

- Instead of a single scalar $\eta_t$, $\Lambda_t$ is a learnable pixelwise, channelwise field with monotonic constraints and boundary clamping. This is novel and intuitively matches heterogeneous cross-modal shifts (texture, contrast, anatomy), and is also grounded by the theoretical analysis.

- Building on the pixelwise, channelwise weight $\Lambda_t$, the authors developed the corresponding forward and reverse processes, enabling the idea to be realistic.

- Experiments span three regimes of difficulty: (i) relatively mild contrast change (MRI T1↔T2), (ii) strong cross-sensor shift (SAR→optical), and (iii) extremely semantic mapping (electroluminescence image → defect mask). The proposed method shows convincing results across different tasks.

### Weaknesses
- The core formulation assumes access to paired (source, target) images during training. However, many cross-modality settings (especially SAR↔optical, certain medical domains) are weakly paired or unpaired in practice. The method’s dependence on paired data limits its applicability, but this is not deeply analyzed. 

- The presentation is mathematically dense. Some key definitions (e.g. how $\Lambda_t$ is actually predicted by the network in practice, how constraints like monotonicity in $t$ are enforced during training, and how the logistic squashing with $\epsilon$ is implemented during backprop) are split across sections and the appendix. This may make reproduction harder for non-experts, despite the promise of releasing code later. Meanwhile, more explanation on the definition equations of $\Lambda_t$ should be included to strengthen readability.

### Questions
- Your method conditions directly on paired $(\hat{x}_0, x_0)$ and uses $x_0$ explicitly in the adaptive mixture at training time. How sensitive is CDTSDE to imperfect pairing or slight misregistration, especially for SAR→optical and PSCDE where pixel alignment can be noisy? Have you tried training with synthetically perturbed / misaligned pairs to evaluate robustness? 

- Many real cross-modality scenarios do not provide paired data (e.g. historical SAR ↔ optical, multi-scanner MRI). Can your formulation be adapted to unpaired or weakly paired data, or is the approach fundamentally limited to paired supervision? Please clarify what breaks if $x_0$ is not available at training time, at least in discussion.

- You compare a global linear schedule $\eta_t$ vs. your spatially varying $\Lambda_t$ and report gains (e.g. Dice 0.46→0.49, Hausdorff 59.5→39.8 on PSCDE). Could you also report an intermediate baseline: a per-channel but spatially uniform schedule (i.e. $\Lambda_t$ depends on channel and $t$, but not $(p)$)? This would help isolate whether most of the benefit comes from spatial adaptivity or just from deviating from a single scalar $\eta_t$.

- Theorem 1 proves that allowing pixelwise $\Lambda(t)$ yields strictly lower path energy $E[d]$ than any global schedule under heterogeneity assumptions. How should we interpret this physically?   Is lower path energy empirically correlated with perceptual realism / fewer artifacts?  Do you ever observe cases where $\Lambda_t$ lowers energy but produces locally inconsistent textures (i.e. visually implausible mixing of source and target in the same region)?

- In which regimes does CDTSDE not help? For IXI (milder contrast shift), you mention improvements are “marginal.” Can you show qualitative counterexamples where CDTSDE produces artifacts (hallucinated structures, topology breaks, etc.) so we understand the limits of the method? 

I look forward to the response from the authors.

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
The paper introduces Cross-Domain Translation SDE (CDTSDE) for Cross-modal image-to-image translation. CDTSDE embeds an adaptive, spatially varying mixing field into the diffusion process. The forward marginal is centered on a per-pixel blend of source and target, and the reverse dynamics add a restoration drift aimed at keeping large steps near the data manifold. Reported results show improved PSNR/SSIM and PSCDE on several datasets.

### Strengths
– Across three modalities/datasets, the method reports better SSIM/PSNR and strong PSCDE structure metrics. 

– Concept of putting domain-shift adaptation inside the dynamics (rather than as external guidance) is principled.

### Weaknesses
– The paper is difficult to follow: many equations and notations are introduced without clear motivation or explanation of each component’s role, making the method hard to reconstruct; it focuses more on how things are done than why they are needed.

– The propose solution is fairly incremental since the core contribution is to propose an adaptive interpolation of $\hat{x}^{src}_0$ and $x_0$, replacing time-varying interpolation of source and target [1]

– Efficiency claims are supported largely by empirical numbers; there is little analysis explaining why the proposed dynamics/sampler should be more efficient.

– The setting effectively reduces to paired translation. A substantial line of prior work on diffusion bridge models already targets this regime and reports SOTA results (e.g., [2,3,4,5]), but these are neither discussed nor compared.

– Fixing a number of training steps and concluding CDTSDE is more efficient may be unfair: different methods have different designs/optimizers. The paper should provide training curves (and, ideally, wall-clock) to show faster convergence.

### Questions
– Which aspects of the mixing field matter most (spatial vs. channel, monotonicity, positional encoding)? Please include ablations. 

– How does the method compare directly to recent DBM baselines on the same splits? 

– What drives the reported efficiency—the restoration drift, the step schedule, or early stopping? 

– Please provide training curves for the method and baselines.

[1] Cui, et al. "Taming Diffusion Prior for Image Super-Resolution
with Domain Shift SDEs", NeurIPS 2024

[2] Zhou, et al. "Denoising Diffusion Bridge Models", ICLR 2024

[3] He, Guande, et al. "Consistency diffusion bridge models." NeurIPS 2024

[4] Liu, et al. “I$^{2}$SB: Image-to-Image Schrödinger Bridge”, ICML 2023

[5] Zheng, et al. “Diffusion Bridge Implicit Models”, ICLR 2025

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper “Adaptive Domain Shift in Diffusion Models via Latent-Space Normalization” proposes a training framework to mitigate the performance degradation of diffusion models under domain shift scenarios. The authors argue that existing fine-tuning or adapter-based methods fail to preserve generative quality when target-domain data are scarce or stylistically distinct. To address this, they introduce an adaptive latent-space normalization (ALSN) module that aligns latent statistics between source and target domains through learned normalization parameters, dynamically adjusted during training. Experiments on multiple domain adaptation benchmarks (Art→Photo, Synthetic→Real) show modest improvements in FID and CLIP similarity compared to baseline fine-tuning or LoRA-based adaptation.

### Strengths
+ Domain adaptation for diffusion models is a growing and challenging topic, and addressing domain shift in generative tasks is an important research direction, especially as diffusion models become widely deployed across diverse domains.
+ The proposed ALSN approach is computationally efficient and easy to integrate into existing diffusion pipelines, making it potentially attractive for practitioners seeking domain-robust generative models.

### Weaknesses
- The core idea, i.e., adjusting latent normalization statistics to align source and target distributions, closely resembles well-known techniques in domain adaptation. The paper primarily recontextualizes these ideas within diffusion models without offering substantial theoretical or methodological innovation. This limits the paper’s conceptual contribution.
- While results are shown, the paper does not convincingly explain why ALSN improves performance or how it interacts with diffusion timesteps and denoising dynamics. There is no analysis of latent trajectory behavior, feature drift, or stability to justify its effectiveness.
- Improvements in metrics such as FID or LPIPS are small (often within variance range), raising doubts about the real impact. The method’s simplicity is a strength, but it also highlights how incremental the advance is.

### Questions
Have you evaluated whether ALSN affects the diversity or mode coverage of generated samples, especially when applied across multiple distinct domains?

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
This paper introduces CDTSDE (Cross-Domain Translation SDE), a novel diffusion-based framework for cross-modality image translation. The key idea is to embed adaptive domain-shift dynamics directly into the diffusion process via a spatially varying mixing field that evolves throughout reverse-time sampling. This dynamic field replaces the conventional fixed linear interpolation between source and target domains, allowing geometry-aware, low-energy paths that stay closer to the data manifold. Extensive experiments on three benchmarks show consistent gains in various matrics.

### Strengths
1. The paper is well-motivated and conceptually clear. It generalizes traditional linear domain-shift formulations to a nonlinear, manifold-aware framework, supported by solid theoretical analysis and proofs.

2. The method is technically sound and demonstrates strong generality, making it straightforward to apply across different diffusion architectures and cross-modality image translation tasks.

3. Extensive experiments on three benchmarks IXI, Sentinel, and PSCDE show consistent gains in various metrics, outperforming GANs (Pix2Pix) and diffusion baselines (BBDM, DOSSR).

### Weaknesses
1. The method reduces to a linear domain-shift scheme when only a single diffusion step is used, implying that it still depends on multiple denoising steps for stable performance. This reliance could pose an efficiency limitation and hinder direct adaptation to flow-matching or single-step generative methods.

2. While the method shows clear conceptual advances, its quantitative improvements over DOSSR on the Sentinel and IXI datasets are relatively minor, indicating that the advantages may be limited in tasks with larger modality discrepancies.

### Questions
Please refer to weakenesses

### Soundness
3

### Presentation
3

### Contribution
3
