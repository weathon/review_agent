# Hyperspherical Latents Improve Continuous-Token Autoregressive Generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Autoregressive (AR) models are promising for image generation, yet continuous-token AR variants often trail latent diffusion and masked-generation models. The core issue is heterogeneous variance in VAE latents, which is amplified during AR decoding, especially under classifier-free guidance (CFG), and can cause variance collapse. We propose SphereAR to address this issue. Its core design is to constrain all AR inputs and outputs---including after CFG---to lie on a fixed-radius hypersphere (constant $\ell_2$ norm), leveraging hyperspherical VAEs.  Our theoretical analysis shows that hyperspherical constraint removes the scale component (the primary cause of variance collapse), thereby stabilizing AR decoding. Empirically, on ImageNet generation, SphereAR-H (943M) sets a new state of the art for AR models, achieving FID 1.34. Even at smaller scales, SphereAR-L (479M) reaches FID 1.54 and SphereAR-B (208M) reaches 1.92, matching or surpassing much larger baselines such as MAR-H (943M, 1.55) and VAR-d30 (2B, 1.92).
To our knowledge, this is the first time a pure next-token AR image generator with raster order surpasses diffusion and masked-generation models at comparable parameter scales.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes SphereAR, a continuous-token autoregressive image generator that prevents variance collapse by enforcing scale-invariant latents: a hyperspherical VAE constrains each token to a fixed-radius hypersphere, and all AR inputs/outputs—including after classifier-free guidance—are projected back to this sphere. A first-order analysis shows normalization removes radial errors, stabilizing multi-step AR decoding. Coupled with a causal Transformer and a token-level diffusion head, SphereAR sets impressive results on ImageNet 256×256: SphereAR-L achieves FID 1.54, surpassing diffusion and masked-generation baselines and matching MAR-H with ~1/2 the parameters. Ablations confirm hyperspherical latents outperform diagonal-Gaussian, fixed-variance, and Gaussian+post-hoc normalization, and that constant-norm refeeding in AR is the key driver of gains.

### Strengths
* This paper introduces a simple but powerful principle—enforcing constant-norm latents for continuous-token AR, including after CFG—to remove scale degrees of freedom. It provides a clean theoretical first-order analysis explaining why normalization eliminates radial error accumulation. Besides, it also distinguishes S-VAE from Gaussian+post-hoc normalization and shows why the former is better aligned with the objective.

* Strong empirical evidence on ImageNet 256×256 with clear SOTA FID at comparable or fewer parameters; comprehensive ablations that validate the effectiveness of each proposed component.

* Well-motivated problem statement tied to variance collapse under CFG; intuitive figures and precise method description; appendices with proofs and architectural details; explicit training and inference settings with reproducibility plan.

### Weaknesses
I think this paper does not contain significant weaknesses. I have a few suggestions and questions:
* It would be better to provide wall-clock (efficiency) comparisons vs MAR/VAR/DiT at matched FID.
* It would be interesting to see if the proposed method can improve the MAE paradigm as well.

### Questions
Please refer to the above section.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles a core limitation in continuous-token autoregressive (AR) image generation—variance collapse caused by heterogeneous latent scales during AR decoding, especially under classifier-free guidance. The authors propose **SphereAR**, which enforces a constant-norm hyperspherical latent space via a hyperspherical VAE (S-VAE) and ensures all AR inputs and outputs are projected back onto a fixed-radius hypersphere. The paper provides theoretical justification showing that normalization eliminates radial error propagation in AR decoding and supports it with extensive empirical evidence on ImageNet-256×256. SphereAR-L (479M) achieves FID 1.54, surpassing diffusion and masked-generation baselines of comparable or even larger parameter scales (e.g., MAR-H, VAR-d30, DiT-XL/2). Ablations further show that hyperspherical latent geometry consistently improves stability and generation quality compared to Gaussian posteriors or post-hoc normalization.

### Strengths
* **Clear motivation with strong theoretical insight.** The paper identifies *scale drift* as the fundamental issue in continuous-token AR and justifies why hyperspherical constraint solves it more elegantly than variance inflation or post-hoc normalization.
* **Significant empirical improvement.** SphereAR sets a new state-of-the-art among AR models and even outperforms strong diffusion and masked-generation baselines with fewer parameters.
* **Well-structured ablation study.** Experiments isolate the contribution of (i) spherical posterior vs Gaussian, (ii) normalization applied only at VAE decoder input vs AR loop, and (iii) post-hoc normalization vs intrinsic spherical latent modeling.
* **Technical contribution beyond engineering tweak.** Introducing hyperspherical VAEs into AR pipelines constitutes a genuinely novel design in continuous-token generative modeling.
* **Good reproducibility.** Training details, architectural configurations, and evaluation settings are consistently documented.

### Weaknesses
* **Limited analysis beyond ImageNet.** All experiments are class-conditional ImageNet generation at 256×256. It is unclear whether SphereAR scales to higher-resolution tasks or more challenging datasets such as LAION-based text-to-image.
* **AR decoding speed not fully addressed.** While normalization improves stability, AR inference remains autoregressive with diffusion steps per token, leading to slower sampling compared to parallel masked generation or next-scale diffusion models. A quantitative runtime comparison would be helpful.
* **Lack of negative results on extreme CFG regimes or resolution scaling.** The theoretical argument suggests strong stability, but empirical evaluation focuses on standard CFG ranges.
* **S-VAE training complexity.** Hyperspherical posteriors (even using Power Spherical) may increase training difficulty relative to standard diagonal-Gaussian VAEs; the paper acknowledges this but does not quantify the stability or convergence trade-off.

### Questions
Please refer to the "Weakness".

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents SphereAR, an autoregressive image generation framework designed to address the instability and performance gap of continuous-token AR models compared to diffusion and masked-generation methods. The key innovation is the use of hyperspherical variational autoencoders (S-VAE) to produce constant-norm latent tokens, enforced throughout AR decoding by projecting all inputs and outputs onto a fixed-radius hypersphere. Theoretical analysis links the scale-invariance to improved stability, and empirical results on ImageNet-1K demonstrate SphereAR’s superior FID and parameter efficiency versus strong baselines. The paper also includes extensive ablations and analyses of architectural choices.

### Strengths
1. Theoretical motivation is clear, with the scale-invariance argument spelled out through formal analysis.
2. Strong Empirical Results:  The clear advantage in both FID and parameter efficiency is validated across multiple regimes (base and large model).
3. Comprehensive Experimental Methodology: Empirical claims are supported by well-documented implementation details (Sec. 4.1 & Appendix C) and a reproducibility statement promising code and hyperparameter release.

### Weaknesses
1. Limited Discussion of Generalization and Failure Cases: The paper focuses almost exclusively on ImageNet-1K at $256 \times 256$ resolution (see Section 4 and qualitative figures in Appendix E). There is no analysis of generalization to other domains (e.g., other image datasets, non-class-conditional tasks, or higher resolutions), nor qualitative exploration of failure modes. This raises questions about the robustness and transferability of the claimed benefits.
2. Empirical Focus Narrowed to FID and Class-Conditional Generation: Table 1 and Figure 4 concentrate on FID, IS, Precision, and Recall. FID is known to have limitations (especially in capturing semantic consistency and diversity)
3. Ablation Duration and Robustness: The ablation experiments in Section 4.3 are run for a much shorter training regime than full-scale models (“50 epochs instead of 400”, cf. page 8). The methodology may not uncover longer-term stability issues or provide sufficiently representative results for convergence characteristics. Moreover, for Table 2 ("Component Contributions"; values not explicitly shown in the main text), further detail and numbers should be presented.

### Questions
Please provide responses to the issues raised in my Weaknesses section.

### Soundness
4

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
This paper points out that current autoregressive image generative models based on continuous tokens often performs unsatisfyingly. To tackle this problem, this paper proposes SphereAR. Specifically, the authors analyzes the importance of scale-invariant tokens for AR generation with continuous tokens. Based on this, the authors propose to replace the original VAE with S-VAE, which parameterizes each token on a unit sphere with the same norm. Then the authors combine the S-VAE with an autoregressive transformer and use the diffusion loss to train the model. Experiments show that SphereAR outperforms all mainstream AR and masked generation models with the same model parameter size.

### Strengths
1. The motivation is clear, and the analysis regarding the importance of keeping the scale of tokens the same is reasonable.
2. Using S-VAE is a straightforward yet effective modification.
3. The experimental results are impressive. The design of the ablation study is also reasonable.

### Weaknesses
Overall, this paper is quite good. I only have the following minor concern.
1. Since the authors use an AR architecture of transformer, which is different from MAR, it would be better to include the performance of combining this architecture with a standard VAE as a baseline, which is a fair comparison.

### Questions
1. The author mentions the different behaviors of masked models and autoregressive (AR) models when both use continuous tokens in Sec.3. I am curious about the underlying reason for this difference. In other words, what fundamentally distinguishes masked models from AR models with continuous token?

### Soundness
3

### Presentation
3

### Contribution
3
