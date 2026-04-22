# MAD: Manifold Attracted Diffusion

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Score-based diffusion models are a highly effective method for generating samples from a distribution of images. We consider scenarios where the training data comes from a noisy version of the target distribution, and present an efficiently implementable modification of the inference procedure to generate noiseless samples.
Our approach is motivated by the manifold hypothesis, according to which meaningful data is concentrated around some low-dimensional manifold of a high-dimensional ambient space. The central idea is that noise manifests as low magnitude variation in off-manifold directions in contrast to the relevant variation of the desired distribution which is mostly confined to on-manifold directions. We introduce the notion of an extended score and show that, in a simplified setting, it can be used to reduce small variations to zero, while leaving large variations mostly unchanged. We describe how its approximation can be computed efficiently from an approximation to the standard score and demonstrate its efficacy on toy problems, synthetic data, and real data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents MAD (Manifold Attracted Diffusion) which is designed for denoising and structure recovery when diffusion models are trained on corrupted or noisy datasets. It modifies the inference dynamics to pull samples toward the data manifold, suppressing off-manifold noise without retraining the model.
Standard diffusion sampling assumes clean training data and tends to reproduce training noise or artifacts. MAD fills this gap by providing an inference-time correction that enforces manifold consistency, enabling cleaner reconstructions from noisy-trained or imperfect models.
The ideas presented here are interesting conceptually, however the application of generation from noisy data training is not that useful in practice. Moreover, the experiments are mainly qualitative with not enough quantitative evaluations.

### Strengths
1.	Clear theoretical formulation. The “extended score” is well-defined and mathematically consistent.
2.	Simple, practical implementation. Requires only minor modifications to inference and reuses pretrained score networks.

### Weaknesses
1. The method is motivated by training on corrupted or noisy data, yet no controlled experiment on real data demonstrate this regime. As a result, the practical need and benefit of MAD are not motivated enough and it appears to fix a problem that the paper never clearly shows.
2. Image degradation at higher attraction strength. Increasing b reduces details and realism, questioning the method’s practicality for general image generation. Low b is equivalient to the regular score and as b increases image quality downgrades.
3. Minor related work references. The paper lacks a structured comparison to related inference-time or manifold-based diffusion approaches. Even if there is no method on this exact task, it would be beneficial to appropriately address other denoising methods. A comparison to some methods in the results section is also suggested (where denoising is illustrated, e.g. Fig 7).
4. No quantitative evaluation. Results remain purely qualitative, some emasures such as FID can be used to evaluate image realness.
5. Minor comment on figures - Figure 1 – plot titles should move to be y-axis labels. Ticks should be larger, it is very hard to read. Ticks are also very small in Figure 2, and values are not clear enough there.
6. No code is provided, hindering future comparisons.
7. The authors mention traditional manifold denoising methods but do not explain why they are irrelevant.
8. Persistent scanning artifacts or logos/watermarks seem like possible sources for noise that need removal in practice. The authors themselves mention compression artifacts. However, there is no examination of the presented method in such contexts.

### Questions
1. The paper claims to address training on corrupted or noisy data, but no controlled experiment on real data actually shows this. Can the authors provide concrete examples or benchmarks where this problem occurs in practice?
2. Increasing the manifold-attraction strength (b) consistently reduces image quality (Figs 3-5). Can the authors explain in what practical scenario the method should be used? Another question here – Section 5.2 and Figs 3-5 seem not relevant to the method (these datasets are noiseless as stated by the authors), why are they presented in the main body, and in such extent? Maybe removing or reducing the space of this part can leave some space for more practical use-cases.
3. Why is there no dedicated “Related Work” section comparing to inference-time denoising, posterior correction, or manifold-regularized diffusion methods?
4. Beyond visual comparisons, can the authors quantify improvement (e.g., denoising metrics, perceptual quality, FID) to justify the practical value of the approach?

### Soundness
2

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
3

### Summary
This paper extends the standard score operator via
$
p \mapsto (1+\gamma) S(p * g_\gamma) + \frac{d}{d\gamma} S(p * g_\gamma),
$
where $S$ is the usual score and $g_\gamma$ is Gaussian smoothing. As $\gamma \to 0$, the extended score reduces to the standard score (when properly defined). This extension accommodates distributions without proper densities (e.g., multi-Dirac measures), partially addressing score blow-ups (Dirac/multi-Dirac scores scale like $1/\sigma$ and diverge).

The core intuition is that the extended score tends to **shrink low-variance directions**. For example, the standard score of an isotropic Gaussian coincides with the extended score of a Dirac. Sampling with the extended score may therefore reduce small variances that, under a manifold hypothesis, correspond to off-manifold noise/deviations in images. In this way, using the extended score implicitly enforces a manifold constraint during sampling—**Manifold Attracted Diffusion (MAD)**.

The extended score can be computed from a learned (standard) score. The authors leverage the EDM setup, select $\gamma(t)=\sigma(t)^p$ (with $\sigma(t)$ the noise level), and propose corresponding correction coefficients and sampler features. Visualizations on toy data, real image datasets (FFHQ, AFHQv2, and ImageNet), and scientific data show a variance-reduction/thresholding effect and suggest potential for producing cleaner images even when training data are corrupted.

### Strengths
1. The mathematical presentation is rigorous and instructive. Extending the predominant score operator is an interesting and timely idea.
2. The figures (esp. Figs. 1–2) are clear and intuitive. Experiments span synthetic, natural image, and scientific data, with seemingly stable and reasonable generations.

### Weaknesses
1. The justification for sampling with the extended score is not explicit. Beyond the Dirac case (line 216-225), the effects are mainly illustrated empirically, and the image-domain interpretations feel somewhat loose. The writing in these parts can be elusive.
2. Related to (1), the roles of the hyperparameters $a$ and $b$ are not explained intuitively. Empirically, increasing $a$ appears to reduce $\gamma(t)$, pulling the extended score closer to the standard one (weakening MAD), while $b$ affects both $\gamma$ and the correction term $m$, and in practice seems to strengthen MAD. This likely necessitates a hyperparameter search, which hurts applicability.
3. MAD roughly doubles computational overhead at inference.

### Questions
1. The sampler requires a derivative with respect to time/noise level. Some recent work shows the denoiser can sometimes omit explicit time conditioning [1]. Would MAD still be applicable in such settings, or does it fundamentally require time-dependent scores?
2. In Fig. 3 (rows 3–4), varying $b$ appears to modulate generation non-monotonically (e.g., a man → woman → man). Do the authors have an interpretation for this zig-zag behavior?

[1] Sun, Qiao, et al. *Is Noise Conditioning Necessary for Denoising Generative Models?* ICML 2025.

*My current rating is not a final assessment, as I am not an expert on diffusion model sampling. I may revise it after discussion with the authors and other reviewers.*

### Soundness
2

### Presentation
3

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
This paper proposes MAD, an inference algorithm for diffusion generation under the setting where the diffusion model was trained on noisy data, but at inference time one wants to generate clean data. The proposed algorithm is based on the intuition that meaningful semantics lie on a lower dimensional manifold, while noise is orthogonal off the manifold. Based on this, at each denoising step, the proposed algorithm calculates the derivative of the score with respect to time, and updates the image using a weighted sum of score and the derivative. The paper provides motivational and intuitive analysis on synthetic settings and on applying the algorithm on diffusion models trained on clean real-world data, and performs denoising experiments on the intended setting on a synthetic dataset and a real dataset EMPIAR-11618.

### Strengths
* This paper looks at an interesting setting of obtaining clean generations from a diffusion model trained on noisy data, and proposes an inference only method that does not require retraining. 
* The proposed method is technically sound and theoretically grounded.
* The illustrative examples and generations on FFHQ, AFHQV2, and ImageNet help gain intuition of the proposed algorithm's behavior.
* Qualitative visualizations on the synthetic data and the EMPIAR-11618 dataset look promising.

### Weaknesses
* The experimental evaluations are limited, relying primarily on qualitative visualizations. For the denoising experiments, it would be helpful to include quantitative metrics, such as FID scores computed against the clean data distribution.
* The two datasets in the denoising experiments have very simple semantics, making it hard to assess the scope of general applicability of the proposed method. It would be helpful to provide evaluations on more complex datasets,  e.g. CIFAR or ImageNet. Although the visualizations in 5.2 involve complex data, they are not a direct application of MAD’s intended setting but used diffusion models trained on clean data instead. 
* The performance appears sensitive to the hyperparameters, and there is no obvious rule-of-thumb across different datasets. This could introduce non-trivial hyperparameter sweep overhead when applying MAD in practice, especially if, e.g. the dataset is complex and involve various sub-categories. 
* It would be helpful to provide some evaluations and analysis on the inference speed and potential speed-performance tradeoffs. By default, each denoising step of MAD invokes two forward passes of the scoring network, which could raise concerns about significant computational overhead. It would be helpful to see, e.g. if MAD can work well with sampling speedup strategies (e.g. the simplest one could be using fewer sampling steps).

### Questions
* It would be helpful to see quantitative evaluations, such as FID with the clean data
* It would be helpful to see more discussion or analysis on the scope and applicability of the proposed method:
    * How well does it scale with dataset complexity, e.g. for datasets that go beyond simple semantics and strong inductive bias like CIFAR, ImageNet?
    * How does the performance look like when there are different levels of corruption/noise in the training data?
    * What kinds of corruption is the method effective against? Is it primarily designed for general noise, or does it also work for other corruption types such as blurring, compression artifacts, or other distortions? e.g. in the synthetic setting, would the method still be applicable if only the first step of blurry blobs was applied without the second step of adding Gaussian noise?
    * It would be useful to have some discussion or analysis on contextualizing the performance of this inference-time approach relative to training-time methods or finetuning with small amount of clean data methods. Is the proposed method's performance significantly worse, indicating a performance-compute tradeoff, or is it comparable? Additionally, a potential relevant inference-time baseline to consider for comparison is truncated sampling [1].

[1] Daras, Giannis, Yeshwanth Cherapanamjeri, and Constantinos Daskalakis. "How much is a noisy image worth? data scaling laws for ambient diffusion." ICLR 2025.

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
3

### Summary
The paper proposes a novel sampling method for generating noiseless images from noisy training data. It mathematically defines an extended score that suppresses small variations while preserving large ones. This formulation motivates noiseless generation, as on-manifold directions typically exhibit large variations, whereas off-manifold directions correspond to small variations. Building on the extended score, the authors introduce a new sampling method and empirically evaluate it on both toy and real-world datasets.

### Strengths
The paper introduces an interesting new concept called the extended score. It provides mathematical proofs of several noteworthy properties of this term. Building on these properties, the authors propose a novel sampling algorithm and demonstrate its effectiveness through empirical experiments. Overall, the paper is well-motivated and clearly structured.

### Weaknesses
1. In Section 3, the paper proves the properties of the extended score under the Multi-Delta and Gaussian distributions with $\gamma = 0$, but then naturally extends these results to more complex real-world data distributions and $\gamma \neq 0$. This introduces a significant gap between theory and practice.

2. As shown in Figures 2–5, the proposed sampling method tends to pull the generated samples toward high-likelihood regions. In other words, although the method effectively reduces off-manifold noise, it also significantly alters the on-manifold distribution. Consequently, the noiseless sampling comes at the cost of reduced diversity. Also, these results are contradict to the statement "leave on-manifold variations unchanged".

3. For the noiseless generation experiments, the paper only presents results in Figures 7 and 8, which are relatively simple or toy examples. Moreover, the results are consistent with my concern above, that the generated samples exhibit very low diversity.

### Questions
1. In the abstract, the paper states: “The extended score … can be used to reduce small variations to zero, while leaving large variations mostly unchanged.” However, I find it unclear how this claim is theoretically justified in Section 3. In particular, Equation $H_0 \delta = S g_1$ (line 165) seems to indicate that the extended score of a delta distribution (variance → 0) is equal to the score of a Gaussian distribution with variance 1. In other words, it appears to increase the variance from 0 to 1, which seems contradictory to the statement that it “reduces small variations to zero.” Could the authors clarify this point further?

2. I would suggest evaluating the proposed method on more complex datasets such as CIFAR-10, FFHQ, or ImageNet. Additionally, when adding noise to images, it would be interesting to explore different types of noise, such as pixel removal or Gaussian blur, to better demonstrate the robustness of the proposed approach.

3. I also notice some conceptual similarities between classifier-free guidance and the sampling method proposed in this paper. Both approaches push the generation process toward high-likelihood regions. It may be worthwhile for the authors to explore how the proposed sampling method could potentially improve generation quality, perhaps offering a more interpretable "classier-free guidance".

### Soundness
3

### Presentation
3

### Contribution
2
