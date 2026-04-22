# Improved denoising diffusion probabilistic models with efficient non-diagonal covariance modeling

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 8, 4

## Abstract
The sampling process of Denoising Diffusion Probabilistic Models (DDPMs) can be accelerated by leveraging second-order information in the form of approximations to the denoising posterior covariance.
  Previous attempts at using such information have used drastic (e.g. diagonal) simplifications of the covariance.
  These do not do justice to the peculiar statistical structure of natural images, which exhibit strong non-diagonal correlations between pixels and color channels, and a slow-decaying power-law frequency spectrum. 
  Here, we develop a novel covariance model that captures these features. Our Kronecker-DCT (K-DCT) model uses a Kronecker-factored decomposition of inter-color  covariances and spatial covariances modeled in the frequency domain using the Discrete Cosine Transform (DCT).
  The use of the DCT reduces the computational complexity from quadratic to log-linear, resulting in negligible computational and memory overhead in the sampling process.
  By learning K-DCT-structured amortizations of the denoising posterior covariance using pre-trained score models on CIFAR-10, Celeb-A, and ImageNet datasets, we show improved performance both in terms of FID and likelihoods compared to previous SOTA denoising samplers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel Kronecker-DCT (K-DCT) covariance model for Denoising Diffusion Probabilistic Models (DDPMs) that captures non-diagonal spatio-chromatic correlations in natural images. The model leverages the Discrete Cosine Transform for efficient computation and shows consistent improvements in FID scores and negative log-likelihoods across CIFAR-10, CelebA, and ImageNet datasets compared to diagonal and low-rank baselines, particularly in the few-step sampling regime.

### Strengths
[1] The K-DCT formulation creatively combines insights from natural image statistics (separable spatio-chromatic correlations) with efficient computational transforms (DCT), going beyond simple diagonal or low-rank approximations.

[2] The technical execution is thorough, with detailed derivations, efficient implementations, and comprehensive evaluations across multiple datasets and metrics.

[3] The paper is generally well-written with clear explanations of the motivation, method, and results. The appendices provide valuable technical details.

[4] The work addresses the important challenge of accelerating DDPM sampling while maintaining sample quality and likelihood performance, with practical improvements demonstrated on standard benchmarks.

### Weaknesses
[1] While consistent, the improvements over diagonal baselines are sometimes incremental (e.g., ~1-2 FID points on CIFAR-10). The paper could better analyze why the significant structural improvements don't translate to larger performance gains.

[2] No comparison against recent distilled models (e.g., MeanFlow, sCM) that achieve superior FID scores, making it difficult to assess the method's position in the current landscape.

[3] The method cannot efficiently compute log-determinants required for exact NLL evaluation (Appendix F.1), limiting its utility for applications requiring precise likelihood calculations.

[4] The discussion of non-Gaussian posteriors in skip-step regimes (Sec. 5) lacks quantitative evidence to support this as a fundamental limitation.

### Questions
[1] Given the substantial improvement in covariance structure modeling (Fig. 1C), why are the performance gains over diagonal baselines relatively modest? Are there specific dataset characteristics or sampling regimes where K-DCT provides more substantial benefits?

[2] How does your method compare against recent one-step distilled models in terms of sampling speed vs. sample quality trade-offs? Would incorporating K-DCT covariance modeling benefit distillation approaches?

[3] Have you explored approximate methods for efficient log-determinant computation (e.g., stochastic estimators) that could make exact NLL evaluation feasible without forming the full covariance matrix?

[4] You mention potential application to audio/speech domains - have you conducted any preliminary experiments to validate this? What modifications would be needed for non-image data?

[5] You suggest input-dependent step size adaptation could help with non-Gaussian posteriors - have you experimented with this, and what were the results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a covariance parameterization for DDPMs, called Kronecker-DCT (K-DCT).The main idea is to better approximate the posterior covariance in DDPMs by explicitly modeling the non-diagonal spatial and chromatic correlations inherent in natural images. The approximately separable chromatic and spatial correlations permit a Kronecker factorization, with the spatial component efficiently modeled in the DCT frequency domain under approximate translational invariance. This structure reduces the computational complexity of full covariance modeling from O(D^2) to O(D \log D), allowing efficient training and sampling. Experiments on CIFAR-10, CelebA, and ImageNet 64×64 datasets show consistent improvements in both FID and negative log-likelihood (NLL) metrics compared to diagonal and low-rank covariance baselines.

### Strengths
1.	The K-DCT parameterization is a good way to model non-diagonal correlations. The combination of Kronecker factorization and DCT-based representation is efficient.
2.	Despite its more complex formulation, the proposed model retains log-linear training and sampling complexity by exploiting the Kronecker and DCT factorizations , having efficient computation of matrix–vector products and DCT-based transformations.
3.	This study evaluates the proposed method on multiple datasets with both linear and cosine noise schedules, comparing six covariance modeling variants. Results consistently show superior FID and NLL performance, especially in the skip-step regime.

### Weaknesses
1.	The K-DCT covariance model fundamentally relies on the assumption that chromatic and spatial correlations are approximately separable and translation-invariant. This assumption justifies the Kronecker-DCT term but is only qualitatively motivated by natural image statistics and a few visual examples. However, the paper does not quantitatively evaluate the validity of this assumption across datasets or denoising stages. The generalizability of this structured covariance form remains uncertain.
2.	Experiments are restricted to low resolution image datasets and the approach has not been tested on higher-resolution (≥256²) settings or within latent-diffusion frameworks. Moreover, generalization beyond the image domain (e.g., audio) is left unverified, which the authors themselves acknowledge as an open question.
3.	Eq.12 introduces three distinct components—diagonal compensation, color-channel correlation, and frequency-domain spectrum—but the paper provides no ablation to isolate their contributions. As a result, it remains unclear which component contributes most to FID/NLL improvements, and whether simpler variants would yield comparable gains.

### Questions
1.	Have the authors quantitatively evaluated how well the separability and approximate translation-invariance assumptions hold across different datasets?
2.	Have the authors evaluated whether the approximate translation-invariance assumption remains valid across different denoising timesteps? Since the spatial correlation structure changes as noise decreases, is this assumption still reasonable in both early and late diffusion stages?
3.	Can the proposed K-DCT covariance model be efficiently extended to higher-resolution datasets or latent-diffusion frameworks such as Stable Diffusion?
4.	How do these components interact during training—do they capture complementary statistical structures, or is there redundancy among them?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a novel approach for parametrizing a non-diagonal covariance matrix in the reverse process of Gaussian diffusion models. The proposed parametrization draws on inductive biases from computer vision related to correlations between pixels and channels in natural images. The authors further develop an efficient training and sampling methodology, building in part on prior advancements aimed at enhancing diffusion models. Experimental results demonstrate the effectiveness of the proposed method within the standard stochastic sampling procedure of classical DDPM. Comparisons are conducted against existing models employing various covariance parametrization strategies.

### Strengths
- The paper is clearly written, with well-motivated arguments and logical structure.

- The proposed training and sampling procedures make the theoretical contributions practically applicable.

- The approach has strong potential to become a commonly adopted component in future work.

- Experimental results clearly highlight the impact of the proposed method.

### Weaknesses
While the experimental comparison is fair, the standard DDPM samplers under few-step sampling regimes perform significantly worse than ODE-based or consistency-based samplers. Additional experiments demonstrating the effectiveness of the proposed parametrization in such alternative settings would further strengthen the contribution and soundness of the paper.

### Questions
See the Weaknesses section.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new method of parameterizing and learning denoiser covariance when training diffusion models. While previous work assumed diagonal and/or low rank formulations, this work used a custom covariance model motivated from empirical observations on image datasets.

Overall this method is a bit ad-hoc and unclear if there is a general principle that can be applied to other data modalities, and its application to unconditional sampling does not seem to improve over state-of-the-art.

### Strengths
This paper is clear and well-written. It motivates a new covariance parameterization taking into account natural image statistics which improves over previous proposed parameterizations. There are also comprehensive experiments evaluating its performance for unconditional diffusion sampling.

### Weaknesses
1. Although this covariance model fits better to natural image statistics, it is still hand-designed and unclear if it will generalize to other data modalities. It would be better if this parameterization is directly learned from data, or if there is a way to directly construct it from measured data statistics.
2. The experimental evaluations mainly focuses on few-step unconditional sampling, but it is unclear how much unconditional sampling benefits from posterior covariance estimation. The numerical FID results are better than the diagonal-covariance baseline, but do not seem to be better than state of the art methods such as EDM [Karras et. al. 2022].

### Questions
1. It would strengthen this paper if this particular method of covariance estimation is applied to using diffusion models to solve inverse problems. In this setting adding noise during the denoising process is more important and there could be potentially bigger improvements than in unconditional sampling.
2. How would the generation results in Table 2 compare to DDIM sampling?
3. Line 215: What is the $||\cdot||_F$ term? In Eq. 8, does "diag" extract the diagonal of a matrix?
4. Is the $(\cdot)^2$ in Eq. 10 acting element-wise on the vector?

### Soundness
3

### Presentation
3

### Contribution
2
