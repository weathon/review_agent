# Learning What Matters: Steering Diffusion via Spectrally Anisotropic Forward Noise

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Diffusion Probabilistic Models (DPMs) have achieved strong generative performance, yet their inductive biases remain largely implicit. In this work, we aim to build inductive biases into the training and sampling of diffusion models to better accommodate the target distribution of the data to model. We introduce an anisotropic noise operator that shapes these biases by replacing the isotropic forward covariance with a structured, frequency-diagonal covariance. This operator unifies band-pass masks and power-law weightings, allowing us to emphasize or suppress designated frequency bands, while keeping the forward process Gaussian. We refer to this as Spectrally Anisotropic Gaussian Diffusion (SAGD). In this work, we derive the score relation for anisotropic forward covariances and show that, under full support, the learned score converges to the true data score as $t\to0$, while anisotropy reshapes the probability-flow path from noise to data. Empirically, we show the induced anisotropy outperforms standard diffusion across several vision datasets, and enables selective omission: learning while ignoring known corruptions confined to specific bands. Together, these results demonstrate that carefully designed anisotropic forward noise provides a simple, yet principled, handle to tailor inductive bias in DPMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores introducing anisotropic noise to the forward process of diffusion models, as a method of controlling the inductive biases present during training to better match that of the training data. This is done by reweighing the added noise in the Fourier domain, either by a radial power-law reweighing of frequencies or by a band-pass filter. This method is then evaluated experimentally on various image datasets.

Although the questions this paper set out to answer are interesting, the proposed method seems ad-hoc and makes choices that are not well justified. The experimental evidence is also not strong, and it is unclear that this method is effective on larger image datasets.

### Strengths
Shaping the added noise in diffusion models to better match the data distribution is important to further improve their performance, and this paper proposes a new method of doing so. There are also extensive experimental evaluations on many image datasets.

### Weaknesses
1. There is not much motivation on why the Fourier basis is chosen, or for the design choices of band-pass/power-law-weighted filters. I am not convinced why adding spectrally anistropic noise is better than white noise for image datasets.
2. The theoretical results only show that the true data distribution can be recovered, but it does not provide any reason for why adding anisotropic noise would be helpful in training and sampling of diffusion models. It would be more compelling if there are results showing what is the optimal noise to add for any given data statistics (e.g. average power spectral density).
3. The results (Fig. 4) only show significant improvement for MNIST, but the limited improvements on larger datasets cast doubt on the practicality of this method on larger-scale and more diverse datasets.

### Questions
1. Instead of the Fourier basis and hand-designed filters, is it possible to learn a basis from training data?
2. Is this equivalent to applying a linear transformation on data, then performing standard denoising using white noise in the transformed space?
3. If the power-spectral density of natural images follow a power law, then adding white Gaussian noise will first corrupt high-frequency components before low-frequency ones. In this case, how would using spectrally anistropic noise be different from using a different schedule for white noise?
4. For FID evaluations, why use the 768-dim layer instead of the standard 2048-dim layer?
5. Is it possible to implement this method on a state-of-the-art baseline (e.g. EDM) and see if generation quality is improved?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work discusses the importance of applying inductive bias on noise in the frequency domain for diffusion-based models. Theoretical analysis are given to show how these biases can be given and why these biases do not hurt the overall diffusion process. Experiments are offered to showcase that by applying such bias, either through the power-law weighting or through two-band mixture, diffusion models can better do de-noising in some specific environments.

### Strengths
1. **The idea** of applying inductive bias on different frequencies of the noise looks interesting and sounds reasonable to me. Like shown in the experiments: some images, e.g. MNIST data, store more information in specific frequency ranges, thus the proposed SAGD performs significantly better on these datasets.
2. **The theoretical introduction** and proof is friendly to readers, where step-by-step equations are provided to clearly identify which biases are added and how.
3. **The experiments are rigorous**: it is great to have error bars even on statistical results like FID etc., which strongly supports the reliability of the experiment results.

### Weaknesses
1. **The Motivation** is not thoroughly demonstrated. While SAGD is designed to put inductive bias on the frequency of noise, it is unclear why this is necessary for real-world image data. SAGD Experiments on real-world images (CIFAR) also do not benefit from such a bias. Therefore, I recommend the authors to give frequency distributions of images in real-world image datasets like CIFAR, ImageNet etc. to further demonstrate the motivation.
2. **Lacking Experiment** on real-world image datasets. The model does not benefit much on real-world image datasets (like CIFAR), or benefits very little from the proposed method (like FFHQ), and the manuscript does not show more experiments in more common-used large-scale datasets like ImageNet or so. I recommend the authors to consider providing experimental results at least on ImageNet, and preferably on even larger datasets like LAION or MSCOCO.
3. **Unclear Applications**. Since most of the experiments are done either on specific (hand-drawn) datasets or on corrupted datasets, it is quite unclear how SAGD could help to advance the diffusion-based models. I recommend the authors to revisit the experiments and to provide specific answers to "Where should I use SAGD and why". This will make the proposed method more beneficial to the society, even if the method is finally found out not working on general real-world image datasets.

### Questions
Please see the weaknesses above. I will consider raising my evaluations if the authors can 1) demonstrate real-world necessities of introducing frequency bias through frequency-domain analysis on real-world datasets; and 2) show results on more real-world dataset that has performance enhancements using the proposed SAGD.

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
2

### Summary
This paper proposes Spectrally Anisotropic Gaussian Diffusion (SAGD), a method to explicitly build inductive biases into Diffusion Probabilistic Models by manipulating the forward noising process. The core idea is to replace the standard isotropic (uniform across frequencies) Gaussian forward covariance with a structured, frequency-diagonal (anisotropic) covariance. The key innovation is to steer which frequency bands are destroyed during the forward process, thereby embedding explicit inductive biases about what information the model should focus on learning. Two variants are introduced:
• plw-SAGD, which emphasizes or suppresses frequencies smoothly through a power-law weighting function, and
• bpm-SAGD, which selectively restricts specific frequency bands.

### Strengths
1). The paper is written in a very strctured manner, that a reader can understand the proposed method.


2). To the best of my knowledge, the proposed method is novel. Additionally, it is simple and practical. For instance, the method modifies only the forward covariance, allowing it to integrate with existing diffusion implementations with minimal code changes, primarily
involving FFTs, per-frequency weighting, and IFFTs. The computational overhead is negligible.


3). SAGD can outperform standard diffusion on several vision datasets (MNIST, CIFAR-10, WikiArt, FFHQ) by aligning the forward noising operation with the data's dominant spectral content.

### Weaknesses
1). Some of the recent works have not been cited. 


2). I would like to know, when w(f) strongly suppresses or amplifies certain bands, does this cause numerical instability or poor conditioning in the FFT/IFFT pipeline?


3). I would like to know why authors specifically normalizes the coordinates to (-0.5,0.5)? Additionally, as the power-law weighting depends on the magnitude of the normalized frequency, would the same still be optimal if the FFT were parameterized on (-1,1) ? If so,
how should w(f) be rescaled to make results invariant to the frequency normalization convention? In other computer vision areas that use normalized coordinates, the coordinate normalization range plays a crucial role in overcoming spectral bias.


4). Did authors explore data-driven approaches to design or adapt w(f) automatically?


5). While the paper conceptually distinguishes SAGD from prior anisotropic or colored-noise diffusion approaches, it does not include quantitative comparisons with these methods. I feel a direct empirical comparison with conceptually similar published works
especially recent anisotropic diffusion or noise-shaping models would be necessary to fully assess the proposed method’s performance and novelty.

6). The paper introduces anisotropic noise using an FFT-based approach to modify noise in the frequency domain. However, as I understood, this ultimately produces spatially correlated noise in the image, why is the frequency-domain formulation necessary? Can
we obtain a similar effect more simply by applying a fixed or learnable spatial filter (such as a convolution kernel) instead of using Fourier transforms?

### Questions
see the weaknesses section

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the impact on performance of manipulating the variance of Gaussian noise in the Fourier basis, which is used to perturbate data in the Forward Process of Denoising Probabilistic Models.

Two methods are proposed:
- plw-SAGD, which performs exponential weighting based on the radius in the Fourier basis
- bpm-SAGD, which extracts only specific bands using a Binary Mask

The paper claims that, depending on the choice of parameters, the method can outperform approaches that use ordinary Gaussian Noise for some cases.

### Strengths
- An ablation study is conducted across various datasets.
- The proof of theoretical consistency is convincing.
- For small-scale datasets such as MNIST, improvements in evaluation metrics are observed.

### Weaknesses
- For large-scale datasets such as Wiki-Art and FFHQ, there are no significant improvements in evaluation metrics. Moreover, as mentioned in Section 3.2.3, for such datasets improvements are observed only when the sweep converges to α → 0, i.e., when the method asymptotically approaches ordinary Gaussian Noise. From this, doubts remain regarding the effectiveness of the method on highly diverse datasets.
- Determining the ideal α parameter requires sweeping, which leaves practical difficulties.

### Questions
- Although the title uses the term Anisotropy, in this paper the manipulation of Gaussian noise appears to be consistently one-dimensional based on the radius in frequency space. Is it appropriate to use the term Anisotropy for a one-dimensional target?
- In plw-SAGD, since f is in [-1/2, 1/2]^2 and w(f) = (r(f) + ε)^α, depending on the value of α the per-frequency energy of the noise overall decreases or increases. The impact from shifting the overall energy level of the power spectrum and the impact from shifting the shape of the energy distribution can be considered separately; which effect is dominant?
- What is the specific number of samples used in Figure 5?

### Soundness
2

### Presentation
3

### Contribution
1
