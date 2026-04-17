# Continuous Space-Time Video Super-Resolution with 3D Fourier Fields

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We introduce a novel formulation for continuous space-time video super-resolution. Instead of decoupling the representation of a video sequence into separate spatial and temporal components and relying on brittle, explicit frame warping for motion compensation, we encode video as a continuous, spatio-temporally coherent 3D Video Fourier Field (VFF). That representation offers three key advantages: (1) it enables cheap, flexible sampling at arbitrary locations in space and time; (2) it is able to simultaneously capture fine spatial detail and smooth temporal dynamics; and (3) it offers the possibility to include an analytical, Gaussian point spread function in the sampling to ensure aliasing-free reconstruction at arbitrary scale. The coefficients of the proposed, Fourier-like sinusoidal basis are predicted with a neural encoder with a large spatio-temporal receptive field, conditioned on the low-resolution input video. Through extensive experiments, we show that our joint modeling substantially improves both spatial and temporal super-resolution and sets a new state of the art for multiple benchmarks: across a wide range of upscaling factors, it delivers sharper and temporally more consistent reconstructions than existing baselines, while being computationally more efficient. Project page: https://v3vsr.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduced INR framework  for arbitrary video super resolution. In particular, the authors utilize 3D Fourier feature embedding to resolve aliasing components during reconstruction. By using 3D representation, they are able to achieve better reconstruction results compared with 2D Fourier embedding or 2D Fourier embedding + B-spline embedding.

### Strengths
- 3D video Fourier Field is proposed to reconstruct 3D arbitary super resolution in video. This is easy to handle and simply explained.
- The proposed VFF shows high jump in quantitative comparisons in Tables 1 & 2. It's meaningful gap.
- Temporal consistency looks very coherent in Fig. 5. The core deliver for vsr is usually accurate prediction of temporal coherency. In that terms, V^3 algorithm shows optimal solution for video super resolution.

### Weaknesses
- I would like to see computational complexity including memory consumption and inference time as table in main text.
- This is regression model, so when we see the results related with texts in Fig.10, we couldn't guess the text information. Nevertheless, the proposed method is highly impactful in quality perspective.
- no detailed analysis on Fourier coefficients. It will be helpful to show Fourier features aligned with video contents such as DFT spectra of the original contents and estimated coefficients.
- No controlling method for weighting spatial or temporal axis, independently. For example, temperature methods sometimes boiled down the effect of reconstruction. Here in VFF, there is no induced effect by user side. VFF automatically estimated Fourier coefficients along temporal and spatial axis.

### Questions
- How does this algorithm behaves for large motion gap such as big fluctuation in optical flow? or abrupt frame changes(key-frame)? In my guess, V^3 depends on 3D Fourier representation which includes time domain resulting in further aliasing components along with temporal axis when they encounter abrupt or big motion changes along with temporal axis

- Isn't there any aliasing results in slomo reconstruction? A task related with slomo is focusing on temporal super resolution, so VFF only provides temporal enhancements which may contradict with spatial super resolution.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes V3, a novel method for continuous space-time video super-resolution (C-STVSR), centered around a new representation called the Video Fourier Field (VFF). VFF models a video as a continuous 3D trigonometric expansion over joint spatial and temporal coordinates (x,y,t), avoiding the common practice of decoupling space and time via optical flow and explicit frame warping. 
Key contributions include:
VFF: A unified, interpretable, continuous video representation using a finite sum of 3D sinusoidal basis functions.
End-to-end framework (V3): A neural encoder predicts VFF coefficients from low-resolution input, enabling arbitrary spatial and temporal upsampling.
Analytical anti-aliasing: Incorporates a closed-form Gaussian point spread function (PSF) for aliasing-free reconstruction at any scale.
State-of-the-art performance: V3 outperforms prior C-STVSR methods by >1.5 dB PSNR across multiple benchmarks while being faster and more memory-efficient.

### Strengths
This paper introduces a completely novel approach by modeling video as a joint 3D Fourier field, establishing a new paradigm in the field of video super-resolution.

This paper leverages the classical principle that "translational motion manifests as phase shifts in the frequency domain" and builds a model that does not rely on traditional optical flow algorithms.

The experimental validation is comprehensive: extensive tests on multiple tasks (such as continuous spatiotemporal super-resolution, spatial super-resolution, and video frame interpolation) and various public datasets consistently demonstrate that its performance significantly surpasses existing state-of-the-art methods, with a notable improvement (PSNR gain exceeding 1.5–2 dB).

### Weaknesses
The paper employs a globally fixed set of frequencies during training, with only independent modulation of amplitude and phase for each voxel. While this enhances stability, it may limit the model's adaptability to diverse motion spectra (e.g., the differences between high-frequency vibrations and slow translations). The paper does not analyze what specific frequencies are learned or explore how these frequencies correlate with different types of motion. Providing ablation studies or visualizations of the learned frequency distributions would more convincingly demonstrate the expressive power of these basis functions.

There is a lack of comparison and review of classical data-driven methods, such as BasicVSR, TTVSR, and FTVSR.

The degradation model is overly simplified: training assumes ideal bicubic downsampling and clean inter-frame subsampling. However, real-world videos often contain noise, compression artifacts, or motion blur. Although Section 5 mentions this as future work, even a brief experiment (e.g., validating the model's robustness under mild noise or JPEG compression) would more strongly support its claimed practical applicability.

### Questions
I'm a bit confused about the method in this paper. The paper proposes modeling a video as a 3D Fourier series, called a Video Fourier Field (VFF). Does this mean that each individual video is represented by its own, independently trained VFF?  

This seems different from conventional data-driven video super-resolution (VSR) algorithms. In traditional VSR, once the model is trained, it can theoretically take any input video and produce a high-resolution output.  

If, in this paper, every video requires its own dedicated VFF representation, how can this approach be applied in real-world scenarios?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel formulation for continuous space-time video super-resolution. Unlike previous approaches that decouple video representation into separate spatial and temporal components and rely on frame warping for motion compensation, the proposed method encodes video as a continuous, spatiotemporally coherent 3D Video Fourier Field (VFF). The resulting model, dubbed V³, achieves substantial improvements in both spatial and temporal super-resolution and establishes a new state of the art across multiple benchmarks. However, certain issues regarding the experimental verification remain to be addressed.

### Strengths
(1) This work represents video as a continuous, spatiotemporally coherent 3D Video Fourier Field, departing from conventional approaches that decouple video representation into separate spatial and temporal components.

(2) The proposed model, V³, substantially improves both spatial and temporal super-resolution performance, setting a new state-of-the-art on multiple benchmarks.

### Weaknesses
(1) The training procedure of V³ lacks clarity. Key implementation details are missing, such as the specific form of the loss function and the value of the hyperparameter σ.

(2) The work would benefit from comprehensive ablation studies. At a minimum, a systematic evaluation of the impact of two key design choices is necessary: the number of basis functions (N) and the value of the hyperparameter σ on final model performance.

(3) The experimental comparisons in Section 4.2.1 have several shortcomings. First, comparisons with Arbitrary-Scale Video Super-Resolution (AVSR) methods are absent, and no visual results from AVSR models are provided. Second, the selection of the two image super-resolution models (LTE and CLIT) in Table 2 is not justified in the text, and the models themselves are not properly cited. Furthermore, as LTE and CLIT are from 2023, including comparisons with more recent methods—such as HIIF [1] would strengthen the evaluation.

[1] 	Yuxuan Jiang, et. al. HIIF: Hierarchical Encoding based Implicit Image Function for Continuous Super-resolution. CVPR 2025.

(4) There are minor typographical errors: a comma is missing in Equation (4), and an unintended line break appears on line 396.

### Questions
(1) The training details, specifically the loss function and the hyperparameter σ, need to be clearly stated.

(2) The experimental comparisons should be revised to include benchmarks against AVSR methods.

(3) Ablation studies must be added to systematically evaluate the impact of core model parameters, such as the number of basis functions N and the hyperparameter σ.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a continuous space–time video super-resolution (C-STVSR) framework that represents an entire video as a spatio-temporally coherent 3D “Video Fourier Field” (VFF)—a finite sum of sinusoidal basis functions. This avoids decoupling space and time and eliminates brittle frame warping. A neural encoder with a large spatio-temporal receptive field predicts local frequencies, phases, and weights for these basis functions from the low-resolution input, enabling sampling at arbitrary spatial and temporal scales.  The VFF can be queried with a closed-form Gaussian point-spread function, yielding aliasing-free reconstruction by frequency-dependent rescaling rather than expensive filtering, which also generalizes well across scales.  Compared to prior C-STVSR methods and local INR approaches, the unified 3D trigonometric expansion captures long-range temporal dependencies, handles occlusions more robustly, and naturally models translation as phase shifts. Experiments show sharper, more temporally consistent results and about a 2 dB PSNR gain on Adobe240 test set, while being faster.

### Strengths
- A continuous-domain video representation that models videos as a single finite trigonometric expansion over (x, y, t) (“Video Fourier Field”), enabling arbitrary spatio-temporal sampling, encoding translation as phase shifts, and supporting analytic, aliasing-free queries.

- The proposed end-to-end C-STVSR framework (V3) whose backbone aggregates over a large spatio-temporal receptive field to predict VFF parameters, leveraging a shared frequency basis for efficiency and stability, and a PSF-aware sampler; the whole system is differentiable and trained end-to-end.

- Strong empirical results: consistent PSNR/SSIM gains across scales on REDS and markedly better video frame interpolation on Adobe240, yielding sharper, more temporally coherent outputs and avoiding warping artifacts typical of optical-flow-based baselines.

### Weaknesses
- Since the entire (x,y,t)-space is partitioned into local, axis-aligned regions, does this induce any discontinuities across region boundaries?

- The description of Sec. 3.3 Conditional Fourier Parameterization is unclear. Is the voxel grid for the local basis expansion sample-specific or shared across all samples? How should inputs with different spatial resolutions and aspect ratios be handled? What exactly is the relationship between the voxel grid and the Video Fourier Field? Please provide a more detailed explanation.

- After training, is the common frequency basis identical for all samples? Have you considered presetting a bank of frequency bases as anchors of frequency basis?

- Why is training conducted on 16× GH200? If you used the same compute cost in training stage as prior methods, how would the performance be?

- How does the method handle severe motion blur arising from fast or large motions?

### Questions
- Given VFF’s continuous spatio-temporal sampling, can the representation be extended to video frame interpolation, prediction and or video generation, while preserving temporal coherence and anti-aliasing guarantees?
- What motivates introducing a Gaussian PSF with tunable variance?
- Why was JAX chosen over PyTorch/TensorFlow?
- Could you suggest an ablation or variant that reduces training cost without degrading final quality?

### Soundness
3

### Presentation
3

### Contribution
3
