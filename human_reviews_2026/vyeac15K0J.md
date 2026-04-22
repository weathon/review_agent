# FaCE: Provable Frequency-Aware Convex Enhancement for Training-Free Low-Light Images

- Avg Score: 3.50
- Decision: Reject
- Scores: 0, 6, 2, 6

## Abstract
We study training-free low-light image enhancement from a convex optimization viewpoint with theoretical guarantees. Building on the Monogenic Fourier Transform (MFT), we introduce Frequency-Aware Convex Enhancement (FaCE), a frequency-aware convex enhancement method for training-free low-light images with guarantees on existence, uniqueness, and stability of the per-image solution. To make the frequency modeling reproducible, we (i) define the low-pass prior via the spectral centroid $(u_c, v_c)$ and an energy–cumulative distribution radius $r_\tau$ (the smallest radius achieving cumulative spectral energy $\tau$), and (ii) select the number of spectral clusters $K$ with a data-driven model-selection rule. This yields a fully training-free pipeline with clear variables and no learned parameters. We prove that a unique, stable solution exists and verify the guarantees via numerical experiments. On standard benchmarks, FaCE attains competitive quality with a per-image solver, and we include in-the-wild qualitative mosaics on real images to highlight practical usefulness. Rather than competing with large learned priors, FaCE complements them as a theoretically grounded, interpretable alternative that requires no training and exposes frequency-band attributions.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper proposes FaCE, a frequency-domain unsupervised low-light image enhancement method based on the Monogenic Fourier Transform. The approach clusters spectral components and defines a piecewise-constant enhancement curve using a few scalar parameters, aiming for interpretability and mathematical guarantees (existence, uniqueness, stability) through a convex variational formulation. The pipeline—transform, clustering, weighting, inverse transform—is conceptually simple.

### Strengths
1. Clear motivation toward unsupervised and interpretable enhancement: The paper emphasizes physically grounded frequency separation and aims for mathematical rigor in an ill-posed problem setting.
2. Simple and intuitive pipeline: The proposed framework is easy to understand and implement.
3. Some positive initial results: Improvements over Multi-Scale Retinex and Zero-DCE on LOL-v1.

### Weaknesses
- Although the paper emphasizes rotation, scale invariance and directional sensitivity of the Monogenic Fourier Transform, the implementation uses only the magnitude for clustering and enhancement weight design. Phase and orientation components are entirely ignored. => The claimed advantages of MFT are not substantiated empirically.

- A Key algorithmic step (clustering) in the spectral domain is insufficiently detailed, reproducibility is limited.

- The paper mentions luminance-chrominance separation but lacks spefics, so hard to assess effect on color fidelity.

- Experiments are conducted only on LOLv1 with two older baselines (MSR, ZeroDCE), need more recent LLIE methods (e.g., RUAS, Zero-IG, etc.)

- there's no non-reference metrics (NIQE, BRISQUE, LOE, LPIPS, etc.)

- need runtime and efficiency analysis and evidence for general effectiveness.

### Questions
1. Monogenic properties are not used in practice. If the method ultimately relies only on Fourier magnitude, why is the Monogenic Fourier Transform necessary at all? Can you show any measurable benefit from phase/orientation information?

2. How is the clustering performed exactly? Please describe:
   - feature vector definition, distance metric, initialization and stopping conditions, whether clustering is per-image or global

3. Why is K = 4 used universally? Is there any data-driven rule or analysis supporting this choice?

4. Which luminance–chrominance transform is used? Is the input linearized or gamma-corrected?

5. How is color consistency preserved during reconstruction?

6. Why are recent state-of-the-art low-light methods omitted, especially those also leveraging optimization, frequency, or physical priors?
Without comparison, performance claims appear unconvincing.

7. Has the method been tested on other datasets (LOL-v2, DICM, LIME, MEF, VV)? If not, how can generalization and robustness be assessed?

8. What is the precise definition of the “numerical stability” metric? How is variance computed? Why do several entries report zero variance, which seems unrealistic?

9. If the solution is guaranteed to be unique and stable, why is no evidence provided on actual behavior across diverse perturbations?
Where is the formal link between the theoretical variable and the parameters actually optimized?

10. What specific part of the enhancement operation is interpretable beyond basic frequency amplification? Can you quantify or demonstrate practical interpretability?

11. Need more ablation studies
    - monogenic vs. Fourier magnitude
    - clustering vs. no clustering
    - low-pass mask vs. none

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FaCE, a training-free LLIE method centered around the MFT. Unlike mainstream approaches, FaCE formulates the enhancement task as a strictly convex per-image optimization problem to ensure the uniqueness of the solution. Specifically, it defines a low-pass prior using the spectral centroid ($u_c, v_c$) and an energy-cumulative distribution radius ($r_{\tau}$), while adaptively selecting the number of spectral clusters $K$ via a data-driven model-selection rule. The core idea is to enhance image illumination by constructing a frequency-aware spectral gain map. The method operates in a fully training-free manner, relying solely on MFT for frequency-domain analysis and inverse optimization for reconstruction. The authors provide mathematical proofs for their claims and validate them on several benchmark datasets.

### Strengths
The core strength of this paper lies in its novelty. It innovatively reconstructs the LLIE problem as an MFT-grounded, strictly convex optimization problem, which represents a significant departure from current methods that rely on empirical learning. The writing is clear, the logic is sound, and the experiments effectively validate its claims.

### Weaknesses
The implementation details regarding key aspects of the method are not sufficiently transparent. First, the experimental section should include comparisons with more recent and advanced unsupervised or training-free LLIE methods, such as UHDFour and ULEFD, which are already mentioned in the related work section. This would help contextualize FaCE's performance relative to the current state of the art. 

Furthermore, the paper claims that key parameters ($K, u_c, v_c, r_{\tau}$) are determined in a data-driven manner, but the description of the underlying strategy is insufficient. How are these parameters precisely calculated or determined for each image? 

As a per-image optimization method, is its computational overhead practical for real-world applications? The paper lacks a discussion on this crucial aspect in its experiments.

### Questions
1.  Could the authors include a comparison with more recent and potentially stronger training-free or unsupervised LLIE methods (e.g., UHDFour, ULEFD) in the experimental section? This would provide a clearer comparison of FaCE's standing within the non-learning paradigm, even if your primary goal is not to compete with learned models.

2.  Could you elaborate on the "data-driven model-selection rule" for selecting $K$ and how the low-pass prior parameters ($u_c, v_c, r_{\tau}$) are chosen based on image frequency? Specifically, what algorithm or heuristic is used to automatically determine these parameters for each input image in a training-free manner?

3.  Could the authors provide an analysis or experimental results on the computational time required for FaCE to process an image of a given resolution?

4.  The paper mentions deriving band-level contribution maps directly from the convex objective, exposing the frequency-to-image pathway. Could you provide an example or a more detailed explanation of how these maps are visualized or used to interpret the enhancement process for a specific image?

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
This paper proposes  Frequency-Aware Convex Enhancement, a training-free framework for low-light image enhancement.
The key idea is to reformulate the enhancement problem as a strictly convex optimization task in the Monogenic Fourier Transform domain.
FaCE decomposes illumination and reflection components in the frequency space, constructs a frequency-aware gain map via unsupervised clustering, and theoretically guarantees existence, uniqueness, and stability of the optimal solution.
The method claims to outperform traditional Retinex-based and deep learning approaches on LOL-v1/v2 datasets without any training process.

### Strengths
1. The paper is well-written and mathematically consistent.
2. The framework is training-free and fully deterministic, which makes it reproducible and computationally lightweight.
3. Implementation is simple, interpretable, and self-contained, offering practical value.

### Weaknesses
1. The term “provable” in the paper merely indicates that the formulation is strictly convex and thus solvable, rather than introducing any new optimization principle or convergence scheme. The proofs of existence, uniqueness, and stability rely on standard convex optimization results (strict convexity + lower semicontinuity).
The paper does not introduce any new theorem or problem-specific analysis that advances the theory of image enhancement.
2. Evaluation is restricted to LOL-v1/v2 datasets. The paper does not include comparisons with recent diffusion-based LLIE methods, which are now standard baselines.
3. The abstract and introduction frame the work as “provable”,” but the core mathematics and methodology remain routine, offering little new insight.
4. While “training-free” sounds appealing, the method’s applicability to modern imaging tasks (night vision, HDR, IR enhancement) is unclear.

### Questions
1. From an algorithmic perspective, the method is merely a standard L2 + TV regularization problem implemented in the frequency domain. What is the substantive difference between your convex formulation and existing variational Retinex models (e.g., Fu et al., TIP 2016)? 
2. Why did you choose the Monogenic Fourier Transform over other common frequency tools such as Gabor or Wavelet transforms?
3. Have you tested FaCE on non-LOL datasets (e.g., dark natural images, night scenes) to verify generality?
4. Could FaCE be combined with deep models as a pre/post-processing step, and would it still maintain convexity and theoretical guarantees?
5. Can you discuss computational complexity compared to typical learning-based enhancement networks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper is well structured and logically organized, adhering to high academic standards. It presents a clear and coherent progression from motivation through methodology, theoretical formulation, results, and finally to conclusion. All equations are mathematically consistent and theoretically sound within the defined framework.

The proposed formulation is elegant, and the proofs are built upon established convex optimization principles, ensuring mathematical rigor. The author has provided relevant and appropriate references to prior literature, demonstrating a strong grounding in existing research.

Framing low-light image enhancement as a strictly convex optimization problem is particularly noteworthy, as it provides strong mathematical guarantees of existence, uniqueness, and stability for the solution. Moreover, the approach eliminates the need for large-scale annotated datasets or lengthy training phases.

Experimental results are compelling: the proposed method achieves competitive PSNR and SSIM scores, outperforming well-known baselines such as Zero-DCE and Retinex-based models. The framework also exhibits superior numerical stability under perturbations, consistent with its theoretical foundations. Overall, the results are convincing, clear, and well supported by theory and experimentation.

### Strengths
The paper demonstrates several notable strengths. It is well-structured, logically organized, and supported by a strong theoretical foundation grounded in convex optimization. The formulation of low-light image enhancement as a strictly convex optimization problem is both innovative and mathematically rigorous, ensuring existence, uniqueness, and stability of the solution. The approach eliminates the need for large annotated datasets or extensive training, making it computationally efficient and practical for real-world use. Experimental results further strengthen the contribution, showing higher PSNR and SSIM values compared to established baselines such as Zero-DCE and Retinex, while maintaining superior numerical stability under perturbations. Overall, the paper combines solid theoretical reasoning with clear empirical validation, presenting a reliable and well-referenced contribution to the field.

### Weaknesses
The comparison is done only with two models i.e Retinex & Zero DCE . More comparison  data is required to check the sustainability of the proposed model.
FaCE achieves moderately  PSNR (~17 dB), outperforming other training-free baselines but falling short of modern deep-learning-based low-light enhancement models.
The manual dependence on λ₁, λ₂, K  limits FaCE’s efficacy as it reduces consistency, adaptability, and generalization. Without automatic calibration, the model’s enhancement quality can vary significantly between images, reducing its overall practical reliability despite its theoretical strength.
Tests mainly on LOL-v1 and LOL-v2 datasets; broader generalization to diverse real-world illumination or camera domains is unproven
The method may serve more as a proof-of-concept than a high-performance system.

### Questions
The comparison is done only with two models i.e Retinex & Zero DCE . More comparison  data is required to check the sustainability of the proposed model.
FaCE achieves moderately  PSNR (~17 dB), outperforming other training-free baselines but falling short of modern deep-learning-based low-light enhancement models.
The manual dependence on λ₁, λ₂, K  limits FaCE’s efficacy as it reduces consistency, adaptability, and generalization. Without automatic calibration, the model’s enhancement quality can vary significantly between images, reducing its overall practical reliability despite its theoretical strength.
Tests mainly on LOL-v1 and LOL-v2 datasets; broader generalization to diverse real-world illumination or camera domains is unproven
The method may serve more as a proof-of-concept than a high-performance system.

### Soundness
2

### Presentation
2

### Contribution
2
