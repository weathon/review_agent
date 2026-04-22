# UltraGauss: Ultrafast Gaussian Reconstruction of 3D Ultrasound Volumes

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Ultrasound imaging is widely used due to its safety, affordability, and real-time capabilities, but its 2D interpretation is highly operator-dependent, leading to variability and increased cognitive demand. 
We present $\textbf{UltraGauss}$: an ultrasound-specific Gaussian Splatting framework that serves as an efficient approximation to acoustic image formation. Unlike projection-based splatting, UltraGauss renders by $\textit{probe-plane intersection}$ with in-plane aggregation, aligning with plane-based echo sampling while remaining fast and memory-efficient. A stable parameterisation and compute-aware GPU rasterisation make this method practical at scale. On clinical datasets, UltraGauss delivers state-of-the-art 2D-to-3D reconstructions in minutes on a single GPU (reaching 0.99 SSIM within $\sim$20 minutes), and a clinical expert survey rates its reconstructions the most realistic among competing methods. To our knowledge, this is the first Gaussian Splatting approach tailored to ultrasound 2D-to-3D reconstruction. Our code is available at: https://www.robots.ox.ac.uk/~vgg/research/UltraGauss/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a specialized Gaussian Splatting framework for ultrasound imaging, designed to reconstruct 3D volumes from traditional 2D scans. Instead of using conventional camera-style projection, it models the acoustic image-formation process via probe-plane intersection rendering. This approach ensures physical consistency with ultrasound acquisition while achieving high computational efficiency.

Empirical results from fetal ultrasound datasets show that UltraGauss consistently outperforms recent models, such as ImplicitVol, RapidVol, and UltraNeRF, in both reconstruction fidelity and speed. When integrated into an end-to-end clinical pipeline using freehand 2D video sweeps, UltraGauss achieved the highest structural similarity and perceptual quality among all models tested.

### Strengths
1. The proposed UltraGauss is an interesting technical advance in 3D ultrasound reconstruction. It replaces traditional ray-based rendering with a probe-plane intersection model that matches real ultrasound physics, giving more accurate and consistent reconstructions. 
2. The method is extremely fast, achieving near–real-time 3D volume generation on a single GPU while maintaining high image fidelity. 
3. Its design is simple and effective, combining Gaussian splatting with a lightweight attenuation model for stable optimization.
4. Experiments show strong quantitative and visual gains over existing methods, and expert evaluations confirm clinical realism.

### Weaknesses
1. The Beer–Lambert shadow model ignores scattering, refraction, and probe beam patterns, which may reduce physical accuracy; the method also assumes precise probe poses;
2. Experiments are limited to fetal ultrasound, leaving its generalization to other organs or imaging conditions uncertain;
3. Evaluation focuses mostly on visual and SSIM metrics, with limited anatomical accuracy analysis;
4. The paper omits details on GPU memory, runtime scaling, and failure cases, which are important for clinical reliability.

### Questions
In addition to the weaknesses that need to be addressed, the authors may want to consider addressing the following questions:

1. How sensitive is UltraGauss to pose estimation noise and incomplete scan coverage?
2. Can the model handle more complex acoustic effects or generalize to cardiac and abdominal ultrasound?
3. How does it compare with ultrasound-NeRF baselines such as AcousticNeRF?
4. Is there a way to estimate reconstruction uncertainty or confidence for clinical safety?
5. What are the typical compute requirements, memory usage, and runtime scaling with scan size?

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
5

### Summary
This paper presents UltraGauss, an ultrasound-specific Gaussian Splatting framework that efficiently approximates acoustic image formation via probe–plane intersection rendering instead of traditional camera projection. It introduces a triangular precision parameterization for stable optimization, a compute-aware χ²-based rasterization pipeline for fast and balanced GPU rendering, and a lightweight Beer–Lambert attenuation model for realistic acoustic shadowing. Validated on clinical fetal datasets and freehand sweeps, UltraGauss achieves minute-level, high-fidelity 3D reconstructions (SSIM≈0.99) on a single GPU, with clinician-rated realism comparable to native 3D ultrasound scans.

### Strengths
The paper shows several notable strengths.

First, its novelty is good — it is the first to adapt 3D Gaussian Splatting to ultrasound reconstruction, proposing a probe–plane intersection rendering mechanism that better captures ultrasound’s plane-based sampling and attenuation behavior. This represents a well-motivated and technically elegant reformulation of Gaussian splatting aligned with acoustic image formation.

Second, the performance is solid and the experiments are very comprehensive. The authors conduct extensive evaluations on both clinical fetal ultrasound volumes and freehand cine sweeps, achieving high-fidelity 3D reconstructions (SSIM≈0.99) within minutes on a single GPU. The ablation studies are detailed and convincing, isolating the effects of the precision parameterization, rasterization bounds, and attenuation modeling, all of which contribute measurable gains.

Finally, the writing is clear and the presentation is well-organized and polished. The figures effectively visualize the differences between camera-based and ultrasound-based rendering, and the overall exposition makes the paper easy to follow even for readers outside the immediate subfield.

### Weaknesses
(1) The paper lacks sufficient discussion and comparison with two highly related works, X-Gaussian and R2Gaussian. In particular, its Beer–Lambert–based attenuation formulation is conceptually similar to the radiative attenuation modeling in R2Gaussian, yet this connection is not explicitly acknowledged. The writing style and overall pipeline design also bear strong resemblance to these two prior works. Without a detailed discussion or quantitative comparison against them, the novelty claim of UltraGauss becomes weaker, as it appears to build upon established ideas without clearly articulating its distinct contributions.


(2) The code and pre-trained models are not submitted, making it impossible to verify the reported performance or reproduce the results. Given that the paper claims substantial efficiency gains and clinical realism, open-sourcing the implementation would be crucial for reproducibility and community adoption.


(3) Ablation interpretation and runtime trade-offs. While the ablation studies are extensive, the analysis remains mostly quantitative. A deeper discussion of why specific design choices (e.g., triangular precision vs. quaternion covariance) improve stability or speed would enhance the reader’s understanding.

(4) Dependence on probe pose estimation. In the end-to-end pipeline, UltraGauss still relies on an external pose estimation module. Errors in pose prediction could propagate into reconstruction quality, yet this dependency and its potential failure modes are not deeply analyzed.

### Questions
Could you clarify how UltraGauss differs conceptually and technically from R2Gaussian and X-Gaussian, given that both works also adopt Beer–Lambert–style attenuation modeling and Gaussian-based volumetric rendering?

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
3

### Summary
This paper introduces UltraGauss, a Gaussian-splatting framework tailored to ultrasonic image formation. Instead of camera-style projection and depth-ordered alpha compositing, the method renders by probe–plane intersection with in-plane aggregation, which better matches plane-based echo sampling. The authors propose a triangular precision (inverse-covariance) parameterization for stability/efficiency, and χ²-based ellipsoidal bounds to confine per-plane rasterization, and a two-phase, load-balanced CUDA pipeline to reject non-intersecting Gaussians and rasterize only relevant 2D windows, as well as a lightweight Beer–Lambert attenuation to approximate acoustic shadowing. On fetal ultrasound, UltraGauss is reported to reach SSIM≈0.99 in minutes on a single GPU and to be preferred by expert sonographers in a reader study.

### Strengths
1. The χ² ellipsoidal bounding and two-phase GPU pipeline are clearly motivated and provide a practical path to minute-level reconstructions with up to ~2M Gaussians. The exposition around Fig. 2–3 and Sec. 4.3–4.4 is crisp.  
2. The lower-triangular precision factorization (Σ⁻¹=LLᵀ with positive diagonals) is a sensible alternative to quaternion-scaled covariances in this setting; it simplifies inversion and Gaussian resampling and appears faster in ablations.
3. Beyond 3D volume metrics, the reader study with experienced sonographers (10 respondents, multi-country) is valuable; the progressive Turing test design is thoughtful and reveals interesting perceptual dynamics.

### Weaknesses
1. The Beer–Lambert shadow term is extremely lightweight; while speed is compelling, the paper concedes deficits in speckle and multiple-scattering effects, where physics-informed baselines sometimes produce more realistic artifacts. The method’s robustness when probe geometry is imperfect is discussed qualitatively but not deeply quantified.
2.  Core quantitative results hinge on 12 fetal brain volumes (Dataset A) and 3 freehand videos (Dataset B). That is narrow in anatomy, vendor, and probe diversity; generalization to abdomen/heart/musculoskeletal, different curvilinear geometries, and pathology is not demonstrated. Claims of broad clinical benefit feel premature without wider coverage.
3. UltraNeRF assumes straight rays/known poses; here, curvilinear inputs are converted and poses are predicted for cinesweeps. It is unclear whether each baseline is given the same pose treatment, regularization, and time/capacity budgets across settings (e.g., RapidVol capacity vs. UltraGauss-300K/2M).
4. The cinesweep evaluation conflates pose and reconstruction error. While that is realistic, the paper does not separate pose noise vs. reconstruction error (e.g., by repeating with GT poses on a subset, or sweeping pose perturbations). Consequently, it is hard to attribute gains to UltraGauss vs. better tolerance to pose jitter.

### Questions
Please refer to weaknesses part, especially points 1&3.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes UltraGaussian, the first Gaussian Splatting method tailored for ultrasound reconstruction. It introduces a rendering strategy and rasterization design guided by ultrasound imaging principles. Experiments on two datasets demonstrate improved performance over previous methods.

### Strengths
- The rendering strategy is thoughtfully designed based on the principles of ultrasound imaging.
- The paper provides a thorough discussion of related work.
- It clearly introduces the background of 3D Gaussian Splatting and ultrasound imaging to aid understanding.
- The datasets and experimental settings are described in detail, including a clinician survey for evaluation.

### Weaknesses
1. **Further clarification on efficiency**: UltraGauss is described as "efficient" in terms of both memory and computation time (lines 013, 044, 107, 122, 211， 477， etc.). However, this claim is not supported by any quantitative results in the current version of the paper.
- Although the time consumption of various methods is presented in Fig. 4, Fig. B1 (Appendix), and Fig. D3 (Appendix), these results do not clearly support the claimed efficiency of UltraGauss.

2. **Clarification for the proposed method**:
- The rendering process described in Eq. 6 and Eq. 7 appears to be weakly motivated, as it mainly adds a background term to fill in empty regions.
- The matrix ${M}$ used in Eq. 10 is semi-definite and cannot represent complex transformations involving non-uniform scaling combined with rotation. Additionally, an ablation study for different $\beta$ should be added to show the effectiveness of the proposed method.
- The lower triangular matrix ${L}$ lacks clear interpretability. It would be helpful to provide some geometric insight or intuition behind its role in the model.
- The rasterization boundaries defined in Eq. 12 only operate in 3D space, which may not accurately correspond to the 2D image space. When the probe pose varies significantly or the viewing angle becomes complex, the projected 3D bounding box may become imprecise or inefficient. The effectiveness of the method should be further validated on datasets with more diverse and challenging probe poses.
- The effect of the different threshold $p\%$ for Gaussian;s probability density utilized in Eq. 12 on time efficiency should be discussed.

3. **Engineering implementation and open-sourcing**: Engineering implementation is an important contribution of this work. It would be valuable to know whether the code will be open-sourced in the future.

4. **Use of heuristics**: Heuristic methods are discussed in Sec. 3.1. However, it is unclear whether any heuristics are actually employed in the proposed method. Clarification on this point would be helpful.

### Questions
Refer to the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3
