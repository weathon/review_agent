# Spectral Domain Neural Reconstruction for Passband FMCW Radars

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
We present SpINR, a novel neural framework for high-fidelity volumetric reconstruction of small, near-field tabletop objects using Frequency-Modulated Continuous-Wave (FMCW) radar. Traditional radar imaging techniques often apply FFT as a black-box post-processing step, discard phase information, and require dense aperture sampling, leading to limitations in sub-centimeter reconstruction and generalization. Our core contribution is a fully differentiable frequency-domain forward model that captures the complex radar response using closed-form synthesis, paired with an implicit neural representation (INR) for continuous volumetric scene modeling. Unlike time-domain baselines and magnitude-only spectral pipelines, SpINR directly supervises the complex frequency spectrum from raw 1D chirps acquired along an arbitrary cylindrical aperture, preserving phase-sensitive spectral spectral fidelity while drastically reducing computational overhead. Additionally, we introduce sparsity and smoothness regularization to disambiguate sub-bin ambiguities that arise at high carrier frequencies and fine range resolutions. Experimental results show that SpINR significantly outperforms both classical and learning-based baselines, with a 52.6\% improvement in reconstruction quality and 32\% improvement in latency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a method for 3D volumetric reconstruction from FMCW radar data.
The approach models the radar signal in the frequency domain using a DFT-based formulation, and employs a neural network that predicts scattering intensity by being trained on the complex frequency response, with regularization terms for smoothness and sparsity. The main claim for novelty and performance gain stem from modeling radar physics directly in the spectral domain, rather than relying on time-domain representations.

### Strengths
1. The paper’s application of volumetric rendering to radar data is interesting. 
2. The inclusion of smoothing and sparsity regularization to address radar spectral issues is an effective way to enhance the method’s stability and robustness.

### Weaknesses
The core novelty claimed by the paper lies in transforming the FMCW signal to the frequency domain and modeling the reflectivity physics in the spectral domain. However, this transformation is a standard step in FMCW processing, where the beat signal is typically converted to the spectral domain to achieve signal compaction and improved signal-to-noise ratio. Since this procedure is common practice, and the rendering process itself is also not new, I find the overall level of novelty limited, both in general and specifically from a machine learning perspective.

### Questions
Given that spectral-domain processing is standard for FMCW signals, what is the core novelty of the paper?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a volumetric reconstruction framework for FMCW radar using a neural network with a forward model defined in the frequency domain via the Discrete Fourier Transform (DFT). Since FMCW radar image inherently lies in the frequency domain, defining the loss in that domain is reasonable.

### Strengths
1. The paper proposes a volumetric reconstruction framework for FMCW radar using a neural network with a forward model defined in the frequency domain via the DFT.

2. Since FMCW radar data inherently lies in the frequency domain, defining the loss in that domain is reasonable and conceptually consistent.

3. The overall idea is straightforward and easy to follow.

### Weaknesses
1.	Paper structure and clarity:
The manuscript is not well structured. A large portion of the content focuses on FMCW vs. pulse radars, which is not directly relevant to the core contribution.
The discussion on beat-signal challenges is repetitive and does not provide new insight.
There are noticeable typos throughout the manuscript, which negatively impact readability.
2.	Overemphasis on well-known theory:
The DFT formulation and explanation of spectral leakage are standard knowledge and could be shortened significantly.
Starting directly from the section “Forward Model with Spectral Synthesis” would already give sufficient context.
3.	Lack of meaningful comparisons:
The paper does not compare against sparse-reconstruction baselines or state-of-the-art deep learning methods for inverse problem.
TF-TS, TF-SS, and BP alone are insufficient to demonstrate the advantages of the proposed approach.
NVR and RadarHD are not suitable comparisons since they address different tasks and use different signal models.

4.	No real-world validation:
All experiments are simulation-based. 
Without real data, it is difficult to assess practical effectiveness, robustness, or applicability in real FMCW systems.

5.	Missing In-depth discussion:
There is no analysis of how start frequency ( f0 ) and other radar configuration impact reconstruction quality.
The influence of DFT size (Dirichlet kernel width) on model performance is not explored.
These analyses could provide deeper insight and potentially lead to better design choices.

### Questions
1.	How robust is the proposed model to different radar configurations (e.g., start frequency f0, bandwidth B, chirp rate, and DFT size/Dirichlet kernel width)? If the model is trained on one configuration, can it generalize to others, or would re-training be required?
2.	Will the method work with real FMCW radar measurements, and do the authors plan to validate on hardware?
3.	Are there reasons sparse-reconstruction or modern neural inverse methods were not included in the comparisons?

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
This paper introduces SpINR, a novel neural framework for high-fidelity volumetric reconstruction from Frequency-Modulated Continuous-Wave (FMCW) radar signals . The authors identify that existing time-domain neural models suffer from optimization instability and computational inefficiency . The core contribution is a fully differentiable, frequency-domain forward model that analytically synthesizes the complex radar spectrum, explicitly accounting for physical phenomena like spectral leakage. This model is paired with an implicit neural representation (INR) to model the continuous 3D scene. The authors also introduce sparsity and smoothness regularizations to disambiguate the reconstruction, particularly at high carrier frequencies . Experiments conducted on simulated data demonstrate that SpINR surpasses both classical and learning-based baselines in reconstruction quality and computational performance

### Strengths
* The paper's primary strength lies in its novel formulation of a fully differentiable, frequency-domain forward model. This is a significant contribution as it directly addresses the well-known instability and inefficiency issues of supervising FMCW beat signals in the time domain. The "analysis-by-synthesis" approach is moved entirely to the spectral domain, which is more physically appropriate for FMCW radar.

* The paper is clearly written and well-motivated. The authors provide a thorough analysis of the challenges in FMCW signal modeling, such as the ill-conditioned nature of time-domain signals and the problem of spectral leakage, which their closed-form model analytically addresses.

### Weaknesses
- The most significant weakness of this paper is the exclusive reliance on simulated data for all experiments. While the simulation is based on a commercial sensor's parameters, a considerable gap exists between idealized RF simulations and real-world radar data, which is affected by complex noise, multi-path interference, and hardware-specific artifacts not perfectly captured by the model. The lack of validation on real-world RF data makes the current experimental results insufficient to prove the method's practical applicability. This is my primary reason for leaning towards rejection. I strongly urge the authors to provide reconstruction results on real-world data during the rebuttal period.

- The paper pairs a standard hash-encoding-based INR with the proposed physics-based forward model. However, the analysis of this architectural choice is limited. It is unclear if this generic architecture is optimal for representing the complex scattering fields of FMCW radar. Specifically, passband signals with high carrier frequencies introduce rapid phase variations. Does the hash encoding have sufficient capacity and resolution to represent these high-frequency spatial-phase relationships without aliasing, or does the burden of reconstruction fall entirely on the smoothness/sparsity priors?

- The method relies on smooth and sparsity regularizations to resolve sub-bin ambiguities, which become more severe at higher carrier frequencies. The paper does not provide an analysis of the model's sensitivity to the hyperparameters $\beta$ and $\gamma$. How critical are these weights to the final reconstruction quality? Furthermore, do these hyperparameters need to be manually re-tuned for different radar parameters (e.g., a different starting frequency $f_0$ or bandwidth $B$), or is the model robust to such changes? A lack of robustness here would limit the method's generalizability across different radar sensors.

### Questions
See Weaknesses

### Soundness
3

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
This paper presents a novel neural volumetric rendering approach for 1D chirps of FMCW radars which operates in the intrinsic frequency domain of the radar signal. It combines a carefully crafted, DFT-aware forward model for FMCW radar sensors with an implicit volumetric scene representation of continuous reflectivity which is predicted by a neural network and used by the forward model to recover radar measurements. The proposed method, SpINR, is evaluated on simple object-centric scenes against baselines and two related competing methods for acoustic signal rendering and radar point cloud enhancement and demonstrates strong performance in these comparisons and satisfactory qualitative results.

### Strengths
1. Novel frequency-domain forward model for FMCW radars. The authors propose a rigorous, theoretically sound forward sensor model for 1D raw chirps of FMCW radars, which incorporates the full DFT computation that is involved in the range-bin-dependent measured signal formation. This model crucially treats the full, complex radar signal, including both magnitude and phase. It is fully differentiable and thus inherits the benefits of neural rendering to the proposed neural representation in terms of indirect supervision of the latter via novel measurement synthesis.

2. Smooth introduction to the topic and presentation of the method. Even though the examined topic of FMCW radar is quite demanding and heavy on spectral theory, the authors have successfully introduced it in a gentle fashion and supported the presentation of their own method on this primer. Despite the complex nature of the notation, the authors have made proper use of it and neatly defined the various quantities at the correct places.

3. Good quantitative and qualitative performance. In the considered, niche context of simple object-centric scenes with rather small ranges from the sensor of less than a meter, both the geometric reconstruction metrics and the image-level synthesis metrics of the method are substantially better than baselines. Performance is also substantially better than that of the two selected competing methods. Moreover, the qualitative results are clearly superior to those of baselines, with good-quality reconstructions.

### Weaknesses
1. Limitation of experimental evaluation to niche scenes and sensor configurations. The authors have constrained their evaluation to the simplistic case of 1D radar chirps and cylindrical apertures, while real-world applications feature at least 2D range-azimuth measurements with scanning configurations. This choice has also limited the range of compared methods in the experiments, excluding related works such as Borts et al. (2024a) and Huang et al. (2024), which are very closely related to the presented approach. The latter works have also demonstrated high-quality results on much more complex real-world scenes, such as urban or indoor scenes, while the results presented in this paper are exclusively on artificial and simplistic object-centric scenes, which avoid the difficult-to-handle multi-path interference effect. Without comparison with the above related works, the advantage of the proposed method compared to previous approaches and FMCW radar neural models remains questionable.

2. No modeling of directive scattering properties of the scene. The neural field which is learned by the proposed model is formulated as $\sigma(\mathbf{x})$, i.e. as only depending on the position of the scatterer in the scene. The viewing direction $\omega$ is not included in the model, even though it could be used to model view-dependent effects, which are commonly consider in radar models under a directivity term. This omission significantly restricts the representational power of the model, as the same scatterer can result in different signal intensities when viewed from different directions by the sensor.

### Questions
Can the authors apply their method to more complex, real-world scenes, potentially extending their model beyond mere 1D chirps, and compare against state-of-the-art neural field approaches for FMCW radars on these scenes? (cf. Weakness 1)

### Soundness
4

### Presentation
3

### Contribution
3
