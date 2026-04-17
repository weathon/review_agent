# Unsupervised Radar Point Cloud Enhancement via Arbitrary LiDAR Guided Diffusion Prior

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
The angular resolution of automotive radar is fundamentally limited by the Rayleigh criterion and the number of antennas, resulting in sparse radar point clouds that hinder high-precision perception. Recent approaches address this limitation by training discriminative or conditional generative models on paired radar-LiDAR data, learning an explicit mapping from radar measurements to LiDAR-like outputs. However, these methods are sensitive to calibration errors and fail when LiDAR data is unavailable. In this work, we propose the first *cross-modality, unsupervised* radar point cloud super-resolution method by formulating radar enhancement as a Bayesian inverse problem. We explicitly model the radar sensing process with a differentiable forward operator, defining the measurement likelihood, and combine it with a LiDAR-trained latent diffusion model that captures the distribution of physically plausible point clouds. During inference, we perform posterior sampling that integrates this generative prior with the radar likelihood, yielding reconstructions that are both geometrically rich and strictly measurement-consistent — without requiring paired radar-LiDAR data. Experiments on the RADIal dataset show that our approach matches the performance of supervised mapping-based methods, while generalizing significantly better to unseen datasets such as K-Radar. These results demonstrate the effectiveness of combining physics-based modeling with distributional priors, offering a robust and practical solution for radar perception in real-world deployments. Our code is available at https://anonymous.4open.science/r/RadarINV2025.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an unsupervised radar point-cloud enhancement method framed as a Bayesian inverse problem. It combines a differentiable radar forward model (likelihood) with a LiDAR-trained diffusion prior to sample dense, geometrically plausible point clouds without paired radar–LiDAR supervision. Claimed contributions include the formulation, a practical posterior sampling procedure, and experiments showing performance near supervised baselines on RADIal with improved cross-dataset generalization (e.g., to K-Radar).

### Strengths
1. This paper frames radar enhancement as a Bayesian posterior sampling problem and leverages a LiDAR-trained diffusion prior to inject strong geometric priors without paired radar–LiDAR supervision—a creative and non-obvious combination.
2. This paper achieves near-supervised performance on benchmarks, indicating a sound design with practical impact by reducing reliance on costly paired data and calibration.

### Weaknesses
Problem framing and positioning.
The method reads more like super-resolution on range–azimuth heatmaps than “point-cloud enhancement.” Please clarify terminology, the signal domain where inference occurs, and the point-cloud generation step. In addition, lines 40–41 attribute sparsity mainly to limited angular resolution, whereas in practice sparsity largely stems from point cloud generation methods (e.g., CFAR) that retain only peak returns; the pre-CFAR range–azimuth–Doppler tensor is much richer. Revise the discussion to reflect this and specify where your method interfaces with this pipeline. Finally, the title’s “arbitrary” claim is not operationalised—either remove it or define and support it rigorously.

Readability and technical clarity.
It is unclear how the masking strategy enables end-to-end training and reduces computation. Provide a complexity analysis, comparisons to alternative schemes, and ablations. For a broad audience, add a concise Preliminaries section on radar sensing (range/angle/Doppler FFTs, data cubes, CFAR, notation). Derivations are also hard to follow: detail the steps from Eq. (4) → (5) and the extension (5) → (1) in the supplement, stating assumptions and approximations.

Applicability and evidence.
Generated radar points differ noticeably from LiDAR “ground truth.” Please explain the physical plausibility of these differences and what application value they provide. Strengthen the paper with downstream experiments (e.g., detection, segmentation, occupancy) to test whether the “enhanced” radar improves task performance.

### Questions
1. The Chamfer Distance unit is unspecified (meters? normalized voxel units?)
2. How about the efficiency of the current method?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an unsupervised radar point cloud enhancement method that leverages a LiDAR-trained latent diffusion model as a prior. The goal is to enhance radar point clouds without requiring paired radar–LiDAR data. The authors formulate the task as a Bayesian inverse problem, combining a physics-based radar forward model with a generative diffusion prior to reconstruct dense, LiDAR-like radar point clouds. Experiments on the RADIal and K-Radar datasets demonstrate that the proposed method achieves performance comparable to supervised approaches and significantly outperforms traditional methods such as CFAR across multiple metrics.

### Strengths
1. The reviewer believe authors' motivation is strong – unsupervised radar enhancement is indeed essential becuase paired radar–LiDAR data is scarce and fragile. 
2. Novel formulation regarding basyesion inverse problem and unsuperived learning with diffusion for radar enhanment.
3. Authors provide their code and showing that their performance outperfomnce CFAR in the cross dataset experiment, which is widely use in almost all radar systems.

### Weaknesses
1. The paper relies on several strong modeling assumptions—particularly the use of Gaussian noise and an idealized, linear radar sensing model and some are not clearly stated or justified. These assumptions may not accurately reflect the statistical or physical characteristics of real radar signals. In the reviewer’s opinion, assuming Gaussian noise before the FFT stage is reasonable and common practice, as it represents thermal receiver noise. However, for radar tensors from datasets such as RADIal and K-Radar, subsequent processing steps like FFT beamforming introduce additional structured artifacts, including deterministic sidelobes and spectral leakage. The omission effects limits the realism of the forward model and could partly account for the false alarms observed in the qualitative results—most notably on K-Radar, which exhibits stronger sidelobe artifacts.
2. Writing can be improved, for example the symbols $𝜖_𝜃$  and $𝜖_𝛿$ appear interchangeably without explicit definition, while $𝜇_t$ and $Σ_t$ are referenced but not introduced. The update term $𝜇_t − z_t$ uses the hyperparameter $λ$$diff$, which is not defined elsewhere. The parameters $γ$ and $ζ$ are inconsistently described between Algorithm 1 and Table 1. The appendix extends the derivation of the Bayesian update but suffers from notation inconsistencies. Clarifying these definitions and ensuring consistent notation throughout would greatly improve readability
3.  Authors target a 2D range–azimuth slice. Many automotive radars are now 3D/4D (with elevation, Doppler). 2D enhancement limit the potential usage to only in BEV space.
4. Another concern is the lack of detail regarding the CFAR baseline. The paper does not specify which variant of CFAR (e.g., CA-CFAR, OS-CFAR, GO-CFAR, etc.) is used, nor how its parameters are selected. These parameters are typically carefully tuned in practical radar systems, and their performance can vary significantly depending on the setting. Without this information, the reported CFAR results may not represent a fair or optimized baseline, which makes the comparison potentially unreliable.

### Questions
The authors propose a radar enhancement system based on a diffusion model and report quantitative improvements across several metrics against Lidar pc. Chamfer Distance and FID-BEV measure geometry but not physical consistency. There’s no analysis of false positives, detection precision/recall, or uncertainty — all key for radar. Also, it remains unclear how these enhancements affect downstream perception tasks compared with using CFAR point clouds or raw radar data. Since the enhanced results still contain noticeable noise, conducting relevant downstream experiments (e.g., object detection or tracking) would substantially strengthen the paper. My main concern lies in the paper’s overall contribution. Introducing an unsupervised radar enhancement approach is interesting, it is still uncertain whether this direction is worthwhile without stronger evidence. More comprehensive experiments or deeper analysis are needed to convince the community of its significance. 

I believe this is an interesting research direction, and I am open to raising my score if the authors effectively address the concerns outlined above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to mitigate the sparsity of radar points for autonomous vehicles, thereby enhancing high-precision perception. It addresses the limitations of existing works in terms of their sensitivity to calibration errors and reliance on LiDAR data. To this end, the authors proposed a model that leverages a LiDAR-based latent diffusion model to solve for radar enhancement as a Bayesian inverse problem. Highlighted results from their experiment are the generalizability of the proposed model when evaluated across different datasets. Despite some promising results, it comes with significant limitations in both **novelty** and **significance of benefits**, as I will discuss in the sections below. Therefore, it is not considered a paper good enough for acceptance from my perspective.

### Strengths
The paper is overall concise and application-oriented, with a clear presentation of problem statement, motivations, methodology, and experiments. The following lists some key contributions based on my understanding:
1. **Parallel Radar Forward Model**: Motivated by the limitations of the discrete process, the authors introduce a parallel radar forward model in Section 3.2 to apply the Fourier Filter Transformation on BEV grids for continualizations.
2. **Consistency Model**: Authors apply the diffusion model to learn an intermediate representation bridging radar and lidar points $\mathbf{z}_{0}$ and enforce consistency constraints during inference with respect to the range-azimuth function $\mathcal{A}$.
3. **Generalizability**: A significant benefit of the model, as suggested by the authors, is highlighted by the strong generalization capability given by the cross-dataset validation in Section 4.2.3. This fits the expectation with the additional Bayesian inference setting.

### Weaknesses
For this application-oriented paper, limitations of the proposed model are evident in terms of its limited novelty and performance:
1. **Limited Novelty.** To the best of my knowledge, neither the concept of formulating Radar enhancement as a Bayesian inverse problem nor the application of combining Diffusion models in bridging LiDAR and radar is a novel idea compared to existing works. Based on this standpoint, the work is considered an incremental contribution to the field, as it replaces prior models in a Bayesian inverse setting with a learned Diffusion model of LiDAR points. This raises the question about its significance.
2. **Significance of Performance.** With the limited novelty above, I would expect authors to showcase an improved performance with the proposed design. However, as listed in Table 2, the proposed method has improved none of the five key metrics. In particular, it has a relatively worse FID in BEV and UCD. In this sense, I would argue that there are potential improvements to be made in the proposed method, given the increase in computational and memory costs associated with incorporating a Diffusion model.
3. **Generalizability.** Despite the promising generalizability demonstrated in Table 3, I wonder if the model can be efficiently scaled up with additional parameters or new data.

### Questions
1. Equation (6) posed an i.i.d. Gaussian noise for the emission distribution. Have you considered other distributions that may better fit the properties of the features, such as the log-normal or von Mises distribution? 
2. Instead of a Diffusion prior, I wonder if it would be more appropriate to directly learn a unified flow-based model from the distribution of lidar points to the distribution of radar features within a unified BEV perspective?
3. What is the sensitivity of sampling quality with respect to the number of measurement update steps $K$ and sampling steps $T$? If the quality is indeed sensitive to these two hyperparameters, how can we balance the trade-off between performance and efficiency?

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
This paper proposes an unsupervised, cross-modality method for radar point cloud enhancement (super-resolution) to address the sparsity issue caused by physical aperture limitations. Unlike conventional supervised methods that rely on paired radar-LiDAR data, this work formulates radar enhancement as a Bayesian inverse problem.  The core contribution: (1) A differentiable radar forward sensing model (as the measurement likelihood, $p(y|x)$); (2) A latent diffusion model trained only on arbitrary (unpaired) LiDAR data (as the prior, $p(x)$, capturing real-world geometric distributions). During inference, the method performs posterior sampling to generate high-resolution point clouds that are both consistent with the radar measurements and possess LiDAR-like dense geometric structure. Experiments on the RADIal dataset show the unsupervised method achieves performance comparable to supervised approaches, while demonstrating significantly superior generalization on the unseen K-Radar dataset.

### Strengths
1. Unsupervised Formulation: The use of an unsupervised method to improve point cloud reconstruction quality is clearly a promising research direction. I believe this approach is particularly valuable for rare and special scenarios where it is often difficult to find LiDAR point cloud data with a matching distribution for paired training.

2. Generalization Ability: As a direct benefit of its unsupervised nature, the proposed method demonstrates strong generalization capabilities when tested on the unseen K-Radar dataset.

3. Novel Formulation: The paper introduces a novel formulation by framing the task as a Bayesian inverse problem. It effectively combines a physics-based radar forward model (the likelihood $p(y|x)$) with a diffusion prior trained on LiDAR data (the distributional constraint $p(x)$).

### Weaknesses
My specific concerns are as follows:

1. **Poor In-domain Performance:** The method's in-domain results are underwhelming. **Table 2** indicates that the proposed approach offers only a marginal improvement over the traditional CFAR baseline, particularly on the commonly used CD and UCD metrics. More critically, the qualitative results in **Figure 4** show that CFAR can produce significantly more accurate reconstructions in some scenes (e.g., the third row), whereas the proposed method introduces numerous artifacts. This discrepancy suggests the quantitative gains might be illusory, potentially stemming from merely generating a denser cloud of points rather than from successfully integrating a meaningful scene prior. Consequently, the method's performance does not demonstrate a clear and significant advantage, even over the CFAR baseline.

2. Questionable Validity of the Prior: The poor results call the core premise into question: is using a LiDAR point cloud as a prior truly sound for this task? A key implementation detail is omitted: did the authors strictly filter the LiDAR training data to only include points within the radar's specific Field-of-View (FOV)? The paper mentions pre-processing LiDAR to a $512 \times 768$ BEV image covering [-90°, 90°], but it is not explicitly stated if this range was chosen to match the radar's FOV. If out-of-FOV data was used to train the prior, this would introduce a strong, incorrect bias, forcing the model to hallucinate points in areas the radar *cannot* see, which could explain the artifacts and poor metrics.

If the authors can substantially address concern #1 (Poor In-domain Performance) and provide a clear, reasonable explanation for concern #2 (Validity of the Prior), I will consider improving my rating.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
