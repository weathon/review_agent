# Diffusion-Based, Data-Assimilation-Enabled Super-Resolution of Hub-height Winds

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
High-quality observations of hub-height winds are valuable but sparse in space and time. Simulations are widely available on regular grids but are generally biased and too coarse to inform wind-farm siting or to assess extreme-weather-related risks (e.g., gusts) at infrastructure scales. To fully utilize both data types for generating high-quality, high-resolution hub-height wind speeds (tens to ~100m above ground), this study introduces WindSR, a diffusion model with data assimilation for super-resolution downscaling of hub-height winds. WindSR integrates sparse observational data with simulation fields during downscaling using state-of-the-art diffusion models. A dynamic-radius blending method is introduced to merge observations with simulations, providing conditioning for the diffusion process. Terrain information is incorporated during both training and inference to account for its role as a key driver of winds. Evaluated against convolutional-neural-network and generative-adversarial-network baselines, WindSR outperforms them in both downscaling efficiency and accuracy. Our data assimilation reduces WindSR's model bias by approximately 20% relative to independent observations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes WindSR, a diffusion-based super-resolution downscaler that conditions on a blended field (observations + simulation via a Gaussian soft-bleed mask with a dynamic impact radius) plus terrain. Experiments claim better accuracy and efficiency than CNN/GAN baselines, and say data assimilation cuts model bias by 10–20%.

### Strengths
The paper targets an important problem: high-resolution estimation of hub-height wind speed, a field that is highly variable yet sparsely observed—making it a natural and impactful application scenario for data assimilation.

It integrates data assimilation with super-resolution by using an observation–simulation blended field as the conditioning input for diffusion denoising, thereby addressing fine-scale detail enhancement and upstream bias correction simultaneously.

Compared with prior work, it introduces a dynamic impact-radius strategy, which improves predictive accuracy when assimilating observational data.

### Weaknesses
#### 

- The major weakness of this work lies in the evaluation. Evaluation feels thin for both SR and DA:
  - Super-resolution baselines are limited to older CNN/GAN models (SRCNN, ESRGAN). Missing stronger, more recent downscalers, especially diffusion/transformer ones (e.g., CorrDiff [1]). Also missing a traditional dynamical downscaling baseline. At least one modern AI method **or** one traditional method would make the comparison more solid. 
  - Data assimilation baselines are also SRCNN/ESRGAN, which are SR models by design. It’s unclear how they’re adapted for assimilation. Classical DA (e.g., 3DVar, 4DVar) isn’t included, and newer AI DA methods (e.g., score-based DA [2], latent DA [3], VAE-Var[4], etc.) aren’t compared. Again, the authors are recommended to include at least one modern AI **or** one traditional DA baseline.
  - It’s not clear whether baselines like SRCNN/ESRGAN also use terrain as input. Without that, the comparison may be skewed. 
  - Although the abstract and introduction highlight WindSR’s efficiency, the paper provides neither theoretical analysis nor empirical evidence to substantiate this claim. Additional clarification and/or experiments are needed.
- The idea of using terrain deviation and speed deviation to determine the dynamic impact radius is interesting, but the thresholds based on terrain and speed deviation aren’t justified, and there’s no ablation/sensitivity to show how these choices affect assimilation quality.

[1] Mardani, Morteza, et al. "Residual corrective diffusion modeling for km-scale atmospheric downscaling." Communications Earth & Environment 6.1 (2025): 124.

[2] Rozet, François, and Gilles Louppe. "Score-based data assimilation." Advances in Neural Information Processing Systems 36 (2023): 40521-40541.

[3] Melinc, Boštjan, and Žiga Zaplotnik. "3D‐Var data assimilation using a variational autoencoder." Quarterly Journal of the Royal Meteorological Society 150.761 (2024): 2273-2295.

[4] Xiao, Yi, et al. "VAE-Var: Variational autoencoder-enhanced variational methods for data assimilation in meteorology." The Thirteenth International Conference on Learning Representations. 2025.

### Questions
- On what basis does WindSR claim an efficiency edge? Please add details (e.g., training/inference time, FLOPs, memory, throughput) and supporting experiments.

- Why pick only CNN/GAN baselines instead of recent methods like CorrDiff? What drove that choice (availability, compute, domain fit)?

- In Table 3, SRCNN/ESRGAN trends vs. impact radius look similar to **WindSR w/o Terrain**. Did your SRCNN/ESRGAN runs include terrain as a conditioning input? If not, could you provide results **with** terrain to make the comparison fair?

- How were the terrain/speed deviation thresholds for the dynamic impact radius chosen? Please add sensitivity or ablation studies.

- There are several inconsistencies and minor errors in notation and figure/text presentation that impede clarity. Please standardize and correct the following:

  - Use a consistent notation for dimensions, e.g., 128×128 (not “128x128” and “128 \times 128” interchangeably).
  - Unify model names, e.g., ESRGAN (not “ESGAN” in some places).
  - Line ~300: change “showed in Figure 3” to “shown in Figure 3.”
  - Figure 4: add units for the x-axis.

### Soundness
1

### Presentation
2

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
This paper presents WindSR, a novel framework combining diffusion-based super-resolution and data assimilation to downscale hub-height wind fields for wind-energy and climate applications. The method addresses a key challenge in wind modeling: limited high-resolution observations and biases in numerical simulations. The contributions include using sparse observations with coarse simulation data using DDPMs. The authors also introduce a novel adaptive Gaussian-kernel blending mechanism that adjusts observation influence based on terrain and wind-speed variability. Lastly, the work contributes terrain-aware conditioning during training and inference for physical realism. The authors include comprehensive experiments with WTK and HRRR against CNN- and GAN-based baselines.

### Strengths
* there is a clear, practically grounded motivation with the need for hub-height wind fields and integrating data assimilation with the sparse data fields
* the dynamic impact-radius blending is a clever and physically motivated way to control spatial assimilation influence
* the authors benchmark against strong baselines (SRCNN, ESRGAN), include ablations for DA and terrain conditioning, and validate using * both synthetic and real-world weather station data. these experiments achieves strong improvements against baselines, particularly in complex terrain and at higher wind speeds
* the formulation of DA within the diffusion process seems conceptually sound and is consistent mathematically

### Weaknesses
* Figure 1 shows an overview of the DDPM process, but it would be more beneficial to have a clear visual overview of the model pipeline (where/how are the DA, terrain conditioning, and diffusion steps integrated? make this clearer in Figure 3)
* similarly, it would be helpful to have a diagram for Algorithm 1 (even in the appendix)
* quantitative comparisons across Table 2 and Figure 4 could benefit from uncertainty estimates in some form
* in the appendix, it would be good to include a brief discussion of computational cost/runtime scaling…efficiency is another dimension that’s important to consider in the practical deployability of these methods
* it would be nice to include some exploration of how WindSR generalizes to unseen regions or across long temporal sequences if possible
* there’s no explicit comparison between fixed vs. dynamic assimilation radii beyond Table 3; further visualization of their spatial influence would strengthen the claim
* in the ablation studies, the contribution of terrain conditioning could be isolated more rigorously with additional statistical metrics
* to more robustly evaluate the physical fidelity of the predictions, consider using additional evaluation frameworks such as generating kinetic energy spectra from the wind fields
* include a discussion of why ground truth data (i.e. from HRRR may come with its own biases); also, the few tower data could at least serve for independent verification, even if not enough for training
* grammar fixes (“observations effectively influences” → “observations effectively influence”; “showed in Figure 3” → “shown in Figure 3”; “data assimilation enabled” → “data assimilation-enabled”)
* formatting fixes (inconsistent citations across the paper are a bit sloppy, use \citep and \citet instead of \cite; inconsistent spacing around em dashes)
* redundant citation formatting (Huang et al. 2024 appears duplicated in the reference list), suggests the use of LLMs for citation generation
* include more explicit comparisons with more diffusion-based models (see SwinRDM from Chen et al. 2023 and DiffDA from Huang et al. 2024)
* Section 2 would benefit from a brief commentary on neural operator methods or score-based data assimilation 
conclude the related works discussion with explicit details on where WindSR fits in and the novelty gap this work fills
the literature review on diffusion models for super-resolution is lacking; include other diffusion-based papers (for example, see Xu et al. 2025)
* if more space is needed, Section 3.1 can be shortened to provide a very high level overview of DDPMs

### Questions
* have you evaluated how WindSR performs when applied to regions not represented in training (e.g., different climate zones or surface roughness regimes)?
* Table 3 compares fixed and dynamic radii numerically, but could you provide a visualization or qualitative example illustrating how dynamic blending modifies local fields?
* the ablations show gains from including terrain, but could you isolate this contribution more rigorously? for instance, how does terrain conditioning affect bias reduction or spatial spectra?
* since HRRR itself is a modeled dataset, how do you account for its inherent biases when treating it as “ground truth”? could the limited tower data, even if sparse, serve as an independent verification set?
* do you plan to release the WindSR implementation and preprocessed datasets (WTK + HRRR) for reproducibility?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes the WindSR model, combining data assimilation with diffusion models for super-resolution downscaling of hub-height winds. Based on DDPM, it merges sparse observations and simulation fields via dynamic radius, integrating terrain info in training and inference. Through comparative experiments, the advantages of the model over CNN and GAN methods are verified.

### Strengths
1. WindSR combines data assimilation with diffusion models for hub-height wind super-resolution, merging sparse observations and simulation fields via dynamic radius to reduce model bias and address sparse observation issues.
2. WindSR balances super-resolution accuracy and wind field uncertainty modeling, producing results with stronger physical consistency and better adaptation to wind field dynamic features.

### Weaknesses
1. This work mentions that "Over complex terrain, winds vary rapidly and remain challenging to simulate in numerical models", but the mechanism of introducing terrain information is relatively vague. How the introduction of terrain information specifically affects the establishment of wind speed fields is a question that requires detailed explanation.
2. The baselines compared in the paper are only traditional CNN and GAN, without considering off-the-shelf super-resolution models proposed in recent years, including diffusion-based assimilation and super-resolution methods, as well as some physics-informed methods.
3. This work only evaluates the model through general metrics and lacks specific assessments of key physical characteristics of wind fields, such as extreme values and wind shear.

### Questions
1. This work only validates the model based on WTK and HRRR datasets. Has the proposed model's performance been tested on datasets from other climate zones?
2. Has the paper tested using architectures other than SR3 as the backbone network? For example, will the performance be further improved when using transformer-based SR as the backbone?
3. Has the paper conducted a comparative analysis of resource consumption during training and inference process with existing methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper applies a diffusion model with data assimilation for super-resolution downscaling of hub-height winds.

### Strengths
- Integrating data assimilation for super-resolution downscaling of hub-height winds.

### Weaknesses
- Methodological novelty is limited. There has been a lot of work on data assimilation and diffusion models. Clarifying the novelty would increase the chance of getting accepted by ICLR rather than simply applying some models.
- Lack baselines. You may consider including more baseline methods for comparisons beyond simple CNN and GAN.
- It would be great to test the generalization by including another study domain.
- No code provided.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
