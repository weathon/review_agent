# Recover Cell Tensor: Diffusion-Equivalent Tensor Completion for Fluorescence Microscopy Imaging

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Fluorescence microscopy (FM) imaging is a fundamental technique for observing live cell division—one of the most essential processes in the cycle of life and death. Observing 3D live cells requires scanning through the cell volume while minimizing lethal phototoxicity. That limits acquisition time and results in sparsely sampled volumes with anisotropic resolution and high noise. Existing image restoration methods, primarily based on inverse problem modeling, assume known and stable degradation processes and struggle under such conditions, especially in the absence of high-quality reference volumes. In this paper, from a new perspective, we propose a novel tensor completion framework tailored to the nature of FM imaging, which inherently involves nonlinear signal degradation and incomplete observations. Specifically, FM imaging with equidistant Z-axis sampling is essentially a tensor completion task under a uniformly random sampling condition. On one hand, we derive the theoretical lower bound for exact cell tensor completion, validating the feasibility of accurately recovering 3D cell tensor. On the other hand, we reformulate the tensor completion problem as a mathematically equivalent score-based generative model. By incorporating structural consistency priors, the generative trajectory is effectively guided toward denoised and geometrically coherent reconstructions. Our method demonstrates state-of-the-art performance on SR-CACO-2 and four real \textit{in vivo} cellular datasets, showing substantial improvements in both signal-to-noise ratio and structural fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on a particular imaging inverse problem in fluorescence microscopy, which is formulated as a tensor completion problem. The authors propose integrating Tucker decomposition into the ADMM framework and compare the ADMM iterations to a diffusion process. The efficacy of this method is guaranteed by recovery theory. The proposed method demonstrates superior performance in super-resolution and denoising tasks compared to other methods.

### Strengths
1. By leveraging Tucker decomposition, the method effectively exploits the intrinsic low-rank property of the data, which largely accounts for the observed performance improvement.
    
2. The method is supported by theoretical guarantees and use of the ADMM framework enhances its interpretability.

### Weaknesses
1. The method appears to be incremental, as the integration of tensor decomposition within the ADMM framework has already been explored in several existing papers (e.g., [1]).
   
 2. While the comparison between the ADMM optimization process and diffusion-based denoising is intuitive given their shared iterative patterns, a key distinction lies in their underlying priors. Diffusion models utilize an implicit prior learned from datasets. In contrast, low-rankness serves as an explicit structural prior. These two types of priors have their own distinct advantages. Therefore, the paper requires a discussion on why simply swapping these priors is a reasonable approach, and why the chosen explicit prior is more suitable for this specific problem than an implicit diffusion-based one.

[1].Yuan, L., Li, C., Mandic, D., Cao, J., & Zhao, Q. (2019). Tensor  Ring Decomposition with Rank Minimization on Latent Space: An Efficient  Approach for Tensor Completion. *Proceedings of the AAAI Conference on Artificial Intelligence*, *33*(01), 9151-9158.

### Questions
1. The paper states that it approximates the complex, true degradation process using a combination of downsampling and noise. However, it is unclear whether the datasets used in the experiments were generated using this approximated degradation or a true degradation process. This point requires clarification.
   
 2. The paper mentions 'Structural Priors in Biological Imaging'. However, the only prior explicitly integrated into the model appears to be low-rankness. The authors should justify this choice and discuss whether other priors specific to biological imaging, or more general image priors (such as continuity or nonlocality [2]), were also considered.
   
3. Clarity is needed regarding the DDPM's training methodology. The paper should explicitly state in the main text whether the DDPM was pre-trained or trained from scratch using paired data.

[2].Liu Y, Yang Y, Cui Z X, et al. Patch-based Reconstruction for Unsupervised Dynamic MRI using Learnable Tensor Function with Implicit Neural Representation[J]. arXiv preprint arXiv:2505.21894, 2025.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents a novel tensor completion framework for fluorescence microscopy (FM) imaging, addressing nonlinear signal degradation and incomplete observations. FM imaging with equidistant Z-axis sampling is modeled as a tensor completion task under uniform random sampling. The framework establishes the theoretical lower bound for exact 3D cell tensor recovery and reformulates the problem as a score-based generative model. Structural consistency priors guide the generative process toward denoised and geometrically coherent 3D reconstructions.

### Strengths
1. Well-written and well-organized, with comprehensive supplementary material.
2. Introduces a tensor completion approach specifically tailored to fluorescence microscopy imaging, addressing a novel research problem that has not been extensively explored in the literature.
3. Derives the theoretical lower bound for exact 3D cell tensor recovery and reformulates tensor completion into a score-based generative modeling framework.

### Weaknesses
### Mismatch Between Noise Model and Physical Imaging Process

In the manuscript, the observation is formulated as  
Y_\Omega = X_\Omega + E_\Omega

(an additive sparse noise model), which is a convenient and commonly used formulation.  
However, in fluorescence microscopy, the primary degradations typically involve **Poisson noise** (due to photon counting), **system PSF** (blur or convolution), **signal attenuation and scattering**, and even **multiplicative effects**.  
Therefore, such a simple additive sparse model may not adequately capture the real imaging process.  

If the equivalence or performance analysis of the proposed method relies on the assumption of *additive Gaussian plus sparse noise*, the paper should **explicitly state this assumption** and **discuss its implications and robustness** when the actual noise deviates from it.  Otherwise, the **generalizability and practical applicability** of the method may be limited.


### Insufficient Experimental Evaluation
The paper lacks analysis and comparison with the latest method **MicroDiffusion** : Implicit Representation-Guided Diffusion for 3D Reconstruction from Limited 2D Microscopy Projections.

How is the **Z-axis sampling rate**  determined, and how does the model perform under different sampling rates?

### Questions
Please see the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper reframes 3D live-cell fluorescence microscopy reconstruction as a tensor completion problem rather than an ill-posed inverse imaging task. The authors show that sparse axial sampling can be modeled as missing tensor entries, enabling recovery via low-rank tensor priors and sparse-noise decomposition. They further prove a formal equivalence between the resulting alternating minimization algorithm and a structurally guided conditional diffusion process, linking classical low-rank projection with modern generative denoising dynamics. The method provides theoretical recovery guarantees and achieves state-of-the-art performance on multiple real biological datasets, producing cleaner, more faithful volumes than existing inverse-problem and deep learning baselines.

### Strengths
- The paper reframes 3D fluorescence microscopy reconstruction as cell-tensor completion rather than a traditional inverse problem, and establishes a provable connection between low-rank tensor recovery (via ADMM) and conditional score-based diffusion dynamics. This theoretical bridge is genuinely novel and goes beyond heuristic model design, giving a principled interpretation of diffusion priors in biological imaging.

- The paper provides clear recovery guarantees under missing-data and sparse-noise settings, and designs an interpretable ADMM solver with separated low-rank and sparse components—well aligned with actual microscopy noise physics (e.g., shot noise and acquisition sparsity). The algorithm is lightweight, transparent, and grounded in theory rather than black-box training.

- Experiments on multiple live-cell datasets show state-of-the-art performance in PSNR/SSIM/LPIPS and visibly better membrane continuity and cellular morphology preservation compared to supervised GAN-based and reconstruction-based baselines. The method avoids hallucinations and preserves structural fidelity, particularly important for biological interpretation.

### Weaknesses
- The paper highlights the diffusion-equivalence result but does not compare against recent unsupervised or self-supervised generative FM restoration methods (e.g., score-based microscopy denoising, inverse-consistent diffusion models, diffusion-based deconvolution pipelines). Without such comparisons, it is hard to quantify whether the proposed tensor-based approach benefits primarily from low-rank structure or from the implicit generative prior interpretation. Adding baselines like self-supervised diffusion denoisers or Plug-and-Play diffusion priors would strengthen the empirical story.

- The paper does not provide sufficient analysis of rank selection, regularization weights, and sparse-noise thresholding. Since these choices likely impact geometry fidelity and hallucination risk, a systematic study (e.g., effect of Tucker rank on membrane sharpness and artifact suppression.

### Questions
- How the method behaves when the underlying cell volume exhibits high intrinsic complexity that is not well-approximated by a low-rank tensor (e.g., dense organelles, cytoskeletal networks, neurites)? 

- How sensitive is performance to the choice of Tucker ranks, regularization parameters, and sparse-noise thresholds? A small ablation or rule-of-thumb guideline would help practitioners avoid overfitting or texture loss.

- While the method avoids GAN-like hallucinations, could low-rank priors oversmooth fine biological details? A controlled stress test—e.g., synthetic volumes with known thin filaments.

### Soundness
3

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
This paper proposes a novel tensor completion framework for 3D fluorescence microscopy imaging that overcomes the limitations of traditional inverse problem methods. The authors reformulate cell volume recovery as a robust low-rank tensor completion problem, establishing theoretical recovery guarantees and proving mathematical equivalence to conditional diffusion models. Using Tucker decomposition with structural consistency priors, the method effectively handles the inherent challenges of fluorescence imaging: nonlinear degradation, incomplete Z-axis observations due to phototoxicity constraints, and high spatially-varying noise. Experiments on SR-CACO-2 and live C. elegans datasets demonstrate state-of-the-art performance with superior noise robustness and temporal consistency, achieving significant improvements in both signal fidelity and structural preservation for biological applications.

### Strengths
1. The paper establishes a novel mathematical equivalence between Tucker-based tensor completion and conditional diffusion models, providing rigorous theoretical guarantees with clear derivations for exact recovery under sparse sampling conditions inherent to fluorescence microscopy.
2.The method achieves robust 3D cell reconstruction without requiring paired high-resolution ground truth data, instead leveraging low-rank structure and sparse noise decomposition directly from incomplete noisy observations.
3.The approach demonstrates state-of-the-art quantitative results across multiple datasets while maintaining exceptional temporal consistency and structural fidelity throughout extended time-lapse sequences (2700s-5400s) under degrading signal conditions.

### Weaknesses
1. Since the paper focuses on biological image reconstruction, relying solely on visual quality metrics lacks sufficient persuasiveness—high PSNR does not guarantee biological correctness, as reconstructions may appear visually plausible yet contain biologically inaccurate structures. Based on Figure 3 and other renderings, the cycle+IPG method gives me a better overall impression, and it performs better in reproducing details compared to the method proposed in this paper.

2. The paper lacks critical discussion on hyperparameter selection, particularly the Tucker ranks (r₁, r₂, r₃) and sparse regularization weight λ₁, providing neither specific values, selection strategies, nor sensitivity analysis, which severely compromises reproducibility and practical applicability.

3. The paper suffers from notational inconsistencies and insufficient explanations—parameter θ in Equation 1 is undefined, the transition between Equations 2-3 lacks clear explanation of tensor relationships (T vs. Y), and similar issues with forward symbol references and inconsistent variable naming persist throughout.

### Questions
1. During testing, how do you determine the rank for a new unseen volume—do you estimate it per-sample or use a fixed value?

2. For λ₁: What value(s) did you use? Does it change across datasets or noise levels? If I choose r = 15 instead of r = 25, or λ₁ = 0.05 instead of 0.5, how much does PSNR drop?

3. Could you provide a unified notation table?

### Soundness
3

### Presentation
2

### Contribution
2
