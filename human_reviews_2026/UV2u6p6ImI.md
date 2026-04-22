# Deep generative priors for 3D brain analysis

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Diffusion models have recently emerged as powerful generative models in medical imaging. However, it remains a major challenge to combine these data-driven models with domain knowledge to guide brain imaging problems. In neuroimaging, Bayesian inverse problems have long provided a successful framework for inference tasks, where incorporating domain knowledge of the imaging process enables robust performance without requiring extensive training data. However, the anatomical modeling component of these approaches typically relies on classical mathematical priors that often fail to capture the complex structure of brain anatomy. In this work, we present the first general-purpose application of diffusion models as priors for solving a wide range of medical imaging inverse problems. Our approach leverages a score-based diffusion prior trained extensively on diverse brain MRI data, paired with flexible forward models that capture common image processing tasks such as super-resolution, bias field correction, inpainting, and combinations thereof. We further demonstrate how our framework can refine outputs from existing deep learning methods to improve anatomical fidelity. Experiments on heterogeneous clinical and research MRI data show that our method produces consistent, high-quality solutions without requiring paired training datasets. These results highlight the potential of diffusion priors as versatile tools for brain MRI analysis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the use of diffusion models as generative priors for solving 3D brain MRI inverse problems, including super-resolution, inpainting, and image refinement. The method combines a pre-trained 3D score-based diffusion prior with explicit forward models to incorporate domain knowledge, enabling the reconstruction of high-quality brain images without paired training data or acquisition parameters. The framework is validated across multiple heterogeneous datasets (clinical, low-field, and pathological MRI), showing improved quantitative metrics and enhanced anatomical fidelity compared to both classical and data-driven baselines.

### Strengths
- The experiments are extensive, covering various datasets and tasks with consistent improvements in image quality and anatomical plausibility over strong baselines.

### Weaknesses
- The main weakness of this work lies in its lack of clear technical novelty. The use of diffusion models as priors for inverse problems has already become a well-established paradigm, with several prior works applying similar formulations to medical and general imaging tasks. In this paper, the authors simply train a standard 3D diffusion model and apply existing posterior sampling methods to a few common MRI problems such as super-resolution and inpainting. There is no apparent methodological innovation or theoretical advancement beyond adapting known techniques to a different dataset. If this work were presented as a benchmark or large-scale evaluation study, it might be reasonable, but for a general ICLR track, the contribution appears incremental and lacks sufficient originality.

- In addition, the likelihood modeling and posterior sampling components rely on heuristic parameter tuning, without principled justification or adaptive estimation. This makes the framework sensitive to task-specific configurations and limits its reproducibility and general applicability. A deeper analysis or ablation on how the likelihood formulation impacts the final performance would be needed.

### Questions
Please see the weakness section.

### Soundness
2

### Presentation
4

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
This paper presents a unified probabilistic framework for 3D brain analysis that bridges the gap between classical and deep learning methods by combining a single, powerful diffusion prior, trained on diverse healthy MRIs , with flexible, task-specific likelihood models at inference time. The authors demonstrate this approach achieves state-of-the-art performance on a range of 3D inverse problems, including super-resolution, bias field correction, inpainting.

### Strengths
1. The writing is clear and easy to follow.

### Weaknesses
1. The work's conceptual novelty is limited, as it primarily synthesizes established components. The core framework of using score-based diffusion models as "plug-and-play" priors for inverse problems is a well-known concept. The paper's specific choice of sampling algorithm (DAPS) is also directly adopted from prior work.The central claim to novelty is combining this prior with a flexible likelihood (Eq. 11). However, this likelihood function itself is not novel; it represents a data-fitting term ("loss function") whose components are explicitly drawn from prior classical methods. The paper cites previous work for the projection matrix $A$ (modeling alignment, blurring, and downsampling) and for the bias field model.Therefore, the paper's contribution is not a new framework, but rather the application of an existing framework (diffusion priors + posterior sampling) to a specific set of 3D medical imaging tasks. The work consists of "plugging" an established likelihood model (from classical methods like UniRes) into an established diffusion prior framework. This can be viewed as an incremental substitution—swapping a classical prior for a diffusion prior—rather than a fundamental conceptual advance.

2. The diffusion prior is trained only on high-quality, healthy brain scans. This creates a strong bias. This assumption may not be optimal for patients whose "healthy" tissue has been subtly altered or deformed by the pathology, potentially limiting the anatomical fidelity of the inpainted regions.


3. Several key ablation studies are missing: 1. Ablation of Likelihood Components: The main restoration likelihood (Eq. 11) jointly models super-resolution ($A$), image alignment, and bias field correction ($b$). The paper never ablates these components. A crucial missing study would be to compare the full model (Eq. 11) against a simpler version that only performs super-resolution (i.e., without the bias field correction term $b$ and the coordinate descent in Eq. 13). 2. Ablation of the Prior's Training Data: The diffusion prior was trained on a large, diverse cohort of T1w, T2w, and FLAIR images. The paper does not provide an ablation showing how performance changes if the prior is trained on a simpler dataset (e.g., only T1w images). This study would be necessary to justify their significant effort in assembling the multi-modal training cohort. 3. Ablation of Sampler Hyperparameters: The paper uses the DAPS sampling algorithm, which has numerous hyperparameters (e.g., Annealing steps, Diffusion steps, Langevin step number). The paper only provides an ablation for the likelihood precision ($\tau_{y}$). 

4. The quantitative comparison for computational cost and model size is missing. The paper provides no data on inference speed or model parameter count for the proposed method or the baselines.

### Questions
Same as the weakness part.

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
This paper proposes a general framework that uses a foundational diffusion prior for refining brain MRI data. To train a reliable diffusion prior, the authors curate a large-scale, artifact‑free brain MRI dataset aggregated from multiple sources. With the trained prior, the method addresses several medical imaging inverse problems and reports strong results across tasks.

### Strengths
- The problem is important and highly relevant to the medical imaging community.
- Both the curated dataset and the trained diffusion prior are valuable contributions to the field.
- The paper is generally well presented and easy to follow.
- Experiments across three task setups are thorough and yield meaningful insights.

### Weaknesses
- Limited technical novelty. From a task perspective, the medical imaging inverse problems studied here are common in the literature and have been extensively explored, even if prior work may not be as comprehensive as this study. From an algorithmic perspective, using diffusion priors for medical inverse problems has also been investigated,for example, Di‑Fusion (Wu et al., ICLR 2025) and DDM² (Xiang et al., ICLR 2023). While this paper may offer a broader or more unified treatment, the core concepts are not new.
- Insufficient dataset/model detail in the main paper. Given that the primary contributions are (i) dataset curation and (ii) training/validating a foundational diffusion prior, the main paper should include more details and statistics highlighting the dataset’s scope and the model’s practical significance. Currently, key information appears to be deferred to the supplementary materials. It would also be helpful to describe procedures for verifying dataset quality and to compare the curated data with publicly available alternatives.
- Need comparisons with more diffuion-related methods. Empirical comparisons with other medical inverse problem solvers, especially recent diffusion‑based state‑of‑the‑art methods (e.g., Di‑Fusion and DDM²), are not sufficient. Including these benchmarks (on a subset of tasks is fine) would strengthen the evidence.

### Questions
The citation style and in‑line references appear inconsistent in places. Could the authors verify and correct the formatting of citations and cross‑references throughout the paper?

### Soundness
3

### Presentation
3

### Contribution
2
