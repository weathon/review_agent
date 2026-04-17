# Doloris: Dual Conditional Diffusion Implicit Bridges with Sparsity Masking Strategy for Unpaired Single-Cell Perturbation Estimation

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 4

## Abstract
Estimating single-cell responses across various perturbations facilitates the identification of key genes and enhances drug screening, significantly boosting experimental efficiency. However, single-cell sequencing is a destructive process, making it impossible to capture the same cell's phenotype before and after perturbation. Consequently, data collected under perturbed and unperturbed conditions are inherently unpaired, creating a critical yet unresolved problem in single-cell perturbation modeling. Moreover, the high dimensionality and sparsity of single-cell expression make direct modeling prone to focusing on zeros and neglecting meaningful patterns. To address these problems, we propose a new paradigm for single-cell perturbation modeling. Specifically, we leverage dual diffusion models to learn the control and perturbed distributions separately, and implicitly align them through a shared Gaussian latent space, without requiring explicit cell pairing. Furthermore, we introduce a sparsity masking strategy in which the mask model learns to predict zero-expressed genes, allowing the diffusion model to focus on capturing meaningful patterns among expressed genes and thereby preserving diversity in high-dimensional sparse data. We introduce \textbf{Doloris}, a generative framework that defines a new paradigm for modeling unpaired, high-dimensional, and sparse single-cell perturbation data. It leverages dual conditional diffusion models for separate learning of control and perturbed distributions, complemented by a sparsity masking strategy to enhance prediction of zero-valued genes. The results on publicly available datasets show that our model effectively captures the diversity of single-cell perturbations and achieves state-of-the-art performance. To facilitate reproducibility, we include the code in the supplementary materials. Code available at \url{https://github.com/ChangxiChi/Doloris}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents Doloris, a generative framework designed to predict single-cell perturbation responses from unpaired data. It uses a dual conditional diffusion implicit bridge (DDIB) to learn separate distributions for control and perturbed cells while connecting them via a shared latent space. To address sparsity, it introduces a separate masking model to predict which genes will be silent, allowing the main diffusion model to focus on modeling active gene signals. The authors demonstrate performance on several public datasets using distributional metrics.

### Strengths
The DDIB-based approach is a reasonable way to model the transition between control and perturbed states without enforcing explicit pairings.

### Weaknesses
- The motivation is unclear, and I don’t think the current motivation has a fair acknowledgement of existing methods. Multiple OT-based perturbation models [1-3], and STATE considers the same problem [4], as well as CellFlow [5]. Variational inference models directly consider sparsity and count distribution [6].
- I don’t think there are anything related to a negative binomial in your method. The method has nothing to do with ZINB at the moment; therefore, the discussion in Appendix A.6 appears problematic.
- It is unclear how the results compare to well-established benchmarks in [4, 7]. In particular, linear baselines or simple averaging have shown better performance compared to deep-learning methods in these cases. 
- The paper lacks an ablation study on the value of multi-step inference versus a one-step approach.
- More comprehensive metrics and evaluations should be employed, e.g., the Cell-eval metrics [4].
- It is unclear how the ground truth in Fig. 5 is derived, and how the authors distinguish biological and technical zeros.

[1] Bunne, Charlotte, et al. "Learning single-cell perturbation responses using neural optimal transport." Nature methods 20.11 (2023): 1759-1768.
[2] Klein, Dominik, et al. "Mapping cells through time and space with moscot." Nature 638.8052 (2025): 1065-1075.
[3] Dong, Mingze, et al. "Causal identification of single-cell experimental perturbation effects with CINEMA-OT." Nature methods 20.11 (2023): 1769-1779.
[4] Adduri, Abhinav K., et al. "Predicting cellular responses to perturbation across diverse contexts with State." bioRxiv(2025): 2025-06.
[5] Klein, Dominik, et al. "CellFlow enables generative single-cell phenotype modeling with flow matching." bioRxiv (2025): 2025-04.
[6] Weinberger, Ethan, Chris Lin, and Su-In Lee. "Isolating salient variations of interest in single-cell data with contrastiveVI." Nature Methods 20.9 (2023): 1336-1345.
[7] Ahlmann-Eltze, Constantin, Wolfgang Huber, and Simon Anders. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines." Nature Methods (2025): 1-5.

### Questions
See weaknesses.

### Soundness
2

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
5

### Summary
This author with this paper introduces DOLORIS, a new generative framework for predicting the gene expression of single cells after perturbation. The work is well-motivated by the unique challenges of single-cell data: it is unpaired, high-dimensional, and sparse. The proposed solution combines a dual diffusion bridge model with a dedicated sparsity masking mechanism.

### Strengths
***S1:*** The application of the DDIB framework is an excellent fit for the unpaired nature of the problem. It provides a principled way to bridge the control and perturbed distributions without imposing strong, potentially incorrect, assumptions about cell-to-cell correspondence.

***S2:*** The sparsity masking strategy is a very clever and pragmatic contribution. High-dimensional, sparse data is a major challenge for many generative models, which can easily waste capacity modeling the uninformative zero values. Decoupling the prediction of sparsity from the prediction of expression values is a clean solution that appears to work very well based on the ablation studies.

***S3:*** The work addresses a problem of significant practical importance in drug discovery and functional genomics. Developing more accurate predictive models for perturbation effects has the potential to dramatically accelerate biomedical research. The paper does a good job of motivating the problem and its real-world impact.

### Weaknesses
***W1*** While the DDIB framework is a good fit, the paper does not provide a strong argument for why it is superior to other generative frameworks for unpaired domain translation (e.g., those based on VAEs or GANs, like CycleGAN). A brief discussion contextualizing this choice would strengthen the paper.

***W2*** The model consists of two key parts: the mask model and the diffusion model. The paper evaluates them in ablation studies but does not analyze their interaction. For example, how robust is the diffusion model to potential errors made by the mask model? Does a slightly imperfect mask lead to catastrophic failures or graceful degradation in the generated expression?

***W3*** The model is conditioned on specific perturbation information. While it is tested on unseen perturbations, it's not entirely clear how it would generalize to entirely new classes of perturbations not seen during training. The representation of the perturbation itself seems crucial, and this aspect is not deeply explored.

### Questions
Beside weakness, I think would be beneficial if authors could briefly answer to some additional questions.

***Regarding the Sparsity Mask:*** Could you elaborate on the potential failure modes of the mask model and how they might impact the final predictions? For instance, if the mask model incorrectly predicts a biologically crucial gene as "silent" (inactive), does your framework have any mechanism to recover, or is that information irrecoverably lost?

***Regarding the DDIB Framework:*** Could you provide more intuition on why the shared latent space is sufficient to learn a biologically meaningful transformation? How does the model ensure that a control cell is mapped to a corresponding perturbed cell, rather than just an arbitrary cell from the perturbed distribution that matches the conditioning information?

***Regarding Computational Cost:*** Your model involves two diffusion models plus a mask model. Could you comment on the computational resources required and the inference time per sample compared to the baseline methods?

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
3

### Summary
The paper introduces Doloris, a novel generative framework for predicting the gene expression of single cells after a perturbation. The paper addresses two main challenges: (1) the data is unpaired, as one cannot measure the same cell before and after perturbation, and (2) gene expression data is extremely sparse. Doloris addresses the unpaired problem by training two separate diffusion models and aligning them through a shared latent space (without explicit pairing). To handle sparsity, the paper proposes a two-part model, namely a mask model that predicts a binary mask of which genes will be silent, and a diffusion model trained only on the active (non-zero) genes.

### Strengths
The dual-bridge framework is a novel and effective solution to the fundamental unpaired data problem.

The paper's emphasis on using distributional metrics (E-distance, EMD) over simple RMSE is well-argued, as these metrics are better suited to capturing the cellular heterogeneity inherent in single-cell data.

### Weaknesses
A significant weakness is the lack of details for the core diffusion models. The paper does not state what neural network architecture is used as the backbone.

The authors state they use a GAT to embed perturbation information. However, this choice seems arbitrary. The paper provides no ablation studies or comparisons to other GNN architectures.

The paper fails to connect its sparsity masking strategy to existing, well-established statistical frameworks. This two-part architecture with one model for the zero/non-zero binary outcome and a second model for the non-zero continuous values, is a direct implementation of a Hurdle Model. Drawing this parallel would have strengthened the paper by grounding its components in classical statistical theory.

### Questions
What is the specific architecture of the diffusion models?

What is the specific architecture of the mask model? Is it also conditioned by the GAT embedding?

I think the paper proposes a complex sampling strategy for the binary mask to ensure global consistency. How does this compare to simply thresholding the mask model's probabilities at 0.5? What was the performance gain from this more complex method?

### Soundness
2

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
The authors tackle the problem of estimating single-cell gene expression under genetic or molecular perturbations, where data is inherently unpaired due to the destructive nature of sequencing. They introduce Doloris, which uses two conditional diffusion models: a "source" model for unperturbed (control) cells and a "target" model for perturbed cells, implicitly aligned via a shared Gaussian latent space using Dual Diffusion Implicit Bridges (DDIB). To handle sparsity (abundant zeros in expression data), they add a mask model that predicts silent genes, allowing the diffusion process to focus on expressed patterns. During inference, continuous expressions are generated via DDIB, then masked and rescaled. The framework conditions on cell types and perturbations.

### Strengths
Single-cell perturbation modeling is difficult because scRNA-seq is destructive. We can’t see “before and after” for the same cell. Data is unpaired and high-dimensional + sparse. Existing models often (i) implicitly force pairing or (ii) ignore the unpaired nature, or (iii) regress to the mean and miss heterogeneity. Doloris learns a source conditional diffusion model for control cells and a target conditional diffusion model for perturbed cells, make them share a latent Gaussian space via ODE mapping, so a real control cell is diffused to latent, then denoised to the perturbed state under a specified perturbation. Additionally, they introduce a mask model that predicts which genes should be silent for a given perturbation and cell-type condition. Losses for diffusion are computed only on expressed genes.

### Weaknesses
1. The abstract boasts SOTA on public datasets, but misses comparison against recent unpaired optimal transport methods like scOT, diffusion-based OT or flow matching based OT. 
2. The shared latent space assumes control and perturbed distributions are bridgeable via Gaussian priors, but how is this validated? In Sec. 3.4, adding noise to unperturbed means to preserve heterogeneity is ad-hoc. Why Gaussian, and what's the sensitivity to σ_ct? If perturbations drastically shift distributions, this might fail.
3. While the method predicts realistic expression distributions, the paper doesn’t show many biological case studies. For example, does the model recover known pathway-level responses?

### Questions
1. In Fig. 2, why those specific genes (AARS, CARS, etc.)? Are they representative, or cherry-picked? Extend to full dataset stats?
2. How robust is the model to perturbation types? For example, does it handle dosage dependent molecular perturbations, or only binary knockouts?
3. Do you have any downstream validation? Beyond metrics, does it recover known biology, like perturbed pathways in KEGG?

### Soundness
2

### Presentation
2

### Contribution
2
