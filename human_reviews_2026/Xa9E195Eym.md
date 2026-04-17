# Collaborative-Reverse Diffusion Models

- Decision: Reject
- Scores: 0, 2, 6, 2

## Abstract
Given sufficient training samples, diffusion models have outperformed in various data generation tasks. However, natural science researches are often challenged by class imbalance and data sparseness. In order to apply the diffusion models to general natural science scenarios, we propose the Collaborative-Reverse Diffusion Models (CoReDM), a novel diffusion-based framework that incorporates a neural collaborative mechanism to improve the generating ability of sparse-class samples. We introduces a new paradigm for integrating the collective similarity into the reverse diffusion processes. The model employs a dual-network architecture consisting of a denoising network and a neural collaborative regularization term, which learns to assign influence weights between samples according to their collective relationships. This design facilitates adaptive knowledge transfer from dense to sparse classes during the denoising. Experimental results on several imbalanced datasets demonstrate that our CoReDM substantially enhances both the quality and diversity of generated samples for sparse classes, outperforming standard diffusion models and a few data augmentation baselines.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes Collaborative-Reverse Diffusion Models (CoReDM), which aim to improve the performance of diffusion models under class-imbalanced and data-sparse conditions. The authors introduce a neural collaborative regularization term that enables information sharing between semantically similar samples during the reverse diffusion process. The paper claims that this collaborative mechanism facilitates adaptive knowledge transfer from data-rich to data-scarce categories, leading to better generation quality and diversity. Experiments are conducted on a material property dataset and a text-to-image setting.

### Strengths
The paper addresses the class-imbalance problem in diffusion models, which is an important and timely topic, especially for scientific domains with limited data availability.

### Weaknesses
**1. Serious issues in correctness and clarity**  
   - Eq. (2): b_j and D_{ij} are scalars, whereas the other terms are vectors, making the equation mathematically inconsistent.
   - Eq. (3): The definition of lambda(t) lacks an expectation over t, and its formulation contradicts the text description. L242-245 state that the collaborative constraint should become stronger as the image becomes cleaner, but the given formula implies the opposite.
   - Eq. (5): w_{ij} is undefined, and the loss does not depend on the parameter phi.
   - Alg. 1: x_i and x_j are never defined.  
   - Similarity matrix D_{ij}: The derivation is not explained. It is also unclear how similar samples x_j are collected during training or inference.

**2. Insufficient experimental validation**
  - Important baselines such as CBDM [1] or other class-imbalanced diffusion models are missing.
   - The experimental datasets are limited. Common benchmarks for class imbalance (e.g., **CIFAR10-LT**) are not tested, making it hard to evaluate general applicability.
   - No visualizations on the T2I results. This prevents qualitative assessment of the claimed improvement. 
   - There is no description of hyperparameter settings or training details (e.g., gamma, lambda_max, optimizer setup), which makes the experiments non-reproducible.
.

----
**Reference**

[1] Class-Balancing Diffusion Models, CVPR 2023

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Collaborative Reverse Diffusion Models (CoReDM), a modification of diffusion models designed to improve generation quality under data imbalance and data sparsity. The approach introduces a collaborative mechanism across samples during the reverse diffusion process. Each sample is influenced by other samples considered similar according to a fixed similarity matrix derived from a pretrained encoder. A secondary network, denoted as S_phi, is said to learn pairwise influence scores b_j = S_phi(x_i, x_j) that adapt the strength of collaboration. Additionally, training includes an inter-sample consistency term intended to enforce that similar samples yield similar noise predictions from the denoising network. The paper claims that these mechanisms enhance generation quality and diversity, especially for scientific datasets with few examples per class.

### Strengths
The motivation is reasonable and addresses a valid gap in diffusion research: adapting diffusion models for domains where data are scarce or imbalanced. The idea of inter-sample regularization through learned collaboration is conceptually appealing and may, in principle, allow better generalization in low-data regimes. The ablation design that compares NoCollab, StaticCollab, and LinearCollab configurations is also well motivated.

### Weaknesses
The paper suffers from major issues of theoretical rigor, methodological clarity, and experimental transparency. Several critical points remain ambiguous or internally inconsistent, making it difficult to assess correctness or reproducibility.
1. Theoretical and methodological issues  
- The most fundamental problem concerns the training of the collaboration network S_phi. In both the text and Algorithm 1, b_j = S_phi(x_i, x_j) is computed, yet this quantity never appears in a differentiable path connected to phi. The consistency loss L_CF only contains D_ij and epsilon_theta, hence grad_phi L_CF = 0. Consequently, S_phi has no learning signal. The later introduction of w_ij, which is never defined, does not resolve this issue. It is unclear how S_phi is actually optimized or whether it is frozen during training. The authors should clarify precisely how gradients reach phi.  
- Related to this, the meaning of inter-sample consistency is vague. The loss penalizes differences in predicted noise between neighboring samples, but there is no theoretical justification for why this improves sample quality or preserves multimodality. Does the consistency term simply enforce smoothness, risking oversmoothing and mode collapse? A more formal justification or empirical study of this behavior is necessary.
- The collaborative sampling update also raises validity concerns. The proposed update replaces the reverse-process mean with a linear combination that omits the stochastic noise term z, thereby breaking the standard DDPM parameterization. The paper provides no derivation linking this update to any approximate reverse transition or score-based formulation. What is the derivation of this collaborative update? Does it approximate any known reverse-process distribution or guidance mechanism? Without such justification, the sampling rule appears ad hoc.
- Time dependence introduces further confusion. The text claims that b_j is independent of t, but Algorithm 1 optimizes phi across all timesteps. If b_j does not vary with t, why does the algorithm loop over t when updating phi? If it does vary, what is the explicit form of this dependence? This ambiguity prevents clear interpretation of the learning process.
- Finally, the computation of b_j itself is unclear. The definition b_j = S_phi(x_i, x_j) implies dependence on both x_i and x_j, yet Algorithm 1 computes b_j once and reuses it for all i. Is b_j global (shared across all x_i) or recomputed per sample pair (b_ij = S_phi(x_i, x_j))? The latter interpretation is more consistent but implies O(n^2) computation per step. Which implementation is correct, and how does this affect scalability?
2. Experimental and empirical issues  
The specification of the similarity matrix D_ij is incomplete. The encoder architecture, its training data, normalization scheme, and whether it is computed using only the training split are all unspecified. If D_ij is computed using validation or test data, this would introduce information leakage. Please clarify how D_ij is built: what encoder, what dataset, and whether it is restricted to k-nearest neighbors or normalized to prevent cluster bias.
The dataset itself poses additional challenges. The inputs are compositional (elemental fractions), meaning they live on a simplex manifold. The paper does not mention closure correction or log-ratio transformation, which are essential to respect compositional constraints. How are these constraints enforced during diffusion? Without proper handling, generated samples may violate physical feasibility. Given this structure, it would also be appropriate to compare against diffusion models defined on the simplex or using log-ratio transformations.
Several experimental details are missing or inconsistent. Dataset statistics, splits, and random seeds are omitted. The metric "70% R² + 30% normalized MSE" is unconventional and not reproducible. The reporting of mean ± standard deviation is inconsistent: several baselines lack standard deviations, and the parentheses in Table 1 appear to denote improvements rather than variability. Please clarify whether these parentheses represent standard deviations or relative gains, and include proper uncertainty reporting for all models.  
Optimization settings are also unclear. Different learning rates are stated across sections, and the training schedule for the collaborative term is not specified. Please reconcile these discrepancies. The image experiments lack information about dataset size, baseline configuration (e.g., Stable Diffusion parameters), and computation of FID, IS, and CLIP scores. These omissions prevent independent verification.  
The paper further fails to compare against existing imbalance-aware or collaborative diffusion methods that it cites in the related work section. Given that the compositional dataset lies on a simplex, the study should also include baselines such as Mirror Diffusion Models or Riemannian Diffusion frameworks designed for constrained domains.  
Finally, computational cost is not discussed. The collaborative term appears to require O(n^2) operations if computed densely, yet no runtime or scaling results are provided. How does this method scale with dataset size, and what are the associated training and sampling costs?
3. Writing and presentation  
The manuscript exhibits numerous notation inconsistencies: the use of L_diff and L_diffusion interchangeably, the omission of the stochastic noise term z in the sampling equations, and inconsistent symbols for D_ij, w_ij, and lambda(t). Algorithmic notation is ambiguous about whether sums are computed over the entire dataset or within mini-batches. The writing also suffers from typographical errors (“sprase”), missing spaces, and underexplained figures. These presentation issues further obscure the contribution.

### Questions
The core idea of inter-sample collaboration in diffusion models is potentially interesting but currently under-specified and mathematically unsound. The training procedure for S_phi lacks a valid optimization pathway, the sampling update deviates from diffusion fundamentals without justification, and the experimental setup is incomplete and potentially irreproducible. For this work to meet publication standards, the authors must provide a principled derivation, consistent notation, full experimental transparency, and credible baselines.

Suggestions for improvement  
- Define a differentiable objective for S_phi. For example, set w_ij = D_ij S_phi(x_i, x_j) and include this weight inside L_CF, ensuring that phi receives a learning signal.  
- Provide a derivation showing how the collaborative update relates to the reverse diffusion process or score guidance.  
- Restore the stochastic term z in the sampling equation and demonstrate empirically that samples remain stable and diverse.  
- Fully specify D_ij: describe the encoder architecture, pretraining data, normalization, and whether k-nearest neighbor sparsity is applied.  
- Handle compositional inputs correctly using log-ratio or simplex-preserving transformations, and include diffusion baselines that operate natively on the simplex manifold.  
- Strengthen experiments with complete dataset statistics, fixed seeds, transparent metric definitions, and full uncertainty reporting (mean +- std).  
- Compare against established imbalance-aware and manifold-constrained diffusion models.  
- Analyze computational scaling of the collaborative term, reporting runtime and memory overhead.  
- Clean up notation, unify symbols, and fix typographical errors to improve clarity.
If these issues are thoroughly addressed and the collaborative mechanism is theoretically grounded and empirically validated, the work could become a meaningful contribution to generative modeling under data scarcity.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes Collaborative Reverse Diffusion Models (CoReDM), introducing the "collaborative regularization" mechanism in the reverse diffusion process: The pre-computed sample similarity matrix and learnable influence weights are utilized to guide the denoised trajectories, thereby enhancing the generation quality and diversity in scenarios of class imbalance/data sparsity.

### Strengths
1. For the sparse small sample problem in natural science scenarios, it is proposed to perform collective similarity regularization in the reverse diffusion stage to achieve knowledge transfer, and the problem description and paradigm shift are clear.

2. Alternately optimize the denoising network and the collaborative regularization network to avoid instability in joint training.

3. A large number of experimental results have proved that the improvement effect of CoReDM is significant.

### Weaknesses
1. What are the quantification of time and video memory overhead?

2. What are the effects of encoder selection and its out-of-domain generalization？

### Questions
1. Is the linear collaborative weight over time optimal?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Collaborative-Reverse Diffusion Models (CoReDM), a novel framework designed to improve the performance of diffusion models on imbalanced datasets with sparse data. The key innovation is the integration of a neural collaborative filtering mechanism into the reverse diffusion process. Specifically, CoReDM adopts two networks, a standard denoising network and a neural collaborative regularization network that learns to assign influence weights between samples based on their similarity. The collaborative reverse process modifies the standard update rule by adding a weighted sum of influences from similar samples, where weights are determined by both a precomputed static similarity matrix and dynamically learned influence scores. Experiments on material property datasets and image generation tasks demonstrate improvements in generation quality.

### Strengths
Addressing class imbalance in diffusion models is an important practical problem, especially for scientific applications where balanced datasets are rarely available.

### Weaknesses
1. The paper lacks a theoretical intuition of why the proposed formulation (2) might help for inference. The proposed formulation is very heuristic.  
2. The paper lacks comparisons with existing methods that deal with class imbalance and data sparseness.
3. The paper lacks a detailed explanation of the class imbalance and data sparseness that appeared in material property datasets, e.g., what is the actual size? What is the degree of imbalance (exact class distribution)? How many classes exist? Moreover, the formulation of the criteria in Table 1, Figure 3 are missing. Although section 4.3.3 mentions some experimental results on LAION-400M, the experimental setup is also vague.

Typos: the caption of section 4.3.1.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
