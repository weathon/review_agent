# Dataset Distillation as Pushforward Optimal Quantization

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Dataset distillation aims to find a small synthetic training set, such that training on the synthetic data achieves similar performance to training on a larger training dataset. Early methods solve this by interpreting the distillation problem as a bi-level optimization problem. On the other hand, disentangled methods bypass pixel-space optimization by matching data distributions and using generative techniques, leading to better computational complexity in terms of size of both training and distilled datasets. We demonstrate that by using latent spaces, the empirically successful disentangled methods can be reformulated as an optimal quantization problem, where a finite set of points is found to approximate the underlying probability measure. In particular, we link disentangled dataset distillation methods to the classical problem of optimal quantization, and are the first to demonstrate consistency of distilled datasets for diffusion-based generative priors. We propose Dataset Distillation by Optimal Quantization (DDOQ), based on clustering in the latent space of latent diffusion models. Compared to a similar clustering method D4M, we achieve better performance and inter-model generalization on the ImageNet-1K dataset using the same model and with trivial additional computation, achieving SOTA performance in higher image-per-class settings. Using the distilled noise initializations in a stronger diffusion transformer model, we obtain competitive or SOTA distillation performance on ImageNet-1K and its subsets, outperforming recent diffusion guidance methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper reframes disentangled dataset distillation as an optimal quantization problem in latent space, adds per-centroid weights (a small but material tweak), and proves a diffusion-pushforward consistency bound; empirically the paper proposes DDOQ , and it beats D4M and is competitive with DiT-guided methods at higher IPC on ImageNet-1K.

### Strengths
1.Casting disentangled DD as optimal quantization ties practice to classical theory; the pushforward bound (Theorem 1) is simple and actionable.

2.Adding weights to latent centroids is near-free and consistently helps (lower W₂; small but real accuracy gains over D4M).

3.Multi-IPC, multiple students (R-18/50/101; MobileNet-V2; EfficientNet-B0; Swin-T) and two generators (LDM, DiT), with cross-teacher/student table.

4.Strong results with DiT. Beats Minimax/IGD on full ImageNet-1K at low IPC; competitive on ImageNette/Woof.

5.Clean pipeline and well-written paper.

### Weaknesses
1. The empirical lift mainly comes from adding weights to D4M-style clustering; it’s an elegant repackaging more than a new core mechanism. A D4M+weights control would isolate weighting vs. barycenter vs. quantizer effects.

2. Theorem 1 requires δ>0 (no bound at exact image time), compact support, and Lipschitz test functionals; constants depend on (δ,T,R,d). Useful, but the gap to practical (non-Lipschitz) training signals remains.

3. Teacher dependence and fairness. Main UNet/LDM tables fix ResNet-18 soft labels (cap at 69.8), but teacher swaps materially affect generalization (Table 3) and DiT tables use different baselines. A teacher-swap on ImageNet-1K main table would de-bias conclusions.

4. Results focus on accuracy; there’s little about wall-clock/GPU hours for distillation + training across methods/backbones. (RDED omitted at high IPC due to compute, another signal that cost matters.)

5. DDOQ underperforms D4M for Swin-T students at IPC50; this suggests hyperparameter or representation-mismatch issues that deserve diagnosis.

### Questions
1. What exactly drives the gains, weights or quantizer? Please add a D4M + weights variant and a DDOQ w/o weights ablation to separate effects.

2. What is the effective latent dimension d used by LDM/DiT in your pipeline, and how do W₂ and accuracy scale with K (log–log plots) to test the K⁻¹/ᵈ prediction?

3. Please replicate Table 2 with a ViT-B and Swin-T soft-label teacher to test robustness of conclusions beyond ResNet-18’s 69.8 cap.

4. Report end-to-end cost (distill + train) for D4M vs DDOQ vs guidance-based baselines, at IPC10/50/200, on fixed hardware.

5. Any insights (e.g., label entropy, weight dispersion, LR/BS sensitivity) on why Swin-T lags with DDOQ in Table 3?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Dataset Distillation by Optimal Quantization (DDOQ), which reframes disentangled dataset distillation as an optimal quantization problem. The method leverages latent diffusion models (LDMs) to represent data distributions in a low-dimensional latent space, performs weighted clustering (CLVQ) to approximate the latent data measure, and uses the generative decoder to reconstruct synthetic training samples.

### Strengths
1. The paper introduces a new conceptual connection between dataset distillation and optimal quantization under the Wasserstein distance framework.
2. The authors propose to use the Wasserstein distance between the distilled latent representations and the original latent data distribution as a quantitative indicator of how well the synthetic dataset approximates the real data distribution.
3. The proposed method, DDOQ, improves performance in higher images-per-class (IPC) settings.

### Weaknesses
1. The introduction looks like a related work and fails to clearly introduce the research gap and specific contributions of the paper. As a result, it is difficult for readers to understand what is novel about this work.
2. The proposed DDOQ method is a modification of D4M, replacing uniform clustering with a weighted k-means step. The overall pipeline (latent clustering, diffusion decoding, and weighted training) remains conceptually similar to previous methods. The framing of the approach as “optimal quantization” seems more like a theoretical reinterpretation than a fundamentally new algorithmic contribution.
3. In Section 3.1, it is not explicitly shown how optimal quantization principles are concretely applied to dataset distillation. The practical novelty seems the introduction of weights, but these weights are simply normalized cluster weights (as shown in Algorithm 1 in line 278) and not directly derived from optimal quantization theory.
4. The paper is mathematically dense and lacks intuitive explanation or visual figures. A framework diagram or flowchart illustrating the DDOQ pipeline would greatly help readers grasp how the theoretical formulation connects to the practical implementation.
5. In Table 2, the caption claims that “the maximum performance for all methods should be 69.8,” but this number does not appear in the table, leading to confusion. Additionally, the DDOQ performance at IPC=10 is substantially worse than RDED, indicating that the proposed method may not generalize well across different IPC settings.
6. The paper lacks a detailed ablation study analyzing the effects of key hyperparameters, such as the latent dimension, the number of quantization points K, and the weighting scheme.

### Questions
Could the authors provide a table comparing the computational cost (e.g., memory usage, and GPU hours) of DDOQ against other baseline methods such as D4M and RDED?

### Soundness
2

### Presentation
1

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
The paper aims to improve representative selection for classes in dataset distillation. It reframes the selection process as an optimal quantization problem and supports the approach with theoretical proofs and experimental validation. The objective is to provide a more principled and theoretically grounded way of selecting better representatives for dataset distillation for sample efficient training. This combination of theory and empirical study gives the work potential value for the dataset distillation research.

### Strengths
1. Frames the dataset distillation problem as an optimal quantization problem and supports the suggested improvements with appropriate theoretical proofs. The theoretical formalization connects optimal quantization error with downstream expectation differences in the image domain, which helps bridge diffusion-based priors and representative selection in a mathematically grounded way.

2. Theorem 1 gives an explicit upper bound for the Wasserstein distance in latent space, which they show is related to the expectation difference in image-space functions after reverse diffusion. This link between latent space distance and image-space fidelity is useful and not previously formalized for these disentangled distillation methods.

3. The complexity of the proposed method is low, but the performance gains are consistent and meaningful compared to baselines.

4. Experiments covering other approaches for the same latent diffusion backbone and comparison of different prior selection methods for DiT are good to show that the gains are from the proposed approach and translate to other backbones.

### Weaknesses
1. The theoretical foundation simply builds upon existing proofs from quantization theory, so the novelty in theoretical contributions is limited. The actual improvements suggested are limited and incremental, as operationally, it simply adds weights to cluster centers.

2. A weighted loss is used while training the student models, which makes it unclear if the improvements are from better diffusion prior selection via clustering or better optimization of student model via weighted loss. An ablation study clarifying this will improve the contribution.

### Questions
1. The bad performance of the Swin-T transformer as a student is attributed to hyperparameters. Providing results with optimal hyperparameters will be helpful to identify if transformer architecture poses some limitation to this work or disentangled distillation in general.

2. Proposition 1 and 2 use the notation δ_{x_i}. Please clarify whether this denotes a Dirac delta function (i.e., a unit point mass at x_i) or some other function indicating the Voronoi cell’s probability mass.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper provides a theoretical interpretation of dataset distillation (DD) by framing it as an optimal quantization problem in latent space. The authors show that modern disentangled DD methods can be understood as approximating the true data distribution via finite quantization of its latent representation.
Building on these observations, they propose Dataset Distillation by Optimal Quantization (DDOQ), which performs weighted clustering (via CLVQ) in the latent space of a pretrained diffusion model and uses the resulting centroids and weights to reconstruct distilled images. Empirically, DDOQ achieves improved performance over prior works such as D4M and RDED on ImageNet-1K, ImageNette, and ImageWoof, with notable advantages in medium IPC regimes and better inter-model generalization.

### Strengths
- **Theoretical contribution**: The key theoretical contribution is the formal link between quantization theory, Wasserstein distance, and dataset distillation consistency. Theorem 1 demonstrates that score-based diffusion preserves the distributional closeness between the raw data and its quantized latent approximation, resulting in consistent gradient expectations during training on the distilled dataset. Corollary 1 further establishes asymptotic convergence rates $\mathcal{O}(K^{-1/d})$ for the approximated data distribution in image space, offering the consistency guarantees for diffusion-based DD methods.
- **Clean problem framing**: Reformulating dataset distillation as a pushforward optimal quantization problem is an insightful contribution that bridges classical quantization theory and modern generative modeling.
- **Comprehensive experiments**: Evaluation covers both UNet and DiT-based diffusion models, multiple IPC regimes, and cross-architecture generalization. The results are consistent and competitive with SOTA baselines.

### Weaknesses
- **Heuristic implementation > theoretical analysis**: The core theoretical argument is that optimal quantization (non-uniform weights) is superior to finding the Wasserstein barycenter (uniform weights). However, the final implementation does not utilize these theoretically derived weights. Instead, it employs an ad-hoc heuristic (Eq. 34) output for "variance reduction." This choice is not justified by the theory and is not ablated. This undermines the central claim that the theory guides the method's design. Rather, the theory appears to be more of a post-hoc justification for the general idea of weighting, while the actual implementation relies on an unexplained heuristic.
- **Lack of practical guidance**: 
    - The paper introduces weighting as a core contribution but provides little analysis of its behavior, interpretability, or sensitivity during training.
    - The bound in Theorem 1 contains a constant C with dependencies that are not analyzed in a practical context (e.g., how to choose diffusion time T to optimize the bound). The convergence rate $\mathcal{O}(K^{-1/d})$ is standard for quantization but does not lead to new insights on how to choose an optimal latent dimension d or number of prototypes K for a given budget.

### Questions
- **More ablations**: Could the authors provide an ablation study on ImageNet-1K (e.g., at IPC 10) that directly compares: (a) no weights (D4M baseline), (b) the theoretically motivated weights, and (c) your proposed heuristic weights? Without these ablations, it is difficult to assess whether the theory is useful in practice or if an arbitrary heuristic is the true source of improvement.
- **Source of performance gains**: Can you quantify the relative error reduction from (a) moving from D4M to DDOQ on a fixed U-Net backbone, versus (b) moving from a U-Net to a DiT backbone using a fixed D4M algorithm? This would help disentangle the contribution of your method from the contribution of the stronger (or different) generative model.
- **OQ utility**: Beyond justifying the use of weights, does the OQ perspective offer any other suggestions for concrete algorithmic improvements? For instance, the $\mathcal{O}(K^{-1/d})$ error rate suggests a trade-off between the number of prototypes K and latent dimension d. Does your theory provide any guidance on how to choose d for a given distillation budget K? If not, what is the practical utility of the theory beyond re-describing weighted k-means?
- **Necessity of weighting in high-dimensional latent spaces**: The paper suggests that the weighting scheme mitigates the mismatch between the limited number of quantization points (IPC) and the high dimensionality of the latent space. Could the authors elaborate on this relationship quantitatively? For instance, is there an empirical or theoretical threshold linking the number of latent samples K and latent dimension d beyond which weighting becomes less critical (e.g., K>>d)?
- **Higher IPC regimes**: If the number of distilled images increases substantially (e.g., IPC@1000), do the authors expect the weighting adjustment to remain influential, or would the raw cluster frequencies already approximate the latent density sufficiently? Clarifying this could help understand whether the proposed weighting primarily benefits low-data regimes or remains beneficial asymptotically.

### Soundness
3

### Presentation
3

### Contribution
3
