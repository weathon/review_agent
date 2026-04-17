# SADUNs: Sharpness-Aware Deep Unfolding Networks for Image Restoration

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
The ability to improve model performance while preserving structural integrity represents a fundamental challenge in deep unfolding networks (DUNs), particularly when handling increasingly complex black-box priors. This paper presents a novel Sharpness-Aware Deep Unfolding Networks (SADUNs), that addresses these limitations by integrating Sharpness-Aware Minimization (SAM) principles with the proximal operator theory. By analyzing the gradient landscape of linear inverse problems, we develop the separable sharpness-aware perturbation and subgradient calculation modules that maintain original network structures while enhancing optimization. Our theoretical analysis demonstrates that SADUNs achieve linear convergence for sparse coding tasks under common assumptions. Crucially, our framework reduces training costs through fine-tuning compatibility and preserves inference speed by eliminating redundant gradient computations via proximal operator properties. Comprehensive experiments validate SADUNs across multiple domains. Moreover, we have validated the improvement of our framework on plug and play single image super resolution tasks, which means that our framework has the potential to expand to more types of deep unfolding networks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates sharpness-aware minimization (SAM) in the context of deep unfolding networks (DUN) for inverse problems. They start from a minimization problem that consists of two terms: data fidelity and the SAM cost, which serves as a regularization to promote minima in flatter regions of the landscape, aiming to achieve better generalization. They propose a solver for the problem based on the majorization-minimization technique. They then design an unfolded architecture based on this solver. The idea is nice, but it is very incremental, and in my opinion, the authors fail to demonstrate the advantage of the SAM within the unfolding scheme, as the performance gains are very marginal. Additionally, they do not present any qualitative results, e.g., to facilitate visual comparisons, which is essential, in my opinion, for a paper on an image reconstruction topic. Overall, the paper provides an interesting perspective but does not convincingly demonstrate the practical advantages of incorporating SAM into DUNs.

### Strengths
The paper is well written, easy to follow, and the method seems to be correct theoretically, and a nice direction to explore.

### Weaknesses
The contribution is incremental, with no significant improvement in performance or efficiency over existing deep unfolding networks. The paper also lacks visual comparisons. Overall, the experimental evidence is not convincing that incorporating SAM provides a meaningful advantage within the unfolding framework.

### Questions
While a method does not need large numerical gains to be valuable, such claims should be precisely stated and justified. The first line of the conclusion mentions “significant performance improvements,” but the tables show only marginal gains. Could the authors clarify what specific improvements they consider significant? Are they referring to convergence behavior, stability, or something beyond PSNR/SSIM? In what scenarios would adding SAM be preferable? For example, again, in terms of robustness, generalization to unseen data, could you design an experiment to show this? If yes, it would be beneficial to include such experiments in the main body of text to show those advantages.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Sharpness-Aware Deep Unfolding Networks (SADUNs) — a new framework that enhances traditional Deep Unfolding Networks (DUNs) by incorporating Sharpness-Aware Minimization (SAM) into the proximal operator optimization process.

### Strengths
Standard DUNs (like LISTA, ISTA-Net, UFC-Net) suffer from optimization instability and limited adaptability when integrating complex black-box priors. The paper introduces sharpness awareness into DUNs to improve convergence and generalization. The framework redefines each DUN iteration as a sharpness-perturbed proximal step, replacing redundant gradient computations via proximal operator properties. The authors show that, under sparse coding assumptions, SADUNs achieve linear convergence

### Weaknesses
Most experiments are still within linear inverse problems (CS, SISR). There is no evaluation on nonlinear or dynamic inverse tasks (e.g., MRI, deblurring, or radar), where SAM’s stability might differ. 

Gains over strong baselines (e.g., UFC-Net) are minor (≈ 0.1 dB PSNR, < 0.002 SSIM), raising questions about practical significance despite theoretical novelty.

The authors claim no inference slowdown, but runtime and FLOPs comparisons are absent.
Quantitative verification (speed vs accuracy trade-off) would strengthen the paper’s engineering relevance.

### Questions
1. How sensitive is SADUN performance to the choice of subgradient approximation (Property 2)? Could replacing ∇̃g with backprop-based gradients further improve stability?

2. The authors are encouraged to empirically verify whether the “flatter minima” achieved via SAM actually translate to better generalization on unseen degradations.

3. The authors are encouraged to include visual comparison results to better illustrate the qualitative advantages of SADUNs over baseline methods.

4. The ablation analysis for SAM hyperparameters (ρ, β) is minimal; only a few static values are shown in tables. A more systematic study (varying ρ/β → performance → stability) would clarify the effect of sharpness control.

### Soundness
3

### Presentation
2

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
This paper proposes SADUNs, a framework that integrates Sharpness-Aware Minimization (SAM) into Deep Unfolding Networks (DUNs) for image restoration tasks. The key idea is to use the properties of proximal operators to efficiently compute a subgradient approximation, which is then used within a Unified SAM formulation to perturb the optimization landscape. This design aims to improve performance without increasing inference time and allows for fine-tuning from pre-trained DUNs. The authors provide a theoretical analysis showing linear convergence for a sparse coding variant (SALISTA-CP) and demonstrate empirical improvements on tasks including synthetic sparse coding, natural image compressive sensing, and single image super-resolution.

### Strengths
A significant advantage of the proposed method is its claim of not degrading inference speed, which is achieved by reusing computations from the unfolding step. The fine-tuning compatibility is also a practical benefit for adapting existing models.

The paper provides extensive experiments across multiple tasks (sparse coding, CS, SISR) and several base DUN architectures (LISTA variants, UFC-NET, ISTA-NET), demonstrating the general applicability of the framework.

### Weaknesses
The actual performance gains reported are often marginal. In many cases (e.g., Table 1, Table 3, Table 4), the improvements in PSNR/SSIM are fractions of a decibel or tiny absolute increments. While consistent, these gains are arguably too small to constitute a major advance. The paper lacks a compelling demonstration of a scenario where SADUNs provide a substantial qualitative or quantitative leap over strong baselines, which is critical for a top-tier conference publication.

The paper successfully demonstrates the "what" (improvements are possible) but falls short on the "why." There is a lack of analysis connecting the sharpness-aware formulation to the observed improvements in image restoration quality. How does the perturbation induced by SADUNs specifically improve the texture generation, artifact suppression, or detail recovery in the output images? A qualitative analysis comparing the loss landscapes or the behavior of attention maps (in transformer-based DUNs) with and without SADUNs would significantly strengthen the claims.

### Questions
No Questions

### Soundness
2

### Presentation
2

### Contribution
2
