# Noise is All You Need: Solving Linear Inverse Problems by Noise Combination Sampling with Diffusion Models

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Pretrained diffusion models have demonstrated strong capabilities in zero-shot inverse problem solving by incorporating observation information into the generation process of the diffusion models. However, this inevitably presents a dilemma: excessive integration can disrupt the generative process, while insufficient integration fails to emphasize the constraints imposed by the inverse problem. To address this, we propose $\textit{Noise Combination Sampling}$, a novel method that synthesizes an optimal noise vector from a noise subspace to approximate the measurement score function, replacing the noise term in the standard Denoising Diffusion Probabilistic Models process. This enables conditional information to be naturally embedded into the generation process without reliance on step-wise hyperparameter tuning. Our method can be applied to a wide range of inverse problem solvers, including image compression, and, in most scenarios, especially when the number of generation steps $T$ is small, achieves superior performance with negligible computational overhead, significantly improving robustness and stability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Noise Combination Sampling (NCS) framework to address the key challenge of balancing observation constraint integration and diffusion process consistency in diffusion-based inverse problem solving. NCS constructs an optimal noise vector (via linear combination of base Gaussian noise from a codebook) to implicitly embed observation information, ensuring the noise remains standard normal and the sampling trajectory stays on the data manifold. It provides a closed-form solution for optimal weights (via Cauchy–Schwarz inequality), unifies existing methods (DPS, MPGD, DDCM) as special cases, and achieves both better performance (higher PSNR, lower LPIPS) and efficiency across tasks like inpainting, super-resolution, and deblurring.

### Strengths
1. The closed-form solution for optimal weights (Theorem 2) is strictly derived using inner products and the Cauchy–Schwarz inequality, eliminating the need for heuristic iterative optimization. This ensures reproducibility—researchers can directly compute \gamma^* without tuning empirical parameters.
2. Unlike baselines (e.g., DPS) that suffer severe quality degradation when diffusion steps T are reduced (to cut costs), NCS maintains high robustness. Experiments show it achieves high-quality results even with small T (e.g., 100 steps vs. 1000 steps), as the optimal noise combination preserves manifold consistency without relying on excessive iterations.
3. This method replaces DDCM’s exponential-complexity noise selection, where C is quantization bins) with NCS’s linear-complexity combination. For example, combining 3 noise vectors achieves inner-product magnitudes comparable to DDCM’s search over 1024 vectors, slashing storage (smaller codebooks) and computation (fewer inner-product calculations) costs.

### Weaknesses
1. This method lacks clear guidelines for selecting codebook size K and base noise distribution. A small K restricts the noise combination space (failing to match rare observation directions), while a large K increases memory usage and inner-product computation time. No adaptive mechanism (e.g., dynamic K based on task complexity) is proposed.
2. The authors state that the NCS method approximates the measurement score through the linear combination of Gaussian noise vectors. Can similar effects be achieved via approximation methods such as Hermite polynomials? After all, Hermite polynomials are the optimal basis functions for Gaussian-related processes.
3. The authors state that the NCS method can be integrated into sampling strategies such as DDPM. Is NCS incompatible with DDIM out of inverse problems, a more commonly used deterministic sampling method? Can it be extended to Rectified flow in video generation models?
4. In the description of Figure 1, the authors state that NCS embeds the measurement score into the optimal noise within an ellipsoidal subspace, which is defined by the span of the noise codebook. I confuse that is this definition related to the one in Equation (10)? Why is the noise obtained from a simple weighted sum explained using the concept of an ellipsoidal subspace?
5. When the authors explain the advantages of NCS, they highlight a key feature: it adheres to conditional constraints while ensuring generation stability. However, most of the experimental results focus on image quality metrics such as PSNR. Although these metrics are suitable for evaluating inverse problems, I am also curious about NCS’s performance in terms of conditional compliance—for example, metrics like image-text alignment.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

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
The paper proposes Noise Combination Sampling (NCS), a method for solving diffusion-based linear inverse problems by replacing the stochastic noise term in the diffusion update rule with a constructed linear combination of Gaussian noise vectors. The combination weights are chosen to align with the estimated measurement score direction, derived via a closed-form expression using the Cauchy–Schwarz inequality. Experiments on standard inverse problems such as inpainting, deblurring, super-resolution, and compression are presented, showing moderate performance gains and faster convergence in some settings.

### Strengths
1. The paper is generally well-written and clearly structured, making the proposed method easy to follow.
2. The idea of modifying the diffusion noise term instead of using explicit gradient guidance is interesting and practically implementable.
3. The authors conduct some ablations such as varying codebook size K and report quantitative metrics across multiple datasets, showing some empirical rigor.

### Weaknesses
1. The review template seems to be missing some elements, such as line numbers and the header stating “Under review as a conference paper at ICLR 2026.” The former, in particular, would have made it easier to reference specific parts of the text when providing comments.
2. NCS primarily reparameterizes the noise term in existing diffusion posterior sampling methods, offering little conceptual or theoretical advancement. The “closed-form” derivation is a straightforward application of the Cauchy–Schwarz inequality, and the claimed improvements over prior approaches are minimal. Overall, the contribution feels incremental rather than a substantive methodological innovation.
3. The paper claims that the constructed noise combination remains Gaussian if the combination weights are independent of the noise codebook. However, in the proposed method, the weights $𝛾_i$ are computed from the inner products between the codebook vectors and the measurement score, creating explicit statistical dependence between $𝛾_i$ and $ε_i$. This invalidates the independence assumption underlying the “Gaussianity lemma” and breaks the formal justification that the composed noise $ε^*_t$ follows $\mathcal{N\mathrm{(\mathbf{0},\mathbf{I})}}$.
4. Methods like DPS and MPGD are reasonable baselines but outdated, as many faster samplers now exist (e.g., ΠGDM, DDNM, DiffPIR, MGPS) that achieve strong performance within 50-100 NFEs. It remains unclear whether NCS-DPS and NCS-MPGD can outperform these state-of-the-art methods, and Table 2 does not clearly demonstrate whether NCS provides any real improvement over DAPS.
5. In Section 3, the authors derive all formulas under the assumption of linear inverse problems, yet later report results for the nonlinear phase retrieval task without providing its explicit formulation or explaining how NCS applies in this setting.
6. The authors claim that NCS unifies most existing gradient-based approaches, but this appears largely superficial, consisting mainly of algebraic reformulations where the guidance term is replaced by a projected noise. There is no shared probabilistic interpretation or derivation showing these methods as genuine special cases of a single framework.
7. The authors report that they used $σ_y$ = 0.05, but the qualitative figures suggest $σ_y$ ~= 0.0 (e.g., Figure 2, box inpainting).

### Questions
1. *Regarding weakness 3:* Can the authors clarify how the Gaussianity of $ε^*_t$ is preserved in practice given that $𝛾$ depends on {$ε_i$}? If the independence assumption is violated, what is the actual distribution of the constructed noise, and how does this affect the validity of the diffusion process?
2. *Regarding weakness 4:* Can the authors explain why stronger and more recent zero-shot inverse solvers such as were omitted from comparison? Additionally, can they clarify whether NCS actually improves DAPS, given the quantitative results reported in Tables 2,3 and 4?
3. *Regarding weakness 5:* How does NCS behave when the forward operator is ill-conditioned or nonlinear? Does the Gaussianity assumption still hold in such cases, and how exactly was phase retrieval implemented or adapted within the proposed framework?
4. Can the authors report the additional runtime and GPU memory consumption introduced by NCS for typical values of K?

### Soundness
2

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
4

### Summary
This paper proposes the Noise Combination Sampling (NCS) framework to address the stability dilemma in solving linear inverse problems with diffusion models. Instead of directly modifying the sampling trajectory, which risks disrupting data manifolds and generative consistency, NCS embeds conditional information into the noise term of Denoising Diffusion Probabilistic Models (DDPM). It synthesizes an optimal noise vector from a predefined noise codebook to approximate the measurement score, with closed-form optimal weights derived via the Cauchy–Schwarz inequality to ensure that the synthesized noise remains standard normal. NCS addresses the core dilemma of diffusion-based inverse problems that excessive integration disrupts generation and insufficient integration ignores constraints by synthesizing optimal noise vectors to embed conditional information, rather than directly modifying the sampling trajectory.

### Strengths
1. NCS embeds conditional constraints into the DDPM noise term instead of directly modifying the sampling trajectory to avoid pushing generated samples off the data manifold and breaking diffusion generation consistency. This design aligns with the intrinsic stochasticity of diffusion models and has solid theoretical motivation.

2. NCS does not redesign the approximation of conditional score and is expected to expand the performance of many existing methods that apply data consistency through conditional gradient score approximation.

### Weaknesses
1. The paper acknowledges unstable results on nonlinear tasks (e.g., phase retrieval) but provides no theoretical analysis on the reason why NCS fails here. For example, whether the noise subspace can approximate non-linear conditional gradient score or if the closed-form weight solution breaks down under non-linear constraints.

2. The choice of noise codebook size K (e.g., 512 for 4× super-resolution and 64 for 8× super-resolution) is purely empirical. In the paper, although K is noted to work "across a broad range", there does not exist a quantifiable relationship between K, data dimensionality, or task complexity, nor does it define the boundary where increasing K reduces noise independence (mentioned in Section 3.2) and yields degraded performance. Ablation studies are also missing for key parameters related to noise combination such as the number of combined noise vectors K on performance. This makes it hard to validate the robustness of NCS to parameter changes.

### Questions
1. Can NCS be adapted to nonlinear inverse problems? If not, what theoretical modifications would be required?

2. Is there any theoretical guideline for K? For example, how should K scale with data dimensionality (d) or task complexity to balance noise expressiveness and independence?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Noise Combination Sampling (NCS), a new approach to solving linear inverse problems using pretrained diffusion models without retraining. The main idea is to replace the standard Gaussian noise term in the denoising process with a synthesized noise vector, formed as an optimal linear combination of Gaussian samples drawn from a codebook. The combination weights are derived via a simple closed-form expression based on the Cauchy–Schwarz inequality, which aligns the noise with the conditional measurement score.

The authors claim that this procedure naturally embeds conditional information into the generation process while avoiding the instability and heavy hyperparameter tuning required by prior guidance-based inverse solvers such as Diffusion Posterior Sampling (DPS) and Manifold-Preserving Gradient Descent (MPGD). They further argue that existing diffusion-based inverse problem solvers can be interpreted as special cases of NCS, including the recently proposed Denoising Diffusion Codebook Models (DDCM). Empirical results on FFHQ and ImageNet show moderate improvements in PSNR and LPIPS, particularly for low sampling step counts (T = 20–100).

### Strengths
1. The paper is clearly written, with logical organization and careful presentation of derivations, figures, and tables. The notation is consistent, and the main idea is easy to follow even for readers without deep expertise in diffusion models.
2. The derivation of the optimal noise combination through the Cauchy–Schwarz inequality is elegant. It provides a compact, closed-form solution that is computationally lightweight and easy to implement.
3. The authors successfully demonstrate that several well-known inverse problem solvers (DPS, MPGD, DDCM) can be interpreted as instances of the same general principle. This unifying viewpoint could be valuable for researchers seeking a more cohesive understanding of guidance mechanisms in diffusion models. Also, across multiple datasets and problem types, the NCS variants perform as well as or slightly better than their corresponding baselines, especially when the number of diffusion steps is small. This consistency suggests that the approach is robust and stable in practice. More importantly, the method requires no extra training and introduces negligible additional computational cost. The linear complexity with respect to the codebook size makes it practical and accessible for a wide range of applications. By relating the approach to DDCM, the paper opens the possibility of extending diffusion-based generative compression with simpler noise quantization schemes.

### Weaknesses
While the paper is neat and clearly executed, the conceptual advance feels incremental and its underlying mechanism insufficiently explored. The proposed idea of aligning a noise combination to a measurement gradient is mathematically simple and closely related to existing guidance schemes. The derivation relies on a direct application of Cauchy–Schwarz, and it is unclear why this reformulation should lead to qualitatively better sampling.

Moreover, the claimed advantages (manifold preservation, stability, robustness to step size) remain intuitive hypotheses rather than demonstrated phenomena. No quantitative measure or theoretical justification is provided for why embedding the conditional information into the noise term, as opposed to the mean term, should improve results. The experimental section reports moderate numerical gains but does not investigate what aspects of NCS drive these improvements—whether due to the noise combination itself, implicit regularization effects, or differences in implementation.

Overall, the work reads as a well-presented reformulation of known methods rather than a fundamentally new contribution. It could be strengthened by deeper theoretical analysis and more targeted experiments probing why the approach works. If the authors are able to answer the following questions in the next section, it would be a big help for us.

### Questions
1. Could the authors provide a theoretical or empirical explanation for why replacing the noise term with an optimally combined version improves stability or reconstruction quality? Is there any evidence (e.g., manifold distance, variance analysis, or effective step size) showing that the NCS trajectory stays closer to the learned data manifold?

2. In practice, NCS appears mathematically similar to taking a guided noise step proportional to $
\nabla_x \log p(y\mid x)$. Could the authors clarify how NCS differs in effect from existing gradient-based corrections? Are there cases where the two produce substantially different trajectories?

3. How sensitive is performance to the codebook size K and the number of combined noise vectors m? Does increasing K always help, or does it introduce variance and instability due to correlation among noise samples? The method is restricted to linear inverse problems. Are there conceptual or mathematical obstacles to extending NCS to nonlinear or learned degradation operators (e.g., differentiable renderers, neural forward models)?

### Soundness
3

### Presentation
3

### Contribution
2
