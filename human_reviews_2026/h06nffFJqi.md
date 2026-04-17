# What Exactly Does Guidance Do in Masked Discrete Diffusion Models

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Masked discrete diffusion models have been gaining popularity recently, and classifier-free guidance, just like its continuous counterpart, has been proposed to enable efficacious conditional generation by discrete diffusion. To quantify the precise effect of discrete guidance, this article considers masked discrete diffusion with arbitrary data distribution in low dimension, so that 
   the distribution that guided masked discrete diffusion samples from, as well as the sampling dynamics, can be analytically and exactly quantified and interpreted. When the full data distribution is a mixture over classes and the goal is to sample from a specific class, guidance amplifies class-specific regions while suppresses regions shared with other classes. This effect depends on the guidance strength $w$ and induces distinct covariance structures in the sampled distribution. Notably, we observe quantitatively different behaviors in $1$D and $2$D.  We also show that for large $w$, the decay rate of the total variation ($\text{TV}$) along the reverse dynamics is double-exponential in $w$ for both $1$D and $2$D. These findings highlight the role of guidance, not just in shaping the output distribution, but also in controlling the dynamics of the sampling trajectory. Our theoretical analysis is supported by experiments that illustrate the geometric effects of guidance and its impact on convergence.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a theoretical analysis of classifier-free guidance (CFG) in masked discrete diffusion models. In 1D, the guided reverse process exactly recovers the tilted distribution. In 2D, deviations emerge: the final distribution is a reweighted version of the tilted distribution, with mass suppressed even in regions overlapping only in projection.

The authors derive closed-form expressions in both settings and show that convergence to the final distribution exhibits double-exponential decay in the guidance strength w. Experiments in higher dimensions (5D, MNIST) confirm that these geometric effects—amplifying private regions and suppressing ambiguous ones—persist beyond low-dimensional cases.

### Strengths
I like this work. The paper is well-written and clear. It presents a novel theoretical insight into how classifier-free guidance (CFG) affects discrete diffusion sampling, with concrete examples and rigorous proofs. The authors also provide high-dimensional experiments to support the generality of their findings.

### Weaknesses
It would be interesting to discuss how this effect manifests in practical applications, and how the misalignment introduced by CFG could be addressed in real-world settings.

### Questions
See weakness.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper develops the first rigorous quantitative theory explaining the role of classifier-free guidance (CFG) in masked discrete diffusion models.
The authors analyze low-dimensional cases (1D and 2D) where the reverse dynamics can be solved exactly. Their results show that:
- Guidance amplifies class-specific regions while suppressing overlapping regions between classes, with overlap vanishing as guidance strength $w$ increases.
- In 1D, the generated distribution exactly matches the tilted distribution $p_{z,w} \propto p(x)p(z|x)^{1+w}$.
- In 2D, deviations emerge but can be expressed in closed form via coupling coefficients $c_x,d_x$.
- The convergence rate of the reverse dynamics exhibits a double-exponential dependence on w in both 1D and 2D.
Empirical illustrations confirm the theoretical predictions.

### Strengths
- Novel theoretical framework: First rigorous analysis of discrete CFG dynamics.
- Analytic tractability: Closed-form results for both 1D and 2D masked diffusion.
- Clear phenomena: Demonstrates class-specific amplification and overlap suppression quantitatively.
- Double-exponential convergence: Elegant link between guidance strength and diffusion rate.
- Bridges gaps: Unifies discrete and continuous CFG theories.
- Empirical alignment: Simulations verify analytical predictions.

### Weaknesses
- Heavy reliance on exact score and continuous-time limit; numerical approximations and learned scores are not analyzed.
- Empirical validation is illustrative rather than large-scale.
- Some proofs deferred to appendices could benefit from intuitive discussion in the main text.
- Limited exploration of $D \ge 3$ behavior; higher-dimensional extension remains conjectural.
- Minor presentation complexity (dense notation, multi-index expressions).

### Questions
- Could the authors extend the proof techniques to approximate scores (learned $s_\theta$) or noisy simulations?
- How sensitive are the observed phenomena to the choice of the forward process (absorbing vs. uniform)?
- Is the double-exponential convergence rate provably tight, or an upper bound?
- In 2D, does the deviation from the tilted distribution scale polynomially or exponentially in overlap size?
- Could these results suggest a new adaptive guidance schedule where w increases dynamically?
- How might the regional weighting structure $A^{z,w}_i$ inform geometry-aware training or regularization?
- Are there potential connections to discrete optimal transport or entropic regularization frameworks?
- Could the asymptotic results be empirically observed in token-level text diffusion models?
- Are the coefficients $c_x,d_x$ interpretable as marginal reweighting factors for token dependencies?
- How might partial overlapping supports in high dimensions influence generalization or calibration?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper extend the analysis of CFG in continuous state diffusion to the masked discrete state diffusion. It has two main results:

* In 1d situation, for masked discrete state diffusion, with direct construct of \\(\hat{Q}\\), we can reach the tilted distribution.
* In 2d situations, it is not that simple.
* This paper also proposes some analysis on multi-guide situation.

### Strengths
* The theoretical analysis and calculations are solid, especially the ability to derive exact distribution results— outperforming continuous-state diffusion on CFG in the 1D case.
* The inclusion of the multi-guidance setting adds depth and richness to the paper.

### Weaknesses
* Unfortunately, the 1D setting is overly simplistic. While the proposed techniques are effective in 1D, they become increasingly complex in * 2D and are difficult to generalize to higher dimensions due to inherent limitations.
* The experimental evaluation relies too heavily on toy examples, which weakens the practical impact of the work.

### Questions
n/a

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a rigorous analysis of classifier-free guidance (CFG) for masked discrete diffusion. Under exact scores and exact reverse dynamics, the authors derive closed-form reverse dynamics and generated distributions in 1D and 2D. In 1D, CFG precisely samples from the tilted distribution pz,w, whereas in 2D, the generated distribution deviates with explicit marginal dependent reweighting, shifting mass from overlapping to class-specific regions. For large guidance strength $w$, the total variation (TV) to the terminal distribution decays in time with a rate that is double-exponential in $w$. The paper features experiments on synthetic 1D, 2D and higher-dimensional toy setups, plus MNIST case studies, which qualitatively support the theory.

### Strengths
(i) The paper provides the first rigorous treatment of CFG in discrete masked diffusion with explicit formulas in 1D and 2D

(ii) The paper provides clear geometric interpretation. Guidance suppresses overlapping regions and amplifies "private" regions, quantified via region-wise weights in 2D

(iii) The paper's simple, targeted experiments align with theory

### Weaknesses
(i) The paper's scope appears limited to masked absorbing diffusion and low dimensions, an extension to higher dimensionality $D > 2$ remains informal

(ii) The paper poses idealized assumptions (exact scores, exact reverse simulation) with little analysis of approximation / discretization error or robustness under practical solvers

(iii) While the main contribution of this work is a theoretical discussion, the scope of experiments remains thin, and the robustness and scalability of results is underexplored

### Questions
In addition to the weaknesses outlined in points (i-iii), I present the following questions for the authors to address:

(1) Proposition 3.3 partitions the space into regions $\mathcal{R}_1, \ldots, \mathcal{R}_4$. Can you provide examples of real data where such region decompositions would be meaningful?

(2) The analysis assumes exact concrete scores. How would score approximation errors change the conclusions?

(3) The findings are specific to masked discrete diffusion, following the absorbing forward process. Would the conclusions still hold for other discrete processes (e.g., D3PM-uniform modelling)?

(4) Strong guidance suppresses shared regions and reduces sample diversity. Is this “loss of diversity” always undesirable, or can it be beneficial in certain applications?

(5) What are the main obstacles to extending the exact analysis beyond 2D, and what aspects of the results are most likely to generalize?

(6) In 2D you introduce marginal coefficients $\lbrace c_x, d_x \rbrace_{1 \leq x \leq N}$ that encode the steering effect of guidance on marginals, consequently influencing the sampling dynamics. What intuition can we build for how these coefficients arise and what they represent?

(7) [minor concern w.r.t. to presentation] Figure 2 feels slightly out of place, can you place it at the top of page 8?

(8) [minor concern w.r.t. to presentation] While Notation was shortly introduced in the preliminaries section, could you provide a simple table of notations in the appendix to further improve overall readability and accessibility of your work?

### Soundness
3

### Presentation
2

### Contribution
3
