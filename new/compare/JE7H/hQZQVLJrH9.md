---
job_id: d3ed7f26-f89d-4db9-8e35-2e03cc7120b4
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: hQZQVLJrH9.pdf
paper: A Unified First-Order Framework for Activation Steering and Data Influence
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies activation steering, influence functions, Jacobians, and generalization bounds for neural networks, which squarely fits ICLR’s focus on representation learning, interpretability, optimization, and learning theory.

## Minimum Quality
Pass ✅.  
The paper is in English and has all core sections: Abstract, Introduction, Background/Notation, Main Theory (methodology), Experiments, Related Work, and Conclusion. While I find nontrivial issues in soundness, positioning, and empirical validation, they do not rise to the level of immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no evidence of hidden prompts, steganographic text, or explicit attempts to manipulate AI-based reviewing in the main content.

---

# Expected Review Outcome:

## Summary

The paper proposes a unified first-order framework linking activation steering and training-data influence functions. By analyzing the parameter–logit and activation–logit Jacobians, the authors introduce Influence-Aligned Steering (IAS), which yields a minimum-norm activation perturbation that, under certain alignment conditions, matches the first-order logit shift from a parameter perturbation or influence re-weighting.  

They derive several theoretical results: a steering–influence equivalence at the data level (Theorem 4.2), alignment-based error bounds and impossibility results (Theorems 5.1 and 6.2), a spectral optimality characterization of steering directions (Theorem 5.3), and a Rademacher-complexity bound for low-rank steering (Theorem 6.1). Experiments on GPT‑2 detoxification and ResNet‑50 (Figures 1–3, Table 1) aim to empirically support the first-order approximation, alignment diagnostics, and the spectral direction heuristic.

---

## Strengths

1. **Conceptual unification of activation steering and influence functions.**  
   The paper identifies that both activation steering (Eq. (2)) and influence-based parameter updates (Eq. (1)) induce *linear* logit shifts governed by $\mathbf{J}_{h\to y}$ and $\mathbf{J}_{\theta\to y}$, then formalizes their relationship via the primal program (P) and its dual. The closed-form IAS solution $\Delta h^\star = J_{h\to y}^\dagger J_{\theta\to y} \Delta\theta$ (Eq. (2) in Section 3.2, Theorem 5.2) is a clean and interpretable construction that many practitioners in activation engineering and influence analysis could find useful.

2. **Clear geometric alignment diagnostics with principal angles.**  
   The use of $\gamma(x)$, the smallest principal-angle cosine between $\operatorname{Im}(J_{\theta\to y})$ and $\operatorname{Im}(J_{h\to y})$, is conceptually appealing. Theorem 5.1 provides a simple bound on the relative logit error in terms of $\sqrt{1-\gamma^2(x)}$, and Theorem 6.2 shows a no-free-lunch result when $\gamma$ is small. Figure 2 (layer-depth ablation) offers concrete evidence that $\gamma$ increases with depth in GPT‑2 Medium, which supports the proposed practical heuristic for layer selection.

3. **Data-level steering–influence equivalence and causal narrative.**  
   Theorem 4.2 and Corollary 1 argue that any small steering vector corresponds to a signed measure over training points whose weighted influence reproduces the same first-order logit shift, and that the constructed measure $\rho_s$ is $\ell_1$-minimal. This gives a neat narrative: steering a behavior is equivalent to re-weighting a minimal set of training examples, which in principle enables tracing a successful steering direction back to “causal” data points.

4. **Spectral optimality perspective for steering directions.**  
   Theorem 5.3 characterizes, under an $\ell_2$ norm budget, the steering vector that maximizes expected first-order logit change as the top eigenvector of a Fisher–influence-style matrix $\Sigma$. This moves beyond hand-crafted or heuristic steering vectors by giving a clear spectral optimization objective, with a concrete, scalable power-iteration procedure in Section 5.3. Figure 3 qualitatively supports that the spectral direction for the “horse” class in ResNet‑50 lies in the tail of the null distribution of random directions.

5. **Attempt to bound generalization impact of low-rank steering.**  
   Theorem 6.1 and inequality (4) adapt Rademacher-complexity results to argue that a rank‑$k$ IAS correction with small magnitude $\alpha$ causes an additive blow-up of order $\alpha L\sqrt{2k/(dn)}$ in empirical complexity. This is a useful sanity check for people deploying low-rank steering: it suggests that modest, low-rank steering will not catastrophically harm generalization as $d,n$ grow.

6. **Some empirical support for first-order approximations.**  
   Figure 1 shows predicted versus actual logit shifts for $n=5000$ prompt-token pairs with cosine 0.978 and slope 1.50. Despite the slope mismatch (discussed below), the near-collinearity supports the claim that first-order IAS roughly tracks influence updates in a small-edit regime.

7. **Figures and table are well chosen to match key theoretical claims.**  
   - **Figure 1** specifically visualizes the central first-order equivalence claim, plotting predicted $\Delta y$ vs. actual $\Delta y$ under IAS/influence; it allows a quick visual assessment of linearity and residual variance.  
   - **Figure 2** nicely captures how $\gamma$ varies with layer index, giving practical weight to the alignment diagnostic.  
   - **Figure 3** tests the spectral optimality concept by comparing the spectral steering direction’s radius against a null distribution of random directions.  
   - **Table 1** directly compares IAS to Contrastive Activation Addition (CAA) on toxicity and perplexity, tying the theoretical contributions to an activation-steering baseline.

---

## Weaknesses

I will be deliberately detailed and critical here; many points are addressable but currently weaken the paper’s case.

1. **Steering–influence equivalence is weaker than the abstract suggests, and the assumptions are under-discussed.**  
   The abstract claims “we prove that, to first order, these techniques are equivalent: any steering vector can be represented as an influence weighting over training data and vice versa.” However, in the main text:  
   - Theorem 4.2 requires that the set $\{\mathcal{I}(z\to x)\}_{z\in \mathcal{Z}}$ spans $\operatorname{Im}(J_{h\to y})$ for exact equivalence; otherwise Eq. (4) only holds up to a residual bounded by $(1-\gamma(x)^2)^{1/2}\|\alpha s\|$. This important condition is only briefly mentioned under “Residual when spans do not match” on **Page 4**, but is not carried through the abstract or conclusions.  
   - The forward direction (steering → influence) and the reverse one (influence → steering) are stated compactly but without fully explicit constructions or proof sketches in the main text. For the reverse direction, the claim “Conversely, any signed weighting $\mathbf{w}$ … admits a steering vector $s_w$ … that realizes the same first-order output shift” assumes nontrivial conditions on the geometry of $J_{h\to y}$ and $J_{\theta\to y}$ and the influence parameter update; these conditions are neither spelled out nor proven in detail.  
   As a result, the equivalence feels more like a conditional correspondence that holds on nice subspaces rather than the near-universal statement suggested in the opening. The paper should clearly delineate the exact conditions (rank assumptions, subspace inclusions, properties of the Hessian surrogate) under which both directions hold.

2. **Mathematical derivations are often sketched rather than fully justified, which undermines some central claims.**  
   Several key statements are given as “idea of proof” or “sketch” without enough detail to check correctness:  
   - **Lemma 4.1** (chain-rule factorization) is stated but the sketch “differentiate $m_\theta$ along the composite map” elides assumptions about differentiability and the dimensions of $m_\theta$. Given that this lemma underlies expressing parameter gradients in terms of activation-space gradients (and is used implicitly in later theorems), a more explicit derivation would be welcome.  
   - **Theorem 4.2** is pivotal, yet its proof is entirely omitted in the main text. Equation (4) asserts $\| \rho_s \|_1 = |\alpha|$, and the converse mapping from $\mathbf{w}$ to $s_w$ is summarized without detailing how the $\ell_1$-minimal measure is constructed, how signed measures interact with the Hessian inverse, and how the bound on $\|s_w\|$ arises. For example, if $\Delta \theta$ is constructed as $\sum_z w_z \Delta \theta_z$ with $\Delta \theta_z = -\epsilon H^{-1} \nabla_\theta \ell(z, \theta)$, it is not obvious how the norm of the corresponding $s_w$ scales strictly as $O(\epsilon)$ without extra Lipschitz or conditioning assumptions on $J_{h\to y}^\dagger J_{\theta\to y} H^{-1}$.  
   - In **Theorem 5.3**, the definition of $\Sigma$ involves a product $J_{\theta\to h}^\top H^{-1} \nabla_\theta \ell(z) \nabla_\theta \ell(z)^\top H^{-1} J_{\theta\to h}$. To reach the conclusion that the top eigenvector of $\Sigma$ maximizes expected first-order logit change, one needs a clear objective derivation, e.g., showing that the expected squared logit shift or its variance equals $s^\top \Sigma s$ for $\|s\|\le B$. This link is asserted but not demonstrated. As it stands, the reader has to infer the objective from standard quadratic-form optimization, but the paper never clearly specifies *which* expected quantity is maximized.  
   - **Theorem 6.1** only states “Sketch. Combine Thm. 2 of Pinto et al. (2024) with the fact that IAS changes only a rank‑$k$ sub-matrix of the layer weight.” This glosses over important details: how exactly does IAS equate to a rank‑$k$ change in the weight matrix at that layer (especially with the pseudoinverse-based construction), and how are the Lipschitz and norm constraints mapped onto the assumptions in Pinto et al.? Without at least a high-level derivation, the bound in Eq. (4) feels somewhat disconnected.  
   Overall, the math is promising but underspecified in the main paper. At ICLR standards, at least one of the core theorems (Theorem 4.2 or 5.3) should be fully derived in the main text, including the precise assumptions and intermediate steps.

3. **The influence-function side inherits well-known fragility issues, which are not addressed.**  
   Section 2 acknowledges the use of a damped Gauss–Newton surrogate for $H_\theta$ and sets $\lambda>0$ for stability, but the paper does not confront the extensive literature on the fragility of influence functions in deep networks (e.g., Basu et al., 2021, already cited). The whole IAS construction relies on accurate approximations of $H^{-1}\nabla_\theta \ell(z)$ and $J_{\theta\to y}$, yet:  
   - There are no experiments examining robustness of IAS to Hessian damping $\lambda$ or mis-estimation of $H^{-1}$, nor any sensitivity analysis.  
   - There is no empirical confirmation that influence estimates used to build IAS are numerically stable on GPT‑2 Medium; e.g., how large is the variance in predicted vs. actual logit shifts across different damping factors?  
   Considering that IAS is pitched as a “practical workflow” for billion-parameter models, this omission is important. Without some robustness study or at least principled discussion of conditioning, step sizes $\alpha$, and Hessian approximation error, the practical reliability of the framework is unclear.

4. **Experiments are thin, lack ablations, and do not directly stress-test the main theoretical predictions.**  
   - **Table 1 (detoxification)** compares IAS only against CAA, with mean toxicity scores (0.0195 baseline vs. 0.0150 CAA vs. 0.0164 IAS) and perplexities (baseline 14333 vs. 13291 CAA vs. 13701 IAS). These are single numbers with no error bars, no significance testing, and no exploration of how performance varies with $\alpha$, layer $\ell$, or rank $k$. IAS even underperforms CAA slightly on both toxicity and PPL in this single configuration, which does not convincingly demonstrate any advantage of the theoretically principled approach.  
   - The first-order equivalence is only evaluated via **Figure 1**, which plots predicted vs. actual logit shifts with cosine 0.978 and slope 1.50. The high cosine is encouraging, but the slope being 1.5 suggests systematic over- or under-scaling in the first-order prediction. There is no further analysis: for example, how does this slope behave across layers, different magnitudes of $\alpha$, or different model architectures? Does adjusting the damping or step size correct this bias? A deeper analysis would greatly strengthen the claim.  
   - **Figure 2** shows that $\gamma$ increases with depth for GPT‑2 Medium, but the paper does not actually leverage this to compare IAS performance at different layers. An obvious experiment would be to run detoxification steering at shallow vs. deep layers and show that higher $\gamma$ correlates with better match between IAS and influence, or better trade-offs between toxicity reduction and PPL.  
   - **Figure 3** (spectral shift significance) displays a histogram of “spectral radius” values for random directions and a dashed vertical line for the estimated spectral direction, with $p = 0.00498$. This is qualitatively interesting, but does not show actual task-level improvements (e.g., increased classification confidence when steering along the spectral direction vs. random). As stated, it mostly confirms the obvious: that the top eigenvector of a covariance-like matrix has larger quadratic form than typical random directions.  
   Overall, the experiments feel like proofs-of-concept rather than the systematic, multi-faceted evaluation one expects for a paper proposing a practical unified framework.

5. **The “causal” data attribution story is oversold relative to what is actually demonstrated.**  
   The abstract and Section 4.1 emphasize that $\rho_s$ “pinpoints the fewest training examples to relabel/remove/examine to reproduce the behavioral change” and “points straight to the most causal training documents.” However:  
   - There is *no* empirical experiment that actually computes $\rho_s$ for a real steering vector, retrieves the top-ranked training examples, and demonstrates that relabeling or removing them modifies the model behavior in the same way.  
   - The minimal-$\ell_1$ property in Corollary 1 is contingent on affine independence of the influence vectors, which is a strong and unverified assumption in realistic large-scale datasets. In overparameterized regimes, affine dependence is very likely, so the uniqueness of the minimal measure and the “fewest examples” interpretation become questionable.  
   - Influence functions are known to be unstable in deep networks, which further complicates any strong causal language.  
   Without at least one experiment where $\rho_s$-identified samples are inspected or edited, this part of the contribution remains speculative.

6. **Insufficient positioning with respect to closely related recent work on activation steering and unified perspectives.**  
   The Related Work section is quite short and omits several directly related recent papers:  
   - **Adila et al., 2026: “Weight Updates as Activation Shifts: A Principled Framework for Steering.”** This work also establishes a first-order equivalence between activation-space interventions and weight-space updates, which is conceptually extremely close to the Jacobian-based primal–dual view here. It should be cited and compared against in Section 3 or 4, especially since both works claim “principled frameworks” for steering.  
   - **Cui & Chen, 2025: “Painless Activation Steering: An Automated, Lightweight Approach for Post-Training Large Language Models.”** This work focuses on automated learning of activation steering vectors, which relates to the spectral recipe in Theorem 5.3 and could provide alternative baselines beyond CAA in Table 1.  
   - **Dan et al., 2025: “A2A: Mechanistic Analysis for Efficient Layer Selection in Activation Steering.”** This is directly relevant to the layer-selection heuristic based on $\gamma$ and Figure 2; at minimum it should be acknowledged and contrasted in Section 5 or 7.3.  
   - **Bigelow et al., 2025: “Belief Dynamics Unify In-Context Learning and Activation Steering.”** This paper also seeks a unifying perspective (Bayesian belief dynamics) on context- and activation-based control of language models. The present work’s “unified first-order framework” should be discussed in relation to this.  
   Currently, the paper positions itself as the “first to give a closed-form map” between steering and influence, but the omission of closely related work that also relates activation shifts to weight updates weakens this claim and may be misleading for readers.

7. **Ambiguous or overloaded notation and missing clarifications in several equations.**  
   There are cases where the notation is either ambiguous or used inconsistently:  
   - In **Equation (2)** on **Page 3**, the same label “(2)” is used previously on **Page 3** for the dual solution, which can cause confusion. The numbering of equations 1–4 is also inconsistent: Eq. (2) appears both as $\Delta y^{\mathrm{SV}}(x)$ and earlier as the dual solution. This makes cross-referencing difficult.  
   - The dimensions of $\Delta y$ and of the loss $\ell$ are not always clear. For instance, Theorem 5.3 states that the achievable change equals $B\sqrt{\lambda_{\max}(\Sigma)}\|\nabla_h f_\theta(x)\|$, but does not clarify if this is in some aggregated scalar metric or in $\ell_2$ norm over logits; the phrase “expected first-order logit change” is vague. One would expect a formal objective like $\mathbb{E}_z[\|J_{h\to y}(x) s\|_2^2]$ or $\mathbb{E}_z[\Delta y^\top \Delta y]$ to be given explicitly.  
   - In Theorem 6.1, the form $\hat f = f_\theta + \alpha U V^\top$ is somewhat confusing: $f_\theta$ maps inputs to logits, while $U V^\top$ is described as an IAS correction at a layer, so strictly speaking it is not additive in the same function space. The mapping between a rank-$k$ change at a weight matrix and the induced functional change should be written more carefully.

8. **Figures and Table are not fully exploited to validate the theory.**  
   While Figures 1–3 and Table 1 are well aligned with the paper’s story, they are rather minimal:  
   - **Figure 1**: Only a global cosine and slope are reported. There is no breakdown by layer, prompt type, or magnitude of the underlying perturbation. Also, the substantial spread around the fitted line indicates notable residual error; quantifying this (e.g., R^2, variance explained, or RMSE) would help readers gauge how “first-order” the behavior really is.  
   - **Figure 2**: It would be natural to overlay, for a fixed steering task, some measure of steering success (e.g., toxicity reduction, logit-shift match) versus layer index alongside $\gamma$. Currently, the figure suggests an interesting geometric pattern but does not connect it directly to performance.  
   - **Figure 3**: The histogram of “spectral radius” for random directions and the vertical line for the spectral direction is one-dimensional; there is no evidence that steering along this spectral direction helps classification, calibration, or robustness.  
   - **Table 1**: There are no standard deviations across seeds, nor any ablations or alternative metrics. Without uncertainty estimates, the small differences between CAA and IAS could easily be noise.

9. **Claims about computational scalability are not substantiated with concrete benchmarks.**  
   The paper repeatedly emphasizes that IAS is computationally cheap (“two backward passes per input” in Section 1, the cost model in Section 2, and “practical workflow” in several places). However, there are no wall-clock runtime measurements, memory estimates, or scaling experiments. For instance, computing $J_{h\to y}^\dagger$ and principal angles $\gamma$ via SVD for large layers could be costly; no evidence is given that this is manageable for modern large models beyond GPT‑2 Medium.

10. **Use of strong language (e.g., “provably insufficient”, “no-free-lunch”) vs. practical regimes.**  
    Theorem 6.2 formally states that if $\gamma(x)\le \rho<1$, then $\|J_{h\to y}\Delta h\|/\|J_{\theta\to y}\Delta \theta\|\le \rho$, which is mathematically sound but essentially restates that orthogonal components cannot be reached. While true, calling this a “no-free-lunch” theorem might be overselling the insight, especially since $\gamma$ in real models (Figure 2) can be fairly high at later layers. This is more of a geometrical sanity check than a deep impossibility result. The paper could moderate its language and devote more effort to empirically validating where, in practice, $\gamma$ is small and steering indeed fails.

Taken together, these issues add up. The core idea is interesting, but the theoretical derivations in the main text are thin, experiments limited, and the positioning against closely related work incomplete.

---

## Potentially Missing Related Work

The following works are directly relevant and should be cited and discussed:

1. **Adila, D., Cooper, J., Yun, A. (2026): “Weight Updates as Activation Shifts: A Principled Framework for Steering.”**  
   - *Relevance*: Establishes a first-order equivalence between weight updates and activation shifts, which conceptually overlaps heavily with the IAS primal–dual view.  
   - *Where to add*: Discuss in Section 3 (primal/dual program) and Section 4 (steering–influence equivalence), explicitly comparing their framework to IAS and clarifying what is new here (e.g., data-level influence interpretation, principal-angle diagnostics, etc.).

2. **Cui, S., Chen, Z. (2025): “Painless Activation Steering: An Automated, Lightweight Approach for Post-Training Large Language Models.”**  
   - *Relevance*: Proposes automated ways to learn activation steering vectors for LMs, which overlaps with the practical goal of IAS and is a relevant baseline for steering performance.  
   - *Where to add*: Related Work (Section 8) and Section 7.1; compare IAS-based steering to an automated method like Painless Steering rather than only to CAA.

3. **Dan, Y., Lin, J., Chen, Q. (2025): “A2A: Mechanistic Analysis for Efficient Layer Selection in Activation Steering.”**  
   - *Relevance*: Addresses efficient layer selection for activation steering using mechanistic analysis, closely related to the use of $\gamma(x)$ and Figure 2 for layer selection.  
   - *Where to add*: Section 5 (alignment and layer-wise composability) and Section 7.3; compare the $\gamma$ heuristic with mechanistic layer-selection approaches.

4. **Bigelow, E., Wurgaft, D., Wang, Y. (2025): “Belief Dynamics Unify In-Context Learning and Activation Steering.”**  
   - *Relevance*: Provides another “unifying” perspective on activation steering, albeit from a Bayesian belief-dynamics angle.  
   - *Where to add*: Section 1 and 8; discuss conceptual similarities and differences in how “unification” is framed.

5. **Shrikumar, A., Greenside, P., Shcherbina, A. (2016): “Not Just a Black Box: Learning Important Features Through Propagating Activation Differences.” (DeepLIFT)**  
   - *Relevance*: DeepLIFT relates activations at different reference points to importance scores and can be seen as an early method for propagating activation changes.  
   - *Where to add*: Section 2 or 8 when discussing interpretability and activation-based explanations.

6. **Lee, J. H., Lanza, S., Wermter, S. (2024): “From Neural Activations to Concepts: A Survey on Explaining Concepts in Neural Networks.”**  
   - *Relevance*: A survey of explaining neural network concepts via activations; useful background for readers on activation-based interpretation.  
   - *Where to add*: Related Work (Section 8), to situate activation steering within the broader family of activation-based interpretability.

7. **Lederer, J. (2021): “Activation Functions in Artificial Neural Networks: A Systematic Overview.”**  
   - *Relevance*: Less directly related, but provides foundational context on activations and their properties; may help clarify assumptions on smoothness and Jacobians.  
   - *Where to add*: Optional citation in Section 2 when discussing model and activation-space Jacobians.

---

## Questions

Author responses that concretely address the following could substantially improve my assessment:

1. **Precise conditions and full proof for Theorem 4.2.**  
   - Please provide, in the rebuttal and (ideally) in a revised main text, a more detailed proof of Theorem 4.2.  
   - Specifically:  
     - Under what exact rank/conditioning assumptions on $J_{h\to y}$, $J_{\theta\to y}$, and the influence parameter update does the equivalence hold in both directions?  
     - How is the $\ell_1$-minimal measure $\rho_s$ explicitly constructed, and how do you guarantee $\|\rho_s\|_1 = |\alpha|$ in the presence of potential affine dependence?  
     - How is the $O(\epsilon)$ scaling of $\|s_w\|$ obtained for the converse mapping $\mathbf{w} \mapsto s_w$?

2. **Clarification of the objective in Theorem 5.3.**  
   - What exact scalar quantity is being maximized by $\mathbf{s}_{\max}$? Is it $\mathbb{E}_z[\|\Delta y(x)\|_2^2]$, or some other functional?  
   - Please include the derivation that shows the expected first-order change can be written as $s^\top \Sigma s$.

3. **Robustness to Hessian approximations and damping.**  
   - Can you provide empirical or theoretical evidence on how sensitive IAS is to the choice of damping $\lambda$ in $(H+\lambda I)^{-1}$? For example, do you observe large changes in the cosine/slope in **Figure 1** when varying $\lambda$?  
   - Are there conditions (e.g., spectral properties of $H$) under which IAS remains stable?

4. **Empirical demonstration of data-level attribution via $\rho_s$.**  
   - Can you add an experiment where you compute $\rho_s$ for a steering vector (e.g., the detoxification task), inspect the top-k training examples, and either show that removing or reweighting them leads to the same behavioral change?  
   - This would significantly bolster the causal-data narrative.

5. **More systematic evaluation of $\gamma$ and steering performance.**  
   - Is it feasible to add experiments that vary the layer $\ell$ and show:  
     - The distribution of $\gamma(x)$ (as in Figure 2),  
     - The match between predicted and actual logit shifts (like Figure 1), and  
     - Task-level performance (e.g., toxicity vs. PPL)?  
   - Such a three-way comparison could test Theorems 5.1 and 6.2 more directly.

6. **Comparison to more recent activation-steering methods.**  
   - Can you clarify how IAS differs empirically and conceptually from works such as “Painless Activation Steering” and “A2A”? For example, could IAS be used as a diagnostic or initialization within those methods?  

7. **Scalability and complexity.**  
   - Please provide practical runtime/memory benchmarks for computing IAS, $J_{h\to y}^\dagger$, and $\gamma$ on GPT‑2 Medium, and discuss whether you have tried larger models. How does the cost compare to more standard steering approaches like CAA or LoRA fine-tuning?

Concrete answers and possibly modest additional experiments would help clarify whether IAS is mainly a conceptual lens or a practically superior method for real systems.

---

## Flag For Ethics Review

No ethics review needed.  

(Tasks involve standard language-model detoxification and ImageNet classification without novel data practices or sensitive subjects.)

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

2: fair.  
The high-level ideas and many of the theorems are plausible and mathematically reasonable, but several core results (especially Theorem 4.2 and Theorem 5.3) are only sketched, with important assumptions left implicit. Empirical support is limited and does not fully test robustness or the claimed practical workflow.

---

## Presentation Rating

2: fair.  
The paper is generally readable and structured, and the figures/tables align with the story, but important derivations are missing from the main text, some notation is ambiguous, and the positioning with respect to closely related work is incomplete.

---

## Contribution Rating

2: fair.  
The conceptual unification of steering and influence via Jacobian geometry and principal angles is interesting and potentially useful, but the strength of the claimed equivalence and causal interpretation is weakened by missing proofs, thin experiments, and incomplete comparison to related frameworks.

---

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper presents an appealing and conceptually coherent framework connecting activation steering and influence functions through first-order Jacobian geometry, with neat ideas like principal-angle diagnostics and spectral steering directions. However, the central equivalence results are insufficiently detailed in the main text, the empirical evaluation is limited and does not fully validate the claimed practical workflow or causal data attribution, and key recent related work is missing. With stronger proofs, more systematic experiments (especially around $\gamma$ and $\rho_s$), and better positioning, this could become a solid contribution; in its current form, it falls short of ICLR’s bar.

---

## Reviewer Confidence

4: confident.  
I am reasonably familiar with influence functions, activation steering, and Jacobian-based analysis, and I have carefully gone through the equations and figures in the main paper. Some details delegated to the appendix could change specific judgments, but they are unlikely to reverse the overall assessment.