---
job_id: c088f8e0-0589-40f6-b801-6d2fb1bb8f73
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: uP6RDWHcs7.pdf
paper: Marginal Flow: A Flexible and Efficient Framework for Density Estimation
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new deep generative modeling framework for density estimation, with applications to simulation-based inference, positive-definite matrices, and manifold learning; this is squarely within probabilistic methods, generative models, and representation learning topics appropriate for ICLR.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Method/Model, Experiments/Results, Related Work, Conclusion) are present and written in English. The method is clearly specified (Equations 1–3, 5–11, 15–23), experiments are reasonably extensive (synthetic, SBI, Wishart mixtures, image manifolds), and there is nontrivial technical content. While there are weaknesses (positioning vs prior work, some methodological and evaluation gaps), they do not amount to a fatal flaw requiring desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No hidden instructions or attempts to manipulate automated reviewing are apparent in the provided main paper content. The text is standard scientific prose without suspicious prompts.

---

# Expected Review Outcome:

## Summary

The paper proposes **Marginal Flow**, a density estimation framework where a tractable parametric family \(q(\mathbf{x}\mid\mathbf{w})\) (often Gaussian with diagonal covariance) is mixed over latent parameters \(\mathbf{w}\) drawn from a learned distribution \(q_\theta(\mathbf{w})\). Instead of explicitly modeling or evaluating \(q_\theta(\mathbf{w})\), the method samples latent codes via an unconstrained neural network \(f_\theta(\mathbf{z})\) that maps base noise \(\mathbf{z}\sim p_{\text{base}}\) to \(\mathbf{w}\), and defines the model density as the Monte Carlo mixture in Eq. (2).  

The authors argue that this setup yields a model that allows (approximate) exact density evaluation and efficient single-step sampling, can handle multi-modality and lower-dimensional manifolds by choosing the base support dimension \(m<d\), and is flexible with respect to the choice of the parametric family \(q(\mathbf{x}\mid\mathbf{w})\). Extensive experiments on synthetic 2D distributions, simulation-based inference benchmarks, Wishart mixture distributions on SPD manifolds, and image latent spaces (MNIST, JAFFE) showcase runtime advantages and qualitative modeling capabilities.

---

## Strengths

1. **Conceptually simple construction with clear mathematical formulation.**  
   The core model is crisply defined via marginalization in Eq. (1) and its Monte Carlo instantiation in Eq. (2), with latent parameters sampled through Eq. (3). This makes the method easy to implement and reason about: it is essentially a learnable mixture of a chosen parametric family \(q(\mathbf{x}\mid\mathbf{w})\) where the mixture locations (and in principle other parameters) are outputs of a neural network. The extension of standard mixture models to a “marginalized over \(\mathbf{w}\)” form, with re-sampling of \(\mathbf{w}_i\) every evaluation, is well explained.

2. **Good discussion and partial analysis of the Monte Carlo estimator bias/variance.**  
   The appendix (Section A.5) gives a nontrivial analysis of the bias and variance of the nested Monte Carlo estimator \(\widehat{\mathcal{L}}_{N,N_c}\) of the KL objective, invoking Theorem A.1 (Hong & Juneja) and deriving Lemma A.2 and Theorem A.3. Equations (20)–(21) quantify the bias and variance scaling in terms of the intrinsic manifold dimension \(m\) rather than the ambient dimension \(d\), which is a meaningful insight for high-dimensional but manifold-structured data. This is an above-average level of rigor for what is basically a mixture-of-experts style model.

3. **Runtime efficiency and empirical scaling evidence.**  
   Figure 3 directly compares runtime (for 100 samples and 100 density evaluations) between Marginal Flow, Normalizing Flows (NF), Flow Matching (FM), and Free-form Flows (FFF) across increasing dimensions \(d\). Marginal Flow is convincingly shown to be substantially faster, particularly for density evaluation (right panel of Figure 3) where NF and FM quickly become slow or even hit OOM, whereas the proposed method scales nearly flat for sampling and much more gently for evaluation. This is consistent with the fact that no Jacobian determinants or ODE solves are needed.

4. **Ability to work with lower-dimensional manifolds, illustrated clearly.**  
   Section 2.3 and Figure 4 provide a very clear and visually compelling demonstration of learning densities supported on a 1D manifold in 2D. The authors set \(m=1<d=2\), and the center panel of Figure 4 shows that Marginal Flow recovers both the manifold and density, whereas Free-form Flows distort the manifold and NF/FM cannot reduce dimensionality (their panels are explicitly marked “No manifold available”). This figure strongly supports the claim that Marginal Flow can learn distributions with support on unknown low-dimensional manifolds.

5. **Multi-modal density modeling without explicit bijectivity constraints.**  
   Section 2.3 and Figure 5 show a multi-modal 2D target density with very few training points, and compare Marginal Flow, FM, NF, and FFF. The Marginal Flow panel (Figure 5, “Marginal Flow”) captures all modes fairly sharply, whereas the other methods visually miss modes or smear them due to inductive biases or training instabilities. This is a nice qualitative piece of evidence that not enforcing bijectivity between a simple base and target can be beneficial, in line with prior criticism of flow bijectivity.

6. **Flexibility in the choice of parametric family \(q(\mathbf{x}\mid\mathbf{w})\), exploited in nontrivial ways.**  
   The paper does not stop at Gaussian mixtures. Section 4.3 sets \(q(\mathbf{x}\mid\mathbf{w})\) to Wishart distributions on SPD matrices (Eq. (4)), with the latent \(\mathbf{w}_i\) parameterizing the scale matrices and a global degree-of-freedom \(\nu\). Figure 9 left shows lower test KL divergence compared to an NF baseline on \(10\times 10\) matrices, while also reconstructing the underlying manifold \(\mathcal{M}\). The right panel of Figure 9 demonstrates that Marginal Flow scales to \(100\times 100\) matrices where the NF baseline is deemed computationally infeasible. This nicely illustrates the modularity of the framework with respect to the choice of \(q(\cdot\mid\cdot)\).

7. **Empirical versatility across tasks (synthetic, SBI, manifolds in image latents).**  
   The experiments are broad:  
   - Synthetic 2D densities: Figure 6 and Figure 13 show that Marginal Flow can learn standard toy distributions (Two Moons, Pinwheel, Swiss Roll, Checkerboard, MoG) via forward KL. Figure 7 plots test log-likelihood vs epochs for various baselines, with Marginal Flow converging significantly faster.  
   - Reverse-KL training: Figure 8 compares Marginal Flow and NFs; the left panel’s bar plot of test reverse KL suggests competitive or better performance for Marginal Flow, and the right panel’s density plots show qualitatively better fits.  
   - SBI benchmark: Figure 14 (Appendix) shows C2ST scores across several tasks, where Marginal Flow is competitive or superior, especially in low data regimes.  
   - Image latent manifolds: Figures 10, 11, and 15–16 illustrate 1D and 2D learned manifolds in VAE latent spaces, with visually smooth interpolations across digits and emotional expressions.

8. **Architecture-agnostic and implementation-friendly.**  
   The method does not require invertible nets or special Jacobian-friendly architectures, just standard MLPs (Section A.1). The paper includes a brief but concrete explanation of efficient log-density computation for Gaussian components using `torch.cdist`, and claims to release code. This is practically valuable.

9. **Clear illustrative figures that aid understanding of the method.**  
   - Figure 1 nicely contrasts directly optimizing a finite set of Gaussians (GMM) vs the marginalization scheme, visually showing that the latter avoids “clumpy” mixture artifacts and yields a smoother density.  
   - Figure 2 decomposes the overall pipeline into three panels: sampling \(\mathbf{w}_i\) (left), evaluating \(q_\theta(\mathbf{x})\) (center), and sampling from \(q_\theta(\mathbf{x})\) (right). The graphical depiction of random vs deterministic nodes makes it very easy to grasp the flow of randomness and computation.

---

## Weaknesses

1. **“Exact density evaluation” claim is overstated and conflates the model definition with its Monte Carlo approximation.**  
   Throughout the paper (Abstract, Table 1, Sections 1–2.2, 5), the authors repeatedly assert that Marginal Flow provides “exact density evaluation”. However, the implemented model in practice is the *empirical mixture* in Eq. (2), which approximates the true marginal density in Eq. (1) via a finite Monte Carlo sum over \(N_c\) random latent draws:  
   \[
     q_\theta(\mathbf{x}) \approx \frac{1}{N_c}\sum_{i=1}^{N_c} q(\mathbf{x}\mid\mathbf{w}_{\theta,i}),
   \]
   with \(\mathbf{w}_{\theta,i}\sim q_\theta(\mathbf{w})\). For any fixed \(N_c\), the resulting density is random and differs from the infinite-sample marginal. The paper partially acknowledges this in Appendix A.5, where Theorem A.1 and Lemma A.2 derive a nonzero bias term
   \[
   \operatorname{Bias}(\widehat{\mathcal{L}}_{N,N_c}) = - \frac{1}{2N_c}\mathbb{E}\Big[\frac{\mathrm{Var}_{\mathbf{z}}(q(\mathbf{x}\mid f_\theta(\mathbf{z})))}{q_\theta(\mathbf{x})^2}\Big] + O(N_c^{-2}), 
   \]
   yet the main text continues to present density evaluation as exact. This matters because the comparisons to NF and other methods in Table 1 and runtime Figure 3 rely on this “exact likelihood” label. A more accurate statement is that, *given fixed \(\{\mathbf{w}_{\theta,i}\}\)*, the density of the finite mixture is exact and cheap to compute, but the learned marginal over \(\mathbf{w}\) is only approximated. This conceptual imprecision weakens the central selling point.

2. **Lack of quantitative likelihood or KL metrics against baselines on key experiments.**  
   While many plots are qualitative, quantitative improvements are seldom documented rigorously:  
   - For forward-KL synthetic experiments (Section 4.1), Figure 6 and Figure 13 primarily show qualitative density plots. Figure 7 shows test log-likelihood trajectories during training, but the axis scale and final values are not discussed numerically, and no table (e.g., final log-likelihoods with standard deviations across seeds) is provided.  
   - For the manifold toy in Figure 4, the comparison with FFF, FM, and NF is purely visual. Quantitative metrics such as Wasserstein distance, log-likelihood, or reconstruction error on held-out samples are missing.  
   - For SBI (Section 4.2), all C2ST results are relegated to Appendix Figure 14, with no numerical summary in the main text. Moreover, the baselines are imported from Draxler et al. (2024), and training details/compute budgets are only sketched.  
   In aggregate, this makes it difficult to assess how robustly the proposed model outperforms or matches baselines, and weakens claims such as “state-of-the-art results” and “orders of magnitude faster *and* as accurate”.

3. **Comparison set and baselines are limited or not always well controlled.**  
   The experimental comparisons are almost exclusively against three specific baselines: Normalizing Flows (with relatively strong ResNet or spline couplings in some experiments), Flow Matching (as in Lipman et al. 2023), and Free-form Flows (Draxler et al. 2024). However:  
   - Other mixture-based or kernel-based flexible density estimators (e.g., modern neural mixture models, kernel density estimators with learned kernels, or hybrid flows with mixture outputs) are not considered, despite being conceptually close. The simplest and strongest baseline here would be a *deterministically parameterized* large GMM where the mixture components are outputs of a neural net (i.e., conditional mixture models) and not resampled per evaluation, to test whether the marginalization aspect truly offers a benefit beyond mere large mixtures.  
   - For the manifold toy in Figure 4, it is unclear whether FFF and NF are given architectures tailored to match the intrinsic dimension or whether they are heavily constrained by their bijectivity. The description “Free-form Flow learns an incorrect manifold and is not able to embed the density in 2D space” is plausible visually, but we do not know if hyperparameter tuning or architectural variations would correct this.  
   - The reverse-KL setup (Figure 8) compares only with NF. Methods specifically designed for score-based or solver-free reverse KL / density-ratio estimation (e.g., recent score-based one-step estimators) are absent, despite being natural competitors in terms of “efficient reverse-KL training”.

4. **Ambiguity about the true modeling capacity vs a standard neural mixture, and lack of formal expressivity analysis.**  
   The motivation section argues that directly optimizing a finite set of mixtures \(\{\mathbf{w}_i\}_{i=1}^{N_c}\) leads to a GMM whose expressiveness is limited by \(N_c\), whereas marginalizing over \(\mathbf{w}\sim q_\theta(\mathbf{w})\) gives a more powerful model. Figure 1 supports this visually: at fixed nominal \(N_c\), the “Optimization / GMM” panel is lumpy while the “Marginalization” panel is smoother. However, the paper never formalizes how the marginalized mixture class compares in capacity to a large deterministic mixture with learned locations and possibly learned weights. In practice, Eq. (2) is still a finite mixture with uniformly weighted components, just with stochastic re-sampling of locations at every evaluation. Without either (i) a theoretical result about approximation properties or (ii) a strong empirical comparison to well-optimized deterministic mixtures under matched parameter budget, it remains unclear whether this stochasticity is intrinsically beneficial or just a convenient regularizer.  

   Additionally, Eq. (15) clarifies that the true model class is the integral \(\int q(\mathbf{x}\mid f_\theta(\mathbf{z}))p_{\text{base}}(\mathbf{z})d\mathbf{z}\). This is essentially a conditional expectation model, akin to certain kernel mixture models. Explicitly connecting to that literature and clarifying where the advantage lies would strengthen the contribution.

5. **Theoretical results are somewhat narrow and not clearly tied back to practical training guidelines.**  
   The bias/variance analysis in Section A.5 and A.6, culminating in Theorem A.3 with constants involving \(\gamma=((1+\sigma^{-2})^2/(1+2\sigma^{-2}))^{m/2}-1\), is derived under specific Gaussian assumptions and a simplified manifold model. While it supports the non-exploding nature of the nested Monte Carlo error in high ambient dimension, the paper does not clearly translate these results into concrete recommendations:  
   - How should practitioners choose \(N_c\) as a function of data size \(N\) or intrinsic dimension \(m\)?  
   - Are the analyzed assumptions (e.g., linear manifold, Gaussian kernels) representative of the much more complex experimental settings (SBI, JAFFE images)?  
   - Figure 17 empirically confirms independence from dimension \(d\), but this is shown on a very stylized Gaussian toy (Eq. (28)–(30)) that is far from, e.g., a latent VAE manifold.  

   This disconnect makes the theoretical analysis feel somewhat decorative rather than central to understanding or using Marginal Flow.

6. **Claims regarding manifold learning and disentanglement in image experiments are anecdotal and under-quantified.**  
   Section 4.4 reports visually plausible manifolds on MNIST and JAFFE. For example, Figure 10 displays rows of digits along a 1D learned latent coordinate, and the authors claim to observe “disentanglement of digits and writing style”. Figure 11 likewise shows faces arranged by emotion; the text states “we observe disentanglement of faces and emotions”. However, these are subjective visual impressions with no quantitative manifold quality metrics (e.g., reconstruction fidelity as a function of manifold dimension, geodesic consistency, or disentanglement scores). Moreover, the manifolds are learned in VAE latent space, and the influence of the VAE prior and architecture is not disentangled from the contribution of Marginal Flow itself. Without more rigorous evaluation, these results demonstrate feasibility rather than strong evidence of manifold learning advantages.

7. **Some experimental details and choices are under-specified or may bias comparisons.**  
   - Table 1 is presented as a central summary of model properties, but several entries are subjective or at least debatable. For example, “Efficient training” for VAEs is marked ✓ while NFs are X, but FFF and EB are annotated with “(✓)” without clarifying the threshold for efficiency. Similarly, NFs are clearly not always inefficient to train in practice, depending on architecture; a more systematic metric would help.  
   - For the runtime experiment (Figure 3), the models are roughly matched at ~100k parameters, but the NF architecture is chosen as 3 coupling layers with splines in Section A.3, then in A.4 for synthetic experiments they use 5 invertible ResNet layers “more expressive (but more computationally expensive) than coupling layers with splines”. The inconsistency in NF architecture across sections makes it hard to compare runtime and accuracy fairly.  
   - In reverse-KL experiments (Section A.4), they use simulated annealing of a temperature \(T_i\) (Eq. (12)), which is a nontrivial engineering detail that could heavily affect optimization. The effect of this annealing, and whether the same trick is applied to NF baselines, is not clearly documented in the main text.

8. **Related work section omits several directly related recent works on flexible density regression and efficient density estimation.**  
   The Related Work (Section 3) focuses on EB, Diffusion, VAE, GAN, NF, Flow Matching, and FFF, but omits a number of density estimation approaches that are conceptually close: mixture-based neural densities, kernel-based flexible estimators, and modern solver-free reverse-KL/density-ratio methods. Some of these (see “Potentially Missing Related Work” below) target exactly the trade-offs of efficient density evaluation and sampling, and some develop geometric or two-step density modeling frameworks somewhat analogous to the proposed marginalization. This weakens the positioning and risks overstating the gap in the literature.

9. **Small but noticeable clarity issues and a minor typo in the main text math.**  
   - On Page 14, an equation labeled as (7) reads  
     \(\mathcal{L}(\theta)=\mathcal{L}(\theta^{\prime})=\mathcal{L}(\theta^{\prime})+\mathcal{L}(\theta^{\prime})+\mathcal{L}(\theta^{\prime\prime}).\)  
     This looks like a clear typo or leftover and is mathematically nonsensical. It slightly undermines trust in the carefulness of exposition, although it does not affect the core method.  
   - Some notation choices are slightly confusing. For instance, both \(N_c\) (number of components) and \(N_e\) (SBI experiments) appear, but the relationship between them and batch sizes is not clearly connected to the bias/variance analysis in Appendix A.5. Also, the term “exact density evaluation” is used loosely as noted above.

---

## Potentially Missing Related Work

1. **Dasgupta, Pati, Srivastava, “A Two-Step Geometric Framework For Density Modeling”, 2017.**  
   - Relevance: Proposes a two-step density modeling framework where a base density is estimated and then geometrically transformed, conceptually similar to marginalization / geometric views the authors adopt (especially in Section A.6 on manifold structure).  
   - Where to cite/discuss: Section 3 (Related Work) and possibly Section 2.3 (Flexibility & manifolds). It would be useful to clarify how Marginal Flow’s marginalization differs from or generalizes such geometric two-step frameworks, especially regarding manifold support and computational efficiency.

2. **Yuan, Jarvis, Wang, “A flexible method for estimating luminosity functions via Kernel Density Estimation”, 2020; and Yuan, Li, Wang, “A flexible method for estimating luminosity functions via Kernel Density Estimation – III. Extending to Multiple Flux-Limited Samples”, 2026.**  
   - Relevance: These works use kernel density estimation as a flexible probability modeling tool, trading off sample-efficiency and computational cost. The proposed Marginal Flow with Gaussian kernels and marginalization (Eq. (1), Eq. (15)) is conceptually related to adaptive kernel density estimators with learned kernel locations.  
   - Where to cite/discuss: After Eq. (1) or in Section 2.1 (Model definition) when discussing universality of kernels (Micchelli et al., 2006), and in Section 3 when describing alternatives like kernel-based density models.

3. **Arpogaus, Kneib, Nagler, “Hybrid Bernstein Normalizing Flows for Flexible Multivariate Density Regression with Interpretable Marginals”, 2025.**  
   - Relevance: Introduces hybrid models combining interpretable marginal distributions with flows for flexible multivariate density regression. Similar high-level goal of balancing interpretability, flexibility, and efficient density evaluation.  
   - Where to cite/discuss: Section 3 (Related Work), possibly in comparison to NF-based density models with rich marginals, highlighting how Marginal Flow differs by marginalizing parameters instead of using bijections.

4. **Chen, Li, Li, “One-Step Score-Based Density Ratio Estimation: Solver-Free with Analytic Frames”, 2025.**  
   - Relevance: Provides a solver-free, one-step method for score-based density ratio estimation, directly addressing computational efficiency of density/ration estimation that could be used for reverse KL-type objectives.  
   - Where to cite/discuss: Section 4.1 “Reverse KL divergence training” and Appendix A.2 “Reverse KL” as a baseline or alternative for reverse-KL objectives that emphasize efficiency, and in Section 3 when discussing score-based and flow-based approaches.

5. **Vetter, Gloeckler, Gedon, “Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models”, 2025.**  
   - Relevance: Proposes NPE-PFN for simulation-efficient Bayesian inference, a direct competitor in the simulation-based inference setting targeted in Section 4.2. The benchmarked SBI baselines in Figure 14 are limited to FFF, FM, NSF, but not to modern PFN-based methods.  
   - Where to cite/discuss: Section 4.2 (SBI) and Section 3. A short discussion comparing Marginal Flow’s training cost and sample efficiency to training-free or few-shot methods like PFN-based SBI would help contextualize the practical impact.

---

## Questions

1. **Clarification on the “exact density” claim and the role of \(N_c\).**  
   - For a fixed \(\theta\), is the *true* model you consider in theory the integral in Eq. (1)/(15), or the empirical mixture with finite \(N_c\) in Eq. (2)?  
   - When reporting log-likelihoods (e.g., Figure 7), are you averaging over multiple resamplings of \(\{\mathbf{w}_i\}\) for each test point, or using a single random realization?  
   - Could you provide a brief discussion in the main paper (not just Appendix A.5) of the magnitude of the estimator bias for the values of \(N_c\) you use in practice (e.g., 128–256)? A small ablation vs \(N_c\) on one synthetic task would help.

2. **Comparison to deterministic neural mixture baselines.**  
   - Can you include an experiment where you compare Marginal Flow to a strong deterministic neural mixture model of similar parameter count, where \(\{\mathbf{w}_i\}\) are *directly* learned parameters or outputs of a network but not resampled per evaluation?  
   - Such a baseline on at least the 2D synthetic densities and the manifold toy (Figure 4) would directly test whether marginalization over \(\mathbf{w}\) is truly essential, or if you effectively outperform simple large mixtures mainly due to architectural/training choices.

3. **Quantitative metrics for manifold/image experiments.**  
   - Could you add quantitative evaluations for the manifold-learning experiments (Figure 4, 10, 11, 15–16)? Examples:  
     - For the 1D manifold toy, measure log-likelihood on held-out samples and the distance between learned and ground-truth manifold.  
     - For MNIST and JAFFE, report reconstruction error distributions when restricting to the learned 1D/2D manifolds vs the full VAE latent.  
   - This would significantly strengthen the claims about discovering manifolds and disentangling style/emotion.

4. **Sensitivity to base dimension \(m\) and component count \(N_c\).**  
   - How sensitive is performance (both training stability and final likelihood/KL) to the choice of intrinsic dimension \(m\) of the base distribution and to \(N_c\)?  
   - Are there scenarios where underestimating \(m\) severely harms performance or leads to spurious manifolds? Some empirical plots showing performance vs \(m\) and \(N_c\) would be very informative, especially given the theoretical results depending on \(m\).

5. **Reverse-KL training and annealing details.**  
   - In Section A.4, you anneal a temperature \(T_i\) in Eq. (12) to aid reverse-KL training. Were the same annealing and tuning techniques applied to the NF baselines for fairness?  
   - Could you clarify whether the improved test reverse KL in Figure 8 (left) persists when both methods are carefully tuned for reverse-KL, and whether the difference is statistically significant?

6. **SBI benchmark coverage.**  
   - Figure 14 suggests strong SBI performance, but the main text glosses over which tasks are most improved and by how much. Could you summarize in the main paper the average rank or mean C2ST difference vs baselines across all tasks, and comment on compute/time differences?  
   - Also, can you clarify why PFN-based SBI methods (if known to you) were not included as baselines, or argue why Marginal Flow targets a different trade-off?

---

## Flag For Ethics Review

No ethics review needed.  

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

2: fair.  
The method is reasonably well specified and appears empirically effective, and the Monte Carlo bias/variance analysis is technically sound under its assumptions. However, the key marketing claim of “exact density evaluation” is conceptually overstated relative to the finite-\(N_c\) implementation, some experimental comparisons and baselines are incomplete, and important questions about the benefit over strong deterministic mixtures remain open.

---

## Presentation Rating

3: good.  
The paper is generally clear and well organized, with helpful figures (especially Figures 1, 2, 3, 4, 5, 9–11) and explicit equations for objectives (Eqs. (5)–(11), (15)–(23)). There are a few clarity issues and at least one clear typo (the strange Eq. (7) on Page 14), and the Related Work is somewhat incomplete, but overall readability is good.

---

## Contribution Rating

2: fair.  
Marginal Flow offers a clean and flexible instantiation of a neural mixture model with a focus on efficiency and manifold support, and the empirical exploration across several domains is interesting. That said, the conceptual novelty relative to existing mixture/kernel and latent-variable density models is moderate, positioning vs closely related recent work is incomplete, and the core advantages over simpler baselines are not yet fully substantiated quantitatively.

---

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper presents a solid and practically appealing framework, with strong runtime characteristics (Figure 3) and convincing qualitative behavior (Figures 4, 5, 9–11). However, the central “exact likelihood” narrative is somewhat misleading in the presence of finite-sample Monte Carlo approximation, quantitative evidence vs strong and appropriate baselines is limited, and the benefit over well-tuned deterministic mixtures is not convincingly established. With additional experiments addressing these points and clearer positioning relative to related density estimators, this could become a strong contribution; in its current form it falls slightly short of ICLR’s bar.

---

## Reviewer Confidence

4: confident.  
I am reasonably familiar with generative modeling, normalizing flows, and mixture-based density estimators, and I have carefully checked the main equations and appendices. While there may be related work I have not seen, my assessment of the technical correctness and experimental sufficiency is unlikely to change dramatically, though a strong rebuttal with additional quantitative results could shift my stance toward acceptance.