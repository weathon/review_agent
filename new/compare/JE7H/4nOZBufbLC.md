---
job_id: 251ad09d-fb2b-42d9-ae5a-5fde1ad11479
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 4nOZBufbLC.pdf
paper: Count Bridges Enable Modeling and Deconvolving Transcriptomic Data
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new generative modeling framework (a discrete bridge process) and applies it to deconvolution problems in transcriptomics, which falls squarely under generative models, probabilistic methods, and applications to biology.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method/Approach, Experiments, Results, Conclusion/Limits) are present. The math is nontrivial and carefully developed, experiments are substantial on both synthetic and real biological data, and the exposition is clear enough to understand and scrutinize; there are no obvious fatal methodological or statistical flaws.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No hidden prompts, instructions to the reviewer, or other manipulative content are present in the paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces **Count Bridges**, a stochastic bridge process on integer lattices based on Poisson birth–death dynamics, yielding closed-form forward and bridge kernels (\(K_{t|0}\), \(K_{s|0,t}\)) that satisfy diffusion-style consistency and admit exact sampling. The authors show how to train distributional denoisers using energy scores and extend the framework to **aggregate-only supervision** via a projection-guided EM-like procedure that treats unit-level counts as latent variables. Empirically, Count Bridges outperform continuous and discrete flow matching on synthetic count benchmarks and are applied to two challenging biological deconvolution problems: nucleotide-level single-cell expression modeling and bulk RNA-seq deconvolution, and reference-free spatial transcriptomics deconvolution into single-cell count profiles.

## Strengths

1. **Well-specified discrete bridge with strong mathematical backing.**  
   The core construction in Section 3 derives a Poisson birth–death bridge on \(\mathbb{Z}^d\) with exact conditional distributions. Proposition 3.1 and the accompanying Appendix A (Propositions A.1, A.6, Theorems A.5, A.9) carefully show that:  
   - The forward process \(X_t = X_0 + B_t - D_t\) (Eq. 6) induces binomial–hypergeometric structure for intermediate times (Eq. 9),  
   - A slack variable \(M_t\) has a Bessel-form posterior (Eq. 11), and  
   - The resulting observable bridge satisfies the key consistency identity (Eq. 1).  
   This is not a heuristic “discrete diffusion” but a rigorously specified bridge kernel with explicit sampling procedures (Algorithms 1 and 2).

2. **Conceptual link to Schrödinger bridges and optimal transport on counts.**  
   The paper does a nice job connecting Count Bridges to entropic OT on \(\mathbb{Z}\). The interpretation in Section 3.1 and Appendix A.2, relating the jump intensity \(\kappa = \sqrt{\lambda_+\lambda_-}\) to the entropic regularization strength, and showing that \(\kappa\to 0\) recovers discrete OT with cost \(|x_1 - x_0|\), provides a clean conceptual parallel to Gaussian bridges with variance \(\sigma\). This is an appealing unification for readers familiar with continuous Schrödinger bridges.

3. **Distributional training objective that respects count geometry.**  
   Instead of reverting to token-wise cross-entropy, the paper uses an **energy score** based on \(\rho(x,x') = \|x-x'\|_2^\beta\) (Sec. 3.2), enabling modeling of the joint distribution \(X_0|X_t\) in high dimensions. The loss is strictly proper under standard assumptions and can be estimated from samples of \(q_\theta\). This addresses a real limitation of categorical discrete diffusion/flow methods, which typically either factorize or autoregress over coordinates.

4. **Aggregate-supervision extension is clearly articulated and reasonably principled.**  
   The EM-like deconvolution framework in Section 4 is well developed. The E-step uses a projection-guided reverse bridge sampler (Algorithm 3) to approximate \(Q_\theta(\mathbf{X}_0|A_0,a_0)\); the M-step optimizes an aggregate-level energy score \(S^A_\rho\). Proposition 4.1 grounds the simple scaling projection \(\Pi(\mathbf{x}_0)_g = a_0 x_{g0} / \sum_{g'} x_{g'0}\) as a generalized KL projection that approximates an exponential tilt of the aggregate-conditional law. Appendices B.1–B.3 provide thoughtful discussion of when deconvolution is statistically identifiable and how identifiability degrades with group size.

5. **Synthetic experiments clearly highlight advantages of the method.**  
   - **Figure 2** (discrete 8-Gaussians to 2-Moons) visually compares trajectories for continuous flow matching (CFM), discrete flow matching (DFM), and Count Bridges (CB). CB trajectories follow OT-like paths that are better aligned with the geometry than DFM, which shows grid-like artifacts. Table 6 quantifies this: CB with energy-score training achieves the lowest MMD (0.0044), W2 (0.0052), and Energy (0.0098), substantially outperforming CFM and DFM.  
   - **Figure 3** and Table 9 show scaling in dimensionality for a low-rank Gaussian mixture. CB maintains low W2 and MMD as \(d\) grows up to 512, while both CFM and DFM degrade notably. This directly supports the claim that Count Bridges scale better in high-dimensional integer spaces.

6. **Biological applications are substantive and well-aligned with the method.**  
   - **Sequence-to-expression and bulk RNA-seq deconvolution** (Sec. 6.2): The model is trained on \(10^6\) PBMC cells with nucleotide-level counts. Table 1 shows that Count Bridges significantly improve nucleotide-level MSE vs a fine-tuned Enformer baseline for both bulk (0.601 vs 2.590) and cell-type-specific predictions (1.410 vs 3.142). Table 3 demonstrates that CB-based deconvolution beats CIBERSORTx and MuSiC on all metrics (JSD, RMSE, Spearman) while additionally producing unit-level count profiles.  
   - **Spatial transcriptomic deconvolution** (Sec. 6.3): Using synthetic Visium-like data from MERFISH, CBs outperform STDeconvolve on cell-type proportion recovery (Table 4: JSD 0.231 vs 0.288, RMSE 0.110 vs 0.177) and also deliver better single-cell count reconstructions than a strong “spot mean” baseline (Table 5: e.g., Energy 8.903 vs 41.717). **Figure 7**’s UMAP of true vs CB-deconvolved cells shows substantial mixing, lending qualitative support that deconvolved expression distributions are realistic.

7. **Exposition and mathematical derivations are unusually thorough.**  
   The main text is reasonably self-contained, and Appendix A provides full derivations: binomial and hypergeometric composition lemmas (A.3, A.4), the slack Bessel posterior (Prop. A.6), and bridge consistency (Theorem A.9). The paper also provides detailed pseudocode (Algorithms 1–4) and practical sampling details (e.g., custom CUDA kernel for Bessel sampling). Overall, this is significantly clearer and more rigorous than many discrete generative modeling papers.

## Weaknesses

1. **Distributional training and sampling details could be better justified and analyzed.**  
   While Section 3.2 introduces the energy-score loss, several important design decisions are left somewhat opaque:  
   - The choice \(\beta = 1\) is stated but not justified; different \(\beta\) can materially affect the geometry of the learned conditional. Some sensitivity analysis or discussion would strengthen the case.  
   - The practical implementation uses \(m=2\) samples for the U-statistic estimator (App. E.2.4), which is quite small and may yield high-variance gradient estimates in high dimensions. There is no discussion of variance, bias, or tradeoffs here, nor ablations varying \(m\).  
   - During sampling (Algorithm 2), the model uses a small number of denoising steps (e.g., 3 NFEs in the nucleotide application), which is impressive, but there is no systematic study of how performance degrades as the grid is coarsened or refined. Given that Eq. (3) relies critically on bridge consistency, some empirical check of stability vs NFE would make the practical sampling story more convincing.

2. **Approximate projection for deconvolution is heuristic and lacks quantitative evaluation of its approximation quality.**  
   Proposition 4.1 positions the scaling projection \(\Pi\) as a first-order approximation to the aggregate-conditional distribution in a large-sample regime, but:  
   - The main text concedes in the Limitations that “the projection step we use is a first-order surrogate and lacks serious theoretical support,” and the conditions behind the first-order tilt argument in Appendix B.1 require asymptotic regimes that may not apply to realistic group sizes.  
   - There is no quantitative analysis of how far the projection-guided sampler (Algorithm 3) deviates from the true conditional \(Q_\theta(\mathbf{X}_0|A_0=a_0,\mathbf{X}_1)\) even on small synthetic problems where exact conditioning is tractable. For example, one could compare marginal or aggregate statistics for a toy 1D sum constraint where exact conditional sampling is easy.  
   - In the main biological applications, the projection is upgraded to a learned module \(\Pi_\psi\) (Sec. 6.2, E.2.3), but again there is no direct evaluation of whether this module approximates the true conditional (e.g., log-likelihoods under a reference model, or comparison with MCMC-based conditioning).  
   This matters because deconvolution is the key advertised application; without either tighter theory or diagnostics, it is hard to know how much of the observed performance is due to luck / regularization vs genuinely correct conditioning on aggregates.

3. **Identifiability discussion is insightful but remains largely qualitative and does not tie back tightly to the empirical regimes.**  
   Appendices B.2 and B.3 give a good formal treatment of factorial cumulants and the CLT collapse for large groups, and Section 6.1 (and **Figure 4**) shows that deconvolution errors indeed grow with group size and Dirichlet concentration parameter \(\alpha\). However:  
   - The recoverability condition (Assumption B.1) and Theorem B.10 are stated in fairly abstract terms with factorial cumulant signatures. The paper does not explicitly check whether the synthetic mixture construction or the biological datasets plausibly satisfy the required additive-identifiability or covariate-coverage assumptions.  
   - In the spatial transcriptomics application, spots contain on the order of 10–50 cells, but the degree of between-spot heterogeneity in cell-type composition is not quantified. Connecting these real-world regimes more explicitly to the theory would help interpret how reliable the deconvolution really is.  
   - There is no discussion of how model misspecification (e.g., units not following the assumed independent product prior under \(q_\theta\)) interacts with identifiability; this is particularly relevant given that the CB denoiser is a highly flexible neural network that can encode complex dependencies.

4. **Comparisons to alternative discrete / count generative models are somewhat limited, particularly on the real biological tasks.**  
   Although the synthetic experiments compare CBs to CFM and DFM, on the transcriptomic tasks the baselines are mostly **analytic or regression-style methods** rather than modern generative count models:  
   - For sequence-to-expression, the only baseline is a fine-tuned Enformer model (Table 1). No comparison is made to more recent single-cell sequence-to-expression models such as scooby (Hingerl et al., 2024) or other scRNA generative models that might be adapted to nucleotide resolution.  
   - For bulk RNA-seq and spatial deconvolution, the baselines (CIBERSORTx, MuSiC, STDeconvolve, RCTD) are sensible and widely used, but they all target cell-type proportions, not unit-level count distributions. There is no comparison to, e.g., a strong negative binomial VAE, diffusion, or flow model trained directly on single-cell counts then coupled with a simpler deconvolution approach.  
   - The **Blackout Diffusion** model (Santos et al., 2023) is mentioned conceptually but not used as an empirical baseline; it is not obvious it could be adapted for these tasks, but a short justification or empirical attempt for at least the synthetic benchmarks would be useful.  
   This does not invalidate the results, but it makes it harder to isolate how much of the performance gain is due to the novel bridge vs the use of a high-capacity distributional denoiser and task-specific conditioning.

5. **Some aspects of the loss and architecture could be better ablated, particularly for biological tasks.**  
   The biological experiments use quite sophisticated architectures (Enformer-based sequence encoders, U-ViT backbones, learned projection modules) and custom training tricks (e.g., EMA, cell-type masking, aggregate conditioning probability \(p_{\mathrm{agg}}\)). However:  
   - There are essentially no ablation studies that disentangle the contribution of the Count Bridge *kernel* from the contribution of architectural choices. For instance, what happens if one uses the same denoiser architecture but trains a continuous diffusion / flow model on log-transformed counts with an appropriate likelihood?  
   - The learned projection module \(\Pi_\psi\) in Sec. 6.2 is only described in detail in Appendix E.2.3, and there is no comparison to using the simpler analytic scaling \(\Pi\) or omitting projection during training but enforcing it only at test time.  
   - For spatial deconvolution, the U-ViT counts+image architecture is quite complex. It would be informative to include a simpler CB model without images, and perhaps a baseline that plugs images into a non-bridge model, to see whether the advantage comes from the bridge or from multimodal modeling per se.  

6. **Some mathematical and notation points could be clarified or tightened.**  
   While the math is overall well done, a few places could benefit from more explicit statements:  
   - In Proposition 3.1 (Eq. 8–9), the conditioning structure for the Bessel slack variable \(M_t\) is described as \(M_t|d_t \sim \text{Bes}(|d_t|; \Lambda_+(t), \Lambda_-(t))\), then the text says “we condition on \(d_t\)” and sample \(M_t\mid d_t\). It would be helpful to make the role of the time-rescaling function \(w(t)\) explicit in this main-text statement, not just in App. A.1.3 / Corollary A.7.  
   - Equation (2) (projective posterior) is central but not re-derived in the discrete case; it holds by analogy via Theorem A.9, yet this link is only implicit. A brief explicit remark in Sec. 3.1 that Theorem A.9 establishes both Eq. (1) and the projective posterior structure for the Poisson bridge would close the conceptual loop.  
   - In the Gaussian background (Sec. 2), there is a minor notational inconsistency between \(r(s,t)\) defined in Eq. (5) and the discussion right after Prop. 2.1 when matching to Albergo et al. (2023). This is not fatal but requires a bit of deciphering.

7. **Evaluation metrics and uncertainty quantification on biological tasks are limited.**  
   - For the deconvolution tasks, the paper reports JSD, RMSE, and Spearman correlation on cell-type proportions (Tables 3, 4, 14) and distributional metrics (MMD, W2, Energy) on reconstructed counts (Tables 2, 5). However, there is no per-gene or per-cell breakdown, error bars are aggregated over only 3 inference seeds, and there is no calibration or uncertainty analysis, even though the method is fundamentally probabilistic.  
   - **Figure 5** (UMAP of true vs deconvolved PBMC cells) and **Figure 7** (MERFISH UMAP) are qualitatively encouraging, but they are not accompanied by cluster purity / neighborhood metrics that would make the visual impressions more rigorous. Given that UMAP can be misleading, relying too heavily on such figures as qualitative evidence is risky without additional quantitative checks.

8. **Some related work in spatial deconvolution and single-cell generative modeling is missing or thinly covered.**  
   The related work section focuses on discrete diffusion, Schrödinger bridges, and classical spatial deconvolution. Newer methods for spatial deconvolution using GNNs and modern generative models for transcriptional dynamics (see next section) are not discussed, which weakens the positioning of the work, especially given that deconvolution is a central application.

## Potentially Missing Related Work

1. **Y. Li and Y. Luo, “STdGCN: Spatial Transcriptomic Cell-Type Deconvolution Using Graph Convolutional Networks”, 2024.**  
   - Directly addresses spatial transcriptomic cell-type deconvolution using a graph neural network over spatial spots.  
   - This is highly relevant to Sec. 6.3 and the comparison to STDeconvolve and RCTD. The authors should discuss it in the Spatial transcriptomic deconvolution part of Sec. 5 and compare conceptually how Count Bridges differ from GCN-based approaches (e.g., CBs model unit-level counts and use a generative bridge; STdGCN models proportions with graph regularization). If possible, adding STdGCN as a baseline on their MERFISH benchmark would strengthen the empirical comparison.

2. **C. Mizukoshi, Y. Kojima, S. Nomura, “DeepKINET: A Deep Generative Model for Estimating Single-Cell RNA Splicing and Degradation Rates”, 2024.**  
   - Proposes a deep generative model tailored to single-cell RNA kinetics, emphasizing stochastic modeling of RNA counts.  
   - While focused on kinetics rather than deconvolution, it is conceptually close to the stochastic modeling of transcriptomic count data. It should be cited in the “Sequence-to-expression models” or in a dedicated paragraph in Sec. 5 discussing generative models for single-cell RNA (alongside VAEs, etc.), clarifying how Count Bridges differ in goal and methodology (bridge-based generative modeling + deconvolution vs kinetic parameter estimation).

3. **X. Liao, L. Kang, Y. Peng, “Multivariate Stochastic Modeling for Transcriptional Dynamics with Cell-Specific Latent Time Using SDEvelo”, 2024.**  
   - Uses SDEs to model transcriptional dynamics with cell-specific latent times, yielding a continuous-time stochastic model of gene expression.  
   - This is relevant to the background on stochastic processes for gene expression (Sec. 6.2) and could be discussed alongside scRNA kinetic/dynamical models, highlighting that Count Bridges focus on integer-valued bridges and deconvolution rather than continuous-time latent dynamics. A short comparison in the Related Work section would be appropriate.

## Questions

1. **Energy score and sampling design:**  
   - How sensitive are your synthetic and biological results to the choice of \(\beta\) in the energy score? Have you tried \(\beta = 1/2\) or \(\beta = 2\), and if so, how did performance and training stability change?  
   - For the nucleotide and spatial applications, how does performance vary as you change the number of samples \(m\) used in the U-statistic estimator? Is \(m=2\) close to optimal, or primarily a computational compromise?

2. **Projection-guided sampler quality:**  
   - On a small 1D or low-dimensional synthetic example where exact conditioning on sums is tractable (e.g., via dynamic programming or MCMC), can you compare your projection-guided sampler (Alg. 3) to the exact conditional in terms of KL divergence, marginal distributions, or summary statistics? Even a small experiment would clarify how accurate the approximation is in practice.  
   - In the settings with a learned projector \(\Pi_\psi\), did you observe mode collapse or pathological behaviors (e.g., always assigning the same unit-level distribution within a group)? Any diagnostics on diversity of deconvolved profiles would be useful.

3. **Ablations on biological architectures and projection module:**  
   - For the PBMC experiment, can you provide ablations: (a) CB without aggregate conditioning / projection; (b) CB with only analytic scaling projection (Prop. 4.1) but no learned \(\Pi_\psi\); (c) a non-bridge model (e.g., continuous diffusion on log-counts) with the same denoiser architecture and projection? This would help isolate the contribution of the Count Bridge kernel.  
   - For the MERFISH spatial deconvolution, have you tried removing image inputs and training CBs only on counts plus noise/time? How much does the image modality contribute relative to the bridge itself?

4. **Comparisons to count-specific or discrete generative baselines:**  
   - Is it feasible to adapt Blackout Diffusion (Santos et al., 2023) or a discrete diffusion like Shi et al. (2024) to your synthetic low-rank Gaussian mixture experiments or even to small-gene-count biological subsets? A short empirical comparison would better support the claim that CBs are a more natural choice for integer-valued count data.  
   - For the sequence-to-expression task, have you considered or tried scooby (Hingerl et al., 2024) or other single-cell sequence-based models as baselines?

5. **Interpretation of UMAP visualizations:**  
   - For **Figures 5 and 7**, can you provide quantitative metrics like neighborhood overlap, adjusted Rand index between clustering on true vs deconvolved cells, or some local density-based score to back up the visual mixing claim? UMAP can be quite sensitive to hyperparameters and random seeds.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core bridge construction and proofs look correct and complete, and the synthetic experiments strongly support the main claims. The main caveat is that the projection-guided deconvolution step is heuristic with limited quantitative validation of approximation quality, though it works well empirically.

## Presentation Rating

3: good.  
The paper is dense but generally well written, with clear mathematical derivations and informative figures and tables. Some notation links (e.g., between main text and appendices) and ablation discussions could be clearer, but overall the exposition exceeds the typical standard.

## Contribution Rating

3: good.  
The paper introduces a principled discrete bridge for counts, connects it to Schrödinger bridges, and extends it to aggregate-only supervision with substantial, nontrivial biological applications. The work is not just a small tweak of existing discrete diffusion models, and it offers a useful tool for count data modeling and deconvolution.

## Overall Rating

8: Accept, good paper (poster).  
The paper makes a solid, technically careful contribution at the intersection of discrete generative modeling and biological deconvolution. The Count Bridge construction is well-justified mathematically, the synthetic benchmarks are convincing, and the biological applications are both ambitious and empirically strong. The main reservations concern (i) the heuristic nature and limited quantitative analysis of the projection-guided deconvolution step, and (ii) somewhat limited comparisons to alternative generative count models on real data. Nonetheless, the strengths clearly outweigh these weaknesses, and the work should be of broad interest at ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion/Schrödinger bridges and discrete generative models, and I carefully read the main derivations and experimental sections. Some biological details and specific baseline implementations are less in my core expertise, but this does not materially affect the evaluation.