=== CALIBRATION EXAMPLE 34 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately captures the paper's central thesis—that eigenvector overlaps between train and test Hessians are the "missing ingredient" beyond spectral analyses. The abstract's claims are generally supported by the results, though calling the fluctuation law "universal" deserves scrutiny (it holds strictly only in the local quadratic regime, and asymptotically in proportional asymptotics for ridge regression). The claim that the framework provides "practical tools for analyzing generalization in modern neural networks" is partially fulfilled: the algorithms scale, but the quantitative predictive power in deep networks is only demonstrated observationally.

---

### Introduction & Motivation

This is the paper's strongest section. The gap between prior work (which equates spectra with loss geometry) and the actual two-operator structure of train-test loss is clearly identified and well-motivated. The prior literature survey is thorough and the positioning against TIC, SAM-family methods, and covariate shift literature is accurate. The contributions list (Section 2) is specific and maps cleanly onto the results.

One concern: the claim that the paper "corrects interpretations that implicitly attribute sample-wise multiple descent to spectrum ill-conditioning" (Chen & Mei, 2022; Mel & Ganguli, 2021) is overstated. Those prior works accurately predict the descent peaks using spectral data. They don't claim spectra *exclusively* explain the phenomenon—they provide spectral predictions that happen to work. The overlap framework provides a finer decomposition, but calling prior work incorrect risks mischaracterizing their claims.

---

### Theoretical Foundations (Section 3.1 & Appendix B)

**Theorem 1 (Fluctuation Law):** The result is correct but mathematically elementary. It amounts to writing E[tr(H_test C_train)] in eigenbasis form—the proof in Appendix B.2 is two equations long (equations 21–25). The genuine contribution is conceptual: insisting that this quantity cannot be simplified to spectral terms alone without the overlap kernel O(λ₁, λ₂). The "universality" claim (Section 3.1.1) is appropriate only within the local quadratic approximation; there is no control on when this approximation is valid in deep networks.

**Theorem 2 (Transfer Law):** This is technically non-trivial. The proof (Appendix B.3) uses operator-valued free probability and Proposition 1's linearization argument. The argument appears correct. However, the freeness assumption—"X is free from A, B"—holds asymptotically for rotationally invariant random matrix ensembles (e.g., Gaussian), not generically. When Theorem 2 is invoked for the neural network experiments (implicitly, via the KPM-based estimators), the freeness assumption is unjustified and not acknowledged. The paper should explicitly flag this gap.

**Noisy gradient descent (Appendix B.2.2):** The extension to the SDE steady-state covariance is clean and non-trivial. This strengthens the case that the framework applies to gradient-based optimization.

---

### Ridge Regression Analysis (Section 3.2 & Appendix C)

This is the paper's most technically rigorous section. Under Gaussian inputs and proportional asymptotics, the free-probability derivation of the exact asymptotic overlap function via Propositions 2–3 is well-executed, and the reduction to published formulas (Appendix C.4) confirms consistency.

**Covariate shift (Section 3.2.1 & Figure 1):** The isospectral rotation experiment is an elegant demonstration. Fixing spectra while rotating eigenspaces isolates the pure overlap effect on test error. Theory and simulation agree well (Fig. 1(c)), providing compelling support for the central claim.

**Multiple descent (Section 3.2.2 & Figures 2–3):** The geometric account is clear and insightful. Figure 3's overlap map showing block-diagonality, with error peaks corresponding to near-null training directions overlapping the sharp test subspace, is convincing. The separated-scales limit analysis in Appendix C.5.1 gives precise conditions under which peaks become singularities. Agreement between theory and simulation (d=5000) is very good.

A concern: the analysis assumes Σ_train, Σ_test have a two-level (or k-level) discrete spectrum—an idealized "two-scale" covariance (Eq. 12). Real data covariances have continuous spectra. The paper validates the predictions only in this discrete setting. How the qualitative multiple-descent story generalizes to more complex covariance structure is not addressed.

Additionally, Figure 2 presents the _total_ test loss L_test, not purely the fluctuation ΔL. The paper correctly decomposes L = L₀ + ΔL, but the peaks in total error arise from both components (see Fig. 2(a)); the claim that they are "explained by overlaps" applies specifically to the fluctuation term ΔL, not necessarily to L₀. The paper acknowledges this implicitly but could be clearer.

---

### MLP Validation (Section 3.3 & Appendix E)

The validation experiment is thoughtful but limited. The networks used—layer widths (5,5,5,1)—are very small and toy. The experimental protocol (train to near convergence, then add noise and retrain) is carefully designed to test the local theory, but the controlled nature raises the question of whether the local quadratic approximation is being put to a meaningful test. For small networks with MSE loss and ℓ₂ regularization (λ=1, which is aggressive), the loss landscape is well-conditioned and the quadratic approximation is likely quite good by construction.

**Figure 4(a,b):** The predicted vs. measured ΔL/L₀ shows the theory works well for small noise but exhibits discrepancy at larger noise, which is expected and acknowledged. The log-scale agreement over several orders of magnitude is genuinely impressive.

**Figure 4(c) (inverse Hessian filtering):** The claim that "large displacements do not translate into large test error since the train and test Hessians are well aligned" is stated based on a qualitative observation (Fig. 6). But the paper does not quantify the alignment or show that it *predicts* the observed error. This is a missed opportunity to quantitatively validate the fluctuation law in the neural network setting.

**Missing:** There is no ablation on network depth, width, nonlinearity, or noise type. It is unclear whether the local theory would hold for ReLU networks, deeper architectures, or classification settings—exactly the settings of practical interest.

---

### Scalable Algorithms and ResNet-20 (Section 3.4 & Appendix F)

**Algorithm (KPM/subspace iteration):** The generalization of KPM from spectral density estimation to overlap functional estimation (Eq. 16) is a natural and useful extension. Using the trace identity tr[G_A^{1/2} G_B G_A^{1/2}] = E_v ‖G_B^{1/2} G_A^{1/2} v‖² to enforce positivity is a nice algorithmic insight.

However, the approximation error introduced by (a) the Chebyshev truncation to degree K, (b) the Gaussian smoothing bandwidth σ, and (c) the stochastic trace estimator variance is not formally analyzed. The paper says "kernel width and approximation degree K are chosen so that the truncated series sufficiently dampens the large-multiplicity near-0 eigenspaces" (Section 3.4), but there is no convergence theorem or error bound. For a practical tool that is intended to be applied to large neural networks, this is a significant gap.

**ResNet-20 experiment (Figure 5):** This experiment demonstrates that the algorithm scales, which is valuable. However, the experiment is purely qualitative: it shows that train–test Hessian overlap changes under class imbalance (Fig. 5(b)) vs. balanced test set (Fig. 5(a)). The paper does not:
- Quantitatively predict what *performance degradation* should result from the observed misalignment using the fluctuation law.
- Compare this prediction to actually observed performance under class imbalance.
- Show that the overlap metric is more predictive than alternative diagnostic quantities (e.g., spectral norms of train/test Hessians, gradient similarity).

The experiment thus demonstrates the existence of misalignment but does not validate the paper's core claim that overlaps *predict* generalization degradation.

---

### Proof Completeness and Rigor

**Proposition 1 (Appendix B.3):** The linearization and operator-valued free probability argument is technically sophisticated. The key step—that g_Bhat(M) factors through B alone—uses the additive subordination law for freely independent pairs. This step requires X to be free from (A, B), which holds asymptotically for Gaussian X but not in general. The paper correctly states this as an assumption but should be clearer about its role in the downstream neural network applications.

**Propositions 2 and 3 (Appendix D):** The derivation via operator-valued free probability and linearization is detailed and self-contained. The reduction of the self-consistent equations to scalar form is correctly executed.

---

### Missing Ablations and Experiments

1. **No quantitative test of the fluctuation law in neural networks.** The paper validates the law in tiny MLPs but never tests whether the predicted ΔL from the KPM-estimated overlap functionals matches actual measured ΔL in the ResNet-20.

2. **No comparison to spectrum-only predictions.** For completeness, it would be informative to show how much a spectrum-only approximation (e.g., O(λ₁, λ₂) = constant) misestimates ΔL in the ridge regression or MLP experiments.

3. **Sensitivity to the overlap estimation parameters** (Chebyshev degree K, bandwidth σ) is not explored.

4. **No limitations section.** The paper ends with a brief Discussion (Section 4) but does not acknowledge key limitations: Gaussian input assumption, local quadratic regime, small MLP size, absence of quantitative deep-net predictions.

---

### Writing & Clarity

The paper is well-written and the conceptual contributions are presented with clarity. The proof sketches in the main body are appropriately brief, with details in the appendix. One structural issue: contributions are listed in Section 2 (not the Introduction), which feels slightly non-standard for ICLR. Some figures (e.g., Figs. 2–3) are dense and would benefit from clearer captions distinguishing which curves/markers correspond to which conditions.

---

### Overall Assessment

This paper makes a genuine conceptual contribution by establishing that local loss geometry is intrinsically bivariate, and that eigenvector overlaps between train and test Hessians are the decisive quantity linking optimization variance to test error. The theoretical core—the fluctuation law, the transfer law, and their application to ridge regression—is technically sound and the ridge-regression analysis of covariate shift and multiple descent is the clearest, most compelling part of the paper. The main weaknesses are: (1) the empirical validation in deep networks is limited and exclusively qualitative—the ResNet-20 experiment shows overlaps change under class imbalance but never quantitatively tests whether the overlap-based predictions match observed performance; (2) the key Gaussian/freeness assumption underlying the transfer law is invoked without justification in the neural network context; (3) the KPM estimator lacks formal error bounds; and (4) there is no explicit limitations section. Despite these concerns, the paper's central thesis is well-supported in the linear setting, and the framework provides a useful new lens for analyzing generalization. For ICLR, the theoretical depth and novelty of the ridge regression analysis are likely sufficient, but the paper would be significantly strengthened by a quantitative end-to-end test of the fluctuation law in at least one realistic deep network setting.

# Neutral Reviewer
## Balanced Review

### Summary
The paper establishes a two-loss geometric framework demonstrating that local generalization error is governed not merely by the individual spectra of training and test Hessians, but by the eigenvector overlaps between their respective eigenspaces. It derives a universal local fluctuation law and an operator-valued free probability transfer law, using these to analytically resolve covariate shift and multiple descent in high-dimensional ridge regression. Finally, it introduces scalable matrix-polynomial estimators for overlap functions, validating the theory on small MLPs and diagnosing train-test misalignment under class imbalance in a ResNet-20.

### Strengths
1. **Clear conceptual and theoretical advance:** The paper rigorously corrects the widespread oversimplification that equates loss geometry with Hessian spectra. Theorem 1 cleanly formalizes how the expected test loss increment decomposes into a bivariate integral over spectral measures weighted by an overlap kernel $O(\lambda_{\text{test}}, \lambda_{\text{train}})$, making the interaction between train and test geometries explicit.
2. **Unified explanation of high-dimensional phenomena:** By deriving explicit asymptotic formulas for the overlap function in anisotropic ridge regression (Eq. 11/13), the authors provide a parsimonious geometric resolution to multiple descent and covariate shift. Figures 1–3 convincingly show that error peaks arise from eigenspace misalignment rather than spectrum ill-conditioning alone, correcting prevalent misinterpretations in the literature.
3. **Practical, scalable algorithmic contribution:** The Overlap-KPM method (Appendix F) elegantly adapts the Kernel Polynomial Method and Hutchinson trace estimation to compute overlap functionals in $O(PK^2 md)$ time. The ResNet-20 experiment (Section 3.4) demonstrates feasibility on commodity hardware and yields highly interpretable structural insights about how class imbalance reshapes two-loss geometry via induced misalignment (Fig. 5, 10).
4. **Controlled empirical validation:** The MLP experiments (Section 3.3) quantitatively verify the local quadratic predictions for displacement covariance and test loss increments across multiple noise scales (Fig. 4a,b). The clear visualization of inverse-Hessian filtering and the strong train-test diagonal alignment (Fig. 6) provide compelling evidence that the quadratic local model accurately captures perturbation dynamics in realistic, non-convex settings.

### Weaknesses
1. **Limited scale and quantitative generalization correlation:** The quantitative validation of the fluctuation law is restricted to very small MLPs (width 5, 4 layers), while the ResNet-20 analysis is primarily descriptive. The paper does not report a systematic, quantitative correlation between measured overlap metrics and actual test gaps across diverse datasets or architectures, which weakens claims about broad diagnostic utility.
2. **Reliance on asymptotic free-probability assumptions:** The theoretical machinery assumes proportional limits ($n,m\to\infty$) and asymptotic freeness, which are idealizations. Real-world datasets, CNN weight-sharing, and transformer architectures often violate these independence assumptions. The robustness of overlap interpretations to finite-size deviations or architectural inductive biases is not empirically explored.
3. **Constant-factor computational cost for modern scale:** While the algorithmic complexity is linear in parameters and examples, the prefactor involves $K^2$ Hessian-vector products per probe. Computing exact Hessian-vector products on modern vision or language models remains prohibitively expensive. The paper notes a "few hours" runtime for ResNet-20 but lacks scaling curves or approximations (e.g., gradient-covariance proxies) needed for immediate adoption at foundation-model scale.
4. **Diagnostic rather than prescriptive:** The framework excels at explaining *why* domain shifts or imbalance hurt performance, but the Discussion relegates "alignment-aware optimization" to future work. Without a proof-of-concept ablation showing that overlap-informed data selection, reweighting, or regularization improves optimization, the practical impact remains primarily analytical rather than algorithmic.

### Novelty & Significance
The paper is highly novel in its explicit treatment of eigenvector overlaps as the fundamental coupling mechanism between train and test geometries, moving decisively beyond the well-trodden literature on sharpness, spectral densities, and edge-of-stability dynamics. By unifying covariate shift, domain generalization, and multiple descent under a single overlap-centered framework rooted in free probability, it corrects spectrum-centric narratives and introduces a mathematically rigorous diagnostic tool. This aligns strongly with ICLR's emphasis on theoretically grounded, scalable insights into deep learning. Its significance lies in providing both exact analytical machinery for linear models and computable estimators for neural networks, with clear potential to inspire overlap-aware training objectives, data curation strategies, and sharper generalization bounds.

### Suggestions for Improvement
1. **Expand empirical scope to demonstrate predictive correlation:** Validate the overlap-fluctuation relationship on medium-scale architectures (e.g., ResNet-50 or ViT-S) and multiple benchmarks (e.g., DomainBed splits or CIFAR-10-C). Report quantitative correlations between overlap dispersion/misalignment metrics and held-out test errors across seeds to strengthen practical relevance.
2. **Discuss and evaluate computational proxies:** Since exact Hessian-vector products are expensive, analyze how well overlaps estimated from empirical gradient covariances (e.g., using batch gradients or K-FAC) approximate true Hessian overlaps. This would bridge the theory-practice gap for modern large-scale training.
3. **Include a prescriptive intervention ablation:** Transition beyond diagnostics by testing a simple overlap-aware intervention, such as a regularization term penalizing train-test misalignment, or a data-subselection strategy that minimizes $O(\lambda_{\text{test}}, \lambda_{\text{train}})$ on harmful subspaces. Even a toy ablation would substantially increase impact and appeal to the ICLR community.
4. **Clarify finite-size and architectural limitations:** Add a brief finite-size scaling analysis or simulation showing how quickly empirical overlaps converge to the free-probability predictions as $d, m$ increase. Explicitly discuss architectural factors (e.g., convolutional locality, normalization layers) that may induce non-free dependencies and suggest how the theory might be adapted or bounded in those regimes.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Causal Intervention in Neural Networks:** Add an experiment manipulating eigenvector alignment in MLPs/ResNets while holding spectra fixed (analogous to Fig 1 ridge regression). Without this, the claim that overlaps *cause* generalization differences in deep nets remains merely correlational.
2. **Predictive Power vs. Sharpness Baselines:** Compare overlap metrics against standard sharpness measures (SAM, trace, entropy) for predicting generalization gaps across hyperparameters. Without outperforming these baselines, the practical utility of the complex overlap estimation is unproven.
3. **Multiple Descent Validation in NNs:** Demonstrate multiple descent curves in the MLP/ResNet experiments correlated with overlap metrics. The paper claims to unify multiple descent under overlap theory, but only shows this for ridge regression.

### Deeper Analysis Needed (top 3-5 only)
1. **Finite-Width Freeness Error:** Quantify the deviation from the free probability assumptions in finite-width networks. The transfer law relies on asymptotic freeness; without bounding this error, the theory's applicability to practical network sizes is unclear.
2. **Spectral vs. Overlap Contribution:** Decompose the test loss increment in the class imbalance experiment into spectral and overlap components. Fig 5 shows both change; without isolating the overlap contribution, the claim that misalignment drives the effect is unsupported.
3. **Quadratic Approximation Validity:** Measure the error of the quadratic approximation vs. true loss displacement in the ResNet experiments. The theory hinges on local quadratic behavior, which may not hold for large steps or highly non-convex regions.

### Visualizations & Case Studies
1. **Layer-wise Overlap Distribution:** Visualize overlap heatmaps per layer for the ResNet experiment. Global averages (Fig 5) hide whether misalignment is driven by the classifier head or feature extractor, which dictates actionable insights.
2. **Training Dynamics of Overlaps:** Plot overlap metrics throughout training epochs. If overlaps are fundamental, their evolution should correlate with generalization gap reduction, not just final state.
3. **Overlap vs. Shift Magnitude:** Plot overlap metrics against varying degrees of domain shift (e.g., corruption severity). This would validate overlaps as a quantitative measure of shift severity as claimed.

### Obvious Next Steps
1. **Overlap-Aware Optimization:** Propose and test a regularizer that explicitly minimizes train-test eigenvector misalignment. The paper suggests this in Discussion but lacks even a prototype experiment to prove feasibility.
2. **Standard Domain Generalization Benchmarks:** Evaluate overlap metrics on DomainBed suites rather than just CIFAR class imbalance. Class imbalance is a specific shift; broader validation is needed for the "covariate shift" claim.
3. **Connection to Existing Bounds:** Derive a formal generalization bound using the overlap fluctuation law. Currently, the law is a local fluctuation result; linking it to global generalization bounds would strengthen the theoretical contribution.

# Final Consolidated Review
## Summary

This paper establishes a two-loss geometric framework for understanding generalization, demonstrating that local loss geometry is fundamentally bivariate: test error under training perturbations depends not only on train and test Hessian spectra but critically on the eigenvector overlaps between their eigenspaces. The authors derive a universal local fluctuation law expressing expected test-loss increments as a bivariate integral over spectral measures weighted by an overlap kernel, prove a transfer law for how overlaps transform under noise using operator-valued free probability, and apply these results to derive exact asymptotic overlap decompositions in ridge regression—resolving multiple descent and quantifying covariate shift. Scalable algorithms based on subspace iteration and kernel polynomial methods enable overlap estimation in large networks, demonstrated on MLPs and a ResNet-20.

## Strengths

- **Conceptual novelty:** The paper rigorously corrects a widespread oversimplification by establishing that local loss geometry is inherently bivariate. Theorem 1 cleanly formalizes how expected test-loss increments decompose into spectral measures weighted by an overlap kernel $O(\lambda_1, \lambda_2)$, making the train-test interaction explicit. This is a genuine theoretical contribution that reconceptualizes the relationship between optimization dynamics and generalization.

- **Unified theoretical explanation:** By deriving exact asymptotic overlap functions for anisotropic ridge regression, the authors provide a parsimonious geometric account of multiple descent and covariate shift. Figures 1–3 demonstrate convincingly that error peaks arise from eigenspace misalignment rather than spectrum ill-conditioning alone, with the overlap map in Figure 3 clearly showing block-diagonal structure corresponding to error spikes.

- **Controlled empirical validation:** The MLP experiments in Figure 4 quantitatively verify local quadratic predictions for displacement covariance and test-loss increments across multiple noise scales, with log-scale agreement over several orders of magnitude. The isospectral covariate shift experiment (Figure 1) elegantly isolates overlap effects by rotating eigenspaces while fixing spectra.

- **Practical algorithmic contribution:** The Overlap-KPM method adapts kernel polynomial methods and Hutchinson trace estimation to compute overlap functionals in $O(PK^2 md)$ time, enabling application to modern networks. The ResNet-20 experiment demonstrates feasibility on commodity hardware and yields interpretable structural insights about class imbalance effects.

## Weaknesses

- **Limited quantitative validation in deep networks:** The ResNet-20 experiment (Figure 5) shows that train-test Hessian alignment changes under class imbalance, but does not quantitatively test whether overlap-based predictions match observed performance degradation. The paper demonstrates the *existence* of misalignment but not its *predictive power* for generalization gaps. A natural validation would compare predicted $\Delta L$ from the fluctuation law (using estimated overlaps) against measured test-loss increments.

- **Restricted scope of neural network experiments:** The quantitative validation of the fluctuation law is limited to small MLPs (layer widths 5,5,5,1) with MSE loss and strong $\ell_2$ regularization ($\lambda=1$), which produce well-conditioned landscapes where quadratic approximations are likely to hold. Whether the framework applies to ReLU networks, deeper architectures, classification settings, or realistic training regimes remains untested.

- **Missing error bounds for overlap estimation:** The KPM algorithm approximates overlap functionals using truncated Chebyshev series with Gaussian smoothing, but no formal convergence theorem or error bound is provided. The paper notes that kernel width and degree $K$ must "sufficiently dampen" near-zero eigenspaces, but this is not quantified. For a practical tool intended for large networks, this gap reduces confidence in the estimates' accuracy.

- **No explicit limitations section:** The paper does not clearly delineate where its theoretical guarantees apply (Gaussian inputs, proportional asymptotics, local quadratic regime) versus where they are heuristic. While the text mentions these assumptions in context, a dedicated limitations discussion would help readers understand applicability boundaries.

## Nice-to-Haves

- **Causal manipulation in neural networks:** An experiment that artificially rotates train/test Hessians while holding spectra fixed (analogous to Figure 1's ridge regression setup) would strengthen causal claims about overlap effects in deep networks.

- **Spectral vs. overlap contribution decomposition:** For the class imbalance experiment, decomposing the test-loss change into spectral and overlap components would isolate whether the observed effect is driven primarily by overlap changes.

- **Training dynamics of overlaps:** Visualizing how overlap metrics evolve through training could reveal whether early-stage alignment predicts final generalization.

- **Comparison to sharpness-based metrics:** Correlating overlap measures against standard sharpness metrics (trace, spectral norm, SAM) would clarify the incremental predictive value of overlaps.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Freeness assumptions invoked without justification for neural networks"** (from harsh critic): This misunderstands the paper. Theorem 2's freeness assumption is used for the ridge regression analysis (where it holds asymptotically for Gaussian designs), not for neural network experiments. Neural network overlaps are estimated directly via KPM without invoking freeness.

- **"Theorem 1 is mathematically elementary"** (from harsh critic): The contribution is conceptual—insisting that the trace formula cannot be simplified to spectral terms alone—not mathematical complexity. This is not a weakness.

- **"Universality claim inappropriate"** (from harsh critic): The paper explicitly states the fluctuation law is within the local quadratic regime. The "universal" framing is appropriate for that regime.

- **"Scope creep demands"** (from spark finder): Demands for overlap-aware optimization algorithms, DomainBed benchmarks, exhaustive architectural ablations, or prescriptive interventions exceed the paper's stated scope, which is establishing foundations and demonstrating feasibility.

- **"Figure 2 presents total loss not fluctuation"** (from harsh critic): The paper correctly decomposes $L_{\text{test}} = L_0 + \Delta L$ and discusses both components; the figure caption and text make this clear.

## Novel Insights

The synthesis of reviews reveals a key insight not explicitly articulated: the paper's two-loss framework naturally suggests a decomposition of generalization error into *routing* and *curvature* components. The overlap kernel $O(\lambda_1, \lambda_2)$ determines how training variance (from low-curvature directions) routes into test-sensitive directions (high-curvature directions). This routing interpretation clarifies why spectrum-only analyses cannot predict generalization under domain shift—even with identical train/test spectra, different eigenspace alignments route variance differently. The framework also suggests that alignment, not just sharpness, is the relevant quantity for understanding why flat minima generalize, potentially bridging sharpness-based and alignment-based perspectives on generalization.

## Suggestions

- Add a quantitative test in the ResNet-20 experiment comparing predicted test-loss increments (from estimated overlaps and the fluctuation law) against measured increments under controlled perturbations.

- Include error analysis for the KPM algorithm—either a formal bound on approximation error in terms of Chebyshev degree $K$ and kernel width $\sigma$, or empirical convergence curves showing stability as these parameters vary.

- Add a brief limitations section explicitly acknowledging: (1) theoretical guarantees require Gaussian inputs or proportional asymptotics; (2) the local quadratic approximation may not hold for large perturbations or highly non-convex regions; (3) neural network validation is preliminary and limited to specific architectures.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 2.0, 4.0]
Average score: 5.6
Binary outcome: Accept
