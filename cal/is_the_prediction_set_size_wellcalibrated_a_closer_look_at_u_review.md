=== CALIBRATION EXAMPLE 47 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is appropriately descriptive, and the abstract correctly previews the three contributions. One concern arises immediately: the abstract frames GPT-2 Small (137M parameters) as evidence of effectiveness "including… GPT-2," implying broad LLM applicability. GPT-2 Small is a decade-old language model far from modern LLM scale; the NLP component of this paper is better described as "a fine-tuned small language model on a niche classification dataset," and the abstract's framing inflates this claim. The abstract also claims to "answer the open question" of whether PSS is well-calibrated, but the answer depends critically on the proposed definition, which itself is a design choice, not a universally agreed-upon standard.

---

### Introduction & Motivation

The motivation is genuine. The gap between coverage guarantees and per-instance reliability of PSS is a real concern, and the paper correctly distinguishes its contribution from conditional coverage (Gibbs et al., 2025) and regression-domain calibration (van der Laan & Alaa, 2024). These distinctions are handled carefully.

However, the introduction introduces "multinomial sampling" as the mechanism for connecting PSS to accuracy, but this is itself a significant design choice that is not motivated here. Why should one sample from the prediction set at all, rather than, say, using set inclusion of the argmax or some other point estimate? The introduction treats this as obvious when it is not. The temperature parameter *t* in multinomial sampling means that the paper is actually defining a *family* of calibration notions rather than a single canonical one, and this tension is underplayed.

The paper claims that "the function between PSS and prediction accuracy is not straightforward, compared to the linear function in traditional confidence-based uncertainty." This characterization of confidence calibration as having a "linear function" is slightly imprecise; it is the *identity* function (P(Ŷ=Y|P̂=p) = p) that defines calibration, which is indeed linear, but this glosses over the difficulty of binning.

---

### Preliminary & Framework (Section 3–4.1)

**The multinomial sampling construction (Definition 4.1) is the paper's foundational choice and deserves sharper scrutiny.** Defining accuracy via E[Acc_t(x) | |S(x)|=k] = f(k) with t as a free temperature is reasonable as a *measurement* device, but it means the calibration definition changes with t. The paper focuses on "Top-1 accuracy" (t→∞) for experiments because it is "most ill-behaved," but this selection is ad hoc. A more principled argument for why Top-1 is the right operating point would strengthen the paper.

**Equation (3)** defines conformal calibration as merely requiring f(k) to be monotonically decreasing, which is an extremely weak requirement—almost any conformalized model will exhibit some degree of monotone decrease in accuracy versus PSS. The interesting question is whether the *specific shape* of the decrease matches a proposed target, which is what the calibration error metrics (Eq. 5) actually measure. The definition as stated is not tight enough to meaningfully distinguish calibrated from uncalibrated models; the real work is done by the choice of target f(k), which is a separate contribution.

**CP-ECE formulations (Eq. 5):** The choice of "Uniform CP-ECE" (equal weight per PSS bin) over the more conventional sample-weighted ECE is presented as being motivated by fairness concerns (citing Mehrabi et al., 2021). While the fairness argument is creative, it is not fully convincing: bins with PSS = 500 (almost no samples) are given the same weight as PSS = 1 (common). This can produce metrics that are dominated by noise from sparsely populated bins, especially on datasets with 1,000 classes like ImageNet where many PSS values will have very few samples.

---

### Calibration Target Function (Section 4.2)

This is the most technically problematic section of the paper.

**The gap in Theorem 4.2:** The theorem shows that under a Dirichlet model for **p** and **q** with the same shape vector **a**, E[**p**·**q**] = Σ_j a_j² = K^{-τ}. This gives a *single scalar* for the expected inner product—it is the expected accuracy integrated over *all* prediction set sizes, not the expected accuracy *conditional on a specific PSS of k*. The theorem does not prove that E[Acc_t(x) | |S(x)| = k] = 1/k^τ as a function of k. The jump from a scalar result (K^{-τ}) to the function 1/k^τ over varying k is never formally justified. The connection to f(k) = 1/k^τ is asserted heuristically ("inspired by the success of the power function") rather than derived.

To state this precisely: Theorem 4.2 proves something about the *marginal* expected accuracy under a distributional assumption, but the calibration target f(k) = 1/k^τ must hold *conditionally* on PSS = k. These are different quantities, and the proof does not bridge them. This is a significant theoretical gap in the paper's central claim.

**The Dirichlet assumption:** Assuming that softmax outputs **p** and the renormalized within-set probabilities **q**^(t) are jointly drawn from Dirichlet distributions with the same shape vector **a** is restrictive. In practice, neural network softmax outputs are known to be poorly modeled by Dirichlet distributions—they tend to be overconfident point masses. The logistic-normal alternative in Appendix A (Theorem A.1) is a useful robustness check, but the O(k^{-3/2}) correction term means the result holds only asymptotically for large set sizes, which may not be the practically relevant regime.

**Empirical validation of the target function:** The paper validates the power function by showing (Fig. 4) that uniform sampling accuracy vs. PSS fits 1/k well. However, 1/k is the trivially expected result under uniform sampling from a set containing the true class exactly once—it is essentially a tautology for the uniform sampling case (t=0). The interesting claim is that 1/k^τ with τ < 1 is correct for t > 0, which is validated empirically but not well-justified theoretically.

**Table 3:** The logarithmic function actually outperforms the power function for Multinomial and Top-1 sampling, but the paper sticks with the power function because it handles the uniform case better. This pragmatic choice means the proposed target function is not universally best, and the paper should be more explicit about this limitation.

---

### CPAC Algorithm (Section 4.3)

**Algorithm 1 (bi-level optimization):** The algorithm alternates between fixing ν (the CP quantile) and optimizing (W, b), then recomputing ν. This is a heuristic approximation to true bi-level optimization, and the paper acknowledges in the conclusion that convergence and generalization are "only validated empirically." For ICLR, this is acceptable if the empirical validation is strong, but (see below) it is not.

**Data reuse concern:** The calibration set D_cal is used for: (a) temperature scaling (existing APS pipeline), (b) CPAC optimization of (W, b), and (c) computing the conformity score quantile ν̂. The coverage guarantee of split conformal prediction relies on D_cal being used only for step (c). Using D_cal for step (b) (CPAC optimization) can invalidate the theoretical coverage guarantee. The paper does not address this. The coverage numbers in Tables 1–2 show that CPAC indeed reduces empirical coverage below the nominal 90% in several settings (e.g., ViT-Large Clean: PS achieves 94.52% coverage, CPAC achieves 92.39%), which may partially reflect this confounding.

**Coverage guarantee validity:** The APS coverage guarantee requires that the calibration set used to compute ν̂ is exchangeable with the test set and not used for model fitting. By optimizing W and b on D_cal and then reusing D_cal for the quantile computation, CPAC likely invalidates the coverage guarantee. The paper does not formally address this, stating only that the method is a "pre-processing step before the quantile computation." This is the most critical practical concern—if CPAC breaks the coverage guarantee, one of CP's key appeals is lost.

**The regularization term** ‖W - I‖²_F + ‖b‖² toward the identity initialization is sensible, but the choice λ=1e-4 via "preliminary experiment" without systematic ablation is insufficient. An ablation over λ should be shown.

---

### Experiments & Results (Section 5)

**CIFAR100:** The paper excludes CIFAR100 from the CPAC comparison ("calibration error on CIFAR100 is not high"), which is a selective reporting concern. If CPAC is a general method, it should be evaluated on all datasets consistently.

**The main results in Table 1 (ImageNet, ViT)** show that CPAC reduces Uni. CP-ECE but often leaves Std. CP-ECE unchanged or marginally improved, while reducing PSS (fewer classes in the prediction set). However, in Appendix C (coverage-fixed comparison), "our method enlarges the PSS compared with the baseline." This is the fair comparison—when coverage is fixed at the nominal level, CPAC actually produces larger prediction sets. The paper acknowledges this but argues the coverage-fixed experiment "is not doable in practice as the test set is unknown." This reasoning is unconvincing: the same coverage fixing could be done on a held-out calibration split. The increased PSS under fixed coverage is a serious concern that undermines the practical utility of CPAC, since smaller PSS (efficiency) is the main practical measure of CP quality beyond coverage.

**Table 2 (GPT-2 on Topic Classification):** The improvements from CPAC are largely within standard deviation across almost all conditions. For example, on Clean data: PS Uni CP-ECE = 6.01 ± 0.53, PS-Full = 6.21 ± 0.37, CPAC = 5.50 ± 0.46—these differences are not statistically significant. The paper does not report statistical significance tests, and given the overlapping standard deviations across most entries in Table 2, the improvements are questionable.

**No baseline CP variants:** The paper only compares CPAC against the standard APS with Platt scaling (PS) and a "PS-Full" variant. Missing comparisons include: RAPS (regularized APS), THR, CRC (conformal risk control), or RSCP (the random-set method of Manchingal et al., 2025 cited in related work). The restriction to a single CP score function (APS) limits generalizability claims.

**The PS-Full baseline** is not clearly defined in the text. From context it appears to be a full linear transformation version of Platt scaling, but its formulation should be stated explicitly for reproducibility.

**The PSS threshold heuristic (PSS < 400 on ImageNet):** Excluding high-PSS samples from CPAC optimization is a design choice with significant impact. PSS < 400 on a 1000-class dataset excludes up to 40% of the class space. The rationale ("we only need to cover (1-α) of all samples") is not clearly connected to why exactly 400 is appropriate, and no ablation over this threshold is provided.

**No statistical significance testing:** With five seeds, the reported standard deviations suggest that many of the gains in Tables 1–2 are within noise. A formal test (e.g., paired t-test across seeds and perturbation conditions) would be essential for ICLR claims.

---

### Limitations & Broader Impact

The conclusion honestly notes the lack of convergence and generalization theory for the bi-level optimization. However, several important limitations go unacknowledged:

1. **Coverage guarantee invalidation** under CPAC (discussed above) is the most critical omission.
2. The definition of "conformal calibration" via multinomial sampling is a design choice, and the paper presents it as if it is the natural definition when it is not universally agreed upon.
3. CPAC's hyperparameter sensitivity (τ, λ, t, PSS threshold, batch size, learning rate grid) means it requires substantial tuning and is not truly a lightweight post-processing step.
4. The paper does not discuss computational cost of CPAC (multiple rounds of quantile recomputation on the calibration set) relative to baselines.

---

### Overall Assessment

This paper addresses a genuinely underexplored dimension of conformal prediction—whether the prediction set size is a well-calibrated uncertainty signal relative to per-instance accuracy. The empirical study demonstrating weak calibration of conformalized classifiers is a meaningful contribution. However, the paper has three interrelated weaknesses that collectively keep it below the ICLR acceptance bar in its current form.

First, the core theoretical justification (Theorem 4.2) does not directly support the proposed target function f(k) = 1/k^τ: the theorem establishes a scalar expected accuracy under a Dirichlet model, not a conditional function of PSS size k, and the extrapolation from one to the other is not formally proven. Second, CPAC invalidates or weakens the coverage guarantee—the defining property of conformal prediction—by reusing the calibration set for both model optimization and quantile computation, and this is neither acknowledged nor mitigated. Third, under the fair (coverage-fixed) experimental comparison, CPAC increases PSS rather than decreasing it, which reverses the apparent efficiency gain and suggests that the reported improvements in unconstrained settings conflate calibration improvement with coverage reduction. These are not cosmetic issues; they go to the heart of whether the contribution is technically sound and practically useful.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates the calibration of uncertainty expressed via prediction set size (PSS) in conformal prediction (CP) for deep classifiers, arguing that small prediction sets should consistently correspond to higher prediction accuracy. The authors define CP calibration, propose a theoretical target function based on Dirichlet assumptions, and introduce a bi-level optimization algorithm (CPAC) to align PSS with accuracy before the quantile computation stage. Empirical results on Vision Transformers, ResNets, and GPT-2 demonstrate that CPAC significantly reduces calibration error compared to standard CP baselines while maintaining coverage guarantees.

### Strengths
1.  **Novel Problem Formulation:** The paper correctly identifies a gap in CP literature: while coverage is guaranteed, the *calibration* of the prediction set size itself relative to reliability is often ignored. Defining CP calibration as a function $f(k)$ relating PSS ($k$) to expected accuracy is a clear and useful conceptual contribution.
2.  **Theoretical Grounding:** The derivation of the calibration target function $f(k) = 1/k^\tau$ using Dirichlet distribution assumptions provides a solid theoretical underpinning for the proposed metric, distinguishing it from heuristic approaches. This theoretical link between the predictive distribution shape and the target curve is a notable strength.
3.  **Comprehensive Empirical Evaluation:** The authors validate the method across diverse modalities (image and text), architectures (CNNs, ViT, LLMs), and perturbation types (noise, blur, dropout). The inclusion of reliability diagrams (Figures 2, 3, 5) clearly visualizes the calibration improvement, adding credibility to the quantitative results in the tables.
4.  **Practical Plug-in Framework:** The CPAC algorithm is presented as a pre-processing step that can be applied to pre-trained models without retraining the entire backbone, which aligns well with the practical constraints of deploying uncertainty quantification in production systems.

### Weaknesses
1.  **Efficiency-Calibration Trade-off:** The paper acknowledges that CPAC often increases the Prediction Set Size (PSS) (e.g., Table 1 appendix vs. Table 4), which can be interpreted as a loss in efficiency. A deeper analysis is needed on whether the gain in calibration is worth the potential increase in computational cost for downstream users (e.g., larger sets might reduce the practical utility of CP for decision-making).
2.  **Sampling-Proxy for Accuracy:** The definition of accuracy relies on multinomial sampling from the prediction set (Eq. 4 and 5). While justified as a decision-making proxy, this introduces additional variance and heuristic dependency not present in standard point-prediction calibration. It remains unclear if this sampling method consistently reflects the actual risk of selecting a specific class within the set for all downstream applications.
3.  **Bi-Level Optimization Complexity:** Algorithm 1 requires iterative bi-level optimization on the calibration set, which is computationally expensive compared to single-stage calibration methods (e.g., temperature scaling). The paper admits theoretical convergence analysis is missing, and the sensitivity of the optimizer to hyperparameters (e.g., learning rate, regularization $\lambda$) requires a more systematic ablation study.
4.  **Assumption Sensitivity:** The target function derivation relies on the Dirichlet distribution assumption for the predictive probabilities. While the paper mentions logistic-normal as an alternative, the robustness of the CPAC method when these assumptions are violated (e.g., in highly adversarial or out-of-distribution settings) is not fully explored.

### Novelty & Significance
The work offers high novelty by shifting the focus of CP research from coverage validity to the internal consistency of uncertainty measures (PSS vs. Accuracy). This is significant because current CP deployments often treat the prediction set size as a generic uncertainty proxy without verifying if it is trustworthy for risk-aware decisions. By formalizing "CP calibration," the paper provides a necessary metric for evaluating the quality of uncertainty estimates in safety-critical applications, potentially influencing future CP standards and implementation in foundation models.

### Suggestions for Improvement
1.  **Analyze Computational Overhead:** Include a comparison of training/inference overhead between CPAC and standard CP baselines (e.g., time per sample, GPU memory usage) to help practitioners assess the cost-benefit ratio of the method.
2.  **Refine Accuracy Metric Discussion:** Add a discussion or ablation on the impact of the sampling temperature $t$ in the accuracy metric definition. Is the calibration result stable across different $t$ choices, or is it overly sensitive?
3.  **Expand Trade-off Analysis:** Provide a more detailed analysis of the PSS increase. If PSS increases, does the "effective" set size (after pruning or ranking) change? A cost-benefit curve (Calibration Error vs. Mean PSS) would clarify the method's impact.
4.  **Strengthen Convergence Argument:** Since the conclusion admits missing theoretical analysis on convergence, include a preliminary convergence study or loss landscape visualization for the bi-level optimization in the main text or appendix to reassure reviewers of the algorithm's stability.
5.  **Address OOD Behavior:** Given the robustness to noise experiments are strong, test the method on a specific Out-of-Distribution (OOD) dataset (e.g., CIFAR-100 for a model trained on ImageNet) to show how PSS calibration behaves under distributional shift.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Coverage Validity with Hold-Out Set:** The method optimizes parameters on the calibration set ($D_{cal}$) before computing quantiles on the same set, violating split CP exchangeability assumptions. You must validate coverage guarantees using a strict 3-way split (Train, Cal-Opt, Cal-Quantile) to prove test coverage is not compromised by overfitting.
2. **Runtime and Complexity Analysis:** Conformal Prediction is valued for low computational overhead, but CPAC uses bi-level optimization. Provide explicit runtime comparisons against standard CP and Temperature Scaling to demonstrate the method is practically viable.
3. **Direct Comparison to CP Temperature Scaling:** You cite Xi et al. (2024) and Dabah & Tirer (2024) but do not include them as baselines. Add these methods to Tables 1 and 2 to prove CPAC offers improvement beyond optimized temperature scaling.
4. **Downstream Utility Task:** Reducing CP-ECE is not inherently useful without demonstrating impact. Include a selective classification or human-in-the-loop experiment where PSS calibration directly improves decision quality or safety.
5. **Ablation on Optimization Rounds:** Table 1 shows results for specific hyperparameters without justification. Provide an ablation study on the number of optimization rounds ($M_{opt}$) to show diminishing returns and justify the computational cost.

### Deeper Analysis Needed (top 3-5 only)
1. **Theoretical Proof of Coverage Preservation:** The paper claims coverage is maintained (Table 1), but optimizing logits on $D_{cal}$ theoretically invalidates the standard split CP guarantee. Provide a theoretical proof or bound showing how exchangeability is preserved despite parameter updates on the calibration data.
2. **Justification for Sampling-Based Accuracy:** The core definition of CP calibration relies on multinomial sampling from the prediction set to define accuracy, which is non-standard. Explain why "sampled accuracy" is a superior metric for set uncertainty compared to traditional set containment probability.
3. **Overfitting Risk on Calibration Set:** Since $W$ and $b$ are fitted to minimize error on $D_{cal}$, there is a high risk of overfitting the calibration distribution. Analyze the gap between calibration loss and test loss to quantify the generalization error of the calibration parameters.
4. **Target Function Robustness:** Table 3 shows the Logarithmic function fits better for Top-1 sampling, yet the Power function is chosen based on Uniform sampling. Analyze why the Power function is preferred despite empirical evidence suggesting alternatives fit specific sampling regimes better.
5. **Sensitivity to Distribution Shift:** The experiments use synthetic noise (Gaussian, Blur), but real distribution shift is more complex. Analyze how CPAC performance degrades under natural distribution shifts (e.g., ImageNet-A/C) compared to standard CP.

### Visualizations & Case Studies
1. **Coverage vs. CP-ECE Trade-off Curve:** Plot coverage violation rates against CP-ECE improvement to expose if calibration gains come at the cost of invalid coverage guarantees.
2. **Subgroup Reliability Diagrams:** Show reliability diagrams split by class difficulty or semantic subgroups to reveal if CPAC biases uncertainty estimates against specific classes (conditional calibration).
3. **Optimization Convergence Plots:** Visualize the bi-level optimization loss curves over epochs to demonstrate stability and confirm the algorithm converges reliably across different seeds and models.
4. **Prediction Set Size Distribution:** Histogram the prediction set sizes before and after CPAC to show exactly which samples are affected (e.g., does it only shrink easy sets or also expand hard ones?).
5. **Case Study on Failure Modes:** Provide specific examples where CPAC corrects a miscalibrated set versus where it fails, to qualitatively validate the method's behavior on hard samples.

### Obvious Next Steps
1. **Implement 3-Way Split Protocol:** Redesign the experimental pipeline to use a held-out set for quantile computation separate from the optimization set to ensure valid theoretical guarantees.
2. **Simplify to Single-Level Optimization:** Investigate whether optimizing $W, b$ without the inner quantile loop achieves similar calibration with significantly reduced computational complexity.
3. **Validate on Real-World Decision Tasks:** Move beyond metric improvement to demonstrate CPAC in a safety-critical application (e.g., medical diagnosis or autonomous driving) where PSS calibration dictates action.
4. **Open Source Calibration Code:** Release the CPAC implementation with clear instructions on hyperparameter tuning, as bi-level optimization is sensitive and difficult to reproduce without detailed code.
5. **Extend to Regression Tasks:** Since the theoretical analysis mentions regression connections, validate whether the CPAC framework extends to regression prediction intervals to broaden the paper's impact.

# Final Consolidated Review
## Summary
This paper investigates an underexplored question in conformal prediction: whether the prediction set size (PSS)—commonly used as an uncertainty signal—is well-calibrated with respect to prediction accuracy. The authors define CP calibration via multinomial sampling from prediction sets, propose a theoretical target function $f(k) = 1/k^\tau$ based on Dirichlet distribution assumptions, and introduce CPAC, a bi-level optimization algorithm to improve calibration. Experiments on ImageNet (ResNet, ViT) and topic classification (GPT-2) demonstrate calibration improvements across various perturbation settings.

## Strengths
- **Novel Problem Formulation:** The paper correctly identifies a gap in CP literature—while coverage guarantees are well-studied, whether PSS meaningfully reflects per-instance reliability is not. Defining CP calibration as a monotonic relationship between PSS and expected accuracy is a clear conceptual contribution that addresses real concerns in safety-critical deployments.
- **Comprehensive Empirical Analysis:** The systematic study of CP calibration across multiple architectures (ResNet50/101, ViT-Base/Large, GPT-2), datasets (CIFAR100, ImageNet, topic classification), and perturbation types (Gaussian noise, blur, dropout, typos) provides strong evidence that CP calibration is indeed weak in standard conformalized models (Figures 2-5, Tables 1-2).
- **Clear Visual Evidence:** The reliability diagrams (Figures 2, 3, 5, 7-14) effectively illustrate the miscalibration problem and CPAC's improvements, making the empirical case accessible.
- **Theoretical Attempt:** The derivation of the target function from Dirichlet assumptions, while imperfect, attempts to ground the calibration target theoretically rather than proposing it purely heuristically.

## Weaknesses
- **Theoretical Gap in Theorem 4.2:** The theorem establishes a marginal expected accuracy $\mathbb{E}[\mathbf{p} \cdot \mathbf{q}] = K^{-\tau}$ under Dirichlet assumptions—not the conditional function $\mathbb{E}[\text{Acc}_t(\mathbf{x}) \mid |S(\mathbf{x})| = k] = 1/k^\tau$ as a function of set size $k$. The extrapolation from the scalar result to the functional form $f(k) = 1/k^\tau$ is motivated empirically ("inspired by the success of the power function") but not formally derived. The paper relies on Figure 4's empirical fit for justification, leaving the theoretical foundation incomplete.
- **Coverage Guarantee Validity Under CPAC:** CPAC optimizes $(\mathbf{W}, \mathbf{b})$ on $D_{\text{cal}}$, then reuses the same $D_{\text{cal}}$ for quantile computation. The split conformal prediction guarantee requires that the calibration set be exchangeable with test data and used only for quantile computation—optimizing on $D_{\text{cal}}$ before quantile computation may invalidate this assumption. The paper does not address this formally, though empirically coverage remains near or above the 90% target (Tables 1-2 show CPAC coverage around 91-94%).
- **Fair Comparison Shows PSS Increase:** Under coverage-controlled comparison (Table 4), CPAC *increases* PSS rather than decreasing it (e.g., Clean: PS PSS 6.20 → CPAC PSS 7.81). The paper acknowledges this but argues coverage control "is not doable in practice as the test set is unknown." This response is unsatisfying—a held-out split could provide fair comparison. The efficiency loss under fair conditions raises questions about practical utility.
- **Marginal Significance on NLP Results:** In Table 2 (GPT-2), many CPAC improvements fall within reported standard deviations (e.g., Clean: PS-Full Uni CP-ECE $6.21 \pm 0.37$ vs. CPAC $5.50 \pm 0.46$). No statistical significance testing is provided.
- **Missing Baselines:** Comparison is limited to APS with Platt scaling. Other CP methods (RAPS, THR, conformal risk control) and the temperature scaling approaches of Xi et al. (2024) and Dabah & Tirer (2024) are not included, limiting generalizability claims.
- **Hyperparameter Sensitivity:** CPAC requires tuning $\tau$, $\lambda$, $t$, $M_{\text{opt}}$, learning rate, and PSS threshold (PSS < 400 on ImageNet). The threshold heuristic lacks justification, and no ablation on $M_{\text{opt}}$ or $\lambda$ is provided.

## Nice-to-Haves
- Runtime and computational overhead analysis comparing CPAC to standard CP and temperature scaling.
- Ablation studies on optimization rounds ($M_{\text{opt}}$) and regularization parameter ($\lambda$).
- Comparison to other CP efficiency methods (RAPS, conformal risk control) beyond APS.

## Removed Points
These points are flagged to be removed, treat them with caution:

- *Critic claim that GPT-2 is too old to demonstrate LLM applicability:* The paper demonstrates the method works on transformer-based language models; GPT-2 is a valid test case for proof of concept. Scope creep to demand testing on larger LLMs.
- *Critic nitpick about "linear function" characterization of confidence calibration:* The identity function $P(\hat{Y}=Y|\hat{P}=p) = p$ is indeed linear in $p$. This precision issue is not substantive.
- *Critic claim that fairness justification for uniform CP-ECE is unconvincing:* Whether to weight bins uniformly or by sample count is a design choice; both metrics are reported in tables, and the paper reasonably motivates uniform weighting for subgroup fairness concerns.
- *Spark finder suggestion to test on OOD datasets like ImageNet-A/C:* This extends scope beyond the paper's stated contribution; the perturbation experiments already assess robustness.

## Novel Insights
The paper surfaces an important distinction between *coverage validity* (the probability that the true label is in the prediction set) and *calibration validity* (whether smaller sets reliably correspond to higher accuracy). These are fundamentally different desiderata: a method can achieve valid coverage while being poorly calibrated. The multinomial sampling construction to connect set-size uncertainty to point-prediction accuracy is a reasonable design choice that enables practical evaluation, though it is a design choice rather than a canonical definition. The empirical finding that pre-trained models show worse CP calibration than randomly initialized ones (Figure 2)—despite higher accuracy—is a counterintuitive observation that warrants further investigation.

## Suggestions
- **Formalize the theoretical claim:** Either strengthen Theorem 4.2 to directly prove the conditional form, or explicitly frame it as empirical motivation with theoretical inspiration rather than derivation.
- **Validate coverage guarantees with proper splits:** Run experiments with a three-way split (train / cal-opt / cal-quantile) to verify that coverage guarantees hold when $D_{\text{cal}}$ is not reused for optimization.
- **Provide statistical significance tests:** Report p-values or confidence intervals for CP-ECE differences, especially for the GPT-2 results where improvements appear marginal.
- **Include baseline CP variants:** Add RAPS and THR baselines to demonstrate that CPAC's calibration improvements are not achievable by simpler modifications.
- **Justify the PSS threshold:** Explain or ablate the PSS < 400 threshold used for optimization on ImageNet.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 2.0]
Average score: 3.5
Binary outcome: Reject
