=== CALIBRATION EXAMPLE 74 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "FF-Erase: Machine Unlearning and Verification for Forward-Forward Models" accurately reflects the paper's scope. The abstract makes two concrete claims: (1) 1.9–3.1× speedup over retraining, and (2) 1.6–3.3% accuracy degradation. Both are substantiated in experiments, so the abstract is well-calibrated.

A minor framing concern: the abstract calls G-MIA a "black-box attack," but as discussed in the method analysis below, this characterization is inaccurate. The abstract should clarify that G-MIA requires access to all intermediate layer goodness scores.

---

### Introduction & Motivation

The introduction clearly identifies two specific technical barriers to applying standard unlearning to FF models: (a) FF's sensitivity to parameter tuning due to BP-free layer-wise greedy optimization, and (b) the difficulty of apportioning the per-layer "penalty" for forgotten data given layer-independence. These are genuine and well-articulated challenges.

However, the paper overstates the practical urgency of FF unlearning. The claim that FF is "particularly well-suited for resource-constrained scenarios such as edge computing" is plausible but weakly supported. FF-based models currently trail BP counterparts significantly in accuracy on complex tasks, limiting the real-world deployment footprint. The motivational claim would benefit from data comparing FF to BP in realistic constrained settings, or at least a citation demonstrating FF deployment in production.

The two key questions are well-posed and serve as a clean structural backbone for the paper.

---

### Related Work

The related work is reasonably comprehensive. The coverage of exact vs. approximate unlearning methods, MIA taxonomy, and FF algorithm variants is appropriate for an 8-page paper.

**Issue 1 – Missing MIA literature**: The G-MIA comparison in §6.1 evaluates against Shokri et al. (2017) as the representative black-box MIA and Nasr et al. (2019) as the white-box baseline. Carlini et al. (2022) ("Membership Inference from First Principles") is cited in the appendix but not included as a baseline, even though it represents the current state-of-the-art and would be a natural point of comparison. The exclusion is not explained.

**Issue 2 – Formal unlearning bounds not discussed**: Recent work on certified/approximate unlearning with differential-privacy-style guarantees (e.g., Sekhari et al., Guo et al.) is cited but not engaged with theoretically. The paper makes no claim about formal guarantees of forgetting, which is fine for an approximate-unlearning paper, but the gap should be explicitly acknowledged rather than silently ignored.

---

### Preliminaries (§3)

The FF training description (Equations 1–3) is correct and reasonably precise. The column-wise L1 norm for goodness in Equation (1) and the softmax-cross-entropy-style loss in Equation (2) are clearly stated.

**One technical ambiguity**: Equation (1) defines `z^l = (h^l − g^l) / sqrt(σ² + ε)`, which appears to be layer normalization of `h^l`. However, `g^l` is defined as `||h^l||₁`, a scalar-valued vector — subtracting a vector of L1 norms from the raw output before normalizing is unusual. The footnote partially clarifies this but leaves it opaque whether the normalization is applied per-class or globally. This matters for reproducibility: it is unclear how the layer normalization interacts with the goodness computation across the J-class structure.

The machine unlearning notation in §3.2 is standard and well-expressed.

---

### Methodology (§4)

**Core Idea:** FF-Erase alternates between "forgetting forward" (KL divergence minimization toward a guidance model's goodness) and "recovering forward" (re-application of the FF training objective on remaining data). The key insight is that using a reference goodness distribution from a data-ignorant guidance model stabilizes optimization and prevents invalid goodness distributions.

**Issue 1 – Does the guidance model itself already solve the problem?** The guidance model is trained exclusively on `D_remain` and is therefore already "ignorant" of the forgetting data. The paper acknowledges (Appendix C.4) that guidance models with 30–50% data and 20–50% epochs achieve 60–71% test accuracy, compared to 81% for full retraining. The authors argue this insufficiency justifies using FF-Erase on top. However, this raises an important question: **why not simply continue training the guidance model longer?** The guidance model at α₁=0.3, α₂=0.5 (t₀≈160s) combined with goodness decrease (≈270s) gives similar total time to just training the guidance model longer. No experiment directly tests whether extended guidance model training converges to RE performance, which would undermine the need for FF-Erase.

**Issue 2 – Asymmetry in the KL divergence direction (Equation 5).** The paper minimizes `D_KL(g^l(x; θ_o) || g*^l(x; θ_g))`, i.e., the forward KL, which drives the original model's goodness toward the guidance model's goodness distribution. This choice is not theoretically motivated. The forward KL (mode-covering) and reverse KL (mode-seeking) have different optimization behaviors. The paper should justify this choice explicitly, especially since the stability argument (avoiding "invalid goodness distributions") is the central motivation.

**Issue 3 – Recovery step hyperparameter K lacks sensitivity analysis.** Algorithm 1 includes recovery every K epochs, and a footnote says "A smaller K leads to better model utility and worse efficiency." However, K is not analyzed in §6.4's ablation study, despite being as critical as α₁ and α₂. The interaction between K and α₁/α₂ is completely unexplored.

**Issue 4 – Early stopping thresholds ε₁ and ε₂.** These are listed as inputs in Algorithm 1 but their values are never reported in the main text or appendix. This is a reproducibility gap.

**Issue 5 – Efficiency formula (Equation 9) is approximate.** The formula `t_unl ≈ α₁·α₂·t_ret + (K⁻¹ + β)·t_ret` treats t₁ as proportional to (K⁻¹ + β)·t_ret, but this ignores the early stopping mechanism. If ε₁ or ε₂ triggers early, t₁ could be much shorter. The empirical claim of 25–35% of t_ret is reasonable but the formula as written is misleading in its generality.

---

### G-MIA (§5)

**Issue 1 – "Black-box" framing is inaccurate.** The most significant conceptual problem in the paper. G-MIA is called a "black-box" attack because it does not require model gradients or weight access. However, it explicitly requires access to the goodness vectors from **all intermediate layers** (`g^1, g^2, ..., g^L`). This is **not** how "black-box" is defined in the standard MIA literature (e.g., Shokri et al. 2017), where black-box access means only the final output logits or predictions are available. G-MIA requires a form of **gray-box access**: model outputs beyond the final prediction layer. In real deployed systems (where FF would plausibly run on edge devices), intermediate layer outputs are rarely exposed. The paper should either: (a) rename the attack as "gray-box" or "intermediate-output" MIA, or (b) explicitly justify why intermediate goodness outputs are accessible in the target deployment scenario.

**Issue 2 – Circularity in evaluation.** The paper proposes G-MIA, then uses G-MIA as the primary metric to evaluate FF-Erase. Both the unlearning method and the evaluation metric are designed by the same authors to exploit properties of the same FF models. This circularity weakens the credibility of results. An independent external verifier (e.g., a model-agnostic verification protocol such as influence function-based analysis, or a held-out adversary) should be included to provide an unbiased assessment.

**Issue 3 – Synthetic data assumption for shadow training.** G-MIA assumes the attacker can generate synthetic data matching the training distribution via model inversion. This assumption is labeled "common" and Fredrikson et al. (2015) is cited, but model inversion for complex datasets like CIFAR-100 is substantially harder than for MNIST. The paper should clarify whether G-MIA's performance degrades when synthetic data quality decreases, as this directly affects its practical utility as a verification tool.

**Issue 4 – G-MIA outperforming white-box MIAs is surprising and under-explained.** The paper reports G-MIA achieving better ACC/AUC than white-box GR and GAP baselines on VGG13 + CIFAR-100. The explanation offered is that "deeper models and complex datasets amplify layer-wise independent training." This informal explanation is insufficient. Mechanistically: why does having goodness vectors (essentially L1 norms of class-specific outputs) provide more membership signal than having full gradients (GR) or all intermediate activations (GAP/ST)? This result warrants deeper analysis — possibly the GR baseline implementation is suboptimal, or the dropout/batch-norm defenses happen to hurt gradient-based attacks more. This is a key claim that deserves more rigor.

---

### Experiments (§6)

**Issue 1 – No statistical significance testing.** All reported accuracy numbers (e.g., 81.61 for RE, 81.58 for D-(0.5,0.5) in Table 1) are point estimates with no variance. Given that differences between FF-Erase variants and the RE baseline are on the order of 0.03–0.6 percentage points, the absence of confidence intervals or standard deviations makes it impossible to assess whether observed differences are meaningful.

**Issue 2 – Fixed forgetting fraction.** All experiments use `|D_forget|/|D_train| = 20%` (mentioned in the text: "we sample 20% of the training data"). Machine unlearning performance is known to be highly sensitive to the forgetting fraction. Unlearning a small number of samples (e.g., 0.1%, 1%) is a fundamentally different problem from unlearning 20%. The paper should test a range of β values.

**Issue 3 – Class-level unlearning not evaluated.** The unlearning literature commonly distinguishes between sample-level forgetting (random samples) and class-level forgetting (removing all samples of a class). The paper only tests random sample forgetting. Class-level forgetting is arguably the more important setting for GDPR compliance (e.g., a user who requests deletion of their data contributes samples predominantly from specific contexts or categories).

**Issue 4 – Datasets are limited to simple benchmarks.** MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 are the evaluation datasets. While these are consistent with the FF literature, the paper cites FF-LSTM and FORWARDGNN as motivating applications, yet evaluates only image classification. Demonstrating generalization beyond image classification would substantially strengthen the paper's contribution.

**Issue 5 – Advanced baselines appear only in the appendix.** The CKA analysis in Appendix C.2 includes Bad Teacher (BT), FATS, and FYE as additional baselines and shows all of them fail or underperform. This is important comparative evidence. Moving at least the summary of these comparisons to the main text (Table 1 or Figure 4) would strengthen the paper's claims significantly.

**Issue 6 – The comparison with direct GA is unfairly one-sided.** The paper tests only gradient ascent and retraining as baselines in the main text. It argues other methods fail (Appendix A), which is likely true. But the evidence in §6.3 sweeps λ from 10⁻³ to 10¹ and shows GA either collapses or fails to unlearn — this is a fair experimental demonstration. The concern is whether any GA tuning regime (e.g., with gradient clipping, smaller learning rates) could work at all. The search space coverage for GA tuning could be more exhaustive.

---

### Limitations & Broader Impact

The paper has no dedicated limitations section. This is a significant omission. Key limitations that should be stated:

1. **No formal privacy guarantees**: FF-Erase is approximate unlearning; no ε-unlearning certificate or DP-style bound is provided.
2. **Dependence on remaining data**: Both mini-retraining and fast-distillation require access to a meaningful subset of `D_remain`. In scenarios where remaining data is small or unavailable, the method degrades (R.G.M result in Table 1 shows 55.53% accuracy collapse when a random guidance model is used).
3. **No non-image-task evaluation**: Despite citing FF-LSTM and FORWARDGNN, the paper evaluates only image classification.
4. **Scalability to larger models/datasets unclear**: VGG13/VGG16 on CIFAR-100 is the largest setting tested; scaling to ImageNet-class tasks is unaddressed.
5. **GPT-5 usage**: The LLM usage disclosure in Appendix D mentions "GPT-5," which does not exist publicly as of the paper's submission date. This should be verified and corrected; it undermines confidence in the disclosure's accuracy.

---

### Writing & Clarity

The algorithm is well-specified in Algorithm 1 and Figure 2. The section structure is logical. The main clarity issues are in the method:
- The explanation of why "valid goodness distributions" cannot be determined in advance (§1) is the central motivation, but the paper never formally defines what constitutes an "invalid" distribution. This weakens the justification for needing a guidance model rather than alternative stabilization techniques (e.g., gradient clipping, KL regularization toward the original distribution).
- Table 1's notation (D-(α₁, α₂) and R-(α₁, α₂)) is explained but not prominently displayed in the table header, making the table hard to parse at a glance.

---

### Overall Assessment

FF-Erase addresses a genuine and underexplored gap — machine unlearning for Forward-Forward algorithm-based models — with a pragmatic and experimentally validated solution. The identification of the two core challenges (optimization instability and layer-wise penalization ambiguity) is the paper's clearest contribution. However, the submission has several issues that need resolution before acceptance at ICLR. The most serious is the mischaracterization of G-MIA as a "black-box" attack when it requires intermediate layer outputs unavailable in standard black-box settings; this conflation weakens both the attack's novelty claim and its value as a practical verification tool. The circular use of G-MIA as the primary evaluation metric for FF-Erase's effectiveness — both being proposed in the same paper — should be addressed with an independent verification signal. Experimentally, the lack of statistical significance testing, the fixed 20% forgetting fraction, the absence of class-level unlearning experiments, and the confinement to simple image benchmarks all narrow the paper's scope below what ICLR's standards require. The unanswered question of whether extended guidance model training alone achieves the same result as FF-Erase also needs resolution to substantiate the method's necessity. Addressing these issues would make a substantially stronger submission; in its current form, the contribution is real but insufficiently validated for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **FF-Erase**, the first machine unlearning framework specifically designed for Forward-Forward (FF) models, addressing the instability of applying standard Backpropagation (BP)-based unlearning methods to FF architectures. Complementing the unlearning framework, the authors introduce **G-MIA**, a goodness-based membership inference attack that leverages hidden layer activations for verification. Experimental results demonstrate that FF-Erase achieves significant efficiency gains (1.9-3.1x faster than retraining) while maintaining model utility, and G-MIA outperforms existing verification methods.

### Strengths
1.  **Problem Identification and Novelty:** The paper correctly identifies a gap in the literature: existing machine unlearning methods fail on FF models due to model collapse caused by the architecture's sensitivity and layer-wise optimization. This is the first work to formalize and solve this problem, as evidenced by the dedicated sections in §1 and §2.
2.  **Effective Solution with Efficiency Gains:** FF-Erase effectively mitigates the collapse issue using a goodness-guided KL-divergence strategy. As shown in Table 1 (Main Text), the method achieves unlearning in ~583s compared to 1107s for full retraining, confirming the claimed efficiency improvements without sacrificing significant utility (maintained ~80% accuracy on Dtest).
3.  **Robust Verification Metric:** The proposal of G-MIA provides a tailored verification tool that is superior to standard black-box attacks (like FL) when applied to FF models. Figure 3 demonstrates that G-MIA achieves higher accuracy (AUC/ACC) than white-box baselines in certain settings, offering a more reliable metric for data owners to verify compliance.
4.  **Comprehensive Experimental Evaluation:** The authors evaluate across multiple datasets (CIFAR-10/100, MNIST, FMNIST) and architectures (TinyCNN, AlexNet, VGG), providing ablation studies on guidance model strategies (§6.4) and layer-wise analysis (Appendix C.2), adding credibility to the findings.

### Weaknesses
1.  **"Black-Box" MIA Definition Ambiguity:** The paper claims G-MIA is a "black-box attack" in Section 5, yet it explicitly states the attacker "obtains the output of the target model... the goodness vectors from all layers." Accessing internal layer activations (goodness vectors) typically constitutes a semi-white-box or feature-extraction attack rather than a standard black-box attack (which relies only on final logits/predictions). This distinction is critical for the claim of "practicality" for data owners without model access.
2.  **Dependence on Guidance Model Quality:** The ablation study in Table 1 and §6.4 indicates that unlearning performance is highly sensitive to the quality of the guidance model. Using a randomly initialized guidance model (R.G.M) leads to a catastrophic drop in utility (ACCt drops to 55.53%). This implies that the overhead and risk of training a reliable guidance model (which requires access to remaining data) must be carefully justified relative to the speedup.
3.  **Limited Scope on Generalization:** The method is strictly tied to FF models' specific "goodness" mechanism. There is no discussion or analysis on whether the *principles* (e.g., distribution guidance) can transfer to standard BP models or vice versa. Given that BP remains the industry standard, the broader impact of the specific FF architecture solution is somewhat limited.
4.  **Lack Theoretical Guarantees:** While the empirical results are strong, there is no theoretical analysis regarding the convergence of the FF-Erase process or bounds on the similarity between the unlearned and retrained models compared to exact unlearning methods (unlike some certified unlearning work).

### Novelty & Significance
The novelty is **high** within the niche of Forward-Forward algorithms. As FF research is an emerging field, providing a compliance tool (unlearning) for it is a timely and necessary contribution. The significance for the broader ICLR community is **moderate**; while the specific solution is FF-centric, the underlying challenge of unlearning in non-BP, layer-wise training regimes is relevant. However, the ambiguity in the "black-box" claim regarding G-MIA slightly detracts from its theoretical rigor compared to standard security metrics.

### Suggestions for Improvement
1.  **Clarify G-MIA Access Model:** Explicitly define the attacker's capabilities. If access to layer goodness vectors is required, classify it honestly as "semi-white-box" or "activation-based" rather than "black-box." Discuss whether this access is realistic for a data owner requesting "right to be forgotten" under GDPR, as they rarely have access to intermediate activations.
2.  **Expand Baseline Comparisons:** Include comparisons with other sophisticated unlearning methods (e.g., Fisher-Exact or Influence Functions adapted for FF) if possible, or explicitly state why they are mathematically infeasible for FF. The current comparison with GA and RE is good, but more baselines would solidify the claim of "effectiveness."
3.  **Refine Efficiency Claims:** Clarify if the 1.9-3.1x speedup accounts for the time to generate the guidance model. Table 1 shows $t_0$ is significant (~25-50% of total unlearning time). A sensitivity analysis on the trade-off between guidance model accuracy and unlearning time would be valuable.
4.  **Address Parser/Formatting Artifacts (for final camera-ready):** While OCR errors are not weaknesses of the science, the manuscript's final version should ensure all equations (e.g., Eq 1, 3, 5) and tables (Table 1, 3) render correctly, as the garbled text currently obscures specific mathematical terms which could confuse reviewers unfamiliar with FF notation.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **SOTA Baselines:** Compare against adapted state-of-the-art BP unlearning methods (e.g., Fisher masking, influence functions) to substantiate the claim that existing methods are infeasible for FF models. Relying solely on Gradient Ascent (GA) is insufficient to prove uniqueness of the FF unlearning challenge.
2.  **Guidance Cost Scaling:** Analyze how unlearning time scales as $|D_{forget}| \to 0$ (where $|D_{remain}| \to |D_{train}|$), since guidance training cost dominates in realistic small-forget scenarios. Without this, the 1.9–3.1× speedup claim is misleading for practical use cases.
3.  **Large-Scale Efficiency:** Evaluate on ImageNet-1k or larger models to verify the efficiency claims hold beyond small benchmarks like CIFAR-10. ICLR standards require efficiency claims to be validated at scales where training costs are actually prohibitive.
4.  **Canonical Privacy Metrics:** Replace accuracy on $D_{forget}$ with rigorous memorization metrics (e.g., likelihood ratio, entropy) to prove actual data removal. Accuracy drop is a weak proxy for privacy and does not guarantee the "right to be forgotten."

### Deeper Analysis Needed (top 3-5 only)
1.  **Circular Verification Risk:** Validate unlearning using standard black-box MIAs (not G-MIA) to ensure effectiveness isn't an artifact of the proposed verification method. Using a self-proposed attack to verify a self-proposed defense creates a high risk of biased evaluation.
2.  **Layer-wise Necessity:** Quantify the privacy leakage difference between unlearning all layers vs. only the final predictor to justify the complex layer-wise approach. If inference relies on the predictor, intermediate layer unlearning must be proven necessary for privacy.
3.  **Collapse Mechanism:** Provide empirical evidence (e.g., gradient norms, goodness variance) explaining *why* GA collapses in FF models specifically, rather than asserting it. Understanding the failure mode is critical to validating the proposed solution's design.
4.  **Hyperparameter Sensitivity:** Analyze sensitivity to $K$ (recovery frequency) and guidance quality ($\alpha_1, \alpha_2$) to demonstrate robustness beyond tuned settings. High sensitivity would undermine the method's practicality for real-world deployment.

### Visualizations & Case Studies
1.  **Goodness Distribution Shift:** Plot histograms of goodness scores for Forget/Remain data to visually confirm distribution matching against Retraining. This directly validates the core mechanism of "goodness-guided" unlearning.
2.  **Cost Breakdown Chart:** Use a stacked bar chart to explicitly separate Guidance Training time vs. Unlearning time vs. Full Retraining time. This exposes whether the efficiency gain comes from the algorithm or the approximated guidance model.
3.  **Layer-wise CKA Heatmap:** Convert Table 3 into a heatmap to intuitively visualize representation similarity across all layers and methods. A table obscures the layer-wise propagation of unlearning effects.
4.  **Stability Trajectory:** Plot accuracy/goodness over epochs for GA vs. FF-Erase to visualize the "collapse" phenomenon dynamically. Seeing the divergence point clarifies the stability advantage of FF-Erase.

### Obvious Next Steps
1.  **Theoretical Bound:** Provide a convergence bound or divergence guarantee between FF-Erase and Retrained models to strengthen theoretical grounding. ICLR expects theoretical justification for novel optimization schemes.
2.  **Guidance Reusability:** Investigate if a single guidance model can serve multiple sequential unlearning requests to amortize the overhead. This determines if the method is viable for continuous unlearning settings.
3.  **Standard MIA Robustness:** Evaluate against established shadow-model MIAs to ensure security claims hold against independent attacks. Reliance on G-MIA alone is insufficient for security claims.
4.  **Code Reproducibility:** Release the full codebase including FF training and unlearning scripts, as FF implementations vary significantly. Reproducibility is a mandatory standard for ICLR acceptance.

# Final Consolidated Review
## Summary

This paper proposes **FF-Erase**, the first machine unlearning framework specifically designed for Forward-Forward (FF) models. The authors identify two key challenges in adapting standard backpropagation-based unlearning methods to FF architectures: (1) optimization instability due to FF's sensitivity to parameter tuning and layer-wise independent training, and (2) difficulty in apportioning per-layer penalties for forgotten data. FF-Erase addresses these through a goodness-guided strategy using a guidance model to stabilize parameter updates, combined with a novel goodness-based membership inference attack (G-MIA) for verification. Experiments demonstrate 1.9–3.1× speedup over retraining while maintaining model utility.

## Strengths

- **Novel problem identification:** The paper correctly identifies that existing BP-based unlearning methods cause model collapse when applied to FF models, and formalizes the unique challenges arising from FF's layer-wise greedy optimization. The empirical demonstration in Section 6.3 (GA leading to either collapse or ineffective unlearning across a wide range of λ values) provides concrete evidence for this claim.

- **Pragmatic solution with demonstrated efficiency gains:** FF-Erase achieves meaningful speedups (Table 1 shows ~426–583s versus 1107s for full retraining) while maintaining model utility (1.6–3.3% accuracy degradation). The ablation study in Section 6.4 demonstrates flexibility in trading off guidance model quality for efficiency.

- **Comprehensive empirical evaluation:** The paper evaluates across four datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100) and multiple architectures (TinyCNN, AlexNet, VGG13/16). Appendix C.2 provides layer-wise CKA similarity analysis showing how knowledge propagates through layers during unlearning.

- **Layer-wise analysis of unlearning effectiveness:** The CKA similarity analysis in Table 3 provides valuable insight into how different unlearning methods affect each layer, revealing that middle layers are most amenable to unlearning while shallow and deep layers retain more shared features.

## Weaknesses

- **G-MIA is mischaracterized as "black-box":** Section 5 states that G-MIA requires "the goodness vectors from all layers" — this is **not** black-box access in the standard MIA literature sense (which means final prediction logits only). Accessing all intermediate layer outputs constitutes gray-box or feature-extraction access, which is rarely available in real deployment scenarios where data owners would verify compliance. The paper should either (a) reclassify G-MIA as "gray-box" and justify why such access is realistic, or (b) demonstrate verification using only final-layer outputs.

- **Circular evaluation concern:** Both the unlearning method (FF-Erase) and the primary evaluation metric (G-MIA) are proposed in this paper and designed to exploit the same FF-specific goodness properties. While Figure 4(c) shows G-MIA scores improving during unlearning, the circularity weakens confidence that FF-Erase provides genuine privacy guarantees. An independent verification signal (e.g., standard black-box MIA, influence function analysis, or external held-out adversary) would strengthen the claims.

- **The guidance model necessity question is not fully resolved:** The paper acknowledges (Appendix C.4) that guidance models trained on 30–50% of remaining data for 20–50% of epochs achieve 60–71% test accuracy. A natural question is: why not simply continue training the guidance model to convergence? Table 1 shows RE takes 1107s while FF-Erase(R)-(0.5,0.5) takes 518s total, so there is real efficiency gain, but the paper does not directly test whether extended guidance model training would match RE performance.

- **G-MIA outperforming white-box methods is under-explained:** Figure 3 shows G-MIA achieving higher accuracy than gradient-based (GR) and activation-based (GAP, ST) white-box attacks on deeper models and complex datasets. The informal explanation that "deeper models amplify layer-wise independent training" does not explain mechanistically why L1-norm goodness vectors provide more membership signal than full gradients or intermediate activations.

- **Limited experimental scope:** All experiments use a fixed 20% forgetting fraction. Machine unlearning difficulty varies substantially with forgetting set size — unlearning 0.1% versus 20% of data presents fundamentally different challenges. Additionally, all evaluations are on image classification; the paper cites FF-LSTM and FORWARDGNN but does not test beyond vision tasks.

- **Missing hyperparameter analysis:** The recovery frequency K is mentioned in Algorithm 1 and a footnote but never analyzed. The early stopping thresholds ε₁ and ε₂ are inputs to the algorithm but their values are never reported, creating a reproducibility gap.

- **Collapse mechanism under-explained:** The paper asserts that GA causes "model collapse" in FF models, but provides only accuracy/G-MIA curves. Gradient norm analysis or goodness variance trajectories would clarify *why* FF is particularly unstable.

## Nice-to-Haves

- Statistical significance testing (confidence intervals or standard deviations) for the reported accuracy numbers, particularly given that differences between FF-Erase variants and RE are often <1 percentage point.

- Class-level unlearning experiments (removing all samples of a class), which is a common GDPR scenario where a user's data predominantly represents specific categories.

- Theoretical bound or convergence guarantee between FF-Erase output and retrained model, even if approximate.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Formal unlearning bounds criticism:** Demanding DP-style certificates or ε-unlearning guarantees is scope creep — the paper explicitly positions FF-Erase as approximate unlearning, and such theoretical bounds are not standard for empirical unlearning papers at ICLR.

- **"GPT-5 usage" disclosure error:** The appendix mentions GPT-5, which does not exist publicly. While this is an error in the LLM usage disclosure, it does not affect the scientific content and is a minor presentation issue.

- **Class-level unlearning as required:** This is presented as a mandatory experiment but is actually a natural extension beyond the paper's stated scope of sample-level forgetting.

- **Theoretical convergence analysis:** ICLR accepts empirical systems papers without theoretical bounds. This would strengthen the paper but is not a requirement.

- **Transferability to BP models:** Criticizing the FF-specific solution for not generalizing to BP models misrepresents the paper's contribution, which is explicitly about the FF architecture.

- **Additional MIA baselines (Carlini et al.):** The paper compares against multiple MIA categories (black-box FL, white-box GR, GAP, ST). While Carlini et al. (2022) represents SOTA, the comparison gap does not invalidate the core claims.

- **Parser/formatting issues:** These are noted as extraction artifacts from the PDF parsing, not issues with the actual manuscript.

## Novel Insights

The layer-wise CKA similarity analysis reveals a striking pattern: middle layers show the lowest CKA similarity (most effective unlearning), while shallow and deep layers retain higher similarity to the original model. This suggests FF models encode general features in early layers and high-level semantic features in final layers — both shared between forgetting and remaining data — while middle layers learn more task-specific, forgettable representations. This architectural insight may generalize to other layer-wise trained models beyond FF.

## Suggestions

- **Clarify G-MIA's access requirements:** Rename it as "gray-box MIA" or explicitly justify why data owners would realistically have access to intermediate goodness vectors. Better yet, evaluate whether verification is possible using only final-layer predictions.

- **Add independent verification:** Include at least one standard black-box MIA (e.g., Shokri et al.) as a secondary evaluation metric to reduce circularity concerns.

- **Report missing hyperparameters:** Provide the values of ε₁, ε₂, and K used in experiments, and add sensitivity analysis for K.

- **Test varied forgetting fractions:** Evaluate at 1%, 5%, and 10% to demonstrate robustness across realistic unlearning scenarios.

- **Address the guidance model efficiency question:** Run an experiment comparing full guidance model training time versus FF-Erase total time to directly justify the method's necessity.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 8.0]
Average score: 5.5
Binary outcome: Reject
