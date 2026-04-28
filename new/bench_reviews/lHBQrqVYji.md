Now I have enough calibration data. Let me write the final review.

## Summary
This paper proposes D-PDDM, a two-stage monitoring algorithm for detecting post-deployment performance deterioration using unlabeled data without requiring training data access during deployment. The method provides theoretical guarantees on false positive rates under non-deteriorating shifts and sample complexity bounds for true positive rates under deteriorating shifts, validated on synthetic data, CIFAR-10.1, and the GEMINI healthcare dataset.

## Strengths
- **Training data-free deployment architecture**: The decoupling of pre-training (computing disagreement statistics Φ and hypothesis subset H_p) from deployment monitoring enables privacy-compliant auditing in regulated domains like healthcare, as the deployment phase requires only compressed statistics rather than raw training data (Algorithm 2, Section 3).

- **Transparent characterization of failure modes**: The paper explicitly identifies and analyzes Regime 2 (Theorem 4.5, Section 4.3.1), where deteriorating shifts occur but ε_q ≤ ε_p, leading to TPR of O(α). Rather than hiding this limitation, the authors provide Figure 3 illustrating how improving base classifier quality (lower ε_f) can mitigate this failure mode.

- **Clear formalization of D-PDD distinction**: Definition 1 vs. Definition 2 articulates the distinction between distribution shift and deteriorating shift, providing a useful theoretical framework. Lemma 2.1 establishes equivalence conditions between PDD and D-PDD under specified assumptions (g = g' and TV(P_x, Q_x) ≤ κ).

## Weaknesses

### Fatal
- **Empirical contradiction in primary healthcare results undermines core robustness claim**: The GEMINI temporal shift experiment contains a critical inconsistency between the text claims and Figure 5 results. Section 5.2 states the temporal shift is "non-deteriorating" (Fig. 5a shows stable AUROC) and claims D-PDDM is "least reactive to detection" with "small False Positive Rate." However, Figure 5(b) shows D-PDDM maintaining detection rates around 0.8-0.9 throughout the period. If the shift is genuinely non-deteriorating (as Fig. 5a indicates), then an 80-90% detection rate constitutes a massive false positive rate, directly contradicting the claim of robustness to non-deteriorating shifts. This inconsistency suggests either the metric is mislabeled (plotting shift detection rather than deterioration detection) or the method fails its primary design goal on the real-world healthcare scenario. The paper's central empirical contribution—demonstrating low FPR on non-deteriorating shifts in high-stakes domains—cannot be validated without resolving this contradiction.

### Major
- **Theory-experiment mismatch in VC dimension assumptions**: The theoretical guarantees (Theorems 4.2, 4.4) rely on sample complexity bounds involving the VC dimension d_p of the hypothesis class H_p, with Corollary 4.3 noting the bounds are meaningful for "linear and forest models among others of manageable VC-dimension" (line 223). However, Section 5.1 states experiments use "neural networks restricted to several layers of ≈32 hidden nodes each" (line 294). While the authors claim this respects "expressivity constraints," the VC dimension of even small neural networks can be extremely large depending on architecture, potentially rendering the theoretical bounds vacuous for the actual implementation. This disconnect between the theory's stated applicability (linear/forest models) and experimental practice (neural networks) undermines the "Provable" claim in the context of presented results.

- **Missing comparison against deterioration-specific baselines**: The paper compares D-PDDM primarily against distribution shift detectors (MMD-D, H-Div, JS-Div, KL-Div) to demonstrate lower false positive rates on non-deteriorating shifts. However, shift detectors are designed to detect any distribution shift regardless of performance impact, so naturally they will flag non-deteriorating shifts. The paper lacks comparison against unsupervised error estimation or deterioration-specific baselines that also operate without deployment labels (e.g., confidence score degradation, entropy monitoring, or methods from Rosenfeld & Garg 2023 adapted to store statistics). Without such baselines, the empirical superiority claim for *deterioration monitoring* specifically remains inadequately supported.

### Minor
- **Regime 2 mitigation relies on unrealistic assumptions for healthcare applications**: The paper's solution to the Regime 2 failure mode is to ensure the base classifier f has very low training error ε_f (Section 4.3.1, Figure 3). However, in high-stakes domains like healthcare—the paper's primary use case—models rarely achieve near-zero training error due to label noise, inherent uncertainty, and the complexity of clinical data. A monitoring system that theoretically fails whenever the base model is imperfect (the standard realistic scenario) represents a structural limitation. While the paper acknowledges this, proposing more robust alternatives (e.g., widening H_p, adaptive thresholds) would strengthen practical utility.

- **Resource overhead analysis absent**: Algorithm 2 requires storing and querying the hypothesis subset H_p during deployment to compute argmax_{h ∈ H_p}. If H_p consists of neural networks (as per experiments), this incurs significant memory and compute overhead compared to statistic-based baselines. The paper claims "efficiently scalable" deployment (line 35, Abstract) but provides no empirical analysis of inference time, memory footprint, or computational cost relative to baselines. Without this, the scalability claim remains unsubstantiated.

### Trivial
- **Figure 5 metric labeling ambiguity**: The y-axis in Figure 5(b) is labeled "TPR@5%" but the scenario is explicitly non-deteriorating. This creates confusion about whether the metric represents true positive rate (which would be undefined for non-deteriorating shifts) or alert rate/false positive rate. Clarifying the metric definition in the figure caption would improve interpretability.

## Nice-to-Haves
- **Disagreement distribution visualization**: Visualizing the distribution of disagreement rates Φ from training distribution P versus disagreement on deployment distribution Q for both deteriorating and non-deteriorating cases would reveal whether the theoretical "separation" actually exists in the healthcare data, strengthening empirical validation.

- **Discussion of concept shift assumption**: Lemma 2.1 assumes g = g' (no concept shift), which may be strong for post-deployment scenarios where labeling criteria evolve. A brief discussion of how D-PDDM might behave under mild concept shift would enhance practical guidance.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point on "Training data-free" claim**: The critic argued that requiring access to H_p during deployment contradicts "efficiently scalable" claims. However, the paper does decouple training data access from deployment—H_p is computed during pre-training and only the compressed subset is used during deployment. This is a valid architectural choice, not a contradiction. The resource overhead concern is retained as a Minor weakness, but the claim that this invalidates "training data-free" is removed.

- **Harsh Critic Point on Lemma 2.1 fragility**: The critic noted that if ε_f is 0.1, the equivalence probability bound (1 - 2ε_f - κ) drops significantly. However, this is a mathematical property of the bound, not an error. The paper acknowledges this through Regime 2 analysis. This is not a weakness but rather a transparent characterization of assumptions.

- **Strength Finder claim about GEMINI showing "low alert rate"**: The Strength Finder claimed Figure 5(b) shows D-PDDM maintaining "low alert rate" on GEMINI temporal shift. This directly conflicts with the figure description stating D-PDDM maintains "high TPR (around 0.8-0.9)." When a strength and weakness disagree on empirical facts, the weakness wins. This strength is removed as it appears to misread the figure.

- **Harsh Critic Point on baseline comparison being "tautological"**: While the comparison against shift detectors has limitations, it is not tautological—demonstrating that a deterioration monitor outperforms shift detectors on non-deteriorating data is meaningful for establishing the method's distinct purpose. The valid concern about missing deterioration-specific baselines is retained as a Major weakness, but the "tautological" framing is removed.

## Novel Insights
The paper's identification of Regime 2—where deteriorating shifts occur but the monitor fails because ε_q ≤ ε_p—provides genuine insight into the limitations of disagreement-based monitoring. The theoretical analysis showing that this failure mode can be mitigated by improving base classifier quality (lower ε_f) offers practical guidance, though the reliance on near-perfect base classifiers in high-stakes domains reveals a fundamental tension in monitoring system design. The explicit decoupling of pre-training and deployment phases to achieve training data-free monitoring is a useful architectural contribution for privacy-sensitive applications.

## Suggestions
1. **Resolve the GEMINI Figure 5 contradiction**: Clarify whether Figure 5(b) metrics represent detection rate (which would be FPR for non-deteriorating shifts) or actual true positive rate. If D-PDDM is detecting 80-90% of the time on non-deteriorating data, this fundamentally contradicts the low FPR claim and must be addressed. Consider re-running the experiment with explicit FPR reporting.

2. **Add deterioration-specific baselines**: Include comparisons against unsupervised error estimation methods (e.g., confidence score monitoring, entropy-based detection, or adapted versions of Rosenfeld & Garg 2023 that store statistics during pre-training) to validate that D-PDDM is genuinely better at deterioration monitoring, not just different from shift detection.

3. **Provide resource overhead analysis**: Report memory footprint, inference latency, and computational cost of storing and querying H_p during deployment compared to baselines. If H_p contains neural networks, quantify the overhead and discuss whether this is acceptable for real-time monitoring in healthcare settings.

4. **Temper theoretical claims or align experiments**: Either acknowledge that the VC dimension-based bounds may be loose for the neural network architectures used, or conduct additional experiments with linear/forest models to demonstrate the theory-experiment alignment. The "Provable" claim should be qualified if the experimental setup does not satisfy the theoretical assumptions.

5. **Discuss Regime 2 alternatives**: Propose or discuss methods to widen H_p or adjust thresholds dynamically to detect deterioration even when ε_f is non-negligible, rather than relying solely on "train f better" as the solution.

## Calibration and Scoring

I compared this paper against several calibration anchors:

**High-scoring anchors (avg ≥ 6):**
- VwCyRQJ51H (6.00): Provides theoretical guarantees on disagreement discrepancy with consistent empirical validation; no theory-experiment mismatch.
- FnbGlnKbIU (5.50): Strong theory for anomaly detection with synthetic experiments that validate theoretical findings, though limited empirical scope.
- pXw0uRTSKT (6.00): Healthcare time-series representations with solid empirical evaluation across three hospital cohorts.

**Medium-scoring anchors (avg ~5):**
- hFxivbAgVP (5.33): Simple empirical method for LLM monitoring; reviewers noted limited technical novelty but accepted for practical utility.
- hyI8cIOU2f (5.00): Strong formalization but empirical weaknesses in analysis depth; ultimately rejected.

**Low-scoring anchors (avg ≤ 4):**
- Kk08XcQCl2 (3.00): Explicit "significant gap between theory and practice" where generalization bounds are too loose; rejected.
- I3spHvRHqo (4.00): Claims non-vacuous bounds for deep learning but reviewers questioned novelty and clarity; rejected.
- 3m3E1TEjtb (3.33): Theory claims O(1/√B) scaling but Figure 2(d) does not support this empirically; withdrawn.
- WvRmaSD2QV (3.00): Central claim "model editing is over" not supported by empirical evidence on only 2 small models; rejected.

**Positioning:** This paper shares critical flaws with the low-scoring anchors: (1) a theory-experiment mismatch similar to Kk08XcQCl2 and I3spHvRHqo where theoretical bounds assume manageable VC dimension but experiments use neural networks, and (2) an empirical contradiction in the primary healthcare results similar to 3m3E1TEjtb and WvRmaSD2QV where figure data does not support the text claims. The GEMINI contradiction is particularly severe—if D-PDDM flags non-deteriorating shifts 80-90% of the time, it fails its core design goal, analogous to WvRmaSD2QV's unsupported central claim.

However, the paper has genuine strengths: clear problem formalization, training data-free architecture, and transparent Regime 2 characterization. The CIFAR-10.1 results appear solid. These prevent it from falling to the 2-3 range of papers with completely unsupported claims.

Compared to hyI8cIOU2f (5.00, Reject), which had strong formalization but empirical weaknesses, this paper's empirical contradiction is more severe (undermining the core claim rather than just limiting analysis depth). Compared to FnbGlnKbIU (5.50, Accept), which had strong theory validated by appropriate synthetic experiments, this paper's theory-experiment mismatch is a significant flaw.

The paper's fatal empirical contradiction and major theory-experiment mismatch place it below the medium-scoring anchors. It aligns most closely with 3m3E1TEjtb (3.33) and WvRmaSD2QV (3.00), but the clearer problem formalization and solid CIFAR-10.1 results justify a slightly higher score than those.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>