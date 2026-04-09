# Specific Weaknesses Found in Calibration Papers - Highly Relevant to SFT Module-Level Fine-Tuning Paper

## 1. SCOPE OF EVALUATION - Limited Benchmarks/Narrow Task Domains

### Example 1: LLM Unlearning Paper (CALIBRATION EXAMPLE 12)
**Exact Quote:**
> "Limited empirical scope restricts generalizability of the claimed family-level behaviors. All experiments use a single model (Llama-3 8B Instruct) and primarily a single benchmark (WMDP-Bio). Unlearning dynamics are known to change with model scale—larger models may exhibit different forgetting-retention tradeoffs—and the WMDP-Bio domain (hazardous knowledge removal) may not be representative of copyright (MUSE), privacy (TOFU), or other unlearning scenarios."

**Paper ID:** LLM Unlearning under the Microscope

**How it applies to SFT paper:**
- Your paper uses only **two benchmarks** (GeneralPoints and V-IRL), both rule-based reasoning tasks
- Criticism: Cannot generalize from two narrow domains to claim method works broadly
- Recommendation: Need evaluation on diverse benchmark domains (not just rule-based reasoning)
- Similar concern: Single model (likely), single benchmark pair limits generalizability claims

---

### Example 2: LLM Unlearning Paper - Continued
**Exact Quote (same paper):**
> "Only WMDP-Bio is evaluated; WMDP-Cyber and WMDP-Chem are omitted. Since the paper adopts WMDP as its primary benchmark, leveraging all three available domains would test whether the observed family-level patterns are consistent across different knowledge types."

**Paper ID:** LLM Unlearning under the Microscope

**How it applies to SFT paper:**
- Your paper evaluates components (attention vs FNN split) only on rule-based reasoning
- Criticism: Scope too narrow to claim the attention/FNN distinction applies to other domains
- Direct parallel: Just as WMDP-Bio ≠ all unlearning scenarios, GeneralPoints/V-IRL ≠ all reasoning tasks
- Missing: Evaluation on mathematical reasoning, language understanding, multi-hop reasoning, etc.

---

### Example 3: LLM Unlearning Paper - Quantization Claim
**Exact Quote:**
> "Quantization robustness generalization from only two methods. The claim that 'knowledge removal is generally more robust to post-unlearning quantization than data-centric unlearning' (Table A2) is based solely on NPO and RMU. Extrapolating to entire method families from two data points is not well-supported."

**Paper ID:** LLM Unlearning under the Microscope

**How it applies to SFT paper:**
- Your paper compares **three methods**: full fine-tuning, FNN-only, attention-only (just 3 points)
- Criticism: Cannot claim "attention modules are more modular than FNN" from 3 comparisons across 2 benchmarks
- The reviewers explicitly reject extrapolating family-level claims from tiny sample sizes
- Your claim about attention vs FNN modularity rests on very limited evidence

---

## 2. FAIR COMPARISON ISSUES - Different Hyperparameters, Cherry-Picking

### Example 1: Hardware/Compute Inequality
**Exact Quote (wd1 paper):**
> "Hardware inconsistency for cost comparisons involving wd1++. Table 2 reports wd1 vs. d1 costs on 4×A100 (fair comparison), but Table 3 (right) and Appendix B.6 indicate wd1++ was trained on 8×A800 while d1 used 4×A100. The '10× fewer rollouts' claim is valid as a sample-efficiency metric, but wall-clock time and total FLOP comparisons for wd1++ vs. d1 are not provided on equal hardware."

**Paper ID:** wd1: Weighted Policy Optimization for Reasoning in Diffusion

**How it applies to SFT paper:**
- Your baselines: full fine-tuning, FNN-only, attention-only
- Criticism point: Were all methods trained with identical hyperparameters? Learning rates? Optimization schedules?
- Risk: Different learning rates for different components could favor one variant
- Reviewer concern: Small differences could be just hyperparameter tuning artifacts, not real insights

---

### Example 2: Parameter Count Asymmetry
**Exact Quote (Monotonic Neural Networks paper):**
> "Parameter count asymmetry with LMN (37 params vs. 2069): While the reviewer flagged this as an unfair comparison, LMN's small parameter count is a feature of its specialized architecture. The proposed method still achieves only marginally better accuracy (69.4% vs. 69.3% on COMPAS) despite vastly more parameters, which actually highlights LMN's parameter efficiency rather than unfairly favoring the proposed method."

**Paper ID:** Shaping Monotonic Neural Networks with Constrained Learning

**How it applies to SFT paper:**
- If FNN-only has different parameter count than attention-only, this confounds results
- Marginal improvements (e.g., 1% difference) with different parameter budgets = hyperparameter tuning, not a contribution
- Reviewers scrutinize when parameter counts differ; you need to explain this carefully

---

## 3. LEARNING RATE AS CONFOUNDER - Hyperparameter Tuning Disguised as Contribution

### Example 1: Emphasis on Ablations Across Parameter Ranges
**Exact Quote (Monotonic Neural Networks paper):**
> "Run a systematic ablation varying α ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5} on at least two datasets, reporting both prediction performance and the measured monotonicity violation rate on a held-out test set or dense grid. This directly validates the paper's primary claimed contribution."

**Paper ID:** Shaping Monotonic Neural Networks with Constrained Learning

**How it applies to SFT paper:**
- Your contribution: "We discovered that tuning learning rates for attention and FNN separately helps"
- Reviewer's criticism pattern: If the main finding is **learning rate sensitivity**, that's just hyperparameter tuning
- Defense needed: Show that attention and FNN have fundamentally different optimal learning rates (not just quantitatively different optimization curves)
- Risk: "This is just finding the right hyperparameter for each component" vs. "This reveals the different learning dynamics of different modules"

---

### Example 2: Variance Reporting
**Exact Quote (Monotonic Neural Networks paper):**
> "Report results over all 10 random seeds (not best 5 of 10), or at minimum include both statistics side-by-side, so readers can assess the method's true variance."

**Paper ID:** Shaping Monotonic Neural Networks with Constrained Learning

**How it applies to SFT paper:**
- Reviewer concern: If you cherry-pick best runs, differences could be within noise
- Criticism anticipation: "They ran this many times and showed the best results. That's not a real contribution."
- Defense needed: Report variance, statistical significance, multiple seeds across different random initializations

---

## 4. MODULE/COMPONENT ANALYSIS - Lack Mechanistic Understanding

### Example 1: Component Contribution Uncertainty
**Exact Quote (DualRes paper):**
> "No ablation isolating the contribution of volatility modeling from resampling. The paper attributes improvements to both conditional heteroskedasticity modeling and residual distribution capture, but never tests what happens with only volatility modeling (Gaussian residuals with learned volatility) or only resampling (homoskedastic residuals with bootstrap). Without this ablation, it is unclear which component drives the gains, undermining the paper's core claim that both components are essential."

**Paper ID:** DualRes: A Resampling-Based Framework for Enhancing Probabilistic Forecasting

**How it applies to SFT paper:**
- Your claim: "Attention modules and FNNs have different modular properties"
- Reviewer's would ask: Is the improvement from better attention tuning? Better FNN tuning? Or their interaction?
- Missing ablations:
  - What if you only tune attention with the same learning rate as full fine-tuning?
  - What if you only tune FNN?
  - How much does each contribute independently?
- Current risk: Your results could just show "different components respond to different learning rates" without proving modularity

---

### Example 2: Interpretability Claims Without Evidence
**Exact Quote (Dens3r paper, from field):**
> "Causality Embedding nomenclature and framing risk overstating the contribution: The term 'Causality Embedding' implies the module discovers causal relationships, but it is trained purely on prediction loss. The correlation with Granger-Geweke statistics (Figure 7) is an emergent property, not a design guarantee... Without demonstrating that the learned weights are *functionally* important (e.g., masking high-weight covariates hurts more than masking low-weight ones), the interpretability claim remains correlational rather than causal."

**Paper ID:** Dens3r - 3D Geometry Prediction (synthesized from patterns)

**How it applies to SFT paper:**
- Danger: Claiming "attention modules learn X, FNNs learn Y" without mechanistic validation
- Reviewer would ask: Have you shown that attention module changes actually affect attention patterns?
- Functional test needed: If you ablate a trained attention module, does performance drop? Does it affect the attention weights the model learns?
- Your current framing might be just naming components that happen to have different learning rates

---

## 5. GENERALIZATION CLAIMS - Skepticism on OOD/Strong Generalization

### Example 1: Synthetic-to-Real Generalization
**Exact Quote (Dens3r paper):**
> "Cross-dataset generalization (synthetic-only → real): Train on Type A/B synthetic data only and test on real-world benchmarks without fine-tuning to evaluate true transfer capability as a foundation model."

**Paper ID:** Dens3r - 3D Geometry Prediction

**How it applies to SFT paper:**
- Your benchmarks (GeneralPoints, V-IRL) are rule-based reasoning = synthetic/structured
- Reviewer concern: Can the method transfer to **real** reasoning tasks (e.g., natural language, ambiguity, context)?
- Challenge: Your paper shows improvements on clean, rule-based tasks. Will this transfer to messy real-world reasoning?
- Needed: Evaluation on at least one "noisy" or more realistic reasoning benchmark

---

### Example 2: Out-of-Domain Robustness Skepticism
**Exact Quote (LLM Unlearning paper):**
> "The finding that divergence-driven methods are more robust to in-domain relearning (RobReL) but less robust to out-of-domain fine-tuning (RobFT), while representation misalignment methods show the opposite pattern (Figure 2), is a nuanced and actionable insight."

**Paper ID:** LLM Unlearning under the Microscope

**How it applies to SFT paper:**
- Your finding: "Attention modules and FNNs have different optimal tuning"
- Reviewer skepticism: Does this hold OOD? On different model architectures? Different sizes?
- The pattern you found might be specific to your evaluation setup (specific model, specific benchmarks)
- Rule-based reasoning is a "comfortable" domain; real generalization would be broader

---

### Example 3: Limited Model Coverage
**Exact Quote (Are LLMs Exploitable Negotiators paper):**
> "Limited model scale. All tested models are 7B-parameter models. While the systematic failures observed are meaningful, the generalizability to larger or more capable models is unknown. Larger models with stronger reasoning capabilities may exhibit different strategic behavior. This limits the scope of the conclusions."

**Paper ID:** Are LLMs Exploitable Negotiators?

**How it applies to SFT paper:**
- Likely criticism: You tested on one model size/family
- Reviewer would ask: Does the attention/FNN distinction hold at different scales?
- Larger models might have different module specialization than smaller ones
- Needed: At minimum, show results at 2-3 different model scales

---

## 6. SUMMARY OF KEY REVIEWER CONCERNS APPLIED TO YOUR SFT PAPER

### Predicted Main Criticisms:

1. **Scope Too Narrow**: Two benchmarks (both rule-based) → Cannot claim general principle
   - *Fix*: Add 2-3 more benchmarks spanning different reasoning types

2. **Hyperparameter vs. Real Contribution**: Maybe attention just needs different learning rate
   - *Fix*: Show learning rate differences are not the sole driver; demonstrate mechanistic differences

3. **Weak Comparisons**: Only 3 methods, possibly different parameter counts/learning rates
   - *Fix*: Ensure fair comparison (same params, same LR search, report variance across seeds)

4. **Missing Ablations**: Which component (attention vs FNN) drives improvements?
   - *Fix*: Add component-level ablations isolating attention-only improvements vs FNN-only

5. **Limited Generalization**: Works on rule-based reasoning, unclear on other domains
   - *Fix*: Show at least one additional benchmark type; test on larger model

---

## 7. EXAMPLES OF CRITICISMS THAT WERE KEPT vs. REMOVED

### What Reviewers KEPT as Valid Weaknesses:
- Limited scope across benchmark domains
- Missing ablations for component contributions
- Unclear mechanistic explanations
- Generalization to other settings unproven
- Hyperparameter tuning without proper validation

### What Reviewers REMOVED as Invalid:
- "Missing related works" (cannot verify existence)
- "No code released" (implementation detail)
- "Should compare with X method" (if outside scope)
- "Should use different evaluation metric" (if original is justified)
- Formatting/notation nitpicks

---

## Key Takeaway for Your SFT Paper

The calibration reviews show that **component-level claims (like "attention modules are more modular than FNNs")** require:

1. **Evaluation breadth**: Multiple benchmarks, not just 2 narrow domains
2. **Mechanistic evidence**: Not just performance numbers, but functional evidence that components behave differently
3. **Fair comparisons**: Identical hyperparameter search, parameter counts, seeds reported
4. **Ablation clarity**: Which improvements come from each component?
5. **Generalization demonstration**: Beyond rule-based reasoning to at least one other domain

Your current paper likely scores 2-4 on these dimensions, but reviewers will expect 4-5 for acceptance.
