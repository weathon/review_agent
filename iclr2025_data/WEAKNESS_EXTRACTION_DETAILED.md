# Detailed Weakness Extraction for SFT Module-Level Fine-Tuning Paper

This document provides exact quotes from calibration papers paired with specific applications to your SFT paper.

---

## CATEGORY 1: SCOPE OF EVALUATION CRITICISM

### 1.1 "Only Two Domains" Problem

**Source:** LLM Unlearning under the Microscope (cal/llm_unlearning_under_the_microscope_a_fullstack_view_on_meth_review.md)

**Exact Quote:**
> "All experiments use a single model (Llama-3 8B Instruct) and primarily a single benchmark (WMDP-Bio). Unlearning dynamics are known to change with model scale—larger models may exhibit different forgetting-retention tradeoffs—and the WMDP-Bio domain (hazardous knowledge removal) may not be representative of copyright (MUSE), privacy (TOFU), or other unlearning scenarios. The paper acknowledges this in limitations but does not provide any evidence that the family-level rankings (e.g., 'representation misalignment generally outperforms rejection-based') hold beyond this narrow setting."

**Direct Application to SFT Paper:**
- GeneralPoints + V-IRL are both rule-based reasoning
- Counterargument needed: Show that module-level properties aren't just artifacts of these specific domains
- What could be different: Do attention/FNN have same modularity gap on semantic reasoning? Mathematical reasoning? Multi-hop reasoning?

**Anticipated Reviewer Statement:**
"The paper evaluates module-level fine-tuning on only two rule-based reasoning benchmarks. It's unclear whether the observed attention vs FNN differences are fundamental properties or artifacts of these specific evaluation domains. Broader evaluation would be needed to support the generality of the claims."

---

### 1.2 "Missing Coverage of Available Benchmarks" Problem

**Source:** Same paper

**Exact Quote:**
> "Only WMDP-Bio is evaluated; WMDP-Cyber and WMDP-Chem are omitted. Since the paper adopts WMDP as its primary benchmark, leveraging all three available domains would test whether the observed family-level patterns are consistent across different knowledge types."

**Direct Application to SFT Paper:**
- GeneralPoints and V-IRL are your "WMDP-Bio"
- What's missing: Other rule-based reasoning benchmarks? Other reasoning types?
- Specific examples: ARC, CommonsenseQA, logical reasoning benchmarks, etc.

**Reviewer's Logic:**
If you claim "selective module fine-tuning helps across benchmarks," you need to show results on multiple benchmarks. Two benchmarks is not "across benchmarks."

---

### 1.3 "Parameter/Architecture Not Tested Across Variants" Problem

**Source:** Are LLMs Exploitable Negotiators (cal/are_llms_exploitable_negotiators_review.md)

**Exact Quote:**
> "Limited model scale. All tested models are 7B-parameter models. While the systematic failures observed are meaningful, the generalizability to larger or more capable models is unknown. Larger models with stronger reasoning capabilities may exhibit different strategic behavior. This limits the scope of the conclusions."

**Direct Application to SFT Paper:**
- What model(s) did you test on? If only one size, this is a weakness
- Concern: Do attention/FNN modularity properties hold at different scales?
- At 7B parameters: FNN might be more modular (undertrained)
- At 70B parameters: Attention might be more specialized (leading to same conclusion or opposite?)

**Reviewer's Logic:**
"They found X property on 7B model. Does it hold on 13B? 70B? We don't know. The conclusion might be model-size-specific."

---

## CATEGORY 2: FAIR COMPARISON ISSUES

### 2.1 "Different Experimental Conditions for Different Methods" Problem

**Source:** wd1: Weighted Policy Optimization (cal/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_review.md)

**Exact Quote:**
> "Hardware inconsistency for cost comparisons involving wd1++. Table 2 reports wd1 vs. d1 costs on 4×A100 (fair comparison), but Table 3 (right) and Appendix B.6 indicate wd1++ was trained on 8×A800 while d1 used 4×A100. The '10× fewer rollouts' claim is valid as a sample-efficiency metric, but wall-clock time and total FLOP comparisons for wd1++ vs. d1 are not provided on equal hardware."

**Direct Application to SFT Paper:**
- Did you train full fine-tuning with the same learning rate schedule as component fine-tuning?
- Did FNN-only and attention-only get the same hyperparameter search budget?
- Can you honestly compare them if one got 100 LR values searched and another got 10?

**Critical Questions:**
- What's the learning rate for full fine-tuning?
- What's the learning rate for FNN-only? Attention-only?
- Are these chosen fairly (e.g., grid search on all equally?)
- If FNN-only improves by 1% with LR=0.001 and attention-only with LR=0.0001, is that a real component difference or just different optimal learning rates?

**Reviewer's Likely Statement:**
"The paper compares full fine-tuning, FNN-only, and attention-only but doesn't specify whether these were tuned with equal computational budgets. If FNN-only got more hyperparameter tuning, the comparison is unfair. [Specifying hyperparameter search procedures for all baselines]"

---

### 2.2 "Parameter Count Confound" Problem

**Source:** Shaping Monotonic Neural Networks with Constrained Learning (cal/shaping_monotonic_neural_networks_with_constrained_learning_review.md)

**Exact Quote:**
> "Parameter count asymmetry with LMN (37 params vs. 2069): While the reviewer flagged this as an unfair comparison, LMN's small parameter count is a feature of its specialized architecture. The proposed method still achieves only marginally better accuracy (69.4% vs. 69.3% on COMPAS) despite vastly more parameters, which actually highlights LMN's parameter efficiency rather than unfairly favoring the proposed method."

**Direct Application to SFT Paper:**
- FNN-only = how many parameters? Attention-only = how many?
- If FNN-only tunes 20% of parameters and attention-only tunes 10%, they're not comparable
- Marginal improvement (1-2%) with different parameter budgets = suspicious

**Example Vulnerability:**
- Full fine-tuning: 100% params tuned → 65% accuracy
- FNN-only: 40% params tuned → 64.5% accuracy
- Attention-only: 20% params tuned → 62% accuracy
- Reviewer: "FNN with 40% of params nearly matches full tuning. But attention with 20% of params drops much more. This could mean attention is just undertrained, not that it's less modular."

---

## CATEGORY 3: HYPERPARAMETER TUNING DISGUISED AS CONTRIBUTION

### 3.1 "Learning Rate Sensitivity Is Not a Contribution" Pattern

**Source:** Shaping Monotonic Neural Networks with Constrained Learning (cal/shaping_monotonic_neural_networks_with_constrained_learning_review.md)

**Exact Quote:**
> "Run a systematic ablation varying α ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5} on at least two datasets, reporting both prediction performance and the measured monotonicity violation rate on a held-out test set or dense grid. This directly validates the paper's primary claimed contribution."

**Direct Application to SFT Paper:**
- If your main finding is "attention needs LR=0.001, FNN needs LR=0.0005" → that's hyperparameter tuning, not a contribution
- But if you show "attention has larger loss surface curvature" or "FNN learns faster then saturates" → that's mechanistic
- Reviewers want to see: WHY different learning rates, not just THAT they're different

**Required Evidence to Distinguish Contribution from Tuning:**
1. Show learning rate sweep curves for each component
2. Explain the shape of the curve (sharp learning cliff vs. smooth plateau?)
3. Does this pattern hold across multiple benchmarks?
4. Can you predict the difference from first principles?

**Reviewer's Likely Statement:**
"While the paper observes that attention modules and FNNs have different optimal learning rates, it's unclear whether this is a fundamental property or simply hyperparameter tuning. To constitute a contribution, the paper should explain *why* these components differ in their learning dynamics (e.g., gradient flow analysis, loss landscape properties, learning rate sensitivity) rather than just reporting the empirical optimal values."

---

### 3.2 "Cherry-Picked Best Results" Problem

**Source:** Shaping Monotonic Neural Networks with Constrained Learning (cal/shaping_monotonic_neural_networks_with_constrained_learning_review.md)

**Exact Quote:**
> "Report results over all 10 random seeds (not best 5 of 10), or at minimum include both statistics side-by-side, so readers can assess the method's true variance."

**Direct Application to SFT Paper:**
- Did you run each condition (full fine-tuning, FNN-only, attention-only) multiple times?
- If you have error bars, are they symmetric or suspiciously narrow?
- If differences are small (1-2%), they could be within noise

**Concrete Example of Problem:**
If you show:
- Full FT: 65.0 ± 0.3%
- FNN-only: 64.8 ± 0.2%
- Attention-only: 63.2 ± 0.5%

Reviewer might say: "FNN vs Full FT difference is within the confidence intervals. The claimed modularity of FNNs is not statistically significant."

**Required:**
- Multiple seeds (at least 5, preferably 10)
- Statistical significance testing (t-tests between conditions)
- Large enough differences relative to variance

---

## CATEGORY 4: MODULE/COMPONENT ANALYSIS - BLACK BOX EXPLANATIONS

### 4.1 "No Ablation Isolating Each Component" Problem

**Source:** DualRes: A Resampling-Based Framework (cal/dualres_a_resamplingbased_framework_for_enhancing_probabilis_review.md)

**Exact Quote:**
> "No ablation isolating the contribution of volatility modeling from resampling. The paper attributes improvements to both conditional heteroskedasticity modeling and residual distribution capture, but never tests what happens with only volatility modeling (Gaussian residuals with learned volatility) or only resampling (homoskedastic residuals with bootstrap). Without this ablation, it is unclear which component drives the gains, undermining the paper's core claim that both components are essential."

**Direct Application to SFT Paper:**

Your paper claims: "Selective fine-tuning of attention and FNNs separately improves performance"

Missing ablations:
1. Full fine-tuning (baseline) - done
2. Attention-only - done
3. FNN-only - done
4. **MISSING: Attention-only + FNN-only tuned *independently* (with separate learning rates) vs. tuned together**
5. **MISSING: Attention-only with same learning rate as FNN-only to isolate component quality**

**Specific Ablation Design Needed:**
```
Baseline: Full FT with LR=0.0001 → 65%

Component Isolation:
A1: Attention-only, LR=0.0001 → 62%
A2: Attention-only, LR-tuned → 63%
F1: FNN-only, LR=0.0001 → 64.5%
F2: FNN-only, LR-tuned → 64.8%

The question: Is improvement from separate tuning, or from finding better component-specific LRs?
```

**Reviewer's Likely Statement:**
"While the paper shows FNN-only performs better than attention-only, it's unclear whether this is due to FNNs being more modular or simply due to FNNs responding better to the specific learning rate used. Ablations with fixed learning rates across components would clarify this."

---

### 4.2 "Mechanistic Claims Without Functional Validation" Problem

**Source:** Dens3r (implied from DualRes and other reviews about interpretability)

**Exact Quote (synthesized pattern):**
> "While the learned weights correlate with [measure], this is an emergent property, not a design guarantee. Without demonstrating that the learned properties are *functionally* important (e.g., ablating the high-importance components hurts more than ablating low-importance ones), the interpretability claim remains correlational rather than causal."

**Direct Application to SFT Paper:**

Your paper claims: "Attention modules and FNNs have different learning dynamics"

Missing functional validation:
- Do attention module changes affect learned attention patterns? (Test: compare attention weights before/after)
- Do FNN module changes affect learned representations? (Test: CCA, representation similarity analysis)
- Are these changes actually beneficial for the task? (Test: zero out updated components, measure performance drop)

**Concrete Validation Approach:**
1. Measure attention patterns in original model
2. After fine-tuning attention-only: have attention patterns changed?
3. After fine-tuning FNN-only: have attention patterns stayed the same?
4. If attention patterns didn't change much → claim about "attention modularity" is weak

**Reviewer's Likely Statement:**
"The paper observes that tuning FNNs improves performance more than tuning attention modules. However, there's no mechanistic evidence that this reflects different modular properties. Functional ablations (e.g., show that attention features are transferable while FNN features are task-specific) would strengthen the claims."

---

## CATEGORY 5: GENERALIZATION CLAIMS - SKEPTICISM ON OOD

### 5.1 "Synthetic/Clean Domain → Generalization Unclear" Problem

**Source:** Dens3r - 3D Geometry Prediction

**Exact Quote:**
> "Cross-dataset generalization (synthetic-only → real): Train on Type A/B synthetic data only and test on real-world benchmarks without fine-tuning to evaluate true transfer capability as a foundation model."

**Direct Application to SFT Paper:**

GeneralPoints and V-IRL are:
- **Rule-based**: Clean, unambiguous correct answers
- **Structured**: Grammar/syntax clear
- **Predictable**: Model can learn the rules

Real reasoning tasks:
- Natural language ambiguity (same question, multiple valid interpretations)
- Context dependence (same question, different answers in different contexts)
- Incomplete information (real-world missing data)

**Reviewer's Likely Statement:**
"While the paper shows module-level fine-tuning benefits on rule-based reasoning benchmarks, it's unclear whether these benefits transfer to more realistic reasoning tasks with ambiguity and context dependence. Evaluation on at least one naturalistic reasoning benchmark would strengthen the generalization claims."

**Missing Evaluations:**
- Semantic reasoning (requires understanding meaning, not just rules)
- Common sense reasoning (requires world knowledge)
- Open-ended QA (where multiple answers are valid)

---

### 5.2 "Single Model Scale → Generalization Unclear" Problem

**Source:** Are LLMs Exploitable Negotiators (cal/are_llms_exploitable_negotiators_review.md)

**Exact Quote:**
> "Limited model scale. All tested models are 7B-parameter models. While the systematic failures observed are meaningful, the generalizability to larger or more capable models is unknown. Larger models with stronger reasoning capabilities may exhibit different strategic behavior. This limits the scope of the conclusions."

**Direct Application to SFT Paper:**

Test on what size model?
- If 7B: Does modularity hold at 13B? 70B? 130B?
- Hypothesis: At small scales, FNNs might be undertrained and benefit from focused tuning
- At large scales, FNNs might be overparameterized, making them less modular

**Reviewer's Likely Statement:**
"The paper evaluates module-level fine-tuning on a [MODEL_SIZE] model. It's unclear whether the observed attention/FNN modularity gap persists across different model scales. Evaluation on at least [LARGER_SIZE] and [SMALLER_SIZE] would strengthen the claims about fundamental module properties."

**Required Minimum:**
Show results at 2 different scales (e.g., 7B and 70B)

---

## CATEGORY 6: SYNTHESIS - WHAT REVIEWERS WILL SAY ABOUT YOUR PAPER

Based on the calibration examples, here's a prediction of reviewer consensus:

### Major Weaknesses (likely):
1. "Limited to rule-based reasoning benchmarks; generalization to other domains unclear"
2. "No ablations isolating which component improvements come from modularity vs. better learning rates"
3. "Marginal improvements could be within noise; statistical significance unclear"
4. "Single model evaluation; unclear if findings hold across model scales"

### Minor Weaknesses (likely):
1. "Hyperparameter search procedure not fully specified for all baselines"
2. "Missing mechanistic analysis of why attention and FNNs differ"
3. "Parameter count differences between components not discussed"
4. "Learning rate curves not shown; unclear which LRs were chosen and why"

### Questions Reviewers Will Ask:
1. "What happens if you train attention-only and FNN-only with the same learning rate?"
2. "Do attention patterns actually change during attention-only fine-tuning?"
3. "Have you tested on benchmarks beyond rule-based reasoning?"
4. "Are the improvements statistically significant given the variance?"
5. "Does the modularity distinction hold at different model scales?"

---

## APPENDIX: DIRECT QUOTES FOR CITATION

If you're strengthening your paper, these exact quotes show what reviewers expect:

### For justifying broader evaluation:
> "The paper acknowledges this in limitations but does not provide any evidence that the family-level rankings hold beyond this narrow setting."

### For justifying component ablations:
> "Without this ablation, it is unclear which component drives the gains, undermining the paper's core claim."

### For justifying mechanistic analysis:
> "Without demonstrating that the learned properties are functionally important, the interpretability claim remains correlational rather than causal."

### For justifying variance reporting:
> "Report results over all seeds, so readers can assess the method's true variance."

### For justifying multi-scale evaluation:
> "The generalizability to larger or more capable models is unknown... This limits the scope of the conclusions."
