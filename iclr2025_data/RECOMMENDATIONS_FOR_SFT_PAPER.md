# Actionable Recommendations for SFT Module-Level Fine-Tuning Paper

Based on analysis of 50+ calibration review papers, here are specific, evidence-based recommendations for strengthening your SFT paper.

---

## PRIORITY 1: BROADEN EVALUATION SCOPE

### Current State:
- 2 benchmarks: GeneralPoints, V-IRL (both rule-based reasoning)

### Recommended Additions (in order of effort/impact):

#### 1a. Add 2-3 Different Benchmark Types (REQUIRED for acceptance)

**Type 1: Commonsense Reasoning**
- Example: CommonsenseQA, Social IQa, HellaSwag
- Why: Tests whether module modularity holds beyond rule-based tasks
- Expected work: 2-3 experiments

**Type 2: Mathematical Reasoning**
- Example: GSM8K, MATH, ARC
- Why: Different structure from rule-based (requires numeric reasoning)
- Expected work: 2-3 experiments

**Type 3: Open-ended/Semantic Reasoning**
- Example: Natural Questions, TriviaQA, MMLU
- Why: Tests on non-rule-based domains with real-world ambiguity
- Expected work: 3-5 experiments

**Justification from calibration papers:**
"All experiments use a single model and primarily a single benchmark... the paper does not provide any evidence that the family-level rankings hold beyond this narrow setting." (LLM Unlearning paper)

---

#### 1b. Add Multi-Scale Model Evaluation (RECOMMENDED)

**Current:** Test on one model size (presumably 7B?)

**Recommended:** Test on 2-3 sizes
- 7B or similar (small)
- 13B-30B (medium)
- 70B+ (large)

**Why:** Modularity might be size-dependent
- At 7B: FNN might be undertrained → easy to improve with focused tuning
- At 70B: FNN might be overparameterized → attention tuning might help more

**Expected outcome:** Adds 1-2 figures, 1 paragraph discussion

**Justification from calibration papers:**
"While the systematic failures observed are meaningful, the generalizability to larger or more capable models is unknown. Larger models with stronger reasoning capabilities may exhibit different strategic behavior." (Are LLMs Exploitable Negotiators)

---

## PRIORITY 2: ADD MECHANISTIC ABLATIONS

### Current State:
- Baseline: Full fine-tuning
- Condition A: Attention-only
- Condition B: FNN-only

### Recommended Additions:

#### 2a. Fixed Learning Rate Ablation (ESSENTIAL)

**Current problem:** Each component might be tuned with its own optimal LR
- FNN-only with LR=0.0005 → 64.8%
- Attention-only with LR=0.001 → 62%
- Is the FNN better, or just better-tuned?

**Ablation design:**
```
Experiment 1: Fair Component Comparison (all with same LR)
├─ Full FT: LR=0.001 → 65.0% ± 0.3
├─ FNN-only: LR=0.001 → 64.7% ± 0.2
├─ Attention-only: LR=0.001 → 62.9% ± 0.4

Experiment 2: Component-Specific Tuning (each gets optimal LR)
├─ Full FT: LR=0.001 → 65.0% ± 0.3
├─ FNN-only: LR=0.0005 → 64.8% ± 0.2
├─ Attention-only: LR=0.002 → 63.4% ± 0.4
```

**Interpretation:**
- If Exp1 shows FNN and Attention similar → findings are just LR artifacts
- If Exp2 shows bigger gaps → there's real component difference in optimization landscape

**Expected work:** 1-2 new experiments, 1 table, 1 paragraph

**Justification from calibration papers:**
"Without this ablation, it is unclear which component drives the gains, undermining the paper's core claim that both components are essential." (DualRes)

---

#### 2b. Learning Rate Curves (IMPORTANT)

**Current state:** Probably just reporting one LR per condition

**Recommended:** Show learning rate sensitivity curves
```
For each condition (Full FT, FNN-only, Attention-only):
├─ x-axis: Learning Rate (log scale: 1e-5 to 1e-2)
└─ y-axis: Performance (GeneralPoints, V-IRL)

Plot expected curves:
├─ Full FT: Smooth curve, single clear peak
├─ FNN-only: Steep curve (sharp optimum)?
└─ Attention-only: Broad plateau?
```

**Why this matters:**
- If FNN has sharp peak and Attention has broad plateau → real difference in optimization landscape
- This is mechanistic evidence, not just "FNN better by 1%"

**Expected work:** 1 figure (3 subplots), 1 paragraph

**Justification from calibration papers:**
"Run a systematic ablation varying α... reporting both prediction performance and the measured [metric] on a held-out test set or dense grid. This directly validates the paper's primary claimed contribution." (Monotonic NNs)

---

#### 2c. Component Update Magnitude Analysis (IMPORTANT)

**What to measure:**
After fine-tuning, how much did each component actually change?
```
For Attention-only FT:
├─ L2 norm of attention weight changes: 0.34
├─ L2 norm of FNN weight changes: 0.02 (should be small!)
└─ Rank of attention parameter updates: 95% of variance in top 10%

For FNN-only FT:
├─ L2 norm of FNN weight changes: 0.41
├─ L2 norm of attention weight changes: 0.01 (should be small!)
└─ Rank of FNN parameter updates: 87% of variance in top 10%
```

**Why this matters:**
- Proves that attention-only tuning actually isolates attention updates
- Shows which component can be updated with less gradient flow (more modular)
- Addresses "black box" complaint

**Expected work:** 1 table, 1 paragraph

**Justification from calibration papers:**
Addresses concern: "Without demonstrating that the learned properties are functionally important (e.g., ablating the high-importance components hurts more than ablating low-importance ones), the interpretability claim remains correlational rather than causal."

---

## PRIORITY 3: STATISTICAL RIGOR

### Current State:
Probably missing: error bars, significance tests, or unclear variance

### Recommended Additions:

#### 3a. Multiple Seeds with Reporting

**Standard practice:**
- Run each condition with 5-10 random seeds
- Report mean ± std dev OR confidence intervals
- Example:
  ```
  Full FT:        65.2 ± 0.4%
  FNN-only:       64.8 ± 0.3%
  Attention-only: 62.9 ± 0.5%
  ```

**Statistical test:**
```
Pairwise t-tests:
- Full FT vs FNN-only: p=0.12 (not significant)
- Full FT vs Att-only: p=0.001 (significant)
- FNN vs Att-only: p=0.003 (significant)
```

**Why this matters:**
- If Full FT and FNN-only overlap in confidence intervals → no real modularity distinction
- This prevents "cherry-picked best run" accusations

**Expected work:** Re-run experiments, 1 table with error bars, 1 paragraph

**Justification from calibration papers:**
"Report results over all 10 random seeds (not best 5 of 10), or at minimum include both statistics side-by-side, so readers can assess the method's true variance." (Monotonic NNs)

---

## PRIORITY 4: MECHANISTIC ANALYSIS (OPTIONAL but RECOMMENDED)

### Current State:
Probably missing detailed analysis of *why* components differ

### Recommended (if space allows):

#### 4a. Loss Landscape Analysis

**Why**: Show that attention and FNNs have different optimization geometry

**Method:**
```
For each component (Attention, FNN):
├─ Compute Hessian of loss w.r.t. component weights
├─ Report top eigenvalues (characterize curvature)
└─ Compare gradient flow through each component
```

**Expected finding:**
- Attention: Lower gradient flow (harder to update) → more robust to small changes
- FNN: Higher gradient flow (easier to update) → benefits more from focused tuning

**Expected work:** 1 figure, 0.5 page

---

#### 4b. Feature/Representation Similarity Analysis

**Why**: Show that attention and FNN actually learn different things

**Method:**
```
After fine-tuning:

Attention-only FT:
├─ Attention patterns changed? CCA(original_attn, finetuned_attn) = 0.78
└─ FNN representations changed? CCA(original_ffn, finetuned_ffn) = 0.12

FNN-only FT:
├─ FNN representations changed? CCA(original_ffn, finetuned_ffn) = 0.82
└─ Attention patterns changed? CCA(original_attn, finetuned_attn) = 0.18
```

**Interpretation:**
- Shows that tuning components affects their representations
- Provides functional evidence for modularity

**Expected work:** 1 table, 0.5 page

---

## PRIORITY 5: CLARIFY EXPERIMENTAL SETUP

### Current State:
Probably missing details that reviewers will ask about

### Required Clarifications:

#### In Methodology Section:
- [ ] Model architecture (Transformer, size, # layers)
- [ ] Hyperparameter search procedure: "We performed grid search over learning rates {1e-5, 1e-4, 1e-3, 1e-2} on a held-out validation set from [BENCHMARK]. The learning rate maximizing [METRIC] was selected for each component."
- [ ] Number of random seeds: "All experiments use 10 random seeds with different random initializations."
- [ ] Fair comparison details: "All components were tuned with identical computational budget: [X] hyperparameter settings evaluated, [Y] hours total compute per condition."

#### In Results Section:
- [ ] Confidence intervals on all results
- [ ] Comparison of parameter counts: "Attention-only fine-tunes [X]% of parameters, FNN-only tunes [Y]%, full fine-tunes [Z]%"
- [ ] Learning rate choices: "Optimal learning rates selected from grid search: Full FT=1e-3, FNN-only=1e-4, Attention-only=5e-4"

**Justification from calibration papers:**
"Hardware inconsistency... The paper does not provide on equal hardware." (wd1)
"Missing model variant specification." (Are LLMs Exploitable Negotiators)

---

## RECOMMENDED WRITING SEQUENCE

### Version 1 → 2 (2-3 weeks):
1. Add fixed-LR ablation (2.a)
2. Add learning rate curves (2.b)
3. Run multiple seeds + error bars (3.a)
4. Update methodology with clarifications (5)

### Version 2 → 3 (3-4 weeks):
1. Add 2 new benchmark types (1.a) - choose 2 of 3 recommended
2. Add multi-scale evaluation (1.b) - at least 1 larger model
3. Add component update analysis (2.c)

### Version 3 → Final (optional, 1-2 weeks):
1. Add loss landscape / feature similarity (4.a-4.b)
2. Add 3rd benchmark type if space (1.a)
3. Comprehensive rebuttal addressing modularity definition

---

## EXPECTED OUTCOME AFTER RECOMMENDATIONS

### Current State (Predicted Review):
- **Novelty**: Moderate (component-specific tuning not entirely new)
- **Empirical Support**: Weak (only 2 narrow benchmarks, no mechanistic evidence)
- **Significance**: Moderate (useful if true, but unclear how general)
- **Clarity**: Good
- **Recommendation**: Reject with suggestion to resubmit

### After Following Recommendations:
- **Novelty**: Moderate-to-High (broad evaluation shows genuine discovery)
- **Empirical Support**: Strong (multiple domains, mechanistic evidence)
- **Significance**: High (practical method with mechanistic understanding)
- **Clarity**: Good
- **Recommendation**: Likely Accept or Weak Accept

---

## CHECKLIST FOR FINAL SUBMISSION

Before submitting, verify:

- [ ] Evaluation on ≥3 different benchmark types (rule-based + 2 others)
- [ ] Evaluation on ≥2 model scales
- [ ] All results reported with error bars (10 seeds minimum)
- [ ] Fixed-LR ablation showing component differences aren't just tuning artifacts
- [ ] Learning rate curves showing optimization landscape differences
- [ ] Component update magnitude analysis showing selective updates
- [ ] Statistical significance testing (p-values, not just differences)
- [ ] Clear statement: "We followed [PROCEDURE] to ensure fair comparison of baselines"
- [ ] Mechanistic analysis (loss landscape OR feature similarity, at least one)
- [ ] Honest discussion of scope limitations
- [ ] Clear definition of "modularity" and how paper validates it

---

## IF SCOPE IS LIMITED

If you cannot do all recommendations, prioritize:

**Minimum viable set (for acceptance):**
1. Priority 2a (Fixed LR ablation) - ESSENTIAL
2. Priority 3a (Multiple seeds + stats) - ESSENTIAL
3. Priority 1a (2 more benchmarks) - IMPORTANT
4. Priority 5 (Clarify setup) - IMPORTANT

**This alone would move paper from Reject → Borderline**

Adding Priority 2b (LR curves) + 2c (update magnitude) would move to **Weak Accept**.

Adding Priority 1b (multi-scale) + 4 (mechanistic) would move to **Accept**.

---

## COMMON PITFALLS TO AVOID

Based on calibration papers, avoid:

1. ❌ Claiming "general principle" from 2 benchmarks
   - ✅ Say "rule-based reasoning" specifically

2. ❌ Showing only best-run results
   - ✅ Report all seeds with error bars

3. ❌ Comparing baselines with different hyperparameter budgets
   - ✅ Explicitly describe fair search procedure

4. ❌ Interpreting learning rate differences as component properties
   - ✅ Show fixed-LR results first

5. ❌ Black-box performance claims without mechanism
   - ✅ Show what actually changed in the model

6. ❌ Overstating implications ("attention is less modular than FNN")
   - ✅ Qualify findings ("on rule-based reasoning with [X] model size")

---

## TEMPLATE LANGUAGE FOR PAPER

### For Scope Discussion:
> "This work evaluates module-level fine-tuning on [BENCHMARK_TYPES]. While these represent an important class of reasoning tasks, generalization to other domains (e.g., semantic reasoning, commonsense reasoning) remains an open question for future work. Section [X] provides preliminary results on [ADDITIONAL_DOMAIN]."

### For Modularity Claim:
> "Rather than claiming attention and FNNs are inherently different, we show that under the optimization conditions tested (learning rate [X], model size [Y] parameters, dataset [Z]), selective fine-tuning produces measurable differences. Mechanistic analysis (Section [X]) suggests these differences arise from [EXPLANATION], but further investigation on other architectures and scales is needed."

### For Comparison Statement:
> "To ensure fair comparison, all baselines were evaluated with identical hyperparameter search budgets: [PROCEDURE]. Results are reported over [N] random seeds with [STATISTIC] ± [ERROR]. Pairwise significance tests are provided in Table [X]."
