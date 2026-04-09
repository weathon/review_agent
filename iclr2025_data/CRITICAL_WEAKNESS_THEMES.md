# Critical Weakness Themes for Scaling RL for LLMs Paper

## Overview
These weakness patterns are extracted from ICLR 2025 human reviews and are highly relevant to a paper on "The Art of Scaling Reinforcement Learning Compute for LLMs" with focus on:
- Predictive scaling using sigmoidal curves
- SCALERL recipe for predictable scaling
- 400,000+ GPU-hours empirical study at 8B scale
- Design choice ablations (loss aggregation, normalization, curriculum, off-policy algorithms)
- Scaling to 100,000 GPU hours with verified performance matching

---

## CRITICAL WEAKNESSES (Must Address)

### 1. Scale Coverage and Generalization
**Severity:** CRITICAL - Core paper contribution
**Pattern:** Evaluation limited to single scale (8B model) with uncertain generalization to other sizes
**Evidence from reviews:**
- "Evaluation only on mathematical reasoning tasks... All experiments on a single model family"
- "Experiments are limited to a single model family... limiting generalization claims"
- Limited architecture scope across different model families is criticized

**For your paper:**
- Explicitly test design choice insights across multiple model sizes beyond 8B (test at 1B, 7B, 13B, 34B if possible)
- Show how sigmoidal scaling parameters change across scales
- Document where insights DO and DON'T transfer to different scales
- Provide guidance on hyperparameter selection at untested scales

---

### 2. Baseline Comparison Rigor
**Severity:** CRITICAL - Undermines claimed superiority
**Pattern:** Baselines not properly tuned, using different model sizes, missing strong comparisons
**Evidence from reviews:**
- "the paper provides no detail regarding the hyperparameter settings of the baseline algorithms, nor how or if they were tuned. This is a critical weakness of the paper"
- "Paper does the opposite and provides no detail regarding hyperparameter settings of baselines"
- "Missing comparisons with important baseline algorithms like PPO"

**For your paper:**
- Document hyperparameters for ALL baseline design choices being compared
- Ensure compute-matched comparisons (if method A is slower, budget more compute for baselines)
- Include established algorithms (PPO, SAC, etc.) at comparable computational budgets
- Provide evidence that design choice improvements aren't just from better tuning

---

### 3. Compute Efficiency Claims Must Account for ALL Costs
**Severity:** CRITICAL - Core claim about efficiency
**Pattern:** Performance gains not justified against computational overhead
**Evidence from reviews:**
- "50% slower per epoch with 2% performance gain, but no comparison with compute-matched baseline"
- "Evaluation metrics do not account for cost of model inference... will observed scaling law still hold?"
- "Training cost impacts not discussed despite being critical for large-scale studies"

**For your paper:**
- Report wall-clock training time for all design choices (not just FPS, include actual time)
- Report inference time costs if relevant to RL setting
- Show scaling curves accounting for inference costs (especially important for LLM RL where inference matters)
- Compare design choices on iso-compute bases, not iso-steps
- Include memory requirements for different design choices

---

### 4. Ablation Study Rigor
**Severity:** HIGH - Critical for empirical study credibility
**Pattern:** Ablations lack statistical significance, confidence intervals, or clear contribution analysis
**Evidence from reviews:**
- "Ablation study fails to provide statistical significance... lacks analysis"
- "Many ablations show minimal performance gains with no explanation"
- Shaded confidence intervals in figures should be explained as std dev or CI

**For your paper:**
- Report confidence intervals (95% CI or std dev) for ALL ablation results
- Conduct significance tests when comparing design choices
- Explain why each component is included (what does it add beyond baseline?)
- Show ablations not just as "with/without" but systematically across components
- Document if some components only help edge cases (and be honest about it)

---

### 5. Scaling Law Clarity and Consistency
**Severity:** HIGH - Paper is about scaling laws
**Pattern:** Scaling trends inconsistent or unexplained; cherry-picked favorable ranges
**Evidence from reviews:**
- "Increasing data ratio doesn't benefit methods equally; trend is inconsistent with no explanation"
- "Scaling law doesn't account for inference costs, making practical curves unclear"
- Non-monotonic behavior needs explanation

**For your paper:**
- Show complete scaling curves (not just favorable regions) for ALL design choices
- If any design choice shows non-monotonic behavior, explain why
- Explain inflection points in sigmoidal curves
- Confirm scaling law holds consistently across random seeds (important for reproducibility)
- Don't cherry-pick model sizes or training regimes

---

## HIGH-PRIORITY WEAKNESSES (Should Address)

### 6. Reproducibility and Implementation Details
**Pattern:** Vague implementation details, missing hyperparameters, no code
**Evidence from reviews:**
- "Description of state, action, reward too general, making it difficult to grasp specific implementations"
- "Many details of baselines and datasets missing"
- "Implementation details vague, making reproducibility difficult"

**For your paper:**
- Provide detailed hyperparameter tables for all design choices and model sizes
- Describe preprocessing, reward normalization, data sampling in detail
- Include pseudo-code or algorithm boxes for novel design choices
- Commit to releasing code for reproducibility
- Specify exact RL environments and benchmark suites used

---

### 7. Real-World Applicability and Generalization
**Pattern:** Methods shown to work in controlled settings but unclear transfer to other domains
**Evidence from reviews:**
- "Sim2Real gap acknowledged but not addressed"
- "Method shows improvements in offline settings but unclear how to transfer"
- Single domain evaluation (e.g., MATH-only) limits generalization claims

**For your paper:**
- Test SCALERL recipe on multiple RL domains (not just one task)
- For LLM RL specifically, test on diverse tasks (reasoning, coding, alignment, etc.)
- Discuss when insights about design choices might NOT hold
- Provide honest assessment of limitations and assumptions
- Consider sim-to-real if applicable

---

### 8. Missing Comparisons with Related Work
**Pattern:** Key baselines or recent methods not compared
**Evidence from reviews:**
- "Missing comparisons with recent methods published in 2023-2024"
- "Paper does not compare with closest baseline methodologies"
- "No comparison with other process reward models"

**For your paper:**
- Compare design choices against all published variants mentioned (e.g., different aggregation schemes)
- Include recent scaling papers as baselines
- Compare specific design choices head-to-head where possible
- Citation of related work should be comprehensive

---

### 9. Hyperparameter Sensitivity and Selection
**Pattern:** Large number of hyperparameters; no guidance on selection
**Evidence from reviews:**
- "Method has large list of hyperparameters which limits practical applicability"
- "Number of clusters selection guidance missing for real-world application"
- "Grid search for hyperparameter, do values generalize to other tasks?"

**For your paper:**
- Conduct sensitivity analysis for key hyperparameters
- Provide clear decision rules for hyperparameter selection (especially for untested scales)
- Show which hyperparameters are critical vs. robust
- Provide ablations on key hyperparameter choices
- Include learning rate schedules, batch sizes, update frequencies for all runs

---

### 10. Domain-Specific Validation
**Pattern:** Results shown in narrow domains without validating broader applicability
**Evidence from reviews:**
- "Experiments limited to single task domain"
- "Results on other tasks show no improvement or negative improvement"
- "Method doesn't work well on other tasks, generalization capability limited"

**For your paper:**
- Test SCALERL on diverse RL tasks (not just one benchmark)
- If focusing on LLM RL, test on multiple LLM domains
- Show scaling behavior is consistent across domains
- Document when design choices help vs. hurt on different tasks
- Provide task-specific recommendations if they differ

---

## MODERATE-PRIORITY WEAKNESSES (Nice to Address)

### 11. Theoretical Justification
- Explain WHY design choices work, not just THAT they work
- Connect to existing theory where relevant
- Avoid overclaiming theoretical contributions

### 12. Statistical Testing
- Use appropriate statistical tests for comparisons
- Report multiple random seeds (at least 5+)
- Use power analysis to determine required sample sizes

### 13. Presentation Quality
- Use clear, consistent notation
- Avoid unnecessary complexity
- Make figures/tables easy to interpret
- Include intuitive explanations alongside equations

### 14. Baseline Details
- Explain why specific baselines were chosen
- Justify exclusion of alternatives
- Provide context for baseline performance

---

## CHECKLIST FOR YOUR PAPER

Before submission, ensure your scaling RL paper addresses:

**Critical Items:**
- [ ] Tested design choices across multiple model sizes (not just 8B)
- [ ] Hyperparameters fully disclosed for all baselines
- [ ] All comparisons are compute-matched or clearly noted
- [ ] Confidence intervals/std dev reported for all results
- [ ] Training time, inference time, memory all documented
- [ ] Scaling curves complete and explained (no cherry-picking)
- [ ] Ablations show contribution of each component with stats

**High Priority Items:**
- [ ] Implementation details sufficient for reproduction
- [ ] Code will be released
- [ ] Tested on multiple RL domains/tasks
- [ ] Recent related work compared
- [ ] Hyperparameter sensitivity analyzed
- [ ] Limitations and assumptions stated clearly
- [ ] Random seed variation reported (5+ runs)

**Nice to Have:**
- [ ] Theoretical justification provided
- [ ] Figures and tables are clear and interpretable
- [ ] Novel insights explained intuitively
- [ ] Recommendations for practitioners provided
- [ ] Discussion of edge cases and when method struggles

---

## Key Themes to Emphasize in Your Paper

1. **Predictability:** Show that scaling behavior is consistent and predictable (not noisy)
2. **Reproducibility:** Provide all details needed for others to reproduce results
3. **Honesty:** Discuss when design choices help/hurt and why
4. **Breadth:** Show insights generalize across scales and domains
5. **Rigor:** Statistical significance, proper baselines, fair comparisons
6. **Practicality:** Clear guidance on using SCALERL recipe at new scales

