=== CALIBRATION EXAMPLE 26 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Accuracy**: The title clearly signals the core components (LLM rule generation, logistic regression, iterative refinement, evaluation) and accurately reflects the paper's scope.
- **Clarity**: The abstract effectively outlines the problem (LLMs generate rules but lack principled aggregation), the proposed framework (RLIE's 4 stages), and the key empirical finding (direct linear combination outperforms prompting the LLM with rules/weights/predictions).
- **Supported Claims**: The claim that prompting LLMs with weighted rules "surprisingly degrades the performance" is well-supported by Table 2. However, the abstract states RLIE achieves "superior over all performance compared to a range of LLMbased methods." Table 1 shows RLIE is highly competitive and robust, but strictly "best" on all datasets (e.g., LLM Detect favors LoRA, and HypoGeniC is close on Citations). The phrasing slightly overstates the empirical dominance; tempering this to "consistently top-tier or state-of-the-art across diverse, low-resource tasks" would be more precise.

### Introduction & Motivation
- **Motivation & Gap**: The motivation is strong. The authors correctly identify that existing LLM hypothesis-generation methods (e.g., HypoGeniC, IO Refinement) either optimize rules independently or iteratively refine a single rule, missing the opportunity to learn a globally calibrated, interacting rule set. Bridging open-vocabulary LLM generation with classical probabilistic rule aggregation is a well-motivated research gap.
- **Contributions**: Clearly stated and logically aligned with the methodology and results.
- **Claim Calibration**: The introduction claims RLIE is "the first to explicitly combine LLMs with probabilistic methods to learn a set of weighted rules." While the specific pipeline is novel, this claim risks overlooking prior neuro-symbolic and neuro-fuzzy literature where neural/LLM outputs are combined via weighted log-linear models or differentiable logic (e.g., Logic-LM, differentiable rule networks). I recommend clarifying the novelty boundary: the contribution is not merely "combining" but establishing the specific engineering paradigm (LLM for local ternary judgment + Elastic Net for global aggregation + error-driven LLM refinement) and empirically analyzing the failure of LLMs to internalize probabilistic signals during inference.

### Method / Approach
- **Clarity & Reproducibility**: The four-stage pipeline is logically structured and generally reproducible. The use of ternary judgments `{−1, 0, +1}` for local rule evaluation is a sensible design choice for modeling abstention/coverage.
- **Assumptions & Justification**: The method heavily assumes the LLM can consistently and deterministically evaluate whether a natural language rule applies to a sample. While the temperature is set to `1e-5` for reproducibility, LLMs are notoriously sensitive to prompt phrasing and context. There is no discussion of how inter-sample or inter-iteration consistency in rule judgments is measured or guaranteed. If the LLM's local judgments are noisy, the subsequent logistic regression layer inherits this noise, potentially compromising weight calibration.
- **Logical/Technical Gaps**: There is a critical technical issue in **Equation 2** (Section 3.1). The equation defines coverage as `(1/N_tr) * Σ I(z_i,j = 0)`. Since `z=0` denotes *abstention*, this formula computes the **abstention rate**, not coverage. The text then states rules are discarded if "coverage [is] lower than a predefined threshold γ." If coverage means "applies to many samples," rules should be discarded when the *application rate* `(Σ I(z ≠ 0)) / N` is low, i.e., when the abstention rate is *high*. As written, Eq 2 combined with the thresholding logic is contradictory and likely a typo. This needs immediate correction for the method to be mathematically sound.
- **Edge Cases**: The pruning strategy in Iterative Refinement (Section 3.3, Step 3) states that if `|H_tmp| > H`, rules are ranked by "individual accuracy on the validation set" and pruned. However, individual rule accuracy ignores the synergy/complementarity that the logistic regression is designed to capture. Pruning based purely on marginal accuracy before global re-weighting might discard weak but highly complementary rules, undermining the purpose of the combiner. A more principled approach would be to fit the full Elastic Net on the superset and drop rules with zeroed-out coefficients, which the framework already computes.

### Experiments & Results
- **Testing Claims**: The experiments directly test the core claims. Table 1 validates RLIE's overall competitiveness, and Table 2 cleanly addresses whether LLMs can utilize weighted rules.
- **Baselines**: Strong and appropriate. Comparing against HypoGeniC, IO Refinement, and standard ICL/Zero-shot baselines is fair. The adaptation of baselines to the same train/val/test splits (Appendix A.2) is commendable.
- **Missing Ablations**: While Table 2 ablates inference strategies and Appendix C ablates `γ`, there are two material missing ablations:
  1. **Component Contribution**: How much does *iterative refinement* actually buy over single-shot rule generation + LR? A comparison of RLIE vs. "RLIE (no iterative loop)" across datasets would isolate the value of the hard-exampling feedback loop.
  2. **Rule Capacity `H`**: The capacity is fixed at `H=10`. How sensitive is the model's performance and sparsity to `H`? Does increasing `H` yield diminishing returns or overfitting, especially given the small training sizes?
- **Statistical Rigor**: Section 4.3 states "Each experiment was repeated at least three times, and we report the mean and standard deviation." However, **Tables 1 and 2 do not report standard deviations or confidence intervals**. With training sets of only 200 samples, performance variance is inevitable. ICLR expects explicit statistical reporting (e.g., `68.3 ± 1.2` or error bars) and ideally statistical significance tests (e.g., paired bootstrap or Wilcoxon) when claiming consistency or superiority. Omitting these weakens the empirical claims.
- **Interpretation of Table 2**: The finding that LLM+Rules+Weights degrades performance is a strong, valuable insight. However, the authors frame it as a limitation of LLMs in "fine-grained, controlled probabilistic integration." An alternative explanation is **prompt interface failure**. Figures 7 and 8 present weights as raw numbers and ask the LLM to "consider how the bias interacts." LLMs are known to perform poorly at numerical arithmetic and probability calibration in free-text. The degradation might stem from suboptimal prompting or lack of structured numerical reasoning tools, rather than an inherent inability to handle probabilistic signals. The authors should explicitly discuss this nuance to avoid overgeneralizing the finding.

### Writing & Clarity
- The paper is generally well-organized, and the pipeline description is accessible.
- The contradiction in **Equation 2** vs. the coverage thresholding logic (discussed in Method) is a significant clarity/technical flaw that must be resolved.
- The transition between local judgment (ternary `{-1,0,1}`) and logistic regression features is clear, but the rationale for mapping abstention to `0` rather than handling it via masking in the loss function is not discussed. Does the model learn a negative weight for a rule to predict "not spam," and how does `0` (abstain) interact with this? Briefly clarifying the feature semantics in Section 3.2 would improve understanding.
- Figure/Tables are informative, though as noted, the lack of error bars in tables impedes clear assessment of result stability.

### Limitations & Broader Impact
- **Acknowledged Limitations**: The Ethics Statement appropriately addresses dataset bias, transparency of learned rules, and the need for human oversight in sensitive domains. The Discussion correctly notes the computational paradigm (LLM for semantics, classical model for probability) and suggests extensions like GAMs or Bayesian LR.
- **Missed Limitations**:
  1. **Scalability**: The method requires `O(N * H * Iterations)` LLM API calls for rule judgment, plus additional calls for iterative refinement. For `N=200`, this is manageable, but for larger datasets or longer texts, this approach becomes computationally expensive and latency-heavy compared to fine-tuning or prompt caching. The paper should acknowledge this cost trade-off.
  2. **Small Dataset Generalization**: Experiments use 200 training samples. While suitable for low-data regimes and hypothesis generation benchmarks, the paper should explicitly state that findings (especially regarding LLM rule internalization and LR calibration) may not directly extrapolate to large-scale, high-resource settings.
  3. **Refinement Instability**: The iterative refinement loop relies on LLM self-correction based on hard examples. There is no reported mechanism to prevent rule oscillation or degradation during refinement cycles beyond validation early stopping. A brief discussion on the stability of the generative feedback loop is warranted.

### Overall Assessment
This is a solid, motivated paper that addresses a meaningful gap in LLM-based reasoning: how to move from ad-hoc, independent rule generation to a principled, globally calibrated rule set. The core insight—that LLMs struggle to reliably integrate explicit probabilistic weights and predictions when prompted in natural language, and that a classical linear combiner remains superior for this task—is a valuable empirical contribution to the neuro-symbolic and interpretability literature. However, to meet ICLR's rigor standards, the authors must address several key issues: correct the logical/typographical error regarding coverage in Equation 2, add statistical significance metrics or confidence intervals to the main results (given the small dataset sizes), include an ablation isolating the contribution of iterative refinement, and broaden the interpretation of the LLM+Weights failure to consider prompt interface limitations versus inherent model deficiencies. With these clarifications and empirical additions, the paper would stand as a strong, reproducible contribution to interpretable AI and LLM reasoning.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces RLIE, a neuro-symbolic framework that integrates LLM-based natural language rule generation with iterative, hard-example-driven refinement and classical elastic-net logistic regression for global weight calibration. Through systematic evaluation across six benchmarks, the work demonstrates that a direct linear combiner consistently outperforms LLM-augmented inference, revealing LLMs' limitations in integrating explicit probabilistic cues despite their strong semantic generation capabilities. 

### Strengths
1. **Clear and Effective Neuro-Symbolic Architecture:** The framework establishes a principled division of labor, leveraging LLMs for local semantic rule evaluation (including abstention) while delegating global aggregation and calibration to a transparent logistic regression model. This design improves interpretability and auditability compared to purely black-box LLM pipelines (Sections 3.2 & 6).
2. **Valuable Empirical Insights on LLM Reasoning Limits:** The hierarchical comparison of four inference strategies yields a counter-intuitive but highly practical finding: injecting rule weights and linear model predictions into an LLM often degrades performance rather than improving it. This provides strong empirical guidance for the community against over-relying on LLMs for numerical/probabilistic reasoning (Table 2, Section 5.2).
3. **Targeted Iterative Refinement Mechanism:** Unlike prior methods that refine based on random or generic validation samples, RLIE explicitly mines hard examples using prediction errors from the statistical combiner. This leads to steady performance improvements and semantically sharper rules across iterations, as qualitatively demonstrated in the case study (Section 3.3, Table 3).

### Weaknesses
1. **Extremely Limited Dataset Scale:** Experiments rely on fixed splits of only 200 training, 200 validation, and 300 test samples per task. This small scale raises concerns regarding statistical reliability, susceptibility to variance, and the generalizability of the proposed framework to realistically sized datasets common in modern ML (Section 4.3).
2. **Ambiguous Mathematical Treatment of Abstentions:** The method encodes LLM ternary judgments as `{-1, 0, +1}` and directly feeds them as features into the linear model `Φ(x)^T β`. Treating an "abstain" (`0`) identically to a neutral feature mathematically conflates missing evidence with explicit negative/positive signals, and the paper does not justify or analyze alternative formulations (e.g., separate coverage indicators or missing-data imputation) (Sections 3.1 & 3.2).
3. **Lack of Statistical Testing and Computational Overhead Analysis:** Results are reported as mean ± standard deviation without formal statistical significance tests (e.g., paired t-tests or bootstrap confidence intervals). Furthermore, the paper omits analysis of computational efficiency, such as the total number of LLM API calls, latency, or cost per iteration, which is critical for evaluating the practical scalability of an iterative neuro-symbolic pipeline (Sections 4.3 & 5).

### Novelty & Significance
**Novelty:** Moderate. While LLM-based hypothesis generation and rule learning are active areas, RLIE's specific coupling of hard-example mining with elastic-net logistic regression is a competent and practical engineering design. It incrementally advances prior LLM inductive reasoning frameworks by introducing a systematic feedback loop that bridges local semantic generation with classical probabilistic calibration.
**Clarity:** High. The manuscript is well-structured, the methodology is thoroughly documented, and the inclusion of prompt templates and hyperparameter settings in the appendix greatly aids comprehension.
**Reproducibility:** High. The paper provides detailed experimental setups, fixed random seeds, specific dataset splits, and clear baseline implementations. The promise of public code release further supports reproducibility, though verifying claims on larger datasets remains a practical checkpoint for immediate replication.
**Significance:** Moderate to High. The core empirical finding—that classical linear combiners outperform LLMs when tasked with integrating weighted probabilistic rules—is highly relevant to the neuro-symbolic and LLM reasoning communities. It offers actionable design principles for building reliable AI systems. However, the significance is partially tempered by the small-scale evaluation and lack of statistical rigor.

### Suggestions for Improvement
1. **Scale and Statistical Validation:** Extend the experimental evaluation to datasets with larger training splits (e.g., 1k–10k samples) to demonstrate scalability. Additionally, incorporate formal statistical significance testing (e.g., Wilcoxon signed-rank or paired t-tests) to confirm that observed performance differences between inference strategies are robust across runs.
2. **Explicit Handling of LLM Abstentions:** Formulate and ablate alternative mathematical treatments for the ternary `{-1, 0, +1}` judgments in the logistic regression stage. For instance, compare the current direct insertion approach against a model that uses a separate binary coverage mask or treats `0` as missing data, and justify the chosen formulation empirically or theoretically.
3. **Computational Efficiency and Robustness Analysis:** Report the computational overhead of RLIE, including average LLM calls per iteration, total training time, and approximate API costs compared to baselines. Additionally, include a brief discussion or ablation on failure modes, such as how the framework performs when initial LLM rules are highly biased or when the linear combiner struggles with inherently non-linear decision boundaries despite rule interactions.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add baselines comparing RLIE to classical interpretable rule learners (e.g., RuleFit, Skope-Rules, GA2M/GAM) because without them, the central claim of advancing "probabilistic rule learning" over established methods is unsubstantiated and may simply reflect weak traditional baselines.
2. Include an ablation that replaces hard-example mining with random subsampling of equal size during iterative refinement, because the framework's performance gains are attributed to error-driven feedback, but no evidence proves targeted selection outperforms cheap random augmentation.
3. Conduct scaling experiments across varying training set sizes (e.g., N=50 to N=2000), because asserting "robustness" and "generalizability" on a fixed 200-sample split fails ICLR's rigor standards and leaves the framework's practical data requirements completely unknown.

### Deeper Analysis Needed (top 3-5 only)
1. Perform statistical significance testing (e.g., paired bootstrap or permuted Wilcoxon) across all runs, because claiming consistent superiority over high-variance baselines using only mean/standard deviation from ~3 runs on 300-sample test sets cannot rule out stochastic noise.
2. Provide quantitative evaluations of rule faithfulness, redundancy, and coverage overlap (e.g., via leave-one-rule-out perturbation or Jaccard similarity metrics), as asserting rules are "compact and semantically clearer" remains purely anecdotal without human readability scores or structural metrics.
3. Diagnose the precise failure mechanism when weights are injected into LLMs via prompt perturbation or log-prob analysis, because broadly attributing performance drops to LLMs' "inability to handle probabilistic integration" ignores confounding factors like instruction-following conflicts, improper weight scaling, or context formatting mismatches.

### Visualizations & Case Studies
1. Provide error-overlap confusion plots comparing Linear-only vs. LLM+Weights predictions to directly reveal whether degradation stems from LLMs correctly ignoring spurious linear signals, incorrectly overwriting correct predictions, or simply failing to parse weight magnitudes.
2. Include reliability diagrams and Expected Calibration Error (ECE) metrics for both the logistic combiner and LLM-augmented strategies, because the paper heavily emphasizes "calibration" and "uncertainty management" but never demonstrates that the output probabilities align with empirical frequencies.
3. Present a qualitative failure-case walkthrough showing an instance where the linear model confidently predicts correctly, but the LLM explicitly overrules it when given weights, exposing the exact reasoning breakdown and validating the claimed "division of labor" necessity.

### Obvious Next Steps
1. Rigorously control for prompt syntax and token complexity across all four evaluation strategies (e.g., matching verbosity and structural templates), because without this, the conclusion that LLMs fundamentally fail at weight integration cannot be separated from poor prompt engineering artifacts.
2. Expand hyperparameter sensitivity analysis for the coverage threshold (γ) and ElasticNet penalties across multiple datasets, as the stability of the global rule selection and claimed "robustness" heavily depend on these choices, and a single sweep on Headlines is insufficient.
3. Explicitly detail which LLM backbones are used for rule generation versus single-rule judgment in each experiment, since Table 1 ambiguously lists different models as the "RLIE backbone," and unclear component attribution undermines reproducibility and fair cost-performance claims.

# Final Consolidated Review
## Summary
RLIE introduces a unified neuro-symbolic framework that couples LLM-generated natural language rules with elastic-net logistic regression for global weighting, augmented by iterative hard-example refinement. Evaluated on six text classification benchmarks, the framework produces compact, interpretable rule sets and demonstrates that a direct linear combiner consistently outperforms strategies injecting rules, weights, or model predictions back into an LLM for inference.

## Strengths
- **Principled division of labor:** The architecture cleanly separates responsibilities, leveraging the LLM for local semantic grounding and abstention-aware rule evaluation, while delegating global aggregation and weighting to a transparent logistic regression model. This yields a robust, auditable pipeline that avoids overwhelming the LLM with complex combinatorial reasoning.
- **Counter-intuitive but highly practical empirical finding:** The hierarchical comparison of inference strategies reveals that explicitly injecting learned rule weights and linear predictions into an LLM consistently degrades performance compared to the linear-only baseline. This challenges the common practice of overloading LLM contexts with probabilistic signals and provides strong empirical guidance for future neuro-symbolic system design.
- **Effective error-driven refinement:** Using prediction errors from the statistical combiner to mine hard examples successfully steers the LLM away from random or redundant generation, leading to measurable accuracy gains and clearer rule semantics across iterations.

## Weaknesses
- **Mathematical inconsistency in coverage calculation (Section 3.1):** Equation 2 defines coverage as `(1/N_tr) Σ I(z_{i,j} = 0)`. Since `z=0` denotes abstention, this computes the *abstention rate*, not coverage. The accompanying text states rules are discarded when "coverage [is] lower than a predefined threshold γ," which would logically require counting non-abstentions `(z ≠ 0)`. As written, the thresholding logic is inverted, creating a fundamental flaw in the filtering mechanism that must be corrected.
- **Missing core ablations:** The paper attributes its robustness to the iterative refinement and hard-example mining loop but omits direct comparisons to (1) single-shot rule generation + logistic regression, and (2) random subsampling of the same number of examples during refinement. Without these, it is unclear how much performance gain stems from the iterative feedback mechanism versus the capacity of the global combiner alone.
- **Inconsistent model attribution & reproducibility concern:** Section 4.3 explicitly states "All experiments involving LLMs utilized gpt-4o-mini," yet Table 1 lists DeepSeek-V3 and Qwen series as backbones for RLIE and baselines. It is unclear which model performs rule generation/judgment versus final inference, and how backbone choices map to reported scores. This ambiguity undermines experimental reproducibility.
- **Suboptimal rule pruning strategy:** In Section 3.3, when the candidate rule set exceeds capacity `H`, rules are pruned based on individual validation accuracy. This ignores inter-rule complementarity that the elastic net combiner is specifically designed to capture, risking the removal of weak but highly synergistic rules before global re-weighting occurs.

## Nice-to-Haves
- Incorporate calibration metrics (e.g., ECE) and reliability diagrams for the logistic combiner and LLM-augmented strategies to empirically verify claims about probability calibration.
- Discuss whether the LLM's failure to utilize weights stems from inherent reasoning limitations or prompt-interface artifacts (e.g., unnormalized weight scales, lack of structured numerical reasoning controls), and ensure prompt verbosity is matched across all four evaluation strategies.
- Report standard deviations directly in the main results tables to align with the claim of 3x experimental repetitions, and provide a computational efficiency breakdown (total LLM API calls/cost per iteration) to assess practical scalability.

## Novel Insights
The paper empirically validates a strict "division of labor" paradigm for neuro-symbolic reasoning: LLMs are most reliable when restricted to local semantic interpretation and candidate generation, while classical statistical models should exclusively handle global probabilistic aggregation. The consistent performance degradation observed when forcing LLMs to explicitly integrate learned numerical weights challenges the prevailing assumption that richer, multi-signal contexts inherently improve decision-making. This suggests that architectural modularity, rather than prompt-based information fusion, is the more reliable pathway to building stable, interpretable neuro-symbolic systems.

## Suggestions
- Immediately correct Equation 2 to compute non-abstention coverage `(Σ I(z ≠ 0) / N)`, or clarify the intended thresholding logic.
- Add ablation studies isolating the iterative refinement loop (full RLIE vs. single-shot) and hard-example mining vs. random sampling to quantify their individual contributions.
- Resolve the backbone inconsistency by explicitly detailing which model handles rule generation, rule application, and baseline inference, and ensure Table 1 aligns with Section 4.3.
- Replace individual-accuracy-based pruning with a principled alternative, such as fitting the elastic net on the full candidate set and discarding rules with zeroed L1 coefficients, to preserve synergistic rule interactions.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
