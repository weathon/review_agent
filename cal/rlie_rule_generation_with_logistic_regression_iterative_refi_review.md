=== CALIBRATION EXAMPLE 23 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately reflects the contribution: a pipeline combining LLM rule generation, logistic regression for weighting, iterative refinement, and evaluation.
- The abstract clearly states the problem (lack of principled rule combination and probabilistic modeling with LLMs), the method (RLIE's four-stage framework), and the key empirical finding (direct linear combiner outperforms LLM-augmented inference).
- The claim that injecting weights and linear predictions "surprisingly degrade performance" is well-supported by Table 2 and Section 5.2. No unsupported claims are present in the abstract.

### Introduction & Motivation
- The problem is well-motivated, tracing a logical progression from classical symbolic/probabilistic rule learning to the expressiveness of LLMs, highlighting the gap in joint rule set optimization and weighted aggregation.
- Contributions are clearly enumerated. However, the claim in Section 2.2 ("we are the first to explicitly combine LLMs with probabilistic methods to learn a set of weighted rules") is likely overstated. Prior work on LLM-as-feature-engineers feeding extracted hypotheses/scores into logistic regression, SVMs, or simple linear heads exists (e.g., in automated machine learning, neuro-symbolic prompt tuning, or instruction-tuned classifier heads). The authors should calibrate this novelty claim by precisely delineating how their pipeline (coverage-based filtering + elastic net + iterative hard-mining) differs technically and conceptually from existing LLM-feature-classifier hybrids.
- The introduction neither over-claims nor under-sells; it accurately sets up an empirical study comparing rule-utilization paradigms.

### Method / Approach
- The method is structurally clear and reproducible in principle. However, several assumptions and design choices require justification or clarification:
  1. **Coverage Definition (Sec 3.1):** The text defines coverage using indicator `I(z=0)` (abstentions). Conceptually, coverage should measure how many examples a rule *applies* to, i.e., `1 - (1/N)∑I(z=0)`. If the current formula is used, the threshold `γ` filters rules based on abstention *rate* rather than applicability. This needs explicit mathematical correction or clarification.
  2. **Feature Encoding (Sec 3.2):** Ternary outputs `{-1, 0, +1}` are fed directly into logistic regression. Treating `0` as a numerical midpoint between negative and positive assumes linear separability on this scale and introduces implicit bias (e.g., abstaining contributes `0` to the logit, identical to a perfectly balanced positive/negative rule). Modeling abstention as a missing value or adding a separate "coverage" feature per rule might better capture the intended semantics.
  3. **Pruning Strategy (Sec 3.3):** When `|H| > H`, the pipeline discards rules based on *individual accuracy* on the validation set. This directly contradicts the purpose of the logistic regression stage, which models *joint* dependencies and handles collinearity. Greedily pruning by individual accuracy will break the weight calibration learned in step 2 and discard complementary rules that only fire in conjunction. A principled approach (e.g., coefficient thresholding, sequential backward selection, or L1-sparsed refitting) should replace individual accuracy ranking.
  4. **Iterative Refinement Termination:** Early stopping uses a margin `δ` over `p` iterations, but these hyperparameters are never specified in Section 4.3 or Appendix. Reproducibility requires exact values.

### Experiments & Results
- The experiments directly address the paper's core question: how to best utilize learned rules. The hierarchical evaluation (E1–E4) is well-designed and strongly supports the conclusion that linear-only inference is superior.
- **Baseline Fairness (Table 1):** The comparison is partially confounded by model scale. RLIE reports results with `Qwen3-Next-80B`, `Qwen3-235B`, and `DeepSeek-V3`, while all non-RLIE baselines use `DeepSeek-V3` (except LoRA). RLIE's bolded/top results come from larger backbones. To claim algorithmic superiority over baselines, RLIE (DeepSeek-V3) must be explicitly compared against baselines (DeepSeek-V3) with statistical testing, isolating the framework's contribution from backbone scaling. The current table obscures this.
- **Missing Ablations:** Critical design choices are unverified:
  - How much does *Iterative Refinement* actually improve performance over a single pass (Step 1 → Step 2)? Without a "One-shot RLIE" baseline, the cost-benefit of iterative hard-example mining is unclear.
  - Why ternary `{-1,0,1}` instead of binary? The assumed advantage of explicit abstention is never empirically validated.
  - Impact of Elastic Net vs. standard L1/L2 or unregularized logistic regression on a 300-sample test set is omitted.
- **Dataset Scale & Statistics:** Splits of 200 train / 200 val / 300 test are extremely small for reliably learning probabilistic rule combinations. With 300 test samples, a 1–2% F1 difference falls within typical sampling variance. Reporting 3-run mean/std is good, but pairwise statistical tests (e.g., bootstrap or McNemar) are necessary to substantiate claims of "consistent top-2" performance. Additionally, tuning elastic net `(λ, α)` on a 200-sample val set carries a high risk of validation overfitting, which should be addressed.
- Results are not cherry-picked; Table 2 honestly shows E4 degradation. However, the lack of confidence intervals or significance testing on small datasets weakens empirical rigor for an ICLR submission.

### Writing & Clarity
- The manuscript is generally well-structured and the pipeline logic in Section 3 is easy to follow. 
- Section 3.1's coverage formula is conceptually confusing (as noted above) and impedes understanding of how rules are filtered. 
- Table 1's layout in the text mixes multiple backbone results in a way that makes quick comparison difficult; separating backbone-specific comparisons into distinct sub-tables or explicitly highlighting the apples-to-apples DeepSeek-V3 vs. DeepSeek-V3 line would improve clarity.
- Section 5.2's discussion of why E4 degrades is thoughtful and clearly explained. No major clarity issues block understanding of the core contribution.

### Limitations & Broader Impact
- The authors acknowledge bias transparency and recommend human oversight for sensitive domains, which is appropriate.
- **Missing Limitations:** 
  1. **Computational Cost:** Each refinement iteration requires ternary LLM judging over the entire training set (or at least the hard set), plus prompt generation. The computational and financial cost of RLIE vs. single-pass baselines (like IO Refinement) is not discussed, which is critical for practical adoption.
  2. **Scalability & Task Breadth:** The framework is evaluated only on small binary text classification tasks. Scalability to larger datasets, multi-class settings, or rule interaction complexity (beyond additive log-odds) is not addressed.
  3. **Paradigm Implication:** The strongest result (E1 > E4) actually shows that the LLM's primary role is *feature engineering*, not reasoning. The paper should explicitly frame this as a limitation/insight: RLIE succeeds precisely because it *decouples* semantic generation from inference, avoiding the known brittleness of LLMs in weighted numerical reasoning. Failing to explicitly state this misses an opportunity to position the work within the broader "LLMs are great predictors but poor calibrated integrators" literature.

### Overall Assessment
This paper presents a clear, well-structured pipeline that synergizes LLM-generated natural language rules with regularized logistic regression and iterative hard-example refinement. The core empirical insight—that a simple linear combiner over LLM-judged rules consistently outperforms prompting the LLM with those same rules, weights, or even model predictions—is valuable, aligns with recent findings on LLM brittleness in probabilistic integration, and offers practical guidance for neuro-symbolic system design. However, the work faces several methodological and empirical hurdles that weaken its standing against ICLR's rigor standards. The extremely small dataset splits (200 train/200 val) risk validation overfitting and limit statistical confidence in marginal performance gains. Missing ablations on iterative refinement, ternary encoding, and backbone-matched baselines leave questions about the true source of performance improvements. Furthermore, the pruning strategy based on individual accuracy contradicts the probabilistic aggregation premise, and computational costs are unaddressed. With stronger empirical validation (larger splits or repeated cross-validation, explicit backbone-matched comparisons, significance testing, ablation of key design choices, and a reframing of the LLM's role as a semantic feature extractor rather than a reasoner), this could become a solid ICLR contribution. In its current form, the contribution is interesting but empirically under-supported for unconditional acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
RLIE proposes a unified framework that combines LLM-driven natural language rule generation with elastic-net regularized logistic regression to learn a probabilistically weighted, iteratively refined rule set. Through systematic evaluation across six real-world binary classification tasks, the authors demonstrate that using the learned linear combiner directly yields superior and more stable performance than injecting the rules, their weights, or reference predictions back into an LLM. The work advocates for a clear neuro-symbolic division of labor: LLMs for local semantic judgment and classical probabilistic models for global aggregation.

### Strengths
1. **Systematic Investigation of Inference Paradigms:** The paper rigorously compares four distinct rule utilization strategies (Linear-only, LLM+Rules, LLM+Rules+Weights, LLM+Rules+Weights+Linear Prediction). The consistent empirical finding that the simplest linear combiner (E1) outperforms LLM-augmented strategies provides actionable, counterintuitive guidance that directly challenges common assumptions in LLM-based reasoning. (Evidence: Section 3.4, Table 2)
2. **Well-Structured, Principled Methodology:** The four-stage pipeline is logically cohesive. The design choices, such as allowing LLM abstention (`-1, 0, +1` ternary outputs) and applying coverage-based filtering before probabilistic weighting, effectively mitigate noise and enforce rule sparsity. (Evidence: Sections 3.1-3.2)
3. **Clear Architectural Insight & Discussion:** The paper articulates a robust neuro-symbolic principle: decoupling semantic generation/local judgment from global probabilistic calibration. The discussion thoughtfully explores how the RLIE architecture can be extended (e.g., GAMs, Bayesian logistic regression, isotonic scaling) without altering the core pipeline. (Evidence: Section 6)
4. **Strong Reproducibility Practices:** The paper provides fixed train/val/test splits, explicit hyperparameters (rule capacity, hard example counts, temperature, coverage threshold), baseline implementation details, and complete prompt templates in the appendix, addressing a major pain point in LLM-based research. (Evidence: Sections 4.2-4.3, Appendix E)

### Weaknesses
1. **Limited Dataset Scale & Statistical Fragility:** All experiments use very small datasets (200 training, 200 validation, 300 test samples). While drawn from the HypoBench benchmark, this scale severely limits statistical power, increases sensitivity to prompt/split variations, and makes claims of "robustness" and "low variance" difficult to substantiate. (Evidence: Section 4.3, Table 1)
2. **Missing Strong Classical/Feature Extraction Baselines:** The baselines are confined to LLM-centric rule learners and a fine-tuned LoRA model. The paper lacks comparison to standard pipelines that treat LLM outputs as features (e.g., extracting rule satisfaction signals via LLM + training a standard LR/SVM/Elastic Net directly), making it unclear whether performance gains stem from the iterative refinement process or simply from using a well-regularized linear model. (Evidence: Section 4.2)
3. **Unverified Consistency of LLM Rule Judgments:** The logistic regression stage treats the LLM's ternary rule evaluations as fixed, reliable features. The paper does not assess intra-run or inter-run consistency of these judgments. Given known stochasticity in LLM reasoning, unreported evaluation noise could artificially inflate or deflate learned rule weights. (Evidence: Sections 3.1-3.2)
4. **Overstated Novelty & Undercontextualized Key Finding:** The claim of being the "first to explicitly combine LLMs with probabilistic methods" overlooks substantial prior work using LLM-generated hypotheses/features as inputs to logistic/ensemble models. Additionally, the degradation observed when injecting weights/predictions into the LLM closely mirrors well-documented limitations in LLM numerical reasoning and complex instruction following, but lacks explicit citation of the relevant calibration/numerical reasoning literature to ground the finding theoretically. (Evidence: Section 2.2 claim, Table 2 findings, Section 5.2)

### Novelty & Significance
**Novelty:** The contribution is incremental rather than groundbreaking. The core architecture (LLM generates candidate rules/features -> classical model aggregates) parallels established feature-selection and boosting paradigms. Novelty primarily resides in the explicit framing of LLM rule satisfaction as a ternary feature space, the iterative hard-example refinement loop, and the structured comparison of inference strategies within the rule-learning context. 
**Clarity:** The paper is well-organized, with a clear problem statement, logical method progression, and readable tables/figures. Minor grammatical issues and parsing artifacts exist but do not impede comprehension. The pipeline and evaluation strategies are explained thoroughly.
**Reproducibility:** High. Fixed splits, explicit hyperparameters, temperature controls, and full prompt templates are provided. The authors commit to releasing code upon acceptance. The main reproducibility risk lies in the small dataset size, which can lead to high variance across different random seeds despite the fixed split.
**Significance:** Moderate to high for the neuro-symbolic and LLM reasoning communities. The empirical demonstration that LLMs struggle with explicit probabilistic integration, coupled with the practical recommendation to use linear combiners for global inference, provides valuable engineering guidance. However, the small scale and limited baseline comparisons prevent it from being a definitive benchmark-setting contribution.

### Suggestions for Improvement
1. **Expand Dataset Scale or Provide Rigorous Statistical Validation:** Evaluate on larger subsets of HypoBench or additional standard NLP classification benchmarks. If dataset size is fixed by benchmark constraints, report bootstrap confidence intervals or results across multiple random seeds to statistically validate that the performance gaps between strategies are significant and robust.
2. **Add Direct Classical/Feature-Extraction Baselines:** Include a baseline that extracts the same ternary rule satisfaction signals (or simple zero-shot LLM predictions) and trains a standalone Elastic Net/LR model without the iterative refinement loop. This isolates the true contribution of the RLIE refinement vs. the linear aggregation component.
3. **Quantify LLM Judgment Reliability:** Report a consistency metric (e.g., test-retest agreement rate or entropy of LLM outputs across multiple sampling passes with `temperature=0`) for the ternary rule evaluations. If inconsistency is detected, discuss how it propagates to weight learning and consider adding a confidence-weighting mechanism.
4. **Strengthen Theoretical Context for LLM Degradation:** In Sections 5.2 and 6, explicitly reference and situate the "LLM + Weights" performance drop within the broader literature on LLM numerical reasoning, probability calibration, and instruction-following limits. This will elevate the empirical observation from an isolated result to a well-grounded contribution.
5. **Analyze Computational Overhead & Rule Capacity Sensitivity:** Provide a brief analysis of training time/LLM API costs per iteration compared to baselines. Additionally, include a sensitivity analysis for the rule capacity parameter ($H$) and the coverage threshold ($\gamma$) across multiple datasets to show computational-performance trade-offs.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Dataset scaling and statistical significance testing:** Add experiments on larger training sets (1k–5k samples) with paired statistical tests, because 200 samples are insufficient to trust logistic regression weight stability or validate generalization claims on standard text classification tasks.
2. **Component ablation:** Isolate the contribution of each pipeline stage by comparing against (i) one-shot rule generation + logistic regression, and (ii) logistic regression without iterative hard example mining, to verify that the refined and iterative components actually drive the reported gains.
3. **Backbone parity correction:** Re-run all baselines and RLIE using identical LLM backbones and prompt templates for rule judgment, as Table 1 inconsistently mixes DeepSeek-V3, Qwen3-235B, and LoRa fine-tunes, making it impossible to attribute performance gains to the framework rather than model capacity.
4. **Computational cost reporting:** Report average LLM API calls, latency, and training memory across methods, because the claimed practical advantage of neuro-symbolic division of labor is unverified if RLIE requires orders of magnitude more inference steps than zero-shot or IO refinement.

### Deeper Analysis Needed (top 3-5 only)
1. **Mechanism of LLM performance degradation:** Quantify how often the LLM overrides correct linear predictions versus ignoring rule weights, because stating that LLMs "struggle with probabilistic integration" is unconvincing without empirical error tracing.
2. **LLM local judgment calibration:** Measure the consistency, bias, and agreement rate of the ternary (-1, 0, +1) rule-satisfaction prompts against ground truth, since noisy LLM feature extraction directly contaminates the logistic regression weights and undermines the claimed reliability.
3. **Rule faithfulness and redundancy assessment:** Evaluate whether the learned rule sets are logically distinct and actually drive predictions (e.g., via feature importance permutation or human-rated redundancy scores), as the paper’s core claim of producing "auditable, composable theories" cannot be supported by downstream accuracy alone.

### Visualizations & Case Studies
1. **Rule weight and coverage trajectories:** Plot the L1-regularized weight magnitudes and coverage rates across refinement iterations for each dataset to visually confirm that the logistic combiner is effectively pruning redundant rules rather than arbitrarily assigning near-zero weights.
2. **Failure case breakdown:** Provide side-by-side example tables showing where Linear-only succeeds but LLM+Weights/LinearPred fails, explicitly highlighting which specific rule weight or abstention signal the LLM misinterpreted or overrode during inference.

### Obvious Next Steps
1. **Human evaluation of interpretability:** Conduct a blind study with domain experts rating rule clarity, actionability, and trustworthiness, because automated metrics cannot validate the claim that RLIE produces "semantically clearer" and "auditable" hypotheses.
2. **Prompt sensitivity and robustness testing:** Systematically vary rule generation and judgment prompts across different LLM instruction-finetunes or random seeds to demonstrate that performance is not artifactually dependent on a single carefully engineered prompt template.
3. **Generalized combiner comparison:** Replace logistic regression with a slightly more expressive but interpretable aggregator (e.g., GAMs or decision stumps) to test whether the linear assumption is the true bottleneck for LLM-augmented inference, directly validating the "division of labor" hypothesis.

# Final Consolidated Review
## Summary
RLIE proposes a four-stage pipeline that leverages LLMs to generate natural-language classification rules, applies elastic-net regularized logistic regression for global weighting, refines rules using hard-example feedback, and evaluates multiple inference strategies. The central empirical finding demonstrates that direct linear aggregation of LLM-evaluated rules consistently outperforms prompting the LLM with the same rules, their weights, or reference predictions, advocating for a strict neuro-symbolic division of labor.

## Strengths
- **Actionable, Hierarchical Evaluation of Inference Paradigms:** The systematic comparison of four inference strategies (Linear-only, Rules, Rules+Weights, Rules+Weights+Prediction) yields a clear, counterintuitive empirical result: simple probabilistic aggregation outperforms LLM-augmented reasoning. This provides concrete, practical guidance for neuro-symbolic system design. (Evidence: Sections 3.4 & 5.2, Table 2)
- **Coherent Pipeline Architecture & Reproducibility:** The framework's division of local semantic judgment (LLM) and global calibration (regularized LR) is logically structured and transparently documented. Fixed data splits, explicit hyperparameters, temperature controls, and full prompt templates in the appendix significantly lower reproducibility barriers common in LLM research. (Evidence: Sections 3, 4.3, Appendix E)
- **Strong Empirical Demonstration of Modality Limits:** The degradation observed when injecting weights and model predictions back into the LLM serves as a valuable empirical case study highlighting the brittleness of LLMs when tasked with controlled numerical integration or multi-source evidence fusion. (Evidence: Table 2, Section 5.2)

## Weaknesses
- **Confounded Baseline Comparisons & Model Scale:** Table 1 mixes RLIE results across vastly larger backbones (`Qwen3-Next-80B`, `Qwen3-235B`) with baselines run exclusively on `DeepSeek-V3`. The top-performing and bolded results rely on significantly more capable models, making it impossible to disentangle algorithmic gains from backbone capacity. Without strict backbone-parity evaluations and statistical testing, claims of framework superiority are unsubstantiated.
- **Small-Scale Experiments & Statistical Fragility:** Training on 200 samples and testing on 300 samples is extremely small for reliably learning stable probabilistic weights or claiming robust generalization. Marginal F1 differences of 1–3% on `N=300` fall well within typical sampling variance. The absence of bootstrap confidence intervals, paired significance tests, or results on larger subsets severely weakens the empirical foundation for ICLR standards.
- **Contradictory Pruning Mechanism & Missing Component Ablations:** When rule capacity is exceeded, the pipeline prunes based on *individual validation accuracy*, directly contradicting the joint dependency modeling of the logistic regression stage and discarding potentially complementary rules. Furthermore, the paper lacks critical ablations: (1) a one-shot RLIE variant (generation + LR only) to verify that iterative hard-mining actually drives performance, and (2) a standard `LLM-features → LR` baseline to isolate whether gains stem from the refinement loop or merely from using a well-regularized linear head on LLM outputs.

## Nice-to-Haves
- Clarify the coverage formula in Section 3.1: the text sums `I(z=0)` (abstentions) but claims to filter by applicability. If thresholds target low abstention rates, specialized but highly accurate rules may be systematically discarded.
- Report computational overhead: average LLM API calls, latency, and memory consumption per iteration versus single-pass baselines would strengthen practical adoption claims.
- Provide failure-case breakdowns or rule-override statistics to quantify how often and why the LLM ignores correct linear predictions or misinterprets weights in E4.

## Novel Insights
The paper successfully crystallizes a pragmatic engineering principle for modern neuro-symbolic systems: LLMs excel at semantic interpretation and local rule evaluation but are fundamentally unreliable as global probabilistic integrators. By strictly decoupling generation/judgment from aggregation, RLIE demonstrates that classical statistical combiners not only stabilize predictions but also preserve interpretability that complex LLM prompting actively degrades. This reframes the LLM's role in rule-based reasoning from an end-to-end reasoner to a high-dimensional, semantically grounded feature extractor, offering a clear pathway toward more calibrated and auditable hybrid architectures.

## Potentially Missed Related Work
- **LLM-as-Feature-Engineers / Automated Feature Synthesis:** Prior work treating LLM-generated hypotheses, rationales, or natural language embeddings as inputs to logistic regression, SVMs, or ensemble methods (e.g., `Singh et al., 2022`, `Ruiz et al., 2023`) should be discussed to accurately position RLIE's novelty within the LLM+classical hybrid literature.
- **LLM Calibration & Numerical Reasoning Limits:** The degradation in E4 strongly aligns with literature on LLM probability miscalibration, arithmetic brittleness, and complex instruction-following failure modes (e.g., `Ouyang et al., 2023` on numerical reasoning, `Tian et al., 2024` on LLM calibration). Citing these would theoretically ground the empirical observation rather than presenting it as an isolated phenomenon.

## Suggestions
- Re-run all main comparisons with strict backbone parity (identical LLM for all frameworks) and report bootstrap confidence intervals or McNemar tests to verify that performance margins are statistically significant given the small dataset size.
- Introduce two ablation baselines: (1) `RLIE-SinglePass` (direct rule generation + LR without iterative refinement) and (2) `Standard LLM→LR` (extract ternary rule satisfactions in one pass + Elastic Net without capacity-based pruning or hard-mining). This will isolate the true marginal contribution of the iterative refinement component.
- Replace the individual-accuracy pruning strategy in Section 3.3 with a principled alternative that respects joint weight learning, such as L1 coefficient thresholding, recursive feature elimination based on joint validation score, or greedy forward/backward selection using the logistic objective.
- Explicitly frame the LLM's role in the introduction and discussion as a semantic feature generator/local evaluator, and contextualize the E4 degradation within established literature on LLM numerical integration and instruction-fidelity limits to elevate the theoretical contribution.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
