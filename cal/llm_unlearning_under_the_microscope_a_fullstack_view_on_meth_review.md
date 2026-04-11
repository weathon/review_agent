=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary
This paper presents a systematic, "full-stack" analysis of LLM unlearning, organizing twelve state-of-the-art methods into a taxonomy of three families: divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning. Its core contributions are a critique of standard multiple-choice question (MCQ) evaluations as myopic, the proposal of open question-answering (Open-QA) metrics to better assess generative behavior and the UE-UT trade-off, and a comprehensive robustness analysis against diverse model-level and input-level attacks, revealing distinct vulnerability profiles across method families.

## Strengths
- **Systematic Taxonomy and Analysis:** The paper provides a clear, well-motivated taxonomy of twelve unlearning methods into three intuitive families (divergence-driven optimization, representation misalignment, rejection-based targeted unlearning). This structuring is a valuable contribution that organizes a fragmented research landscape and effectively guides the subsequent comparative analysis.
- **Novel Evaluation Perspective:** The introduction and empirical demonstration of Open-QA metrics (e.g., entailment score) as a necessary complement to standard MCQ evaluations is significant. The paper provides compelling evidence (e.g., Table A1, Fig. 1) that MCQ can mask critical issues like over-forgetting and degraded generation quality, offering a more nuanced and accurate view of the fundamental UE-UT trade-off.
- **Comprehensive Robustness Study:** The robustness analysis is thorough, examining multiple distinct attack vectors (in-domain relearning, out-of-domain fine-tuning, quantization, jailbreaking). The finding that robustness profiles differ across attacks and method families (e.g., divergence-driven methods are more robust to in-domain relearning but less so to out-of-domain fine-tuning) is an important and nuanced insight that advances the field's understanding of unlearning vulnerabilities.

## Weaknesses
### Major:
- **Proposed Open-QA Metric is Conceptually Misaligned with Unlearning Goals:** The entailment score (ES) proposed for evaluating Unlearning Effectiveness (UE) in Open-QA measures factual consistency between the model's output and the original correct answer. A low ES is desired. However, this metric cannot distinguish between a model that safely refuses to answer (e.g., "I don't know")—an ideal unlearned behavior—and one that generates incorrect but potentially harmful content. Both would score low on ES. This conflation undermines the paper's central claim that Open-QA metrics "better capture generative performance and reveal the inherent UE–UT tradeoff," as the metric does not adequately measure the safety or appropriateness of the unlearned model's generative behavior.
- **Limited Generalizability of Taxonomic Claims:** The paper's conclusions about the behaviors and trade-offs of different method families (e.g., divergence-driven optimization causes over-forgetting) are drawn almost exclusively from experiments on the WMDP-Bio benchmark. To substantiate these generalized claims about method *families*, the core experiments (especially those in Figures 1 and 2) should be validated on at least one other major benchmark (e.g., TOFU, MUSE) to demonstrate that the observed behaviors are inherent to the methodologies and not specific to the WMDP dataset or domain.
- **Incomplete Hyperparameter and Optimization Analysis:** The comparative evaluation of 12 methods, while extensive, lacks a systematic ablation study of key hyperparameters (e.g., the regularization weight λ, the β in NPO) that control the UE-UT trade-off. Furthermore, the tuning protocols differ across families (e.g., some parameters are grid-searched, while for "all other methods" configurations are taken from prior work). Without demonstrating that each method was optimized to its fullest potential under a consistent resource budget, the fairness of the comparative rankings (e.g., RMU being the "strongest" in its family) is uncertain, and the observed trade-offs may reflect optimization artifacts rather than fundamental methodological properties.

### Minor:
- **Superficial Explanations for Observed Phenomena:** The paper excellently documents *what* happens (e.g., divergence-driven methods cause logit collapse, different robustness profiles) but often provides only post-hoc, surface-level explanations for *why* (e.g., "degraded generative capacity inadvertently hinders their ability to reveal sensitive knowledge"). A deeper mechanistic analysis—such as probing representation dynamics, analyzing gradient distributions, or more detailed loss landscape studies—would significantly strengthen the contribution by moving from correlation to causation.
- **Inconsistent Use of Proposed Metrics in Robustness Analysis:** The paper introduces Open-QA metrics (UEOpen-QA) as crucial for evaluation but then primarily uses only MCQ-based accuracy (UEMCQ) to report robustness against relearning, fine-tuning, and jailbreaking (Figures 2, 4). To be consistent and to see if attack success differs between answer selection and generation, the robustness analyses should also be reported using the proposed UEOpen-QA metric.
- **Validation of the Entailment Score Metric is Limited:** The entailment score (ES) is central to the new evaluation paradigm, but its validation is somewhat limited. The paper would benefit from a discussion of the chosen NLI model's reliability for this task, potential failure modes, and an ablation study on the sensitivity of ES to the few-shot prompt format. A small-scale correlation with human judgments of answer quality/safety would help establish the metric's validity.

### Trivial:
- **Figure and Presentation Clarity:** Some figures (e.g., Figure 2) are dense and could be presented more clearly. The description of Figure 4(b) is slightly ambiguous regarding the connected points.

## Nice-to-Haves
- **Inclusion of a Strong "Approximate Retraining" Baseline:** Including a retraining-on-the-retain-set-only baseline (even on a subset) would help contextualize the performance gaps of all 12 methods and show how close current methods are to the ideal, computationally expensive standard.
- **Expanded Utility Evaluation:** Incorporating an assessment of general conversational or instruction-following utility post-unlearning (e.g., using MT-Bench) would provide a more holistic view of the utility retention trade-off, which is critical for practical deployment.
- **Multi-Scale Model Experiment:** All experiments use Llama-3-8B. A single experiment with a larger-scale model (e.g., 70B) on a subset of methods would help check if the taxonomic findings and trade-offs hold at different scales, strengthening the paper's guidance for future work.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strength: "The paper is well-written / the topic is important."** (Removed as generic; does not identify something specific this paper does well that most do not.)
- **Weakness: "Lack of justification for method selection."** (Removed as scope creep; the paper explicitly examines twelve *representative* methods to build a taxonomy, not an exhaustive survey. Demanding justification for every omitted method is unreasonable.)
- **Weakness: "The robustness analysis is purely empirical."** (Weakened and moved to Minor; while deeper mechanistic insight is desirable, comprehensive empirical evaluation and correlation analysis constitute a valid and significant contribution in this area.)
- **Weakness: "Reproducibility concerns about undisclosed final hyperparameters."** (Removed as a nitpick; the paper provides hyperparameter search ranges in Appendix A, and reporting exact final values for 12 methods is often impractical and not a standard requirement for a broad comparative analysis of this scale.)

## Suggestions
- **Address the Metric Misalignment:** Revise the Open-QA evaluation for UE. Consider a metric that explicitly scores for safe refusal (e.g., "I don't know") versus incorrect or harmful generation, potentially using a classifier or a refined NLI setup that can identify refusal responses.
- **Strengthen Generalizability:** Conduct a pivotal experiment applying the core evaluation (UE/UT trade-off from Fig. 1) on one additional benchmark (e.g., TOFU or MUSE) to confirm that the observed family-level behaviors are not dataset-specific.
- **Deepen Causal Analysis:** Add a focused analysis (e.g., probing classifiers on retain knowledge, analyzing per-layer gradient norms) to build a more mechanistic explanation for why divergence-driven methods lead to logit collapse and over-forgetting, and why robustness profiles differ.
- **Perform a Focused Hyperparameter Sensitivity Study:** For 2-3 key methods from different families, include an ablation showing how the UE-UT Pareto front shifts with the primary trade-off controlling hyperparameter (e.g., λ). This would directly address concerns about fair comparison and illuminate the tunability of the trade-off.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 4.0]
Average score: 3.5
Binary outcome: Reject
