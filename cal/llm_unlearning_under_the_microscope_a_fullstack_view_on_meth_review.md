=== CALIBRATION EXAMPLE 25 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "LLM Unlearning Under the Microscope: A Full-Stack View on Methods and Metrics" is appropriately descriptive. The abstract accurately previews the paper's three contributions: a taxonomy of 12 methods, Open-QA metrics for UE/UT evaluation, and a multi-faceted robustness analysis. The claim that current MCQ evaluations "overstate success" is a strong one and is reasonably (if narrowly) substantiated in the body. One concern: the abstract implies the work is broadly generalizable, yet virtually all experiments use a single model (Llama-3 8B Instruct) on a single benchmark (WMDP-Bio). This scope limitation is not telegraphed upfront.

---

### Introduction & Motivation

The motivation is genuine: MCQ-based evaluation dominates the LLM unlearning literature, and this may obscure meaningful distinctions between methods. The introduction appropriately situates the work relative to prior surveys and benchmarking studies.

A notable parsing artifact aside, the text references "the key research question we aim to address is: [Q]" but the actual question Q is missing from the extracted body—it appears to have been in a formatted call-out box. This is a structural oddity that impedes follow-through in the introduction.

The differentiation from most closely related concurrent works (Feng et al., 2025; Che et al., 2025; Hu et al., 2025) is stated but shallow. Feng et al. (2025), titled "Existing large language model unlearning evaluations are inconclusive," has substantial thematic overlap with this paper's Section 4. The introduction asserts three novelties, but the boundary between this work and Feng et al. needs sharper articulation.

---

### Section 3: Taxonomy of Unlearning Methods

The three-way categorization (divergence-driven optimization, representation misalignment, rejection-based) is conceptually clean and useful as an organizing framework. That said, several concerns arise:

**Novelty of the taxonomy.** The groupings are fairly intuitive and largely recoverable from the original papers. GA/GradDiff/NPO/SimNPO as "divergence-driven," RMU/RR/TAR/LAT as "representation misalignment," and IDK/DPO/IDK+AP/ELM as "rejection-based" do not require surprising analytical insight. The paper's value must come from what insights the taxonomy enables—which it does in Sections 4–5, but the taxonomy itself is not a contribution of independent weight.

**ELM classification.** ELM (Gandikota et al., 2024) aligns model outputs with a reference model prompted with a refusal-inducing prefix. Classifying this under "rejection-based targeted unlearning" alongside IDK is reasonable, but ELM still operates via KL-divergence-like alignment with a reference distribution, placing it closer to divergence-driven methods mechanistically. This ambiguity deserves acknowledgment.

**Boundary cases.** DPO-for-unlearning is presented as a "rejection-based" method. However, DPO inherently involves both positive (retained) and negative (forgotten) preference signals. The paper's classification treats DPO as primarily rejection-based, but this simplification merits discussion.

---

### Section 4: Beyond MCQ — Rethinking Evaluation

This is the paper's most substantive conceptual contribution.

**The core argument is sound but the evidence is thin.** The central claim—that MCQ-based UE is misleading because an unlearned model may have internally disrupted its generation capacity (not actually forgotten the knowledge)—is compelling and important. However, the main illustrative evidence in Table A1 involves a single question and two unlearning methods. One example is insufficient to establish that this phenomenon is systematic and widespread. A quantitative assessment of how often MCQ and Open-QA disagree, across all 12 methods, would dramatically strengthen this claim.

**The Open-QA metric is not truly free-form.** A critical methodological tension: the paper's key argument is that MCQ fails to capture "generative behavior," yet the proposed Open-QA evaluation (Entailment Score, ES) uses few-shot prompts that constrain the model to output answer choices in MCQ format (e.g., "C. tiger"). By construction, this still gates evaluation on the A/B/C/D letter selection, merely assessed via NLI rather than exact match. This substantially weakens the claim that ES captures free-form generation; it is better described as a soft-scoring MCQ variant. The paper should either (a) acknowledge this limitation explicitly, or (b) demonstrate evaluation on truly open-ended forget-set queries without format constraints.

**NLI model selection is not validated.** ES is computed using the "tasksource" NLI classifier (Sileo, 2023), which is a relatively obscure model. There is no ablation showing that ES is robust to the choice of NLI backbone, and no correlation with human judgments is reported. Given that ES is positioned as the paper's primary novel metric, this omission is significant. Different NLI models may produce different absolute scores and different method rankings.

**Hyperparameter selection criterion is ambiguous.** All methods are tuned via grid search. Which metric was used to select the final hyperparameters? If MCQ-based UE/UT determined the optimal configuration, then the experiments comparing MCQ vs. Open-QA performance are potentially biased: methods may have been optimized for MCQ performance, making their Open-QA underperformance a methodological artifact rather than a genuine characteristic of the algorithm family. This is a serious confound that the paper does not address.

**The ES metric for UE is counterintuitive.** High ES on the forget set means the model still gives correct answers—so low ES is "good" unlearning. But this conflates "the model produces incoherent text" with "the model has genuinely forgotten"—precisely the over-forgetting problem the paper criticizes. There is no metric proposed to distinguish genuine forgetting (the model cannot recall the fact) from degeneracy (the model cannot generate coherent text at all). The paper acknowledges over-forgetting as a problem but the proposed ES metric does not resolve this ambiguity.

**The UT Open-QA benchmarks (IFEval, GSM8K) are reasonable additions**, and the finding that NPO degrades instruction-following (IFEval) while maintaining MMLU accuracy is novel and important. This portion of the analysis is the most convincing empirical contribution of Section 4.

---

### Section 5: Robustness Assessment

**Conceptual contribution is clear.** Distinguishing in-domain relearning (analogous to adversarial robustness) from out-of-domain fine-tuning (analogous to distribution shift robustness) is a useful and underexplored framing. The finding that these two dimensions do not correlate—and indeed that method families show *reversed* rankings—is genuinely insightful.

**The jailbreaking–relearning correlation (Fig. 4b) is the paper's most novel empirical finding.** Showing that RobJA correlates more strongly with RobReL than with RobFT provides a meaningful mechanistic interpretation: both adversarial prompting and in-domain relearning probe forget-domain knowledge, while out-of-domain fine-tuning represents a different perturbation regime. This connection was not established in prior work and deserves more prominent discussion.

**Quantization analysis (Fig. 3) is underdeveloped.** The claim that "knowledge removal is generally more robust to post-unlearning quantization than data-centric unlearning" is supported by Table A2, but Table A2 contains only RMU on MUSE and RMU/NPO on WMDP—which is severely limited. NPO is not shown on MUSE in the table, and no other methods are evaluated for quantization on MUSE. This weakens the generalization claim substantially.

**The "illusion of improved unlearning" under aggressive quantization** is an important concept (Section 5, discussion before Fig. 3), but the paper only tests 4-bit quantization. A sweep over quantization levels (e.g., 8-bit, 6-bit, 4-bit, 3-bit) would show where the transition from genuine robustness to false robustness occurs.

**In-domain relearning setup is potentially too weak.** Relearning uses only 100 steps at a small batch size. It is not demonstrated that this represents a worst-case relearning attack; more steps or a higher learning rate might show sharper differences. The claim that divergence-driven methods are more robust to in-domain relearning depends on whether 100 steps is the right probe level.

**No statistical uncertainty is reported** anywhere in Sections 4 or 5. All results are shown as point estimates. Given that unlearning methods are sensitive to random seeds and learning rate choices, the absence of variance measures makes it difficult to assess whether observed differences (e.g., NPO at 0.28 vs. RMU at 0.27 in UEMCQ before attack in Fig. 2) are meaningful.

---

### Experimental Scope and Generalizability

**This is the paper's most serious structural weakness.** All major experiments use a single model (Llama-3 8B Instruct) on a single benchmark (WMDP-Bio). The paper mentions MUSE only in a brief appendix table with a subset of methods. At ICLR's standard, claims about the relative properties of method families (e.g., "divergence-driven optimization is generally more resilient to in-domain relearning") need to hold across at least two models and ideally two benchmarks. The Llama-3 8B family has specific architectural properties (GQA, RoPE, etc.) that may influence how these methods behave. Whether the findings hold for Mistral, Llama-2, or larger models is unknown.

The paper would be substantially strengthened by:
- One additional model (e.g., Zephyr-7B or Llama-3 70B)
- Full evaluation on MUSE or TOFU alongside WMDP

---

### Writing & Clarity

Section 3 has a notable structural issue in the extracted text: the content introducing Divergence-Driven Optimization (with Eq. 2 and discussion of GA/GradDiff/NPO/SimNPO) appears after the discussion of representation misalignment and rejection-based methods. This is a PDF extraction artifact per the review instructions, but it suggests the paper's section organization may be non-standard in ways worth reviewing.

The discussion of "over-forgetting" is spread across multiple sections (Section 4 body text, Fig. A1, Table A1) without a single clear definitional treatment. A formal definition distinguishing over-forgetting from degeneracy would improve clarity.

---

### Limitations & Broader Impact

The Limitations section (Appendix D) is honest about three limitations: incomplete method coverage, focus on GCG-based jailbreaking only, and reliance on automatic metrics. However, the most important limitation—single-model, single-benchmark scope—is not explicitly listed. The broader impact discussion is appropriate for the subject matter.

---

## Overall Assessment

This paper presents a useful, empirically grounded analysis of LLM unlearning methods, and its central insight—that MCQ-based evaluation is insufficient and that robustness must be analyzed across multiple attack types—is correct and timely. The discovery that jailbreaking robustness correlates more strongly with in-domain relearning than out-of-domain fine-tuning is the paper's most technically novel finding. However, the contribution has meaningful weaknesses that prevent it from meeting ICLR's bar in its current form. Most critically: (1) the proposed Open-QA metric (ES) still constrains outputs to answer-choice format via few-shot prompting, undermining the core claim about capturing generative behavior; (2) all primary experiments use one model on one benchmark, making family-level claims difficult to trust; (3) the hyperparameter selection criterion is unspecified, potentially confounding the MCQ vs. Open-QA comparison; and (4) no statistical uncertainty is reported despite the sensitivity of unlearning to optimization settings. The paper is closest in spirit to a rigorous benchmarking study, but the experimental rigor expected of such studies at ICLR—multiple models, confidence intervals, careful metric validation—is not met. A significant revision addressing the Open-QA metric's design, adding at least one additional model/benchmark, and providing statistical reliability measures would substantially improve the contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents a comprehensive, full-stack analysis of recent LLM unlearning methods by categorizing twelve representative approaches into three methodological families: divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning. It critically demonstrates that conventional multiple-choice question (MCQ) benchmarks offer a narrow and often misleading view of unlearning effectiveness and utility retention, advocating for Open-QA metrics to better capture generative behavior. Finally, it provides a fine-grained robustness assessment across in-domain relearning, out-of-domain fine-tuning, quantization, and jailbreak attacks, revealing distinct vulnerability patterns and interdependencies across method families.

### Strengths
1. **Principled and Structured Taxonomy:** The paper effectively synthesizes a fragmented research landscape by grouping twelve recent methods into three clear families based on their optimization objectives and mechanistic principles (Sec. 3). This categorization provides a highly usable conceptual framework that helps researchers quickly contextualize design choices and trade-offs.
2. **Compelling Empirical Critique of MCQ-Only Evaluation:** The authors rigorously demonstrate the limitations of answer-selection metrics by showing that MCQ can mask severe generative collapse. For instance, Fig. 1 and Appendix B reveal that divergence-driven methods like NPO achieve high MCQ success through logit collapse (suppressing all options), whereas representation misalignment (RMU) reshapes relative distributions without destroying generative capacity. This directly addresses a critical blind spot in current literature.
3. **Nuanced Robustness Analysis:** The paper distinguishes between multiple model-level attack vectors (in-domain relearning vs. out-of-domain fine-tuning vs. quantization) and correlates them with input-level jailbreaks (Fig. 2, Fig. 4). The finding that jailbreak robustness aligns more closely with in-domain relearning than out-of-domain fine-tuning, coupled with the observation that method families exhibit complementary robustness profiles, provides actionable guidance for designing resilient unlearning pipelines.

### Weaknesses
1. **Limited Experimental Scope and Generalization:** All primary experiments are conducted on a single architecture and scale (Llama-3 8B Instruct) using predominantly one benchmark (WMDP-Bio). While supplementary references to MUSE exist in the appendix, they are not integrated into the main analysis. For ICLR, findings claiming to guide community-wide evaluation practices require validation across multiple model families (e.g., Mistral, Qwen), sizes, and diverse unlearning tasks (e.g., TOFU for synthetic memorization) to ensure the observed trade-offs are not benchmark- or architecture-specific artifacts.
2. **Methodological Tension in the Open-QA Framework:** The proposed Open-QA evaluation relies on an Entailment Score (ES) that uses few-shot prompting to restrict model outputs to an `A/B/C/D` format before applying an NLI model (Appendix A). This constraint contradicts the goal of capturing free-form generative behavior, as it forces the model into a multiple-choice paradigm during inference. Consequently, the metric may conflate format compliance with actual knowledge retention and limits applicability to truly open-ended generation scenarios.
3. **Reproducibility and Implementation Gaps:** While Appendix A lists hyperparameter grids and step counts, it omits critical details required for reproducibility: random seeds, exact computational budget (GPU hours/memory), precise adversary parameters for jailbreaking (e.g., GCG iteration counts, token budgets), and no mention of open-sourcing the evaluation pipeline. ICLR places strong emphasis on reproducibility, and these omissions hinder independent verification and adoption of the proposed evaluation suite.

### Novelty & Significance
- **Novelty:** Moderate-High. The paper does not introduce a new unlearning algorithm but offers high conceptual and analytical novelty by restructuring the evaluation paradigm. The distinction between MCQ and Open-QA perspectives, the granular breakdown of robustness vectors, and the identification of over-forgetting mechanisms represent fresh contributions to a field suffering from metric fragmentation.
- **Clarity:** High. The manuscript is well-structured, with clear sectioning, intuitive figure design, and logical progression from taxonomy → evaluation → robustness. The mathematical formulations and methodological descriptions are accessible to both practitioners and theorists.
- **Reproducibility:** Low-Moderate. The experimental protocol is described at a high level, but missing seeds, attack specifications, and absence of code/public artifacts limit immediate reproducibility. The reliance on proprietary or external components (e.g., GPT-4o for QA reformatting, specific NLI models) without detailed versioning further complicates exact replication.
- **Significance:** High. The paper directly addresses pressing community pain points: inconsistent evaluation standards, hidden utility degradation, and fragmented robustness claims. Its findings are likely to shape future benchmarking standards and influence how ICLR reviewers assess unlearning submissions going forward.

### Suggestions for Improvement
1. **Expand Experimental Validation:** Incorporate at least one additional model architecture/size (e.g., Mistral-7B or Qwen-14B) and a secondary benchmark like TOFU or MUSE into the main text. Demonstrating that the MCQ vs. Open-QA discrepancy and robustness trade-offs persist across architectures will significantly strengthen the generalizability claims expected at ICLR.
2. **Refine and Validate the Open-QA Methodology:** Decouple the ES metric from strict A-D format constraints. Consider supplementing it with a truly open-ended metric, such as instruction-tuned LLM-as-a-judge evaluation or semantic fidelity scores that do not restrict output format. Additionally, report NLI model calibration or inter-metric agreement to validate ES reliability across diverse generation styles.
3. **Strengthen Reproducibility Documentation:** Provide a complete reproducibility checklist in the appendix: explicit random seeds for unlearning and attacks, exact GCG/jailbreak generation parameters, hardware specifications, and a clear commitment to release code/evaluation scripts. If proprietary APIs (e.g., GPT-4o) were used for data preprocessing, describe alternatives or provide the prompt/template to enable fully open reproduction.
4. **Clarify Taxonomic Boundaries and Hybrid Methods:** Briefly address how hybrid approaches (e.g., methods combining representation misalignment with preference optimization) fit into the three-family taxonomy. Discuss whether methods that fall outside this categorization exist and how the proposed evaluation framework would adapt to them, ensuring the taxonomy remains robust to future methodological innovations.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Pareto Frontier Curves:** Plot full UE vs. UT trade-off curves for each method family rather than single hyperparameter points. Without frontiers, claims that one family "outperforms" another are unsubstantiated and likely depend on specific tuning choices.
2. **Stronger Robustness Attacks:** The current relearning attack (100–250 steps) is too weak to verify robustness claims against serious adversaries. Increase fine-tuning duration or data size to demonstrate whether knowledge is truly erased or merely suppressed temporarily.
3. **Statistical Significance Testing:** All results are reported as single-run point estimates without error bars. Run multiple seeds to prove that observed differences (e.g., NPO vs. RMU on Open-QA) are statistically significant and not initialization noise.
4. **Cross-Dataset Validation:** The core claim about MCQ vs. Open-QA divergence relies heavily on WMDP. Validate these findings on TOFU or MUSE to ensure the metric discrepancy is not specific to the biosecurity domain or dataset structure.

### Deeper Analysis Needed (top 3-5 only)
1. **Metric Contradiction:** The paper claims Open-QA captures "free-form generation," yet Appendix A states outputs are restricted to MCQ options (A–D) via few-shot prompting. Analyze why this constrained generation differs fundamentally from standard MCQ accuracy to justify the new metric.
2. **Collapse vs. Unlearning Distinction:** When divergence methods produce nonsensical text (Table A1), distinguish whether this is successful unlearning or model collapse. Current metrics penalize both equally, obscuring whether the model is safe or simply broken.
3. **Hyperparameter Sensitivity:** Analyze how sensitive the UE–UT trade-off is to regularization strength ($\lambda$) across families. If one method family is vastly more sensitive, the comparison is unfair without showing performance across a wider range of tuning parameters.
4. **Taxonomy Novelty:** Explicitly differentiate the proposed 3-family taxonomy from prior surveys (e.g., Zhang et al., 2024; Maini et al., 2024). Without clear theoretical distinction, the taxonomy risks being viewed as a superficial regrouping rather than a novel contribution.

### Visualizations & Case Studies
1. **Qualitative Generation Examples:** Provide clear, side-by-side generated responses for Forget vs. Retain queries across all three families. Current tables are insufficient to validate whether "nonsensical" outputs are consistent refusals or incoherent gibberish.
2. **Knowledge Recovery Trajectories:** Plot UE accuracy vs. fine-tuning steps during robustness attacks instead of just final states. This reveals the *rate* of knowledge return, distinguishing true robustness from delayed recovery.
3. **Expanded Loss Landscapes:** Visualize loss landscapes for representatives of all three families, not just TAR and RMU+LAT. This is necessary to support the claim that divergence-driven methods have smoother landscapes contributing to robustness.

### Obvious Next Steps
1. **Human Evaluation of Open-QA:** Replace automatic Entailment Scores with human ratings for generated responses. Verify if automatic metrics align with human perception of "forgotten" vs. "broken" to validate the proposed evaluation pipeline.
2. **Compute Efficiency Reporting:** Report training time and GPU hours for each method. A "Full-Stack" view is incomplete without analyzing whether robust methods incur prohibitive computational costs compared to baselines.
3. **Standardized Robustness Protocol:** Propose a fixed benchmark protocol for relearning attacks (steps, data ratio, LR). Current ad-hoc settings make it difficult for future work to build upon or fairly compare against these robustness claims.

# Final Consolidated Review
## Summary

This paper provides a comprehensive "full-stack" analysis of LLM unlearning methods, contributing a taxonomy of twelve representative methods grouped into three families (divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning), a critical evaluation of conventional MCQ-based metrics alongside proposed Open-QA metrics, and a fine-grained robustness assessment across model-level and input-level attacks. The central insight is that MCQ-based evaluations obscure genuine generative behavior and that robustness dimensions (in-domain relearning vs. out-of-domain fine-tuning) exhibit distinct profiles across method families.

## Strengths

- **Compelling empirical critique of MCQ-only evaluation:** The paper demonstrates that MCQ accuracy can mask severe generative collapse. Figure A1's logit analysis shows that NPO achieves high UEMCQ by uniformly suppressing all answer options rather than genuinely forgetting—a finding with practical implications for how unlearning success is assessed. The quantitative results across all 12 methods in Figure 1 provide systematic evidence beyond a single example.

- **Novel robustness analysis with actionable insights:** The distinction between in-domain relearning (analogous to adversarial robustness) and out-of-domain fine-tuning (analogous to distribution shift) is conceptually valuable. The finding that jailbreak robustness (RobJA) correlates more strongly with in-domain relearning (RobReL) than out-of-domain fine-tuning (RobFT) provides a meaningful mechanistic interpretation that was not established in prior work.

- **Useful taxonomy that enables downstream insights:** While the categorization itself is intuitive, the paper leverages the taxonomy to reveal family-level patterns: divergence-driven methods exhibit stronger in-domain relearning robustness but are more prone to over-forgetting (Figure 1b), while representation misalignment methods show stronger out-of-domain fine-tuning robustness (Figure 2). This structure-to-insights mapping is the paper's genuine contribution.

- **UT Open-QA reveals hidden utility degradation:** The finding that NPO maintains MMLU accuracy but degrades substantially on IFEval/GSM8K (Figure 1b) is important and not visible through MCQ-only evaluation.

## Weaknesses

- **Open-QA metric design contradicts its stated purpose:** The paper claims Open-QA captures "free-form generation," yet Appendix A states that outputs are constrained to A/B/C/D format via few-shot prompting before applying the NLI model. This gates evaluation on the same answer choices as MCQ, merely soft-scoring via entailment rather than exact match. The metric may therefore fail to capture what it claims: truly open-ended generative behavior on forget-set queries without format restrictions. This tension is significant for a paper whose central thesis is that MCQ evaluation is insufficient.

- **ES metric conflates successful unlearning with model degeneracy:** High over-forgetting (producing nonsensical text) and successful knowledge removal (producing coherent but incorrect responses) both yield low ES scores. The paper acknowledges over-forgetting as a problem but the proposed metric does not distinguish between these two fundamentally different outcomes. A model that outputs gibberish and a model that genuinely cannot recall have indistinguishable ES values.

- **Limited experimental generalization:** All primary experiments use Llama-3 8B Instruct on WMDP-Bio. Table A2 provides limited quantization results on MUSE with only RMU (no NPO), and other benchmarks are not integrated into the main analysis. Family-level claims about robustness patterns (e.g., "divergence-driven optimization is generally more resilient to in-domain relearning") require validation across additional architectures to ensure they are not specific to Llama-3's GQA or RoPE characteristics.

- **NLI model selection lacks validation:** The Entailment Score relies on a single NLI classifier (tasksource, Sileo 2023) without ablation or correlation with human judgments. Different NLI models could produce different absolute scores and potentially different method rankings, which is concerning for a metric positioned as the paper's primary evaluation contribution.

- **Hyperparameter selection criterion is unspecified:** The paper describes grid search ranges but does not state which metric determined optimal hyperparameters. If MCQ-based UE/UT was used for selection, the subsequent MCQ vs. Open-QA comparison is confounded: methods may have been optimized for MCQ performance, making Open-QA underperformance a tuning artifact rather than an intrinsic property.

- **No statistical uncertainty reported:** All results appear as single-run point estimates. Given the known sensitivity of unlearning methods to random seeds and learning rates, the absence of confidence intervals makes it difficult to assess whether observed differences (e.g., NPO vs. RMU in Figure 2) are meaningful.

## Nice-to-Haves

- **Pareto frontier curves for UE–UT trade-offs:** Plotting full trade-off curves rather than single hyperparameter points would substantiate claims that one family "outperforms" another.

- **Truly open-ended evaluation format:** Removing the A/B/C/D constraint from Open-QA evaluation would align the metric with its stated purpose of capturing free-form generation.

- **Human evaluation of ES metric:** Validation that low ES corresponds to human perception of "forgotten" rather than "broken" would strengthen the metric's credibility.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Missing research question Q in introduction:** This is a PDF extraction artifact acknowledged by the reviewer, not an actual paper problem.

- **Single example for MCQ-OpenQA discrepancy:** The reviewer claimed only Table A1 supports this finding, but Figure 1 provides quantitative analysis across all 12 methods. This criticism misreads the paper.

- **Taxonomy lacks novelty:** The reviewer claimed the groupings are "intuitive." However, the paper's contribution is not the taxonomy itself but the insights it enables in Sections 4–5—family-level robustness patterns and over-forgetting mechanisms. The taxonomy serves its purpose.

- **Writing issues from PDF extraction:** Formatting artifacts noted by the reviewer are parser issues, not paper problems.

## Novel Insights

The correlation between jailbreak robustness (RobJA) and in-domain relearning robustness (RobReL)—versus the weaker correlation with out-of-domain fine-tuning (RobFT)—provides a mechanistic interpretation: adversarial prompting and in-domain relearning both probe forget-domain knowledge through similar perturbation regimes, while out-of-domain fine-tuning represents a fundamentally different attack surface. This insight suggests that improving robustness to one type of attack (e.g., relearning) may transfer to others (e.g., jailbreaking) in ways that out-of-domain fine-tuning robustness does not. Additionally, the observation that divergence-driven methods achieve MCQ success through logit collapse (uniformly suppressing all options) rather than distribution reshaping reveals a blind spot in current evaluation practices: the method "works" by becoming unable to generate coherent responses, not by genuinely forgetting.

## Suggestions

- Decouple the Open-QA evaluation from the A/B/C/D format constraint to properly capture free-form generative behavior. Consider evaluating responses to open-ended forget-set queries without predefined answer choices, using semantic similarity or factual consistency metrics.

- Report statistical uncertainty (at minimum, standard deviations across multiple seeds) for all quantitative results to establish that differences between methods are statistically meaningful.

- Clarify the hyperparameter selection criterion: state explicitly which metric(s) determined the final configuration for each method.

- Validate the Entailment Score metric by reporting results with at least one alternative NLI model or, ideally, correlation with human judgments on a sample of responses.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 4.0]
Average score: 3.5
Binary outcome: Reject
