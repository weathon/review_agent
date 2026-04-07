=== CALIBRATION EXAMPLE 45 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "A Full-Stack View" implies comprehensive breadth — multiple models, architectures, benchmarks, and task types. In practice, all core experiments use a single model (Llama-3 8B Instruct) and a single benchmark (WMDP-Bio). This disconnect between the framing and the actual scope is the paper's most persistent weakness. The abstract correctly identifies that MCQ-based evaluation overstates unlearning success, and the claim that robustness requires finer-grained distinction (in-domain relearning vs. out-of-domain fine-tuning) is well-supported. However, the claim of delivering a "full-stack revisit" is inflated relative to what is delivered.

---

### Introduction & Motivation

The problem is well-motivated and the three contributions are clearly stated. The paper correctly identifies that neither the evaluation community nor the methods community has produced a systematic cross-method comparison including Open-QA. The differentiation from concurrent work (Che et al., 2025; Hu et al., 2025; Feng et al., 2025) is made, but the differentiation from Feng et al. (2025), which makes a very similar argument about current unlearning evaluations being inconclusive, deserves much more explicit treatment. The paper should directly compare what questions Feng et al. (2025) leaves open and how the present work answers them.

The "research question Q" is referenced at lines 76–79 but the actual question text appears to be missing from the extracted paper body (likely a PDF/section-ordering issue), making the introduction somewhat incomplete as written.

---

### Section 3 – Taxonomy

The three-family taxonomy (divergence-driven, representation misalignment, rejection-based) is reasonable and covers the major methodological variants. However, several concerns arise:

**Selection criteria are absent.** The paper examines "twelve representative methods" but never explains the inclusion/exclusion criteria. Why these 12 and not others? Methods like DEPN (Wu et al., 2023), ECP (Liu et al., 2024a), or pruning-based approaches are mentioned in the related work but absent from the taxonomy. The reader cannot assess whether the comparison is representative or cherry-picked.

**Taxonomy novelty is modest.** Groupings similar to this taxonomy (gradient-based, representation-based, output-targeted) appear in prior surveys. The paper should make a stronger argument for why this particular three-way split provides the most useful analytical lens — particularly why "rejection-based" deserves to be its own family rather than being a variant of divergence-driven optimization (since IDK and DPO for unlearning also use loss-based updates).

**The formulation in Eq. (1)** is standard but the paper uses it mainly as organizational scaffolding. It would strengthen the taxonomy to discuss, for each family, *what* in Eq. (1) distinguishes them (e.g., what ℓ_f is optimizing over, what the geometry of the loss landscape looks like) rather than just describing each method.

---

### Section 4 – Beyond MCQ: Evaluation of UE and UT

This is the paper's most substantive contribution, and the motivating example (Table A1 showing NPO producing "@nate@nate..." on Open-QA while achieving low MCQ accuracy) is compelling. However, several methodological concerns weaken the argument:

**Entailment Score design conflates format with knowledge.** The ES metric is computed after prepending 2-shot format examples that explicitly show the model "the required output format in the multiple-choice setting (e.g., 'C. tiger')." If these demonstrations nudge the model toward outputting a specific letter (say "B"), and the correct answer for a forget query *happens* to be "B," the NLI model may classify it as entailment even though the model is not demonstrating domain knowledge — it is demonstrating format compliance. The paper partially acknowledges this ("The purpose is solely to ensure that the model outputs remain restricted to the given options") but does not report what happens without few-shot examples or how sensitive results are to the choice of demonstrations. This is a significant reliability concern for the UE_Open-QA metric.

**NLI model domain mismatch.** The ES relies on Sileo (2023), a general-domain NLI classifier, to assess entailment for WMDP questions about biosecurity, synthetic biology, and chemical weapons. There is no validation that this NLI model performs reliably on this specialized domain. A simple calibration experiment (e.g., reporting NLI model accuracy on held-out WMDP QA pairs with known entailment/contradiction) would substantially strengthen the metric's credibility.

**No statistical significance.** Figure 1 reports numeric differences between methods (e.g., DPO's UTOpen-QA vs. IDK+AP's UTOpen-QA) without confidence intervals or any indication of variance across runs. Given that 125 steps of unlearning with a grid search over a small hyperparameter range is involved, results may be sensitive to seed and hyperparameter choices.

**UT benchmark selection is unexplained.** The paper adds IFEval and GSM8K for UTOpen-QA but does not justify this choice over alternatives like HELM instruction-following, HumanEval, or MT-Bench (which is already mentioned in the WMDP paper). The finding that "divergence-driven optimization" degrades IFEval and GSM8K may partly reflect sensitivity to these specific task types rather than a general utility collapse.

**The claim about rejection-based methods (Section 4, "Rethinking rejection-based methods")** is the paper's most interesting new insight: DPO for unlearning preserves utility better than previously recognized. However, the mechanism proposed ("the presence of a positive preference signal") is speculative and the proposed fix (warm-starting IDK+AP with DPO, Fig. A2) is shown for a single data point without ablation over the number of DPO warm-start steps, which is a key hyperparameter.

---

### Section 5 – Robustness

**The in-domain/out-of-domain distinction is valuable** and the analogy to adversarial vs. OOD robustness is apt. The observation that divergence-driven methods resist in-domain relearning better but representation-misalignment methods resist out-of-domain fine-tuning better is a genuine insight.

**The jailbreaking analysis (RobJA) is methodologically inconsistent.** Earlier in the paper, the authors argue that MCQ-based evaluation is insufficient and that Open-QA is necessary. Yet Figure 4(a) — the jailbreaking robustness figure — is presented *only* in terms of UEMCQ. If the main thesis is that MCQ misses important generative behavior, then a jailbreak attack that successfully elicits coherent harmful text at the generation level (even if MCQ accuracy is low) would be undetected by this measure. The paper needs UEOpen-QA results for jailbreaking to be internally consistent.

**The RobJA ≈ RobReL correlation claim (Fig. 4b)** is described as a "positive correlation" but the paper provides no quantitative correlation coefficients, p-values, or confidence intervals. The figure appears to show a scatter plot, but given only 12 methods, this is a very small-n correlation. With 12 data points and no significance test, the "positive correlation" claim is not firmly established.

**Quantization analysis is thin.** Table A2 reports results only for NPO and RMU (2 out of 12 methods) on MUSE Books, and only NPO/RMU on WMDP. The claim that "knowledge removal is generally more robust to post-unlearning quantization than data-centric unlearning" rests on 2 methods × 2 benchmarks × 2 bit-widths. Extending this to even 4-5 methods would significantly strengthen the conclusion.

**Fig. 3's comparison of quantization robustness** uses 4-bit quantization as the single representative setting. Different methods may be differently sensitive to 4-bit vs. 8-bit. The paper could benefit from showing the full quantization curve (full → 8-bit → 4-bit) for a broader set of methods.

---

### Single-Model, Single-Benchmark Scope (Cross-Cutting Major Concern)

All results in the main paper use Llama-3 8B Instruct on WMDP-Bio. The paper's taxonomy and insights are presented as general properties of method *families*, yet:

- It is unknown whether the finding "representation misalignment methods outperform divergence-driven optimization on the UE-UT tradeoff" holds for 13B, 70B, or non-Llama architectures (e.g., Mistral, Qwen).
- WMDP-Bio has a specific structure (no prior fine-tuning on the forget set; MCQ-format evaluation set; biology-focused content). It's possible the insights about rejection-based methods' relative utility are specific to this domain.
- The MUSE results in the appendix cover only 2 methods (NPO, RMU) under quantization, and the TOFU/WHP benchmarks mentioned in the taxonomy are absent from any experimental comparison.

For a paper claiming a "full-stack view" with insights about method *families*, validation across at least 2 model sizes or 2 benchmarks for the central UE-UT tradeoff analysis (Section 4) should be considered a minimum standard.

---

### Writing & Clarity

The paper is generally clear in its conceptual contributions. The section ordering in Section 3 is confusing (likely a PDF artifact), with the taxonomy discussion split non-contiguously. The research question "Q" referenced in the introduction is not visibly rendered in the available text. These structural issues should be resolved.

The paper's 9-page main body is quite compact for a study covering 12 methods × 4 robustness dimensions × 2 evaluation modalities; some findings (especially jailbreaking, quantization) feel rushed and underexplained.

---

### Limitations & Broader Impact

The limitations section (Appendix D) appropriately acknowledges limited method coverage, limited attack types, and reliance on automatic metrics. However, it conspicuously omits the single-model limitation — arguably the most significant constraint on the paper's generalizability. The broader impact section is generic and adds little substance.

---

## Overall Assessment

This paper makes a genuine and useful contribution to the LLM unlearning literature by (1) providing a structured taxonomy of 12 methods organized around three methodological families, (2) demonstrating through concrete examples and systematic experiments that MCQ-based evaluation can misrepresent both unlearning effectiveness and utility retention, and (3) unpacking the underappreciated distinction between in-domain relearning robustness and out-of-domain fine-tuning robustness. The motivating example (NPO producing incoherent tokens that fool MCQ metrics) is compelling and the conceptual framing of robustness dimensions is valuable. However, the central empirical claims rest almost entirely on one model (Llama-3 8B Instruct) and one benchmark (WMDP-Bio), which significantly limits the generalizability of the family-level conclusions. The proposed Open-QA metric (entailment score) has unresolved design issues — the few-shot prompting may conflate format compliance with knowledge retention, and the NLI model has no domain validation. The jailbreaking analysis is reported only in MCQ terms, creating an internal inconsistency with the paper's own thesis. Several key claims (RobJA ≈ RobReL correlation, quantization robustness advantage of knowledge removal) lack statistical rigor. At ICLR, this sits below the acceptance bar in its current form: the core insights are valuable but the empirical foundation needs broader model/benchmark coverage and the metric design needs additional validation before the family-level generalization claims can be trusted.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents a comprehensive empirical study of LLM unlearning, proposing a taxonomy of 12 methods across three families and introducing open question-answering (Open-QA) metrics to complement standard multiple-choice evaluations. Through extensive experiments on the WMDP benchmark with Llama-3 8B, the authors demonstrate that MCQ metrics often mask over-forgetting and utility loss, revealing fundamental trade-offs between unlearning effectiveness and retention. They further provide a multi-faceted robustness analysis covering model-level attacks and input-level jailbreaking, offering actionable guidelines for future algorithm design.

### Strengths
1.  **Critical Evaluation of Standard Metrics (Sec 4, Fig 1):** The paper convincingly argues that MCQ accuracy is insufficient for unlearning evaluation. Evidence is provided in Table A1 and Fig 1-(a), showing that divergence-driven methods like NPO can achieve high unlearning effectiveness on MCQ while simultaneously destroying generative capacity on the same queries (over-forgetting), a nuance missed by prior work.
2.  **Comprehensive Robustness Analysis (Sec 5, Fig 2-4):** Unlike prior work often focusing on single attack types, this work systematically evaluates in-domain relearning, out-of-domain fine-tuning, quantization, and jailbreaking. The findings that resilience to model-level perturbations does not guarantee input-level security (Fig 4) provide a necessary correction to current robustness assumptions.
3.  **Actionable Methodological Improvements (Appendix A2, Fig A2):** Beyond diagnosis, the paper offers prescriptive solutions, such as the proposal to warm-start IDK+AP with DPO to mitigate utility loss. This adds significant value to the community by offering a concrete strategy to address identified failure modes.

### Weaknesses
1.  **Limited Experimental Scope:** The evaluation relies primarily on a single model size (Llama-3 8B) and benchmark family (WMDP-Bio), limiting confidence in generalizing findings regarding utility retention (e.g., GSM8K/IFEval performance) to larger or different architecture families.
2.  **Dependency on External NLI Models for Open-QA:** The proposed Entailment Score (Sec 4) relies heavily on an external NLI model to judge generative output validity. Without validating the NLI model's bias or calibration against human judgment, there is uncertainty about whether the Open-QA UE metrics are more reliable than the established MCQ baseline.
3.  **Taxonomy Ambiguity on Rejection Methods:** Grouping Direct Preference Optimization (DPO) solely under "rejection-based targeted unlearning" (Sec 3) conflates general safety alignment techniques with specific forget-set unlearning, potentially biasing the comparison against methods designed specifically to modify weights for forget data (e.g., RMU, GradDiff).

### Novelty & Significance
**Novelty:** Moderate to High. While the taxonomy synthesizes existing work, the specific proposal for Open-QA metrics as a standard alongside MCQ and the granular distinction between in-domain and out-of-domain robustness attacks offer a new lens for the community.
**Significance:** High. As unlearning moves towards deployment, current evaluation standards are insufficient for safety-critical applications. This work challenges the community to adopt more rigorous generative metrics, which could prevent deployment of "unlearned" models that fail on real-world tasks or relearn forgotten data upon minor fine-tuning. It aligns well with ICLR's interest in empirical rigor and safety evaluation.

### Suggestions for Improvement
1.  **Corroborate Open-QA Metrics:** Validate the Entailment Score findings against human evaluation or additional generation-based metrics (e.g., BLEU/ROUGE on generation tasks) to ensure the Open-QA UE metrics are more reliable than the MCQ baseline, addressing the NLI dependency concern.
2.  **Expand Model Scope:** Include at least one additional large model architecture (e.g., Gemma-7B or Llama-3 70B) to verify if the trade-offs between divergence-driven and representation methods hold across scales, particularly for quantization robustness (Fig 3).
3.  **Clarify Conceptual Definitions:** Refine the Introduction and Conclusion to distinguish between "unlearning" for specific data removal versus "safety alignment" for rejection-based methods (e.g., DPO), preventing conceptual confusion regarding the nature of the weight updates in Fig 1.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Multi-Domain Validation:** Extend main text results beyond WMDP-Bio to include MUSE (copyright) or TOFU (fictional) to prove the Open-QA vs. MCQ tradeoff is not domain-specific. Without this, the claim of a "Full-Stack View" is unsupported and likely overfit to biosecurity knowledge.
2. **Human Evaluation of Open-QA Metrics:** Validate the Entailment Score (ES) metric against human judgments of factual correctness. If the NLI model used for ES is biased against unlearned model outputs, the core claim that Open-QA reveals "over-forgetting" is unreliable.
3. **Statistical Significance Testing:** Report mean and standard deviation over at least 3 random seeds for all main figures (Fig 1-4). Unlearning optimization is high-variance; without error bars, the ranking of methods (e.g., SimNPO vs. NPO) may be noise rather than signal.

### Deeper Analysis Needed (top 3-5 only)
1. **Hyperparameter Sensitivity (Pareto Fronts):** Plot UE vs. UT curves across a range of regularization strengths ($\lambda$) rather than single optimal points. This ensures the observed method family differences are not artifacts of unequal tuning budgets.
2. **Attack Convergence Analysis:** Demonstrate that the 100-step in-domain relearning attack is sufficient to reach asymptotic knowledge recovery. If the attack stops prematurely, the reported "robustness" of divergence-driven methods is artificially inflated.
3. **NLI Model Bias Audit:** Analyze whether the NLI model used for Entailment Score penalizes refusal-style outputs common in rejection-based methods. Without this, the low UE scores for rejection methods may reflect metric bias rather than actual unlearning failure.

### Visualizations & Case Studies
1. **Qualitative MCQ vs. Open-QA Mismatches:** Provide specific input-output examples where MCQ indicates success but Open-QA reveals hallucination or leakage. This concretely demonstrates the evaluation gap beyond aggregate scores.
2. **Loss Landscape Comparisons:** Extend the loss landscape visualization (currently only TAR vs. RMU+LAT) to compare Divergence vs. Representation families. This is needed to visually substantiate claims about why one family is more robust to relearning than the other.
3. **Utility Degradation Curves:** Plot utility metrics (MMLU/GSM8K) against unlearning steps to show *when* over-forgetting occurs. This reveals whether utility loss is immediate or gradual, informing the mechanism of degradation.

### Obvious Next Steps
1. **Adaptive Robustness Evaluations:** Replace standard jailbreak prompts with adaptive attacks optimized against the specific unlearning defense (e.g., optimizing against the refusal pattern). Current input-level robustness claims are weak against adaptive adversaries.
2. **Compute-Normalized Comparison:** Include training time and memory overhead for each method. Methods like NPO+SAM are computationally heavier; without efficiency metrics, the practical utility of the robustness gains is unclear.
3. **Forget-Retain Overlap Analysis:** Evaluate performance on cases where forget and retain data distributions overlap. Real-world unlearning rarely has disjoint sets, and performance here is critical for practical deployment claims.

# Final Consolidated Review
## Summary

This paper provides a systematic study of LLM unlearning methods, organizing 12 representative approaches into three methodological families (divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning). The authors demonstrate that conventional MCQ-based evaluation of unlearning effectiveness and utility retention can mask important failures in generative behavior, and introduce Open-QA metrics to address this gap. They further analyze robustness across multiple attack dimensions (in-domain relearning, out-of-domain fine-tuning, quantization, and jailbreaking), revealing that robustness to different attack types requires different methodological approaches.

## Strengths

- **Compelling demonstration of MCQ evaluation limitations:** The paper provides concrete evidence that MCQ-based evaluation can misrepresent unlearning success. Table A1 shows that NPO-unlearned models achieve low MCQ accuracy on forget queries (selecting incorrect options) while producing incoherent outputs on the same queries in Open-QA format. Figure 1(b) further demonstrates that divergence-driven methods achieve similar UT_MCQ to representation misalignment methods but substantially worse UT_Open-QA, revealing over-forgetting that MCQ metrics miss entirely.

- **Systematic multi-dimensional robustness analysis:** The paper examines four distinct attack types—in-domain relearning, out-of-domain fine-tuning, quantization, and jailbreaking—within a unified framework. The finding that divergence-driven methods resist in-domain relearning better while representation misalignment methods resist out-of-domain fine-tuning better (Figure 2) is a substantive insight that helps practitioners choose methods based on threat models.

- **Clear conceptual framework for robustness:** The analogy between in-domain relearning (adversarial robustness) and out-of-domain fine-tuning (OOD robustness) provides a useful lens for understanding why different method families exhibit different robustness profiles. The correlation analysis in Figure 4(b) between jailbreaking and in-domain relearning robustness offers a principled way to think about attack relationships.

## Weaknesses

- **Empirical claims rest on a single model and single benchmark:** All main-paper experiments use Llama-3 8B Instruct on WMDP-Bio. The paper presents family-level generalizations ("representation misalignment generally outperforms divergence-driven optimization on the UE-UT tradeoff") without validation across model scales, architectures, or domains. The appendix includes MUSE results for only NPO and RMU under quantization—insufficient to establish that findings generalize beyond the specific experimental setup. For a paper claiming a "full-stack view," this scope limitation significantly constrains confidence in the broader conclusions.

- **Entailment Score metric has unresolved design issues:** The Open-QA evaluation uses few-shot prompting to constrain outputs to A–D format, then applies a general-domain NLI model (Sileo, 2023) to judge entailment. Two concerns arise: (1) the few-shot format demonstrations may nudge models toward specific letter patterns that could inflate ES when these coincide with correct answers; (2) no validation is provided that the NLI model reliably judges entailment for specialized biosecurity content. The paper acknowledges the format motivation but does not analyze sensitivity to demonstration choice or validate the NLI model's domain competence.

- **Internal inconsistency in evaluation methodology:** The paper's central thesis is that MCQ metrics provide "only a narrow perspective" and "obscure the actual generation behavior." Yet Figure 4(a), which presents the primary jailbreaking robustness results, reports only UEMCQ. If MCQ misses important generative failures for the main evaluation (as the paper argues), it may also miss coherent harmful text elicited by jailbreak prompts. The paper should report UE_Open-QA for jailbreaking to be methodologically consistent with its own argument.

- **No statistical significance testing:** Figures 1–4 report point estimates without confidence intervals, standard deviations across runs, or significance tests. Unlearning optimization is known to be sensitive to random seeds and hyperparameter choices. Without variance quantification, it is unclear whether observed differences between methods (e.g., SimNPO vs. NPO in Figure 1(c)) reflect genuine performance gaps or optimization noise.

- **Missing research question in paper text:** The introduction states "the key research question we aim to address is:" followed by "To tackle (Q), we first draw methodological insights..." The actual question text appears to be absent or improperly formatted, creating a structural gap in the paper's motivation.

## Nice-to-Haves

- **Multi-domain validation:** Extending the UE-UT tradeoff analysis to MUSE (copyright) or TOFU (fictional entities) would strengthen claims that the MCQ/Open-QA gap is not domain-specific.

- **Pareto frontier analysis across hyperparameters:** Plotting UE vs. UT curves across a range of regularization strengths (λ) would clarify whether method family differences persist under equal tuning budgets, or whether apparent advantages stem from unequal hyperparameter search.

- **Attack convergence analysis:** Showing that 100-step in-domain relearning reaches asymptotic knowledge recovery would establish that the reported robustness values are not artificially inflated by early stopping.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Taxonomy novelty criticism:** The harsh reviewer suggested the taxonomy lacks novelty because similar groupings appear in prior surveys. However, the paper's contribution is not the taxonomy per se but the systematic empirical comparison enabled by it. Taxonomies are organizational tools; their utility is demonstrated through the insights they facilitate.

- **UT benchmark selection demand:** The reviewer questioned why IFEval and GSM8K were chosen over alternatives. These are reasonable, well-established benchmarks for instruction-following and mathematical reasoning. Requesting justification for every benchmark choice is scope creep—the selected benchmarks cover relevant utility dimensions and are widely used.

- **Mechanism speculation criticism for DPO:** The reviewer argued that the DPO utility preservation mechanism is speculative. However, the empirical observation (DPO preserves utility better than IDK+AP) stands regardless of mechanism explanation completeness. The warm-start proposal (Figure A2) provides practical value even with incomplete mechanism understanding.

- **Quantization coverage complaint:** The claim that quantization analysis is "thin" because it covers only NPO and RMU is weakened by recognizing that quantization is one of four robustness dimensions analyzed. The paper provides substantial novel analysis elsewhere, making this a valid but not critical limitation.

## Novel Insights

The correlation between jailbreaking robustness (RobJA) and in-domain relearning robustness (RobReL)—but not out-of-domain fine-tuning robustness (RobFT)—offers a novel lens for understanding unlearning defense. The analogy to adversarial vs. OOD robustness in machine learning is apt: both jailbreaking and in-domain relearning represent "worst-case" adversarial scenarios in their respective domains (input-space vs. weight-space), while out-of-domain fine-tuning represents distribution shift. This suggests that practitioners focused on adversarial robustness may need to consider in-domain relearning and jailbreaking together, while those concerned with general fine-tuning safety should focus on representation misalignment methods. The finding that divergence-driven methods achieve MCQ-based unlearning effectiveness by "collapsing logits" rather than removing knowledge (Figure A1) provides mechanistic insight into why these methods exhibit over-forgetting—a pattern not highlighted in prior work.

## Suggestions

- Add validation of the Entailment Score metric: report agreement with human judgments on a sample of WMDP question-answer pairs, or at minimum show that the NLI model performs reliably on domain-specific content through calibration experiments.

- Report UE_Open-QA results for jailbreaking robustness to maintain methodological consistency with the paper's core thesis about MCQ limitations.

- Include at least one additional model size (e.g., Llama-3 70B or a different architecture family) to strengthen generalizability claims, or explicitly frame conclusions as specific to 8B-scale models on WMDP-Bio.

- Add confidence intervals or standard deviations across multiple runs with different seeds to establish that observed differences between methods are statistically meaningful.

- Fix the missing research question (Q) in the introduction to complete the paper's motivation structure.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 4.0]
Average score: 3.5
Binary outcome: Reject
