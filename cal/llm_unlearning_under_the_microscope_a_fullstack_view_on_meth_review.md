=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary

This paper presents a systematic investigation of LLM unlearning, organizing twelve methods into three families (divergence-driven optimization, representation misalignment, rejection-based targeted unlearning) and arguing that conventional MCQ-based evaluation provides an incomplete picture. The authors introduce Open-QA metrics (particularly entailment score via an NLI model) to capture generative behavior post-unlearning, and provide a multi-faceted robustness analysis across in-domain relearning, out-of-domain fine-tuning, quantization, and jailbreaking attacks.

## Strengths

- **Compelling demonstration that MCQ evaluation obscures real generative behavior.** The paper provides concrete evidence—Table A1 shows NPO/RMU appearing successful under MCQ but producing incoherent gibberish under Open-QA, and Figure 1 quantifies this systematically across all 12 methods. This is a timely and important empirical critique that the community needs.
- **Revealing family-specific robustness profiles.** The finding that divergence-driven methods are more robust to in-domain relearning (RobReL) but less robust to out-of-domain fine-tuning (RobFT), while representation misalignment methods show the opposite pattern (Figure 2), is a nuanced and actionable insight. The correlation between jailbreak robustness (RobJA) and relearning robustness (Figure 4b) further connects previously separate lines of inquiry.
- **The IDK+AP warm-started with DPO mitigation (Appendix B, Figure A2)** demonstrates that the paper's analytical framework directly yields practical improvements, not just diagnostic observations.
- **Logit-level mechanistic evidence for over-forgetting** (Figure A1) showing NPO collapses option logits to near-zero while RMU reshapes their relative ordering provides a concrete, interpretable explanation for the MCQ–Open-QA divergence.

## Weaknesses

### Major:

- **The entailment score (ES) metric is insufficiently validated, which undermines the paper's core methodological contribution.** The ES relies entirely on an external NLI model (Sileo, 2023) to judge whether unlearned outputs entail forbidden knowledge. No validation is provided for this NLI model's reliability on the biosecurity/cybersecurity domain of WMDP, where technical paraphrase and synonymy could easily cause both false positives (hallucinated text accidentally entailing the answer) and false negatives (correct knowledge expressed in unfamiliar terminology classified as non-entailing). Since the paper's central claim is that Open-QA evaluation is *better* than MCQ, the burden of proof is on demonstrating that the proposed alternative metric is trustworthy. Without any domain-specific validation, sensitivity analysis to the NLI model choice, or comparison against human judgments, this key pillar remains unestablished.

- **Limited empirical scope restricts generalizability of the claimed family-level behaviors.** All experiments use a single model (Llama-3 8B Instruct) and primarily a single benchmark (WMDP-Bio). Unlearning dynamics are known to change with model scale—larger models may exhibit different forgetting-retention tradeoffs—and the WMDP-Bio domain (hazardous knowledge removal) may not be representative of copyright (MUSE), privacy (TOFU), or other unlearning scenarios. The paper acknowledges this in limitations but does not provide any evidence that the family-level rankings (e.g., "representation misalignment generally outperforms rejection-based") hold beyond this narrow setting. Given that the paper aspires to deliver "actionable guidance for designing and evaluating future methods," this is a significant gap.

### Minor:

- **DPO's categorization under "rejection-based targeted unlearning" creates taxonomic ambiguity.** DPO is fundamentally a divergence-based objective (optimizing against a reference policy), and the paper itself introduces it in the divergence-driven section (Eq. 2) before re-categorizing it. The justification—that DPO is *adapted* for rejection—makes the taxonomy depend on application intent rather than mechanism. This conflation may confuse readers and weakens the claimed clean separation between families.

- **Quantization robustness generalization from only two methods.** The claim that "knowledge removal is generally more robust to post-unlearning quantization than data-centric unlearning" (Table A2) is based solely on NPO and RMU. Extrapolating to entire method families from two data points is not well-supported.

- **Single checkpoint for relearning attack severity.** In-domain relearning uses exactly 100 fine-tuning steps (Appendix A). Whether this constitutes a sufficient attack is unclear; a convergence analysis (UE vs. fine-tuning steps) would make robustness claims more rigorous. If 100 steps only partially recovers knowledge, the reported robustness rankings could be misleading.

- **Only WMDP-Bio is evaluated; WMDP-Cyber and WMDP-Chem are omitted.** Since the paper adopts WMDP as its primary benchmark, leveraging all three available domains would test whether the observed family-level patterns are consistent across different knowledge types.

### Trivial:

- **The Introduction states "the key research question we aim to address is:" but the explicit question appears missing** (likely a formatting artifact from PDF extraction). This does not affect the scientific content.

## Nice-to-Haves

- Computational cost comparison across the 12 methods (training time, GPU hours, memory), which would strengthen the paper's claim to "actionable guidance."
- Human evaluation of a sample of Open-QA outputs to validate (or calibrate) the automatic entailment score, even on a small subset.
- Statistical significance tests or variance across multiple runs for the UE–UT tradeoff rankings in Figure 1.
- Loss landscape visualizations (à la Figure A3) for methods across all three families, not just TAR and RMU+LAT.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Missing no-unlearning baseline in robustness figures"** (Spark Finder): The Original (pre-unlearned) model is included in Figures 2–4 as a baseline. The reviewer missed this.
- **"Need concrete examples showing MCQ success with Open-QA failure"** (Spark Finder): Table A1 already provides exactly this—NPO and RMU select the wrong MCQ option (apparent success) but produce gibberish in Open-QA (revealed failure).
- **"Robustness metric definition is confusing"** (Harsh Critic): The paper explicitly states "lower accuracy (lighter color) indicates stronger robustness" and plots post-attack UE, which is standard. The phrasing is clear.
- **"Missing research question"** (Harsh Critic): The text shows "the key research question we aim to address is:" followed by "To tackle (Q)..." — this appears to be a PDF parsing artifact, not an authoring error. Even if real, it is a trivial formatting issue.
- **Demand for theoretical framework/guarantees** (Harsh Critic, drawing from external reviews): This is an empirical systems/benchmarking paper. Demanding theoretical privacy guarantees is scope creep inconsistent with the paper's stated contribution.
- **Demand for error bars/confidence intervals** (Harsh Critic/Spark Finder): For a large-scale comparative study of 12 methods across multiple attack dimensions, single-run evaluation is the community norm. Flagging as nice-to-have rather than a weakness.

## Novel Insights

The correlation between input-level jailbreak robustness (RobJA) and model-level in-domain relearning robustness (RobReL)—but not out-of-domain fine-tuning robustness (RobFT)—is a genuinely novel observation. It suggests that the adversarial perturbation structure (domain-aligned vs. domain-agnostic) may be more predictive of robustness transferability than the perturbation modality (input vs. weight). If validated at scale, this could inform unified defense strategies: methods hardened against relearning may simultaneously gain jailbreak resistance, while out-of-domain fine-tuning robustness requires separate architectural considerations.

## Suggestions

- Validate the entailment score metric by (a) comparing ES against human annotations on a subset of WMDP outputs, (b) testing sensitivity to the choice of NLI model, and (c) analyzing failure cases where ES and MCQ disagree to confirm ES is capturing meaningful differences rather than NLI noise.
- Add results on at least one additional model scale (e.g., Llama-3 70B) or one additional WMDP domain (Cyber or Chem) to support the generality of family-level claims.
- Clarify the DPO categorization by either (a) acknowledging its dual nature explicitly in the taxonomy discussion, or (b) redefining the families along purely mechanistic rather than intent-based lines.
- Report a convergence curve for in-domain relearning (UE vs. fine-tuning steps) rather than a single checkpoint, to ensure robustness rankings are not sensitive to attack duration.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 4.0]
Average score: 3.5
Binary outcome: Reject
