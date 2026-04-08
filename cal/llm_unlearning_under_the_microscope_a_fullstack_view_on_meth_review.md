=== CALIBRATION EXAMPLE 24 ===

# Final Consolidated Review
##Summary

This paper presents a systematic investigation of LLM unlearning, organizing twelve methods into three families (divergence-driven optimization, representation misalignment, rejection-based targeted unlearning) and arguing that conventional MCQ-based evaluation overstates unlearning success. It introduces Open-QA metrics (entailment score) to capture generative behavior and provides multi-faceted robustness analysis across model-level attacks (relearning, fine-tuning, quantization) and input-level jailbreaking.

## Strengths

- **Principled taxonomy that reveals family-level structure:** The categorization into three methodological families is not merely organizational—it enables the paper's key insight that each family exhibits characteristic failure modes. For instance, divergence-driven methods achieve UE by collapsing logits (Fig. A1), while representation misalignment reshapes relative distributions. This family-level framing makes the comparative findings more interpretable and generalizable than method-by-method comparisons.

- **Compelling demonstration that MCQ evaluation is insufficient:** The paper shows that NPO and RMU can achieve identical UEMCQ while having vastly different UEOpen-QA (Fig. 1a), and that NPO's UTMCQ appears comparable to RMU's while its UTOpen-QA is much lower. This is a substantive and well-supported finding that directly challenges prevailing evaluation practice. The logit analysis in Fig. A1 provides mechanistic grounding for why: NPO uniformly suppresses all candidate scores while RMU reorders them.

- **Multi-faceted robustness analysis with non-obvious findings:** The discovery that divergence-driven methods are more resilient to in-domain relearning while representation misalignment is more robust to out-of-domain fine-tuning (Fig. 2) is a nuanced and practically important result. Similarly, the observation that RMU+LAT fails to improve over RMU despite adversarial design—explained via loss landscape analysis (Fig. A3)—is a valuable negative finding.

- **Rehabilitation of rejection-based methods:** The paper makes a persuasive case that DPO has been underrated due to MCQ-centric evaluation: it achieves the best UEOpen-QA among rejection-based methods while maintaining UTOpen-QA comparable to the original model, a profile invisible under MCQ alone.

## Weaknesses

- **Single model and primary dataset limit generalizability of family-level claims:** All main results use Llama-3 8B Instruct on WMDP-Bio. The paper makes strong claims about the characteristics of entire methodological families (e.g., "divergence-driven optimization generally achieves better UEMCQ and UEOpen-QA"), but whether these hold across model scales (70B), architectures (Mistral, Gemma), or unlearning domains (MUSE for copyright, TOFU for privacy) is unknown. The supplementary MUSE analysis in Table A2 is limited to quantization only and involves just two methods. This is a meaningful gap for a paper whose core contribution is establishing family-level properties.

- **Tension between the free-form generation argument and the ES evaluation design:** The paper's central argument is that MCQ evaluation is insufficient and free-form generation should be assessed. However, the ES metric uses few-shot prompting to constrain model outputs to MCQ-style format (e.g., "C. tiger"), as acknowledged in Appendix A: "The purpose is solely to ensure that the model outputs remain restricted to the given options (A–D), which makes the subsequent NLI evaluation reliable." This partially undermines the free-form evaluation claim—models are still being steered toward option-style outputs rather than generating truly open-ended responses. The Table A1 examples (showing nonsensical text like "@nate@nate...") suggest that unconstrained generation would reveal even more dramatic differences, making the constrained evaluation a conservative rather than faithful assessment of generative behavior.

- **No variance reporting or significance testing for comparative claims:** All figures report single-run results without confidence intervals, error bars, or statistical tests. Claims like "NPO's UTOpen-QA is much lower" than RMU's or that one family "generally outperforms" another are made without quantifying uncertainty. For large-scale LLM evaluation, single-run MCQ accuracy is often stable, but ES-based metrics on generative outputs could have meaningful variance. This is a gap for a paper whose contribution rests on comparative empirical claims.

- **Correlation claims about robustness dimensions lack quantitative support:** The paper states that "RobJA patterns align more closely with RobReL than with RobFT" (Section 5), which is a key insight connecting input-level and model-level attacks. However, no correlation coefficient, statistical test, or quantitative measure of alignment is provided. Given that this finding could inform defense design (if model-level and input-level robustness share mechanisms), it deserves rigorous quantification.

- **Attack budget inconsistency between RobReL and RobFT evaluations:** In-domain relearning uses 100 fine-tuning steps while out-of-domain fine-tuning uses 250 steps (Appendix A). When the paper concludes that divergence-driven methods are more robust to relearning but less robust to fine-tuning, this asymmetry could partly reflect the stronger attack budget for RobFT rather than inherent family properties. A matching attack budget or explicit discussion of this design choice is needed.

- **Missing formulation of the research question (Q):** The introduction states "the key research question we aim to address is:" but the question itself does not appear—only "To tackle (Q)..." follows. While this appears to be a LaTeX formatting error, it leaves the study's precise focus ambiguous. For a systematic analysis paper, an explicit research question anchoring the experimental design is important.

## Nice-to-Haves

- **Human evaluation on a subset of outputs** to validate that ES correlates with human judgment of unlearning quality and generation coherence. The paper acknowledges this limitation but it remains a meaningful gap for safety-critical evaluation.

- **Computational cost comparison** across methods (e.g., SAM and IRM add significant overhead). Practical method selection depends on both effectiveness and cost; without this, the "actionable guidance" is incomplete.

- **Mechanistic analysis of why families differ under attacks.** The paper shows that divergence-driven methods resist relearning while representation misalignment resists fine-tuning, but doesn't explain why (e.g., via weight change analysis or activation patterns). This would elevate the taxonomy from classification to understanding.

- **Truly unconstrained Open-QA evaluation** alongside the format-constrained ES, to assess whether the paper's findings hold or strengthen when models generate fully free-form responses.

- **Ablation isolating robustness design components** in the starred methods (NPO+SAM vs. NPO+IRM vs. NPO), to provide more granular guidance on which specific modifications drive robustness gains.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Figure and equation readability issues** — Attributed to PDF parsing artifacts, not the submitted paper.
- **Missing method details for TAR, LAT, RMU+LAT** — The paper summarizes these as extensions of RMU with meta-learning or adversarial training and references the original papers; for a taxonomy/comparison paper, this level of detail is appropriate.
- **Code availability, random seeds, exact model version** — Reproducibility nitpicks outside the scope of what's expected in a submission; the Appendix provides substantial implementation details.
- **Taxonomy boundaries are blurry (DPO under rejection-based also uses divergence)** — The paper's categorization is based on the *target* of unlearning (rejection behavior vs. untargeted divergence from reference), not solely on loss structure. DPO is classified as rejection-based because it treats rejection as the preferred response, which is a coherent and defensible distinction.
- **Demand for multiple additional benchmarks (MUSE, TOFU, RWKU)** — The paper uses WMDP and provides supplementary MUSE analysis. While more benchmarks would strengthen the paper, demanding comprehensive coverage of all available benchmarks is scope creep for what is already a substantial empirical study.
- **Broader impact section is "generic"** — Formatting/style critique; the section adequately covers both positive and negative societal implications.
- **Compute requirements not discussed** — Moved to nice-to-have; useful but not a core flaw for an evaluation/taxonomy paper.
- **Missing related works** — Per rules, cannot confirm existence of uncited works.

## Novel Insights

The paper reveals an important asymmetry in how unlearning methods fail: divergence-driven methods achieve unlearning by *destroying* generative capacity on forget queries (logit collapse), while representation misalignment achieves it by *reordering* internal signals (logit reshaping). This distinction explains why the former over-forgets in open generation while the latter preserves utility—a finding that MCQ evaluation completely obscures. The connection between jailbreaking robustness and in-domain relearning (both being "adversarial" in nature) versus out-of-domain fine-tuning (being an "OOD" perturbation) is a fruitful framing that draws an analogy to the adversarial vs. distributional robustness distinction in classification, suggesting that unlearning defenses might productively borrow from that literature.

## Suggestions

- Add at least one additional model (e.g., a 70B variant or different architecture) on WMDP-Bio to test whether family-level findings generalize, even if only for a subset of methods.
- Report correlation coefficients for the RobJA–RobReL/RobFT alignment claim, and explicitly discuss the attack budget asymmetry between relearning and fine-tuning evaluations.
- Include a small set of truly unconstrained open-generation examples (without few-shot format steering) alongside the ES-constrained evaluation, to demonstrate that the paper's findings are conservative lower bounds on the MCQ-vs-Open-QA gap.
- Explicitly state the research question (Q) that the study addresses—this appears to be a formatting omission but is important for framing.

---

**Axis assessments:**

- **Novelty:** Moderate. The taxonomy organizes known methods rather than proposing new ones; the key novelty is the Open-QA evaluation lens and the multi-faceted robustness comparison, which reveals previously obscured family-level properties.

- **Technical soundness:** Mixed. The experimental design is comprehensive in scope, but the single-model/single-dataset limitation, lack of variance reporting, attack budget inconsistency, and tension in the ES evaluation design reduce confidence in the generality of the claims.

- **Empirical support:** Adequate for the main UE-UT findings (the MCQ vs. Open-QA gap is clearly demonstrated), but insufficient for broad family-level generalizations and the robustness correlation claims, which lack quantitative support.

- **Significance:** High. The finding that MCQ evaluation systematically misrepresents unlearning effectiveness and utility retention has direct implications for how the community should benchmark future methods, and the robustness analysis provides practical guidance for method selection.

- **Clarity:** Good overall, with a clear organizational structure following the taxonomy-evaluation-robustness arc. The proliferation of metric acronyms (UEMCQ, UEOpen-QA, UTAvg, RobReL, RobFT, RobQT, RobJA) is manageable but a notation table would help. The missing research question is a noticeable gap.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 4.0]
Average score: 3.5
Binary outcome: Reject
