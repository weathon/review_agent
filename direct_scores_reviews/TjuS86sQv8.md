## Summary
This paper benchmarks leading large language models (LLMs) and vision-language models (VLMs) against human cognitive norms on the Wechsler Adult Intelligence Scale Fourth Edition (WAIS-IV), covering Verbal Comprehension (VCI), Working Memory (WMI), and Perceptual Reasoning (PRI). The study finds that most models score at or above the 98th–99.5th percentile on VCI and WMI while consistently performing near or below the 1st percentile on PRI, with notable variability driven by model scale. The authors also conduct WAIS-IV standard discrepancy analyses to characterize intra-index cognitive profiles across models.

---

## Strengths

- **Use of a clinically validated instrument with professional scoring.** Unlike ad-hoc LLM benchmarks, the paper employs the actual WAIS-IV—a rigorously normed, standardized psychometric tool—and has clinical psychologists score open-ended responses per official manual protocols. This is a concrete methodological improvement over prior work (Ilić & Gignac, 2024) that used non-validated subtest proxies.

- **Standard discrepancy analysis applied to AI.** The paper borrows the WAIS-IV's built-in discrepancy analysis framework (Tables 3–4) to probe *relative* strengths and weaknesses within models, not just absolute scores. The consistent finding that Information far outpaces Similarities/Vocabulary within VCI (significant for all models, including Gemini Nano at p < .05), and that Digit Span outpaces Arithmetic within WMI, is a nuanced characterization going beyond simple composite rankings.

- **Clear and striking PRI deficit finding.** The finding that all tested multimodal models perform at or below the 10th percentile on PRI—with most at <1st percentile—is a well-controlled, internally consistent result. Because PRI items consist of novel visual patterns rather than retrievable factual content, the deficit is not plausibly explained by memorization and is consistent with known VLM limitations. The across-developer consistency (GPT, Gemini, Claude all fail) strengthens this finding.

- **Quantified inter-generational improvement in visual reasoning.** The jump from Claude 3 Opus to Claude 3.5 Sonnet in Matrix Reasoning (0.1th → 25th percentile) and Figure Weights (0.1th → 50th percentile) provides a concrete, quantified improvement trajectory against a human-normed scale, which is more interpretable than relative benchmark deltas.

---

## Weaknesses

### Fatal
None that completely invalidate every finding. The PRI results remain credible. However, the two issues below severely compromise the VCI and WMI claims specifically.

### Major

- **Test contamination renders VCI and WMI scores uninterpretable as measures of cognitive ability.** The WAIS-IV is a commercially published instrument whose specific items are documented in academic literature and accessible online. LLMs trained on web-scale corpora almost certainly encountered WAIS-IV items, answer keys, and worked examples during pretraining. The most direct evidence in the paper itself is the Information subtest: *every single model* achieves 18–19/19 (99.6th–99.9th percentile), including Gemini Nano, which scores 1/19 on Similarities and 4/19 on Vocabulary. It is implausible on any account of genuine cognitive ability that Nano is maximally knowledgeable but severely deficient in verbal reasoning; the parsimonious explanation is item-level memorization. The paper's limitations section acknowledges only "non-standard administration" in one brief sentence and does not name contamination as a validity threat. Because VCI and WMI constitute the primary positive claims of the paper, this is a fundamental credibility problem, not a peripheral limitation.

- **Modality mismatch fundamentally changes what Digit Span measures.** In humans, Digit Span is an auditory short-term memory task: digits are read aloud at one per second, and subjects must retain them in phonological working memory. In the paper's administration, the full digit string is presented as a written text prompt stored externally in the model's input context. There is no phonological loop, no decay, and no capacity limit of the kind that creates the human normative gradient. The near-perfect scores achieved by nearly every model on all three Digit Span subtests (Forward 18/19, Backward 18–19/19, Sequencing 17–19/19 for most models) reflect reading back a text string—trivially easy for any transformer—not working memory. The resulting WMI composites (≥99.5th percentile for all but Gemini Nano) cannot validly be compared to human norms for this construct. The paper's own abstract calls these scores evidence of "exceptional capabilities in the storage, retrieval, and manipulation of tokens"—which is true but tautological—while simultaneously comparing the scores to human normative percentiles as though they measure the same thing.

- **No inter-rater reliability statistics reported.** The paper states that responses were scored by one of two clinical psychologists, with ambiguous cases reviewed jointly. However, no inter-rater reliability metric (Cohen's kappa, intraclass correlation coefficient, or percent agreement) is reported. In any study using subjective human scoring as its primary measurement procedure, this is a basic methodological requirement. Without it, the consistency of the scoring process cannot be evaluated.

### Minor

- **p < .15 used as a significance threshold without justification.** Tables 3 and 4 use `* p < .15` as one significance indicator. Using p < .15 as a threshold for significance is non-standard (even p < .10 is rarely accepted without explicit justification as "marginal"). The paper offers no rationale. This inflates apparent significance for several discrepancy claims and should be either justified or revised to p < .10 with appropriate framing as exploratory.

- **Positive Manifold claim is asserted without formal analysis.** The paper states that "the Positive Manifold… fails to hold for when including PRI." This is based on six models, and no correlation coefficients or formal tests are reported. The claim may well be correct, but the evidence presented is purely informal.

- **Model stochasticity is unaddressed.** The paper does not report temperature settings or whether models were tested once or multiple times per item. Single-run, potentially non-deterministic responses introduce unquantified measurement variance that could affect scaled scores, particularly on open-ended verbal items.

- **No testing timestamps.** The paper does not report when each model was accessed. Models may have been updated between testing sessions. For longitudinal comparisons (e.g., GPT-3.5 → GPT-4), undocumented version drift is a confound.

### Tiny

- **Prompting protocol not fully specified.** The paper states prompts followed WAIS-IV manual instructions (zero-shot, no CoT). This should be explicitly confirmed in the methodology rather than inferred, for reproducibility.

- **Chance-level performance for PRI not reported.** PRI subtests are multiple-choice (5 options for Matrix Reasoning). A random baseline would help contextualize the near-floor PRI scores.

---

## Nice-to-Haves

- A contamination probe using novel, never-published items (e.g., newly constructed analogues of WAIS-IV tasks) would be highly valuable to distinguish retrieval from reasoning.
- A PRI error typology—categorizing failures as image-parsing errors vs. logical reasoning failures—would deepen the visual deficit finding considerably.
- AI-specific normative baselines (pooling performance across many models) could complement human percentile comparisons for future work.
- Visualizations of specific PRI failure cases (matrix inputs + model output) would make the visual deficit more concrete and compelling.
- Reporting confidence intervals or stability across multiple runs for key scores would increase measurement credibility.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Weakness (Review 2): Assumption of normal distribution.** This misunderstands the method. The paper does not assume AI scores follow a normal distribution; it places each model's point score against the pre-existing human normative distribution (a standard psychometric practice). No distributional assumption about AI is required. Removed.

- **Weakness (Reviews 2 & 3): Limited mechanistic insight / no architectural explanation.** The paper is explicitly a behavioral benchmarking study. Demanding architectural or mechanistic analysis is scope creep for this contribution type. Weakened to nice-to-have level.

- **Weakness (Review 1): "Circular logic" in framing.** The critic argues that using a human-normed test assumes the test measures the same constructs across biological and artificial systems. The paper explicitly acknowledges this by noting the administration is non-standard and by framing the exercise as behavioral benchmarking with known limitations. The concern is reasonable philosophically but the paper is not claiming construct equivalence in a strong psychometric sense—it is comparing behavioral outputs. Removed as overstated.

- **Strength (Review 2): "Covers a diverse range of proprietary models."** This is generic; nearly any modern LLM benchmarking paper does this. Removed.

- **Strength (Review 2): "Use of clinical psychologists adds validity often missing in automated evaluations."** Retained in main strengths in a more specific form; the generic version ("adds validity") is removed.

---

## Novel Insights

The most genuinely novel observation across the reviews is the interplay between the Information subtest anomaly and the contamination concern: Gemini Nano's extreme performance dissociation—near-maximum on Information (19/19) while near-floor on Similarities (1/19) and low on Vocabulary (4/19)—constitutes internal evidence within the paper's own data of item-level memorization rather than general verbal ability. This dissociation pattern, consistent across models to varying degrees (every single model achieves 18–19 on Information regardless of overall scale), is a natural contamination detection signal that the authors could analyze more formally. Additionally, the finding that the within-VCI discrepancy between Information and Similarities is *larger in smaller models* (Gemini Nano: +11 SD points; GPT-4o: +2.33 SD points) may indicate that larger models acquire genuine verbal reasoning that narrows the gap, while smaller models are more purely retrieval-dependent—a hypothesis that, if substantiated with contamination-controlled items, would be a meaningful contribution to understanding scaling effects.

---

## Suggestions

1. **Contamination probe (highest priority):** Construct novel, unpublished analogues of at least the Information and Similarities subtests (e.g., items about events post-training-cutoff) and re-run the analysis. If Information scores drop substantially while Similarities scores are stable, this confirms contamination. If both drop uniformly, it supports genuine ability measurement. This single experiment would substantially increase or undermine confidence in all VCI/WMI claims.

2. **Address Digit Span modality mismatch explicitly:** Add a section discussing why text-presented Digit Span cannot be equated to auditory Digit Span under human norms, and qualify WMI composite claims accordingly. The paper's framing—"exceptional capabilities in the storage, retrieval, and manipulation of tokens"—is accurate; the comparison to human WMI percentiles is not.

3. **Report inter-rater reliability:** Provide Cohen's kappa or ICC for the clinical psychologist scoring, even retrospectively on a sample of responses. This is essential for scientific credibility.

4. **Replace p < .15 with p < .10 (or less):** Or explicitly label all such findings as exploratory with Bonferroni-adjusted thresholds given the number of comparisons made across Tables 3 and 4.

5. **Report testing dates and model version strings:** Capture exact API versions and access dates to allow future replication and to clarify generational comparisons.

6. **Formal Positive Manifold analysis:** Report pairwise Pearson or Spearman correlations across VCI, WMI, and PRI scores across the ten models, with appropriate caveats about the small sample size. Even a simple correlation matrix would support or undermine the Positive Manifold claim.

---

## Evaluation on Key Axes

- **Novelty:** Moderate. Using the actual WAIS-IV with clinical scoring is a methodological step beyond prior proxy-based work, and the PRI deficit finding is presented with appropriate scope. However, "LLMs are good at text, bad at vision" is broadly known; the contribution is the specific quantification and profile analysis.
- **Technical soundness:** Below average. The contamination and modality mismatch issues are serious and underexamined. The statistical practices (p < .15, no inter-rater reliability) further undermine rigor.
- **Empirical support:** Mixed. The PRI findings are well-supported and internally consistent. The VCI/WMI claims are confounded by contamination and modality issues to a degree that the paper does not adequately acknowledge.
- **Significance:** Moderate. The behavioral profiling approach and the striking VCI–PRI dissociation are of broad interest to the AI evaluation community, but the methodological problems limit the strength of the conclusions.
- **Clarity:** Generally good. Tables are well-organized, results sections are structured, and the discrepancy analysis is presented accessibly.

MY FINAL SCORE: <pineapple>4.6</pineapple>