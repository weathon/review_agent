## Summary

This paper benchmarks 10 leading LLMs and VLMs (from OpenAI, Google, and Anthropic) using three indices of the Wechsler Adult Intelligence Scale, Fourth Edition (WAIS-IV): Verbal Comprehension (VCI), Working Memory (WMI), and Perceptual Reasoning (PRI). The central finding is a striking dissociation: models perform at or above the 98th–99.9th human percentile on VCI and WMI, yet consistently fall in the "Extremely Low" range (≤10th percentile) on PRI across all tested VLMs. The authors administer subtests via adapted prompts scored by clinical psychologists and perform WAIS-IV-standard discrepancy analyses to identify within-domain relative strengths and weaknesses.

---

## Strengths

- **Clinical scoring with inter-rater validation:** Unlike most LLM benchmarking work, responses were scored by two trained clinical psychologists using WAIS-IV standardized scoring criteria, with consensus procedures for ambiguous cases. This adds meaningful rigor over automated or crowdsourced scoring, particularly for open-ended subtests like Similarities, Vocabulary, and Comprehension where partial credit and nuance matter.

- **WAIS-IV discrepancy analyses reveal non-trivial within-domain patterns:** The paper goes beyond composite index scores to conduct formal WAIS-IV discrepancy analyses. These reveal consistent cross-developer patterns: Information > Similarities/Vocabulary across all VCI-capable models (suggesting stronger crystallized recall than abstract verbal reasoning), and Digit Span > Arithmetic within WMI (suggesting stronger token manipulation than numerical word-problem reasoning). The Gemini Nano case — proficient in encoding but failing at manipulation — is a particularly crisp dissociation.

- **The PRI failure is a cross-developer, cross-generation finding:** The low PRI finding is not an artifact of any single developer's system. All six multimodal models fall below the 10th human percentile, with five below the 1st. This convergent finding across architecturally distinct systems constitutes meaningful empirical evidence of a common limitation in current VLMs, not noise.

- **Figure Weights exception reveals a theoretically interesting sub-dissociation:** Claude 3.5 Sonnet scores at the 50th human percentile on Figure Weights (a visually-presented algebraic balancing task) while remaining at the 0.1–0.4th percentile on Picture Completion and Visual Puzzles. This pattern — algebraic visual reasoning substantially better than holistic spatial/scene understanding — is a meaningful within-PRI dissociation that is underexplored in the paper but has genuine theoretical implications for understanding VLM visual processing.

- **Population-normed framework yields interpretable, absolute benchmarks:** Expressing LLM performance in WAIS-IV percentiles against a 2,200-person representative normative sample provides a more interpretable reference point than arbitrary leaderboard scores, and enables the discrepancy analyses that yield the paper's most nuanced findings.

---

## Weaknesses

### Fatal
None. The paper has real methodological problems, but the core PRI deficit finding is cross-validated across developers and is unlikely to be explained away by any single confound.

### Major

- **Training data contamination is entirely unaddressed — a central validity threat for VCI.** The Information subtest asks factual questions (e.g., "Who was the first president of the United States?"). All 10 models score 19/19 (99.9th percentile ceiling). The WAIS-IV is a commercially published instrument whose items are widely discussed in clinical training materials, textbooks, study guides, and online forums — sources almost certainly present in the training corpora of all tested models. The paper attributes these ceiling scores to "exceptional crystallized knowledge" and "long-term memory recall," but cannot distinguish genuine world-knowledge generalization from item-level memorization. Critically, this contamination concern extends beyond Information: Vocabulary, Comprehension, and Similarities items (e.g., "How are a kite and an airplane alike?") are the kinds of generic verbal items that may also appear in training data. The absence of any contamination discussion, let alone any control (e.g., novel parallel items, training-data search), is the most significant methodological gap in the paper.

- **Construct validity of WMI scores is asserted, not established.** The paper explicitly acknowledges that test adaptations "provided the GenAI models with an advantage due to their ability to access the full context while generating responses" (§2.1), but treats this as a minor caveat rather than the central construct validity issue it is. The WAIS-IV WMI subtests (Digit Span, Arithmetic, Letter-Number Sequencing) were designed to probe a cognitively limited, time-decaying biological working memory system. LLMs with persistent context window access are not performing the same cognitive operation. Achieving 99.9th percentile on Digit Span does not demonstrate "exceptional working memory" in any psychologically meaningful sense — it demonstrates that the model can access a digit string it has already seen, which is a trivially expected property of any context-attending system. The paper's abstract claim of "exceptional capabilities in the storage, retrieval, and manipulation of tokens" is the right framing, but the paper then maps this directly onto WMI percentile rankings as if the construct equivalence is established. The paper should explicitly distinguish "context-window token manipulation" from "working memory" as a psychological construct, and interpret the WMI results accordingly.

- **PRI failure modes are not disaggregated — visual encoding integrity is unverified.** The paper attributes the low PRI scores to an inability to "understand the meaning or relationship in visual representations." However, a competing explanation — that models fail because the image-to-model encoding pipeline (image resolution, format, prompt encoding of visual stimuli) is degraded or lossy — is not addressed. If a VLM cannot correctly identify the elements present in a Matrix Reasoning image due to input encoding failures, the model is penalized for perception, not reasoning. Given that Figure Weights (where the visual structure is more algebraically explicit) shows dramatically better performance than Picture Completion and Visual Puzzles (which require holistic scene understanding), there is already internal evidence for heterogeneous failure modes. Without qualitative error analysis or encoding integrity checks, the claim of a "profound inability to reason on visual information" may conflate perception failures with reasoning failures.

- **Single administration per model; temperature and inference parameters unreported.** No model is tested more than once. For models with non-zero temperature, different runs would yield different scores. The paper does not report what temperature settings were used, whether inference was deterministic, or any other API parameters. A single-point estimate with no variance measure is insufficient to support strong claims about a model's capability profile, particularly for subtests where a single item error changes the scaled score substantially.

### Minor

- **The p < .15 significance threshold is non-standard and poorly justified.** The paper uses `* p < .15` and `** p < .05` notation throughout Tables 3 and 4. Using p < .15 as a threshold for reporting statistical significance diverges substantially from standard convention (typically p < .05 or p < .01). These should either be reframed as exploratory trends, or the threshold should be justified.

- **Age norm choice (25–29 years) is not justified.** The authors selected the 25–29 age bracket without explanation. While this choice is defensible (young adult peak is a meaningful reference point), it should be explicitly justified; different age brackets would yield different percentile rankings, and readers cannot assess the sensitivity of the conclusions to this choice.

- **Positive Manifold conclusion is overextended.** The paper concludes (§4) that "the Positive Manifold holds when VCI and WMI are considered, and fails to hold when including PRI." With only three index scores, two of which hit ceiling for most models, and no replication across administrations, this is a descriptive observation rather than a test of the Positive Manifold hypothesis. The conclusion should be framed accordingly.

- **Inter-rater reliability is procedurally described but not quantified.** The paper states that two psychologists scored and resolved disagreements by consensus, but reports no inter-rater reliability metric (kappa, percent agreement). For subtests with nuanced verbal responses (Similarities, Comprehension, Vocabulary), this is an important validity indicator.

- **Anomalous model responses are unresolved.** The paper notes that retaining phrases like "Listen" caused models to produce responses such as "I am a text-based chat assistant and cannot hear the numbers." It is unclear how these off-task responses were scored (zero? discarded?) and whether they affected any model's total score.

### Tiny

- **Table 4 column ordering appears to differ from Table 2.** Table 4 lists OpenAI models as "GPT-3.5 Turbo | GPT-4o Turbo | GPT-4 Turbo," while Table 2 lists them as "GPT-3.5 Turbo | GPT-4 Turbo | GPT-4o." This ordering inconsistency could lead to misreading of the data.

- **"First of its kind" claim is overstated.** The paper describes itself as "the first of its kind approach to benchmark GenAI against human norms of intelligence," while simultaneously citing Ilić & Gignac (2024), who used WAIS-IV subtest proxies for LLMs. The paper's contribution is a more rigorous and complete WAIS-IV administration — a meaningful advance, but not the first such effort.

---

## Nice-to-Haves

- **Include open-weight models (e.g., Llama, Mistral, Qwen).** Open models allow partial disentanglement of training data, parameter count, and architecture from performance — exactly the factors the authors note are unavailable for proprietary models. Even a single open model would strengthen the analysis considerably.

- **Ablate prompting strategies (chain-of-thought, few-shot, zero-shot).** Current results conflate inherent capability with prompt sensitivity. A prompting ablation on a representative subtest would clarify how much scores depend on elicitation rather than capability.

- **Run text-only models on text-described PRI items.** To isolate whether PRI failure is visual encoding or abstract reasoning, administer text-transcribed versions of PRI items (as matrix descriptions, verbal puzzle descriptions) to text-only models and compare. This would directly address the perception-vs.-reasoning confound.

- **Provide qualitative error analysis for PRI.** Showing specific model outputs for Matrix Reasoning or Visual Puzzles failures — whether models misidentify visual elements, apply wrong logical rules, or simply guess — would make the PRI deficit finding much more actionable for vision researchers.

- **Release exact prompt templates and scoring scripts.** Currently in Appendix B (unavailable in the reviewed version). A public repository with prompts, image preprocessing steps, and scoring rubrics is necessary for reproducibility.

- **Compare model performance to a human control group administered the text-adapted version.** The adapted text prompts differ from standard oral WAIS-IV administration. Without a human group receiving the same text-adapted version, it is impossible to know whether the adapted format itself changes the normative reference point.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Harsh Critic: "Exclusion of open models is unjustified and is a major weakness."** While open models would be informative, the paper's stated contribution is benchmarking SOTA proprietary models, and 10 models across three major providers is a reasonable scope for a descriptive study. Moved to Nice-to-Have.

- **Harsh Critic: "The 25–29 age norm flatters models by using peak performance norms."** The direction of bias is not as straightforward as claimed. Using peak human norms for WMI/VCI would actually make it *harder* (not easier) to achieve high percentiles, since human performance is highest at that age. The age norm concern is retained as a minor issue (unjustified choice), but the specific framing that it inflates scores is factually incorrect for VCI/WMI.

- **Harsh Critic severity on construct validity of WMI as "devastating" / "the central threat to the paper's interpretive claims."** The paper explicitly acknowledges the context-window advantage and frames results around "token manipulation" rather than claiming to measure identical constructs. The validity concern is real and retained as a Major weakness, but framing it as fatal to the entire paper is excessive given the authors' own caveats.

- **Positive Reviewer: "Methodological adaptation issues (omitting 'read the problem one more time') alter standardization."** The paper acknowledges this as a limitation, and the adaptation is reasonable given the test constraints. The off-task response issue is retained as a Tiny weakness, but the general adaptation concern (omitting repetition prompts) is not a meaningful validity threat.

---

## Novel Insights

The most genuinely novel insight emerging from the synthesis of these reviews — beyond the paper's own stated contributions — is the **within-PRI sub-dissociation**: Claude 3.5 Sonnet scores at the 50th human percentile on Figure Weights (visual-algebraic reasoning with explicit symbolic structure) while remaining near floor on Picture Completion and Visual Puzzles (holistic scene understanding and spatial reasoning). This pattern is consistent across the reviewed data and suggests that current VLMs may not have a uniform "visual processing deficit" but rather a specific deficit in scene-level and spatial visual understanding, while being better equipped for tasks where the visual representation can be decomposed into explicit symbolic or algebraic relationships. This is a theoretically important distinction — it implies that the gap between human and VLM visual reasoning may be partially addressable by reframing visual tasks in more structured, symbolic encodings, rather than requiring advances in raw visual perception. The paper notices this pattern but does not theorize it; a deeper analysis could yield a meaningful contribution.

---

## Suggestions

1. **Add a contamination control.** Generate a small set of novel, non-public WAIS-analogous items for each VCI subtest and compare performance to the published items. A significant drop in Information or Comprehension scores on novel items would confirm contamination; a stable score would strengthen the crystallized knowledge interpretation.

2. **Report inference parameters.** Specify temperature, top-p, max tokens, and API access dates for all models. For studies aiming at reproducibility benchmarks, these are not optional.

3. **Disaggregate PRI failure modes qualitatively.** For each of the four PRI subtests (Matrix Reasoning, Visual Puzzles, Figure Weights, Picture Completion), show 2–3 representative model responses alongside the correct answer, and categorize failures as (a) visual primitive misidentification, (b) correct perception but wrong reasoning, or (c) complete non-answer. This would transform the PRI finding from a score report into a mechanistic claim.

4. **Reframe WMI conclusions more carefully.** Distinguish explicitly between "context-window token access and manipulation" (what is actually being measured) and "working memory" as a psychological construct (what the test was designed to measure in humans). The finding that models excel at token sequencing while showing relative weakness in arithmetic word problems is genuinely interesting — but it should not be presented as evidence of superior human-equivalent working memory.

5. **Run multiple administrations on a subset of models.** Even 3–5 repeated administrations at non-zero temperature for one model per developer would provide variance estimates and demonstrate whether the observed discrepancy patterns (e.g., WMI >> PRI) are stable.

6. **Justify the p < .15 threshold or reframe as trends.** Either provide statistical justification (e.g., exploratory analysis with Bonferroni correction waived) or remove the asterisk notation for p < .15 findings and describe them as "trends" or "descriptive patterns."

---

**Evaluation summary:**
- **Novelty:** Moderate — full WAIS-IV with clinical scoring advances over prior proxy-based work, but the research direction is established.
- **Technical soundness:** Weak-to-moderate — the methodology has real gaps (no contamination control, single administration, construct validity of WMI acknowledged but not resolved, PRI failure mode ambiguity).
- **Empirical support:** Moderate — 10 models across three developers produces convergent findings, but single-run data with unreported inference parameters limits confidence in any individual result.
- **Significance:** Moderate — the PRI dissociation is a practically meaningful, cross-validated finding; the VCI/WMI results are largely consistent with expectations but provide useful population-normed quantification.
- **Clarity:** Acceptable — the paper is readable and the tables are informative, but the p < .15 notation, the construct-validity conflation in the WMI discussion, and the missing appendices detract from clarity.