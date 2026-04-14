## Summary
This paper administers adapted versions of the Wechsler Adult Intelligence Scale (WAIS-IV) to ten leading LLMs and VLMs, scored by trained clinical psychologists, to benchmark model performance against human normative distributions. The study finds near-ceiling performance on Verbal Comprehension (≥98th percentile) and Working Memory (≥99.5th percentile) for most models, contrasted with uniformly severe deficits on Perceptual Reasoning (<10th percentile for all multimodal models). The core empirical contribution is documenting a stark and consistent modality gap across independent model families, with some evidence that the PRI deficit can be partially overcome with architectural advances (Claude 3.5 Sonnet vs. Claude 3 Opus on Figure Weights).

---

## Strengths

- **Clinical-grade scoring protocol**: Unlike automated AI benchmarks, WAIS-IV items are scored by trained clinical psychologists with consensus adjudication for ambiguous cases. This is methodologically stronger than automatic scoring and adds legitimacy to verbal/reasoning scoring that requires qualitative judgment.

- **Cross-developer consistency of the PRI deficit**: The near-floor PRI performance holds across OpenAI, Google, and Anthropic models—different training regimes, data pipelines, and multimodal architectures—making the finding robust against model-specific confounds. A 60–100 point standard-score gap between WMI and PRI is not a marginal effect; it is enormous by psychometric standards.

- **Granular subtest-level analysis reveals meaningful within-domain patterns**: The paper goes beyond composite indices and identifies informative dissociations: Digit Span vs. Arithmetic within WMI (token manipulation vs. mathematical reasoning), Information vs. Similarities/Vocabulary within VCI (retrieval vs. abstract reasoning), and Figure Weights vs. Picture Completion within PRI (quantitative reasoning vs. visual perception). These patterns add scientific texture beyond "LLMs are good at language."

- **Demonstration that PRI performance is improvable**: The Claude 3 Opus → Claude 3.5 Sonnet jump on Figure Weights (0.1th → 50th percentile) and Matrix Reasoning (0.1th → 25th percentile) provides concrete evidence that the visual reasoning gap is not architecturally fixed, which is a constructive finding for the field.

---

## Weaknesses

### Fatal
None. The central empirical finding—a large, cross-consistent modality gap between text-based and visual tasks—is robust to most methodological concerns. However, the paper over-interprets these findings as direct windows into cognitive constructs rather than performance on adapted psychometric tasks.

### Major

- **Construct validity of WMI tasks is critically underexamined.** The paper explicitly acknowledges (Section 2.1) that some adaptations "provided the GenAI models with an advantage due to their ability to access the full context while generating responses." For Digit Span, this is not a minor advantage—presenting a digit sequence as a visible text prompt converts a transient auditory working memory task into visible-string transformation, a qualitatively different cognitive operation. Crucially, Table 5 shows that *every single model, including Gemini Nano*, achieves an identical LDSF of 9 (top 17.5% in humans)—even while Nano fails catastrophically at Digit Span Backwards and Letter-Number Sequencing. The uniformity of LDSF across models of wildly different capability levels is a clear signal that the task adaptation may be measuring attention to visible tokens rather than genuine transient storage capacity. The paper does not analyze this, instead presenting near-ceiling WMI scores as evidence of "exceptional capabilities in storage, retrieval, and manipulation." The claim requires at least an acknowledgment that the construct may not transfer.

- **Training-data contamination is unaddressed, and the Information subtest is an obvious confound.** Every model tested, including Gemini Nano (which scores in the 23rd percentile overall on VCI), achieves a scaled score of 19 (99.9th percentile) on Information. This ceiling uniformity across models of vastly different abilities is more consistent with memorization of factual knowledge from training corpora than with a genuine psychometric latent ability. The paper interprets this as evidence of "exceptional crystallized knowledge" without considering that general-knowledge questions of the WAIS-IV type are thoroughly represented in web-scale text. Without any contamination control (e.g., novel isomorphic items), this subtest provides no meaningful evidence of cognitive ability, yet it inflates VCI composites for every model. The paper should either treat this as a confound or provide evidence it is not.

- **Reproducibility is insufficient.** The paper does not report: (a) API version identifiers or access dates, (b) decoding temperature or sampling parameters, (c) whether system prompts were standardized, (d) whether conversations were reset between items, (e) whether safety filters or tool use were active, (f) number of runs per item, or (g) image format/resolution for multimodal inputs. For proprietary models with periodic backend updates, these details are essential for reproducibility. A single deterministic evaluation per item is insufficient to establish reliable percentile claims, particularly for open-ended verbal subtests where response variation exists.

- **Inter-rater reliability for psychologist scoring is not reported.** The paper states answers were scored by one of two clinical psychologists with consensus adjudication for ambiguous cases, but it does not report: the proportion of responses requiring adjudication, Cohen's kappa or ICC across independently double-scored items, or whether scorers were blinded to model identity. Several WAIS-IV verbal subtests (Similarities, Vocabulary, Comprehension) require graded qualitative judgment. Without reliability statistics, the scaled scores for open-ended subtests rest on an unverified assumption of consistent scoring.

### Minor

- **The PRI deficit may partially reflect image encoding and API interface limitations, not only visual reasoning ability.** Picture Completion uniformly scores 1 (0.1th percentile) across all models, while Figure Weights for Claude 3.5 Sonnet reaches 10 (50th percentile). This large within-PRI heterogeneity suggests task-specific factors—potentially including image resolution through API interfaces, how WAIS figures were reproduced and embedded, and instruction format—contribute to the score profile. The paper does not investigate whether failures are due to perceptual encoding, instruction-following, or reasoning errors. Providing qualitative error analysis for even a few representative PRI items would substantially strengthen the claim of "profound inability to interpret and reason on visual information."

- **The statistical notation for discrepancy tables may confuse ICLR readers.** The use of `* p < .15` and `** p < .05` follows WAIS-IV clinical convention—these denote rarity of observed discrepancy patterns in the human normative sample, not inferential tests over repeated model evaluations. For an ML audience, this notation is easily misread as claiming statistical significance of model-level effects. A brief clarifying note would prevent misinterpretation.

- **The justification for using 25–29 age-group norms is absent.** While this is arguably a reasonable default (peak cognitive performance age range), the paper does not explain the choice or assess whether conclusions change under other normatively comparable age bands (e.g., 18–19, 45–54). For an evaluation that converts raw to normed scores as its central measurement procedure, norm selection should be explained.

- **The Positive Manifold observation is asserted rather than analyzed.** The paper states the Positive Manifold "holds" for VCI/WMI and "fails to hold" when PRI is included. Given that only 5–6 models have all three indices and PRI is dominated by floor effects, this observation requires more analytical care—or should simply be presented as a descriptive note rather than a finding.

### Tiny

- The discussion's speculation that the visual deficit "may require separate specialized architecture for visual and auditory processing with enhanced interaction capabilities, as is the case with vertebrates" is too speculative relative to the data. It should be framed more clearly as a hypothesis.

---

## Nice-to-Haves

- **Include at least some open-weight models** (e.g., LLaMA 3, Mistral, Qwen) to enable community replication and comparison without API dependency.
- **Correlate WAIS subtest ranks with standard AI benchmark ranks** (MMLU, HellaSwag, MMMU, etc.) to establish convergent validity and calibrate what WAIS performance predicts.
- **Test paraphrased or novel isomorphic WAIS items** alongside originals to provide a partial contamination control, especially for Information and Vocabulary.
- **Provide a text-only version of PRI items** (describing the matrix/puzzle in text rather than images) to disentangle visual encoding limitations from abstract reasoning deficits—this would address a scientifically important question about where the PRI failure originates.
- **Report multiple evaluation runs** with variance estimates for at least a subset of models and subtests, to characterize response variability.
- **Deepen the technical discussion** of PRI deficits: is the bottleneck visual tokenization, attention over spatial structure, or cross-modal grounding? This would make the findings more actionable for the ICLR audience.

---

## Removed Points

*These points are flagged as removed; treat them with caution as they are either factually inaccurate about the paper, scope-creep criticisms, or otherwise not well-grounded.*

- **[REMOVED] Criticism that the paper draws causal conclusions about parameter count, training data, etc.**: Section 2.2 explicitly states these conclusions are "limited due to the lack of public disclosure" and describes them as qualitative comparisons. The paper is appropriately hedged.
- **[REMOVED] Criticism of the paper for not including all WAIS-IV subtests (PSI, Block Design)**: The paper explicitly justifies these exclusions on methodological grounds (administration fidelity), which is reasonable and well-motivated.
- **[REMOVED] Criticism that base-rate comparisons "statistically treat models as human subjects"**: The discrepancy tables use human normative base rates as a reference frame for interpretation, not as inferential tests over model samples. This is a legitimate (if imperfect) psychometric framing, and the paper's note explains it as such.
- **[REMOVED] Claim that the WAIS-IV norms for age 25-29 are "arbitrary"**: Any norm band chosen would be arbitrary to some degree; 25-29 is a defensible choice for peak performance; criticizing it as arbitrary without evidence it changes conclusions is unproductive.
- **[REMOVED] Request for confidence intervals/repeated runs as a hard requirement**: For closed-ended structured items (Digit Span, Arithmetic, etc.), deterministic or near-deterministic behavior reduces this concern substantially. It is a legitimate methodological nice-to-have, not a fatal flaw, and is moved to Nice-to-Haves above.
- **[REMOVED] Criticism that the title overstates scope**: The title "The Cognitive Capabilities of Generative AI" is broad but common for papers establishing evaluative frameworks; it is not materially misleading given the abstract.
- **[REMOVED] Criticism of not establishing formal Positive Manifold statistics with sample-size analysis**: The paper describes this as an observation, not a formal test; demanding correlational modeling from N=10 proprietary models is unreasonable scope creep.
- **[REMOVED — original strength] "The paper is well-written and well-structured"**: Too generic.

---

## Novel Insights

The most genuinely novel observation in the paper—underappreciated even by the paper itself—is the dissociation *within* WMI: Digit Span (visible text) reaches near-uniform ceiling while Arithmetic shows consistent relative weakness across all model families and generations. This suggests that what the WMI is measuring in these adapted tasks is not a unified working memory construct but two distinct operations: in-context symbolic string transformation (which LLMs do trivially) and multi-step numerical reasoning under constraint (which is genuinely harder). This dissociation, if properly controlled for the construct validity problem, would be scientifically informative and worth investigating more deeply—it maps cleanly onto the distinction in ML between attention-based sequence manipulation and compositional arithmetic reasoning. The companion observation that Information scores are uniformly at ceiling regardless of model size (including Gemini Nano), while Similarities and Vocabulary show meaningful cross-model variance, points toward a clean empirical distinction between retrieval of encoded facts and generative reasoning over language concepts—a dichotomy that the paper touches on but could develop further as a standalone finding.

---

## Suggestions

1. **Add a brief construct-validity analysis for Digit Span**: Report whether any model ever fails LDSF (given perfect uniformity across all models at score 9). If all models trivially max this subtask, frame it explicitly as a task-adaptation artifact and remove it from strong capability claims. A simple control—embedding the digit sequence within a longer distractor context to require selective attention—would tell readers whether genuine span capacity or trivial visible-token matching is being measured.

2. **Treat Information as a potential confound rather than a finding**: Perform a sensitivity analysis of VCI composites excluding the Information subtest. If Gemini Nano's VCI drops substantially and other models' relative rankings shift, it confirms that Information is driven by training-data coverage rather than verbal reasoning ability.

3. **Report inter-rater reliability explicitly**: Calculate Cohen's kappa or ICC on a double-scored subsample and report it in Section 2.1. This is a basic psychometric reporting requirement and its absence undermines confidence in subtest scaled scores for qualitative tasks.

4. **Provide qualitative error analysis for PRI**: Present example model responses (including the stimulus image description, model output, and psychologist scoring rationale) for representative failed PRI items across at least two subtests. This would substantially clarify whether failures are primarily perceptual, instructional, or reasoning-based.

5. **Be explicit throughout that percentile comparisons to human norms are heuristic under adapted administration**: Add a standing caveat (once in methods, once in results, once in discussion) that these scores are indicative rather than psychometrically equivalent to human WAIS-IV administration, and that conclusions about underlying cognitive constructs should be read accordingly.

---

**Evaluation Summary:**
- *Novelty*: Moderate. Using clinical psychologist scoring and full WAIS-IV index structure is a genuine methodological advance over prior AI psychometric work, but the conceptual territory is established.
- *Technical soundness*: Below expectations for ICLR. The WMI construct validity problem and contamination gap are substantive, not peripheral.
- *Empirical support*: Mixed. The PRI findings are robust and cross-validated across developers. The VCI/WMI findings are empirically solid as task-performance descriptions but over-interpreted as cognitive-construct evidence.
- *Significance*: Moderate to high for AI evaluation and safety communities; the modality gap finding is meaningful and the framework is useful if properly scoped.
- *Clarity*: Good overall structure; clinical psychometric terminology should be better bridged for an ML audience, and the distinction between benchmark performance and cognitive construct measurement needs consistent clarification throughout.