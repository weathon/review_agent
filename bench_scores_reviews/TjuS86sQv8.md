## Summary
This paper benchmarks ten leading LLMs and VLMs against human performance using adapted subtests from the Wechsler Adult Intelligence Scale–Fourth Edition (WAIS-IV), focusing on Verbal Comprehension (VCI), Working Memory (WMI), and Perceptual Reasoning (PRI). The primary finding is a striking dissociation: models score at or above the 99th human percentile on text-based VCI and WMI tasks, while multimodal models score at or below the 10th percentile on visual PRI tasks. Clinical psychologists score all responses, and human population norms are used to convert raw scores to percentiles and perform discrepancy analyses.

---

## Strengths

- **Population-normed comparison framework.** Unlike ad-hoc AI benchmarks, the WAIS-IV provides validated, age-stratified normative distributions. Mapping model outputs into this framework produces a quantitatively interpretable profile of relative strengths and weaknesses that most LLM evaluations lack—and this specific choice enables discrepancy analyses (e.g., WMI vs. VCI) that would be impossible with typical benchmarks.

- **Expert scoring protocol.** Recruiting two trained clinical psychologists to score responses according to WAIS-IV manual criteria is a meaningful step above automated string-matching, particularly for open-ended subtests like Similarities, Vocabulary, and Comprehension where partial-credit scoring involves genuine judgment.

- **Striking and consistent VCI–PRI dissociation.** The magnitude of the gap—99th+ percentile on text-based tasks vs. <1st percentile on visual tasks, replicated across all six multimodal models and three developers—is a robust empirical signal. Claude 3.5 Sonnet's improvement over Claude 3 Opus specifically on Matrix Reasoning and Figure Weights (but not Picture Completion) further narrows the locus of the deficit toward abstract pattern reasoning rather than sheer image parsing, which adds structure to the finding.

- **Within-index dissociations illuminate sub-capability structure.** The Gemini Nano profile (intact Digit Span Forward, collapsed Digit Span Backwards and Letter-Number Sequencing) is a specific, coherent data point for understanding how constrained-capacity models degrade: encoding survives, manipulation collapses. Similarly, the near-universal relative weakness of Arithmetic versus Digit Span across larger models is an informative and underappreciated finding.

---

## Weaknesses

### Fatal
*None that completely invalidate all findings.* The PRI dissociation is robust enough to survive most methodological objections. However, the WMI "working memory" claims and the specific percentile figures are substantially undermined by the issues below.

### Major

- **Construct invalidity of Digit Span as a working-memory measure for LLMs.** In humans, Digit Span Forward tests transient auditory attentional encoding; Backwards and Sequencing test active manipulation of a decaying trace. When administered as a text prompt, the sequence is persistent in the model's context window, transforming the task from working-memory load into context retrieval and reordering. The abstract's claim that models "exceeded the 99.5th human percentile in working memory" rests on this conflation. The near-universal perfect scores and identical longest-span values (LDSF = 9, LDSS = 9 for virtually all models; Table 5) further confirm that the adapted task is not discriminating—it has been trivially re-indexed. This is the single most important construct validity problem in the paper and directly undermines a headline claim.

- **Uncontrolled training-data contamination for VCI and WMI.** The WAIS-IV is a published clinical instrument whose items, practice materials, and worked examples are broadly available online. The near-unanimous perfect or near-perfect scores on the Information subtest (18–19/19 across all but one model; Table 2) are entirely consistent with memorization of known test content rather than novel retrieval of crystallized knowledge. The paper does not include any novel-item controls, parallel-form comparison, or contamination analysis. Without this, the VCI and WMI percentile claims are uninterpretable: the paper cannot distinguish "cognitive ability" from "training-set familiarity," which is precisely the distinction that motivates using a population-normed test in the first place.

- **No prompt-sensitivity analysis or reproducibility controls.** The paper does not report decoding temperature, number of trials per item, session reset policy, or API version dates. For proprietary commercial APIs that change continuously, results obtained on undated API calls are not reproducible. Even modest temperature variation can alter Arithmetic and Similarities scores substantially. A single-run point estimate with no uncertainty quantification provides no basis for confident rank-ordering of models or for the discrepancy significance claims.

- **Human normative statistics applied without validity transfer.** Converting raw scores to age-normed scaled scores, and then applying WAIS discrepancy base rates and critical values as inferential thresholds, presupposes that the psychometric model (item-response structure, inter-index covariance, normative distribution) is the same for AI systems as for the human standardization sample. It is not, and no validation is offered. The p-values in Tables 3–5 are properties of the human normative sample, not uncertainty estimates over model behavior. Reporting these as "statistically significant relative strength/weakness" for AI models conflates a human-normed reference with inferential statistics. At minimum, these tables should be reframed as descriptive comparisons to human discrepancy distributions, not significance tests.

- **Image preprocessing and PRI administration details absent.** The PRI tasks were administered as images to multimodal models, but the paper provides no information about image resolution, format, rendering pipeline, or API-specific preprocessing. Commercial VLM backends differ substantially in how they tokenize and compress images. Since PRI performance is the paper's most consequential finding, the omission of these details makes PRI results non-reproducible and leaves open whether observed floor effects are driven by genuine visuospatial reasoning failures or by preprocessing artifacts that render stimuli unrecognizable.

- **No item-level or error analysis.** The paper reports only aggregate subtest scores. For PRI, where nearly all models score at or near the minimum (1/19 on Matrix Reasoning and Picture Completion; Table 2), there is no analysis of what models actually output, whether errors are systematic, and whether failures occur at image parsing, spatial relation extraction, or response selection. This gap makes the "profound inability to interpret and reason on visual information" claim vague and the discussion about specialized architecture speculative.

- **Model versioning and reproducibility.** Commercial model names alone (e.g., "GPT-4o," "Claude 3.5 Sonnet") are insufficient identifiers because these endpoints are silently updated. Exact API version strings and evaluation dates are missing, meaning another researcher cannot reproduce the results even in principle.

### Minor

- **Arithmetic timing and Digit Span modality confound.** In human WAIS-IV administration, Arithmetic is delivered under time pressure with mental calculation only. The paper does not state whether timing constraints were enforced or whether models were permitted chain-of-thought steps. If neither constraint was applied, Arithmetic scores are inflated relative to human norms. Similarly, phrases like "Just say what I say" and "I'm listening!" artifacts (reported in Section 2.1) reveal incomplete adaptation; it is unclear how these interface-error responses were scored.

- **Inter-rater reliability not reported.** The paper mentions that ambiguous items were reviewed jointly, but provides no inter-rater reliability coefficient, count of ambiguous items, or scoring rubric for open-ended responses. For Similarities, Vocabulary, and Comprehension, where partial-credit scoring is judgment-dependent, this omission is important for assessing scoring quality.

- **Selection of 25–29 age band is unjustified.** The paper applies norms for 25–29 year olds without explaining why this band is appropriate for AI systems. Since all normative comparisons in the paper flow from this choice, the rationale should be stated, or the paper should show insensitivity to band choice.

- **Ceiling saturation limits model differentiation.** Most models achieve scaled scores of 19/19 on Information, Digit Span, and LN Sequencing (Table 2), and identical longest span values (Table 5). When the instrument saturates for the majority of models, it cannot rank-order models or track improvement. The claim that "training data and parameter count are resulting in significant advances in cognitive ability" is not supportable from ceiling-saturated subtests.

- **Positive manifold claim is under-supported.** With only six multimodal models, saturated VCI/WMI scores, and no formal correlation analysis, the claim that the positive manifold "holds for VCI and WMI and fails when PRI is included" is a post-hoc observation about mean levels, not a structural claim about inter-test correlations. The modality confound (VCI/WMI are text-in/text-out; PRI is image-in/text-out) is a simpler explanation for the break than a claim about cognitive architecture.

### Tiny

- The paper leaves adapted prompt templates in an appendix but does not specify whether prompts were identical across sessions or whether any prompt ordering effects exist.
- The distinction between text-only and multimodal models means VCI/WMI comparisons span all ten models while PRI spans only six; several claims in the discussion generalize across both sets without flagging this asymmetry.

---

## Nice-to-Haves

- A **text-only baseline for PRI**: administer textually-described versions of Matrix Reasoning and Figure Weights to the same models to isolate whether poor PRI scores reflect a visual encoding bottleneck or a reasoning bottleneck. This would directly strengthen the paper's central interpretive claim.

- **Novel parallel-form items** for VCI and WMI: construct structurally matched items not plausibly present in training data. If performance degrades significantly, the contamination concern is confirmed; if performance is stable, the crystallized-knowledge framing gains credibility.

- **Comparison to a retrieval-only baseline** (e.g., a search engine or embedding lookup) on the Information and Vocabulary subtests. This would help bound how much of the high VCI performance reflects reasoning versus verbatim retrieval.

- **Qualitative PRI case studies**: showing actual model inputs (the WAIS images) and model responses for 2–3 Matrix Reasoning and Picture Completion items would substantially clarify whether failures are at perception or reasoning, and would make the "profound inability" characterization more concrete and convincing.

- **Open-weight model replication**: including at least one reproducible open-weight model (e.g., LLaMA-family or Mistral-family) would allow prompt ablations, temperature sweeps, and independent replication that proprietary APIs cannot support.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Reviewer 2: "Methodological Contradiction on Visual Tasks"** — This is a misread. The paper clearly states that VCI/WMI were converted to text prompts for all models, and PRI was administered to multimodal models as images (because they "require both image recognition"). There is no contradiction; VCI/WMI ≠ PRI in terms of modality. The *lack of detail* about how PRI images were prepared is a legitimate concern, but the supposed logical contradiction does not exist.

- **Harsh Critic: Copyright/test security concerns** — The use of WAIS-IV items in a research context is standard in the psychometric and clinical psychology literature. Raising copyright as a scientific weakness is outside the scope of peer review.

- **Harsh Critic: Ethical/anthropomorphism/public discourse risks** — These are legitimate societal concerns but are not scientific weaknesses of the paper that affect its claims or validity. They belong in a broader impact section, not in a weakness list.

- **Harsh Critic: "Title overstates the contribution"** — The title is admittedly broad, but titling conventions vary widely across ICLR submissions. This is a pure formatting/presentation nit.

- **Reviewer 2: "Limited Machine Learning Insight" / lack of architectural analysis** — The paper explicitly scopes itself to a comparative benchmarking study, not an architectural analysis. Demanding mechanistic architectural explanation (attention mechanisms, visual encoders) is scope creep. It is reasonable to note as a nice-to-have but not a weakness of a benchmarking paper.

- **Harsh Critic: "The paper lacks deep engagement with AI evaluation literature"** — The relevant citation comparison was made by the harsh critic without citing specific omitted works. Per instructions, missing related works claims are removed since external sources cannot be verified.

---

## Novel Insights

The reviews collectively surface one non-obvious observation worth highlighting: the within-PRI dissociation reported in the paper—Claude 3.5 Sonnet showing dramatic improvement in Figure Weights (quantitative/analogical visual reasoning) and Matrix Reasoning (abstract pattern recognition) over Claude 3 Opus, while Picture Completion (detection of missing visual detail) showed no improvement—suggests that the visual deficit is not monolithic. Tasks requiring symbolic or relational abstraction over visual input may be more tractable for current architectures than tasks requiring holistic scene understanding or anomaly detection. This suggests a productive decomposition of "visual reasoning" into abstraction-over-visual-relations versus perceptual-completeness, which future work could operationalize with targeted interventions.

---

## Suggestions

1. **Replace or supplement Digit Span with a genuine manipulation task**: Use symbol sequences (emoji, invented glyphs, or non-ASCII characters) that cannot be trivially tokenized or memorized. Report performance on both original and novel-symbol versions side by side to assess what the WMI score actually measures.

2. **Introduce contamination controls before reporting VCI percentiles**: Construct novel Information and Vocabulary items matched in difficulty to the original WAIS items but using facts/words unlikely to appear in training corpora. Report performance on both and acknowledge the gap as a contamination upper bound.

3. **Provide item-level PRI error analysis**: For each PRI subtest, show at minimum a breakdown of response types (no response, wrong option, formatting failure, semantically coherent wrong answer) across the 6 multimodal models. This would replace the current vague "profound inability" framing with actionable evidence about where exactly the bottleneck lies.

4. **Specify API versions and evaluation dates** for all models, or re-run on pinned open-weight models, to make the study reproducible.

5. **Reframe the discrepancy tables**: Relabel Tables 3–5 to make clear that the p-values and base rates are *human normative references* for context, not inferential statistics computed over AI model uncertainty. This is a presentation fix that would substantially reduce the risk of misinterpretation.

6. **Run a temperature/prompt sweep on at least one representative model**: Even a 3-temperature × 3-prompt-phrasing grid on Similarities and Arithmetic would quantify how sensitive the scaled scores are to administration choices, providing at least a rough uncertainty bound on the reported percentiles.