=== CALIBRATION EXAMPLE 22 ===

# Final Consolidated Review
## Summary

ChemTable introduces a large-scale benchmark of 1,382 real-world chemical tables curated from top-tier chemistry journals (2015–2024), with expert-annotated cell layouts, logical structures, and domain-specific labels including SMILES mappings. It defines two core tasks—table recognition (structure/content extraction with fine-grained retrieval and molecular recognition) and table understanding (descriptive and reasoning QA across 9,886 instances)—and evaluates 7–10 multimodal LLMs, revealing significant gaps between current models and human experts, particularly in molecular structure interpretation and fine-grained cell-level alignment.

## Strengths

- **Fills a genuine and important gap in scientific AI evaluation.** Existing table benchmarks (WikiTQ, FinQA, SciTab, MMTab) lack chemical-domain specificity—none incorporate molecular structures, SMILES mappings, or chemical symbolic conventions. ChemTable is the first benchmark to systematically test multimodal models on tables where visual molecular diagrams, symbolic notation, and domain-specific abbreviations are interleaved within structured data. This is a concrete, needed contribution.

- **Rigorous, multi-phase annotation pipeline with strong quality control.** The three-phase annotation process (Section D.3)—coarse demarcation → fine-grained cell-level annotation → SMILES linking—combined with inter-annotator agreement metrics (IoU 0.96 for cell boundaries, 0.94 for cell content text, 0.99 for SMILES extraction; Appendix J) demonstrates substantial annotation care. The inclusion of 2,122 manually annotated complex reasoning questions alongside rule-based and LLM-assisted generation provides layered coverage.

- **Revealing and well-documented failure modes.** The qualitative case studies (Appendix M) are a standout contribution: they demonstrate concrete, distinct failure types—fine-grained positional misalignment (Figure 15), visual-style hallucination (Figure 16), domain-specific symbol misbinding (Figure 17), and multi-hop reasoning breakdown despite correct intermediate steps (Figure 18). These go beyond reporting accuracy numbers and provide actionable diagnostic information for the community.

- **Thoughtful evaluation design with domain-adapted metrics.** The replacement of normalized edit distance with the Tanimoto coefficient for SMILES-containing cells (Section 4.1) is a principled adaptation that avoids penalizing valid structural isomorphisms. The three-way input modality comparison (Image/HTML/Hybrid; Figure 5) and the unanswerable-question analysis (Table 5) add valuable dimensions beyond standard benchmark reporting.

## Weaknesses

- **Automated grading reliance with limited human validation.** The primary evaluation for table understanding uses GPT-4.1-nano as a binary classifier (Section 5.1), with only 20% human verification reporting 96.8% agreement. For a benchmark intended to serve as a community standard, this is a meaningful concern: the agreement rate between the two human verifiers themselves is never reported (if humans disagree often, the GPT agreement metric becomes less informative), and the 96.8% figure aggregates across all question types—it may be lower for the most important reasoning categories. The benchmark's reliability as an evaluation tool depends on the trustworthiness of its grading, and this pipeline needs stronger validation.

- **Data filtering introduces potential size-dependent bias.** Section 3.3.4 describes filtering out questions that Qwen-2.5-7B answered correctly on a first attempt, explicitly to increase difficulty. This creates a benchmark that is, by construction, harder for smaller models—potentially exaggerating the observed performance gaps between model sizes. The paper does not acknowledge this as a limitation, nor does it report what fraction of questions was removed or how results change with and without filtering. This should at minimum be disclosed and its impact analyzed.

- **No decomposition of recognition vs. reasoning failures.** When a model fails at table understanding, it is unclear whether the failure originates from misreading cell content (an OCR/recognition error) or from incorrect reasoning over correctly parsed content. Without this diagnostic decomposition, the benchmark's guidance for model improvement is limited—one cannot tell whether to invest in better visual parsing or better chemical reasoning. The paper's own Figure 5 (Text QA outperforming VQA) hints that recognition is a bottleneck, but this is not systematically analyzed across question types.

- **Human baseline comparability is under-specified.** Appendix L states that five chemistry experts answered questions with "scratch paper and a basic calculator," but it is unclear whether humans viewed the same table images as the models or had access to parsed/HTML representations. Given the paper's own finding that Text QA outperforms VQA (Figure 5), this input-modality difference could substantially affect the human–model gap. Additionally, no inter-annotator agreement is reported for human answers to reasoning questions, making it difficult to assess the reliability of the human baseline.

- **Limited sub-domain coverage within chemistry.** The selected journals (ACS Catalysis, JACS, Organic Letters, Angewandte Chemie, Chem) are heavily oriented toward organic synthesis and catalysis. Condition optimization and substrate screening tables alone comprise >50% of the dataset. Inorganic chemistry, materials science, analytical chemistry, and biochemistry table formats—each with distinct symbolic conventions and visual encodings—are not represented. This limits the benchmark's ability to generalize claims about "chemical table understanding" as a whole.

- **Unanswerable question distribution is not reported.** Table 5 evaluates model behavior on unanswerable questions (missing content, ambiguity, missing style), but the paper does not report how many such questions exist in the dataset, their proportion relative to answerable questions, or their distribution across question types. Without this, it is impossible to assess whether the unanswerable-question evaluation reflects a meaningful test or a sparse edge case.

## Nice-to-Haves

- Fine-tuning experiments showing that training on ChemTable improves model performance would strengthen the benchmark's utility claim beyond pure evaluation.
- A direct comparison with a general-domain table benchmark (e.g., evaluating the same MLLMs on PubTabNet or FinTabNet using the same metrics) would isolate whether observed limitations are chemistry-specific or reflect general table understanding gaps.
- Statistical significance testing or confidence intervals for model comparisons, particularly where differences are small (e.g., TEDS-Struct 93.12 vs. 92.58).
- Performance breakdown by the six table types defined in Section 3.1.1 (condition optimization, substrate screening, etc.) to reveal which experimental reporting formats are systematically more challenging.
- Explicit analysis of the proportion and impact of LLM-generated vs. human-written questions on evaluation outcomes.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Questioning existence/availability of GPT-5, Claude-4.5-Sonnet, Gemini-2.5-Pro as baselines.** Hard rule: the paper cites these models; they are assumed to exist and be available.
- **Complaints about garbled Table 3 and Table 4 formatting.** This is a PDF extraction artifact, not a paper flaw.
- **Missing related works on molecular property prediction, reaction yield prediction, chemical OCR.** Hard rule: cannot flag missing related works without external source verification.
- **Nitpick about abstract saying "over 1,300" vs. exact 1,382.** The abstract uses correct approximate language; 1,382 is indeed "over 1,300."
- **Nitpick about tokenizer choice (Qwen2.5-7B) for computing unique tokens.** Trivial implementation detail.
- **Demand for safety/impact analysis of incorrect chemical extraction.** Scope creep—this is a benchmarking paper, not a deployment paper.
- **Concern about compute requirements for evaluating 78B models.** Nitpick about accessibility; not a weakness of the benchmark itself.
- **Demand for copyright concerns to be "prominently stated in Section 3."** The paper addresses copyright in Appendix S; this is a formatting preference, not a substantive flaw.
- **Reproducibility concern about table images being subject to publisher copyright.** The paper explicitly acknowledges this in Appendix S and releases under CC BY-SA 4.0 where possible. This is a well-known constraint for scientific benchmarks derived from published literature and does not invalidate the contribution.

## Novel Insights

The most striking finding across the reviews and the paper itself is the *multi-hop reasoning breakdown pattern* revealed in Figure 18: Claude-4.5-Sonnet correctly identifies the target row and states "Entry 4 matches both criteria exactly" in its chain-of-thought, yet outputs a value from a different column (the base "LDA" instead of the entry index "4"). This suggests that the bottleneck for complex chemical table reasoning is not in retrieval or numerical comparison, but in the *final mapping step*—correctly routing a resolved row back to the requested schema field. This is a distinct and previously underappreciated failure mode that would not be captured by coarse accuracy metrics alone, and it implies that improving MLLM performance on scientific tables may require better schema-grounded output training rather than just better OCR or more chemical knowledge.

## Suggestions

- Report the inter-annotator agreement between the two human verifiers used for the 20% GPT-grading validation, and break down the GPT-vs-human agreement by question category (reasoning questions likely have lower agreement than descriptive ones).
- Disclose the fraction of questions removed by the Qwen-2.5-7B difficulty filter and, if possible, report a comparison of model rankings with and without filtering to quantify the introduced bias.
- Add a systematic error decomposition: for a sample of failed reasoning questions, determine whether the model's OCR of the relevant cells was correct, and report the proportion of failures attributable to recognition errors vs. reasoning errors.
- Clarify in Appendix L what input modality human annotators received (image only, or also HTML/text), and report inter-annotator agreement for human answers on reasoning questions.

## Axis Evaluations

- **Novelty:** High. First benchmark specifically designed for chemical tables with integrated molecular structure recognition, SMILES evaluation, and chemistry-domain reasoning. Fills an identified gap that existing general and financial table benchmarks do not address.
- **Technical soundness:** Moderate. The benchmark construction and annotation are thorough, but the automated grading pipeline, difficulty-filtering bias, and under-specified human baseline introduce methodological concerns that partially undermine confidence in the reported numbers.
- **Empirical support:** Moderate-to-strong. Extensive evaluation across 7–10 models with fine-grained task breakdowns and informative qualitative case studies. However, the lack of error decomposition and the gap in statistical rigor (no significance tests, no ablation of filtering) limit the depth of conclusions that can be drawn.
- **Significance:** High. The benchmark addresses a real and growing need as MLLMs are increasingly applied to scientific literature. The documented failure modes provide concrete targets for improvement, and the domain-specific evaluation (molecular recognition, chemical symbol interpretation) tests capabilities that general benchmarks miss.
- **Clarity:** Good. The paper is well-structured with detailed appendices. The annotation protocol, question taxonomy, and evaluation setup are clearly described. Some methodological details (human baseline conditions, filtering impact) could be more transparent.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0]
Average score: 4.5
Binary outcome: Reject
