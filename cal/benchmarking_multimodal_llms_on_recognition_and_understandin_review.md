=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
## Summary
This paper introduces **ChemTable**, a benchmark for multimodal understanding of chemical tables drawn from real chemistry literature. The benchmark covers both **table recognition** (structure/content extraction, value/position retrieval, molecular diagram recognition) and **table understanding** (descriptive and reasoning QA), and the experiments show a consistent gap between current MLLMs and human experts, especially on molecule-centric, style-sensitive, and fine-grained alignment tasks.

## Strengths
- **The benchmark targets a genuinely underexplored multimodal regime that general table benchmarks do not capture well.** The paper is not just “tables in science”; it focuses on chemical tables that jointly contain text, numerical conditions, footnotes, formatting cues, and embedded molecular diagrams. This is concretely reflected in the task design: e.g., molecular recognition to SMILES, function-based QA, benzene-ring counting, and yield/condition reasoning.
- **The annotation schema is unusually rich and operationalized in a way that supports multiple evaluation settings.** Beyond cell boxes and text, the dataset includes logical row/column structure, title/reaction/substance/annotation segmentation, style markup (bold/color/italics), and links from structures to SMILES (Appendix D). The conversion from logical annotations to markup is explicitly specified in Algorithms 1–2, which makes the recognition setup much more concrete than many benchmark papers.
- **The paper contributes useful diagnostic findings rather than only leaderboard numbers.** The most compelling empirical result is not the raw ranking, but the mismatch between high global structure scores and poor cell-level grounding: e.g., Table 3 reports TEDS/TEDS-Struct in the high 80s/90s while value/position retrieval remains very low (roughly 17–34 and 33–53 ACC). This isolates a real weakness of current MLLMs: they can often “parse the table” coarsely while failing at exact localization.
- **The qualitative analysis is strong and specific.** Appendix M identifies concrete failure modes that are highly relevant for future model development: miscounting rows despite correct OCR, hallucinating style/color cues, misbinding stereochemical footnotes, and failing at the final schema-selection hop even when intermediate reasoning is correct.
- **Including a human baseline is valuable and reveals that the hardest chemistry-aware tasks remain far from solved.** In Table 4, humans are near ceiling on several tasks and still ahead on chemistry-heavy or style-heavy ones, which supports the paper’s claim that chemical tables expose limitations not visible in easier general-domain settings.
- **The benchmark includes unanswerable questions and analyzes abstention behavior.** This is a meaningful addition in scientific QA, where tables often contain ambiguity, missing values, or style-based references. Table 5 gives a more realistic picture of reliability than accuracy alone.

## Weaknesses

###: Fatal
None.

### Major:
- **The QA construction pipeline relies heavily on LLM generation and LLM-based filtering, which weakens the benchmark’s claim to be a clean measure of independent scientific reasoning.**  
  This concern is real, though not as fatal as the harsh review suggests. The paper states that descriptive questions are derived from annotations, “simple reasoning questions are generated using GPT-4.1,” and “more complex reasoning questions and visually descriptive tasks” include **2,122 manually annotated questions**. Since the dataset contains **9,886 QA pairs**, a large fraction is indeed not human-authored. In addition, Section 3.3.4 filters difficulty by running **Qwen-2.5-7B** and randomly discarding questions it gets right on the first try.  
  Why this matters: this design can shape the benchmark toward the language patterns and difficulty profile induced by the generator and the filtering model, rather than purely reflecting naturally occurring user questions or expert-authored chemistry reasoning. It does not invalidate the benchmark, but it does limit how strongly one should interpret model rankings as measuring “objective” domain reasoning.
- **The evaluation for table understanding depends substantially on an LLM judge, and the validation presented is not strong enough to fully rule out judge bias on the hardest chemistry-specific cases.**  
  Section 5.1 says open-ended QA is scored by **GPT-4.1-nano** with binary correctness, and Appendix G reports **96.8% agreement** with humans on a 20% sample. That is reassuring, but incomplete. The paper does not break down agreement by question type, and the hardest cases in this benchmark are precisely the domain-specific symbolic ones where plausible-but-wrong chemical answers are likely.  
  Why this matters: if grader reliability is uneven across categories, the headline accuracy numbers in Table 4 may be more trustworthy for short descriptive items than for chemistry-heavy reasoning or ambiguous symbolic cases.
- **The paper does not sufficiently disentangle failures of visual/structural recognition from failures of reasoning.**  
  This is one of the most important missing analyses. The benchmark evaluates recognition and understanding separately, and Figure 5 compares image/text/hybrid input for InternVL3, but there is no clean experiment giving models **ground-truth structured tables** and comparing that to image-only QA across the full benchmark. The paper itself provides evidence that recognition is a major bottleneck: Table 3 shows poor value/position retrieval, and Figure 5 indicates text/hybrid inputs help.  
  Why this matters: several claimed “reasoning” failures may actually be upstream perception failures. Without a stronger controlled analysis, the paper cannot cleanly attribute deficits to chemistry reasoning versus table parsing.
- **Some benchmark claims are broader than what the experimental design directly supports.**  
  The paper sometimes frames its findings as revealing general limits of “domain-specific reasoning” in scientific intelligence, but some supporting analyses are narrow. For example, the modality comparison in Figure 5 appears to be run only on **InternVL3**, so the conclusion that hybrid input is generally best across models should be stated more cautiously. Similarly, CoT ablation is only reported for **GPT-4.1** (Appendix C), so it cannot explain cross-model differences in reasoning robustness.

### Minor
- **There is a noticeable inconsistency in how the dataset size is described.**  
  The main text says the dataset comprises “over **1,300** tables,” Table 2 reports **1,382** images, while Appendix D.3 states “we constructed a high-quality dataset comprising **1,500 fully annotated chemical table images**.” This likely reflects different stages or filtering, but the paper should reconcile these counts explicitly.
- **The benchmark is stronger as an evaluation dataset than as a training resource, but this intended use is not stated sharply enough.**  
  Given the scale (roughly 1.3k–1.5k tables), ChemTable is meaningful for benchmarking and perhaps targeted fine-tuning, but not obviously for training broad multimodal table models from scratch. The paper occasionally gestures toward enabling model development more generally, and it would benefit from a clearer statement of intended use.
- **The recognition baseline suite is narrower than ideal for a benchmark paper centered on table recognition.**  
  The paper evaluates multiple MLLMs and includes DECIMER in molecular recognition analysis, but it does not compare against established non-MLLM table extraction pipelines for the recognition task. This does not break the paper’s MLLM benchmarking story, but it limits how informative the recognition results are for the broader table-understanding community.
- **The molecular-recognition metric adaptation is promising but underspecified.**  
  Replacing edit distance with a chemistry-aware similarity measure for molecular cells is sensible. However, the paper does not give enough detail in the main text about the exact conversion/scoring pipeline for molecule predictions, making it hard to assess how robust the metric is to representation differences.

### Trivial
- **Human evaluation methodology could be reported with more statistical detail.**  
  Appendix L explains that three of five chemistry experts answered each question and results are averaged, but variance or agreement statistics for the human baseline would make the ceiling estimate more informative.
- **Some analyses that are compelling in the appendix deserve more quantitative support in the main text.**  
  In particular, claims about molecular complexity, footnotes, and multimodal interference would be stronger with systematic feature-performance correlations rather than primarily qualitative examples.

## Nice-to-Haves
- Add a controlled **ground-truth-HTML QA** evaluation across the main models to quantify how much of the gap is due to perception versus reasoning.
- Report **LLM-judge agreement by category**, especially for function-based QA, molecular recognition, and symbolic/footnote-heavy questions.
- Include a few strong **non-MLLM recognition baselines** for table extraction to better position the recognition difficulty.
- Clarify the exact **dataset splits**, and reconcile the 1,382 vs. 1,500 table counts.
- Provide more granular analysis linking error rates to table properties such as molecule density, merged-cell ratio, or footnote count.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The benchmark cannot claim domain-specific reasoning because Appendix H says questions must be answerable without specialized chemical knowledge.”**  
  This criticism overstates a real tension. Appendix H indeed says annotators should ensure answers are “directly found in the table or logically inferable from it without requiring specialized chemical knowledge.” But the paper’s tasks clearly still include **domain-specific representations and conventions**—e.g., molecular diagrams, stereochemical footnotes, function-based chemistry QA, SMILES extraction, yield/condition reasoning. The benchmark is therefore still domain-specific in input and notation, even if it avoids requiring external chemistry facts not grounded in the table. This is a limitation in scope framing, not a fatal contradiction.
- **Claims about unreleased, unverifiable, or questionable existence/status of cited models or tools.**  
  Per instruction, these are removed.
- **Potential data contamination with model pretraining.**  
  This is speculative and not evidenced from the paper. It should not be used as a concrete weakness here.
- **Copyright/license concerns based on doubting practical release utility.**  
  The paper explicitly discusses licensing in Appendix S. Without external legal analysis, this should not be elevated as a review criticism.
- **Formatting/extraction complaints about tables in the parsed submission text.**  
  These are artifacts of the extracted text, not paper issues.
- **Complaints that the paper misses related work.**  
  Not included per instruction.

## Novel Insights
The most interesting synthesis across the reviews and the paper itself is that ChemTable’s strongest contribution is not merely introducing a chemistry benchmark, but exposing a **three-layer failure stack** in current MLLMs: (1) coarse structure parsing can look strong under global metrics, (2) exact grounding to cells and symbols remains weak, and (3) chemistry-specific reasoning failures are often inseparable from those grounding errors. In other words, the benchmark suggests that “scientific reasoning” deficits in MLLMs may often be **grounding-and-binding deficits first, reasoning deficits second**. This is a useful reframing for future work: improving symbolic scientific QA may require better multimodal alignment and schema binding as much as better reasoning.

## Suggestions
- Run a main-table experiment where models answer the same QA set under three conditions: **image only**, **ground-truth HTML only**, and **image + ground-truth HTML**. This would directly separate perception failure from reasoning failure.
- Provide **per-category judge validation** for GPT-4.1-nano, especially on chemistry-specific and ambiguous questions, not just an overall 96.8% agreement.
- Make the QA provenance more transparent: report exactly how many questions per category are rule-based, LLM-generated, manually authored, and manually revised.
- Reconcile the dataset-size discrepancy and clearly define the released split(s).
- Add at least one or two strong traditional recognition baselines for the table-recognition portion, so the benchmark is informative beyond MLLM-vs-MLLM comparison.
- Tone down broad claims about general scientific reasoning where the evidence is currently based on single-model ablations or narrow analyses.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0]
Average score: 4.5
Binary outcome: Reject
