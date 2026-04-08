=== CALIBRATION EXAMPLE 23 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately reflects the contribution — a benchmark paper covering recognition and understanding of chemical tables. The abstract states "more than 9,000 QA instances," which aligns with Table 2 (7,344 + 2,542 = 9,886). However, the abstract's claim that "closed-source models lead overall but still fall short of human-level performance" is only partially true: Table 4 reveals that on several subtasks (e.g., *Annotation Description*, *Numerical Statistics*), multiple closed-source models already exceed human-level performance. This selective framing requires correction.

---

### Introduction & Motivation

The motivation is clear and well-stated. Chemical tables are indeed a distinct and underexplored modality combining symbolic, numeric, and graphical elements. The six-category table taxonomy is helpful for scoping the benchmark.

**A minor but real gap:** The introduction cites ChartQA and ChartX as related benchmarks for figures/charts, but does not cite existing work on scientific table understanding for chemistry specifically (e.g., ChemTables in the information extraction community, or ORD datasets). A more thorough literature survey of cheminformatics table-extraction tools (beyond what appears in Section 2) would better justify the claim that the domain is entirely unexplored.

---

### Dataset Construction and Annotation (Section 3.1)

This is the heart of the paper and deserves the closest scrutiny.

**1. Inconsistent dataset size claims.** The abstract says "over 1,300 tables," Table 2 reports 1,382 total images, and Appendix D.3 states "1,500 fully annotated chemical table images." These three numbers are mutually inconsistent and never reconciled. How many images are actually in the benchmark?

**2. No train/test split description.** For a benchmark paper, this is a serious omission. How is the data partitioned for evaluation? Is the entire dataset an evaluation-only test set? Is there a validation split? This affects how results from fine-tuned models would be reported and how users should interpret leakage risks.

**3. Data contamination is not addressed.** All 1,382 tables come from a small set of high-impact journals (ACS Catalysis, JACS, Angewandte Chemie, etc.) published between 2015–2024. The models being evaluated — especially GPT-5, Gemini-2.5-Pro, Claude-4.5, and GPT-4.1 — have training data cut-offs that likely include the source publications. There is no contamination analysis, no attempt to assess whether models have "memorized" tables from these papers, and no discussion of this risk. For a benchmark whose primary purpose is to evaluate models, this is a fundamental concern.

**4. Difficulty filtering is methodologically questionable.** Section 3.3.4 filters out questions that Qwen-2.5-7B answers correctly on the first pass ("randomly discarded"). This introduces two problems: (a) the benchmark difficulty is artificially calibrated against a specific model (Qwen-2.5-7B), potentially biasing results against similarly-sized models; and (b) the "random" discarding is undefined in terms of fraction — how many questions were removed? What fraction of the original pool was retained? Without these details, the filtering cannot be reproduced or assessed for bias.

**5. Repeated questions.** Table 2 shows 7,344 descriptive questions but only 1,512 unique questions — each unique question appears ~4.9 times on average. This means that model accuracy on repeated questions may be inflated or biased by systematic errors. The paper should explain what varies between repeated instances (is it different tables with the same question template?) and analyze whether accuracy differs across repetitions.

**6. Copyright and licensing (Appendix S).** This appendix is referenced in Section 3.1.1 but is entirely absent from the paper. Table images extracted from ACS, Wiley, and RSC publications are subject to strict copyright. This is not a trivial concern — several benchmark papers have faced significant legal challenges for exactly this use case. The authors must clarify licensing arrangements before any public release can be considered credible.

---

### Table Recognition (Section 4)

**7. TEDS* is insufficiently described.** The paper introduces a modified TEDS that replaces normalized edit distance with Tanimoto coefficient for cells containing molecular graphs (TEDS*). This is a reasonable approach, but: (a) no ablation is provided showing how often molecular cells occur and what impact the modification has on the overall metric; (b) the Tanimoto coefficient is computed on fingerprints — which fingerprint type (Morgan, MACCS, etc.) and radius? These choices can substantially affect similarity scores and must be specified.

**8. Value Retrieval and Position Retrieval accuracy (~30–54%) is not contexticalized.** Table 3 shows very poor fine-grained retrieval across all models. However, the authors do not report a random baseline or a simple majority-class baseline, making it impossible to know whether these numbers represent meaningful model capability or near-chance performance.

**9. Missing comparison to specialized table recognition models.** The paper frames itself as benchmarking MLLMs, but for the table recognition task specifically, there are well-established methods (TableFormer, MASTER, TableNet, etc.) that are not evaluated. Even a brief comparison to one specialized method would contextualize how far MLLMs are from the state of the art in table structure recognition.

**10. Molecular recognition comparison (Figure 3).** The comparison between MLLMs and DECIMER on molecular formula recognition is informative, but the paper states "real-world diagrams" vs. "synthetic diagrams" without defining what synthetic means in this context (rendered from SMILES? augmented?). The sample sizes for each condition are not reported, making it impossible to assess statistical significance.

---

### Table Understanding (Section 5)

**11. GPT-4.1-nano as judge: circularity and validity.** The paper uses GPT-4.1-nano to classify model answers as correct/incorrect. This creates multiple problems:
- For GPT-5 and GPT-4.1 outputs, the judge is a weaker version of the evaluated model, creating a potential circularity (the judge may favor patterns from its own model family).
- The agreement metric in Equation 1 is defined as |J_human ∩ J_GPT| / |J_sampled|, which measures only how often *both* judge an answer as correct — it ignores agreement on *incorrect* judgments. This is not a proper agreement metric; a Cohen's kappa or full 4-cell agreement matrix is needed.
- 96.8% agreement sounds high, but what is the base rate? If 80% of answers are trivially correct, this number is uninformative.

**12. Human baseline is incomplete.** Appendix L states that "human performance is only reported for tasks that require chemical expertise or complex reasoning; purely descriptive element-level tasks are not annotated by humans." This means the headline comparison "MLLMs vs. human" is not a full comparison — for ~40% of the task categories, there is no human baseline at all. The paper should either complete the human evaluation or be more explicit that the comparison is partial.

**13. Inconsistency in model set.** Table 3 (recognition) evaluates 7 models; Table 4 (understanding) evaluates 10 models, with GPT-5, Gemini-2.5-Pro, and Claude-4.5 appearing only in understanding. No justification is given for why these three models were not also evaluated on recognition tasks. The lack of consistency makes cross-task comparisons difficult.

**14. No statistical significance testing.** Throughout Tables 3, 4, and 6, many claimed differences (e.g., open-source vs. closed-source, forward vs. inverse prediction) are reported without confidence intervals or statistical tests. Given the modest sample sizes for some subtask categories (e.g., a category with ~50 questions), several observed differences may not be statistically significant.

---

### Analysis Sections (Sections 5.3, Appendix A–C)

**15. Hybrid QA finding is unsurprising.** The observation that Hybrid QA (text + image) outperforms VQA-only is expected and mechanistically trivial: more information → better performance. The interesting scientific question would be *when* does image input add unique value beyond the text, and whether models can correctly attribute evidence to the visual modality. This is not explored.

**16. Response-length analysis (Appendix B) is underpowered.** The claim that "accuracy peaks at moderate response length" is analyzed by binning outputs into length ranges. The authors note that "sample counts drop sharply at extreme lengths, limiting statistical confidence" but report the results anyway. Given the small bin sizes at extremes, this analysis is inconclusive as presented.

**17. Forward vs. inverse reasoning (Appendix A).** The claimed asymmetry (91.74% vs. 86.82% for GPT-4.1) is described as a "fundamental challenge," but a ~5% gap on two highly correlated subtasks is a modest effect. The analysis needs statistical testing, and the claim about "abductive reasoning" is speculative.

---

### Writing & Clarity

**18. Structural incoherence in Section 3.3.** The section numbering is: 3.3.3 (Reasoning Questions) appears before 3.3.1 (Task Definition) and 3.3.2 (Descriptive Questions) in the actual PDF. While this is likely a PDF rendering artifact, the logical flow is still disrupted in reading, suggesting the paper was assembled under time pressure.

**19. Table 4 row ordering.** Table 4 presents human performance in the last column under "Human" but does not clarify under which conditions humans were tested (with or without chemistry knowledge, time limits, etc.). Appendix L clarifies this, but the table caption should at minimum cross-reference it.

---

### Limitations & Broader Impact

**20. Narrow chemistry scope.** The benchmark is predominantly organic synthesis tables (condition optimization + substrate screening = >50%). This is a real but narrow slice of chemistry. The scope excludes computational chemistry, spectroscopy, materials science, and biochemistry tables. The authors do not acknowledge this limitation.

**21. Static benchmark problem.** The authors benchmark GPT-5 and Gemini-2.5-Pro, implying the benchmark was constructed and evaluated against very recent models. There is no discussion of how quickly the benchmark will be "solved" or how it will be maintained/extended as models improve. Given the rapid pace of MLLM development, this is a practical concern for a community resource.

**22. No error analysis.** Despite claiming key insights about "symbolic understanding gaps" and "limited transferability," the paper provides no qualitative error analysis. What specific types of molecular structures cause failures? What kinds of domain-specific questions do models most frequently misanswer? This analysis is more informative than aggregate accuracy numbers and is expected in a mature benchmark paper.

---

### Overall Assessment

ChemTable addresses a real and underexplored problem: evaluating MLLMs on multimodal chemical tables, which combine structured layouts, symbolic notation, and embedded molecular graphics. The breadth of annotation (cell-level structure, SMILES mapping, multi-type QA) and the scale of expert involvement represent genuine effort. However, the paper has several serious weaknesses that limit confidence in the results. The inconsistent dataset size (1,300 vs. 1,382 vs. 1,500 tables), the missing copyright appendix, the absence of contamination analysis for models trained on the same literature, and the incomplete/methodologically flawed human baseline together undermine the core claim that ChemTable provides a reliable evaluation platform. The use of GPT-4.1-nano as a judge with a non-standard agreement metric is a further validity concern. The paper also lacks train/test split definition, statistical significance reporting, and qualitative error analysis. For ICLR, dataset papers must be particularly rigorous in methodology and reproducibility. In its current form, ChemTable represents a valuable dataset construction effort, but the benchmark evaluation framework requires substantial revision before the results can be considered authoritative. The contribution is promising but not yet ready for publication without major revisions addressing the issues above.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces ChemTable, a benchmark dataset comprising over 1,300 real-world chemical tables and ~9,900 question-answering pairs, designed to evaluate multimodal large language models (MLLMs) on table recognition and domain-specific understanding. The authors benchmark a diverse suite of modern open-source and proprietary MLLMs against human experts, revealing that while models achieve strong structural parsing, they significantly underperform on fine-grained alignment, molecular structure recognition, and complex chemical reasoning. ChemTable fills a critical gap in scientific AI evaluation by capturing the multimodal, symbolic, and semantic complexities unique to chemical literature.

### Strengths
1. **Addresses a Clear and Timely Gap:** The paper effectively identifies that existing scientific benchmarks focus heavily on charts, figures, or general-domain tables, neglecting the dense, symbol-heavy, and visually embedded nature of chemical tables. The dataset's focus on real-world peer-reviewed literature (2015–2024) ensures high ecological validity for AI4Science applications.
2. **Comprehensive and Rigorous Evaluation:** The evaluation covers both recognition (structure reconstruction, value/position retrieval, SMILES extraction with Tanimoto-adjusted metrics) and understanding (descriptive and reasoning QA), benchmarking 7–10 state-of-the-art MLLMs alongside a human expert baseline. Results are consistent and highlight clear capability gaps, particularly in arithmetic-heavy and domain-specific tasks.
3. **Strong Quality Control and Filtering:** The construction pipeline includes expert annotation, multi-stage QA generation, and automated difficulty filtering to remove trivial examples. The use of unanswerable questions and the validation of the automated LLM evaluator (GPT-4.1-nano) against human judgments (96.8% agreement) demonstrate careful attention to benchmark reliability.
4. **Rich Diagnostic Analysis:** Beyond leaderboard scores, the paper provides valuable insights through modality comparisons (Hybrid > Text > VQA), model behavior on unanswerable queries (Table 5), performance decay over multi-hop depth, and detailed qualitative case studies (Section M). These analyses clearly diagnose failure modes like visual-style hallucination, footnote misbinding, and schema misalignment.

### Weaknesses
1. **Heavy Reliance on LLM-Generated and LLM-Graded Content:** A substantial portion of reasoning and function-based questions are synthesized by GPT-4.1/Claude, and answers are evaluated via binary GPT-4.1-nano classification. While validated, this introduces potential circularity and may limit the benchmark's ability to stress-test models beyond the reasoning envelope of the generator/evaluator models. It also risks penalizing semantically valid but lexically divergent model outputs.
2. **Ambiguity in Human Baseline Coverage:** The paper notes that human performance is only reported for tasks requiring chemical expertise or complex reasoning (Section L), but does not explicitly map which specific sub-tasks in Table 4 include human scores. Without a complete breakdown and inter-annotator agreement statistics for QA answers, the human baseline's comparability across all categories remains partially opaque.
3. **Dataset Accessibility and Copyright Constraints:** While Appendix S outlines licensing efforts and source attribution, compiling tables from high-impact publishers (ACS, Wiley, etc.) often involves restrictive copyright. Claiming a CC-BY-SA 4.0 license for the compiled dataset may conflict with underlying publisher terms, potentially complicating public distribution, reproducibility, and broader community adoption.
4. **Evaluation Metric Granularity:** The reliance on exact-match/edit-distance for descriptive QA and binary correctness for open-ended reasoning may be too rigid for scientific contexts where multiple valid phrasings or partial credit exist. Models that demonstrate correct logical steps but minor numerical rounding differences or alternative valid nomenclature may be unfairly penalized.

### Novelty & Significance
The novelty is incremental but practically significant. The specific focus on chemical tables as a distinct multimodal reasoning challenge, combined with the detailed annotation schema linking table cells to SMILES representations and stylistic features, is a solid contribution. The paper aligns well with ICLR's emphasis on rigorous evaluation, understanding LLM limitations, and AI for scientific discovery. ChemTable provides a necessary diagnostic lens for the AI4Science community, exposing bottlenecks that general benchmarks obscure, and will likely serve as a foundational resource for future model training and evaluation in domain-specific visual reasoning.

### Suggestions for Improvement
1. **Diversify Question Origin and Evaluation:** Increase the proportion of fully human-authored reasoning questions to mitigate generator bias. Implement an evaluation protocol that incorporates semantic tolerance or step-wise partial credit for multi-step reasoning, reducing over-reliance on strict string matching or binary LLM judgment.
2. **Clarify Baseline Coverage and Agreement:** Explicitly detail which question categories in the main results table include human evaluations. Report inter-annotator agreement for the human answers, and ensure the human baseline process (e.g., time limits, allowed tools) is fully transparent for fair model-to-human comparison.
3. **Resolve Licensing and Access Pathways:** Provide a clear, legally compliant distribution strategy for the dataset (e.g., controlled academic access, hashed image links to source papers, or a fully synthetic/derived subset for unrestricted use). This will ensure reproducibility and prevent copyright-related barriers to community adoption.
4. **Deepen Diagnostic Metrics:** Supplement the qualitative case studies with quantitative failure analysis, such as tracking hallucination rates by molecule complexity, attention visualization for cell alignment errors, or correlation between table density (cells/images per area) and recognition accuracy. This would strengthen the paper's utility for model developers aiming to close the identified gaps.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No fine-tuning experiments** — The paper claims ChemTable will "facilitate future advancements" but provides zero evidence that training on this benchmark actually improves model performance. Add fine-tuning results showing models improve after training on ChemTable, otherwise the benchmark's utility claim is unsupported.

2. **Missing comparison against existing scientific table benchmarks** — The paper claims chemical tables are uniquely challenging but never evaluates the same models on SciTab or other scientific table benchmarks to demonstrate the performance gap. Without this, the claim that ChemTable is more difficult is unverified.

3. **No ablation on molecular structure complexity** — The paper identifies molecular recognition as a key bottleneck (Figure 4) but doesn't systematically vary molecular complexity (e.g., number of rings, atoms) to show where exactly models fail. This undermines the specificity of the "molecular formulas pose a bottleneck" claim.

4. **Domain-specific model evaluation is buried in appendix** — ChemVLM and Table-LLaVA comparisons appear only in Appendix K, not main results. For a paper claiming to benchmark domain-specific understanding, specialized chemistry models should be in the main evaluation table.

5. **No test-time scaling or reasoning strategy ablation** — The CoT ablation (Appendix C) is minimal. ICLR expects analysis of whether increased compute/reasoning steps helps on hard tasks. Show whether more reasoning tokens closes the gap on multi-hop questions.

### Deeper Analysis Needed (top 3-5 only)
1. **Error attribution is unclear** — When models fail on molecular structures, is it OCR failure, chemical knowledge gap, or visual parsing? The paper conflates these. Add error categorization showing what fraction of failures are due to each cause, otherwise "molecular recognition bottleneck" is too vague.

2. **No correlation between table complexity and performance** — Tables vary in size, number of embedded images, and structural complexity. Show performance vs. these metrics to demonstrate the benchmark actually captures difficulty gradients, not just random variation.

3. **Human evaluation agreement needs per-task breakdown** — The 96.8% agreement rate (Appendix G) is reported overall but not per question type. If agreement is low on reasoning questions, the automatic evaluation reliability for the paper's key claims is questionable.

4. **No analysis of why closed-source models outperform** — The paper states closed-source models dominate but doesn't analyze what capabilities drive this (better OCR, more chemical knowledge, better reasoning?). Without this, the analysis doesn't guide future model development.

5. **Missing failure mode frequency statistics** — Case studies (Section M) show failure types but don't report how common each failure is across the dataset. A rare failure mode showcased prominently misrepresents the actual bottlenecks.

### Visualizations & Case Studies
1. **Confusion matrices for question types** — Show which question types models confuse with each other (e.g., does the model answer "Yield" questions when asked about "Conditions"?). This reveals systematic misunderstanding vs. random errors.

2. **Attention/heatmap visualization on table cells** — For failed QA pairs, show which table regions the model attended to. This reveals whether models look at the right cells before answering incorrectly, distinguishing reasoning failures from retrieval failures.

3. **Performance vs. molecular structure complexity scatter plot** — Plot accuracy against molecular properties (molecular weight, number of rings, etc.) for the molecular recognition task. This shows whether failure is gradual or catastrophic as complexity increases.

4. **Side-by-side success/failure pairs for the same table** — Show cases where models succeed on descriptive questions but fail on reasoning questions for the same table. This isolates reasoning capability from table parsing capability.

### Obvious Next Steps
1. **Include fine-tuning results** — A benchmark paper at ICLR should demonstrate that training on the benchmark improves performance. Fine-tune at least one open-source model on ChemTable and show gains.

2. **Add temporal generalization test** — Since tables span 2015-2024, evaluate whether models trained on earlier years generalize to later years. This tests whether the benchmark captures evolving scientific conventions.

3. **Cross-dataset transfer evaluation** — Test whether models fine-tuned on ChemTable improve on other scientific QA tasks. This demonstrates the benchmark's broader utility for scientific understanding.

4. **Include more recent domain-specific baselines** — The paper evaluates general MLLMs but should include more chemistry-specific models (e.g., ChemLLM, Mol-LLaVA) to establish a stronger domain-specific baseline.

# Final Consolidated Review
## Summary

ChemTable introduces a benchmark of 1,300+ real-world chemical tables from peer-reviewed literature, with expert annotations for cell layouts, logical structures, and SMILES mappings. The benchmark supports two tasks: table recognition (structure extraction and content retrieval) and table understanding (descriptive and reasoning QA). Evaluation of 7–10 multimodal LLMs reveals that while models achieve strong structural parsing, they significantly underperform on molecular recognition, fine-grained retrieval, and domain-specific reasoning—highlighting clear gaps relative to human experts.

## Strengths

- **Fills a genuine gap in scientific AI evaluation:** Chemical tables are distinct from general-domain tables in their combination of symbolic notation, embedded molecular graphics, and domain-specific conventions. The paper convincingly argues that existing benchmarks (ChartQA, SciTab, MMTab) lack this specialized multimodal complexity.

- **Comprehensive annotation schema:** Beyond standard table structure, the dataset includes SMILES mappings for molecular structures, text formatting annotations (bold, italics, color), and domain-specific question types (yield/conditions, benzene ring counting, function-based QA). The inter-annotator agreement (0.96 IoU for cell boundaries, 0.94 for cell content) demonstrates annotation quality.

- **Multi-faceted evaluation design:** The benchmark includes table recognition, value/position retrieval, molecular recognition (SMILES extraction), descriptive QA, and multi-hop reasoning questions. The inclusion of unanswerable questions (Table 5) tests model calibration and self-awareness.

- **Diagnostically rich failure analysis:** The case studies in Appendix M identify concrete failure modes—visual-style hallucination (Figure 16), footnote misbinding (Figure 17), and schema-field misalignment (Figure 18)—which provide actionable insights for model developers.

- **Novel molecular recognition task:** Using Tanimoto coefficient on molecular fingerprints to evaluate SMILES extraction from embedded diagrams is a meaningful contribution that goes beyond standard text-based table benchmarks.

## Weaknesses

- **Inconsistent dataset size reporting:** The abstract states "over 1,300 tables," Table 2 reports 1,382 images, and Appendix D.3 claims "1,500 fully annotated chemical table images." These discrepancies undermine confidence in the reported scale and must be reconciled with explicit clarification of the final dataset size.

- **Missing data partitioning documentation:** The paper does not specify whether the dataset is train-only, test-only, or split into train/validation/test partitions. For a benchmark intended to evaluate models, clarity on contamination prevention and reproducible evaluation protocols is essential.

- **LLM-as-judge introduces potential circularity:** The paper uses GPT-4.1-nano to classify answers as correct/incorrect. While human validation shows 96.8% agreement, this agreement metric measures overlap on *correct* classifications only (Equation 1), not full confusion matrix agreement. Additionally, for GPT-5 and GPT-4.1 outputs, the judge model is from the same family, potentially introducing systematic bias.

- **Difficulty filtering may penalize certain model families:** Section 3.3.4 filters out questions that Qwen-2.5-7B answers correctly on the first pass. This calibration against a specific model risks biasing the benchmark against models with similar capabilities or training data, and the fraction of questions discarded is not reported.

- **Human baseline is partial:** Human performance is reported only for tasks requiring chemical expertise or complex reasoning. For subtasks like descriptive element-level QA, no human baseline exists, making the human–model gap analysis incomplete.

- **No comparison to specialized table recognition models:** The recognition task evaluates general MLLMs but excludes established table structure recognition methods (e.g., TableFormer, MASTER). Including at least one specialized baseline would contextualize how far general-purpose models are from domain-specific SOTA.

- **Domain-specific model evaluation buried in appendix:** ChemVLM and Table-LLaVA comparisons appear only in Appendix K. For a benchmark emphasizing chemical domain specificity, these models should be in the main evaluation tables.

- **No statistical significance testing:** Tables 3, 4, and 6 report point estimates without confidence intervals or significance tests. Given modest sample sizes for some subtask categories, several claimed differences may not be statistically meaningful.

## Nice-to-Haves

- **Fine-tuning experiments:** Demonstrating that models improve after training on ChemTable would strengthen the claim that the benchmark is useful for advancing scientific understanding capabilities.

- **Per-task human agreement breakdown:** The overall 96.8% agreement between GPT-4.1-nano and human judges should be disaggregated by question type to verify reliability on the most challenging reasoning tasks.

- **Error attribution for molecular recognition:** When models fail on molecular structures, the paper does not separate OCR failures from chemical knowledge gaps from visual parsing errors—this would guide targeted improvements.

- **Correlation between table complexity and performance:** Analyzing how accuracy varies with table size, number of embedded images, or molecular complexity would validate that the benchmark captures meaningful difficulty gradients.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Appendix S is absent"**: The harsh critic claimed the copyright appendix was missing, but Appendix S is clearly present in the paper content (page 34), detailing CC-BY-SA 4.0 licensing and source attribution.

- **"Section 3.3 numbering is incoherent"**: The numbering in the paper (3.3.1 Task Definition, 3.3.2 Descriptive Questions, 3.3.3 Reasoning Questions, 3.3.4 Data Filtering) follows a logical order. Any perceived incoherence is likely a rendering artifact or reviewer confusion.

- **"Data contamination is a fundamental concern"**: While contamination is a reasonable consideration, claiming it as a "fundamental concern" without evidence that models have memorized these specific tables is speculative. The paper would benefit from a brief discussion, but this is not a fatal flaw.

- **"Demand for train/test splits"**: Many benchmarks are evaluation-only (test sets), and the paper appears to position ChemTable this way. While documentation of the intended use is needed, demanding training partitions is scope creep.

## Novel Insights

The forward–inverse reasoning asymmetry (Appendix A.2) is a genuinely novel finding: models perform ~5% better on forward prediction (conditions → yield) than inverse inference (yield → conditions). This suggests that scientific reasoning is not symmetric—models learn the natural direction of experimental workflows but struggle with abductive reasoning. This directional gap has implications for how AI assistants should be designed for scientific discovery: they may be better at predicting experimental outcomes than at reverse-engineering conditions from results.

## Suggestions

- **Reconcile dataset size across all mentions:** Audit every occurrence of table/image counts in the paper and ensure consistency. A single table in the methods section should definitively state the final dataset composition.

- **Add confidence intervals to main results:** Report bootstrap confidence intervals or standard errors for accuracy metrics, especially on subtask categories with fewer than 100 questions.

- **Move domain-specific model comparisons to main results:** Promote ChemVLM and Table-LLaVA from Appendix K to Table 4, even if only in summary form, to establish domain-specific baselines.

- **Clarify evaluation protocol:** Explicitly state whether ChemTable is intended as an evaluation-only benchmark or includes training partitions. If the latter, describe the split methodology and contamination safeguards.

- **Add Cohen's kappa for judge agreement:** Replace the current overlap metric with a proper inter-rater agreement measure that accounts for both correct and incorrect classifications.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0]
Average score: 4.5
Binary outcome: Reject
