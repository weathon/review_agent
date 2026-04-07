=== CALIBRATION EXAMPLE 43 ===

# Harsh Critic Review
Now I have enough to write a thorough review. Let me compose it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is accurate. The abstract makes several strong claims—"urgent need," "significant challenges," "realistic testing platform"—that are well-supported by the subsequent results. One tension worth noting: the abstract states "closed-source models lead overall but still fall short of human-level performance," yet Table 4 shows that several proprietary models (GPT-5, Gemini-Pro) are highly competitive with or exceed the human baseline on multiple numerical and logical subtasks (e.g., Value Comparison, Find Min/Max, Trend Analysis). The abstract framing that humans uniformly lead is too sweeping.

---

### Introduction & Motivation

The motivation is well-articulated and genuine. Chemical tables are underserved by existing multimodal benchmarks, and the authors correctly identify why: they combine symbolic notation, embedded molecular structure images, domain-specific abbreviations, and implicit conventions in a way that general-purpose benchmarks do not capture.

The contributions listed are clearly stated. However, one claim deserves scrutiny: "Closed-source models lead overall but still fall short of human-level performance" is repeated from the abstract but is challenged by the actual results in Table 4, where humans are only evaluated on a subset of tasks and are beaten by top models on several (e.g., Numerical Statistics subtasks where humans score 47–55 vs. GPT-5 at 86–92). This appears to be a case where the conclusion was written before carefully examining the data for all sub-tasks.

---

### Dataset Construction (Section 3.1)

**Journal selection and scope:** Restricting to five elite journals introduces a systematic bias toward high-performing, well-formatted tables. Tables from less prestigious or interdisciplinary venues might have noisier layouts, unusual formats, or different conventions—and are absent from the benchmark. The authors do not acknowledge this coverage gap or discuss how it might affect generalizability.

**Table categorization:** More than 50% of tables come from condition optimization and substrate screening categories. This heavy skew toward organic synthesis workflows limits the benchmark's claim to represent "chemical tables" broadly. Inorganic chemistry, analytical chemistry, materials science, and biochemistry tables have very different structures and are largely absent. The authors should either qualify the scope more precisely (e.g., "organic reaction tables") or demonstrate coverage of diverse chemistry subfields.

**QA question generation pipeline:** The benchmark uses a four-stage annotation pipeline (rule-based, LLM-assisted, manual, function-based), which is reasonable in design. However, several sub-issues arise:

1. *Difficulty filtering (Section 3.3.4)*: Questions answered correctly by Qwen-2.5-7B on the first attempt "were randomly discarded." This filtering strategy is underspecified—what proportion was discarded? How does random discarding (rather than principled selection) affect the distribution of question types and difficulty levels? This could selectively remove questions that are easier for all models, but if Qwen-7B happened to excel in certain types, those categories become underrepresented.

2. *Diversity rewriting*: GPT-4.1 was used to rewrite repetitively phrased questions. This risks introducing linguistic artifacts characteristic of GPT outputs, which could systematically advantage or disadvantage certain models.

3. *Function-Based QA validation (Section F.4)*: Questions are accepted if GPT-4.1 and Claude-3.7-Sonnet agree across three answering rounds. This validation criterion is **circular**: it accepts questions that existing strong MLLMs can answer consistently—which means it filters *out* questions where these models fail, potentially removing the hardest and most discriminating examples from this category. The resulting Function-Based QA subset may be easier than intended for the models that were also used to validate it.

**Missing train/test split:** The paper does not specify whether ChemTable is intended purely as a held-out evaluation benchmark or whether a training split is provided. This is a critical detail for reproducibility and for fair comparison of fine-tuned vs. zero-shot approaches.

---

### Table Recognition (Section 3 & 4)

**Metric design:** The use of TEDS with Tanimoto coefficient substitution for molecular-formula cells is a thoughtful choice, but the specification is incomplete. The Tanimoto coefficient is a real-valued similarity, while standard TEDS uses normalized edit distance—the integration of these two similarity functions within the tree-edit-distance framework is not formally defined. How exactly does the Tanimoto score enter the TEDS computation? At what fingerprint type and bit depth? Is there a threshold above which a match is considered "correct"? Without this, other groups cannot reproduce this metric.

**Task definition inconsistency (Sections 3.2.1–3.2.2):** Section 3.2.1 defines table recognition as "format mapping from images to sequences" (i.e., full table reconstruction). But Section 3.2.2, headed "Evaluation Protocols," actually introduces three *sub-tasks*: Value Retrieval, Position Retrieval, and Molecular Recognition. These are retrieval tasks, not the same as sequence-level reconstruction. The Value and Position Retrieval sub-tasks are more naturally "understanding" tasks in spirit. The conflation blurs what "table recognition" means in this benchmark.

**Value and position retrieval results (Table 3):** The retrieval accuracy numbers (~29–54%) are strikingly low. The paper diagnoses this as a model limitation, but an alternative explanation is that the row/column indexing scheme is ambiguous (as illustrated by the Gemini case study in Figure 15, where caption lines are counted as rows). The authors should report how many errors are attributable to indexing convention mismatches vs. genuine perception failures.

**No end-to-end table recognition baseline:** The paper evaluates MLLMs in a zero-shot generation paradigm, but no comparison is made to specialized table recognition systems (e.g., TableFormer, OTSL, or other TEDS-optimized models) on the chemical data. Including such baselines would contextualize how far behind MLLMs are compared to dedicated systems.

---

### Table Understanding: Experiments & Results (Sections 5.1–5.2)

**Evaluation judge circularity:** The primary judge for open-ended QA is GPT-4.1-nano. At the same time, GPT-4.1 is used to generate a large fraction of the QA pairs (Section F.2) and is the second-strongest tested model (Table 4). GPT models evaluating GPT-generated ground truth, when GPT is also a system under test, creates a systematic advantage for the GPT family. The 96.8% human–judge agreement (Section G) is reported as a single percentage overlap rather than as a standard inter-rater reliability statistic (Cohen's kappa or Fleiss' kappa), which is harder to interpret; agreement is easier to achieve when most answers are obviously right or wrong.

**Human baseline scope:** The human baseline is only applied to "tasks that require chemical expertise or complex reasoning" (Section L). Descriptive element-level tasks do not get human annotations. This is a pragmatic decision but makes the human baseline column in Table 4 inconsistently defined across rows—some "–" entries are absent not because humans failed but because they were never evaluated. This should be made explicit, and descriptive task human performance should ideally be included.

**Human baseline scale:** Only five annotators, three per question, averaged. For tasks where expert disagreement is possible (e.g., enantioselectivity interpretations), three-person majority vote may not reliably represent "expert human performance."

**Data contamination risk:** The benchmark uses papers from 2015–2024, and the tested models include GPT-5, Claude-4.5-Sonnet, and Gemini-2.5-Pro, all of which have training cutoffs overlapping this date range. The high performance of top models on certain QA categories (e.g., 90%+ on Yield and Conditions) could partly reflect memorization of specific papers rather than genuine reasoning. The authors do not discuss contamination risk or provide any analysis (e.g., comparing model performance on post-2023 vs. pre-2023 papers as a proxy).

**Missing modality breakdown for recognition:** Figure 5 (modality ablation for InternVL3-78B) is insightful, but is only shown for one model on a subset of tasks. Extending this to all models would be a much stronger ablation.

**Statistical significance:** No confidence intervals, standard deviations, or significance tests are reported for any results. For a benchmark paper submitted to ICLR, the absence of significance analysis is a real weakness. This is especially important given the small number of questions in some sub-categories (e.g., Benzene Rings Count—count not specified, but likely small) where performance gaps may not be reliable.

---

### Qualitative Analysis (Appendix M)

The four case studies (fine-grained recognition, visual-style grounding, domain-specific notation, multi-hop reasoning) are well-chosen and genuinely illuminating. The Gemini caption-counting error, the GPT-5 yellow-column hallucination, the Claude footnote mislink, and the Claude multi-hop column-swap error each reveal a distinct failure mode. These serve as strong motivation for the benchmark's design choices and should be promoted to the main paper.

---

### Writing & Clarity

The main paper has a structural problem: Section 3.3 opens with subsection 3.3.3 (Reasoning Questions) *before* 3.3.1 (Task Definition) and 3.3.2 (Descriptive Questions). This ordering makes the paper genuinely confusing to read—the task is defined after one of its sub-components is already introduced. Additionally, section numbering in the appendix is inconsistent (some sections like "I Dataset Distribution" appear after "J Annotator Information" in text order), suggesting late-stage reordering.

The left-sidebar floating text in Table 1 (describing ChemTable) is a stylistically awkward choice that interrupts the flow of Section 3.1.

---

### Limitations & Broader Impact

The limitations section is absent from the main body—there is no dedicated limitations discussion. The brief ethics statement confirms human annotation was performed under fair conditions and that copyright issues are addressed in "Appendix S," which does not appear in the visible text. For a benchmark that scrapes copyrighted journal content, this is a significant omission.

Additional failure modes not discussed:
- **Benchmark saturation risk:** On several sub-tasks (Value Comparison, Find Min/Max, Trend Analysis), GPT-5 and Gemini-Pro already score 86–94, approaching the ceiling. The benchmark may need more headroom before it becomes saturated by frontier models.
- **Language scope:** All QA is in English. Chemistry literature includes significant content in German, Chinese, and Japanese (some of the journals indexed cover non-English content). The benchmark implicitly evaluates English-only understanding.
- **Scope inflation:** The claim that "9,000 QA instances" constitute a diverse benchmark is undermined by the fact that 7,344 of these are "descriptive questions" derived from only 1,512 unique question templates (Table 2). Many descriptive questions are likely near-duplicate surface variations over the same table cells.

---

### Overall Assessment

ChemTable addresses a real and underexplored gap: no existing benchmark adequately targets the multimodal, domain-specific challenges of chemical tables in peer-reviewed literature. The annotation infrastructure is carefully designed, inter-annotator agreement metrics are strong, and the qualitative case studies are genuinely insightful. The benchmark succeeds in demonstrating that current MLLMs have meaningful and diagnosable weaknesses on chemical tables, particularly in molecular recognition, fine-grained structural alignment, visual-style grounding, and multi-hop chemical reasoning.

However, several issues need to be addressed before acceptance at ICLR. First, the circular validation of Function-Based QA using the same models that are then tested on it is a methodological flaw that could bias difficulty estimates. Second, the lack of any data contamination analysis for models trained on post-2015 scientific literature is a critical oversight given the dataset's temporal overlap with model training corpora. Third, the Tanimoto-modified TEDS metric is incompletely specified, undermining reproducibility. Fourth, the claim of broad chemical coverage is overstated—the benchmark is heavily skewed toward organic synthesis reaction tables—and the abstract's claim about human superiority is inconsistent with the actual results in Table 4. Fifth, the absence of statistical significance testing throughout is below ICLR standards for empirical claims. Addressing these issues would substantially strengthen a contribution that, despite these concerns, represents a useful step toward rigorous evaluation of MLLMs in scientific chemistry contexts.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces **ChemTable**, a large-scale benchmark comprising 1,300+ real-world chemical tables from peer-reviewed literature, designed to evaluate multimodal large language models (MLLMs) on table recognition and understanding tasks that integrate symbolic notation and molecular structures. The dataset supports two core tasks—recognition and understanding—containing over 9,000 expert-annotated QA instances, including descriptive queries and complex reasoning problems. Evaluation reveals that while current MLLMs perform well on general layout parsing, they struggle significantly with domain-specific elements like molecular structure recognition and fine-grained visual-semantic alignment compared to human experts.

### Strengths
1.  **Domain-Specific Rigor and Diversity:** The dataset is uniquely curated from high-impact chemistry journals (e.g., JACS, ACS Catalysis) over a decade, capturing the multimodal complexity of scientific tables that general benchmarks miss. It specifically addresses the challenge of embedding molecular structures (e.g., SMILES conversion) and chemical conventions within tabular layouts, which is a critical gap left by existing benchmarks like SciTab or PubTabNet.
2.  **Comprehensive Evaluation Framework:** The paper establishes a robust evaluation suite including a human performance baseline, comparison of open-source and closed-source models, and analysis of unanswerable questions (Table 5). The inclusion of modality comparison (Text vs. VQA vs. Hybrid QA) provides valuable insights into optimal input strategies for scientific understanding.
3.  **Transparency and Reproducibility:** The authors provide extensive documentation in the appendices, detailing annotation protocols (Section D), prompt templates (Section Q), and release plans for code and data. The evaluation protocol uses consistent decoding configurations (Appendix L) and includes an automated verification mechanism with high human-agreement rates (96.8%) for QA correctness.

### Weaknesses
1.  **Dependence on Proprietary Models for Evaluation:** The evaluation of answer correctness for open-ended QA relies on GPT-4.1-nano (Section 5.1). While verified against humans, this introduces a closed-source dependency that may conflate the model's specific biases with ground truth, limiting independent reproducibility of the evaluation results by researchers without API access.
2.  **Mixed Annotation Pipeline Quality:** While complex reasoning questions are manually annotated, the paper acknowledges that simple reasoning questions were generated using GPT-4.1 (Section 3.1.2). This semi-automated approach risks introducing LLM-specific biases or lower quality in simpler reasoning samples compared to fully human-driven datasets.
3.  **Benchmarking Reliance on Latest Proprietary Weights:** The reported performance gap is heavily influenced by the inclusion of future-dated proprietary models (e.g., GPT-5, Gemini 2.5, Claude 4.5 in Section 5.1). While reflecting state-of-the-art capabilities, this focus may overshadow the analysis of open-source models' potential for scientific reasoning and could exacerbate the "black-box" nature of benchmark progress, as open researchers cannot easily reproduce the full baseline set.

### Novelty & Significance
**Novelty:** The paper makes a significant contribution by bridging the gap between general table understanding and domain-specific chemical knowledge. Existing benchmarks focus on general layouts (ChartQA) or text-heavy scientific tables (SciTab), but ChemTable is the first to systematically integrate molecular graphics *inside* tables as a core evaluation factor, adding a unique dimension of chemical reasoning.

**Clarity:** The structure is logical, moving from problem motivation to dataset construction, tasks, experiments, and analysis. The technical details of annotation and evaluation are clearly defined, though reliance on future model versioning (e.g., GPT-5) requires context for a current audience.

**Reproducibility:** High. The release of the dataset, annotation tools, and evaluation scripts in a public link (Section 3, Link [1]) supports replication. The explicit decoding parameters and prompt templates aid in reproducing the specific experiments.

**Significance:** This benchmark addresses a high-impact niche: AI-driven scientific discovery. By providing realistic challenges (e.g., extracting SMILES from images, interpreting footnotes), it sets a new standard for evaluating the limits of MLLMs in specialized scientific reasoning, likely to influence future model development for scientific research tools.

### Suggestions for Improvement
1.  **Diversify Evaluation Metrics:** To enhance reproducibility and reduce dependency on closed-source evaluators, consider introducing domain-specific metrics (e.g., chemical validity checking via RDKit for SMILES extraction) alongside the GPT-based correctness metric. This would allow open-source models to be evaluated more transparently.
2.  **Address Bias in Generated Questions:** Provide a detailed analysis or ablation on the quality difference between purely human-annotated reasoning questions versus GPT-assisted ones. Demonstrating that the LLM-generated questions still capture distinct reasoning patterns would strengthen the dataset's reliability.
3.  **Expand Open-Source Baseline Analysis:** While proprietary models show high performance, the paper should offer more actionable insights into why specific open-source architectures fail (e.g., visual tokenization limits for SMILES). Adding a specific failure mode analysis section for open-weight models would make the paper more valuable to the broader open-source community.
4.  **Clarify Future Model Versions:** Given the presence of model versions like GPT-5 and Gemini 2.5, clarify whether these are standard assumptions for the submission context or indicate a simulation. If this is a current submission, ensure these names are accurate or discuss the limitations of relying on hypothetical future models in the discussion.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Contamination-Controlled Evaluation:** Evaluate models completely disjoint from the data generation pipeline (e.g., non-OpenAI families) because using GPT-4.1 for filtering (Sec 3.3.4) and then benchmarking GPT-4.1 invalidates the performance ceiling.
2. **Modular Pipeline Baseline:** Compare end-to-end MLLMs against a composite baseline (Oracle Table Structure + DECIMER for molecules + LLM for reasoning) to isolate whether failures stem from multimodal integration or lack of domain tools.
3. **Human Evaluation Scale:** Clarify the exact sample size for human baselines per task type, as Section 5.1 contradicts Table 4 regarding which tasks humans annotated (e.g., Table Dimensions).
4. **Few-Shot Prompting Ablation:** The main results rely on zero-shot prompts (Appendix Q); ICLR standards require testing if few-shot examples close the performance gap before claiming inherent model limitations.

### Deeper Analysis Needed (top 3-5 only)
1. **Error Propagation Analysis:** Quantify how much of the QA failure is due to upstream table recognition errors (OCR/Structure) versus downstream reasoning errors, as Table 3 shows poor retrieval but QA results conflate the two.
2. **Domain Knowledge Dependency:** Verify if "reasoning" questions actually require chemical knowledge or just generic arithmetic/logic by testing non-expert humans on legible text versions of the tables.
3. **Metric Validity:** Analyze the correlation between TEDS scores and downstream QA accuracy to ensure structure recognition metrics actually predict task success.
4. **Model Family Bias:** Investigate performance variance between models used for data generation (GPT-4.1 family) versus unrelated architectures to rule out training data leakage.

### Visualizations & Case Studies
1. **Step-wise Failure Visualization:** Provide examples showing exactly where the process breaks (e.g., correct OCR but wrong cell alignment vs. correct alignment but wrong chemical interpretation).
2. **Molecular Structure Errors:** Visualize specific cases where SMILES generation failed due to visual ambiguity versus model hallucination to distinguish perception from knowledge gaps.
3. **Human vs. Model Reasoning Trace:** Contrast the reasoning chain of a human expert versus the model's CoT on complex multi-hop questions to expose logical vs. factual errors.

### Obvious Next Steps
1. **Create a Contamination-Free Test Set:** Include a subset of data generated and filtered exclusively by humans to ensure future benchmarking is unbiased.
2. **Define Tool-Use Protocols:** Establish whether external tools (like chemical decoders) are permitted in the benchmark definition, as current results punish models for not having specialized plugins.
3. **Standardize Human Baseline:** Conduct a larger-scale human evaluation with clear time-tracking to establish a realistic "expert ceiling" rather than a small-sample estimate.

# Final Consolidated Review
## Summary

ChemTable introduces a benchmark of 1,300+ chemical tables from peer-reviewed literature to evaluate multimodal large language models on table recognition and understanding tasks. The benchmark supports two core tasks: structure/content extraction and question answering across descriptive and reasoning categories, with over 9,000 annotated QA instances that uniquely integrate molecular structures and domain-specific notation within tabular layouts.

## Strengths

- **Novel domain-specific focus**: This is the first benchmark to systematically target chemical tables that combine symbolic notation, embedded molecular structure images, and domain-specific conventions—a genuinely underserved niche between general table understanding and scientific figure analysis. The focus on molecular graphics within tables (SMILES extraction from diagrams) is particularly distinctive.

- **Comprehensive evaluation framework**: The paper evaluates 7-10 MLLMs across both recognition and understanding tasks, includes human baseline comparisons, and provides ablation studies on input modality (Text QA vs. VQA vs. Hybrid QA). The comparison of how models handle unanswerable questions (Table 5) adds practical insight beyond raw accuracy.

- **Strong qualitative case studies**: Appendix M provides well-chosen failure mode analyses—visual-style grounding failures, footnote misinterpretation in stereochemical contexts, multi-hop reasoning breakdowns—that diagnose real limitations rather than just cataloging errors.

- **Substantial annotation infrastructure**: The three-phase annotation pipeline with inter-annotator agreement metrics (IoU 0.96 for cell boundaries, 0.94 for text content) demonstrates commitment to data quality. The 96.8% agreement between human and GPT-4.1-nano evaluation provides empirical support for the automated grading approach.

## Weaknesses

- **Circular validation in Function-Based QA generation**: Section F.4 describes validating questions using GPT-4.1 and Claude-3.7-Sonnet—"if all answers are consistent and correct across rounds, the question is accepted." Since these same models are then evaluated on the benchmark, this filters out questions where capable models struggle, potentially biasing difficulty estimates downward for these model families. This methodological flaw could inflate performance for GPT and Claude variants specifically.

- **No data contamination analysis**: Tables are sourced from 2015-2024 publications, and evaluated models include GPT-4.1, Claude-3.7-Sonnet, and Gemini-2.5-Flash—all with potential training data overlap with this period. The paper provides no analysis of contamination risk (e.g., comparing performance on pre-2020 vs. post-2023 papers as a proxy). This is a critical oversight for a benchmark intended to measure genuine reasoning capability.

- **Incomplete metric specification for molecular recognition**: Table 3's footnote mentions substituting Tanimoto coefficient for normalized edit distance in TEDS for molecular formula cells, but the exact integration is unspecified: fingerprint type, bit depth, and threshold for "correct" matching are not provided. Without this, other researchers cannot reproduce the TEDS* scores.

- **No statistical significance testing**: Across all experiments, no confidence intervals, standard deviations, or significance tests are reported. For a benchmark paper at ICLR, this is below expected standards—the observed gaps between models (e.g., Claude-3.7 vs. Gemini-Pro on reasoning tasks) cannot be statistically validated.

- **Limited chemistry subfield coverage**: Over 50% of tables are condition optimization and substrate screening tables from organic synthesis workflows. Inorganic chemistry, analytical chemistry, materials science, and biochemistry tables are underrepresented relative to the claim of representing "chemical tables" broadly. The abstract and title should more precisely scope the contribution (e.g., "organic reaction tables from high-impact chemistry journals").

- **Absence of train/test split specification**: The paper does not clarify whether ChemTable is intended purely as a held-out evaluation benchmark or whether training/development splits are provided. This ambiguity hampers reproducibility for fine-tuning experiments.

- **Inconsistent human baseline scope**: Human performance is only reported for tasks "that require chemical expertise or complex reasoning" (Appendix L), meaning descriptive tasks have no human ceiling. The "–" entries in Table 4 are ambiguous—readers cannot distinguish "humans weren't evaluated" from "humans would perform poorly."

- **No specialized table recognition baseline comparison**: The paper evaluates MLLMs in zero-shot fashion but does not compare against dedicated table recognition systems (e.g., TableFormer, OTSL, or DECIMER for molecular recognition). Including such baselines would contextualize how far MLLMs lag behind domain-specialized tools on structure extraction.

## Nice-to-Haves

- **Contamination-free subset**: Creating a manually filtered test set validated without LLM involvement would ensure unbiased future benchmarking.

- **Modular pipeline baseline**: A composite baseline (Oracle table structure + DECIMER + LLM reasoning) would isolate whether failures stem from perception or reasoning.

- **Error propagation analysis**: Quantifying how much QA failure is attributable to upstream recognition errors vs. downstream reasoning errors would clarify failure attribution.

- **Non-English table coverage**: Chemistry literature includes significant German, Chinese, and Japanese content; the English-only scope limits real-world applicability assessment.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that humans underperform GPT-5 on numerical tasks**: The harsh critic states that humans score 47-55 on numerical statistics while GPT-5 scores 86-92. This is factually incorrect—Table 4 shows human scores of 100 on Value Comparison, Find Min/Max, and Calculate Sum, and 98.20 on Calculate Average. Humans are at or near ceiling on all reported tasks.

- **Section numbering error claim**: The critic claims Section 3.3 opens with subsection 3.3.3 before 3.3.1/3.3.2, but the paper's actual structure follows correct ordering: 3.3.1 Task Definition → 3.3.2 Descriptive Questions → 3.3.3 Reasoning Questions → 3.3.4 Data Filtering.

- **Benchmark saturation concern**: The critic suggests the benchmark may saturate because some tasks show 86-94 scores. However, key tasks like Molecular Recognition (14-52% for open-source models), Color Description (~50%), and Function-Based QA (~70%) show substantial headroom. The concern overstates saturation risk.

## Novel Insights

The paper reveals a fundamental asymmetry in model behavior: models perform reasonably well on structural table recognition (TEDS-Struct 93-96%) but catastrophically fail on fine-grained retrieval within those same tables (Value Retrieval 17-54%). This suggests that MLLMs capture "gist-level" table structure without precise positional grounding—a finding with implications beyond chemistry. Additionally, the modality ablation (Figure 5) shows Hybrid QA outperforming both pure text and pure image inputs, suggesting that optimal scientific table understanding requires redundant encoding across modalities rather than relying on either alone.

## Suggestions

- Add explicit contamination analysis: Compare model performance on papers before/after a median date (e.g., 2020) or report the overlap between ChemTable sources and known model training corpora.

- Specify the train/validation/test split structure and document whether the benchmark is intended for zero-shot evaluation only or permits fine-tuning.

- Provide the complete TEDS* computation formula, including fingerprint specification for Tanimoto scores.

- Add standard error bars or confidence intervals for all reported metrics, particularly for small-sample subtasks.

- Clarify scope in the title/abstract to reflect that this is primarily organic synthesis reaction tables from elite English-language journals, not a comprehensive representation of all chemical tables.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0]
Average score: 4.5
Binary outcome: Reject
