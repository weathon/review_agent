# MR$^2$-Bench: Going Beyond Matching to Reasoning in Multimodal Retrieval

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 4, 8

## Abstract
Multimodal retrieval is becoming a crucial component of modern AI applications, yet its evaluation lags behind the demands of more realistic and challenging scenarios. Existing benchmarks primarily probe surface-level semantic correspondence (e.g., object–text matching) while failing to assess the deeper reasoning required to capture complex relationships between visual and textual information. To address this gap, we introduce MR$^{2}$-Bench, a reasoning-intensive benchmark for multimodal retrieval. MR$^2$-Bench presents the following critical values: 1) all tasks are reasoning-driven, going beyond shallow matching to effectively assess models’ capacity for logical, spatial, and causal inference; 2) it features diverse multimodal data, such as natural images, diagrams, and visual puzzles, enabling comprehensive evaluation across content types; 3) it supports complex queries and documents containing multiple images and covers diverse retrieval scenarios, more accurately reflecting real-world applications. Our benchmark contains 1,309 curated queries, derived either from manual collection and annotation or from selective consolidation of public datasets. Despite achieving strong results on existing benchmarks, current state-of-the-art models still struggle on MR$^{2}$-Bench: for example, the leading Seed1.6-Embedding model attains a Recall@1 of 77.78 on MMEB, but only 9.91 on MR$^{2}$-Bench. This substantial performance gap highlights both the increased challenge posed by our benchmark and the pressing need for further advances in reasoning-intensive multimodal retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new benchmark, MR2-Bench, for "reasoning-intensive multimodal retrieval." The benchmark covers three meta-tasks (multimodal knowledge retrieval, visual illustration search, and visual relation reasoning) and twelve sub-tasks, comprising a total of 1,309 queries. It emphasizes flexible interleaved text-image formats, single/multi-image inputs, multi-domain coverage, and evaluation criteria requiring logical, spatial, and causal reasoning (Table 1 and Table 2). The authors systematically evaluate eleven text and multimodal embedding retrieval models using nDCG@10 as the primary metric, concluding that current SOTA methods fail significantly on MR2-Bench. For example, Seed-1.6-Embedding achieves Recall@1 = 77.78 on MMEB but only 9.91 on MR2-Bench, with an average nDCG@10 of 30.68. The authors further present two enhancement techniques: retrieval-oriented Query Rewriting (using GPT-5 to generate step-by-step reasoning and reformulate queries) and Reranking (using explicit multimodal re-rankers that “reason before scoring”), both yielding substantial improvements in average performance (Table 4, Figure 2).

### Strengths
1. Clearly defines “reasoning” as the prerequisite for relevance judgment, distinguishing MR2-Bench from prior benchmarks focused on shallow matching. It introduces data types rarely included in retrieval evaluation before, such as visual puzzles, mathematical visual proofs, and economic charts, making the task design both novel and necessary.

2. The benchmark allows both queries and candidates to include multiple images or interleaved text-image formats, reflecting real-world document retrieval structures rarely seen in existing benchmarks.

3. Task and data statistics are clearly presented. The three meta-tasks and twelve sub-tasks specify query and corpus scales (Table 2), covering diverse scenarios from natural science to visual reasoning.

4. The result analysis captures the essence of “why it is difficult”: multi-image relations, spatial reasoning, and abstract figures represent systematic weaknesses of current methods.

5. The main findings are straightforward: SOTA models significantly degrade on MR2-Bench; caption enhancement notably improves text retrievers; and introducing “reason-before-ranking” strategies is effective.

6. The quantified performance gap (e.g., Seed-1.6 Recall@1 comparison between MMEB and MR2-Bench) strongly supports the claim that reasoning-based retrieval is more challenging and valuable for research than prior evaluations.

### Weaknesses
1. The positive document selection and hard negative sampling in the multimodal knowledge retrieval subset rely on GPT-5 for initial filtering, followed by expert review. Some hard negatives are GPT-5-generated paragraphs that are “topic-relevant but unhelpful.” Although human validation is included, this pipeline introduces unquantified model bias and potential data contamination risks (e.g., “tool consistency bias” with GPT-5 also used in the reranking/reformulation stage). The paper should provide stricter inter-annotator agreement metrics and cross-model sensitivity analyses.

2. No statistics are provided on annotation consistency (e.g., IAA/Kappa) or multi-round review ratios, making it difficult to assess the stability and transferability of labeling quality.

3. The text retrievers replace all images with captions generated by the same model (Qwen2.5-VL-7B) using fixed prompts, which may induce systematic bias favoring certain methods (e.g., those aligned with the captioning style). The paper does not report sensitivity to different caption models or prompts, nor does it compare human-written summaries versus automatic captions.

4. The total of 1,309 queries is relatively small compared with general benchmarks such as M-BEIR (190,000), and sub-task corpus sizes vary widely (e.g., Mathematics: 944 vs. Economics: 7,572), potentially introducing uneven weighting in averaged metrics. Weighted indicators or significance tests across tasks are recommended.

5. Although the benchmark emphasizes “free-form documents and interleaved multi-image inputs,” evaluation remains limited to offline retrieval metrics (nDCG/Recall). Adding out-of-distribution queries, cross-domain transfer (e.g., new chart sources or species), and analysis linking retrieval quality with end-to-end RAG answer accuracy would strengthen claims of real-world applicability.

6. External data sources such as economic charts (World Bank reports), natural images (INQUIRE-Rerank), and datasets like RAVEN/CSS/VASR are mentioned in the main text/appendix, but there is no consolidated ethical section detailing licensing, re-distribution permissions, or potential copyright/portrait issues. The paper should include a dedicated ethics subsection enumerating data sources, licenses, usage boundaries, and de-identification procedures.

### Questions
Questions correspond to the issues listed in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents MR²-Bench, a new benchmark designed to evaluate reasoning-intensive multimodal retrieval. Unlike prior benchmarks that mainly assess surface-level alignment (e.g., text–image matching), MR²-Bench focuses on logical, spatial, and causal reasoning across diverse modalities such as natural images, diagrams, and visual puzzles. It also supports complex queries and multi-image documents, providing a more realistic retrieval setting. Experiments show that even strong models like Seed1.6-Embedding perform poorly, revealing substantial challenges and opportunities for future multimodal retrieval research.

### Strengths
- The paper contributes a useful benchmark resource for studying reasoning-intensive multimodal retrieval, addressing a gap in how current datasets mostly test pattern matching rather than reasoning ability.

- It demonstrates the practical effects of reasoning-oriented strategies such as query rewriting and reranking, offering the community simple baselines that are informative before scaling to larger models.

- The authors conduct a comprehensive evaluation, covering multiple unimodal and multimodal retrievers (nearly 300 results across 12 tasks), which helps clarify the current limitations of existing methods.

### Weaknesses
- The paper mainly focuses on assembling datasets, defining tasks, and benchmarking existing models. It does not introduce any new architectures, algorithms, or training approaches, which limits its contribution beyond dataset curation.

- The experiments rely solely on nDCG@10 as the numeric metric, yet the paper does not explain why this particular choice is sufficient or why other standard retrieval metrics (e.g., Recall@K, MRR) are omitted.

- Although the benchmark emphasizes “reasoning-intensive” retrieval, the paper does not clearly define how reasoning difficulty is measured or annotated. The boundaries between logical, spatial, and causal reasoning tasks are sometimes ambiguous.

- I do not find any public release of the MR²-Bench dataset or code repository. For a benchmark paper, dataset accessibility is essential for community adoption and reproducibility.

If I miss some information, please correct me.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work purpose a new multimodal retrieval benchmark that requires complex reasoning. The benchmark contains comprehensive and various sub-tasks. The author evaluate many baseline setups for the newly purposed benchmark, ranging from different established text-only retriever to multimodal retriever. The author evaluate naive retrieval setup, query-rewritting, reranking setup as baselines.  The author finds out that the purposed benchmark remains challenging for current systems.

### Strengths
- Propose a valuable new benchmark centered on difficult and challenging multimodal retrieval tasks that requires complicated reasoning, a valuable contribution to the community given that the author would release all reproducible artifacts upon acceptance.
- Very comprehensive baseline experiments setup provided, ranging from various retrievers to different reranker setups. 
- The presentation is good and easy to follow; appendix has very comprehensive details that most author expect to have.

### Weaknesses
- The “puzzle” subset appears almost impossible to solve with current methods. Could the authors provide some intuition on how one might approach this task? Furthermore, does it truly benefit from a retrieval or RAG (Retrieval-Augmented Generation) framework for LLMs, or does it mainly test pure visual reasoning instead? While I appreciate the novelty of treating such visual-relation problems as retrieval tasks, they may be closer to classification-type reasoning challenges. For example, in Figure 1(h), if hard negative examples are included, the retriever must correctly identify the target among visually similar candidates. More insights on this design motivation would strengthen the paper. A similar clarification would also be useful for the mathematics subset.
- Minor: 
	- In **Table 2**, it would be helpful to include an additional column showing the **total number of queries and corpus size combined**.
	- The authors could also report **recall metrics** for reranking performance in the appendix to provide a more complete evaluation.
	- In the appendix, it would be helpful to **summarize the data sources** of each sub-task—e.g., whether they were adapted from existing datasets or newly collected.

### Questions
When performing retrieval for each task in Table 3, do you construct a combined retrieval database across all sub-tasks, or maintain separate databases for each sub-task?

### Soundness
3

### Presentation
3

### Contribution
3
