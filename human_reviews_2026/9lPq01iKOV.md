# Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Retrieval augmentation has primarily been studied in limited settings, such as factoid question answering; more challenging, reasoning-intensive benchmarks have seen limited success from minimal RAG. In this work, we challenge this prevailing view on a set of established, reasoning-intensive benchmarks: MMLU, MMLU Pro, AGI Eval, GPQA, and MATH. We identify a key missing component in prior work: a usable, web-scale datastore aligned with the breadth of pretraining data. To this end, we introduce CompactDS: a diverse, high-quality, web-scale datastore that achieves high retrieval accuracy and subsecond latency on a single-node deployment, making it suitable for academic use. Its core design combines a compact set of high-quality, diverse data sources with in-memory approximate nearest neighbor (ANN) retrieval and on-disk exact search. Using CompactDS, a minimal RAG pipeline achieves consistent accuracy improvements across all benchmarks and model sizes (8B--70B), with relative gains of 11% on MMLU, 34% on MMLU Pro, 26% on GPQA, and 14% on MATH. No single data source suffices alone, highlighting the importance of diversity of sources (web crawls, curated math, academic papers, textbooks), and a combination of ANN and exact search is shown to be critical for balancing usability and accuracy. Finally, we show that our in-house datastore even outperforms commercial search engines like Google Search. We release CompactDS and our retrieval pipeline as a fully reproducible alternative to commercial search, supporting future research exploring retrieval-based AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CompactDS for RAG in LLMs, a web-scale datastore built from filtered web crawl CommonCrawl plus specific math, academic, and educational texts to better cover content relevant to the evaluation tasks. To efficiently access the datastore, the paper proposes a two-stage retrieval mechanism: on-memory approximate nearest neighbor search using an build index followed by on-disk exact search. The aggressive content filtering enables single-node low-latency RAG use because the full datastore can fit <500GM RAM device. The paper evaluates a simple RAG pipeline on CompactDS on reasoning benchmarks like MATH and GPQA and language understanding (MMLU), and releases the datastore for public use.

### Strengths
1. The paper is generally well-written and clear.
2. Open-sourcing and publicly releasing the datastore will be a plus for the research community when the industry datastores are proprietary and behind paywalls.
3. MassiveDS and other non-filtered datastores can be impractical due to large memory footprint; a compact datastore like CompactDS will help the research community.
4. The paper compares various approaches for retrieval-based generation, not just embedding-based RAG but also agentic systems based on classical search engines. This grounds the paper's contribution in several popular approaches.

### Weaknesses
1. While the paper explicitly describes the data sources in Sec 3.1, the justification for each filtering step is unclear. For example, in line 134... why take the union of two datasets then filter using a classifier with a certain threshold? Is the target to reduce to a maximum number of words in the corpora, or to maximize some validation split in the evaluation setup? This problem persists in the whole of Sec 3.1, which is a key contribution of the paper. Without this justification, design choices for data filtering seem arbitrary and do not help build intuition on what makes a datastore "good" or "bad".
2. The summary in Sec 3.2 is not fully clear, but could improve significantly with a visual diagram spanning the top of this page. Specifically, it's unclear to me which K disk-stored embeddings are retrieved in the 2nd stage of the 2-stage process.
3. It is difficult to be convinced that the 2-stage process is significantly better than the 1-stage (ANN only) process, comparing rows CompactDS-ANN only and CompactDS in Table 1. MMLU doesn't really improve and GPQA varies with different parts. There is improvement in MMLU Pro, AGI Eval, and MATH though. It would be useful if the authors discuss and interpret these results further.
4. Since the paper argues that MassiveDS is too massive, which makes it impractical, it makes sense to evaluate it in Table 1 and report all benchmarks (not just MMLU in Table 2).
5. Significant drops in performance on GPQA and MATH after decontamination (Sec A.1) suggest that CoT is more important for reasoning than retrieving passages. This counters the paper's argument that retrieval could be important for math reasoning tasks.

### Questions
1. Line 68: Why does removing weakest contributing sources degrade performance? One would expect that the most similar documents would still be retrieved, and so the performance would be unaffected. Why is "long tail diversity" important (line 284)?
2. Line 116: While this work does not focus on agentic RAG, do you think it can still work with RAG in an interactive setting? Say when the agent requests a tool call/search string but we retrieve non-exact document hits (e.g., in cases where the search string doesn't exist in the document corpora verbatim)?
3. Line 192: What is the time and space overhead at training time, for indexing with IVFPQ and exact embeddings?
4. Line 197: Need some clarity in understanding the 2-stage process. After q is encoded as E_exact(q), which K disk-stored embeddings are used? Are these the K passages retrieved in the 1st stage of IVFPQ? If that is so, wouldn't the lossy IVFPQ retrieval hurt the second stage? If that's not the case, then finding K passages with exact search in the full index would be extremely time-intensive.
5. Table 1: CompactDS vs No Retrieval on Bio is worst (39.7 vs 47.4) but the relative gain is still positive? This seems incorrectly computed, or a typo.
6. Line 250: good to point to Table 7 with the search-o1 results.
7. Table 4: GPQA scores are consistently worse for CompactDS compared to NoRetrieval for all LLMs except Llama3.1 8B. Is there any insight into this? To me, arguing that CoT is extremely competent at 70B scale to improve NoRetrieval is not a great argument.
8. Line 303 and Table 3: What is the oracle reranking? This needs explanation here.
9. Line 410 and Table 6: If the PDFs cannot fit in context at all, how do we get evaluation results for the rows without LM Reranking?

### Typos and editorial suggestions
1. Line 171: spacing in argTopk...
2. Table 1: it would make sense to only present No Retrieval, CompactDS rows, and any other baselines in this table. Then another table for ablation on single-source datastores.
3. Line 248: demends -> demands?
4. Line 256: point to the appendix with other LLM results.
5. Fig 5: the font is extremely small even on my large monitor. I cannot read the math expressions at all. I can barely read the plots elsewhere, but this plot is really hard even after zooming in.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper revisits the assumption that retrieval-augmented generation (RAG) is ineffective on reasoning-intensive benchmarks. The authors build a large and diverse datastore called COMPACTDS, combining academic, mathematical, and educational sources, and evaluate a simple retrieval-augmented generation setup without complex pipelines. To make it deployable on a single node, COMPACTDS integrates in-memory approximate nearest-neighbor (IVFPQ) retrieval with on-disk exact search to balance recall and latency. The proposed approach consistently improves accuracy across reasoning benchmarks and model scales, showing that even a minimal retrieval system can enhance reasoning performance when supported by a sufficiently rich and diverse datastore. COMPACTDS also matches or outperforms commercial search engines (e.g., Google Search) and complex agentic RAG systems such as Search-o1.

### Strengths
The paper is clearly written and presents strong empirical evidence that retrieval remains useful for reasoning tasks when the datastore is comprehensive and high quality. The motivation is well grounded in an important and practical question. The retrieval design is simple yet effective, showing that careful data construction and indexing choices can yield significant improvements over prior baselines. Experiments are systematic, covering multiple model families and evaluation settings, and the results convincingly support the paper’s main claim. The analysis provides useful insight into how retrieval complements reasoning models. Releasing COMPACTDS as an open, reproducible resource would represent a meaningful contribution to the community.

### Weaknesses
The main limitation of the paper lies in its limited conceptual novelty. The contribution is primarily an engineering improvement through large-scale data construction rather than a new retrieval or reasoning algorithm. Although the results are strong, it remains unclear how well COMPACTDS generalizes beyond the specific reasoning benchmarks used in evaluation. Since the datastore composition is closely aligned with those benchmarks, the observed gains might partly reflect domain overlap rather than general reasoning improvement. It would be helpful to understand how the system performs on more general or non-reasoning tasks, and how flexible the datastore is when retrieval data need to be updated frequently, which is an important advantage of RAG systems in practical use.

### Questions
- How well does COMPACTDS generalize to new reasoning datasets or more general tasks outside the benchmarks used in this paper?
- How sensitive is the final performance to the strength and relative similarity between $E_{Approx}$ and $E_{Exact}$? Does the consistency of their retrieved results significantly influence performance?
- Is there a quantitative analysis of retrieval quality (e.g., precision and recall) to verify that improvements stem from better retrieval?
- Have the authors examined whether retrieval mainly aids factual grounding or intermediate reasoning steps?
- Did the authors attempt a fine-tuned retriever trained directly on reasoning tasks, and how would it compare to COMPACTDS?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors claim that past works have shown little success in leveraging minimal RAG pipelines to improve performance on reasoning-intensive benchmarks because there are no practical, web-scale databases that capture the breadth of the pretraining distribution. At the same time, a RAG database must be compact to minimize decoding latency. The authors introduce CompactDS to meet these constraints. CompactDS is a curated mix of data that is initially drawn from a wide variety of sources and filtered to reduce redundancy and data contamination. In addition, the authors introduce a two stage retrieval pipeline: a course-grained lossy nearest neighbor search to narrow down the candidate set of passages followed by an exact inner product search conducted on-device to minimize retrieval latency.

### Strengths
-	Well written and motivates why current RAG pipelines are insufficient due to lack of scope and high retrieval latency
-	Demonstrate improvement on a variety of benchmarks using RAG
-	Considers problems in actual deployment (ie on device fine grained search after coarse-grained candidate search to minimize latency)
-	Upper-bounding the RAG performance with the oracle baseline makes sense and demonstrates how close this “autonomous” RAG system is to a curated ICL baseline

### Weaknesses
-	The paper introduces CompactDS as a means to improve reasoning performance using RAG but focuses mainly on multiple-choice question style evals that aren’t particularly reasoning intense. The MATH results are particularly interesting, but I would have liked to see the upper bound performance on this task (few shot prompting/ICL). My main concern is that the authors claim RAG helps with reasoning performance, but it seems hard to separate the boost from fact retrieval from the boost in reasoning performance, especially on MCQ style questions with non-reasoning models (it seems like in Table 4, retrieval seems to hurt GPQA performance across the board).
-	Is there a better way to gauge the accuracy of the two-stage retrieval pipeline beyond comparing downstream performance to the oracle? I am concerned because the oracle passages are chosen from the initial pool provided by the lossy retrieval.

### Questions
-	When would you use RAG over hand-curating examples per topic? You could curate examples for each of these baselines that might boost performance further. Is the claim that RAG allows an agent to operate autonomously? Could you illustrate what a system like this would look like? The total dataset size is still hundreds of GB of RAM, and the course grained search probably can’t be done on a small edge device.
-	Are you concerned that filtering heavily for contamination directly conflicts with the goal of RAG (retrieving relevant passages).
-	What would MASSIVEDS performance look like (efficiency and accuracy) when using the two stage retrieval pipeline?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CompactDS, a diverse datastore retrieval. The paper discusses the pipeline on generating this datastore and the method to efficiently serve the system. The paper claims that LLMs can perform better on reasoning-intensive tasks with RAG on this datastore.

### Strengths
1. The paper release a datastore for retrieval.
2. The paper includes a method to efficiently retrieve passages that reduce RAM needed.
3. The experiments are conducted on different model sizes.

### Weaknesses
## Improvements on reasoning-intensive benchmarks
I am not sure if the experiments fully support the claim, especially claiming that the proposed RAG system improves **reasoning extensive** tasks.
1. MMLU and MMLU Pro are more knowledge related. These two benchmarks ask more about model's knowledge. I am not familiar with AGI Eval but from Table 10 it seems like it is MC and no CoTs are allowed when the model answers. While the model still needs to reason internally even without CoT, I think "reasoning-intensive" means those benchmarks where the model need to produce very long CoT (?) Usually those are math and coding benchmarks like AIME, AMC, HMMT, LiveCodeBench for reasoning models.
2. The author includes MATH and GPQA where CoTs are allowed. While there are some improvement, how much improvement comes from the retrieval? For non-reasoning model like Llama-3 8B, perhaps a better baseline is to use a fixed, detailed prompt curated by a strong model like GPT-5 that teaches the model to plan, reflect, backtrack (and maybe some examples).
3. There should be more experiments on reasoning models. The paper includes Qwen3-8B in Table 10 but I am confused about the result (see Questions section below). Reasoning models achieve non-trivial scores in benchmarks like AIME so I wonder how does reasoning model perform on those challenging benchmarks.

## Other clarifications needed.
Please see questions below.

### Questions
1. In Table 4, Qwen3-8B has MATH accuracy 56.3. However, from Qwen3's technical report (https://arxiv.org/abs/2505.09388) the accuracy for MATH-500 is 97.4 (with reasoning; Table 17) or 87.4(without reasoning; Table 18). While MATH-500 is a subset of MATH, I wonder why there is such discrepancy?
2. In Table 10, what does "chat format" mean? Does it mean that the first three evals are selected based on model's perplexities on the choices and the last two are answered in QA?
3. Table 7 is confusing. Why the upper half table has MATH-500 "No Retrieval" accuracy 83.2 but lower half table with MATH-500 "No Retieval" accuracy 91.0?

### Soundness
2

### Presentation
2

### Contribution
2
