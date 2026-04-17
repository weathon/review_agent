# MRMR: A Realistic and Expert-Level Multidisciplinary Benchmark for Reasoning-Intensive Multimodal Retrieval

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
We introduce MRMR, the first expert-level multidisciplinary multimodal retrieval benchmark requiring intensive reasoning. MRMR contains 1,435 queries spanning 23 domains, with positive documents carefully verified by human experts. Compared to prior benchmarks, MRMR introduces three key advancements. First, it challenges retrieval systems across diverse areas of expertise, enabling fine-grained model comparison across domains. Second, queries are reasoning-intensive, with images requiring deeper interpretation such as diagnosing microscopic slides. We further introduce Contradiction Retrieval, a novel task requiring models to identify conflicting concepts. Finally, queries and documents are constructed as image–text interleaved sequences. Unlike earlier benchmarks restricted to single images or unimodal documents, MRMR offers a realistic setting with multi-image queries and mixed-modality corpus documents. We conduct an extensive evaluation of 4 categories of multimodal retrieval systems and 14 frontier models on MRMR. The text embedding model Qwen3-Embedding with LLM-generated image captions achieves the highest performance, highlighting substantial room for improving multimodal retrieval models. Although latest multimodal models such as Ops-MM-Embedding perform competitively on expert-domain queries, they fall short on reasoning-intensive tasks. We believe that MRMR paves the way for advancing multimodal retrieval in more realistic and challenging scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a multimodal retrieval benchmark with an emphasis on reasoning-intensive retrieval across 23 domains.
In the proposed benchmark, the authors introduce three distinct types of retrieval tasks -- knowledge retrieval, theorem retrieval, and contradiction retrieval.
The paper evaluates various multimodal retrieval setups (e.g., text-only, two-stream, multimodal, and document-as-image models) covering several frontier models on the MRMR benchmark. The results suggest that existing multimodal retrievers struggle on reasoning-intensive and contradiction retrieval tasks compared to text-only retrievers.
Overall, this paper presents a meaningful contribution to the community, though its technical contribution is limited.

### Strengths
1. Wide domain coverage: MRMR spans 23 domains across six disciplines.
2. Integration of contradiction retrieval: adding the negation/contradiction retrieval testbed is practically useful and can be a good initiative for the community.
3. Comprehensive evaluation: the experiments cover a wide range of retrieval architectures and models.
Human-verified annotation and corpus quality control: All positive documents are validated by human experts, and the corpus is cleaned under human supervision, making MRMR more realistic compared to fully synthetic benchmarks.

### Weaknesses
1. Limited technical contribution beyond task composition: it's unclear about the main technical contribution beyond composing tasks (e.g., reasoning-intensive, image-text interleaving, multidisciplinary) from prior works.
2. Limited dataset size: as a benchmark spanning 23 domains, size of the dataset (1502 queries) is relatively small and therefore may limit statistical robustness.
3. Writing clarity: Certain sections could be written more clearly to improve readability and logical flow, particularly in the methodology and evaluation descriptions.

### Questions
1. In table 2, the third last row (L181), what does it mean that negation task has 4 documents in total (D=4), shouldn't it be 200*4=800?
2. L220, should "solvable" be "unsolvable"?
3. Can you discuss similar works such as $\text{MR}^2$ bench [1] on how MRMR differentiates from them?
4. Can you add prompts in the appendix that you use to synthesize the dataset (e.g., synthesize negation candidates for the contradiction retrieval task)?

[1] Zhou, Junjie, et al. "MR $^ 2$-Bench: Going Beyond Matching to Reasoning in Multimodal Retrieval." arXiv preprint arXiv:2509.26378 (2025).

### Soundness
4

### Presentation
3

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
This paper introduces MRMR (Multidisciplinary Reasoning-intensive Multimodal Retrieval), a novel benchmark designed to evaluate retrieval systems in expert-level, reasoning-intensive, and multimodal tasks. Unlike previous benchmarks that focus on general domains, MRMR spans 23 domains, including medicine, engineering, science, and business, and features 1,502 expert-annotated queries. The tasks are categorized into three types: Knowledge Retrieval, Theorem Retrieval, and Contradiction Retrieval. MRMR challenges retrieval models not only to identify semantically relevant documents but also to perform deep reasoning and logical deduction, such as identifying contradictory concepts or retrieving relevant theorems for complex calculations. The dataset includes interleaved image-text sequences, reflecting the real-world scenario where queries and documents are multimodal. The paper evaluates a variety of state-of-the-art multimodal retrieval systems and finds significant room for improvement, particularly in reasoning-intensive tasks.

### Strengths
* MRMR provides a unique and comprehensive framework to evaluate multimodal retrieval systems on expert-level, reasoning-intensive tasks across multiple disciplines. It addresses the gap in existing benchmarks that fail to capture the complexity of real-world, domain-specific applications, such as medical diagnoses and engineering design.
* The benchmark spans 23 domains, providing a holistic evaluation of retrieval systems across various expert fields. This diversity helps to assess the models' generalizability and specialization across disciplines like medicine, business, engineering, and the sciences.
* By using interleaved image-text sequences, MRMR more accurately reflects real-world retrieval tasks, where queries and documents are rarely confined to a single modality. This design enhances the practical relevance of the benchmark.

### Weaknesses
1. Have the authors considered the scenario where the documents highly relevant to a given query come from multiple different disciplines? For example, the user’s question might require the synthesis of documents from multiple disciplines to provide a comprehensive answer. This kind of situation frequently arises in real-world applications. Did the authors consider this when designing the benchmark?

2. In the overall design of the benchmark, the authors separated documents by discipline for the experiments. However, creating a large database by mixing documents from all disciplines and performing retrieval might present a more challenging setting. This cross-disciplinary retrieval scenario could provide an even more difficult test for the models.

3. The authors discuss retrieval within specific disciplines, which is an interesting topic, but they have not addressed cross-disciplinary retrieval. For instance, if the query includes an image of a sculpture from the art domain and the user wants to retrieve mathematical theorems (from the math domain) related to the geometry of the sculpture, this is a very interesting problem. I look forward to the authors' response and discussion on this question.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Summary:
Authors repurpose MMMU-Pro and use other synthetic methods to generate a multimodal retrival dataset that has expert domain, reasoning intensive and contradition style questions.

### Strengths
Strengths:
- Addresses gaps in the present day benchmarks which mostly rely on surface level/minimal reasoning.
- Clear writing on how each datapoint was created. 
- Also provide hard negatives which can be useful to train models.

### Weaknesses
Weakness:
- Positives of each question are curated independetly. Positives of all documents are put together to construct the corpus. How was it ensured that positives of other queries are not positives of a given query.
  - What were false negatives filtered?
  - Similarly, negatives mined for other queries either by GPT-Search/Human mismatch or by PIN14 can turn out to be potential positves of other queries.
- Smaller dataset compared to sme of the MTEB/MMEB datasets.
- Evaluation only benchmark. It would have been significantly more useful had a training dataset been provided.
- Query expansion results for stronger retriever models missing. It is important to understand if query exapansion with multimodal/text retriver models still falls short on this benchmark.
- Weak qualitative analysis. A category wise qualitative analysis would have been much more helpful.

### Questions
-

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce a dataset focusing on 'Multidisciplinary and Reasoning-intensive Multimodal Retrieval' MRMR with many key contributions with respect to existing multi-modal retrieval dataset. From the perspective of somebody likely to utilize this dataset, there are two practical high-level contributions: (1) MRMR provides validated retrieval results for a subset of the MMMU-Pro question source based on high-quality datasources (PIN-14M + web pages for {Knowledge: Art, Medicine, Science, Humanities} and Bright + web pages for {Theorem: Math, Physics, Engineering, Humanities}; and (2) MRMR develops an interesting reasoning-intensive "contradiction" problem and dataset that is applicable to agentic LLM settings (and currently has low performance with existing systems). In contrasting this work with existing MMIR benchmarks, some key contributions of MRMR (also pointed out by the authors) include: (1) multi-disciplinary expert-level questions (most benchmarks focus on the general knowledge of Wikipedia), (2) reasoning-focus with more complex IR instructions (most benchmarks focus on semantic matching or text IR/QA), and (3) image-text interleaving in queries and data (most MM benchmarks are VQA-focused with a single image). 

For each question type {Knowledge, Theorem, Contradiction}, the authors first describe their method for selecting questions, collecting/validating positive and hard negative results, and adding a background set of reasonable negative examples to construct the corpus. Based on the MRMR dataset, the authors conduct a large scale ablation study over four multi-modal retrieval setups (Text models with image captions, Text and image two-stream models with vector function, Multimodal model with merged images, and Multimodal models with document as image) with 14 state-of-the-art candidate models, demonstrating: (1) text-retrieval based approaches presently have the best performance for these knowledge/reasoning-required tasks, (2) there is ample room for improvement across the board, and (3) there is large variance in performance across different tasks. This study was able to power a reasonable failure-space analysis to point toward research directions that may result in tangible improvements.

### Strengths
Some strengths of this work include:
- As primarily a resources paper, the authors clearly defined what type of questions they targeting, clearly motivated why this would result in a useful benchmark, and demonstrated that a large-scale ablation with the benchmark surfaces interesting (albeit not entirely surprising) conclusions.
- The authors clearly contrast MRMR with related datasets (Table 1) and clearly describe the collection procedure in the various settings (Table 2 and Sections 3.2-3.4) -- which are generally sensible and required a significant amount of work to collect with high-quality.
- MMMU-Pro is a good dataset to derive challenging IR problems with.
- The 'contradiction' dataset is innovative and I expect will inspire additional work in other challenges IR cases that require specific instructions and will lead to more interesting agentic LLM responses.
- A good benchmark paper introduces a non-trivial dataset, makes it easy for others to build on, and provides non-trivial baseline performance results -- which MRMR does.

### Weaknesses
Some weaknesses of this work include:
- The 'contradiction' family of questions is the most interesting from an innovation perspective, but underdeveloped in my opinion. To begin with, even if I am being a bit pedantic, these technically aren't all contradictions (e.g., Figure 1 vehicle design is 'non-compliance'). Also, in looking through the specific cases, 'negation' is a bit contrived (albeit shown to perform poorly in the empirical results) and I can think of other cases and possibly a general hierarchy of "matching a description that isn't there". Basically, it is somewhat preliminary and likely deserves its own study.
- The dataset is high-quality, but relatively small as compared to other related datasets in Table 1. Minimally, I would recommend discussing why a smaller number of high-quality examples is more valuable with respect to detecting improvements.
- Based on the process used, it wasn't clear if there would be any license issues with respect to using this in commercial settings (even if only for academic purposes). I am assuming this isn't an issue, but it would likely affect its impact.
- The analysis in Section 5 is underdeveloped.

### Questions
- It would be useful for you to clarify any licensing issues for academic work in commercial settings.
- In terms of the ablation study, it would also be interesting in the discussion to point out any alignment or contradictions for similar systems on related datasets (e.g., wikiHow).

### Soundness
3

### Presentation
3

### Contribution
3
