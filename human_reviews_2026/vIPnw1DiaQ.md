# $\pi$-CoT: Prolog-Initialized Chain-of-Thought Prompting for Multi-Hop Question-Answering

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Chain-of-Thought (CoT) prompting significantly enhances large language models' (LLMs) problem-solving capabilities, but still struggles with complex multi-hop questions, often falling into circular reasoning patterns or deviating from the logical path entirely. 
This limitation is particularly acute in retrieval-augmented generation (RAG) settings, where obtaining the right context is critical. 
We introduce **P**rolog-**I**nitialized **C**hain-**o**f-**T**hought ($\pi$-CoT), a novel prompting strategy that combines logic programming's structural rigor with language models' flexibility.  $\pi$-CoT reformulates multi-hop questions into Prolog queries decomposed as single-hop sub-queries. These are resolved sequentially, producing intermediate artifacts, with which we initialize the subsequent CoT reasoning procedure. Extensive experiments demonstrate that $\pi$-CoT significantly outperforms standard RAG and in-context CoT on multi-hop question-answering benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents \pi-CoT, a prompting framework that has an LLM first generate Prolog queries from multi-hop natural language questions, then execute each via SLICE to build a KB. SLICE essentially calls LLM for fact extraction or verification. Experiments on multiple benchmarks report some performance improvements over baselines due to the symbolic prolog-guided decomposition and reasoning structures.

### Strengths
Overall, the program framing is motivating, and the idea of combining symbolic scaffolding with Prolog queries with LLM for knowledge extraction and reasoning is conceptually straightforward. The intermediate Prolog queries and per-hop passages improve interpretability and reliability over the reasoning trace. Evaluation across multiple datasets establishes the validity of the method. The analysis in Figure 3 particularly indicates the method's stability even as the reasoning step increases.

### Weaknesses
1. There already exist many similar approaches generating Prolog from LLMs for improving arithmetic reasoning and multi-hop QA, like "Reliable Reasoning Beyond Natural Language", "ProSLM: A Prolog Synergized Language Model for explainable Domain Specific Knowledge Based Question Answering", "Arithmetic Reasoning with LLM: Prolog Generation & Permutation", "Thought-Like-Pro: Enhancing Reasoning of Large Language Models through Self-Driven Prolog-based Chain-of-Though". A comparison or discussion with these works is missing in the paper. 
2. In open-domain results (Table 1, 2) and in-context datasets (Table 4a), \pi-CoT does not significantly outperform the baselines. Why do the gold passages also diminish the value of Prolog's symbolic guidance?
3. No ablation studies to isolate contributions of components like Prolog and SLICE chaining. Consider adding an ablation by removing specific components from the prompt when generating the final answer (Section 4.3). 
4. It appears that the predicate definitions, question templates, and statement templates are LLM‑generated with few-shot examples. How often are these definitions incorrect, and how would this affect the downstream performance?

### Questions
1. See weakness for details.
2. Issues in writing: Tables 1–2 label the proposed method as “Memento (Ours)” rather than π‑CoT; “Tukey BSD” is likely a typo for Tukey HSD; typo "Quirell" at line 284

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
The paper proposes $\pi$-CoT (Prolog-Initialized Chain-of-Thought), a prompting strategy designed to enhance large language models’ (LLMs) reasoning in multi-hop question-answering tasks.

$\pi$-CoT first translates natural language questions into Prolog queries that decompose reasoning into a sequence of sub-queries. Each sub-query is executed through a SLICE module (Single-step Logical Inference with Contextual Evidence), which uses the LLM to extract or verify relevant facts from unstructured text and update a symbolic knowledge base. The intermediate results including facts, retrieved passages, and natural-language notes are then used to initialize the final chain-of-thought reasoning step, combining symbolic rigor with neural flexibility.

Experiments on HotpotQA, 2WikiMultiHopQA, MuSiQue, and PhantomWiki show consistent or improved performance over RAG, IRCoT, and HippoRAG baselines.

### Strengths
[S1] Interesting conceptual perspective. The perspective of reliable decomposition with formal method is interesting. It offers a principled way to constrain reasoning trajectories while keep the subtasks manageable for LLMs during multi-hop inference.

[S2] Writing quality is good. The paper is in general well-written and easy to follow. It also appropriately situates itself among prior works (e.g., IRCoT, Self-Ask, GraphRAG, HippoRAG 2).

[S3] Comprehensive evaluation. The evaluation is done on both real-world datasets (HotpotQA, MuSiQue, 2WikiMultiHopQA) and synthetic multi-hop dataset (PhantomWiki), which provides insights from both realistic and controlled scenarios.

### Weaknesses
[W1] Improvement can be inconsistent across datasets. On real-world datasets (e.g., HotpotQA, MuSiQue), $\pi$-CoT performs comparably to baselines, with statistical significance only on certain datasets like 2WikiMultiHopQA. The claimed “significant outperforming” does not hold uniformly. Also, I think it would be good to also make clear that only prompting-based methods are compared in the table. As latest SOTA on the datasets are way higher. E.g., finetuned approaches on HotpotQA is ~10% higher. (https://hotpotqa.github.io)

[W2] Potential high cost. The approach expands every query into multiple Prolog sub-queries, and each step involves a separate LLM call plus intermediate reasoning tokens. Although Tab. 2 reports retriever efficiency, the number of total LLM calls and token usage per question is not quantified. This may make $\pi$-CoT expensive. It would be great that analyses can be provided on that front. 

[W3] Dependency on accurate semantic parsing. The pipeline critically relies on the LLM’s ability to correctly translate NL questions into Prolog queries and definitions (Sec. A.1). When the LLM fails to produce the right predicates or variable bindings, downstream steps will propagate errors, potentially cancelling out the benefits.

### Questions
[Q1] How many total LLM calls or tokens are required per question (vs. standard RAG or IRCoT)? Could parts of the pipeline (e.g., deterministic Prolog execution) be done without invoking the LLM?

[Q2] Are all intermediate steps truly necessary in the LLM prompt for final performance? Have you tried skipping fact-verification or compressing intermediate notes?

[Q3] For queries like “Who is the female Polish scientist who won the Nobel Prize in 1903”, if the query was decomposed to q1= woman(X), q2=scientist(X) etc, then the intermediate states could contain too many entities. How is this type of token explosion handled?

[Q4] I think there might be a typo in the method name in Tables 1 and 2, I think the name Memento was not used elsewhere in the paper.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces π-CoT (Prolog-Initialized Chain-of-Thought), a prompting strategy that integrates Prolog into LLM reasoning. It decomposes complex multi-hop questions into Prolog sub-queries, each solved step-by-step using the SLICE (Single-step Logical Inference with Contextual Evidence) module, and uses the resulting symbolic traces to initialise the final CoT reasoning step. Notably, both the generation of Prolog sub-queries and the SLICE procedure (translation and answering of sub-queries) are performed by LLMs. The authors evaluate their method on the HotpotQA, 2WikiMultiHopQA, MuSiQue, and PhantomWiki datasets.

### Strengths
- The paper presents an interesting approach to addressing the limitations of combining CoT and RAG by incorporating neural-symbolic reasoning through the use of Prolog.
- The paper is clearly written, with well-organised explanations and helpful illustrations.

### Weaknesses
- The experimental setup appears somewhat arbitrary and selective. For example, in Table 3, the retrieval model differs from that used in Tables 1 and 2. Is there a specific reason for this? It seems that the baselines in Tables 1 and 2 could also be evaluated under the retrieval model of Table 3 for a fairer comparison. Similarly, in Section 5.2 / Table 4, why are the PW-S and PW-M datasets not used in the experiments of Tables 1–3?

- The paper lacks robust analysis regarding the Prolog component. Were the processes of generating, translating, and answering Prolog sub-queries executed without errors? Did the authors observe any parsing or execution errors (as they are done by LLMs) during these steps?

- The empirical evidence for Prolog improving multi-hop reasoning is limited. In Table 1, π-CoT’s advantage over IRCoT on HotpotQA diminishes on MuSiQue. As the authors note (line 374), prior work has shown that many HotpotQA examples can be solved with single-hop reasoning, whereas MuSiQue was explicitly designed to mitigate this issue. Given that, is there sufficient evidence that Prolog-based reasoning truly benefits multi-hop reasoning? Alternatively, could one interpret that while Prolog offers structured neuro-symbolic reasoning, it may introduce constraints that limit performance on multi-hop tasks (for instance, through error propagation across sub-query resolutions)?

### Questions
- In Table 2, the authors measure efficiency solely by the number of retriever calls. However, does the proposed method increase the overall computational cost compared to other RAG + CoT baselines, considering the cost of generating Prolog queries and resolving each sub-query through the LLM, as well as the expanded input length and generation overhead? If so, should it perhaps be compared not only to standard CoT but also to inference-time intervention methods with similar computational budgets?

- Why are statements evaluated as false simply ignored? Could incorporating them as additional context in the final CoT reasoning be beneficial?

- (Minor) In Tables 1 and 2, the proposed method is labelled as Memento (Ours)

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper improves the multi-hop reasoning abilities of LLMs by using a prolog inspired approach. They propose pi-CoT which first translates the reasoning question into a prolog program and then executes the prolog program in a step-by-step manner using their approach called SLICE which combines LLM-based reasoning with the semantics of prolog execution. They show that pi-CoT outperforms baselines on several datasets due to its ability to better handle multi-hop multi-branch reasoning than baselines.

### Strengths
* The paper is clearly written.
* The method is novel and is a useful way to combine the benefits of a symbolic system (prolog) with the knowledge and natural language reasoning abilities of LLMs.
* Results are presented with variances and appear mostly significant.

### Weaknesses
* The method’s dependence on prolog potentially limits this method to a specific set of problems.
* It is not clear what Memento refers to in the tables. Bolding in the tables is also confusing. This makes the results very hard to understand.
* Cost/latency is not evaluated.

### Questions
* How does the cost of pi-CoT compare to baselines? Does it perform more RAG lookups or more LLM inference calls than the baselines?
* How often were errors due to incorrect formulation of the problem as prolog in the initial step? For problems that are harder to formalize as prolog, this method will inevitably fail.
* How are concepts like negation handled? This seems like it could lead to a blow up of the knowledge base if the prolog program is not written in a smart way.

### Soundness
3

### Presentation
3

### Contribution
3
