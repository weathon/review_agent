# MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains

- Avg Score: 5.00
- Decision: Accept (Oral)
- Scores: 4, 4, 6, 6

## Abstract
With the increasing demand for step-wise, cross-modal, and knowledge-grounded reasoning, multimodal large language models (MLLMs) are evolving beyond the traditional fixed retrieve-then-generate paradigm toward more sophisticated agentic multimodal retrieval-augmented generation (MM-RAG). Existing benchmarks, however, mainly focus on simplified QA with short retrieval chains, leaving adaptive planning and multimodal reasoning underexplored. We present MC-Search, the first benchmark for agentic MM-RAG with long, step-wise annotated reasoning chains spanning five representative reasoning structures. Each example specifies sub-questions, retrieval modalities, supporting facts, and intermediate answers, with fidelity ensured by HAVE (Hop-wise Attribution and Verification of Evidence), resulting in 3,333 high-quality examples averaging 3.7 hops. Beyond answer accuracy, MC-Search introduces new process-level metrics for reasoning quality, stepwise retrieval and planning accuracy. By developing a unified agentic MM-RAG pipeline, we benchmark six leading MLLMs and reveal systematic issues such as over- and under-retrieval and modality-misaligned planning. Finally, we introduce Search-Align, a process-supervised fine-tuning framework leveraging verified reasoning chains, showing that our data not only enables faithful evaluation but also improves planning and retrieval fidelity in open-source MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MC-SEARCH, a comprehensive benchmark designed to evaluate and advance agentic multimodal retrieval-augmented generation (MM-RAG) systems, particularly focusing on structured, stepwise long-horizon reasoning. MC-SEARCH comprises 3,333 high-quality, HAVE-verified examples encompassing five distinct multi-hop reasoning topologies (including serial, parallel, and cross-modal forks), each annotated with granular sub-questions, modalities, evidence, and intermediate answers. The paper also proposes novel process-level evaluation metrics that go beyond traditional answer accuracy, enabling fine-grained analysis of stepwise retrieval, planning, and reasoning. Extensive benchmarking is conducted on six leading MLLMs, and a process-supervised fine-tuning scheme (SEARCH-ALIGN) leveraging the new dataset is presented, showing notable improvements for open-source models.

### Strengths
**Rigorous Dataset Construction:** MC-SEARCH addresses a clear gap in existing multimodal RAG benchmarks by providing long, structured, and verified reasoning chains spanning five rich reasoning topologies. The dataset construction meticulously filters for non-redundancy and necessity at each reasoning step using the HAVE protocol, resulting in high annotation quality

**Process-Level Evaluation:** The paper moves beyond answer-level benchmarking by introducing stepwise “Hit per Step,” Rollout Deviation, and LLM-as-a-Judge metrics, directly quantifying the chain of reasoning, retrieval fidelity, and error types. This is a step change from prior black-box evaluation paradigms.

New Method: They propose SEARCH-ALIGN, a process-supervised fine-tuning framework for MLLMs that leverages verified reasoning chains to align model behavior

### Weaknesses
- Although the model coverage is broad, several open-source baselines used are not state-of-the-art for their respective modalities, which may overstate SEARCH-ALIGN’s improvements. The paper does not sufficiently justify the exclusion of other competitive open-source MLLMs or recent RAG architectures such as UniRAG and MIRAGE from direct comparison.

- While Table 3 highlights clear performance gains from SEARCH-ALIGN, the contribution of individual components—planning, retrieval, and modality selection—is analyzed only qualitatively. A more rigorous ablation is needed to determine whether the model genuinely learns better planning strategies or simply imitates supervised reasoning steps.

- Given the use of LLMs for data construction, filtering, and evaluation, to what extent might there be annotation artifacts or overfitting to the verifier’s error modes? How do the authors ensure the robustness of results in light of potential circularity?

### Questions
- I’m a bit confused about why the reasoning process is represented as a graph rather than a chain.

- Why didn’t you include ROUGE and MRFS scores [1] in your evaluation?

- Why didn’t you incorporate retrieval-related metrics in your benchmark analysis?

- How does your method Search-ALign perform compared to other RL-based agentic optimization approaches?



[1] Pan Z, Luo H, Li M, et al. Chain-of-action: Faithful and multimodal question answering through large language models[J]. arXiv preprint arXiv:2403.17359, 2024.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MC-SEARCH, a new benchmark for evaluating multimodal agentic Retrieval-Augmented Generation (MM-RAG). The authors argue that existing benchmarks are too simple, focusing on short reasoning tasks. To address this, they make four main contributions: 

1. The MC-SEARCH benchmark, a dataset of 3,333 examples featuring long reasoning chains (avg. 3.7 hops) organized into five distinct reasoning topologies. 

2. A data filtering process called HAVE (Hop-wise Attribution and Verification of Evidence) to ensure each reasoning step is necessary and non-redundant. 

3. New process-level metrics (e.g., Hit per Step, Rollout Deviation) to evaluate intermediate reasoning steps, not just the final answer. 

4. The SEARCH-ALIGN framework, a process-supervised fine-tuning method that uses the benchmark's annotations to improve the capabilities of open-source models.

### Strengths
The paper's primary strengths lie in its thoughtful benchmark design and its focus on process-level evaluation.

Structured Benchmark Design: The introduction of five explicit reasoning topologies is a significant contribution. It moves beyond simply creating "long" chains and provides a structured way to diagnose specific model failures (e.g., a model failing on Parallel Forks but succeeding on Linear Chains). This offers a more granular analysis than existing benchmarks.

Rigorous Data Curation: The HAVE process is a commendable effort to improve benchmark quality. By programmatically filtering out trivial or redundant reasoning steps, the authors have likely created a more challenging and reliable testbed for genuine reasoning abilities, addressing a common weakness in synthetically generated datasets.

Process-Oriented Evaluation: The push for metrics beyond final answer accuracy is timely. In complex agentic tasks, intermediate failures are often hidden. Metrics like Hit per Step (HPS) and Rollout Deviation (RD) provide a clearer view of how and where models fail, which is critical for future development.

Demonstrated Training Utility: The inclusion of the SEARCH-ALIGN framework is a strong point. By showing that the detailed annotations can be used for process-supervised fine-tuning to improve open-source models, the authors prove that MC-SEARCH is a valuable resource for both evaluation and model development.

### Weaknesses
The paper's contributions are undermined by significant weaknesses, primarily an overstatement of novelty and key methodological limitations.

Novelty overstatement — The paper's central claim of being the "first benchmark for agentic MM-RAG with long, step-wise annotated reasoning chains" is not well-supported. Prior work such as Dyn-VQA benchmark was specifically designed for dynamic, multi-hop questions requiring complex, adaptive retrieval and also introduced a self-adaptive planning agent. The paper must reposition its contribution not as being the first, but as providing uniquely structured and verified reasoning chains, and it needs to properly differentiate itself from Dyn-VQA and other related works like WebQA, MRAG-Bench, and MMSearch.

Potential model bias — While the HAVE pipeline partially mitigates bias via cross-model filtering (Qwen2.5-VL and Gemini-Pro), Gemini models still dominate generation, filtering, and evaluation phases. This may tune the benchmark toward Gemini’s reasoning style.

Simplified retrieval setting — The evaluation uses a top-1 retrieval setup, which does not reflect realistic RAG conditions where irrelevant documents co-exist. The conclusions regarding over/under-retrieval and SEARCH-ALIGN efficacy may not fully generalize to top-k retrieval.

Limited justification for reasoning topologies — Although the five topologies are intuitive, the paper does not provide empirical or theoretical justification for their selection (e.g., frequency in real-world tasks). Including such evidence would improve the framework’s validity.

Metric rigidity — The Hit per Step (HPS) metric relies on exact evidence matching, which could penalize models finding semantically equivalent alternatives. The authors do mention a semantic alignment mechanism for structural comparison, but an integrated soft-matching variant could make evaluation fairer.

### Questions
How does MC-SEARCH differ empirically and conceptually from Dyn-VQA and MMSearch beyond having predefined topologies? Could the authors quantify the added diagnostic value of these structures?

How do the authors ensure benchmark neutrality given that Gemini models are used for both generation and evaluation? Have they tested whether non-Gemini models (e.g., GPT-4o, Claude, QwenVL) are unfairly disadvantaged?

How might the conclusions about SEARCH-ALIGN change under a top-k retrieval setting where irrelevant evidence must be filtered dynamically?

What criteria guided the selection of the five reasoning topologies? Are these empirically grounded (e.g., observed in task distributions) or designed heuristically?

Could HPS be complemented by a semantic similarity–based variant, ensuring agents that retrieve equivalent but non-identical evidence are not penalized?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a multimodal retrieval-augmented generation benchmark for agentic paradigm. It has 3333 examples averaging 3.7 hops with sub-questions, retrieval modalities, supporting facts, and intermediate answers ensured by HAVE. It also proposes some process-level metrics to judge the reasoning quality and retrieval performance and planning fidelity.

### Strengths
1. this submission is well-prepared, especially in figures, tables, and appendix.

2. the key contribution of this submission is obvious and makes sense.

3. the process metrics are actually needed things in multi-step reasoning tasks. 

4. the experiments and analysis are comprehensive and high-quality.

### Weaknesses
Many related work may help authors to enhance the completeness of the submission:

1. evaluation on robustness of resisting harmful information is also interesting in RAG-based agentic framework [1].

2. "multi-modality" may also extend to SQL-based database, query-rewriter-based web-search, and even more [2].

3. token usage (input and output) and the number of retrieval callings are also helpful to enhance the benchmark [3].



[1] Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain

[2] Chain-of-Action: Faithful and Multimodal Question Answering through Large Language Models

[3] RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems

### Questions
1. how to evaluate and ensure that Qwen2.5-VL-7B is good at judging each hop is both necessary and non-redundant? is there any mannual double-check and fine-tuning methods or?

2. please refer to weakness section for more contents.

### Soundness
3

### Presentation
3

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
This paper is well-motivated, presenting a novel perspective for improving retrievers by tackling hard tasks. We are convinced by the sufficient baselines used to verify the proposed method and believe it inspires future work on more challenging tasks. I have several minor concerns, including potential unreliability in using LLMs for relevance evaluation, the lack of a simpler majority vote baseline, prohibitive computational costs, and some lack of clarity in the agent selection process. We further seek clarification on case analyses for hard samples, the potential benefit of incorporating more specialized agents.

### Strengths
This paper is well-motivated. It tries to solve hard tasks in a new perspective of view, improving retriever. Sufficient baseline retrievers adopted to verify the proposed method to improve the retriever. This work inspires the future direction on solving more challenging tasks like BrowseComp.

### Weaknesses
1. Previous work has shown that asking LLMs themselves to evaluate the relevance of queries and documents are not that reliable [[1]](https://arxiv.org/abs/2505.21870). Applying Code Agent and CoT Agent is also within this paradigm. It would be better if there are experiments conducted to verify the relevance between inconsistency (among Code and CoT Agent) and query difficulty.
2. The initial CoT agent's decision ($y^{g}_0$) is only overturned if all L discussion groups unanimously disagree. Thus, it seems that a simpler majority vote is a more standard and intuitive baseline, but no comparison is offered.
3. Prohibitive computational cost makes the approach infeasible for generating datasets of any significant size. (1 hard example=~58 API calls and 100k tokens)
4. Lack of Clarity on Agent Selector and Validator.

### Questions
1. Line186 & 208. Inconsistency among Code Agent and CoT Agent indicates a hard sample. Could the author give any case analysis?
2. Related to Weakness1, will it be more trustworthy by incorporating more Agents besides Code Agent and CoT Agent? E.g., agents that specialize searching. Or, the current setup is enough.
3. Typo? Line243: $y_{i,k}^{t}$ and $y_{i,k}^{t}$.

### Soundness
3

### Presentation
3

### Contribution
3
