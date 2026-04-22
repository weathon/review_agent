# MIRAGE: MULTI-HOP INTERLEAVED REASONING AND RETRIEVAL-GROUNDED EVIDENCE

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities
across a wide spectrum of natural language tasks, especially information retrieval
and comprehensive reasoning. However, existing benchmarks typically evaluate
these abilities dependently, failing to capture the process with interleaving and in-
tegrated reasoning and retrieval for complex queries, which is commonly needed
in real-world applications. To address this limitation, we propose a novel bench-
mark—MIRAGE—the first benchmark specifically designed to assess an LLM’s
ability of step-wise reasoning on a decomposed query and iterative retrieval based
on intermediate reasoning outcomes across multiple interactions, ultimately syn-
thesizing a coherent and comprehensive answer. MIRAGE consists of 579 real-
world queries spanning diverse domain including Finance, Legal, Technology,
Academia. We collected both open-sourced dataset and real-world forum con-
versation as our data sources, and we further curate the dataset into a multi-turn
format along with a consicse question and comprehensive answer. To support fur-
ther development and scalability, we also introduce a data generation pipeline for
benchmark expansion. We evaluate state-of-the-art approaches on the our bench-
mark and find that none exhibit consistently strong performance, underscoring
their lack of ability of interleaving reasoning and retrieval as well as the need for
further advancement. Our benchmark provides a foundation for future research
into more robust and dynamic information-seeking agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MIRAGE, a benchmark designed to evaluate llm-based RAG systems on the interleaved task of multi-hop retrieval and evidence-grounded reasoning. 
The authors claim that this interleaved RAG capabilities is critical for real-world problem-solving, but this spectrum is not well addressed by existing benchmarks.
Therefore, this paper propose MIRAGE, which consists of 579 instances across four domain (Legal, FInance, Tech, Academy). Both of them needs llms to generate sub-queries iteratively, retrieve knowledge from adversarial corpus, to find the final answer.

The authors conduct experiments comparing closed-book LLMs, single-turn RAG, multi-hop RAG, and interleaved RAG systems, finding that even SOTA RAG frameworks achieve at most 61.5% Fact-Score, with failures attributed to premature reasoning and distractor vulnerability.
The paper positions MIRAGE as a public resource to advance agentic reasoning capabilities in future research.

### Strengths
1. This paper introduces the MIRAGE benchmark for Retrieval-Augmented Generation (RAG), designed to address challenges such as interleaved retrieval and noise resistance in the era of AI agents. The benchmark focuses on comprehensive evaluation of the retrieval-augmented workflow, assessing not only the accuracy of final answers but also capabilities like robustness to retrieval interference. It also provides relevant environments with both standard and adversarial settings, ensuring a high degree of reproducibility.

2. The authors evaluated current academic and industrial models and products on MIRAGE, drawing preliminary conclusions from the results.

### Weaknesses
1. What are the key distinctions between the benchmark proposed in this paper and previous or subsequent RAG benchmarks?
    - Pre-agent era benchmarks: Multi-hop QA datasets such as HotpotQA and MuSiQue.
    - Agent-era benchmarks: Web browsing benchmarks represented by BrowseComp.
    - Both types of benchmarks involve multi-turn, interleaved retrieval and generation, making them suitable for evaluating agent-era reasoning models. Moreover, some multi-hop QA benchmarks evaluate not only the correctness of the final answer but also the accuracy of intermediate retrieval steps.

2. The proposed benchmark has a relatively small sample size, containing only 500+ instances, which limits its applicability to evaluations in narrow domains.

3. The benchmark lacks sufficient difficulty. A high-quality benchmark should possess foresight by focusing on challenges that current models struggle to solve. For example, when BrowseComp was introduced, most existing methods achieved only single-digit scores. In contrast, the benchmark discussed here appears to lack challenge, as some models already achieved scores above 60 at its release. This raises the question of whether the benchmark fails to provide meaningful guidance for future model optimization due to its limited difficulty.

### Questions
please see weakness section for my quesiton.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MIRAGE, a benchmark designed to evaluate multi-hop interleaved reasoning and retrieval-grounded evidence generation. It aims to simulate realistic RAG workflows where reasoning and retrieval alternate across multiple steps. The dataset covers several specialized domains (finance, legal, technology, academia) and includes human verification and adversarial distractor construction. Experiments benchmark various LLM-based RAG systems under single-hop, multi-hop, and interleaved settings, accompanied by detailed error analyses distinguishing planning vs. synthesis failures.

### Strengths
1. The paper targets a timely and practically relevant problem: evaluating RAG systems that require iterative retrieval and reasoning.
2. The benchmark is well-structured, with clear design motivation, adversarial distractor generation, and human verification steps.
3. Experiments are comprehensive, covering multiple reasoning configurations and offering useful error analyses (e.g., planning vs. synthesis failures).
4. The paper is clearly written and easy to follow, with reproducible methodology and well-presented results.

### Weaknesses
1. The paper’s identified gap: evaluating interleaved reasoning and retrieval, feels primarily conceptual rather than substantive. Similar processes have been studied in prior RAG and agentic frameworks (e.g., ReAct, RARR). MIRAGE’s contribution lies more in its framing and benchmark packaging than in a fundamentally new task formulation.
2. Given that MIRAGE emphasizes multi-hop reasoning, the evaluation could better reflect step-level performance (e.g., scoring per hop, planning accuracy, intermediate retrieval relevance). The current setup uses rather standard metrics and separates "reasoner" and "retriever" evaluation without integrating them into a unified process-level metric. This limits the benchmark’s ability to assess true multi-hop reasoning quality. Also, there's a typo in Table 6, Reasoner instead of Resoner.

### Questions
1. The core task of interleaved reasoning and retrieval appears conceptually similar to prior agentic frameworks (e.g., ReAct). Could the authors please elaborate on what substantively differentiates the MIRAGE task formulation from these existing paradigms? Is the novelty primarily in the challenging, domain-specific corpus and the adversarial distractors, or is there a fundamental difference in the required reasoning process itself?
2. Given that MIRAGE is explicitly designed to evaluate the interleaved process of multi-hop reasoning, the evaluation in Table 6  seems focused on end-to-end performance (Fact-Score, nDCG@10) rather than the quality of the intermediate steps. Could the authors comment on why process-level metrics (e.g., planning accuracy of sub-queries, or hop-by-hop retrieval relevance) were not included? How can the benchmark effectively diagnose "Planning Failures" without such metrics?

### Soundness
3

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
3

### Summary
This paper introduces MIRAGE, a benchmark designed to evaluate LLMs' ability to dynamically interleave multi-hop retrieval and evidence-grounded reasoning. The benchmark comprises 579 tasks across four domains (Legal, Finance, Technology, Academia), constructed from real-world conversational data through a three-stage pipeline: (1) sourcing conversational seeds, (2) synthesizing complex reasoning tasks, and (3) constructing adversarial retrieval corpora. Comprehensive experiments reveal a stark performance gap that the best academic system achieving only 61.5% task success, with interleaved RAG agents significantly outperforming single-turn and structured multi-hop approaches. The authors identify two primary failure modes: planning failures and synthesis failures.

### Strengths
1. The paper addresses a genuine evaluation gap. MIRAGE is designed to evaluate the interleaved process where agents must dynamically decide when to retrieve and what to search for.
2. The paper includes comprehensive evaluation across multiple architectures (single-turn, multi-hop, interleaved, commercial systems).
3. The paper is well-written with clear motivation, good use of examples, and visualizations.

### Weaknesses
1. 579 tasks is relatively small for a benchmark, especially when split across 4 domains.
2. The paper does not report statistical significance of the results.
3. Domain distribution is highly skewed (Legal: 216, Technology: 184, Academia: 110, Finance: 69). This imbalance may affect the generalizability of conclusions.

### Questions
1. BERTScore might be unsuitable for such kind of abstractive, evidence-grounded answers. For future papers, if the authors can only pick one metric to report, which one would the authors recommend?
2. Would it be feasible to conduct a human evaluation on some samples (e.g., 100 examples) to validate the automated metrics and provide qualitative insights?
3. Can you provide a breakdown of results by domain? Are certain domains systematically harder?
4. Are there any considerations for avoiding potential benchmark fitting in the future?
4. "disconnect" -> "disconnection" in line 52; "Soucing" -> "Sourcing" in Figure 1. "Resoner" -> "Reasoner" in Table 6.

### Soundness
3

### Presentation
4

### Contribution
3
