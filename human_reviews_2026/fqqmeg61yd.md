# RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Large language models (LLMs) have achieved impressive performance on knowledge-intensive tasks, yet they often struggle with multi-step reasoning due to the unstructured nature of retrieved context. While retrieval-augmented generation (RAG) methods provide external information, the lack of explicit organization among retrieved passages limits their effectiveness, leading to brittle reasoning pathways. Recent interpretability studies highlighting the importance of structured intermediate reasoning further align with this perspective. 
We propose Retrieval-And-Structuring (RAS), a framework that dynamically constructs question-specific knowledge graphs through iterative retrieval and structured knowledge building. RAS interleaves targeted retrieval planning with incremental graph construction, enabling models to assemble and reason over evolving knowledge structures tailored to each query. On seven knowledge-intensive benchmarks, RAS consistently outperforms strong baselines, achieving up to 8.7\% and 7.0\% gains with proprietary and open-source LLMs, respectively. Our results demonstrate that dynamic, question-specific knowledge structuring offers a robust path to improving reasoning accuracy and robustness in language model generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a planning-based iterative RAG method for knowledge-intensive LLM generation. It conducts experiments on several datasets to verify its effectiveness.

### Strengths
1. The paper is clear and well-written, with a reasonable motivation and considerable insight.

2. The experiments are sufficient and provide a comprehensive analysis.

3. The reproduction is excellent, with code and extensive reproduction details provided.

### Weaknesses
1. The backbone used in the experiments appears somewhat outdated. We recommend using a more recent backbone, such as Claude-4 and LLama-3.1. Alternatively, consider the GPT-4/5 series or the Qwen3 series.

2. Please provide a more detailed analysis of the core differences between this paper and other iterative RAG methods.

### Questions
See weaknesses above. Please respond to these cons.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a RAG framework for structuring the context in a graph form for improving knowledge-intensive tasks. In detail, this framework is composed of sub-question generation, retrieval-and-structuring, and terminal-and-answering. Additionally, this paper also proposes a training framework to enhance the model to be compatible with this framework. Then the comprehensive evaluation over five tasks shows the advantages of new framework over closed and open source models.

### Strengths
1. I appreciate this paper for utilizing the graph form representing knowledge and an elegant workflow to address complex and knowledge-intensive tasks. 
2. The experiments are solid across many classic knowledge-intensive tasks, and the improvement is good.
3. This framework is general for closed and open-source LLMs, where the code is open-source.

### Weaknesses
1. This paper lacks the motivation for why organizing context in a graph form improves the performance of RAG. 

2. The involved benchmarks are outdated. There are many new graph-rag datasets for reasoning on domain knowledge. The results on such datasets are necessary to improve the quality.

[1] GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation
[2] When to use Graphs in RAG: A Comprehensive Benchmark and Analysis for Graph Retrieval-Augmented Generation

### Questions
See weakness!

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Retrieval-And-Structuring (RAS), a framework designed to enhances reasoning in large language models for knowledge-intensive tasks. Unlike traditional RAG approaches that retrieve but fail to organize context, RAS dynamically builds question-specific knowledge graphs through iterative retrieval and structured knowledge synthesis. The framework consists of three key stages: planning (to identify knowledge gaps), retrieval and structuring (to form factual triples), and graph-based answering. Experiments across seven diverse benchmarks demonstrate gains across multiple datasets.

### Strengths
+ The paper introduces RAS, a dynamic, query-specific knowledge graph construction framework that avoids inefficiencies of global KG indexing and eliminates costly offline graph building, achieving pay-per-query scalability.
+ The method is evaluated comprehensively on multiple datasets across both open-source and closed-source LLMs, demonstrating consistent and generalizable effectiveness.
+ The framework improves reasoning transparency and provides a structured way to bridge retrieval and reasoning in a unified pipeline.

### Weaknesses
- The paper lacks concrete examples demonstrating how graph representations outperform plain text in enhancing reasoning accuracy. It remains unclear which aspects of reasoning (e.g., factual grounding, compositional reasoning, or logical chaining) benefit most from graph structuring.
- Moreover, since RAS and RPG are trained on different datasets of different scales, and the appendix shows that training set size directly affects performance, the gains of RAS over RPG on open-source settings (insignificant on half of the datasets) might stem from data scale rather than the graph mechanism itself.
- In principle, clearer graph structures should lead to larger improvements on more complex reasoning tasks, yet the paper does not analyze this correlation or provide corresponding comparisons to validate the claim that graph structuring enhances reasoning.
- There are several presentation issues, such as missing “top-2” formatting in Table 1 (best/second-best not consistently marked) and missing figure references in the appendix.

### Questions
Since graph extraction can be viewed as a form of information organization and denoising, complex sentences with multiple conditions might lose contextual nuances when converted to triples.
Why not combine graph representations and original retrieved text during answering, to preserve both structure and completeness?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
LLMs perform well on knowledge-intensive tasks but struggle with multi-step reasoning due to unstructured retrieved context, a limitation RAG shares. Aligning with structured intermediate reasoning insights, the authors propose RAS—a framework that dynamically builds question-specific knowledge graphs via iterative retrieval and incremental graph construction. Evaluated on seven benchmarks, RAS outperforms baselines by up to 8.7% (proprietary LLMs) and 7.0% (open-source LLMs), validating dynamic, query-tailored knowledge structuring as effective for enhancing reasoning accuracy/robustness. Data and code are publicly available.

### Strengths
1. The paper is easy to follow.
2. The studied problem is important.
3. The proposed method is evaluated on an extensive set of datasets.

### Weaknesses
1. Important related work is neglected. Constructing dynamic knowlege graph for sovling complex reasoning tasks in RAG is a well-studied area. Representative work, such as SG-Prompt, ERA-CoT, and KnowTrace, is not analyzed in the paper, especially KnowTrace. Compared with these existing work, the novelty and technical contribution are limited.
2. Important baselines are missing in the experiments. It is not clear the true performance of the proposed method among existing work.

### Questions
Please refer to the weakenesses.

### Soundness
2

### Presentation
3

### Contribution
2
