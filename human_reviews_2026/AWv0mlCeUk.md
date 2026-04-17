# Discourse-Aware Retrieval-Augmented Generation via Rhetorical Structure Modeling

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 6

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as an important means for enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages as flat and unstructured text, which prevents the model from capturing structural cues and constrains its ability to synthesize dispersed evidence and to reason across documents. Although a few recent approaches attempt to incorporate structural signals, each remains restricted to shallow representations such as entity graphs or dependency edges and thus fails to capture hierarchical discourse organization. To overcome these limitations, we propose Discourse-RAG, a structure-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk rhetorical structure theory (RST) trees to capture local coherence hierarchies and builds inter-chunk rhetorical graphs to model cross-passage discourse flow. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Discourse-RAG achieves a new state-of-the-art ROUGE-L score of 42.4 on ASQA dataset and improves LLM Score by 12.79 points over standard RAG on Loong benchmark. These findings underscore the important role of discourse structure in advancing retrieval-augmented generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Discourse-RAG, a retrieval augmented generation pipeline that makes the model explicitly use discourse structure. It first parses each retrieved chunk into an RST like tree and then links chunks with rhetorical relations to show support, elaboration, or conflict. A final planning stage guides generation using this structure. In evaluation it outperforms standard RAG and other structure aware baselines on long context QA, ambiguous QA, and scientific summarization.

### Strengths
- It uses one pipeline where chunk level discourse trees feed into a cross chunk graph, and both are used to guide generation.

- The method is tested on three tasks, with two Llama models, in both open and closed settings, and it beats 2025 RAG baselines, including on ASQA.

- The ablations, noise tests, and chunk size tests show that the method improves across different settings.

### Weaknesses
- While training-free, the pipeline requires multiple LLM calls per query (RST parsing per chunk, O(k²) pairwise relation inference, planning, generation). Cost grows quickly with larger k and the paper should discuss the latency and token counts.

- I noticed that all LLM benchmarks use Llama 3.x models. Since Qwen was already used for embeddings, why not include Qwen models in the main comparisons as well?

- Discourse quality is not validated. All trees and relations come from an LLM prompt rather than a parser with known accuracy. If the LLM segments poorly the whole pipeline can weaken.

- Relation set may be too big. The method uses many fine grained discourse labels but does not show which ones actually help. A smaller set might work the same.

- Evaluation scope is narrow. Results are mostly on English, long context, clean inputs. It is unclear how well this works on multilingual or noisy data.

### Questions
- Did you evaluate the accuracy of your LLM-generated RST trees against gold-standard annotations (e.g., on RST-DT)? 

- Have you considered integrating neural discourse parsers or non-RST frameworks (e.g., entity grids, coherence models)?

- In cases where Discourse-RAG underperforms standard RAG (if any), what are the failure modes? Are they due to incorrect rhetorical parsing, poor planning, or something else?

- Beyond automatic metrics (ROUGE, LLM Score), was there any human assessment of coherence, faithfulness, or readability? LLM-based scoring can be biased.

### Soundness
3

### Presentation
2

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
This paper introduces Discourse-RAG, a novel training-free framework designed to address a key limitation in standard Retrieval-Augmented Generation (RAG): the tendency to treat retrieved documents as a flat, unstructured "bag of facts." This "flat structure" problem leads to intra-chunk structural blindness and inter-chunk coherence gaps, hindering the model's ability to synthesize evidence and reason.

### Strengths
S1. The paper identifies a clear and important limitation of standard RAG (its "flat structure") and proposes a novel, linguistically-grounded solution that directly addresses it.
S2. The method achieves state-of-the-art performance on multiple, diverse benchmarks (long-doc QA, ambiguous QA, summarization), demonstrating its effectiveness and generalization ability.
S3. The paper is clear, well-illustrated, and reproducible thanks to the detailed appendices.

### Weaknesses
W1. This method is computationally expensive. The proposed pipeline requires an enormous number of LLM inference calls per query. As described in the methodology, for a top-k retrieval, the framework k calls for intra-chunk RST tree construction, k * (k – 1) calls for inter-chunk rhetorical graph construction and 1 call for planning.
W2. The entire framework's success is predicated on the LLM's ability to function as a high-quality, zero-shot RST parser. This capability is assumed, not proven.

### Questions
Q1: Why did the authors not include an intrinsic evaluation of the RST parser against a gold-standard dataset? How can we be confident that the generated structures are faithful and not just plausible-sounding hallucinations that happen to guide the LLM?
Q2: Did the authors compare the full, complex RST parsing against simpler structural signals? For example, what is the performance if only explicit discourse markers (e.g., "however", "because", "in contrast") are used to build the inter-chunk graph, without any RST tree parsing?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Discourse-RAG, a retrieval-augmented generation framework that explicitly models intra- and inter-chunk rhetorical structures using Rhetorical Structure Theory (RST) and rhetorical planning to improve coherence and factual consistency in long-context reasoning. It demonstrates strong empirical results on multiple benchmarks (Loong, ASQA, and SciNews) across varying document lengths, outperforming several state-of-the-art RAG baselines. While the approach is somewhat heavy and empirically oriented, its clear performance gains and conceptual novelty justify acceptance, provided reproducibility and efficiency details are strengthened.

### Strengths
1. The idea of introducing rhetorical trees and inter-chunk discourse graphs into RAG is original and conceptually well-motivated, bridging discourse analysis and generative reasoning.
2. The method is tested on diverse, long-context benchmarks with detailed ablations and robustness studies (chunk size, Top-k, noise, and structure perturbations), giving credibility to the empirical claims.
3. Discourse-RAG outperforms strong baselines (StructRAG, MAIN-RAG, RQ-RAG) in both accuracy (LLM Score, EM) and factuality (SummaC, SARI), particularly on large-context and noisy retrieval scenarios.

### Weaknesses
1. Both intra- and inter-chunk discourse structures rely on LLM prompting for RST parsing, raising concerns about reproducibility, cost, and stability.
2. While results show improvements, the paper doesn’t deeply explore why rhetorical modeling helps or how structural cues propagate through the generator beyond surface correlations.
3. The multi-agent setup (parsing, graphing, planning) introduces significant preprocessing latency and complexity, which may limit real-time or large-scale deployment; no efficiency analysis is reported.

### Questions
1. Provide a quantitative evaluation of RST parsing accuracy and its impact on final performance (e.g., noise sensitivity to incorrect discourse trees).
2. Include a runtime and cost comparison versus other RAG systems (e.g., StructRAG, MAIN-RAG) to demonstrate practical feasibility.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents Discourse-RAG, a rag framework that explicitly models discourse structures. It works via a three-stage pipeline: 1) constructing intra-chunk RST trees to identify core vs. supporting information, 2) building inter-chunk rhetorical graphs to model relationships, 3) using a discourse-aware planning module to generate a blueprint for the final answer. Experiments on ASQA, Loong, and SciNews benchmarks show that Discourse-RAG achieves strong performance.

### Strengths
The paper is well written, and gets strong performance with good baselines.
The method can be plugged in any setup without any fine-tuning. 
The components are ablated.

### Weaknesses
There is no analysis on how the method scales (in terms of cost (tokens) and latency) with higher top-k settings.

### Questions
What are the tradeoff with offline indexing and creation of the RST trees for the whole dataset?
How does the method scale in terms of latency, tokens with respect to higher top-k, different chunk sizes, bigger documents?

### Soundness
3

### Presentation
3

### Contribution
3
