# SAGE: Sufficiency-Aware Implicit Graph Exploration for Long Context Reasoning

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Large Language Models (LLMs) have achieved impressive progress in natural language processing, but their limited ability to retain long-term context constrains performance on document-level or multi-turn tasks. Retrieval-Augmented Generation (RAG) mitigates this by retrieving relevant information from an external corpus. Recently, graph-based RAG systems have shown promise for long-context reasoning. However, these methods face some challenges: high preprocessing computational costs, static graph architectures with fixed node and edge semantics, and complex parameter finetuning requirements that could limit their practical adoption. Recognizing that modern LLMs possess substantially improved reasoning capabilities, we propose SAGE, a dynamic implicit graph exploration framework that eliminates the need for explicit graph construction while preserving multi-hop reasoning benefits. Experiments are conducted on challenging long-context QA benchmarks, including NovelQA and Marathon. Our approach consistently outperforms strong baselines across these datasets. Additionally, it reduces storage and runtime requirements by over an order of magnitude. These results show that high-quality retrieval can be achieved through LLM-driven text exploration without relying on static preprocessing or vector representations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SAGE (Sufficiency-Aware implicit Graph Exploration), a retrieval-augmented framework for long-context reasoning that avoids explicit graph construction and dense vector indexing. Instead, SAGE treats documents as an implicit graph of chunks and lets an LLM iteratively (i) generate query-conditioned keywords, (ii) retrieve candidate chunks lexically, (iii) assess sufficiency using an objective importance score that measures each chunk’s causal impact on the answer distribution via noise perturbations, and (iv) expand along spatial (context neighbors) and semantic (topic neighbors) edges only when evidence is insufficient. On NovelQA and Marathon, SAGE reports higher accuracy than MiniRAG/RAPTOR across multiple LLaMA scales while cutting storage and preparation time by orders of magnitude, since it stores only the original text and skips any pre-built graphs/embeddings.

### Strengths
1. Pre-built graphs are costly and rigid, and embedding-based methods add storage overhead and reduce interpretability in long-context settings.
2. The method’s combination of implicit graph exploration, keyword-based retrieval, and sufficiency-aware stopping forms a clear and interpretable pipeline.
3. The importance score, based on light perturbation and KL evaluation, helps prevent premature stopping and reduces hallucination.
4. It is highly efficient: almost zero preprocessing, only original text stored (≈1×), yet it matches or outperforms MiniRAG/RAPTOR, especially with 8B/70B models.
5. Ablation studies show the benefit of each component (voters, expansion, importance signal), and performance is robust across context lengths.

### Weaknesses
1. Lexical retrieval ceiling: While interpretability is a plus, relying on keyword overlap risks recall issues for paraphrased, non-lexically aligned evidence; a hybrid (lexical + light semantic) comparison would strengthen claims of “embedding-free superiority.”  
2. Benchmark scope: Only multiple-choice QA (NovelQA/Marathon). It would be valuable to add free-form QA and multi-document synthesis (e.g., GovReports, Qasper, LongFact) to test robustness beyond MCQ cueing effects.  
3. Importance metric assumptions: The KL→cosine approximation assumes near-constant variance/softmax temperature; an empirical sensitivity study (temperature, noise distributions, #samples) would bolster soundness.  
4. Counting branch: Counting is handled via a special pipeline. This is practical, but it highlights that SAGE still needs task-specific forks; a unified controller that chooses among skills/tools could generalize better.  
5. Comparisons vs. stronger agentic baselines: Given the “LLM-as-retriever/agent” lineage, adding ReAct- or Toolformer-style agentic retrieval baselines (without embeddings) would clarify where SAGE’s sufficiency gating provides the most benefit. The set of compared methods in the paper is currently too limited.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces SAGE, a framework for dynamic implicit graph exploration. SAGE aims to preserve multi-hop reasoning capabilities in RAG by treating documents as implicit knowledge graphs where relationships are discovered on-demand, thereby eliminating the need for explicit graph construction.

### Strengths
N/A

### Weaknesses
1. The ad-hoc handling of "counting" problems, as described in Section 4.1, seems superfluous and undermines the method's generality. The technique employed for this specific task is highly intuitive and appears to offer minimal technical contribution.

2. The paper's overall presentation is poor, suffering from unclear writing and numerous formatting errors, which makes the methodology difficult to follow. For instance:

    a. A "DisSim" function is mentioned on line 329, with a reference to Appendix B.1 for implementation details. However, Appendix B.1 describes an "Imp()" function, and "DisSim" is never mentioned again.

    b. Algorithm 3 is included in the paper but is never referenced or explained in the main text.

    c. There are multiple distracting formatting errors, such as missing parentheses for citations (e.g., lines 112, 194) and improper spacing around parentheses (e.g., line 412).

4. The related work section is insufficient and misses several critical citations. The paper fails to discuss or position itself against highly relevant recent work, such as:
```
@inproceedings{wang2024knowledge,
title={Knowledge graph prompting for multi-document question answering},
author={Wang, Yu and Lipka, Nedim and Rossi, Ryan A and Siu, Alexa and Zhang, Ruiyi and Derr, Tyler},
booktitle={Proceedings of the AAAI conference on artificial intelligence},
volume={38},
number={17},
pages={19206--19214},
year={2024}
}
@article{li2024graphreader,
title={Graphreader: Building graph-based agent to enhance long-context abilities of large language models},
author={Li, Shilong and He, Yancheng and Guo, Hangyu and Bu, Xingyuan and Bai, Ge and Liu, Jie and Liu, Jiaheng and Qu, Xingwei and Li, Yangguang and Ouyang, Wanli and others},
journal={EMNLP},
year={2024}
}
```

### Questions
N/A

### Soundness
2

### Presentation
1

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
The paper proposes SAGE, a sufficiency-aware implicit graph exploration framework designed to enhance long-context reasoning for Large Language Models (LLMs). 

Instead of relying on explicit graph construction or vector embeddings, SAGE enables dynamic implicit graph traversal over text chunks using LLM reasoning capabilities. The framework introduces a sufficiency evaluation mechanism to decide when enough information has been retrieved, guided by an objective importance score that measures how much a text chunk influences the model’s reasoning. 

Experiments on NovelQA and Marathon benchmarks show that SAGE achieves SOTA accuracy and efficiency, outperforming graph-based methods like MiniRAG and Raptor, while reducing preprocessing time and storage by orders of magnitude. The authors argue that this embedding-free, reflective retrieval paradigm marks a paradigm shift toward more interpretable and adaptive long-context retrieval systems.

### Strengths
- SAGE eliminates the need for embedding and graph construction, significantly enhancing retrieval efficiency for long-context QA.
- SAGE introduces an objective, effective, and novel method for evaluating the importance of evidence in RAG.

### Weaknesses
- The paper's description of the methodology lacks clarity, making it difficult to correlate the main text, the algorithm, and the pipeline in Figure 1.
- The experiments are insufficient. The paper compares too few baselines and does not include a comparison with the GraphRAG paradigm mentioned at the beginning.
- The ablation study design has problems: firstly, it lacks experiments on the impact of iteration rounds $T$ on results; secondly, it fails to provide comparisons between with and without importance scores under non-default values of voter_num and recall_index.
- Experiments are limited to LLaMA family models. The generalization to other LLMs is not verified.

### Questions
Refer to Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an LLM-based RAG approach that iteratively retrieve sentences. The major motivation is to treat sentence as retrieval units and perform graph-like search trajectory without the need of pre-building a graph. Specifically, it identifies keywords in each turn, retrieve sentences by keyword-based matching, and evaluate evidence importance by measuring the effects of it on the LLM's output. Experiments are mainly comparing with some graph-based baselines that show the reduced offline cost and higher performance.

### Strengths
- This paper is inspired by graph-based approaches while proposes a method that mimic a similar behavior without graph construction. This work is thus well motivated
- The proposed objective importance metric is interesting and seems to be effective. It worth further studying and could be applied in different cases as well

### Weaknesses
- The methodology is essentially two unrelated parts, one on counting queries and another one on the retrieval algorithm. This makes the paper lose a focused contribution, and more essentially, the experiments also do not show how the two parts contribute to the performance
- The selection of baselines is insufficient. For example, there are quite a number of graph RAG approaches and also some prompt-based iterative RAG methods without graph.
- Because the method does not involve training, it would be expected to evaluate on more than just open-source models like Llama.

### Questions
Please see weaknesses above

### Soundness
2

### Presentation
3

### Contribution
2
