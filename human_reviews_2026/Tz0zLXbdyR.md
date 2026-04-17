# Dynamic Retrieval AugmentedGeneration Based on The Knowledge-Aware of Large Language Models

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Retrieval-augmented generation (RAG) has proven effective in enhancing open-domain question answering by supplementing language models with external knowledge. However, current approaches often rely heavily on the model’s internal confidence scores to decide whether retrieval is necessary. This overreliance, coupled with the tendency of language models to show overconfidence, results in excessive and sometimes redundant retrieval operations. Moreover, conventional RAG workflows commonly incorporate coarse-grained retrieved documents directly into the generation process, introducing noise and semantic drift that can compromise answer quality. To address these limitations, we propose DynaRAG, a dynamic knowledge-aware framework built on a three-stage optimisation strategy. First, a hybrid question-knowledge similarity space and a lightweight threshold prediction network are constructed that learns query-adaptive decision boundaries to control retrieval triggering more precisely. Second, we generate multigranularity semantic variants to perform targeted retrieval and rank documents using a newly introduced knowledge importance scoring mechanism, thus improving the relevance and specificity of the retrieved content. Third, a prompt-guided large language model synthesises the final answer based on the refined input of selected knowledge. Extensive experiments demonstrate that DynaRAG achieves average improvements of approximately 11% in EM and 14% in F1 on six representative QA benchmarks. Evaluated against a diverse suite of retrieval-augmented baselines, DynaRAG consistently improves accuracy and efficiency, underscoring its robustness and adaptability in knowledge-intensive tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DynaRAG, a dynamic retrieval-augmented generation framework. The system first predicts whether external retrieval is necessary through a thresholding module, then performs multi-granularity query reformulation and document ranking to select useful knowledge before feeding it to the generator. Experiments on multiple QA benchmarks show improved EM and F1 scores, and the authors claim the framework reduces redundant retrieval while improving answer quality.

### Strengths
- The paper is clearly written and the modular pipeline is easy to follow.

- It evaluates multiple RAG baselines across several QA datasets, and the empirical improvements are consistent.

- The motivation—avoiding redundant retrieval and improving knowledge selection—is relevant to the RAG community.

### Weaknesses
- Novelty is insufficient, as similar dynamic RAG ideas have already appeared in works like Self-RAG, SKR, and Adaptive-RAG.

- Modern slow-thinking and tool-use LLMs can already perform autonomous dynamic retrieval, making the proposed pipeline unnecessary in practice.

- The computational overhead and latency of the multi-stage design are not quantified, so its real benefit over simpler alternatives is unclear.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The DynaRAG framework proposed in this paper innovatively designs a three-stage dynamic optimization mechanism of "Knowledge Reliability Evaluation (KRE) - Knowledge Distillation Enhancement (KDE) - Retrieval Enhancement Generation" to address the two core issues of "retrieval triggered over reliance model confidence" and "introduction of noise into coarse-grained documents" in traditional RAG systems.

### Strengths
The KRE module breaks through the limitations of traditional RAG's "static threshold" or "single confidence judgment", and achieves query adaptive retrieval decision-making through "mixed similarity calculation (cosine similarity+KL divergence)+lightweight threshold prediction network"

### Weaknesses
1. The relevant work section did not thoroughly compare the core differences between DynaRAG and these works, making it difficult to highlight the positioning of the method in the optimization of the entire RAG process.

2. Verified only on the Wikipedia knowledge base and not extended to domain specific knowledge bases such as medical PubMed and financial FiQA document libraries, it is not possible to verify the applicability of DynaRAG in domain specific scenarios.

3. One of the core pain points of traditional RAG is "latency caused by redundant retrieval", while DynaRAG claims to "reduce redundant retrieval", but does not quantify the "percentage reduction in retrieval times" or "end-to-end inference time comparison". The additional computational cost caused by the generation of query variants in KDE modules has not been analyzed.

4. Suggest adding hallucination assessment experiments (such as using FaithDial tool or LLM-as-a-study to determine factual consistency), and designing specialized tests for the "conflict document retrieval" scenario to verify the anti-interference ability of the method.

### Questions
Will the differences in attention mechanisms among different LLMs (such as Llama and GPT series) affect the reliability of scoring?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to enhance LLM performance in question answering (QA) with retrieval-augmented generation (RAG) and introduces DynaRAG, a dynamic, knowledge-aware framework based on a three-stage optimization strategy. Specifically, the Knowledge Reliability Evaluator (KRE) adaptively controls retrieval by integrating LLM embeddings with a joint question–document similarity space. The Knowledge Distillation Enhancer (KDE) further refines retrieval quality through multi-granularity query reformulation and cross-document re-ranking before passing results to the LLM for generation. Experiments on single-hop, multi-hop, and knowledge-intensive QA benchmarks demonstrate higher accuracy and reduced redundant retrieval.

### Strengths
1. The motivation and idea of improving the integration of externally retrieved knowledge with LLMs in RAG are good and interesting. Indeed, most existing solutions in RAG simply concatenate the retrieved knowledge with the input and send it to LLMs in a shallow way, which may not fully leverage the knowledge and can lead to errors. 
2. The experiments and ablation studies look thorough and can demonstrate the design choices of each component in the proposed method (i.e., KRE, KDE)

### Weaknesses
1. **Writing Quality and Consistency**
    
    The overall writing of the paper requires significant improvement. There are numerous typos, inconsistent terminology, and unclear expressions. For example:
    
    - The term *KRE* is referred to as **Knowledge Reliability Evaluator** throughout the paper, but at **Line 159**, it is called **Knowledge Reliability Estimator**.
    - Several notations and symbols used in the equations are not defined, leading to confusion. For instance, in **Eq. (1)** , in and **Eq. (3)**, the term T_{temp} is not explained.
    - There are multiple typographical and formatting issues, including missing spaces and inconsistent variable usage (e.g., **Lines 226, 229, 234, 255**, **Eq. (8)**, and **Line 293**). A careful proofreading is necessary to ensure clarity and consistency.
2. **Lack of Conceptual and Technical Clarity**
    
    The paper lacks clear explanations of several key technical concepts and design choices, which may confuse readers. For example:
    
    1. At **Line 107**, the distinction between *query-based* and *inference-based* methods is unclear. Based on the current description, the two approaches appear nearly identical.
    2. In **KRE**, it is not explained where the *candidate documents* originate. How are these documents initially retrieved before KRE evaluates the necessity of retrieval? Moreover, there is a mismatch between the textual description and **Figure 1 (Stage 1)**, as the figure does not mention any retrieval process.
    3. The paper does not describe how the *adaptive threshold prediction network* is trained or evaluated. What are its performance metrics? **Line 198** mentions that annotations were created to train this network, but it remains unclear how numerical values in the range [0,1][0,1][0,1] were assigned to represent a query’s dependency on external knowledge.
    4. The choice of **BERT** for embedding extraction is questionable, considering that more recent embedding models outperform BERT in both accuracy and efficiency. The rationale for this choice should be justified.
3. **Experimental Evaluation and Efficiency Analysis**
    
    Although the paper provides several ablation studies, it lacks detailed evaluation and comparison regarding the performance of *KRE’s retrieval necessity prediction*. In addition, given the proposed three-stage pipeline, an efficiency analysis compared to existing methods is missing. Specifically, the paper should report:
    
    - How many retrieval steps are reduced through KRE’s adaptive mechanism?
    - What is the overall inference time compared to baseline or prior RAG systems?

### Questions
Please see above

### Soundness
2

### Presentation
2

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
The paper presents Dynamic RAG, which consists of two main modules: (1) a Knowledge Reliability Evaluator (KRE) that decides whether retrieval should be triggered based on the similarity between internal and external knowledge (while this part is not very clear), and (2) a Knowledge Distillation Enhancer (KDE) that re-ranks retrieved documents using pooled token-level entropy and token-level attention scores. The re-ranked documents are then incorporated into the prompt, and the LLM generates an answer. Experimental results on various benchmarks demonstrate that the proposed RAG framework with KRE and KDE outperforms baseline methods.

### Strengths
1.	The two proposed modules, KRE and KDE, for enhancing RAG are well motivated, novel, and technically interesting.
2.	The experimental results demonstrate that the proposed Dynamic RAG achieves improved performance over the baseline methods.

### Weaknesses
1.	The presentation of the technical content is not very clear. In particular, although the similarity metric is computed in Eq. (1), it is unclear when a retrieved document d_j becomes available at that point. Some notation definitions are also missing. Moreover, in Eq. (1), the divergence and similarity metrics are simply added, even though they represent opposite relationships.
2.	The technical novelty is not entirely clear. In the KDE module, attention-based re-ranking has been explored in prior studies, but the paper does not sufficiently discuss how the proposed approach differs from or improves upon existing advanced re-ranking methods.
3.	Although the experiments show improved performance over the baselines, it remains unclear whether the proposed methods are compared against state-of-the-art approaches.

### Questions
1.	What are the definitions of h^int_q and h^ext_d_j?

### Soundness
2

### Presentation
2

### Contribution
2
