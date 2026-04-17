# Scaling Knowledge Graph Construction through Synthetic Data Generation and Distillation

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Document-level knowledge graph (KG) construction faces a fundamental scaling challenge: existing methods either rely on expensive large language models (LLMs), making them economically nonviable for large-scale corpora, or employ smaller models that produce incomplete and inconsistent graphs. We find that this limitation stems not from model capabilities but from insufficient training on high-quality document-level KG data. To address this gap, we introduce SynthKG, a multi-step data synthesis pipeline that generates high-quality document-KG pairs through systematic chunking, decontextualization, and structured extraction using LLMs. By fine-tuning a smaller LLM on synthesized document-KG pairs, we streamline the multi-step process into a single-step KG generation approach called Distill-SynthKG. Furthermore, we repurpose existing question-answering datasets to construct KG evaluation datasets and introduce new evaluation metrics. Using KGs produced by Distill-SynthKG, we also design a novel graph-based retrieval framework for RAG. Experimental results demonstrate that Distill-SynthKG not only surpasses all baseline models in KG quality (including models up to eight times larger) but also consistently improves in retrieval and question-answering tasks. Additionally, our proposed graph retrieval framework outperforms all KG-retrieval methods across multiple benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a data-centric pipeline, SynthKG, to construct document-level, ontology-free KGs via (i) sentence-boundary chunking, (ii) decontextualization to enforce consistent entity mentions across chunks, and (iii) LLM-based extraction of entities, propositions, and triplets. It then distills the multi-step pipeline into a smaller LLM, Distill-SynthKG (D-SynthKG-8B), capable of single-step KG generation for a whole document. The authors also repurpose multi-hop QA datasets to create proxy triplets and propose semantic coverage metrics for KG evaluation, and design a Proposition–Entity bipartite graph retriever with optional LLM re-ranking for RAG. Experiments on MuSiQue, 2WikiMultiHopQA, and HotpotQA show higher KG coverage and competitive or better retrieval/QA versus larger baselines and GPT-4o KGs, while using a much smaller model.

### Strengths
S1. One-step distilled model (8B) achieves large-model quality on KG coverage/retrieval, improving efficiency (fewer LLM calls).

S2. Proposition–Entity graph retriever with LLM re-ranking shows consistent gains and strong comparisons to dense retrieval and prior KG-RAG systems.

S3. Data-centric pipeline that converts a fragile prompt-only process into learnable patterns; propositions act as interpretable, retrievable units.

### Weaknesses
W.1 Decontextualization locality assumption (preceding-chunk context suffices) is empirically supported but may break on long-range coreference or cross-section entity normalization (e.g., legal/biomedical corpora). A stress-test on documents with longer entity re-mention distances would strengthen generality claims.

W2. How robust is decontextualization when entities are not re-mentioned locally (e.g., reappear after several pages)? Any fallback using document-level coreference?

W3. For evaluation via proxy triplets, how do you control GPT-4o style drift (atomic vs abstractive propositions) that may advantage a particular retriever? Any normalization?

W4. In Graph+LLM, how sensitive are results to M (initial propositions), N-hop, and K (final chunks)? Please provide stability ranges / guidance.

w5. Please include more recent GraphRAG baselines in recent years.

### Questions
Please see my above w1-w5 for details.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SynthKG, a data synthesis pipeline that creates high-quality document-KG pairs to address the lack of training data for document-level KG construction. They then fine-tune a smaller LLM on this synthesized data, resulting in Distill-SynthKG, which excels at single-step KG generation. Distill-SynthKG significantly outperforms existing models in KG quality and downstream tasks like retrieval and question-answering. The authors also propose new KG evaluation methods and a novel graph-based retrieval framework for RAG. While this paper introduces SynthKG for data synthesis and Distill-SynthKG for efficient KG generation with promising results, its core innovation appears limited to leveraging LLMs for decontextualization. Furthermore, the experimental validation is insufficient, lacking crucial details on the Distill-SynthKG fine-tuning process, an ablation study on decontextualization's impact, and comprehensive comparisons with a wider range of LLMs and advanced methods, thus undermining the robustness of its claims.

### Strengths
Please see summary

### Weaknesses
1. The description of how Distill-SynthKG is fine-tuned from a large language model is insufficiently detailed. Crucially, the specific inputs and outputs of this fine-tuning process are not clearly defined.

2. Decontextualization is presented as a core component of SynthKG, yet the paper lacks an ablation study demonstrating its specific impact on performance. The absence of such an analysis makes it difficult to assess its contribution.

3. The experimental section omits comparisons with a broader range of contemporary large language models, limiting the generalizability and competitive standing of the proposed method.

4. The current experimental setup and results do not provide robust or comprehensive evidence to fully substantiate the paper's claims and conclusions.

5. The overall innovation presented in this paper appears limited, primarily leveraging existing large language models for a decontextualization operation rather than introducing fundamentally new paradigms.

### Questions
1. Could you elaborate on the fine-tuning process for Distill-SynthKG, specifically detailing the inputs provided to the large language model and the expected outputs during this phase?

2. Given that decontextualization is a central tenet of SynthKG, what is the observed impact on performance when this step is removed or altered? A detailed analysis of this would be highly beneficial.

3. The current comparisons are primarily against unprocessed large language models. How does your method fare against more advanced, state-of-the-art approaches in various tasks? For instance, what are the comparative results for QA tasks with different established methods, and what is the performance impact when decontextualization is removed from your proposed method in these contexts?

4. Please clarify the fundamental distinctions between your proposed approach and methods relying on sophisticated prompting techniques. Could comparable performance be achieved solely through advanced prompting strategies without the multi-step SynthKG pipeline?

### Soundness
2

### Presentation
2

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
This paper tackles the scalability bottleneck in document-level, ontology-free knowledge graph (KG) construction for retrieval-augmented generation (RAG) systems. Current approaches either rely on costly large language models (LLMs) or use smaller models that yield incomplete and inconsistent KGs—limitations attributed to a lack of high-quality training data rather than intrinsic model capabilities. To address this, the authors propose SynthKG, a multi-step pipeline for data synthesis that creates high-quality document-KG pairs through chunking, decontextualization, and structured extraction with LLMs. They introduce Distill-SynthKG, a smaller LLM fine-tuned on this synthetic data, capable of end-to-end document-level KG generation in a single step. The authors also present a framework for evaluating KGs, repurposing multi-hop QA datasets as KG evaluation benchmarks, and introduce semantic matching metrics designed for ontology-free KGs. Furthermore, they propose a novel graph-based retrieval methodology utilizing propositions as rich semantic retrieval units within the KG.

### Strengths
- The writing and figures in the paper are clear.
- The discussion around knowledge graph (KG) construction in the context of GraphRAG is meaningful, especially considering the high API cost of current LLM-based construction methods.
- The proposal to fine-tune a smaller LLM for accurate and efficient KG construction is novel.

### Weaknesses
- The main concern is the potential accumulation of errors throughout the pipeline. In the proposed method, an LLM is first prompted to extract entities, relations, and relevant propositions to construct the knowledge graph. This process is used to distill and fine-tune a smaller model. However, the data used for distillation is not from ground truth, but rather generated by the initial LLM, which may contain errors. As the smaller model is then trained on this potentially noisy data, it may inherit these biases and inaccuracies.
- The generalization ability of the trained KG construction model in real-world scenarios is uncertain. Since the training relies solely on the IndustryCorpus dataset, there is no guarantee that the model will generalize well to new domains. As KG construction is a key part of the GraphRAG pipeline, it is important to discuss the model’s generalizability to out-of-domain scenarios.
- The experiments do not include a comparison with more representative baselines for KG construction, such as HIPPORAG2 [1].

[1] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models

### Questions
Refer to the Weaknesses section above.

### Soundness
2

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
The paper introduces SynthKG, a multi-step data synthesis pipeline that creates high-quality document–KG pairs via document chunking, decontextualization, and LLM-based structured extraction of entities, relations, propositions.   The authors repurpose multi-hop QA datasets to build KG evaluation datasets and propose coverage metrics (semantic similarity and keyword-based) for KG assessment. Based on KGs produced by Distill-SynthKG, they design a graph-based retrieval framework that progressively retrieves propositions, related triplets, and text chunks leveraging the graph structure for RAG. Experiments show Distill-SynthKG surpasses baselines.

### Strengths
S1: Pipeline-level clarity makes the approach reproducible with standard components.

S2: Empirical emphasis on compositional queries aligns with KG advantages over pure vector search.

### Weaknesses
W1: Distillation objectives and training signals need precise specification; optimization of edge weights/pruning thresholds should be detailed and justified.

W2: The mechanisms for error correction and robustness are underexplored.

W3: Limited failure-mode analysis (rare entities, emergent relations, long-tail concepts) and cross-domain robustness.

### Questions
Q1: How are edge weights defined and optimized during distillation? Are they learned with retrieval feedback or heuristic scores? 

Q2: Does the system include cross-document coreference and alias normalization? How does this impact graph compactness and recall?

Q3: How are temporal or conditional relations handled (time-scoped facts, event causality) to avoid misleading retrieval when collapsing during distillation?

Q4: What are the observed trade-off curves across corpora sizes and domains? How stable are results under parameter changes?

Q5: How does the distilled KG integrate with text retrieval and rerankers? Are there benchmarks showing gains over strong vector-only baselines under equal resources?

### Soundness
2

### Presentation
3

### Contribution
2
