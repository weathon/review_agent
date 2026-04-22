# HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 4

## Abstract
Retrieval-Augmented Generation (RAG) enhances the response capabilities of language models by integrating external knowledge sources. However, document chunking as an important part of RAG system often lacks effective evaluation tools. This paper first analyzes why existing RAG evaluation benchmarks are inadequate for assessing document chunking quality, specifically due to evidence sparsity. Based on this conclusion, we propose HiCBench, which includes manually annotated multi-level document chunking points, synthesized evidence-dense question answer(QA) pairs, and their corresponding evidence sources. We also propose HiChunk, a hierarchical document structuring framework using fine-tuned LLMs and the Auto-Merge retrieval algorithm to enhance retrieval quality. Experiments demonstrate that HiCBench effectively evaluates the impact of different chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves better chunking quality within reasonable time consumption, thereby enhancing the overall performance of RAG systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study proposes the HiChunk framework, which employs a fine-tuned LLM to parse the hierarchical structure of documents, thereby overcoming the limitations of traditional linear chunking. The Auto-Merge retrieval algorithm is introduced, which can dynamically adjust the retrieval granularity based on queries and adaptively merge hierarchical chunks. The paper validates the necessity of evidence-intensive evaluation through HiCBench and demonstrates the advantages of the combination of HiChunk and Auto-Merge.

### Strengths
1. The paper identifies and defines the evidence sparsity issue in existing RAG benchmarks. When the answers to QA pairs rely solely on a few sporadic sentences in the document, existing benchmarks fail to effectively evaluate the performance of chunking strategies.
2. The experiments in Table 4 demonstrate the effectiveness of HiCBench. On traditional sparse benchmarks, the performance differences among various chunking methods are negligible, whereas on HiCBench, the advantages of HC200+AM are amplified.
3. The paper proves the superiority of HiChunk in multiple dimensions, including chunking accuracy, evidence recall rate, and RAG response quality.

### Weaknesses
1. Studies such as LongRefiner[1] have already proposed leveraging the hierarchical nature of documents to optimize RAG. The HiChunk framework presented in this paper shares similarities with such hierarchical document optimization approaches.
2. The merging decisions of the Auto-Merge algorithm rely on a series of sophisticated rules. However, the paper does not provide ablation experiments or sensitivity analyses for these critical design elements. This renders the algorithm akin to a product fine-tuned for specific tasks, and its robustness remains to be further verified.
3. Furthermore, the paper does not explore whether a LLM can achieve equally proficient hierarchical structure parsing solely through prompting. It is likely that the fine-tuned model is merely reproducing capabilities it already possesses, thus casting doubt on the necessity of fine-tuning.

**reference**

[1] Jin, Jiajie, et al. "Hierarchical document refinement for long-context retrieval-augmented generation." arXiv preprint arXiv:2505.10413 (2025).

### Questions
1. If HiChunk has already generated semantically complete text chunks, why not directly perform retrieval on these finest-grained semantic chunks?
2. Many real-world documents possess a depth structure far exceeding three levels. How should this situation be addressed?
3. During the training of the HiChunk model, has it failed to learn effective modeling and differentiation of deeper hierarchical structures beyond L3?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This study constructs a document structuring framework named HiChunk based on a fine-tuned LLM. This framework identifies chunking points and hierarchical levels within documents through text generation tasks. For hierarchical structures, the research devises an Auto-Merge retrieval algorithm that controls the upward merging of child nodes based on specific conditions, achieving query-based adaptive granularity of retrieved chunks. Experimental results indicate that evidence-intensive QA tasks in HiCBench are more suitable for evaluating chunking quality.

### Strengths
1. The paper points out the limitations of current RAG benchmarks in assessing chunking effectiveness, which stem from evidence sparsity. When the evidence relied upon by a query is confined to only a few sentences within a document, the benchmark loses its ability to distinguish the performance of different chunking methods.
2. Comparative experiments reveal that on sparse datasets such as GutenQA and OHRBench, the differences in metrics among various chunking strategies are minimal. In contrast, the evidence-intensive nature of HiCBench successfully identifies the improvements in recall rate and response quality brought about by high-quality chunking methods.
3. In the comprehensive evaluation of the RAG pipeline, HC200+AM maintains high response quality while also taking into account an acceptable time cost.

### Weaknesses
1. The training data for the model are all explicitly structured documents with clear paragraphs. The HiChunk model may overly rely on explicit formatting rather than semantics for hierarchical segmentation. Therefore, a more thorough explanation of the effectiveness and robustness of this method is required.
2. When conducting baseline comparisons, the paper only selects three baseline methods. It is necessary to include more baseline methods or conduct experiments solely using LLMs without fine-tuning.

### Questions
1. The paper mentions in Section 4 that the training set is enhanced by randomly shuffling document sections. However, given that different sections vary in length, would it be better to process the data following a curriculum learning approach that starts with shorter sections and progresses to longer ones?
2. At line 224, the authors propose a new concept, hierarchical drift, but lack detailed explanation and demonstration.

### Soundness
2

### Presentation
2

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
This paper presents a complete study to address the document chunking problem, which significantly affects the performance of RAG systems. It introduces a new benchmark, HiCBench, featuring diverse QA formats to evaluate the impact of chunking, and proposes a rule-based iterative approach, HiChunk, to alleviate issues resulted from chunked contexts. However, despite the claimed focus on document chunking, the work appears more related to studies on agentic RAG. The paper overlooks well-established document chunking in this area and shows limited technical novelty compared to existing methods. Moreover, the overall presentation quality requires a major revision before it can be considered suitable for formal academic publication.

### Strengths
- The paper presents a new benchmark, HiCBench, which is created to evaluate the impact of document chunking on RAG system performance.
- It also proposes a rule-based iterative approach aimed at optimizing the influence of chunking within RAG systems.

### Weaknesses
- The evaluation procedure of HiCBench is not clearly defined. It remains unclear what specific evaluation methods and performance metrics are used to assess the proposed approach.
- The paper does not provide a clear explanation of how data annotation for HiCBench was conducted. 
- It is unclear why a new question-answering dataset was created when several well-established benchmarks already exist, such as NarrativeQA, 2WikiQA, and MuSiQue.  
- The proposed iterative chunking and retrieval process appears to be entirely rule-based. The authors may consider leveraging agentic approaches recently explored in RAG systems [1,2].
    - [1] Search-o1: Agentic Search-Enhanced Large Reasoning Models
    - [2] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning 
- The paper does not discuss existing research on document chunking techniques, including recent innovations like late chunking and landmark embedding, which are relevant for long-context retrieval [3,4]. 
    - [3] Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models
    - [4] Landmark Embedding: A Chunking-Free Embedding Method For Retrieval Augmented Long-Context Large Language Models
- The manuscript suffers from numerous typos, grammatical errors, and misleading descriptions, which significantly affect readability and make the paper currently unsuitable for academic publication.

### Questions
Please see the comments provided above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors (1) propose a novel RAG benchmark with manually annotated multi-level document chunking points, synthesised evidence-dense QA pairs, together with their respective sources; and (2) introduce a novel framework for structuring the chunks based on fine-tuned LLMs and an auto-merge retrieval algorithm.

### Strengths
- Timely and practical insights for document chunking on information dense benchmark
- Authors introduce a novel benchmark, and a hierarchical structuring framework to dynamically adjust the granualrity of retrieval chunks 
- Multiple QA generation strategy
- Comprehensive comparison of methodologies

### Weaknesses
- Besides average word count and sentence, it would be have been interesting to quantify and compare information density between benchmarks
- The training and use of the fine-tuned model lacks explanation, how well do the training instructions generalise to different document types? 
- Quite complex writing with some redundancy, which could benefit from simplification

### Questions
/

### Soundness
3

### Presentation
3

### Contribution
2
