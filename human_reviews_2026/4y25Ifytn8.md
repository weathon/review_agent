# CFT-RAG: An Entity Tree Based Retrieval Augmented Generation Algorithm With Cuckoo Filter

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Although retrieval-augmented generation(RAG) significantly improves generation quality by retrieving external knowledge bases and integrating generated content, it faces computational efficiency bottlenecks, particularly in knowledge retrieval tasks involving hierarchical structures for Tree-RAG. This paper proposes a Tree-RAG acceleration method based on the improved Cuckoo Filter, which optimizes entity localization during the retrieval process to achieve significant performance improvements. Tree-RAG effectively organizes entities through the introduction of a hierarchical tree structure, while the Cuckoo Filter serves as an efficient data structure that supports rapid membership queries and dynamic updates. The experiment results demonstrate that our method is much faster than baseline methods while maintaining high levels of generative quality. For instance, our method is more than 800% faster than naive Tree-RAG on DART dataset. Our work is available at https://github.com/TUPYP7180/CFT-RAG-2025.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method named CFT-RAG, which aims to accelerate the entity retrieval process in Tree-RAG by introducing a Cuckoo Filter. The authors identify that traversing entity trees in a hierarchical knowledge base is the primary performance bottleneck for Tree-RAG. The core contribution is the application and enhancement of the Cuckoo Filter by adding a temperature variable to prioritize the retrieval of frequently used entities and using a block linked list to efficiently store entity locations.

### Strengths
The paper addresses a practical and important problem. Retrieval latency is a critical barrier to the real-world application of Tree-RAG. Besides,  the experimental results show that CFT-RAG achieves a significant speedup in retrieval compared to baselines, including a naive Tree-RAG and more complex ANN-based methods, while maintaining comparable answer accuracy. This provides strong experimental proof of its effectiveness.

### Weaknesses
1. Misrepresentation and unfair comparison of text-based RAG: the introduction claims that text-based RAG is slow due to complex language processing, which is misleading. In practice, Text-Based RAG leveraging efficient vector indexes (e.g., FAISS/HNSW) is often one of the fastest retrieval methods. Besides, the experimental setup is unfair. The reported retrieval times for Text-Based RAG are extremely long, suggesting that the authors likely used a brute-force search without any indexing. This is not a reasonable or competitive baseline, which undermines the paper's claims about its speed advantage.
2. The core of this work is based on an existing data structure: the Cuckoo Filter. The main contributions are incremental improvements rather than a new framework or a fundamental shift in the RAG paradigm. Therefore, the conceptual novelty of this work is relatively limited.
3. All experiments were conducted by selecting only 36 questions from each dataset. This sample size is too small and may lead to coincidental results, casting doubt on the statistical significance and generalizability of the findings.

### Questions
1. Your Text-Based RAG baseline appears to lack a vector index, causing it to be unusually slow. Standard RAG implementations typically use efficient indexes like FAISS, and their retrieval speed often surpasses that of Tree-RAG. Could you clarify the specific implementation of this baseline and explain why a standard indexed approach was not used for a fairer comparison?
2. Could you provide a computational cost analysis for the entity forest construction process (especially the relationship extraction)? How does this offline cost scale with the size of the knowledge base?

### Soundness
2

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
4

### Summary
CFT-RAG proposes an acceleration method for Tree-RAG by integrating an improved Cuckoo Filter into the retrieval process. The method optimizes entity localization in hierarchical tree structures through two novel designs: a temperature variable that prioritizes frequently accessed entities, and a block linked list that efficiently stores entity addresses. Experimental results on MedQA, AESLC, and DART datasets show that CFT-RAG achieves up to 800% faster retrieval than naive Tree-RAG while maintaining comparable accuracy.

### Strengths
1. The integration of Cuckoo Filter reduces retrieval time substantially (e.g., 1.78s vs. 15.88s on DART) while preserving accuracy, addressing a key bottleneck in Tree-RAG scalability.  
2. The temperature variable and block linked list optimize both temporal and spatial efficiency, enabling dynamic updates and reduced memory fragmentation without extra storage costs.

### Weaknesses
1. Limited Generalization to Non-Tree Structures: The method is tailored for hierarchical tree-based knowledge bases and may not directly apply to graph-based or unstructured data formats without modifications.  
2. Some key experimental settings are missing: What are the hyperparameters of the proposed method? Are the baselines you chose run in Python or C++? These settings are key to ensuring fair experiments.

### Questions
See the above weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the efficiency problem in tree-based RAG and enhances entity-based information retrieval using the Cuckoo filter. The Cuckoo filter is adapted for entity-based retrieval by sorting buckets according to the temperature of the block-linked list, where each block contains three addresses in the forest. This improved Cuckoo filter contributes to faster processing and better overall performance.

### Strengths
- The problem statement regarding the efficiency of RAG is reasonable.
- The introduction of a cuckoo filter equipped with bucket sorting and a block-linked list improves efficiency while enhancing accuracy.

### Weaknesses
- The main idea is to introduce the cuckoo filter into Tree-RAG, which limits its contribution to an engineering technique rather than a conceptual advance.
- The cuckoo filter can only be applied to questions containing multiple entities, raising concerns about its applicability to entity-poor questions.
- The baselines are limited, as the experiments were conducted with only simple baselines. More advanced text-based, graph-based, and tree-based RAG methods may be required to better demonstrate the effectiveness of the proposed method.
- The rationale for selecting benchmarks is missing. It is unclear whether the chosen benchmarks include questions rich in entities, which might make the experimental settings favorable to the proposed method.

### Questions
**Writing Suggestions**
- Citations are missing.
  - L38: "Enhancing the retrieval speed and accuracy of the knowledge base is crucial for boosting ~"
  - L45–L46: "Knowledge bases in RAG systems are mainly of three types: …"
  - L84: Cuckoo Filter, and L87: Bloom Filter
  - L152: No citation for Tree-RAG.
  - 4.1 Baselines 

**Duplicate Sentence**
- L115: "The main advantage of Cuckoo Filter is its support for dynamic updates.”

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
2

### Summary
This paper addresses the limitation of Tree-RAG that as the dataset and tree depth grow, the time required to locate and retrieve relevant entities within the hierarchical structure significantly increases. A cuckoo filter is interated to optimize Tree-RAG. The experimental shows that the proposed method reduces the retrieval time while maintaining the accuracy.

### Strengths
The result of the paper is promising. In the reported benchmark datasets, the retrieval time reduces significantly without sacrificing the accuracy.

### Weaknesses
The paper lacks detailed analysis on the impact of the introduced method. For example, what is the gains of the method compared to the T-Rag method as the dataset grows or reduces?
The paper is not theoretically novel. It is more like an engineering practice.

### Questions
Is there any limitations about the CFT-RAG compared to T-RAG?
How does the performance change when the dataset is much smaller or much larger?

### Soundness
2

### Presentation
3

### Contribution
3
