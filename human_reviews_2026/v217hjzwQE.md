# LOGOS: Precision Retrieval via Logical Document Graphs for Retrieval-Augmented Generation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) systems struggle with long documents because conventional retrieval methods provide noisy, page-level context that degrades generation quality. These methods are fundamentally limited by treating documents as a linear sequence of pages, which breaks the crucial logical dependencies—like tables, paragraphs, or references—that span across page boundaries. To overcome this limitation, we propose Precision Retrieval via Logical Document Graphs for Retrieval-Augmented Generation (LOGOS), a new RAG method that achieves precision retrieval by modeling a document's intrinsic logical structure. LOGOS transforms a document into a graph where semantic regions are nodes and logical connections are edges, effectively bridging page breaks. A Graph Neural Network then generates fine-grained, context-aware representations for each node, enabling a more concise and semantically relevant context for the generator. Extensive experiments on the ViDoRe and MMDOCIR benchmarks show that LOGOS sets a new state-of-the-art, significantly outperforming strong baselines by up to 2\%  in average Recall@1.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LOGOS, a RAG framework designed for long multimodal documents. It addresses the issue of context fragmentation caused by linear chunking. The core innovation is to model documents as logical graphs where semantic regions are nodes and their relationships are edges. A Graph Neural Network (GNN) is then used to generate context-aware embeddings for each node, aiming for more precise retrieval. The method is evaluated on retrieval benchmarks, where it outperforms recent state-of-the-art models.

### Strengths
The conceptual shift from treating documents as linear sequences to structured graphs is highly original and significant. This approach directly tackles a fundamental limitation in how RAG systems process complex, real-world documents. Besides, the methodology is robust, leveraging components for layout analysis and graph representation learning. The ablation and sensitivity analyses are thorough and provide strong evidence for the contribution of each component. Moreover, the paper is well-written, and the core idea is presented clearly with effective visualizations.

### Weaknesses
1. The paper targets to RAG, but the experiments are exclusively focused on retrieval metrics like Recall@K. This represents a significant gap between the paper's claims and its empirical validation.
2. The necessity of the complex GNN-based information fusion is not fully justified. A simpler, more intuitive baseline seems plausible: one could group semantically related text blocks, link them to the images/tables they explicitly reference, and then index only the text embeddings. Retrieval would be performed on text, with linked visuals retrieved alongside. The paper fails to argue why the GNN's expensive message-passing mechanism is superior to such a heuristic-based linking approach.
3. The "w/o GNN" ablation is the most critical one for assessing the core contribution, yet its exact configuration is not described. Without knowing how the embeddings are generated in this variant, it's difficult to interpret the reported 10.1-point performance drop and fairly evaluate the GNN's role.

### Questions
1. Could you please clarify the experimental setup for the "w/o GNN" ablation study? 
2. Why did you choose to evaluate only on retrieval benchmarks instead of an end-to-end generation task like question answering? Could you provide any results demonstrating that LOGOS's superior retrieval leads to more accurate or factual generated outputs?

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
3

### Summary
This paper proposes LOGOS, a framework for Retrieval-Augmented Generation (RAG) targeting long and complex documents. Instead of typical chunks, LOGOS decomposes documents into fine-grained semantic regions and constructs a cross-page heterogeneous graph, embedded by GNN. Experiments on two benchmarks demonstrate Recall@K performance improvement over baselines. Ablation study and sensitivity analysis are conducted to verify the contributions of main components and the impacts of hyperparameters.

### Strengths
- Clear motivation for the noise caused by a broken logical structure in long documents
- Reasonable design and exploitation of heterogeneous graph structure to reflect cross-page semantic relationships
- Empirical experiment results on two benchmark datasets, where the proposed framework consistently outperforms baselines

### Weaknesses
- A motivating example is not directly addressing the core idea of this work
    - Figure 1 mainly illustrates how excessive context could cause a performance drop at some point, but it does not show the direct necessity for logical-structure modeling, which weakens the empirical justification of the motivation.
    - The quality and clarity of Figure 2 are low, which makes it difficult to interpret the pipeline details
- There is a lack of sufficient discussion on existing graph-based document structuration and RAG approaches
    - Connecting semantic components as a graph and fusing them through GNN seems reasonable, but it appears conceptually straightforward, too
    - It would be encouraged to specifically clarify the explicit challenges in extracting semantic components, connecting them to construct a graph, or fusing this information to be effective for RAG
    - Related work focuses briefly on benchmarks, without including a sufficient methodological comparison with existing efforts
- Methodological design needs more improvement and justification
    - Overall performance improvements are marginal; given no additional context of task difficulties in the relevant tasks and datasets, it is hard to justify the substantial engineering additions used in the framework
    - No qualitative analysis was provided to specifically illustrate how the logical graph benefits retrieval performance compared with existing approaches
    - Edge linking strategies seem to rely mostly on heuristics, and there is a lack of analysis on which edge types contribute most

### Questions
Please see the weakness

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
3

### Summary
The authors look at the challenge of retrieval from large documents having cross references, images , tables running across pages. They use a VGT transformer to segment the pages and then use a semi-heurestic technique (but reasonable) to map the components into a graph and then build graph embeddings out of it. This leads to better performance than other methods.

### Strengths
The problem is an important problem and the basis of their solution around documents having a graph structure is intuititve. Also their graph formulation is heurestic driven (e.g. spatial adjancency / cross-page continuation) but perfectly reasonable - whilst edge cases may happen it should cover most real world cases. 

Their results are clearly better (numerically if not statistically) in most cases than baseline models.

### Weaknesses
The paper provides a good experimental setup for a real problem however it is not clear how the setup is being trained and why such a training objective will lead to better results.  

Specific comments below:-

1. Semantic similarity (211-213) - how do the authors propose to compute semantic similarity across modalities. 
2. Explicit reference - it is not clear as to how regex based search is scalable for a large dataset.
3. I did not follow the loss function in the section 3.3 . How is this trained?
4. I am confused on the online querying part. The first issue is the multimodal nature of the content but the query being most often text. But more importantly what is not clear how a text query will lead to a better mapping if the resultant vectors are GNN-enhanced. Suppose for example, the author gives an exact text it will not retrieve that since the GNN- enhancement will have rotated (and potentially scaled - no comments are there on normalization) the vectors in the db
5. Although the authors provide results on two datasets but it would have been better if they had provided some examples (even as supplementary material) on the dataset and queries as well as some error analysis results.
6. Table 1 and 2 should have statistical significance analysis - at least is the proposed model statistically better than the second best performing model in each case? 
7. The authors should have done a more comprehensive literature survey on the use of graphs for document structure in multimodal retrieval. 

Minor comment:-

1. Line 106 Vidore/MMDOCIR benchmark needs a citation where it is first referred to outside of the abstract.
2. A very special case so can be considered minor - the reading order mentioned in lines 206-211 is not valid for two-column documents like academic papers.  Some books also use this format.

### Questions
please address the concerns in the weaknesses section

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
4

### Summary
The paper proposes LOGOS, a precision retrieval method for long-document RAG. It segments pages into semantic regions (paragraphs, tables, images), converts the document into a cross-page heterogeneous graph, uses a shared text encoder (with VLM-generated textual summaries for visual regions) plus an R-GCN to produce context-aware node embeddings, and performs node-level retrieval. It has achieved strong performance on ViDoRe and MMDocIR.

### Strengths
1. Captures an important problem in RAG: content across pages is related and should not be encoded independently.

2. Method is reasonable and well-motivated; reported performance is strong.

3. Ablation study shows the improvement contributed by each component.

### Weaknesses
1. Building graphs over documents is well studied in RAG (e.g., GraphRAG and its follow-ups), but the paper lacks comparison and discussion in this area. Comparisons are needed to justify the novelty of this work.

2. The text encoder is under-specified.

3. The paper claims that small-snippet chunking addresses page-level chunking issues, but this is not directly tested. It is recommended to add a page-level baseline (e.g., concatenate the content of each page and input it as long text to the text encoder).

4. Missing related work like  “Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models,” which also focuses on building connections between chunks.

### Questions
1. Which text encoder is used?

2. Is the ground truth at the page level? If so, how is the retrieved chunk mapped to page-level retrieval?

### Soundness
3

### Presentation
2

### Contribution
2
