# Graph Tokenization for Bridging Graphs and Transformers

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
The success of large pretrained Transformers is closely tied to tokenizers, which convert raw input into discrete symbols. Extending these models to graph-structured data remains a significant challenge. 
In this work, we introduce a graph tokenization framework that generates sequential representations of graphs by combining reversible graph serialization, which preserves graph information, with Byte Pair Encoding (BPE), a widely adopted tokenizer in large language models (LLMs).
To better capture structural information, the graph serialization process is guided by global statistics of graph substructures, ensuring that frequently occurring substructures appear more often in the sequence and can be merged by BPE into meaningful tokens.
Empirical results demonstrate that the proposed tokenizer enables Transformers such as BERT to be directly applied to graph benchmarks without architectural modifications.
The proposed approach achieves state-of-the-art results on 14 benchmark datasets and frequently outperforms both graph neural networks and specialized graph transformers. This work bridges the gap between graph-structured data and the ecosystem of sequence models.
Our code is available at \href{ https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers }{\color{blue}here}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new graph tokenization framework that transforms graphs into discrete sequences through reversible traversal and BPE techniques. The proposed approach effectively preserves structural information, allowing standard Transformer models to process graph data and achieve competitive performance when compared to state-of-the-art graph-based models.

### Strengths
1. The paper presents a graph tokenization method that allows standard Transformers to process graph data without altering the model architecture.
2. The proposed frequency-guided serialization strategy establishes a bidirectional mapping between graphs and sequences, ensuring the reversibility and determinism.
3. The paper conducts the experiments on 12 benchmark datasets and mostly outperforms both graph neural networks and graph transformers.

### Weaknesses
1. A key strength of this approach is its proposed reversible graph tokenization method. However, the experiments predominantly emphasize the encoding aspect, with limited exploration of its reversibility.
2. Lacking the discussion or comparison with strong baselines and relevant works. Several existing graph serialization studies, such as those based on chemical rules [1], ring/functional group approaches [2], and models towards the order of token-sequence [3-4]. A detailed comparison and discussion of the advantages of the proposed method would significantly strengthen the paper’s contribution.
3. Inadequate justification and analysis of the subgraph frequency heuristic.

[1]. Advancing Molecular Graph-Text Pre-training via Fine-grained Alignment, KDD 2025.
[2]. Expressivity and Generalization: Fragment-Biases for Molecular GNNs, ICML 2024.
[3]. A Graph is Worth K Words: Euclideanizing Graph using Pure Transformer, ICML 2024.
[4]. GraphGPT: Graph Learning with generative Pre-trained Transformers, arXiv:2401.00529.

### Questions
1. Could the authors provide more details on how each token in the vocabulary is embedded in the model?
2. In previous work, such as PS-VAE [5], a graph-structured tokenization method similar to BPE that merges subgraphs is used. Could the authors clarify how the proposed approach differs from PS-VAE?
3. For graph-structured datasets, could the authors discuss the impact of subgraph vocabulary size on model training and experimental performance?

[5]. Molecule Generation by Principal Subgraph Mining and Assembling, NeurIPS 2022.

### Soundness
3

### Presentation
3

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
This paper proposes a graph tokenization framework that converts graphs into discrete token sequences by combining reversible graph serialization (frequency-guided Eulerian circuits or Chinese Postman Problem) with Byte Pair Encoding (BPE). The tokenized sequences aim to enable standard off-the-shelf Transformers (BERT, GTE) to process graph-structured data.

### Strengths
Approach: The idea of combining reversible graph serialization with BPE, with formal analysis showing why existing methods fail to satisfy both reversibility and determinism is interesting.

Empirical Performance: The method achieves state-of-the-art results on 12 out of 13 benchmarks and it also demonstrated that properly tokenized graphs enable standard Transformers to outperform specialized graph architectures without any modifications.

Efficiency Gains: BPE compression achieves 6-10× reduction in sequence length with 2-3× training speedup on Figure 2.

Interpretability: The learned vocabulary on Figure 3 shows that BPE discovers chemically meaningful substructures like benzene rings and functional groups, it seems it provides interpretable tokens aligned with domain knowledge.

### Weaknesses
Scope: The paper only evaluates graph-level classification and regression tasks. Despite claiming to "bridge graphs and transformers," node-level and edge-level prediction tasks are entirely absent from evaluation. The authors acknowledge this limitation (Appendix A.1) but provide no experimental validation of their proposed solutions. This significantly undermines the claim of providing a general framework for graph learning. 

Continuous Feature Problem: The framework fundamentally requires discrete labels (mostly just on chemical domain), making it incompatible with graphs that have continuous node/edge features without lossy quantization. This is a critical limitation since many real-world graphs (social networks, knowledge graphs, citation networks) have rich continuous attributes. The authors acknowledge this conflicts with their "faithful and reversible representation" principle but offer only speculative solutions without validation.

Scalability: While the paper discusses O(|E|) complexity for Feuler serialization, there's no evaluation on truly large-scale graphs. The largest graphs tested have ~284 nodes on average (DD dataset). No experiments on OGB-scale datasets (millions of nodes/edges) or analysis of how the method scales. The Transformer's fixed context window limitation is mentioned but not empirically investigated.

Insufficient Baselines: The baseline comparison omits several recent and relevant methods: (1) recent graph transformers like GraphGT, Graphormer, and NAGphormer are not compared; (2) other recent serialization-based methods beyond GraphMamba such as GPatcher; (3) graph LLM models like G-Retriever or InstructGLM that also aim to connect graphs with language models. The claimed state-of-the-art status is therefore difficult to fully verify.  The paper incorrectly positions GCN (Kipf, 2016) as the GNN foundation, overlooking earlier spectral GNN methods and the rich history of graph neural networks. Important categories like heterophilic GNNs are completely absent despite being highly relevant for understanding when message passing succeeds or fails, this is a critical consideration for evaluating serialization-based alternatives (not to mention some graph partition based methods like Graph-ViT-MLPMixer).

Limited Ablation Studies: Several design choices lack thorough ablation: (1) the number of BPE merges K (fixed at 2000) has no sensitivity analysis; (2) the frequency guidance mechanism's individual components are not ablated; (3) why edge-level patterns (triplets) specifically, rather than larger motifs? (4) The comparison with G2T-FM methods (mentioned in related work) is absent from experiments despite being highly relevant.

No Analysis of Failure Cases: The method performs poorly on COLORS-3 (Table 8) where node-counting is required, this exposes fundamental limitations of edge-traversal approaches. However, there's no systematic analysis of when and why the method fails. The paper would benefit from characterizing graph properties (size, density, label distribution) that predict success or failure.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a novel graph tokenization scheme designed to provide graphs with discrete features as input to standard Transformers. The method involves two main steps: 
* First, it employs frequency-guided Eulerian Circuit and Chinese Postman Problem (CPP) approaches to achieve reversible and deterministic serialization of the input graph. 
* Second, it uses Byte Pair Encoding (BPE)—a standard in LLMs—to greedily combine frequently occurring token substrings. 

The authors validate these results by combining their tokenizer with bidirectional BERT and GTE Transformers, comparing their performance against various GNNs and demonstrating the efficacy of BPE in terms of sequence compression.

### Strengths
The proposed approach is elegant and cleanly integrates graph inputs with established LLM best practices. The combination of BPE with graph walks is clever, well-motivated, and useful for inputting graphs directly into Transformers, effectively sidestepping the need for specialized GNN architectures.

The experiments appear strong and thorough. They demonstrate compelling compression rates from BPE and show that the method beats a variety of GNN models, although I note that I lack full context for this specific research area.

The authors are direct about the limitations of their work, specifically regarding continuous features and graph-level tasks, and they suggest reasonable future steps.

### Weaknesses
Regarding reversibility: Output sequences consist only of standard labels (Eq. 1), not unique node identifiers. In graphs with many identically labeled nodes (e.g., large carbon lattices), how does the decoder $f^{-1}$ explicitly distinguish between returning to a previously visited node versus arriving at a new node with the same label? Is reversibility guaranteed for all labeled graphs without positional markers? For one particular example, what structure can be accurately recovered from the molecular path in Figure 1?

The serialization example in Figure 1 is confusing. The numbering is difficult to follow (especially as it does not appear to start with 1 as the first edge), and it is unclear why the right-most 'C' appears after 'Br' and 'Cl' in the sequence, given the labeling.

### Questions
While the paper mentions generative tasks via GPT-style models as an application, it only evaluates on classification/regression using BERT/GTE. Would pairing the tokenization scheme with autoregressive transformers be able to solve similar supervised graph learning tasks as well (given their general-purpose successes in language modeling)?

It would be interesting to better understand the BPE patterns, lengths, and statistics. Does the learned vocabulary consist mostly of small node/edge tokens, or are there many longer sequence tokens?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposed GraphTokenizer, which essentially just tokenizes the graphs s.t. they can be easily consumed by Transformer based models. The method contains two steps, where the authors first use a deterministic and reversible (in opposite to random walks) sterilization method to transfer the graph data into corpus of sequences, and then use BPE to tokenize them. Once the graphs are tokenized, the following becomes natural as in language modeling. The proposed method performed well across many evaluated benchmarks.

### Strengths
1. While the high level idea is not new, the proposed work executed the graph tokenization fairly well, and showcased good performance in the tasks that it can handle.
2. In the evaluations, the proposed method showed good performances as well as efficiency.

### Weaknesses
The limitations on page 14 are very on point. 

1. As limitations 1&2 mentioned, the focus on only graph-level tasks with discrete features significantly constrains the scope of this work. With such constrains, the proposed method almost only makes sense on protein and chemical graphs, where nodes are atoms, etc.
2. All three limitations kinda showed a theme that this proposed work might not be very suitable for larger graphs such as social networks.

### Questions
n/a

### Soundness
3

### Presentation
2

### Contribution
2
