## Human Reviewer 1

### Summary
the paper proposes a graph tokenizer that utilizes multi-task graph self-supervised objectives to train a graph encoder, which can capture local interaction and have memory efficiency

### Strengths
1. paper is well written and easy to follow
2. experiment demonstrate strong results compared with baseline methods.
3. motivation and methodology is clear

### Weaknesses
NO

### Questions
NO

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes Graph Quantized Tokenizer (GQT) that considers three key properties such as local interactions, memory efficiency, and robustness & generalization for graph Transformer. Different existing works that train tokenizer and Transformer architecture in an end-to-end manner, GQA decouples the training tokenizer from the Transformer. In other words, it first trains the tokenizer and then learns graph Transformer architecture given the tokenized graph. From the experimental results, GQA shows the best performance compared to other baselines on 7 out of 8 datasets. It also achieves the best performance on large-scale datsets.

### Strengths
- The design of tokenizers in graph Transformers is really an important research topic. The paper well describes the existing works of graph Transformers and tokenizers in the related works.
- The paper is easy to follow.
- The experiments of the paper demonstrate that the proposed method is effective with multiple experiments including homophilious, heterophilious and large-scale datasets.

### Weaknesses
- I think that the novelty of the paper is limited.
   - One of the main contributions highlighted in this paper is quantized tokenization for graph Transformers. However, the paper simply combines graph neural networks for dealing with graph-structured data and existing residual vector quantization for the quantization. 
   - Another contribution is multi-task self-supervised learning. The paper uses two self-supervised learning objectives such as DGI and GMAE2, which are not proposed self-supervised training scheme in this paper. I think that using two self-supervised objectives to improve the effectiveness of the model is hard to become a good contribution in top-tier ML conferences.

- More experiments need to be included.
   - The author claims that the proposed quantization method improves the robustness and generalization of the model. But, there is no experimental result concerning the robustness and generalization.
   - In the similar manner, the validation of efficiency needs to be added to empirically prove that the proposed quantization is effective and efficient.
   - Also, the paper only deals with node classification tasks. It would be better if the paper included other graph-related tasks, such as node regression and graph classification, which have been actively dealt with in other graph Transformer papers.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes a self-supervised method, named GQT, for graph token learning. GQT adopts the residual vector quantization to generate graph tokens and trains the model with mutual information maximization, graph reconstruction loss and distillation loss minimization. Given the learned codebook, GQT augments the input graph with edges between similar nodes and generates a node sequence for each node with topK-related nodes in personalized PageRank. Extensive results on node classification are provided to demonstrate the effectiveness of GQT.

### Strengths
- This paper decouples the graph tokenizer from transformer training, which poses an interesting thread for developing graph transformers.
- The proposed method achieves competitive performance on node classification, including heterophilic and homophilic datasets.
- GQT scales to benchmarks with large graphs.

### Weaknesses
- As the core module in GQT, the authors did not provide a sufficient description of the residual vector quantization (nor in the appendix). How does GQT learn the mapping from graph nodes to the codebook tokens? The mapping relation $X_Q$ is introduced in L194/199 and never used again in the following context.
- The writing of section 5.2 is rather arbitrary. I cannot fully comprehend the embedding matrix $X_T$, which is 'trained end-to-end with the Transformer'. How does $X_T$ be learned in GQT? In L333-334, the authors argue that 'adding this explicit aggregated token leads to better performance compared to initializing $X_T$ with $C$. Does this mean that $X_T$ is initialized with $C$ or not?
- The operator $[\cdot,\cdot]$ in Eq 14/15 lacks a definition.
- The GQT (or tokenizer) is claimed to (1) capture long-range dependencies; (2) less memory footprint; (3) improved generalization capabilities, which lack of supportive analysis in the experiments.

### Questions
-  How does GQT perform with different sizes of codebooks ($K$, $c$ and $d_c$).
- The graph codebook learning resembles node clustering. Can authors provide a comparison between these two types of methods? Or how does node clustering perform in the graph codebook learning?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This work attempt to design a graph tokenizer that utilizes graph self-supervised learning to train a encoder. And then leverage the Transformer to capture long-range dependencies. Specifically, the authors adapts Residual Vector Quantization to learn hierarchical discrete tokens for reducing memory requirements and improving generalization capabilities.

### Strengths
1. The proposed Graph Quantized Tokenizer utilizes discrete tokens, which can significantly reduce memory requirements.

### Weaknesses
1. The major concern of this manuscript is the experiment part. According to [1], classic GNNs (e.g. GCN, GraphSAGE and GAT) can also achieve excellent performance on node classification tasks. However, the proposed GQT obtains lower performance on almost all datasets. This makes me worry about the actual contribution of this work.

2. The novelty of the proposed method is limited. There are already some works try to utilize vector quantization to learn discrete tokens for graph learning [2,3]. And there are also existing work utilize graph augmentation and Personalized PageRank to generate a sequence per node [4].

3. There are also some concept errors in the manuscript. For example, the authors mention that "Recent studies on applying Large Language Models (LLMs) to graph-related tasks have found the representing ..." in the introduction section. But most recent methods about combining LLMs with graph learning can only work on textual-attributed graph, and they just leverage LLMs to capture better textual  understanding [5, 6].


[1] Luo, Yuankai, Lei Shi, and Xiao-Ming Wu. "Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification." arXiv preprint arXiv:2406.08993 (2024).

[2] Yang, Ling, et al. "VQGraph: Rethinking graph representation space for bridging GNNs and MLPs." The Twelfth International Conference on Learning Representations. 2024.

[3] Xia, Jun, et al. "Mole-bert: Rethinking pre-training graph neural networks for molecules." (2023).

[4] Fu, Dongqi, et al. "VCR-graphormer: A mini-batch graph transformer via virtual connections." arXiv preprint arXiv:2403.16030 (2024).

[5] Li, Yuhan, et al. "A survey of graph meets large language model: Progress and future directions." arXiv preprint arXiv:2311.12399 (2023).

[6] Jin, Bowen, et al. "Large language models on graphs: A comprehensive survey." IEEE Transactions on Knowledge and Data Engineering (2024).

### Questions
1. Could the authors provide the explanation about why GQT obtains lower performance than classic GNNs?

2. I think the authors need to re-emphasize the innovation of this work.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
5

---

## Human Reviewer 5

### Summary
In this work, the authors propose GQT, a transformer-based framework for graph prediction tasks. GQT consists of a variety of proposed components, including a custom training loss and a tokenizer trained separately with a self-supervised learning approach. The tokenizer leverages vector quantization, resulting in a highly efficient tokenization that can be fed into a standard transformer encoder and applied to very large graphs with up to 2.5 million nodes. Empirically, GQT outperforms a variety of GNN and transformer baselines on 16 out of 18 transductive node classification problems, including both homophilic and heterophilic graphs.

### Strengths
- **S1**: The authors evaluate their model on a wide variety of transductive node classification datasets, both homophilic and heterophilic and up to 2.5 million nodes. The empirical results are impressive and indicate that GQT is a versatile approach to transductive node classification.
- **S2**: The authors present an ablation study in Table 4 and Table 5 on the various components of their approach. The study indicates that most components help with improving the overall model performance; see **Q1** for a follow-up question on the ablation study.
- **S3**: The approach for learning a tokenizer using vector quantization is highly interesting and should be further explored in future research. In particular, I expect that GQT can be readily adapted to other types of graph learning problems (such as inductive settings) while being highly efficient.

### Weaknesses
- **W1**: While the results are impressive, the authors evaluate exclusively on transductive node classification problems. In my opinion, the authors should be more upfront about this limitation or include results for inductive settings, e.g., the inductive tasks in [https://arxiv.org/abs/2206.08164](https://arxiv.org/abs/2205.12454).
- **W2**: Section 4.2 introduces the training loss with three different loss terms. However, the authors do not provide any intuition or justification for these choices. As such, it is difficult for a reader to understand why these choices were made and how crucial they are to the overall framework. A similar issue appears in Sections 5.1 and 5.2.
- **W3**: In the introduction in L35-36, the authors state that “GTs possess an expressive power equivalent to the 2-Weisfeiler-Lehman (WL) isomorphism test”, citing Kim et al. (https://arxiv.org/abs/2207.02505).  The transformer used in Kim et al. is entirely different from the transformers used in transductive node classification and in particular, different from GQT. As such, this reference does not serve as a motivation to the approach present in this work. Further, note that the GQT uses a GNN to learn local interactions and would suffer from the same expressivity limitations as GNNs, leading to the next statement in L37-38: “This [the 2-WL expressivity of GTs] surpasses the expressive power of message-passing GNNs, which are limited to the 1-WL test”. This statement is incorrect. The 1-WL and 2-WL tests as defined in Kim et al. are equivalent in expressive power. Finally, in L38-39, the authors state that “[…] a Transformer with sufficient attention heads can match or exceed the expressive power of a second-order invariant graph network, outperforming message-passing GNNs”, again citing Kim et al. Here, the authors omit the fact that transformers can only achieve said result when receiving as input $O(n^3)$ number of tokens, resulting in a transformer with $O(n^6)$ runtime and memory complexity. Again, this approach is entirely different to GQT and does not serve as a motivation for GQT. I strongly recommend the authors to adjust the introduction, remove the incorrect claims and provide a proper motivation for their approach.

### Questions
- **Q1**: Which components of GQT are also applicable to baselines? For example, while the tokenizer is certainly unique to GQT, adding semantic edges and PPR features to nodes may also improve baseline performance. I wonder whether the authors can give any indication as to whether the improved performance of GQT stems largely from components unique to GQT or whether their proposed components may be useful in improving the results of transformers for transductive node classification in general.
- **Q2**: Table 4 and 5: Are the standard deviations missing from these tables or was the ablation study performed only on a single seed? In case the study was performed on a single seed, I strongly recommend to repeat the ablation study for at least three seeds to verify that the results are statistically significant.
- **Q3**: Figure 1, the experimental section, the ablation studies as well as the conclusion mention “structural gating”. However, this term is never formally introduced. I suppose the authors refer to the gating mechanism in Equation 16. If this is the case, I encourage the authors to define properly define the term there.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4