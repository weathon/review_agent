# NOCL: Node-Oriented Conceptualization LLM for Graph Tasks without Message Passing

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Graphs are essential for modeling complex interactions across domains such as social networks, biology, and recommendation systems. Traditional Graph Neural Networks, particularly Message Passing Neural Networks (MPNNs), rely heavily on supervised learning, limiting their generalization and applicability in label-scarce scenarios. Recent self-supervised approaches still require labeled fine-tuning, constraining their effectiveness of their generalization. Meanwhile, Large Language Models (LLMs) excel in natural language tasks but face significant challenges when applied to graphs, including preserving reasoning abilities, managing extensive token lengths from rich node attributes, and being limited to textual-attributed graphs (TAGs) and a single level task. To overcome these limitations, we propose the Node-Oriented Conceptualization LLM (NOCL), a novel framework that leverages two core techniques: node description, which converts heterogeneous node attributes into structured natural language, extending LLM from TAGs to non-TAGs. 2) node concept, which encodes node descriptions into compact semantic embeddings using pretrained language models, significantly reducing token lengths by up to 93.9% compared to directly using node descriptions. Additionally, our method employs graph representation descriptors to unify graph tasks at various levels into a shared, language-based query format, paving a new direction for Graph Foundation Models.. Experimental results validate NOCL competitive supervised performance relative to traditional MPNNs and hybrid LLM-MPNN methods and demonstrate superior generalization in zero-shot settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a framework that integrates Large Language Models (LLMs) into graph learning without message passing neural networks (MPNNs). The proposed method NOCL uses graph representation descriptors, which serialize graph structures into textual form, allowing all graph tasks (node-, edge-, and graph-level) to be reformulated as language-based queries. This makes it possible to apply instruction tuning via LoRA to train LLMs for graph understanding tasks without any architectural changes. Experiments on benchmark datasets (ogbn-arxiv, Cora, PubMed, MUTAG, ogbg-molhiv) demonstrate competitive supervised performance and strong zero-shot generalization.

### Strengths
1. MPNN-Free Paradigm: The work proposes an new framework that departures from message-passing methods, directly aligning graph learning with LLM capabilities.
2. Efficient Node Concept Encoding: NOCL encodes these node descriptions into compact semantic embeddings using pretrained language models (PLMs), reducing token length.

### Weaknesses
1. The datasets for experiments are quite limited. The authors only used five datasets in the paper to test the performance of NOCL. For a GFM paper, it is expected that more datasets are investigated.
2. The authors chose molecule datasets as the representatives of non-TAG graphs. However, the text descriptions of atom nodes were already realized in [1]. [1] is not cited or discussed in this paper. And there should be experiments on more non-TAG graphs from other domains to enhance the paper's contribution.
3. Although the token sequence is , NOCL only uses one-hop neighborhood for the target nodes. It is still questionable that wether NOCL can scale to real-world large graphs or capture long-distance dependency and global graph properties.
4. There are several typos in the paper which hinder the readers' understanding of the paper. For example, the token length in Line 251 is not supposed to be "4 + 2n + 3n".
5. Overall, the novelty of the paper is limited, and the contributions authors claimed and advantages over other methods are not fully justified.

[1] One for All: Towards Training One Graph Model for All Classification Tasks

### Questions
1. How does NOCL ensure consistent outputs when node ordering in graph descriptors changes? do the node descriptors capture adjacency symmetries and structural motifs effectively? What’s the effect of random edge ordering on predictions?
2. Can NOCL handle graphs with millions of nodes or multi-hop ego graphs? What is the computational behavior as graph size increases, given the fixed token window of LLMs? 
3. Can NOCL produce interpretable reasoning chains for the prediction it makes?
4. Zero-shot performance is shown only across similar academic graph datasets. Can the model generalize to domains like social graphs or knowledge graphs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes NOCL (Node-Oriented Conceptualization LLM) — a framework for graph learning without message passing. The authors argue that conventional MPNNs (e.g., GCN, GAT) suffer from limited generalization and that prior LLM-based graph models are restricted to textual-attributed graphs. NOCL introduces two core components: (1) Node Description — transforming heterogeneous node features into structured natural language, and (2) Node Concept — encoding these descriptions into compact embeddings using pretrained language models to drastically reduce token length (up to 93.9%). The model reformulates all graph tasks (node, edge, and graph level) into unified text-based queries, enabling end-to-end reasoning by LLMs without message passing. Experiments across five datasets show that NOCL achieves competitive or superior performance compared to both MPNNs and hybrid LLM-MPNN methods, especially in zero-shot generalization and efficiency.

### Strengths
1) The idea of representing graph elements as language through node descriptions and “node concepts” is conceptually elegant and well-motivated, bridging structured graph representation and natural language reasoning. 2) The design is technically clean — avoiding MPNNs entirely while maintaining efficiency through compact embeddings and LoRA-based instruction tuning, which demonstrates thoughtful engineering. 3) Empirical results are convincing: NOCL matches or surpasses supervised MPNNs and prior LLM-based graph models while offering strong zero-shot generalization and significant computational savings.

### Weaknesses
1) Despite the strong narrative, the methodological novelty may be seen as incremental — the key steps (description-to-embedding encoding and prompt-based task formulation) largely extend existing text-to-graph ideas without fundamentally new architecture or training objective. 2) The evaluation scope is relatively narrow: only five datasets with small graphs and simple tasks are tested, leaving open whether NOCL scales to large, complex graphs or dynamic graph settings. 3) The approach sacrifices permutation invariance and structural inductive bias — while the paper acknowledges this, the empirical section does not adequately analyze robustness or ordering sensitivity, which weakens claims of generalization.

### Questions
N/A

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
This paper proposes the method that enable LLMs to understand the TAG and non-TAG and perform the original graph specific tasks without involving any message passing network. Specifically, the method basically verbolize the node features into texts and then instead of inputting the raw node descriptions, it creates node embedding and train a connector for the LLM to understand this embedding. Finally, it involves graph representation descriptors to let the LLM answer the desired tasks.

### Strengths
- This paper correctly identifies several limitations of previous methods with LLM+graph including the lack of generalization ability from GNN based models and the problem with long texts with descriptions or even neighbor descriptions. Also, it includes the non-TAG graphs that previous work overlooked.
- The use of mainly LLM as a unified and generalized solution for graph-based tasks is reasonable, clean and follow the trends of LLM evolvements and potential of only LLM based graph foundation model
- They include various tasks, including node, graph and link tasks into this framework

### Weaknesses
- The node description method might not be that novel, also the way it applied to non-TAG graph might not be optimal and unified for all types of non-TAG graph. The process of having node concept as embedding produced from PLM and connector require good training of connector and high-quality PLM, the connector also might suffer alignment issue with LLM under limited training data.
- There are more GNN+LLM baselines that are more up to date can be discussed and compared in this case. I think more focus of baselines should be on GNN/MPNN + LLM instead of purely GNN or purely LLM.
- The current descriptor is sequential which breaks the permutation invariance in the original graph nature, the performance is highly dependent on the order of the node. There could be more ablation study on this point. 
- The datasets included are not enough to demonstrate the effectiveness of this method, probably include more TAG datasets and more molecule based datasets to demonstrate the performance on TAG and non-TAG respectively.

### Questions
- Do you think the method can be extended to more hops and for larger graphs?
- For the node concept embedding, why we only choose the connector to be one linear layer, is it sufficient to accomplish the job to let LLM read the embedding well. Also, can we train the connector together with the LLM, will it result in better alignment?

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
This paper introduces NOCL, a message-passing-free framework that reformulates graph learning as a language understanding task. By converting node attributes into natural language descriptions and encoding them into compact embeddings, NOCL enables large language models to efficiently handle both textual and non-textual graphs. Experiments demonstrate competitive performance in both supervised and zero-shot settings.

### Strengths
1. The node description mechanism extends LLMs to non-text-attributed graphs such as molecular graphs, substantially broadening their applicability. It effectively unifies node-, link-, and graph-level tasks within a cohesive instruction-tuning framework.
2. Reformulating diverse graph tasks as natural-language comprehension queries is powerful, enabling unified multi-level instruction tuning under a single framework.
3. The paper provides validation across multiple datasets. NOCL not only achieves in supervised settings but also significantly outperforms baselines on classification tasks. Its zero-shot results further demonstrate effective use of the LLM.

### Weaknesses
1. The paper should include comparisons with more recent baselines, such as GOFA [1], to more comprehensively demonstrate NOCL’s advantages.
2. For non-TAG datasets, the node descriptions rely on manually crafted templates. How sensitive is the model’s performance to the quality or phrasing of these templates?
3. Although the paper mentions that NOCL’s performance depends on the PLM choice, it would be helpful to show how much different PLMs affect performance and generalization.

**Reference**

[1] GOFA: A generative one-for-all model for joint graph language modeling, ICLR, 2025.

### Questions
See the above **Weaknesses**.

### Soundness
3

### Presentation
2

### Contribution
2
