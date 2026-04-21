# Unleashing the Potential of Text-attributed Graphs: Automatic Relation Decomposition via Large Language Models

- Avg Score: 5.25
- Decision: Reject
- Scores: 5, 6, 5, 5

## Abstract
Text-attributed graphs (TAGs) integrate textual information with graph structures, offering unique opportunities for leveraging language models to enhance node feature quality. However, our extensive analysis reveals that the downstream task performance on TAGs is hindered by the graph structure itself; treating diverse semantics(e.g. “advised by”, “participates in”) as a singular relation (e.g. hyperlinks). By decomposing conventional edges into distinctive semantic relations, we discover significant improvement in GNNs’ downstream task performance. Motivated by this, we present **RoSE** (Relation-oriented Semantic Edge-decomposition), a novel framework that leverages large language models (LLMs) to automatically decompose graph structures into different semantic relations without requiring expensive human labelling or domain expertise. **RoSE** consists of two stages: (1) identifying semantic relations via an LLM-based generator and discriminator, and (2) decomposing each edge into corresponding relations by analyzing raw textual contents associated with connected nodes via an LLM-based decomposer. The decomposed edges provided by our framework can be applied in a model-agnostic, plug-and-play manner, enhancing its versatility. Moreover, **RoSE** achieve state-of-the-art node classification results on various benchmarks and GNN architectures.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a prompting-based method for augmenting edge types in graphs. It contains several stages, such as relation generation, relation filtering, and relation decomposer. Experimental results on node classification show that the augmented edge types are useful.

### Strengths
1. Using LLM to augment the predefined graph structure is interesting.
2. The paper is well-written and easy to follow.
3. Amount of experimental results are provided to demonstrate the effectiveness of the proposed method.

### Weaknesses
1. My biggest concern is that the proposed method is not as totally automatic as they claimed, since for graphs from different domains, this method requires a large effort to try the best prompt for graph descriptions and relation filtering guidelines.
2. The proposed efficient relation-type annotation strategy seems too simple. The assumption that the neighbor nodes should have a similar relation type with the central node is not realistic. For example, a paper can cite one paper via a positive sentiment but cite another via a negative sentiment.
3. The proposed method can be viewed as an edge classification method. The authors should include some datasets with ground-truth edge labels to show how much the proposed method can approximate the ground-truth edge labels. In the experiment, the authors should also include the node classification performance with the ground-truth edge labels for comparison.

### Questions
see weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a framework RoSE (Relation-oriented Semantic Edge-decomposition) to enhance the representation learning of Graph Neural Networks by decomposing edges of TAGs into semantically meaningful relations. The authors highlight the limitations of traditional TAG methods that treat all edges uniformly and demonstrate that automatic edge decomposition using LLMs significantly improves node classification performance. The framework operates in two stages: (1) identifying potential relation types with an LLM-based generator and discriminator, and (2) categorizing each edge into identified relation types using LLMs. Extensive experiments show that RoSE leads to up to a 16% improvement in node classification accuracy on benchmark datasets.

### Strengths
1. The introduction of RoSE as an automated method to address the challenge of edge oversimplification in TAGs is interesting. 
2. The authors provide extensive experiments that demonstrate substantial improvements across multiple datasets and GNN architectures. 
3. The introduction of an efficient edge-sampling strategy to minimize LLM queries is a thoughtful addition.

### Weaknesses
1. One of the biggest concerns is edge oversimplification in TAGs may not be a problem. There is a type of graph called textual-edge graphs, where edges are annotated with rich text information [1, 2].
2. Although RoSE introduces methods to improve efficiency, the initial decomposition process may still be resource-intensive for practitioners working with massive TAGs. A more detailed analysis of the computational cost compared to traditional methods would strengthen the paper.
3. The relation discriminator's role and effectiveness are crucial to RoSE, yet the description of its implementation and performance impact lacks clarity. Further explanation or ablation studies would help clarify its significance.

[1] Li, Zhuofeng, et al. "TEG-DB: A Comprehensive Dataset and Benchmark of Textual-Edge Graphs." arXiv preprint arXiv:2406.10310 (2024). 
[2] Ling, Chen, et al. "Link Prediction on Textual Edge Graphs." arXiv preprint arXiv:2405.16606 (2024).

### Questions
1. Can you provide a more in-depth analysis of the computational cost associated with RoSE, especially for large-scale datasets? A direct comparison with simpler baselines could give a clearer picture of the trade-offs involved.
2. Can you elaborate on the design and function of the relation discriminator, including specific examples and results from ablation studies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper focus on the study of including LLMs in enhancing the textual meaning of edge connections in the graph, and then use multi-relation types graph to improve the prediction performance compared to using single-type graph structure. The authors first use a LLM generator to generate the possible types of edges, and another LLM discriminator is adopted to choose the appropriate edge type(s). A multi-relational GNN is applied on the obtained graph for prediction targets.

### Strengths
1. The presentation of this paper is clear and easy to follow.
2. The overall idea of enhancing the edges with semantic level types is interesting.
3. The results from human labeled edge types have shown promising potential of adopting enriched semantic edges idea.

### Weaknesses
1. The efficiency is a main concern. As the proposed framework requires LLM on both edge types generation and discrimination, it requires significant amount of time in these steps for both training and inference stage. This largely harms the practical potential of this method.
2. The experimental results on different datasets varied a lot. On some datasets such as Cora and Pubmed, the improvement of proposed method is very limited. Can authors give intuitive reasons behind it?
3. In some cases the semantic description between two nodes may not be able to be classified into several types, or require some domain knowledge. How the proposed method can handle those situations?

### Questions
Will the number of generated types (5, 10, 20, etc) affect the results a lot?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper identifies the problem that conventional edges on text-attributed graphs hinder the representation learning process of GNNs on downstream tasks when integrated with advanced node features. To resolve the issue the edges are decomposed into semantic relations by the LLM. This has also been proposed as a data enhancement method where edge attributes are absent. 
The relations are identified using an LLM-based relation generator and discriminator through prompting. To avoid labeling every edge and making a lot of LLM queries, similar relations are applied to similar semantic relations nodes with a threshold value.

### Strengths
The paper presents good experiments with multiple runs to prove their hypothesis. Multiple GNN architectures have been used on various datasets which adds to the fact the technique works.

### Weaknesses
1. I have concerns about the novelty of the ideas presented. Generating edges' relations through LLM is not new; it has been shown to be the basic strategy for combining LLMs and GNNs. 
2. As asked in the "Question section" I am not sure that the time and space complexity added would be worth the accuracy increase. Presenting those numbers would make the argument stronger.

### Questions
What is time increment for each process? The time table only has comparison on the sampling approach vs not using the approach. What is the time takes for - 
1. Assigning edge labels
2. Training with edge attributes

### Soundness
2

### Presentation
2

### Contribution
2
