# GraphQ-LM: Scalable Graph Representation for Large Language Models via Residual Vector Quantization

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Large Language Models (LLMs) have demonstrated remarkable proficiency in diverse language-centric tasks, yet their application to structured graph data presents unique challenges, particularly in efficiently tokenizing graph elements. While graphs offer powerful structural representations, existing methods for interfacing them with LLMs, such as creating distinct token embeddings for every node, face significant scalability limitations: the input vocabulary for the LLM grows linearly with the number of nodes, hindering applicability to large-scale graphs. Drawing inspiration from vector quantization's success in compressing information in domains like audio and vision, we introduce a novel approach to represent graph node features for LLMs. Our method, GraphQ-LM, employs Residual Vector Quantization (RVQ) to encode continuous node features into a compact sequence of discrete tokens derived from fixed-size codebooks. These "graph tokens," representing structural feature information, are seamlessly integrated with textual attributes of nodes and their neighborhoods, forming a rich, multimodal input for the LLM. By aligning the codebook's embedding dimension with that of the LLM and jointly training the RVQ module with the LLM, we learn graph-aware representations optimized for downstream tasks like node classification. Extensive experiments demonstrate that GraphQ-LM not only achieves state-of-the-art performance but, crucially, offers a scale-free tokenization strategy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Large Language Models (LLMs) have shown significant success in a wide range of language-centric tasks, but applying them to structured graph data presents unique challenges. One of the main obstacles is efficiently tokenizing graph elements. Traditional methods, such as creating distinct token embeddings for every node, face scalability issues: the input vocabulary for the LLM grows linearly with the number of nodes, making it impractical for large-scale graphs.

To address this, the authors propose GraphQ-LM, a novel approach that leverages Residual Vector Quantization (RVQ) to encode continuous node features into a compact sequence of discrete tokens. These tokens are derived from fixed-size codebooks, similar to vector quantization techniques used in audio and vision domains for compression. The resulting "graph tokens" represent structural feature information and are integrated with the textual attributes of nodes and their neighborhoods to form a rich, multimodal input for the LLM.

The key innovation in GraphQ-LM is the alignment of the codebook’s embedding dimension with that of the LLM. Additionally, the RVQ module is jointly trained with the LLM, enabling the system to learn graph-aware representations that are optimized for downstream tasks, such as node classification. Through extensive experiments, the authors demonstrate that GraphQ-LM not only achieves state-of-the-art performance but also offers a scale-free tokenization strategy, overcoming the scalability limitations of traditional methods.

### Strengths
1. GraphQ-LM introduces a scale-free tokenization strategy by using Residual Vector Quantization (RVQ), which ensures that the vocabulary size does not grow with the number of nodes. This is a crucial advantage over traditional methods, making the approach more suitable for large-scale graph tasks.

2. By integrating both graph structural features (represented as graph tokens) and textual attributes of nodes, GraphQ-LM creates a rich, multimodal input that allows the LLM to leverage both the graph's structural and semantic information. This fusion of modalities helps the model better understand the graph context.

3. A comprehensive experimental result is provided and the experimental results demonstrate that GraphQ-LM achieves state-of-the-art performance on tasks like node classification, showcasing its practical effectiveness in graph-based machine learning problems.

### Weaknesses
1. While RVQ helps with compressing node features, it may lead to some loss of information due to the quantization process. Depending on the task, this compression might reduce the model's ability to capture fine-grained details from the graph structure, especially for highly complex or heterogeneous graphs.

2. While the paper shows strong results on node classification tasks, the evaluation could be expanded to other graph-related tasks (e.g., link prediction, graph generation) to demonstrate the versatility and robustness of the model across different types of graph applications.

3. The joint training of the RVQ module with the LLM adds complexity to the training process. Fine-tuning the RVQ module to work seamlessly with the LLM may require additional computational resources and sophisticated training strategies, particularly for large graphs.

### Questions
Investigate whether other forms of vector quantization or compression methods could further improve the accuracy of the model while maintaining its scalability, especially for more complex graphs. 
To further validate the model’s effectiveness, extend the evaluation to additional graph tasks beyond node classification, such as link prediction or graph generation, to demonstrate its broader applicability.

### Soundness
3

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
4

### Summary
GraphQ-LM introduces a scalable framework that encodes continuous node features into compact discrete tokens via Residual Vector Quantization (RVQ), enabling efficient graph representation for LLMs.

### Strengths
1. It is important to try to use LLM to enhance the performance of downstream tasks.
2. The proposed method achieves competitive results on three datasets while also reducing storage costs.

### Weaknesses
1. The originality seems limited. The core contribution of the paper is to learn discrete tokens using RVQ. However, [1] also uses RVQ to learn discrete tokens and reduces memory requirements and enhances generalization. In addition, a large number of VQ concepts [2] are used for graph learning. 

2. According to the classification in [3], this paper belongs to LLM as Predictor. However, both LLM as Predictor/Encoder/Aligner contain a lot of work. However, the paper only emphasizes InstructGLM in LLM as Predictor. Discussion with the latest text-attributed graphs methods and experimental comparison is needed to increase the credibility of the experiment.

3. The dataset is too small. LLM uses a 3B scale, but still uses Cora. It is recommended to use a larger TAG dataset. 

4. Eq. 2-4 are not very readable. 

5. What is the effect of using VQ directly in the ablation experiment?

[1] Wang, Limei, et al. "Learning graph quantized tokenizers." arXiv preprint arXiv:2410.13798 (2024).

[2] Lin, Qika, et al. "A Survey of Quantized Graph Representation Learning: Connecting Graph Structures with Large Language Models." arXiv preprint arXiv:2502.00681 (2025).

[3] Jin, Bowen, et al. "Large language models on graphs: A comprehensive survey." IEEE Transactions on Knowledge and Data Engineering (2024).

### Questions
See in Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address the scalability challenge of applying Large Language Models (LLMs) to large graphs, where vocabulary grows linearly with the number of nodes, this paper proposes GraphQ-LM. The framework uses Residual Vector Quantization (RVQ) to encode high-dimensional node features into a short sequence of discrete "graph tokens" from a fixed, shared codebook. This design breaks the conventional O(n) scaling bottleneck, enabling sub-linear storage growth. These compact tokens are combined with textual attributes to create an efficient, multimodal input for the LLM. Experiments show that GraphQ-LM achieves state-of-the-art accuracy on multiple benchmarks while using significantly smaller LLMs and reducing node representation storage costs by orders of magnitude.

### Strengths
1. This paper accurately identifies and addresses a critical bottleneck for LLMs on graphs: the linear growth of vocabulary with graph size. Applying RVQ for node feature tokenization is a clever and novel idea that provides a scalable paradigm for integrating LLMs with large graphs.
2. GraphQ-LM not only achieves competitive accuracy on multiple benchmarks but, more importantly, demonstrates remarkable efficiency.
3. The authors conduct a comprehensive set of experiments. The method is benchmarked against a wide range of strong baselines, including GNNs, Graph Transformers, and other LLM-based methods. Crucially, detailed ablation studies (e.g., "w/o RVQ") clearly isolate and validate the contribution of the core components.

### Weaknesses
1. This paper highlights impressive gains in storage efficiency, the discussion on inference latency is limited.
2. The current method primarily quantizes node features, while graph structure is represented indirectly by including neighbors in the prompt. This may limit the model's ability to capture complex structural priors.
3. The framework is validated exclusively on the node classification task. The paper provides limited discussion on how the approach and its scalability benefits would extend to other crucial graph tasks, such as link prediction or graph-level classification.

### Questions
1. On Inference Efficiency: Table 5 reports latency in "ms per query". To provide a more complete picture of practical efficiency, could the authors report the total inference time required to classify the entire test set of ogbn-arxiv and compare it directly with a fast GNN baseline like GraphSAGE?
2. While the 'w/o RVQ' experiment is convincing, a 'w/o Text' ablation would be a valuable addition to further disentangle the respective contributions of the textual features and the learned graph tokens. Have the authors performed this experiment, using only graph tokens in the prompt without any natural language text?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces GraphQ-LM, a framework for scalable graph integration with LLMs via residual vector quantization (RVQ). Instead of the prevailing approach of assigning a separate LLM token to every graph node which results in an impractically large vocabulary, GraphQ-LM discretizes node features into a fixed-length, shared code sequence using multiple residual quantizers. These quantized “graph tokens” are interleaved with natural node text and neighborhood attributes in a soft prompt, enabling efficient LLM-based inference.

### Strengths
Visualization and Interpretability: Figure 3 offers clear evidence that RVQ token assignments are discriminative and remain well-utilized per class, confirming that the approach avoids codebook collapse and preserves class-specific structure.
Resource Efficiency and Scalability: The ablation and cost analysis show that the memory footprint scales modestly from small to large graphs, a non-trivial achievement given the notorious inefficiency of language-as-graph  baselines.
Effective Compression Without Accuracy Loss: Table 1 demonstrates that, across classic GNN backbones (GCN ChebNet GraphSAGE, GAT), discrete RVQ-tokenized features nearly match or outperform original continuous features on node classification, supporting the central premise that node attributes can be highly compressed without harm.

### Weaknesses
Lacks sufficient literature review and novelty. Regarding work on graph tokenization, this concept has already been addressed in VQGraph [1]. Meanwhile, Dr.E [3] (which seems to adopt RVQ) have extended graph tokenization to LLM4Graph, and [4] provides a further review on this topic. However, it appears that the authors have not demonstrated the differences from and advantages over these existing works.

Limited Evaluation Beyond Node Classification

[1]VQGraph: Rethinking Graph Representation Space for Bridging GNNs and MLPs
[3]Multi-View Empowered Structural Graph Wordification for Language Models
[4]A Survey of Quantized Graph Representation Learning: Connecting Graph Structures with Large Language Models

### Questions
It is hoped that the authors will demonstrate the comparisons with and differences from existing work (See W1).

### Soundness
2

### Presentation
3

### Contribution
2
