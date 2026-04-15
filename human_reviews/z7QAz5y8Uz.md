# FoGE: Fock Space inspired encoding for graph prompting

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
Recent results show that modern Large Language Models (LLM) are indeed capable of understanding and answering questions about structured data such as graphs. Existing proposals often use some description of the graph to create an ``augmented'' prompt fed to the LLM. For a chosen class of graphs, if a well-tailored graph encoder is deployed to play together with a pre-trained LLM, the model can answer graph-related questions well. Existing solutions to graph-based prompts range from graph serialization to graph transformers. In this work, we show that the use of a parameter-free graph encoder based on Fock space representations, a concept borrowed from mathematical physics, is remarkably versatile in this problem setting. The simple construction, inherited directly from the theory with a few small adjustments, can provide rich and informative graph encodings, for a wide range of different graphs. We investigate the use of this idea for prefix-tuned prompts leveraging the capabilities of a pre-trained, frozen LLM. The modifications lead to a model that can answer graph-related questions -- from simple graphs to proteins to hypergraphs -- effectively and with minimal, if any, adjustments to the architecture. Our work significantly simplifies existing solutions and generalizes well to multiple different graph-based structures effortlessly.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes FoGE, which is a parameter-free Fock-space-based method to generate graph representations. The method can encode arbitrary graphs into embeddings and train a linear adapter that aligns these embeddings with a frozen LLM via prefix-tuning. It can handle various graph types, including simple graphs, hypergraphs, and attributed graphs, and the experiments show that this method can achieve competitive performance with baseline models.

### Strengths
- This paper proposes a theoretical lossless Fock-space-based method to generate graph representation and can handle graphs with attributes. The use of parameter-free encoding simplifies the design compared to other GNNs or graph transformers.
- This method generalizes across various types of graphs and the experiments validate its performance on many tasks. It generates graph embedding that is not specific to each task and performs better than other specialized models in some cases.

### Weaknesses
- It is not clear whether the LLM contributes to understanding graph structures or merely functions as a text generation module. The interaction between the LLM and graph embeddings should be further explored. For example, it would be insightful to see how the graph embeddings perform if they were used to train a simple MLP or other lightweight models for downstream tasks. This would help clarify whether the LLM adds value beyond generating natural language outputs.
- In Table 5, the proposed method underperforms relative to GraphLLM across several tasks, which brings concerns about whether FoGE can consistently achieve better performance in advanced graph reasoning tasks. 
- The experimental settings are not thoroughly discussed, and details about the training process and hyperparameters are missing. Providing more transparent implementation details would improve the reproducibility of this work.

### Questions
- Could you explain how to obtain the size vector $s$ in the encoding process?
- Table 4 highlights the performance on hypergraphs compared to zero-shot and few-shot scenarios. Could you provide more details on how to generate the corresponding prompts?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
1

### Summary
The paper presents a parameter-free graph encoder using Fock space representations to improve LLMs in answering graph-related questions. This simple approach provides rich encodings for diverse graphs, enabling effective prefix-tuned prompts for pre-trained LLMs, simplifying and generalizing existing methods.

### Strengths
Integrating graph encoders with LLMs to address graph-related tasks is a convincing approach.

### Weaknesses
1. **Unclear motivation**: The motivation of this paper is not clear. Many previous works leverage graph encoders to learn graph representations for integration with LLMs [1,2], and the overall framework appears similar.
2. **Potential confusion with graph prompting methods**: In this work, the authors use graphs to prompt LLMs, which could be confused with existing graph prompting methods [3,4,5]. The authors should discuss the differences between these approaches to help readers better understand the purpose of this work.
3. **Impact of graph encoder choice**: Will the choice of graph encoder affect the performance of the proposed method? Some analytical experiments may provide a better understanding of the impact of the graph encoder choice.


[1] Chen, Runjin, et al. "LLaGA: Large Language and Graph Assistant." Forty-first International Conference on Machine Learning.\
[2] Tang, Jiabin, et al. "Graphgpt: Graph instruction tuning for large language models." Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2024.\
[3] Liu, Zemin, et al. "Graphprompt: Unifying pre-training and downstream tasks for graph neural networks." Proceedings of the ACM Web Conference 2023. 2023.\
[4] Sun, Mingchen, et al. "Gppt: Graph pre-training and prompt tuning to generalize graph neural networks." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.\
[5] Yu, Xingtong, et al. "Few-shot learning on graphs: from meta-learning to pre-training and prompting." arXiv preprint arXiv:2402.01440 (2024).

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposed a novel approach to obtain powerful and model-agnostic graph representations that can be used as prompts to augment LLM’s capabilities of answering graph-related questions. By leveraging Fock spaces, a concept from mathematical physics, they achieved almost lossless task-agnostic graph embeddings capturing the diverse graph structure and information. A lightweight linear adapter is then adopted to map the rich Fock-space inspired embeddings into an LLM’s embedding space, making prefix-tuning effective for various graph-related tasks.

### Strengths
1. By leveraging Fock-space inspired graph encoding, this approach achieves nearly lossless parameter-free graph embeddings, containing rich graph structure information.
2. They show that a simple linear layer can map the task-agnostic graph embeddings into the LLM embedding space, offering a low-computational approach to align graph representations with pre-trained language models.
3. They conducted experiments to validate the informativeness of Fock-space inspired graph embeddings on basic graph-understanding tasks using a small neural network. Experiments also show that their embedding method performs the best in unsupervised approaches and is competitive with specialized supervised methods.
4. This graph encoding method can adapt to different graph types, node-level embeddings and hypergraphs, making it suitable for a wide range of applications.
5. They demonstrated that by using fock-space graph embedding, the LLM can do better at graph understanding and reasoning.

### Weaknesses
1. Although the method is task-agnostic, it might underperform on specialized tasks without additional tuning.
2. Although the method can achieve nearly lossless parameter-free graph embeddings, there’s no study whether it can effectively capture high-order interactions between nodes, which is important in complex graph reasoning.

### Questions
1. Can you explain how the node representations $p_i$ and extra vector size vector $s$ are obtained? Will these representations affect the Fock-space inspired embeddings?
2. How much performance gap is this task-agnostic graph encoding method with specially tuned ones?
3. Can Fock-space inspired graph encoding understand high-order interactions between nodes?

### Soundness
3

### Presentation
2

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
Inspired by Fock space, the paper proposes a training-free graph encoder approach to align graph embeddings with the LLM's embedding space. Specifically, the paper uses a parameter-free scheme to obtain graph embeddings and then trains a linear layer to align these embeddings with the LLM's embedding space, enabling the model to handle graph tasks effectively with minimal adjustments to the architecture.

### Strengths
it is a novel idea to use fock space inspired method to obtain graph embedding.

### Weaknesses
- Lack of efficiency experiments. I agree with the authors that the graph encoding is parameter-free and efficient. However, the complexity is dominated by the LLM rather than the GNN, even though the LLM is frozen. When using large LLMs such as LLaMA-7B, the overall training time will not differ significantly whether the graph encoder requires training or not. Therefore, my concern is that the training-free graph encoder does not offer a noticeable efficiency improvement in terms of real training/application.

- Justification of the use of LLMs. Why LLM should be used in graph reasoning task (without text attribute). It make senses to introduce LLM to help with textual graph tasks, because such tasks need the text reasoning ability, world knowledge from LLM. However, for traditional graph tasks such substructure count, shortest path etc, I didn;t see the necessity of using LLMs.

- Comparison with RAG. In the introduction, the authors mention RAG and compare it with prefix tuning. However, I don’t see the relevance of this comparison. Prefix tuning is designed to adapt a model’s attention to new contexts with minimal parameter updates, whereas RAG combines retrieval mechanisms with generation to enrich the model’s knowledge base dynamically. The distinction between the two methods is significant, and it would help if the authors clarified why RAG was introduced here or provided a more targeted comparison relevant to graph encoding.

- Imprecise Statements. The statement, "RAG-based approaches for graphs primarily involve converting graphs to text, while prefix tuning with graphs uses modules to extract richer, task-relevant structures, requiring larger sample sizes and higher compute power," is unclear. Could you add references to RAG-based approaches where "graphs primarily involve converting graphs to text"? For example, in [1], retrieval is performed over graphs instead of text, and in [2], text is organized into graphs for retrieval.

References:

[1] G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering, NeurIPS 2024, https://arxiv.org/pdf/2402.07630

[2] From Local to Global: A Graph RAG Approach to Query-Focused Summarization, https://arxiv.org/pdf/2404.16130

### Questions
Do you need to train separate adapters for each dataset, or is there a unified adapter for all datasets?

### Soundness
2

### Presentation
2

### Contribution
2
