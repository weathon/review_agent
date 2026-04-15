# Parameter Efficient Graph Encoding for Large Language Models

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
How can we best encode structured data into sequential form for use in large language models (LLMs)? In this work, we introduce a parameter-efficient method to explicitly represent structured data for LLMs. Our method, GraphToken, learns an encoding function to extend prompts with explicit structured information. The encoding function in GraphToken uses graph neural networks to effectively transfer the relational inductive biases in the structured data to a LLM. Unlike other work which focuses on limited domains (e.g., knowledge graph representation), our work is the first effort focused on the general encoding of structured data to be used for various reasoning tasks. We show that explicitly representing the graph structure allows significant improvements to graph reasoning tasks. Specifically, we see across the board improvements - up to 73% points - on a wide variety of node, edge and, graph-level tasks on benchmarks for graph reasoning (GraphQA) and molecular property prediction (ChemLLMBench).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces GraphToken, a novel parameter-efficient approach for encoding graph data and solving graph reasoning tasks with LLMs. GraphToken leverages GNNs to encode graph information into continuous encodings, which contain explicit structural information. Such encodings are passed to LLMs to handle graph reasoning tasks for node, edge, and graph levels. The GraphToken method retains frozen LLM parameters, which reduces computational requirements compared to other Graph + LLM work. The experimental results on GraphQA and molecular property prediction from ChemLLMBench indicate that smaller LLMs equipped with GraphToken can outperform much larger models.

### Strengths
- The use of GNNs to encode graph structures into LLM-compatible tokens is a good contribution, as it addresses a gap in the current approaches of LLMs for graph data.

- The experiments cover a wide range of models and tasks, with different LLMs and GNN architectures, also evaluated on node, edge, and graph-level reasoning tasks, and chemical property prediction benchmarks.

- The ability to boost smaller LLMs to surpass larger ones has important implications for resource-efficient ML and real-world deployment, which may also contribute to LLM reasoning study. 

- The paper is clearly written and easy to follow.

### Weaknesses
- While the paper shows strong results, a more detailed analysis of where GraphToken underperforms would strengthen the contribution. For example, it would be helpful to know if there are specific graph types or structures where the method struggles. Maybe provide a breakdown of performance across different graph sizes or structures, or to include error analysis on specific examples where GraphToken struggles.

- One limitation of the method is that it requires access to an open-source LLM to compute gradients. This could be a challenge for 1) the stronger proprietary LLMs have no such access, limiting the method’s applicability in certain contexts. 2) computing gradients for larger models can be expensive, even though the LLM parameters are not tuned.

- Adding a more explicit discussion of limitations can strengthen the paper. For example, generalization to larger or more complex graphs.

### Questions
- The results show that larger models have an advantage over the proposed GraphToken + small model for the connected nodes task. Are there any insights or analysis about why this is the case? Why do larger models perform particularly well for this task? Can you provide a more detailed analysis of the connected nodes task, perhaps including visualizations or examples that illustrate why this task might be particularly challenging for GraphToken.

- The method is claimed to be parameter efficient. Any runtime analysis or comparison to the baselines?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel approach to integrating graph information into frozen large language models (LLMs). Given recent findings that graph serialization may be inefficient, the authors propose encoding the graph through a specialized neural network, whose output is then fed into the LLM via soft prompting/prefix tuning.

### Strengths
The manuscript is well-organized and clearly written, making it easy to follow. The experimental section is robust, featuring evaluations on two diverse dataset collections—GraphQA and ChemLLMBench—as well as a detailed ablation study exploring the impact of different graph encoders.

The proposed model shows significant performance gains over traditional graph serialization techniques, utilizing a relatively compact graph encoder with around 1-2 million parameters.

Overall, the contributions in this paper are significant as it is one of the first (if not the first) works that use a graph encoder in adjacency with a frozen LLM, and addressing the concerns could further solidify its impact.

### Weaknesses
1. Comparisons with GraphLLM: In line 156, the authors reference another graph-encoder-based approach (GraphLLM) suggesting it is limited to specific graph types. However, GraphLLM claims compatibility with simple graphs, such as those in GraphQA, and can also handle graphs containing additional textual information. In contrast, GraphToken may not accommodate such graph structures. This distinction should either be clarified in the text or, if I have misunderstood, an experimental comparison with GraphLLM could help elucidate the model’s advantages more directly.

2. Task-Dependent Encoder Output: Line 202 notes that the graph encoder’s output is tailored to the specific downstream task and there is no universal approach.
- First, this could be considered a form of data leakage. To address this, the experimental section should at least include a performance evaluation for tasks such as “node degree” utilizing a single token (i.e., the same graph encoder for all tasks). 
- Additionally, the fact that, in some cases, a single token per node/edge is utilized raises questions about the scalability to larger graphs (graphs with significantly more nodes than ~12 which is the average size in  GraphQA). Has the model been evaluated with larger graphs? Insights into its performance under these conditions would be valuable.
- Lastly, such as reliance on the downstream task raises some concerns around how the model could be actually deployed as a general-purpose Large Graph+Language Model. Further clarification on this point would strengthen the paper’s contributions.

### Questions
In Section 7, it is noted that experiments were conducted on TPUs; however, details on the model’s memory requirements are not provided.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method to combine graph models and LLMs to query graph structural properties better. It uses a graph encoder, such as a GNN or a graph transformer, to embed a graph into graph tokens and combine the graph tokens with text/question tokens as the input to a downstream LLM, for generation. Various experiments are conducted, and the method shows strong improvement over existing baselines on graph and molecule properties prediction tasks.

### Strengths
- The idea of the paper is clearly conveyed, and the writing is easy to follow.

- While most existing GNN+LLM work focuses on text-attributed graph learning ability, this paper addresses an important issue on querying graph structures.

### Weaknesses
- While I agree graph property querying is crucial to align graph information with LLM, the presented work is only weakly connected to LLMs. In this work, the LLM merely serves as a decoder that answers "yes" or "no" based on the embeddings generated from a graph encoder, and it is not much different from an MLP classifier. The goal of incorporating an LLM is to use its generalizability and human-interpretability. Take your generalization study as an example, a meaningful integration with LLM is when you train the model with triangles and cycle check tasks, you can directly use the model to tackle bipartite check tasks by only changing the question to "Is the graph bipartite?" Can you justify why the integration of an LLM is necessary when your model seems to be a supervised-task-only model?

- While the paper is from a graph structure querying objective, similar architectures are proposed by many existing (though possibly contemporary works). Just to name a few approaches that encode graphs with a graph encoder and align that with an LLM [1,2,3,4]. The discussion of this line of work is missing from the paper, and I think the connection of the proposed work to PEFT is relatively weak.

- The learning task design is over-simplified and can be easily captured by a graph model or position encodings. It is like including the answer in the question, but only in an implicit way. If I understand correctly, all tasks are trained separately, and the model cannot learn from similar tasks (cycle check and triangle count can help each other). I suggest a discussion on why these tasks are helpful and meaningful and why not incorporate more tasks and train them together to build a large graph property prediction model.

- While the proposed approach outperforms LLM baselines on the molecule property prediction tasks, it actually underperforms graph models (the authors should consider adding graph models into baseline comparison). This relates back to my first weakness, that is, the performance boost seems to be solely attributed to the use of a stronger graph model and has very little to do with an LLM. To show that LLM is necessary, you should consider providing results that show that adding LLM improves the performance over the "graph-encoder-only" variant, or, the model directly generalizes to unseen molecule targets.

[1] Tang, Jiabin, et al. "Graphgpt: Graph instruction tuning for large language models." Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2024.

[2] Kong, Lecheng, et al. "Gofa: A generative one-for-all model for joint graph language modeling." arXiv preprint arXiv:2407.09709 (2024).

[3] Chen, Runjin, et al. "Llaga: Large language and graph assistant." arXiv preprint arXiv:2402.08170 (2024).

[4] He, Yufei, and Bryan Hooi. "UniGraph: Learning a Cross-Domain Graph Foundation Model From Natural Language." arXiv preprint arXiv:2402.13630 (2024).

### Questions
Please see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This submission addresses the interesting problem of using large language models (LLMs) for graph reasoning tasks. To transform structured graph data into a sequential form suitable for LLMs, the authors propose a parameter-efficient method called GraphToken. This approach leverages graph neural networks to learn graph embeddings, which are then concatenated with text embeddings to form the inputs for LLMs. However, the paper could benefit from a clearer clarification of the motivation and technical contributions, as well as the inclusion of additional baselines and experimental results.

### Strengths
S1. This submission presents an interesting solution to integrate the graph reasoning task with LLMs.

S2. The presentation is clear and easy to follow.

S3. The empirical results demonstrated by authors significantly outperform baselines methods.

### Weaknesses
**Weakness**

W1. The motivation is not well-explained.

W2. The technical contributions are insufficiently detailed.

W3. Important literature on graph foundation models is missing.

W4. There is a lack of efficiency comparisons.


**Concerns**

C1. In the motivation, the paper mainly criticizes prior graph foundation models that use text descriptions as input for graphs, leading to the claim: “This highlights the need to explore better and more expressive ways of representing structured data to an LLM.” However, the authors overlook the fact that previous graph foundation models were built to integrate seamlessly with any open-source or closed-source LLM, aiming to externally drive LLMs to perform graph reasoning tasks. In other words, this should highlight the need to explore better and more expressive ways of representing structured data to an LLM **in a convenient and flexible way**. The graph representation and embedding fusion method proposed in this paper may not achieve this goal.

C2. From a technical perspective, this submission employs a classical mechanism of embedding fusion using a graph encoder and a text encoder. The authors need to clarify what the technical contributions of this mechanism are.

C2-1. To present the technical contributions, the authors could elaborate on the core issue: “There is often no clear choice in what order to sequentially write the graph data.” The authors are expected to explain what research challenges this straightforward approach aims to address.

C3. The paper needs to discuss and include more graph foundation models for experimental comparison.

C3-1. As mentioned in the paper, a classic approach to enabling LLMs for graph reasoning is transforming graphs into text descriptions. Although this approach may seem intuitively difficult to compare directly with the proposed embedding fusion method, the authors should still discuss and include it in experiments to demonstrate the superiority of their approach.

C3-2. Experimentally, the tasks considered in this paper do not cover some of the more challenging problems addressed by current graph foundation models. For example, graphWiz [1] focuses on classic graph problems such as bipartite, topology, triangle, Hamiltonian, and subgraph problems. The authors should discuss these tasks and consider including them.

C4. As discussed in C1, the paper’s technical approach combines parameter-efficient fine-tuning to drive LLMs for graph reasoning tasks. Therefore, the authors may need to discuss the efficiency of the proposed method to address concerns raised in C1.

Minor Concern:

C5. Lines 33-42 seem more like an introduction to the graph RAG task rather than the graph foundation model task.

**Reference**

[1] GraphWiz: An Instruction-Following Language Model for Graph Computational Problems, KDD 2024.

### Questions
Please check C1 to C4.

### Soundness
2

### Presentation
2

### Contribution
2
