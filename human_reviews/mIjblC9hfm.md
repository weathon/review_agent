# GOFA: A Generative One-For-All Model for Joint Graph Language Modeling

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Foundation models, such as Large Language Models (LLMs) or Large Vision Models (LVMs), have emerged as one of the most powerful tools in the respective fields. However, unlike text and image data, graph data do not have a definitive structure, posing great challenges to developing a Graph Foundation Model (GFM). For example, current attempts at designing general graph models either transform graph data into a language format for LLM-based prediction or still train a GNN model with LLM as an assistant. The former can handle unlimited tasks, while the latter captures graph structure much better---yet, no existing work can achieve both simultaneously. In this paper, we first identify three key desirable properties of a GFM: self-supervised pretraining, fluidity in tasks, and graph awareness. To account for these properties, we extend the conventional language modeling to the graph domain and propose a novel generative graph language model GOFA. The model interleaves randomly initialized GNN layers into a frozen pre-trained LLM so that the semantic and structural modeling abilities are organically combined. GOFA is pre-trained on newly proposed graph-level next-word prediction, question-answering, structural understanding, and information retrieval tasks to obtain the above GFM properties. The pre-trained model is further instruction fine-tuned to obtain the task-solving ability. Our GOFA model is evaluated on various downstream datasets unseen during the pre-training and fine-tuning phases, demonstrating a strong ability to solve structural and contextual problems in zero-shot scenarios. The code is available at https://github.com/JiaruiFeng/GOFA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces three key properties that an effective Graph Foundation Model (**GFM**) should possess: self-supervised pretraining, fluidity, and graph understanding. To meet these goals, the authors propose Generative One-for-All (**GOFA**) , which integrates a Language Model Decoder with a Graph Language Encoder, which strategically interleaves trainable Graph Neural Network (GNN) layers into frozen Large Language Model (LLM) layers.

The GOFA model is pre-trained on a diverse set of datasets, including MAG240M, Pubmed and Arxiv, Wikikg90mv2, WikiGraph, and Ultrachat200k. Notably, the WikiGraph dataset is created specifically for this work and it is derived from the existing WikiText dataset.

The pretraining results validate the effectiveness of incorporating GNN layers into LLMs, while instruction-tuning experiments further demonstrate GOFA's strong zero-shot generalization capabilities.

### Strengths
The paper is well-structured and easy to follow. Its key strengths can be summarized as follows:

1. The authors identify three crucial properties of an effective graph foundation model: self-supervised pretraining, fluidity, and graph understanding. I strongly agree with these as important considerations.

2. The paper introduces GOFA, which operates under a graph completion framework to enable large-scale self-supervised learning. Furthermore, GOFA demonstrates impressive performance in both zero-shot and fine-tuning scenarios when compared to LLM baselines.

3. One potential concern in developing a foundation model is whether it might negatively impact the capabilities of existing LLMs. However, the results in Table 1 suggest that GOFA's training does not degrade the performance of the original pre-trained LLMs. This is a critical design criterion when building a graph foundation model.

### Weaknesses
1. The authors suggest that integrating GNN layers into LLMs offers a theoretical advantage over standard LLMs when it comes to modeling graph structures. However, I did not find any accompanying mathematical formulations or equations to substantiate this claim in the appendix.

2. GOFA performs significantly worse than the LLM baselines on the SceneGraphs dataset, despite the authors attributing this to the instruction-tuning data containing information-dense texts. It's concerning that a domain-specific foundation model, despite careful design, does not demonstrate competitive performance.

3. As the authors note, using a frozen LLM compressor while tuning only the GNN layers appears questionable. This approach might impede optimal alignment between information from various modalities, potentially limiting the model's overall effectiveness.

### Questions
I have two questions for the authors:

1. Dataset quality: I would like to inquire whether the authors thoroughly validated the content generated in their constructed WikiGraph dataset. Given that the dataset is created by prompting large language models (LLMs), there is a known issue with the potential for LLMs to produce hallucinated or inaccurate outputs. Could the authors elaborate on the measures they took to ensure the accuracy and quality of the dataset? How did they mitigate the risks of including erroneous or noisy data in the final dataset?

2. Could the authors provide details about the pre-training time when using 8 NVIDIA A100-SXM4-80GB GPUs?

### Soundness
3

### Presentation
4

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
The authors introduce GOFA, a general-purpose Graph-Language Model, which proposes an efficient but somewhat intricate approach to encoding a graph into a collection of vectors suitable for input into a large language model (LLM). The model is trained on a large dataset collection using substantial computational resources (8 A100 80GB GPUs).

### Strengths
GOFA is trained on multiple datasets and it appears to be indeed a general-purpose Large Graph Language Model.

The paper is well-structured, with a comprehensive presentation of both quantitative and qualitative results across the main text and appendix.

### Weaknesses
1. **Evaluation of Graph Structure Utilization.** In Figure 1 and Section 3.4, the authors outline four pre-training tasks: (a) sentence completion, (b) question answering, (c) structural understanding, and (d) information retrieval. However, apart from the structural understanding task, it is somewhat unclear how these tasks benefit from graph information. For instance, tasks like sentence completion and question answering might be achievable by directly providing the raw text input to a generic LLM, possibly yielding similar outcomes. Similarly, in Figure 2, the sentence completion and question-answering tasks are primarily language-oriented and may not require graph-based augmentation. Even in Figure 3, the abstract completion task appears somewhat independent of the graph structure. 
\
It would be helpful if the authors could include a comparison where an LLM alone performs these tasks, highlighting any areas where it falls short. Alternatively, if GOFA matches or surpasses the LLM’s performance while adding distinct capabilities, this would serve as a compelling justification for GOFA’s design. Currently, the experimental section primarily focuses on comparing GOFA to LLMs in the structural understanding task (Q3), where graph augmentation evidently improves performance. Additionally, introducing more graph-relevant questions, as seen in recent works (e.g., Fatemi et al.), could further substantiate the advantages of incorporating graph structure in the evaluations.

2. **Comparison with Related Work.** The proposed approach bears similarities to recent models such as GraphToken (https://arxiv.org/abs/2402.05862) and GraphLLM (https://arxiv.org/abs/2310.05845). A detailed comparison, particularly of the advantages and limitations of GOFA relative to these methods, would strengthen the discussion. Including an experimental comparison could also help clarify GOFA’s unique contributions and situate it within the broader context of graph-enhanced LLM research.

### Questions
1. Could the authors clarify how the NOG is represented in practice? Specifically, is a special token used for this purpose?

2. Could the authors provide more details about the graph samples used? How many nodes on average are processed by GOFA on each forward pass and how capable is it to provide accurate answers as the graph size increases?

3. How many tokens are reserved for the graph input in the final question-answering LLM (purple tokens in Fig. 3)? Is it a fixed number or does it depend on the graph's size?

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
4

### Summary
This paper extends language modeling to the graph domain and proposes a new generative graph language model, GOFA. The GOFA architecture interleaves frozen LLM layers with trained GNN layers, adding structure-aware processing layers to the LLM while keeping the original LLM intact and providing processing paths for regular language modeling through gating mechanisms. Input graphs are transformed into Text-Attributed Graphs (TAGs), and a next-token-prediction framework is adopted over various graph datasets in natural language, serving as a general self-supervised pretraining objective over graph-structured data. GOFA shows strong performance on benchmark datasets versus previous works building foundation models for graph-structured data using LLMs, and provides a flexible framework for integrating LLMs with GNNs.

### Strengths
- The authors offer detailed discussion about the desired attributes of a graph foundation model, including the ability to generalize to new tasks and data domains without additional finetuning of the foundation model, i.e. zero-shot settings.
- The authors also give detailed discussion on previous methods for incorporating GNNs into LLM-based architectures (LLMs as predictors and LLMs as an enhancer), and discuss the drawbacks of previous approaches in building general graph foundation models. The graph foundation model should account for the unique structural information of input graphs and generate graph representations depending on the input prompt. It should preserve the LLM’s prompt following capabilities while also learning graph structure and semantic information jointly.
- The authors propose an architecture which designs a graph language encoder that encodes input TAGs into a fixed number of tokens while accounting for both the semantic information of node and edge text as well as the structure of the graph. Their model controls computational complexity by only doing message-passing on each index of the memory tokens, and feeding a fixed number of vectors to the LLM decoder for the downstream graph task.

### Weaknesses
1. It is unclear whether the results in Table 2 of the paper represent a true zero-shot setting, since the authors mention in Section 5.2 and Appendix F.4 that the model is instruction-finetuned on node classification and link prediction examples. The datasets in Table 2 are unseen, however samples of the same task on other datasets has been seen by the model during instruction finetuning, making it seem like this is a generalization task to unseen datasets rather than a zero-shot setting where the model has never seen node classification samples before.
2. Additionally, given that the authors emphasize GOFA as a graph foundation model, more comprehensive benchmarking across more graph datasets would better support this claim. Examples include Amazon Products, molecular graph datasets such as OGBN-molhiv, or protein-protein interaction datasets such as OGBN-proteins. Further benchmarks would better support the claim that GOFA supports diverse downstream tasks on graph-structured data across a variety of graph datasets.
3. The authors do not benchmark against any traditional GNN baselines. The main comparison remains other LLM and LLM-based graph models, however a graph foundation model should compare against traditional GNN baseline models to indicate performance relative to much lower capacity GNN models on the same tasks.

### Questions
1. More clarity on the degree to which the GOFA saw examples of node classification and link prediction during instruction finetuning would improve clarity on whether the results can be called a zero-shot setting, or whether GOFA is generalizing seen tasks to new datasets unseen during pretraining and instruction finetuning.
2. More benchmarking results across graph datasets as well as including traditional GNN baselines would greatly strengthen the empirical results of GOFA, and support its claim as a graph foundation model.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a new joint graph and language model that can perform different graph and language tasks. The model is a combination of an LLM compressor, GNN layers and an LLM decoder. The model can be instruction-tuned to solve many graph-structured tasks, and at the same time, it remains effective on language tasks. The model is also more efficient than plain LLMs solving the same tasks, since it separates the input into different nodes and edges.

### Strengths
1. The paper addresses an important and challenging problem of building generic and effective models for different modalities.
2. The proposed model, while being a combination of existing methods, is logical and original. Especially, applying the GNN layers on top of memory tokens seems a very good idea to exchange information in an efficient and flexible way.
3. The experiments are sufficiently large, diverse and challenging.
4. The results are overall strong and promising.

### Weaknesses
1. The model is designed more for the tasks where nodes/edges are rich in text. It would be interesting and more impactful to include tasks (in the pretraining and/or eval) that are more numeric, e.g. molecule prediction/generation such as ZINC or from open graph benchmark. While the SOTA specialized GNNs are probably going to work better in those cases, it would be interesting to see the gap.

2. Related to above, one potential issue of this approach is that representing node/edge features can be extremely inefficient when the features are high-dimensional (e.g. a node is an image or some other array data).

3. What's the SOTA GNN results in Table 2? It's not a direct comparison, but useful to know for reference.

4. The examples in Fig 2 do not seem to require much knowledge of the graph, i.e. a plain LLM should be able to solve these tasks well. The example does not show what are the nodes and what is the meaning of edges and why those are important for the tasks. It seems like all the information exists within the node.

5. In Table 4, it's unclear if LLM-N and GOFA-F trained on the same data? Also does LLM-N have the complexity of O((V k)^2) as discussed in L298? 

Minor:

- In the "LLMs as predictors" some representative papers like GraphLLM are not discussed.

### Questions
1. Do the authors use a pretrained llm decoder or only a pretrained compressor?

2. From eq. 3, since GNN is not indexed by k, it appears to be shared across all the K tokens. Is it correct? 

3. In the graph task, the authors "set the default target nodes to all nodes in the graph". How the authors combine generations from each node intto a single answer/prediction?

### Soundness
3

### Presentation
3

### Contribution
3
