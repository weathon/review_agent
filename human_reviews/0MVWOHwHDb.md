# Retrieval-Augmented Language Model for Knowledge-aware Protein Encoding

- Decision: Reject
- Scores: 6, 5, 3, 6

## Abstract
Protein language models often struggle to capture the biological functions encoded within protein sequences due to their lack of factual knowledge (e.g., gene descriptions of proteins). Existing solutions leverage protein knowledge graphs (PKGs), using knowledge as auxiliary encoding objectives. However, none of them explored the direct injection of correlated knowledge into protein language models, and task-oriented knowledge integration during fine-tuning, making them suffer from insufficient knowledge exploitation and catastrophic forgetting of pre-trained knowledge. The root cause is that they fail to align PKGs with downstream tasks, forcing their knowledge modeling to adapt to the knowledge-isolated nature of these tasks. To tackle these limitations, we propose a novel knowledge retriever that can accurately predict gene descriptions for new proteins in downstream tasks and thus align them with PKGs. On this basis, we propose Knowledge-aware retrieval-augmented protein language model (Kara), achieving the first unified and direct integration of PKGs and protein language models. Using the knowledge retriever, both the pre-training and fine-tuning stages can incorporate knowledge through a unified modeling process, where contextualized virtual tokens enable token-level integration of high-order knowledge. Moreover, structure-based regularization is introduced to inject function similarity into protein representations, and unify the pre-training and fine-tuning optimization objectives. Experimental results show that Kara consistently outperforms existing knowledge-enhanced models in 6 representative tasks, achieving on average 5.1% improvements.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Kara model, which integrates protein knowledge graphs (PKG) directly into protein language models (PLM) to enhance understanding of biological functions encoded in protein sequences.

### Strengths
It uses a novel knowledge retriever to predict gene descriptions for new proteins during both pre-training and fine-tuning stages, which helps in aligning with PKGs and improves knowledge retention.

These tokens enable token-level integration of knowledge and structural information into protein representations, enhancing the model’s ability to handle high-order knowledge.

### Weaknesses
The performance of the model heavily relies on the quality and the extent of the PKGs used, which might limit its application if relevant knowledge graphs are incomplete or outdated.

While the model shows improvements in task-specific contexts, its ability to generalize across broader protein types or different biological conditions remains uncertain.

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
How to effectively transfer knowledge from knowledge graphs to large language model is a challenging task. In this paper, the authors is the first to propose a novel knowledge retriever, named Kara, that directly injects the correlated knowledge into protein language models and aligns the protein knowledge graph with downstream tasks. Specifically, the contextualized virtual tokens is designed to enable the direct injection of knowledge and high-order structure information into protein representations. Extensive experimental results, arranging from amino acids contact prediction to semantic similarity inference, demonstrate the superior performance of proposed Kara.

### Strengths
In general, the paper is clearly expressed and organized. The authors' innovation of direct injecting the protein knowledge graph into large language model to explore the knowledge-aware protein representation learning, which will have some implications for the biomedical field. In addition, the experiments in the discussion section demonstrate that the virtual tokens and structure-based regularization are good at capturing high-order information of protein knowledge graph from a novel perspective.

### Weaknesses
The Introduction needs to provide more background information, such as the specific role of Knowledge Graphs (KGs) in this context, the benefits they offer, and the rationale behind exploring KG-based methods.

### Questions
1.How is the ProteinKG25 knowledge graph selected? There are many other well-known protein-related multi-omics knowledge graphs, such as PharmKG (Briefings in bioinformatics, 2021), CKG (Nature biotechnology, 2022). Do the types and numbers of entities and relationships affect model performance?

2.The knowledge graph contains only positive samples for interaction-based tasks. Did the authors incorporate negative sampling during training? If so, please provide additional details on how this was implemented.

3.ProteinKG25 is used as the KG, but the model is evaluated on six representative tasks, it is unclear how the entities of these datasets are linked to the knowledge graph.

4.Where is Table 7? If I missing 

5.From many experimental results (e.g., Table 3 and Table 6), we can see that KeAP has achieved comparable performance to Kara. Please describe the difference between the two methods in detail, and be curious about the complexity and number of parameters of the two methods.

6.The experimental design is good, however there are one limitations that preclude the reader to understand how generalizable the method is. Only one protein embedding method (ProtBert) is tested for the pre-trained embeddings.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces Kara that uses information from protein knowledge graphs to improve protein language models. Kara directly injects relevant knowledge during both pre-training and fine-tuning phases, utilizing a knowledge retriever that predicts gene descriptions for new proteins. Kara involves introduction of several key components: Contextualized Virtual Tokens that fuse knowledge and structural information into protein representations; Knowledge injection both at post-training and fine-tuning stages; Retrieval of relevant proteins and graph information with a dense retriever.

### Strengths
- The performance improves on most tasks (following the same experiment tasks and settings as Ontoprotein) compared to Ontoprotein and KeAP.

- The encoding style of Kara combines strengths of Ontoprotein and KeAP: Ontoprotein uses contrastive pretraining to first obtain structure-intensive graph embedding and then inject into language model, while KeAP direct encodes related knowledge in tuples with language encoder. Differently, Kara encodes 1-hop GO entity as knowledge, and 2-hop entities as structure to provide more detailed graph knowledge for the protein language model. 

- The knowledge retriever maps new protein sequence to GO entities, which could make it possible to generalize to proteins not directly covered by the knowledge graph.

### Weaknesses
1. My major concern of this work is its technical contributions, which closely follows OntoProtein and KeAP. The main improvement of Kara compares to Ontoprotein and KeAP is that it encodes both structural information (relations in GO) and knowledge (knowledge stored in each triple) within the contextualized virtual tokens. Ontoprotein uses the same pipeline to encode protein knowledge graph and inject embeddings into the language model, so the technical contributions are minor. 

2. The structural regularization (Eq. 6) obtained from two-hop entities might be weak or misleading. ProteinKG25 is a sparse knowledge graph and entities involve not only proteins as well as biological processes and molecular functions. What is the percentage of proteins that have 2-hop protein neighbors and are the neighbors all functional similar ? Neighbors may not be similar proteins but could be proteins that could interact with each other. Their function may not be similar.

### Questions
1. Protein downstream tasks often require different kinds of knowledge, e.g. PPI requires knowledge about the functions and relations of the two proteins, contact prediction requires evolutionary and structural knowledge. Wonder if the authors could further provide insights on how knowledge & structural information differentiate across tasks. For example, why introducing more graph structural knowledge could improve the performance on contact prediction.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Instead of implicitly modeling knowledge information, the paper proposes knowledge-aware retrieval-augmented protein language model (Kara), which enables consistent knowledge-augmented modeling when applied to downstream protein-related tasks. During the pretraining stage, the authors try to model the structural information in the protein knowledge graph, such as neighboring and high-order connectivity. In the fine-tuning stage, a knowledge retriever is used to bridge the optimization gap between pretraining and fine-tuning, allowing for seamlessly adapting to knowledge updates.
The authors conduct extensive experiments and demonstrate that this unified knowledge modeling process consistently outperforms existing knowledge-enhanced models across six protein-related tasks.

### Strengths
1. The method effectively carries the external knowledge injected during pretraining into downstream, significantly mitigating catastrophic forgetting. Additionally, the knowledge retrieval process not overly complex.
2. The proposed relation-GO combinations further enhance retriever’s ability to recall the informative knowledge.
3. The authors demonstrate the methods’ effectiveness across multiple tasks and conduct thorough ablation studies, such as the effect of without the neighboring information during inference.
4. The paper is well-written and clear

### Weaknesses
1. If the protein belongs to an under-studied or new protein family, does this retrieval method have certain limitations, especially when these proteins have very low sequence identity to known (trained) proteins? It would be better to include experiments on under-studied proteins to demonstrate, possibly by a simulated way that splitting clusters of low-identity proteins into training and validation sets.
2. Further, does the method have potential to uncover patterns of new proteins and their associations with existing?

### Questions
I have listed my questions in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
