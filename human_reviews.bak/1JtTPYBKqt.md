# Neural Architecture Retrieval

- Decision: Accept (poster)
- Scores: 8, 8, 6

## Abstract
With the increasing number of new neural architecture designs and substantial existing neural architectures, it becomes difficult for the researchers to situate their contributions compared with existing neural architectures or establish the connections between their designs and other relevant ones. To discover similar neural architectures in an efficient and automatic manner, we define a new problem Neural Architecture Retrieval which retrieves a set of existing neural architectures which have similar designs to the query neural architecture. Existing graph pre-training strategies cannot address the computational graph in neural architectures due to the graph size and motifs. To fulfill this potential, we propose to divide the graph into motifs which are used to rebuild the macro graph to tackle these issues, and introduce multi-level contrastive learning to achieve accurate graph representation learning. Extensive evaluations on both human-designed and synthesized neural architectures demonstrate the superiority of our algorithm. Such a dataset which contains 12k real-world network architectures, as well as their embedding, is built for neural architecture retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This is a very interesting paper, considering the burgeoning field of neural architecture designs which has made it increasingly challenging for researchers to position their contributions or establish relationships between their designs and existing ones. The paper presents a novel problem of neural architecture retrieval, aiming to automate the discovery of similar neural architectures. The authors identify the limitations of existing graph pre-training strategies, which are unable to handle the computational graphs in neural architectures due to the graph size and motifs. They propose a creative solution by dividing the graph into motifs, which are then used to reconstruct the macro graph, addressing the identified issues. Moreover, the introduction of multi-level contrastive learning is put forth to attain precise graph representation learning. The paper boasts extensive evaluations of both human-designed and synthesized neural architectures, showcasing the superiority of their proposed algorithm. The authors also construct a valuable dataset comprising 12k real-world network architectures along with their embeddings, laying a solid foundation for neural architecture retrieval. This endeavor may address a pressing issue in the domain with pre-training an embedding database for finding and comparing neural architectures. The paper potentially opens up a new domain within the neural architecture community, paving the way for more organized and efficient exploration in this field by building an embedding database, and the video demo shows promising applications.

### Strengths
* The motivation behind the paper is well-articulated and resonates with the ongoing challenges faced by researchers in situating their contributions amidst a plethora of existing neural architectures. The introduction of Neural Architecture Retrieval as a solution to automate the discovery of similar neural architectures is timely and could significantly alleviate the existing bottleneck.
* The methods used in the paper sound and well-justified. The logic behind each step of the solution is sound and reasonable, showcasing a thorough understanding of the challenges at hand. For instance, the approach to addressing the repeated design of blocks is practical and applicable, demonstrating a commendable level of methodological rigor and a pragmatic stance.
* The evaluation metrics, particularly the information retrieval scores, are promising, especially when applied to the nas dataset. This suggests that the proposed methods are effective and could potentially set a new standard in evaluating neural architectures. 
* Contribute a 12k real-world computational graph and its corresponding embedding database. The availability of such a dataset could spur further research and development in the domain of neural architecture retrieval and related areas.

### Weaknesses
* Page 5 Section 3.4 Eq 4: The objective of the first stage may have a better way. The encoder encodes the architecture into motifs, and then concatenates the embeddings.  Directly sampling the highest-frequency motifs $H$ to represent the main design for large models may more reasonable, especially considering that the concatenation of motif embeddings cannot backpropagate the gradients without two stages.

* Page 7 Section 4.3: The details of the evaluation part are lacking. Although it covers the mainstream information retrieval scores, the parameters of the computational score are still unclear. For example, when testing the NDCG, is graded relevance or non-graded relevance used? It would be better to provide a formula here.

### Questions
Page 5 Section 3.4 Eq 4: What role does the context graph $G_s$ play?

Page 6, Section 4.1: How were the real-world repositories collected, and how were the models obtained? Will the 12k real-world computational graphs, along with corresponding labels, be made available for follow-up work?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents a new problem of Neural Architecture Retrieval, which aims to find similar neural architectures from a large collection of existing and potential ones. The paper introduces a novel graph representation learning framework that leverages the motifs of neural architectures and contrastive learning to achieve accurate and efficient retrieval. The authors also construct a new dataset of 12k real-world neural architectures with their embeddings and evaluates the proposed method on both real-world and NAS architectures, showing its superiority over baselines. However, the paper has some issues that need to be addressed. My detailed comments are as follows.

### Strengths
1.	The paper presents a new problem called Neural Architecture Retrieval (NAR). This problem is to find similar neural architectures quickly and easily from a big pool of existing and possible designs. It’s a smart way to make the search for neural architectures simpler and more efficient.

2.	The authors propose a novel and easy-to-understand graph representation learning framework that addresses the computational graph in neural architectures. This framework adopts motifs of neural architectures and multi-level contrastive learning for accurate graph representation learning.

3.	The paper contributes the community by creating a new dataset with 12k real-world network architectures and their embeddings. This dataset is specifically designed for neural architecture retrieval and demonstrates the effectiveness of the retrieval algorithm. It’s a helpful benchmark for everyone working in this field.

4.	The experimental results on both human-designed and synthesized neural architectures benchmarks verify the effectiveness of the proposed method.

### Weaknesses
1.	The proposed NAR appears to focus solely on the topological similarity of architectures. However, it is important to note that the similarity between architectures can vary across different tasks or datasets. For instance, certain architectures might yield comparable results in image classification but diverge significantly in performance when applied to other tasks. Could the authors provide additional insights and elaborations on this matter?

2.	The authors introduce a motif-level contrastive learning approach, wherein the corresponding context graph is treated as the positive sample. However, this raises a concern as there may be other context graphs that also encompass the same motifs, yet they are deemed as negative samples, which seems unreasonable. Could the authors provide further clarification and justification for this aspect of their methodology?

3.	It is unclear how the authors collect/annotate the ground-truth label of the architecture dataset. Without such labels, it is infeasible to calculate the correlation between the different networks. Please provide more details.

4.	In the "Related Work" section, it would enhance the manuscript's thoroughness if the authors could offer a more comprehensive discussion on mainstream NAS methods. This should include an exploration of reinforcement learning-based NAS methods, as referenced in [A-G].

[A] Designing Neural Network Architectures using Reinforcement Learning. ICLR 2017.

[B] Learning Transferable Architectures for Scalable Image Recognition. CVPR 2018.

[C] UNAS: Differentiable Architecture Search Meets Reinforcement Learning. CVPR 2020.

[D] Breaking the Curse of Space Explosion: Towards Efficient NAS with Curriculum Search. ICML 2020.

[E] Contrastive Neural Architecture Search with Neural Architecture Comparators.  CVPR 2021.

[F] Disturbance-immune Weight Sharing for Neural Architecture Search. Neural Networks 2021.

[G] Towards Accurate and Compact Architectures via Neural Architecture Transformer. TPAMI 2021.

### Questions
1. In Page 2 line 8, “exact” should be “exactly”.

2. In Page 2 line 11, “through” should be “by”.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper defines a new question called Neural Architecture Retrieval (NAR) which returns a set of similar neural architectures given a query neural architecture. To address NAR problem, this paper proposes to split the graph into several motifs and rebuild the graph through treating motifs as nodes in a macro graph. Then this paper uses two-level pretrain task to train the architecture representation for retrieval.

### Strengths
1. This work proposes a large new NAS dataset of 12k real-world neural architectures rather than pre-define search space. In my opinion, this thing makes a lot of sense for the NAS field which can explore the diversity of search space species architectures.

2. The idea uses motifs to encode architecture and reduce the graph size is novel and reasonable to encode architecture and capture the connection between structures in one architecture.

3. The two-level pretrain task to train the architecture's embedding is reasonable and effective.

### Weaknesses
1. In my opinion, the NAR problem to find similar architectures for the query architecture doesn't seem particularly significant for real-world usage, Can the author point out the need for this problem and more application scenarios?

2. Some retrieval-based papers[1-3] for giving a query dataset to search architectures should be discussed.

3. Another question is that I want to know how much performance can be achieved by retrieving similar models only based on the architecture corresponding to the text description or the code by a language mode like Chatgpt or other language-based retrieval model.

Refs: 

[1] Task-Adaptive Neural Network Search with Meta-Contrastive Learning. NeurIPS 2021.

[2] MetaGL: Evaluation-Free Selection of Graph Learning Models via Meta-Learning. ICLR 2023

[3] Retrieving GNN Architecture for Collaborative Filtering. CIKM 2023

### Questions
One small question is that I can't find how the ground truth topk set is constructed, can the author describe it in detail?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
