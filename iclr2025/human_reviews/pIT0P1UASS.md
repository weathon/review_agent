## Human Reviewer 1

### Summary
This paper studies the scaling laws for foundation models on temporal graphs. It proposes a large number of temporal graphs with various sizes but sharing the same domain. It proposes a training pipeline called TGS-train to train a temporal graph foundation model on a number of temporal graphs and show that the more data is used in training, the better the zero-shot performance of the model on unseen datasets.

### Strengths
1. Nice topic.
2. Nice writing and I feel no difficulty understanding most parts.
3. This work can lead to further exploration in graph foundation model, which is a heated ongoing research area.

### Weaknesses
1. Unclear technical details and problem statements. Please refer to my questions below.
2. One thing I am concerned about is that scaling law does not only refer to more training data. Bigger model size with deeper networks is also strongly related to the scaling law in my opinion. So for me the title of this paper is somewhat overexaggerated, considering the current focus of this paper is just giving an existing TGNN model with more training data. I think the workload of this paper is not enough for ICLR. The claims made by the authors are intuitive. But for me it is not fully supported with the presented experiments. Please refer to the following points and the questions below.
3. The current baseline models are not enough from my point of view. I would prefer to see the performance of all the models presented in TGB [1] and DyGLib [2]. 
4. In the datasets proposed in the submission, different graphs share a lot of nodes. This makes me wonder whether we can merge different temporal graphs into a huge one, with edges labeled with the type of tokens. In this sense, the multi newtork assumption just disappears. So I think I cannot get the point of introducing the idea of multi network. 
5. And I also think the problem setting in this submission is not zero-shot anymore because by training on the graphs with a set of nodes (addresses), even if we want to predict the behaviour of these nodes related to different tokens, the transaction patterns are decided by the addresses (the users) but not only by the token type. For example, I have a node representing a person and I have seen this person trading on token A and B. Now I learn from this person’s transaction history on A and B and wish to use it to predict the bahavior on token C. We already have prior knowledge of this node and predicting the property on token C becomes not zero-shot anymore. I hope this problem can be well-discussed in rebuttal and I wish the authors can notice me if I have misunderstandings.

[1] Huang, Shenyang, et al. "Temporal graph benchmark for machine learning on temporal graphs." Advances in Neural Information Processing Systems 36 (2024).

[2] Yu, Le, et al. "Towards better dynamic graph learning: New architecture and unified library." Advances in Neural Information Processing Systems 36 (2023): 67686-67700.

### Questions
1. Line 153. What do you think of studying CTDGs? I saw you mentioned in DTDGs there are graph dynamics at specific time intervals. But imagine a CTDG with a group of events, each pair of node has its own temporal patterns which can also indicate temporal dynamics. For me, I think DTDGs are discretized CTDGs which will naturally lose part of temporal information, so I think studying CTDGs in your paper scope is more meaningful.
2. Line 169. Could you give definitions of temporal global efficiency, temporal-correlation coefficient, and temporal betweenness centrality? What is temporal graph property prediction? Could you explain by giving examples based on your proposed dataset?
3. Line 282. To be honest, IID training on timeline has been widely used in training temporal knowledge graphs, so I am not sure whether it is that important to point it out in the main body of your paper. I feel it is a slight waste of space.
4. Line 393. I do not understand why you put ablation before your main results. The datasets studied in ablation is omitted so that I do not even know what are you testing with your models. Please notice me if I missed something important.
5. Appendix E is important. I suggest putting it into the main body of the paper. It shows the performance gain is not due to the bias in data selection, but due to the quantity of the training data.
6. The last point I want to mention is that I found nowhere explicitly pointing that the datasets for training your MN series networks are completely disjoint from your evaluation datasets. I guess the “zero-shot” comes from this setting and the MN networks can do inference on the datasets unseen in training. But I prefer that you mention this early in your paper and then the readers can avoid guessing.

### Soundness
1

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper presents a novel study on the transferability and scaling behavior of Temporal Graph Neural Networks (TGNNs) across multiple temporal graphs. The authors introduce the Temporal Graph Scaling (TGS) dataset, comprising 84 ERC20 token transaction networks, and propose the TGS-train algorithm for pre-training TGNNs on multiple networks. The study empirically demonstrates that larger pre-trained models on temporal graphs can achieve enhanced downstream performance on unseen networks, adhering to a neural scaling law. The paper claims this is the first empirical demonstration of transferability to unseen networks in temporal graph learning. The authors also showcase that their multi-network model outperforms fine-tuned TGNNs on thirteen out of twenty unseen test networks using zero-shot inference.

### Strengths
1. The introduction of the TGS dataset is a significant contribution to the field, providing a rich resource for researchers to study temporal graph learning and foundation models.
2. The paper provides empirical evidence that the neural scaling law, previously observed in NLP and CV, also applies to temporal graph learning, which is a valuable insight for model training and generalization.
3. The demonstration of transferability of pre-trained models to unseen networks is a crucial step towards developing foundation models for temporal graphs, with potential applications in various domains.

### Weaknesses
1. The novelty of the paper is limited to meet the standard of the ICLR. From the data perspective, the authors construct Ethereum and ERC20 Token Networks. In my opinion, it is not comprehensive enough. For example, temporal datasets like DBLP and Stack Overflow in the paper "WinGNN: Dynamic Graph Neural Networks with Random Gradient Aggregation Window". From the method perspective, there seems no architecture or algorithm for modeling.
2. The writing is not clear, e.g., Temporal Graph Property Prediction in preliminaries and construction of Ethereum and ERC20 Token Networks. In methodology, equations are needed to clarify.
3. The effect of scaling needs to be further explored. Table 2 is not enough to verify this strong standpoint.

### Questions
1. What is temporal global efficiency, temporal-correlation coefficient, and temporal betweenness centrality in Line 169 of page 4?
2. Figure 2 is not very clear, how do you conduct token parse?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper focuses on a problem that is to estimate the evolution of an unseen network/graph given some observed temporal graph in the same domain. In this paper, a new data that includes 84 temporal networks/graphs is built for temporal graph property prediction task. For learning from this data, a new method TGS-train is proposed which is used to train a temporal GNN across multiple temporal graphs. The whole idea is clear. Most of GNN approaches on multiple graphs are used on drug discovery, while the temporal GNN methods focus on a single graph. This paper combines these two directions and proposed a reasonable data and method. And this paper explored the graph scaling laws on temporal graph learning which is very useful. It is reasonable that training the TGNN using more temporal graphs will benefit the performance.

### Strengths
Strengths:

1.	This paper presents a temporal graph scaling dataset that includes 84 ERC20 token transaction networks. All further analysis about the scaling behavior, and transferability is based on this data.

2.	This paper presents a new training method TGS-training to enable the temporal graph network training on multiple graphs.

### Weaknesses
Weaknesses:

1.	Although this paper introduces the transferability of the proposed method, this method is hard to be extended on the graph in a different domain. The general transferability is still limited. 

2.	The training method is based on HTGN. The TGNN architecture is not proposed by this paper. The key technical modifications are the shuffling and resets. The technical contribution of these two mechanisms is very limited. Shuffling the training temporal graphs is a very straightforward method. It is not clear what the main architecture and method differences between the proposed method with the HTGN are.

### Questions
The proposed training method is very similar to HTGN. Beside shuffling and resets, do we have any other architecture upgrading or modification?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper introduces the Temporal Graph Scaling (TGS) benchmark, containing 84 temporal graphs derived from Ethereum transaction networks. Besides, the paper investigated the training scaling potential for temporal graph neural networks (TGNNs), proposes the first algorithm for pre-training TGNNs on multiple temporal graphs and study how transferrable a pre-trained temporal graph model is to unseen networks. The work demonstrates that by pre-training on a substantial set of temporal graphs, the multi-network model can be directly applied to unseen token networks, delivering better performance compared to individual models trained specifically on those test networks.

### Strengths
- New Benchmark for Temporal Graph Scaling: This work introduces a new benchmark specifically designed for examining temporal graph scaling, featuring an extensive collection of networks. This benchmark enhances the ability to evaluate temporal graph scaling low across a diverse set of temporal graphs.

- A Multi-Graph Training Algorithm: The authors propose a novel multi-graph training algorithm that shows strong effectiveness on the new benchmark. This approach advances the potential for generalizing training across multiple temporal graphs.

- Promising Empirical Results: The study demonstrates the potential of pre-training temporal graph neural networks (TGNNs) across temporal graphs, highlighting the great potential of pre-training strategies for TGNNs.

### Weaknesses
- Limited Ablation Study on Scaling: The ablation study for the TGS-train algorithm—specifically in terms of memory resetting and data shuffling—was only conducted on MN-4 and MN-8. Given the paper’s focus on scaling laws in TGNN pre-training, extending this study to larger networks, such as MN-32 and MN-64, would provide a more comprehensive understanding of scaling effects.

- Restricted Evaluation on Prediction Tasks: The authors use network growth, defined by edge counts, as the prediction target. However, other common tasks for temporal graphs, such as node classification and link prediction, are not included. Including these tasks could enhance the relevance of the benchmark for a wider range of temporal graph applications.

### Questions
Clarification on Technical Details in Section 5: Some implementation details are unclear. For example, has a random historical embedding been used, and how exactly does this mitigate catastrophic forgetting? A more detailed explanation on this technique and its benefits would be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3