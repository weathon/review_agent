# DURENDAL: Graph deep learning framework for temporal heterogeneous networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 6, 3

## Abstract
Temporal heterogeneous networks (THNs) are evolving networks that characterize many real-world applications such as citation and events networks, recommender systems, and knowledge graphs. Although different Graph Neural Networks (GNNs) have been successfully applied to dynamic graphs, most of them only support homogeneous graphs or suffer from model design heavily influenced by specific THNs prediction tasks. Furthermore, there is a lack of temporal heterogeneous networked data in current standard graph benchmark datasets. Hence, in this work, we propose DURENDAL, a graph deep learning framework for THNs. DURENDAL can help to easily repurpose any heterogeneous graph learning model to evolving networks by combining design principles from snapshot-based and multirelational message-passing graph learning models. We introduce two different schemes to update embedding representations for THNs, discussing the strengths and weaknesses of both strategies. We also extend the set of benchmarks for TNHs by introducing two novel high-resolution temporal heterogeneous graph datasets derived from an emerging Web3 platform and a well-established e-commerce website. Overall, we conducted the experimental evaluation of the framework over four temporal heterogeneous network datasets on future link prediction tasks in an evaluation setting that takes into account the evolving nature of the data. Experiments show the prediction power of DURENDAL compared to current solutions for evolving and dynamic graphs, and the effectiveness of its model design.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces DURENDAL, a deep learning framework tailored for THNs. DURENDAL adapts to evolving networks and offers two methods to update embeddings. Through testing on new datasets, including one from a Web3 platform and an e-commerce site, DURENDAL proves to be more effective in predictive tasks compared to existing models.

### Strengths
1. The authors provide a large number of experiments to analyze the effectiveness of the model.
2. THGs are worth exploring.

### Weaknesses
1. The shortcomings of other THNs aren't clarified clearly. For instance, what does *easily incorporate state-of-the-art designs from static GNNs* mean?  And what are the specific drawbacks of these methods? The current presentation lacks clarity, diminishing the paper's motivation when compared to other THNs.  Besides, related work should be cited in the introduction section.
2. This paper's contribution is limited for ICLR standard. The authors primarily employ the ROLAND framework and conventional techniques for heterogeneous graphs. Despite its efficacy, it lacks innovation, potentially falling short of ICLR's acceptance criteria.
3. Recent studies on THNs warrant citation and comparison.
* (1)Fan, Yujie, et al. "Heterogeneous temporal graph neural network." Proceedings of the 2022 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2022.
* (2)Yang, Qiang, et al. "Interpretable Research Interest Shift Detection with Temporal Heterogeneous Graphs." Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining. 2023.
4. The presentation of this paper is poor.  Additionally, there are typographical errors in the article, such as writing THG as TNH.

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes an extension of ROLAND for discrete-time temporal heterogeneous graphs (where the temporal graph is described as a sequence of snapshots), together with two new datasets that can be used for evaluation. The main methodological novelty of the paper is how to incorporate an aggregation mechanism across various edge types through two possible different schemes: update-then-aggregate and aggregate-then-update. For what concerns the datasets, two new temporal heterogeneous graphs are introduced in the paper: TaobaoTH (a dataset of user behaviour provided from Taobao - an online shopping platform) and SteemitTH (a dataset of user interactions from Steemit - a blockchain-based social network). The proposed model is evaluated on multi-relation and mono-relational link prediction on the proposed datasets plus two other datasets that have already been used in the literature (GDELT18, ICEWS18). The approach appears to perform well on the considered datasets when compared to 9 selected baselines.

### Strengths
The paper is overall well written, easy to follow and with good references for people that might be approaching the field of dynamic graphs for the first time. While the approach appears rather straightforward (especially when compared to ROLAND), experimental results look promising on the considered dataset. The introduction of new datasets is also something that the community will most likely benefit from.

### Weaknesses
As it might have emerged from my comments in the “Strengths” section, the approach appears to be a not particularly original improvement over ROLAND (unless my understanding is wrong, the main addition is the introduction of an aggregation mechanism across multiple relations and the use of heterogeneous GNNs for feature extrapolation). On top of this, while yes the method appears to show good results on the considered datasets compared to the baselines, I’ve some doubts about the experimental evaluation. In particular, have the baselines considered in the experiment been tuned for the dataset? Taking for instance TGN from Rossi et al, the model was not evaluated on any of the datasets used in the paper. As such, if such architecture was not tuned (as instead the proposed approach was), we might be observing lower performance for such methods (as well as the other baselines), which are simply due to a suboptimal architectural choice. I’d greatly appreciate if the authors could comment on this in their rebuttal

### Questions
Besides what highlighted above, I have a few questions / comments that I would the authors to address:

1) Many methods appear to achieve on TaobaoTH a PR AUC that is consistent with random guessing in a balanced binary classification problem. This suggests that many models are actually not learning anything meaningful on that dataset. Can you please clarify why this might be the case? 

2) The fact that TGN and CAW do not compute in their implementation the MRR, I believe it’s not a good reason to avoid computing such statistics for these methods. I’d encourage the authors to fix the implementation in this case to provide a better comparison of all methods.

3) I’m confused why many MRRs appear equal to 0.5, can the authors provide some details on the implementation they used for this and how negatives were sampled?

4) In section 4 it is stated that “minimum number of snapshots to allow live-update evaluation” is four, can you provide some details on why that is the case? From algorithm 2 in ROLAND my understanding is that 2 steps are enough for live update evaluation

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work proposes a generic framework of adapting static heterogeneous GNNs to the dynamic setting through two types of schemes: Update-then-Aggregate (UTA) and Aggregate-then-Update (ATU). The authors also introduce two new datasets of dynamic heterogeneous graphs (TaobaoTH and SteemitTH) for future benchmarking. The proposed method achieves better performance in future link prediction tasks on all four datasets.

### Strengths
1. The designed framework is generic and can be integrated with any static heterogeneous GNNs. Given its simplicity and wide adaptivity, it can facilitates future research on dynamic heterogeneous graph learning.
2. This work introduces two new benchmark datasets of dynamic heterogeneous graphs, including one dataset from e-commerce recommendation and one dataset from blockchain-based online social network. Specifically, the TaobaoTH is of a relatively large size with ~360k nodes.
3. The designed method achieves a better performance compared to the existing baselines including static GNNs, and dynamic GNNs.

### Weaknesses
1. Based on my understanding of the differences between dynamic graphs and temporal graphs, I think it would be better if this work is positioned for dynamic heterogeneous networks instead of temporal heterogeneous networks. Dynamic networks are snapshot-based networks, i.e., aggregating edges and nodes within certain time windows, which is exactly what this paper considers. In contrast, temporal networks are more dynamically changing where each edge is associated with a timestamp (not a snapshot).
2. It is not clear what scheme for the proposed method is applied in Table 2.

### Questions
1. What are the differences or new aspects between the existing Taobao benchmark (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Taobao.html#torch_geometric.datasets.Taobao) and the one introduced by this paper?

2. The number of edges of TaobaoTH is even smaller than the number of nodes. Can you elaborate why this graph is so sparse?

3. The evolutivity of TaobaoTH is extremely low. Does it mean there are very few new edges across snapshots? Or is it because at different snapshots, edges are repetitive (e.g., user viewed an item at snapshot-1 and viewed the same item at snapshot-2)? On this question, I think it's also worth reporting the repetitive metrics of the datasets.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes DURENDAL, a training framework for temporal heterogeneous networks. It introduced two training schemes, Update-Then-Aggregate and Aggregate-Then-Update, which are different aggregation methods for training. It then benchmarks the performance on four datasets.

### Strengths
1. Benchmarking dynamic heterogeneous graphs is important.
2. Two datasets are introduced by transforming the original open datasets.
3. Experiments on performed.

### Weaknesses
1. The mechanism of why DURENDAL outperforms baselines is unclear.
2. The comparison of UTA and ATU is not clear. System-level (e.g. run time, memory usage) evaluation might be helpful.
3. More commonly used datasets are needed if the paper wants to be a benchmark paper (e.g. Open Academic Graph).

### Questions
1. Where is the figure for Aggregate-Then-Update (ATU)?
2. Why does DURENDAL have better accuracy than baselines?
3. Why does not the paper compare with [1]?

[1] Hu, et al. "Heterogeneous graph transformer." Proceedings of the web conference 2020.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
