# Structural-Entropy-Based Sample Selection for Efficient and Effective Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 8

## Abstract
Sample selection improves the efficiency and effectiveness of machine learning models by providing informative and representative samples. Typically, samples can be modeled as a sample graph, where nodes are samples and edges represent their similarities. Most existing methods are based on local information, such as the training difficulty of samples, thereby overlooking global information, such as connectivity patterns. This oversight can result in suboptimal selection because global information is crucial for ensuring that the selected samples well represent the structural properties of the graph. To address this issue, we employ structural entropy to quantify global information and losslessly decompose it from the whole graph to individual nodes using the Shapley value. Based on the decomposition, we present $\textbf{S}$tructural-$\textbf{E}$ntropy-based sample $\textbf{S}$election ($\textbf{SES}$), a method that integrates both global and local information to select informative and representative samples. SES begins by constructing a $k$NN-graph among samples based on their similarities. It then measures sample importance by combining structural entropy (global metric) with training difficulty (local metric). Finally, SES applies importance-biased blue noise sampling to select a set of diverse and representative samples. Comprehensive experiments on three learning scenarios --- supervised learning, active learning, and continual learning --- clearly demonstrate the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a sample selection method with structural entropy. It first introduces the structural entropy of a graph into the sample selection task, and then applies the Shapley value to decompose the structural entropy of a graph to the node-level structural entropy. At last, it designs a score for sample selection with the node-level structural entropy.

### Strengths
1. The idea is interesting and reasonable.
2. The experiments are sufficient and the experimental results are good.

### Weaknesses
1. The main techniques used in this paper are all existing methods, e.g. the structural entropy and Shapley value. The paper seems an application of these techniques in the sample selection task. The paper should clarify its novelty and technical contributions.

2. The paper derives Eq.(3) from the Shapley value, and then removes the second term, leading to Eq.(4), which is used in the method. However, the motivation is unclear. Why should we remove the second term? What is the advantage of removing this term? The ablation study of this term should be conducted in the experiments. The paper only claims that they want to design a global metric and the second term is a local metric. I do not think it's convincing enough. What is the advantage of only considering global information? Moreover, in Eq.(5), the method combines the global score S_e and the local score S_t. It seems a little contradictory. In Eq.(4), they do not want the local term but in Eq.(5) they need the local term. It seems strange and not well-motivated.

3. Eq.(5) needs more justification. Why should we multiply S_e and S_t? What is the advantage? Why not just sum S_e and S_t, or combine them with other forms? I think more ablation study is needed to justify the effectiveness of Eq.(5).

4. In the experiments, the experimental setup should be introduced in more detail. For example, what is the difference between the supervised setting and the active learning setting? They both need to select some samples to train the model and then is used to predict the test data.

### Questions
Please see Weaknesses.

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
This paper proposes a novel sample selection method that can capture both local and global information of data simultaneously. Specially, first, they use a $k$-NN to construct the sample graph of original data. Second, they employ structural entropy to measure global information and use training difficulty to capture the local information of the graph. At lats, they utilize the importance-biased blue noise sampling to select a set of diverse and representative samples. 

The main contribution of this paper is to propose a new global metric for sample selection.

### Strengths
1.The writing of this paper is good and easy to follow.

2.This paper has a certain degree of innovation, introducing a structural entropy as a global metric for sample selection.

### Weaknesses
1. Hypergraphs [1] can capture more complex relationships, and I believe that if the authors had used hypergraphs instead, the results would have been even better. I would be delighted to see the authors add new experiments to verify my hypothesis.

2. How would the authors define informative and representative samples? They mention these concepts multiple times, but do not provide a detailed explanation.

3. Taking sample as a node is not very reasonable, as it treats the sample as a whole and ignores the local information contained in the sample itself. Perhaps using a Hyper-class representation [2] would yield better results. I would be delighted to see the authors conduct new experiments to verify my hypothesis.

4. The experimental results do not show the variance, so I hope the authors can make up the variance.

5. Just a little confused, why most of the comparison algorithm's experimental results are not as effective as random selection? So what's the meaning of those comparison algorithms? Or are those comparison algorithm chosen by the author appropriate? Are the parameters of the those algorithms not optimally tuned?

6. The effect of blue noise sampling (BNS) is not as good as message passing (MP) when the sampling rate is greater than or equal to 20%, why not directly use MP? Simply because BNS does not require hyperparameter tuning, it sounds far-fetched.

7. As can be seen from Figure 4, (b) has better boundary division, while (c) has several categories of samples grouped together. It seems that (b) is better. How can we understand this phenomenon?

[1] Learning with hypergraphs: Clustering, classification, and embedding.

[2] Hyper-class representation of data.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

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
The paper proposes a Structural-Entropy-based sampling (SES) method for data selection. This approach integrates global structural information, quantified using structural entropy, and local information (training difficulty) to choose informative and representative samples for training. The authors show that  incorporating global structural information can improve the quality of sample selection. Traditional methods often focus on local data properties, such as training difficulty, while ignoring broader connectivity patterns. SES constructs a sample graph using k-nearest neighbor (kNN) relationships to model sample similarity and applies structural entropy at a node level to evaluate each sample’s significance in preserving the overall data structure.

### Strengths
1, The method utilizing the encoding tree effectively measures local structural information across different scales, providing valuable insights.  Various experiments on different learning tasks demonstrate SES’s effectiveness.

2, Structural entropy is decomposed from the overall graph level to individual nodes, resulting in a node-level definition of structural entropy. By leveraging properties of the Shapley value, authors show that the Shapley value of a node can be computed in linear time relative to the number of edges.

### Weaknesses
1, The overall idea is relatively simple and incremental, mainly based on the concept from Li and Pan 2016. Also, the proposed method is very heuristic without solid theoretical validation. For example, in line 212-213, "Given our emphasis on ..., we only use....". Can you give more detailed explanation? In my opinion, this is not a serious claim for a research article.  In line 352, you say "10X speedup" because you use only 10\% of dataset. Can you provide the realistic experimental time for this claim? Do you count the construction time of your sample? 

2, While the Shapley value of a node can be computed in linear time relative to the number of edges, the edges in a kNN graph is O(k|X|). It is still  time consuming to calculate the Shapley values for all nodes of the encoding tree, especially for large dataset.

3, The author does not clearly explain why preserving the global structure of the dataset is beneficial or provide theoretical guarantees regarding its impact on performance in tasks such as supervised, active, and continual learning. Similarly, the rationale for prioritizing samples with high local information (i.e., training difficulty) is insufficiently justified.

4, The blue noise sampling method effectively promotes diversity, yet the balance between selecting challenging samples and maintaining sample diversity could be further clarified. A comparative analysis with methods that explicitly optimize for diversity, such as clustering-based selections, would help to clarify the advantages of SES.

5, The method relies on the quality of the embedding used to construct the kNN graph. If the embedding representation is poor, structural entropy may not accurately capture the global structure of the data.

6, Some experimental parts are not sufficient. For example, in the continual learning part, they only consider three baselines and two memory sizes (100, 200). Moreover, do you consider both Class-Incremental Learning and Task-Free Continual Learning?

### Questions
1, The construction method of the Encoding tree is challenging. The author mentioned the Huffman tree construction; how does this affect the conclusions of this article?

2, Equation (5) in the paper calculates the overall importance score of the node. Is this importance score node-wise or point-wise sampling? A data point may appear in multiple nodes, does this affect the proposed sampling method?

3, Have you considered other ways of combining global structural entropy and local training difficulty indicators, rather than through the multiplication method (S(u) = Se(u) * St(u))?

4, Can you provide more theoretical explanations or proofs to demonstrate why structural entropy, as a global indicator, is helpful for considered tasks (e.g. supervised, active, and continual learning)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies sample selection which aims to extract a small, representative subset from a larger dataset. The authors introduce a novel sample selection scheme, termed Structural-Entropy-based Sample Selection (SES), which uses and extends the concept of "structural entropy" (Li and Pan, 2016), which assesses how nodes and edges within a graph are hierarchically organized to form multi-level communities. The proposed scheme seeks to address the limitations of existing selection methods, which often prioritize local information and neglect the broader, global context of the samples.

The algorithm begins by constructing a k-NN graph G for the dataset based on similarity. It then calculates the structural entropy value for each node in G (referred to as node-level structural entropy). While structural entropy was originally defined for an entire graph rather than individual nodes, the authors extend this concept using the Shapley value method. Roughly speaking, node-level structural entropy calculates the average increase in structural entropy when a node is added to all potential subgraphs of G. After that, each node is assigned an importance value, which is the product of its structural entropy and training difficulty. An importance-biased blue noise sampling method is then used to select samples. instead of sampling solely based on importance scores, this sampling process prevents the selection of overly similar samples, thus maintaining diversity within the selected subset.

The effectiveness of the SES scheme is demonstrated through experimental studies and compared to other selection methods in various tasks such as supervised learning, active learning, and continual learning.

### Strengths
Overall, the proposed algorithm is intriguing due to its general empirical performance and some of the novel ideas it introduces. The concept of node-level structural entropy, which quantifies the contribution of an individual node to the global structural entropy of a graph, could be of independent interest. The experiments indicate that the proposed method outperforms existing selection methods in many common learning tasks. Furthermore, ablation studies validate the contribution of each module (node-level structural entropy and importance-biased blue noise sampling) to the overall effectiveness.

### Weaknesses
I believe the results could benefit from stronger theoretical justification. While the high-level ideas are reasonable, the mathematical properties are not mentioned in this paper. A deeper discussion on these aspects should be provided. Especially I am interested in what are the mathematical properties of the node-level structural entropy.

### Questions
Are there anything you can prove about the node-level structural entropy values?

### Soundness
3

### Presentation
3

### Contribution
3
