# Accurate Link Prediction via PU Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 8, 8

## Abstract
Given an edge-incomplete graph, how can we accurately find the missing links? The link prediction in edge-incomplete graphs aims to discover the missing relations between entities when their relationships are represented as a graph. Edge-incomplete graphs are prevalent in real-world due to practical limitations, such as not checking all users when adding friends in a social network. Addressing the problem is crucial for various tasks, including recommending friends in social networks and finding references in citation networks. However, previous approaches for link prediction rely heavily on the given edge-incomplete (observed) graph, making it challenging to consider the missing (unobserved) links during training. In this paper, we propose PULL (PU-Learning-based Link prediction), an accurate link prediction method based on the positive-unlabeled (PU) learning. PULL treats the observed edges in the training graph as positive examples, and the unconnected node pairs as unlabeled ones. PULL effectively prevents the link predictor from overfitting to the observed graph by proposing latent variables for every edge, and leveraging the expected graph structure with respect to the variables. Extensive experiments on five real-world datasets show that PULL consistently outperforms the baselines for predicting links in edge-incomplete graphs.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this study, the authors tackle the issue of link prediction in edge-incomplete graphs. They introduce PULL, a novel method that applies positive-unlabeled (PU) learning, considering observed links as positive and non-observed ones as unlabeled. PULL iteratively labels 
new edges with current link predictor. Extensive testing on real-world datasets confirms PULL's superiority over existing methods, marking a significant advancement in practical applications such as social media friend recommendations or academic citation analysis.

### Strengths
1. The authors study an important and practical problem as the fundamental assumption is that the un-present edges are not ALL negative.
2. The proposed method is simple and practical to implement on top of any link predictor.
3. The authors carry out empirical evaluation on five real-world dataset with comparison to several state-of-art methods.

### Weaknesses
1. Though the authors frame the problem and solution as PU learning, the proposed algorithm is an iterative method to carry out label augmentation. The technique have already been applied to several domains like CV and information retrieval.
2. The proposed hidden-variable framework is not necessary for the task if no edge potential on the link is assumed. It degenerates to a simple label augmentation with the current link predictor.
3. The empirical evaluation can be improved in following aspects: (1) As the proposed can be applied on top of any link predictor, it would be interesting to see whether the method can boost the performance of the baselines method; (2) Given the error bar, the proposed method is not statistical significant compared to baselines. Moreover, AUPRC would be a more sensitive metric under imbalanced labels. (3) All the graphs used in the experiments are small with ten of thousand of nodes. It would be interesting to see results on larger graphs.

### Questions
Please see weaknesses section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a novel method called PULL, which leverages Positive-Unlabeled Learning (PU learning) to train link predictors more effectively. The approach is backed by both theoretical analysis and empirical tests. The article's experimentation encompasses five real-world datasets, and the results indicate that PULL significantly outperforms existing link prediction methods, achieving state-of-the-art performance. By addressing the inherent limitations of conventional techniques, the PULL method offers a more robust and accurate solution for link prediction in edge-incomplete graphs.

### Strengths
1．The PULL method employs latent variables to account for hidden relationships between unconnected node pairs, thereby effectively addressing the inherent uncertainty in edge-incomplete graphs. This is particularly advantageous as missing links are prevalent in real-world network environments.
2. The article demonstrates that the PULL method achieves state-of-the-art performance on five different real-world datasets, outperforming existing link prediction algorithms. 
3. The inclusion of a theoretical analysis adds a layer of interpretability to the PULL method, enhancing its credibility and making it easier to understand its underlying mechanisms.

### Weaknesses
1. The PULL method incorporates latent variables and necessitates the construction of an expectation graph, potentially leading to high computational complexity. This could limit the method's scalability and applicability to large-scale datasets, requiring more computational power and time.

2. The selection of baselines for comparison could be broadened to include more recent methods, thereby providing a more comprehensive evaluation.

### Questions
Previous link prediction methods have relied too heavily on a given edge-incomplete graph and assumed that the edges of a given graph are all fully observed, without considering unobserved missing links. This approach causes the link predictor to overfit a given edge-incomplete graph, which reduces the accuracy of the prediction.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes PULL (PU-LEARNING-BASED LINK PREDICTOR), an accurate method for link prediction in edge-incomplete graphs. The PULL proposed with the thought that in real-world scenarios the presence of missing edges is frequently observed. Under this assumption, without the consideration of edge-incomplete graph, it may degrade the link prediction performance. Thus, it is important to consider the uncertainties of the given graph to obtain accurate linking probabilities between nodes. Expecting to propose PULL, this paper conducts a theoretical analysis of PULL, studying its relationship with the EM algorithm. The PULL achieves state-of-the-art link prediction performance. In summary, the contributions of this paper are as follows:
	Proposing PULL, an accurate method for link prediction in graphs
	Conduct a theoretical analysis of PULL, studying its relationship with the EM algorithm.
	Through the experiments on five datasets, PULL achieves state-of-the-art link prediction performance
The manuscript demonstrates notable originality with the introduction of a new method named PULL. This method has potential implications in the domain of graph data mining, which could be of considerable significance. In terms of linguistic presentation, the article is well-composed. The flow between sentences appears logical, making the content coherent and accessible to the reader. The authors have taken care to provide a structured proof of PULL, which strengthens the paper's theoretical grounding. Regarding the experimental section, the authors have provided a detailed description of their experiments, followed by an in-depth analysis of the data. 
Overall, this paper could be a significant algorithmic contribution, I would be willing to increase the score.

### Strengths
Introduce the consideration of edge-incomplete graph, which has a significant contribution
The theoretical analysis and extensive experiments support the claims of the paper
The authors provide the code and datasets, enhancing the reproducibility of the results

### Weaknesses
The paper could provide more detailed explanations of the experimental setup, such as the hyperparameter settings and the choice of evaluation metrics.

### Questions
How does the proposed method handle large-scale graphs？ Are there any scalability issues？
Are there any limitations or potential drawbacks of the proposed method that need to be considered？
Can the proposed method be extended to handle multi-relational graphs？

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose the PULL method for link prediction using positive-unlabeled (PU) learning. The basic idea is to treat all edges as positive samples while treating all non-edges as unlabeled rather than negative samples. They then use a graph convolutional network (GCN) as a link predictor to predict edge probabilities for all the non-edges (unlabeled samples) and turn the ones with the highest probabilities into edges for the next iteration of learning. (In this sense, it shares some similarities with self learning, where predictions are used as labels to re-train a classifier.) The authors further demonstrate an equivalence between their learning algorithm and the well-known expectation-maximization (EM) algorithm under an independence assumption. They show improved link prediction accuracy on several real data sets and demonstrate that they typically achieve maximum accuracy at around the true number of edges in a graph.

*After author rebuttal:* The authors have done a thorough job of answering the reviewers' questions and addressing their concerns with a much improved revision. I have raised my score to 8 to reflect the improvement in quality.

### Strengths
- Creative approach using PU learning for link prediction, a task that seems to be an ideal fit for the assumptions of PU learning.
- Detailed empirical investigation of several research questions that go beyond improvement in accuracy. I found the analysis of accuracy both with and without the $\mathcal{L}_C$ term as a function of the number of iterations to be very insightful.
- Well written paper that was easy to understand and explains the PULL method with an appropriate level of detail.

### Weaknesses
- While the overall framework is quite principled, some ad-hoc tweaks seem to be necessary to get it working well, such as the way the number of selected edges $K$ is increased or the need for the additional loss function $\mathcal{L}_C$.
- ~~There is some other related work on PU learning for link prediction that is not cited (Gan et al., 2022; chapter 4 of Hao, 2021). While they also use PU learning, the approaches seem to be different from what the authors propose, so I don't think it limits the novelty of this paper and find it to be only a minor weakness.~~ This weakness was addressed by the authors in their revision during the discussion phase and no longer applies.

References:
- Gan, S., Alshahrani, M., & Liu, S. (2022). Positive-Unlabeled Learning for Network Link Prediction. Mathematics, 10(18), 3345.
- Hao, Y. (2021). Learning node embedding from graph structure and node attributes (Doctoral dissertation, UNSW Sydney).

### Questions
1. Hao (2021) present their PU learning-based link predictor as a wrapper that could be applied to a variety of different GNNs. Could your proposed PULL approach generalize beyond the GCN that you use in this paper? If so, I think that would be a better comparison--apply your PU learning approach to each GNN and then compare the accuracy of your PU-based model with the normally trained model.
2. Why keep the top $K$ with the highest predicted probabilities? It seems like taking some sort of random sample based on the probabilities might be a better approach to avoid too much self reinforcement, which could lead to oversmoothing. Taking a random sample could eliminate the need for the $\mathcal{L}_C$ term.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
