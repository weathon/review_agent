# Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks

- Avg Score: 5.67
- Decision: Accept (poster)
- Scores: 6, 5, 6

## Abstract
While graph neural networks (GNNs) are widely used for node and graph representation learning tasks, the reliability of GNN uncertainty estimates under distribution shifts remains relatively under-explored. Indeed, while post-hoc calibration strategies can be used to improve in-distribution calibration, they need not also improve calibration under distribution shift. However, techniques which produce GNNs with better intrinsic uncertainty estimates are particularly valuable, as they can always be combined with post-hoc strategies later. Therefore, in this work, we propose G-$\Delta$UQ, a novel training framework designed to improve intrinsic GNN uncertainty estimates. Our framework adapts the principle of stochastic data centering to graph data through novel graph anchoring strategies, and is able to support partially stochastic GNNs. While, the prevalent wisdom is that fully stochastic networks are necessary to obtain reliable estimates, we find that the functional diversity induced by our anchoring strategies when sampling hypotheses renders this unnecessary and allows us to support G-$\Delta$UQ on pretrained models. Indeed, through extensive evaluation under covariate, concept and graph size shifts, we show that G-$\Delta$UQ leads to better calibrated GNNs for node and graph classification. Further, it also improves performance on the uncertainty-based tasks of out-of-distribution detection and generalization gap estimation. Overall, our work provides insights into uncertainty estimation for GNNs, and demonstrates the utility of G-$\Delta$UQ in obtaining reliable estimates.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper extends the ∆-UQ model to graph learning to enhance intrinsic GNN uncertainty estimates. It proposes several training techniques to introduce partial stochasticity for node and graph-level classification tasks. Extensive experimental results validate the effectiveness of the proposed method. Interestingly, the graph-level training method can be applied to pre-trained graph models.

### Strengths
1. Extensive experimental results on various datasets and setups demonstrate the effectiveness of the proposed method.

2. The introduced method is fairly simple and easy to understand. It seems that it can be applied to many existing models in a plug-and-play manner, making it flexible and extendable.

3. The paper also explores the potential of applying this method to pre-trained models, which is an interesting aspect.

### Weaknesses
1. Some notations and definitions in the paper are unclear, such as the symbols used for samples and distributions, and the shape of tensors during concatenation.

2. The experiments conducted focus solely on classification tasks. Consideration for regression tasks seems to be missing.

3. There are limited performance gains observed with the use of pre-trained models.

These points are further detailed below.

### Questions
1. In Section 2 (Preliminaries), the notation $\\mathbf{X}$ is somewhat unclear. It seems to represent some distributions, but there is also the concatenation $[\\mathbf{X} - \\mathbf{C}, \\mathbf{C}]$, which is described as "channel-wise concatenating two images". Please clarify the notations regarding distributions, samples and training (test) sets.

2. On page 3 (Section 3.1 Node Feature Anchoring), where are $\\mathbf{A}_c$ and $Y_c$ defined?

3. On page 4, "the anchor/query node feature pair" is defined as $[\\mathbf{X}_i − \\mathbf{C}|| \\mathbf{X}_i]$, where $\\mathbf{C}^{N \\times d} \\sim \\mathcal{N} (\\mu, \\sigma^2)$. How are the shapes of $\\mathbf{X}_i$ and $\\mathbf{C}$ aligned?

4. For the node classification task, anchoring is sampled from a learned Gaussian. The notations used seem unclear to the reviewer. Could you elaborate on the learning procedures, the loss function used, and the dimensions of the Gaussians? During inference, stochasticity arises from sampling from such Gaussians. Is it possible to display qualitative results of the learned Gaussians, including details such as variance and mean?


5. a) For the graph classification task, there seems to be no extra learnable component to create the anchors, but the subsequent MPNN layers are modified to accept features of $d_r \\times 2$ dimensions. How does this subtle change affect the number of trainable parameters?\
b) The stochasticity appears to arise from the random shuffle of the node features across the entire batch, and the subsequent selection of anchoring $\\mathbf{c}^{1 \\times d}$. Does creating anchors from a Gaussian distribution work in this case?\
c) Conversely, is it feasible to use intermediate representations to construct anchors for the node classification task? Technically, it seems like these could substitute for the learned Gaussian.\
d) Furthermore, when randomly shuffling the node features across the entire batch (potentially involving cross-graph shuffling), are there observable impacts attributed to the size of the mini-batch?

6. Can this method be generalized to graph/node/edge level regression tasks?

7. The design of shuffling across batches in this paper appears conceptually somewhat similar to the mixup method [1] in the feature space. A comparison between them would be interesting.

8. In Fig. 6, Tables 4/5, the effects of applying G-∆UQ on top of the pretrained model (GPS) seem very marginal.

[1] mixup: Beyond Empirical Risk Minimization, ICLR 2018.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposed a new training framework inspired by the stochastic anchored training in computer vision. In the framework two (partial) stochastic anchoring techniques (node feature anchoring and hidden layer anchoring) are designed for sake of the better uncertainty estimation and calibration in GNNs. The experiments on node classification and graph classificaiton validate the effectiveness of the framework.

### Strengths
1. The idea that introducing the anchoring training in GNNs is new and interesting.

2. The paper conducted comprehensive evalutions on the calibration of GNNs under different settings. Specifically, the evaluation on calibration of GNNs under distribution shift is new and significant. The experimental results show that on most tasks the proposed method can achieve lower calibration error.

### Weaknesses
1.Since the paper focused on calibration and uncertainty estimation of GNNs. The concepts of calibration and uncertainty estimation of GNNs should be introduced in the paper.

2.The paper doesn't provide sufficient discussion or theories to justify the methods provided in the paper. For instance,  in node feature anchoring why authors sample the anchors from the Gaussian Distribution? How to determine the value of $u$ and $\sigma$?

3.The paper claimed that the framework can improve the uncertainty estimation. However, how the method can improve the uncertainty estimation is still unclear.

4.The organization of the experiment part is messy. More setup details should be clarified.

5.Some typos in the paper and the title of Table 2 is missing.

### Questions
See weakness

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a training framework for GNNs that is designed to improve the intrinsic uncertainty estimates. The adopted strategy is to adapt the principle of stochastic data centering to graph data. This involves introducing novel graph anchoring techniques. The paper demonstrates that the methodology can support partially stochastic GNNs. Experimental results in the paper suggest that the partial stochasticity is sufficient; it also has the advantage of providing a mechanism for incorporating pre-trained models. The paper reports experiments investigating the impact of covariate, concept and graph size shifts and demonstrates that the proposed technique leads to better calibrated GNNs, both for node and graph classification. Additional experiments illustrate how the approach performs for out-of-distribution detection and generalization gap estimation.

### Strengths
S1. The paper introduces a novel approach for improving the intrinsic uncertainty estimates of GNNs by translating the stochastic centering strategy to the graph domain. This is non-trivial, both for node and graph classification.
 
S2. The paper reports on experiments exploring (i) node classification under distribution shift (concept and covariate shifts); (ii) calibration under distribution shift for graph classification; (iii) how the approach impacts the calibration of more expressive models such as graph transformers. The experiments are thorough and examine multiple interesting questions. 

S3. The experiments support the interesting observation that the network does not need to be fully stochastic in order to provide improved uncertainty estimates. This paves the way to combine the p

### Weaknesses
W1.	Some of the methods employed to translate stochastic centering to the graph domain appear somewhat heuristic, or are at least the text describing the methodology does not provide sufficient detail to perceive the design principles. For example, the node feature anchoring fits a Gaussian distribution, but there is no explanation as to why a Gaussian is selected and no discussion as to whether the mismatch between the fitted anchor distribution and the feature distribution has a negative impact or could be a concern. The text states that the introduction helps to “manage the combinatorial stochasticity induced by message passing” but it does not elaborate on this to explain why or how. For the graph anchoring, there is a random shuffling of the node features over the entire batch. There is no discussion of this design choice – it doesn’t seem obvious to me that this is the only thing one could choose to do (or the optimal). 

W2.	Distilling the methodological contributions, we see that they involve: (i) node anchoring via fitting a Gaussian distribution and drawing an anchor from this fitted distribution; (ii) hidden layer anchoring by randomly shuffling the node features after the r-th layer. After these steps to construct appropriate anchors in the graph domain, there is effectively a standard application of the stochastic centering approach. The technical methodological contribution is thus not particularly substantial. On the other hand, the experiments are thorough and provide a good balancing contribution. 

W3.	Some of the results are not presented in a particularly helpful manner and are not described or discussed in much detail. For example, the observations for node classification essentially boil down to “our method works better”. The table contains interesting elements such as the proposed method failing to improve (WebKB, CBAS – Concept) or substantially increasing (Cora) the ECE when combined with Dirichlet calibration. But there is no discussion of this. In general there is not a significant attempt to draw detailed conclusions from the obtained results – similar comments apply to Section 5.3 where again the conclusions are “both pretrained and end-to-end G-∆UQ outperform the vanilla model on 7/8 datasets” and “G-∆UQ variants improve OOD detection performance over the vanilla baseline on 6/8 datasets”. Insights beyond “works better” make a paper much stronger and more insightful. 

There is room for improvement in some of the figures and the explanations of how they are being interpreted. Figure 2 is a particular case in point – L1, L2, L3, N/A are not clearly defined. The text states that “READOUT anchoring generally performs well” but does not explain how we should interpret the figure to come to this conclusion. It’s not obvious what “performs well” means – what is an acceptable deterioration in performance. The behaviour over datasets and architectures differs considerably and should be discussed. 

In terms of assessing the variability of performance, the paper reports standard deviations over a few trials, but does not make any attempt to assess the statistical significance of the results or to specify confidence intervals on the reported means.

### Questions
Q1. Why does the proposed method appear to work relatively poorly (at least in terms of often making ECE worse and sometimes considerably worse) when used in conjunction with Dirichlet calibration in the node classification setting?
Q2. Figure 2 is challenging to understand. Insufficient detail is provided. What is the meaning of L1, L2, L3, N/A? What does it mean “generally performs well across datasets and architectures” – which bars am I comparing to draw this conclusion? The accuracy and ECE behaviour seems quite inconsistent across different datasets and architectures – sometimes increasing, sometimes decreasing, sometimes going up then down. How do I determine when something “performs well”?
Q3. “We emphasize that this distribution is only used for anchoring and does not assume that the dataset’s node features are normally distributed.” This sentence seems to imply that it does not matter if the anchoring distribution matches the data distribution or not. But then why fit the Gaussian distribution to the training data? The fitting procedure implies that there is a need to have an anchor distribution that matches (at least to some degree) the feature distribution. So how close does it need to be to a Gaussian distribution?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
