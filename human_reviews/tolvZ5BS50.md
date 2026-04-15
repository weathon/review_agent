# ELU-GCN:  Effectively Label-Utilizing Graph Convolutional Network

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3

## Abstract
The message-passing mechanism of graph convolutional networks (i.e., GCNs) enables label information to be propagated to a broader range of neighbors, thereby increasing the utilization of labels. However, the label information is not always effectively utilized in the traditional GCN framework. To address this issue, we propose a new two-step framework called ELU-GCN. In the first stage,  ELU-GCN conducts graph learning to learn a new graph structure (i.e., ELU-graph), which enables GCNs to effectively utilize label information. In the second stage, we design a new graph contrastive learning on the GCN framework for representation learning by exploring the consistency and mutually exclusive information between the learned ELU graph and the original graph.  Moreover, we theoretically demonstrate that the proposed method can ensure the generalization ability of GCNs. Extensive experiments validate the superiority of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces the ELU-GCN framework to enhance the effective utilization of label information in Graph Convolutional Networks (GCNs).  By optimizing the graph structure to create an ELU-graph and employing contrastive learning to extract complementary information between the original and new graphs, ELU-GCN significantly improves the supervision of unlabeled nodes.  Experimental results show that ELU-GCN improves the generalization ability of the model and is superior to the existing methods on multiple datasets.

### Strengths
1. ELU-GCN demonstrates effectiveness across heterophilic and homophilic graphs.
2. The proposed ELU-graph ensures that labels influence both labeled and unlabeled nodes more effectively, resulting in improved representation learning.
3. By integrating contrastive loss, ELU-GCN extracts complementary information between the original and optimized graphs, preserving essential information while filtering out noise. 
4. The theoretical analysis and experimental results of this paper are complete.

### Weaknesses
1. The framework relies on various hyperparameters that need careful tuning. 
2. Experimental evaluation indicators are relatively homogenous.
3. Insufficient abstracts and introductions.

### Questions
1. The abstract should be enriched with a brief description of the shortcomings of the past methods and the innovative nature of the proposed method.
2. In the introduction, LU-GCN is mainly divided into three categories, but it is also mentioned that the above two methods ignore the critical importance of neighboring label utilization in semi-supervised scenarios, which creates ambiguity. 
3. The classification accuracy results in Fig. 2 are introduced abruptly, without explaining the experimental setup.
4. The parameter sensitivity of the objective function and the convergence analysis experiment can be added appropriately.
5. The visualization of Fig. 3 can be compared by adding some plots from the ablation experiments.                                   
6. the evaluation indexes used in Table 1 and Table 2 are not reflected, and the evaluation indexes are relatively single.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The proposed ELU-GCN framework enhances traditional GCNs by improving label information utilization through a two-step process. First, it learns a new graph structure (ELU-graph) that better integrates label data. Then, a graph contrastive learning method is applied to improve representation learning, ensuring generalization and achieving superior experimental results.

### Strengths
- The paper is clearly written and easy to understand.
- The theoretical analysis is appealing.
- Experimental performance is superior than selected baselines.

### Weaknesses
- The novelty is limited.
- The motivation is not convincing and well thought out.
- The experiments are not comprehensive.

### Questions
- Novelty and motivation. Several works have studied the GNN with label propagation, e.g., GCN-LPA, which has also be adopted as baseline.
- Following the first comment, the paper overstates its contributions, as it is not the first to address GCN limitations in label information utilization.
- According to Definition 2.2, if "the prediction of GCN is consistent with the output of LPA," one might question why not directly use LPA predictions, given their importance. And this may also be the reason why this work does not provide superior performance (compared to the real SOTAs), as Definition 2.2 gives a false direction to GCN.
- Technique contribution is minor, generally speaking, this work is a supervised graph contrastive learning with LPA as data augmentation.
- So many baselines within the graph structure learning and graph contrastive learning domains are missing, weakening comparative analyses.
- Detailed experiment setup is missing, such as the GCN layer, the learning rate, the training epoch and so on. 
- It is unclear how baseline results were obtained, particularly as the NodeFormer results are significantly lower than reported in its original paper.
- In Sec. 3.2.4, the definition of generalization gap is not a good metric for model generalization ability, as there is a trivial solution when the training loss and validating loss are the same without convergence. Small loss gap does not necessarily lead to good generalization ability.
- Actually, as shown in Figure 4, the validation loss of ELU-GCN is worse than GCN, which may even suggest that GCN is much better.
- No hyper-parameter analysis for $\beta$ in Eq.7|8|9, threshold in Line 256, and $\eta$ in Eq.11.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces ELU-GCN, a two-step framework designed to enhance the utilization of label information in the scenario of semi-supervised learning. In the first step, ELU-GCN learns a graph structure, termed the ELU graph, to improve the label propagation and utilization in GCNs. In the second step, the framework incorporates graph contrastive learning to enhance representation learning by examining the consistency and exclusive information between the ELU graph and the original graph.

### Strengths
- The authors highlight an interesting perspective on GNNs, the utilization of label information, and provide experiments to aid understanding
- The paper is well-written, with a clear and easy-to-follow presentation that effectively conveys the authors' motivation and methodology
- Provide some theoretical analyses.

### Weaknesses
- The motivation is somewhat confusing. Among the two challenges mentioned by the authors, the first has already been well-discussed in [1], while the second is closely related to graph homophily/heterophily. Homophily is defined based on true labels, and the graph structure this work aims to learn is actually highly homophilous. I am concerned about whether the label-based perspective can provide insights beyond the existing homophily/heterophily frameworks.
- The authors have conducted some theoretical analyses, but some of these do not seem to effectively support the proposed method or the authors' claim. For instance, regarding Theorem 2.1, the authors need to clarify what additional contribution it provides beyond [1]. By Theorems 2.3 and 2.4, the authors conclude that their proposed method ensures good generalization ability of GNNs. However, this conclusion is based on the assumption in Appendix B.5 that the predictions of the pre-trained GCN closely match the true labels. This raises questions, as it suggests that if the predicted labels are indeed close to the true labels, the original graph structure should already be adequate, making further learning of the graph structure unnecessary. Conversely, if the predicted labels differ significantly from the true labels, Theorem 2.4 would no longer hold.
- The scalability of the proposed method requires further discussion. A theoretical analysis of its time and space complexity is necessary. Additionally, the datasets used in the experiments are now quite small. It is recommended to conduct experiments on the OGB datasets, at least including the OGB-arXiv dataset.
- The discussion of related work and the setup of baseline methods are insufficiently comprehensive. The proposed method involves LPA, GCL, and graph heterophily, yet I found few relevant discussion on these, making it difficult to position this paper. Additionally, as mentioned earlier, the authors’ perspective is highly related to heterophily and incorporates GCL techniques, so the comparison should include GCL models, heterophilous GNNs, and some SOTA methods like GCNII.

[1] Unifying Graph Convolutional Neural Networks and Label Propagation.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper proposed a new two-step framework called ELU-GCN to effectively utilize the label information. Extensive experiments validate the superiority of the proposed method.

### Strengths
The idea of this paper is very good, and the two-stage approach is very interesting. The experimental results look quite competitive.

### Weaknesses
The formulas and descriptions in this article are difficult to understand, and I don’t fully understand them (it may also be because I am not particularly familiar with this field). Therefore, I will skip the weaknesses and mainly list my comments on this article in the question section.

It may be that I overlooked some properties when deducing. If the author can convince me, I am willing to improve my score.

### Questions
> **Q1. Confirmation of symbols.**

- "The first $m$ points $x\_i\, (i \leq m)$ are labeled as $Y\_l$": Does the subscript $l$ mean that the entire $Y$ has a label? Or is it an index for traversal? Because the first part writes each $\mathbf{x}\_i$ individually, and the second part writes the whole $\mathbf{Y}\_l$, it makes people think that it might be a typo of $\mathbf{Y}\_i$.
- "Given a classification function $f:\mathbf{X}\to\mathbb{R}^{n\times c}$, ..." : it looks like it should be changed to $f:\mathbb{R}^{n\times n}\to\mathbb{R}^{n\times c}$.
- $V\_l$, $f\_{ik}$ and $f\_{jk}$ in Eq. (1): What do these notations denote? Authors should introduce them.
- Eq. (10):  Since it is a logarithmic form, it is recommended not to use the fractional form, which is indeed not conducive to understanding. In addition, according to line 188 in test-flan.py in the anonymous link, $e^{\overline{\mathbf{P}}^\top\overline{\mathbf{P}}+\tilde{\mathbf{P}}^{\top}\tilde{\mathbf{P}}}$ should be taken element by element instead of the matrix exponential. The author should modify the form or make special instructions.
```python
intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
```

> **Q2. Theorems and definitions are ambiguous.**

- Theorem 2.1: Can this theorem be expressed in formula form?
- Definition 2.2: The GCN effectively utilizes label information if the prediction of GCN is consistent with the output of LPA, i.e., 
$$V\_{\mathrm{ELU}} = \\{V |\texttt{LPA}(\mathcal{G}) = \texttt{GCN}(\mathcal{G})\\}.$$
It seems that Def 2.2 is a definition of a GCN, but Eq. (2) seems confusing, is the GCN should be consistent with the output of LPA on any graph $\mathcal{G}$, or the $\mathcal{G}$ is a given one? Moreover, what does Eq. (2) mean? Does it mean a subset of the whole node set $V$ belonging to the given $\mathcal{G}(V,E,\mathbf{X},\mathbf{Y})$? The current definitions are too vague, and I would be very happy if the author could explain to me in detail what they really mean.

> **Q3: The derivation of Eq. (6) seems to lack some support.**

- "...,  the second term can make the subsequent matrix inversion more stable. " What is the subsequent matrix, it seems to refer to $(\mathbf{HH}^\top+\beta \mathbf{I})$ in Eq. (7)? Or the matrix $S$? Judging from the proof in the appendix, I tend to believe the former explanation. It seems that this is the form of Eq. (7) first and then Eq. (6) pieced together. Why choose this unnatural form in the narrative?
- Since the Frobenius norm has been adopted, why is the regularization term not simplified by the Frobenius norm (and the summation term loses the superscript)? As a result, there is a typo ($\beta\mathbf{S}$ should be $\beta\mathrm{Tr}(SS^\top)$) in the second line of Eq. (18).
- Eqs. (19-20) seem a bit strange. The result of Eq. (19) I calculated is 
$$\frac{\partial \mathcal{L}}{\partial \mathbf{S}}=2(\mathbf{Q}^{(i)}-SH)(-\mathbf{H}^{\top})+2\beta \mathbf{S}=\textcolor{red}{-2\mathbf{Q}^{(i)}\mathbf{H}^{\top}}+2\mathbf{SHH}^{\top}+2\beta \mathbf{S},$$
And the Eq. (20) should be 
$$\mathbf{S}=\textcolor{red}{\mathbf{Q}^{(i)}\mathbf{H}^{\top}}(\mathbf{HH}^{\top}+\beta \mathbf{I})^{-1}.$$
I don't know whether $\mathbf{Q}^{(i)}\mathbf{H}^{\top}$ is a symmetric matrix, so that it can return to the form used by the author. It seems that this proposition does not hold. This formula seems to involve the core of this article, so I was unable to continue reviewing subsequent derivations and experimental results. However, it is worth mentioning that since the matrix shapes are aligned, whether my or the author's derivation is wrong, the derivation of complexity seems to be correct.

### Soundness
3

### Presentation
1

### Contribution
2
