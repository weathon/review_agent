# Scalable and Certifiable Graph Unlearning: Overcoming the Approximation Error Barrier

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6

## Abstract
Graph unlearning has emerged as a pivotal research area for ensuring privacy protection, given the widespread adoption of Graph Neural Networks (GNNs) in applications involving sensitive user data. Among existing studies, certified graph unlearning is distinguished by providing robust privacy guarantees. However, current certified graph unlearning methods are impractical for large-scale graphs because they necessitate the costly re-computation of graph propagation for each unlearning request. Although numerous scalable techniques have been developed to accelerate graph propagation for GNNs, their integration into certified graph unlearning remains uncertain as these scalable approaches introduce approximation errors into node embeddings. In contrast, certified graph unlearning demands bounded model error on exact node embeddings to maintain its certified guarantee.

  To address this challenge, we present ScaleGUN, the first approach to scale certified graph unlearning to billion-edge graphs. ScaleGUN integrates the approximate graph propagation technique into certified graph unlearning, offering certified guarantees for three unlearning scenarios: node feature, edge and node unlearning. 
  Extensive experiments on real-world datasets demonstrate the efficiency and unlearning efficacy of ScaleGUN. Remarkably, ScaleGUN accomplishes $(\epsilon,\delta)=(1,10^{-4})$ certified unlearning on the billion-edge graph ogbn-papers100M in 20 seconds for a 5,000 random edge removal request -- of which only 5 seconds are required for updating the node embeddings -- compared to 1.91 hours for retraining and 1.89 hours for re-propagation. Our code is available at https://github.com/luyi256/ScaleGUN.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper examines and highlights the efficiency issue of existing certified graph unlearning approaches. The authors show that the culprit is the costly re-computation of graph propagation. To address this challenge, the authors propose ScaleGUN, an efficient certified graph unlearning method that integrates the approximate graph propagation technique to speed up the process. Despite being introduced with additional approximation errors, ScaleGUN's model error is bounded with theoretical guarantees.

### Strengths
1. Certified graph unlearning is an important research topic with potentially broach impact in both academia and industry.
2. The theoretical analyses and proofs in this paper are mostly rigorous.

### Weaknesses
1. The assumption, analysis, and proposed methodology seem to be limited to SGC-like models (GNNs with pre-computed graph propagation). The efficiency issue may also not be evident when the backbone model is not SGC (e.g., multi-layer GCN).
2. The main contribution of this paper mainly lies in theory, while the methodology part seems to be just a simple combination of InstantGNN and CGU.
3. The clarity and presentation of this paper can be further improved. For example, it feels quite strange when the authors suddenly bring up PageRank in Line 100 without any bedding or motivation.

### Questions
1. Would the efficiency issue still be a problem when CGU/CEU uses other GNNs as the backbone model (e.g., multi-layer GCN)? Is ScaleGUN compatible with other GNN models? Would the model error of ScaleGUN be still bounded when using other GNNs as the backbone model?
2. What is the main difference between the approximate propagation module in ScaleGUN and that in InstantGNN?
3. It would be better if the authors can also plot the bounds of other baseline methods (CGU and CEU) in Figure 2, so as to show that ScaleGUN is on par with them.
4. To avoid ambiguity and improve readability, the authors are advised to explain or formally define $\nabla\mathcal{L}(\mathbf{w}^\star,\mathcal{D})-\nabla\mathcal{L}(\mathbf{w}^\star,\mathcal{D^\prime})$. The reader may incorrectly interpret that $\mathcal{L}(\cdot)$ here is the loss function of all training samples, whereas it is actually the loss function of those removed.
5. There are a few typos or grammatical errors, such as
    1. Page 1 Line 049 & Page 2 Line 83: the gradient residual norm is missing the vector differential operator $\nabla$.
    2. Page 4 Line 188: "parallel" should be corrected as "parallelly"

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors study the important problem of certified graph unlearning, where they focus on developing scalable graph unlearning algorithms that can work on extremely large graphs such as ogbn-papers100M. They identify the scalability issue of prior works that require recomputing the graph propagation, and alleviate such issue with an efficient push-forward algorithm that is commonly utilized in efficient PageRank computation. The certified unlearning guarantees are established and extensive experiments are done for large graph datasets, which demonstrates the superiority of the proposed method compared to prior works.

### Strengths
- Identify the scalability issue of the prior graph unlearning works.
- Combining error analysis of push-forward operation and certified graph unlearning framework in a clever way.
- Extensive experiment, especially conducted on graphs with at most 100M nodes.

### Weaknesses
I personally do not find major weaknesses in the paper. Yet some minor points can be addressed to improve the manuscript further.

- The notations can be hard to digest for readers who are not familiar with prior works.
- The difference in analysis compared to prior works should be further emphasized and elaborated.
- I feel the main experiments should be based on the setting $\delta=1/n$ or $\delta=1/|E|$ instead (currently deferred to Appendix), which provides a more meaningful privacy implication. 

## Detail comments

Overall it is a joyful read. The authors study the important certified graph unlearning problem, where they identify the scalability issue in prior works and resolve it by introducing approaches from efficient PageRank computation methods. In theory, while both the error analysis of updating push-forward operation and certified graph unlearning are known, it is still not trivial to combine them correctly. I feel the authors made a great contribution here. On the practice side, the authors test with large graphs such as ogbn-papers100M, which is the largest graph to be tested in the context of certified graph unlearning to the best of my knowledge. The proposed algorithm makes an important step toward the practical certified graph unlearning approach, which I find it as another great contribution of the paper. 

There are still some minor comments that can further improve the clarity and quality of the papers. First, I feel the notations can be further improved. Some notations are not even clearly defined, albeit I roughly understand the meaning since I am familiar with the field. For instance, what exactly is the definition of $\hat{D}$ and the other quantities with \hat? I guess it implies the quantities related to approximate propagation regarding equation (2) but it would be great to further elaborate the details. Also, it would be weird to say we have some “approximate dataset $\hat{D}$”. If I understand correctly, we only have $D,D^\prime$ in practice, and the “approximate quantity” is only regrading to $Z, Z^\prime$. I encourage the authors to think more carefully about the choice of the notation here.

Another comment is that I feel the difference in analysis compared to prior works is not emphasized enough and clearly explained. For instance, I cannot fully understand why the authors can deal with propagation matrix $P = D^{-1/2}A D^{-1/2}$ and the prior works only can do $P=D^{-1}A$. What exactly is the issue and what techniques that the authors use to resolve such issue? Now I only roughly know that it is handled by examining the approximation error of equation (2), but I feel the explanation can be clearer.

Finally, I personally think the authors should focus on the privacy setting of $\delta = 1/n$ or $1/|E|$ instead of a fix $\delta = 10^{-4}$, especially the number of edges and nodes are larger than 100M in the largest graph. As also pointed out by the authors, this is the common standard in the privacy graph learning literature. While the claim would not change and is fair (since all tested approaches follow the same privacy constraint), it is still preferable to make the setting as meaningful as possible in terms of privacy.

### Questions
1.	What is the definition of $\hat{D}$ and other quantities with \hat?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes ScaleGUN, a scalable certified unlearning framework for PPR-based GNNs. In particular, the authors propose a lazy local propagation mechanism to improve the efficiency of the forward pass. In a theoretical perspective, this paper proposes theoretical guarantees on the unlearning efficacy of the proposed lazy local propagation framework. Experimental results show that ScaleGUN largely reduces the latency for unlearning.

### Strengths
- The proposed ScaleGUN achieves an impressive speedup on some of the largest graph datasets while ensuring a theoretical guarantee.
- In the theoretical perspective, ScaleGUN extends previous certified unlearning for linear GNNs to PPR-based GNNs which includes a forward approximation error.
- The paper is overall well-organized. Technical details are introduced. Open-source codes are provided.

### Weaknesses
- The scope of ScaleGUN is still limited to linear GNNs. However, existing studies [1,2,3] have already extended certified graph unlearning to general GNN architectures. This limitation can reduce the significance of the proposed ScaleGUN.
- For the unlearning technique, ScaleGUN simply follows previous designs (based on influence function). The technical novelty is limited.
- Several latest certified graph unlearning baselines [1,3] are missing in the experiments.
- The noise variance is determined by some hyperparameters, e.g., $c_1$ and $\gamma_2$. How to determine the value of them in the practical scenarios?
- Some typos need to be fixed, such as the reference to CEU is incorrect.

[1] Wu, Jiancan, et al. "Gif: A general graph unlearning strategy via influence function." Proceedings of the ACM Web Conference 2023. 2023.

[2] Wu, Kun, et al. "Certified edge unlearning for graph neural networks." Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023.

[3] Dong, Yushun, et al. "Idea: A flexible framework of certified unlearning for graph neural networks." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

### Questions
Please see the Weaknesses

### Soundness
3

### Presentation
3

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
This paper studies the certified graph unlearning problem with controlled propagation approximation to accelerate the unlearning speed over large-scale datasets. The main contribution of this paper comes from combining the ideas from forward push and graph unlearning with theoretical guarantees. Specifically, the forward push is designed to control the impact when we try to remove an edge or node feature from the graph dataset. Overall the idea of this paper is intuitive and the presentation is easy to follow. The empirical evaluations also show significant improvement in the unlearning efficiency compared to previous graph unlearning baselines when the model performance is comparable.

### Strengths
1. This paper aims to resolve one of the major bottlenecks of current certified graph unlearning methods, which is the high complexity of repropagating the features over the new graph topology. Approximate updates, adapted from dynamic PPR propagation techniques, are used to accelerate this repropagation process. To align with the certified (graph) unlearning analysis, the authors also provide theoretical analysis to bound the extra approximation error, validating that the privacy requirement is satisfied.

2. The empirical performance on large-scale datasets show promising gains in the unlearning efficiency while the model performance is comparable.

### Weaknesses
1. The linear model assumption is still needed with the new analysis. Also, most of the analysis (i.e., forward push part and graph unlearning part) seems similar to previous methods.

2. One of the key points, why updating the embeddings of a small local neighborhood is enough, is not rigorously explained in the main text. I am not sure the statement "Consider an edge removal scenario, for example, where edge (u, v) is targeted for removal. Only node u, node v, and their neighbors fail to meet Equation (2) due to the altered degrees of u and v." is enough here. Please see the Questions Section for more details.

### Questions
In Lemma 3.2 Eq (2), the authors claim that "Consider an edge removal scenario, for example, where edge (u, v) is targeted for removal. Only node u, node v, and their neighbors fail to meet Eq (2) due to the altered degrees of u and v." But I think besides the one-hop neighbors, more nodes could be affected thus violating the Eq (2). This is mainly because both $q^{l-1}(t)$ and $d(t)$ can change in this process, not just $d(t)$. For example, suppose there is a line connection between $a-b-c$, and we remove one edge incident to $a$. In this case, $a$ and $b$ would definitely violate Eq (2). But for $c$, since $q^{l-1}(b)$ would change, $q^{l}(c)$ and $r^{l}(c)$ would also need to adjust. I am wondering if the authors can explain the details here in a more rigorous way and maybe discuss how it will affect the following analysis.

### Soundness
3

### Presentation
4

### Contribution
3
