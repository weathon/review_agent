# Generalization of Spectral Graph Neural Networks

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 5, 6, 6, 3, 3, 6

## Abstract
Spectral graph neural networks (GNNs) have achieved remarkable success across various applications, yet their generalization properties remain poorly understood. This paper bridges this gap by analyzing the impact of graph homophily and architectural choices on the generalization of spectral GNNs. We derive a general form of uniform transductive stability for spectral GNNs and provide an explicit stability analysis for graphs with two node classes, providing a comprehensive framework to understand their generalization. Based on this stability analysis, we establish a generalization error bound, demonstrating that better stability leads to improved generalization. Our theoretical findings reveal that spectral GNNs generalize well on graphs with strong homophily or heterophily but struggle on graphs with weaker structural properties. We also identify conditions under which increasing the polynomial order in spectral GNN architectures may degrade generalization. Empirical results on synthetic and real-world benchmark datasets align closely with our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explored the generalization capabilities of Spectral GNNs, focusing on the relationship between generalization capabilities, graph homophily, and node class distribution in node classification tasks. It also investigated the connection between the polynomial order in the architecture of spectral GNNs and generalization error. Through rigorous theoretical analysis and experimental validation, the paper showed that spectral GNNs performed worse on graphs with weak homophily or heterophily but generalized well on graphs with strong homophilic or heterophilic structures. At the same time, it also provided a detailed analysis of the relationship between polynomial order and generalization error.

### Strengths
1. The structure of the paper is clear and easy to read, and it is well-organized；

2. The paper is highly theoretical, offering a rigorous theoretical derivation process for its research problem；

3. The paper is somewhat pioneering, providing a great perspective on the study of the generalization performance of spectral GNNs.

### Weaknesses
1. The research is based on generalization error. In my opinion, the authors should also incorporate additional metrics to validate the accuracy of their theory. For instance, comparing training and test loss over a specific number of epochs would closely align with the notion of generalization error. Additionally, computing one or two sets of generalization error on both synthetic and real-world datasets used in the paper would further illustrate their point effectively.

2. Datasets are small in experiments. The authors might consider using larger datasets, such as those mentioned in link [1], to further validate their theory. Only a few additional experiments, perhaps just one or two, would be sufficient to make the point, as certain properties of Spectral GNNs may differ when applied to larger datasets.

3. In line 512-571, the author wrote that "These observations align with our theoretical analysis." However, in Figure 4, the test accuracy for JacobiConv and in Figure 4 for ChebNet does not show a linear relationship with the K value, and the curves exhibit significant fluctuations. This seems inconsistent with the conclusions drawn by the authors in Proposition 16. Could the authors provide an explanation for the discrepancy between their conclusions and the experimental results observed in Figure 3 and Figure 4?

4. It would be better to provide codes to ensure the reproducibility.

&nbsp;&nbsp;&nbsp;[1] https://ogb.stanford.edu/docs/nodeprop/

### Questions
Refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the impact of node label distribution and polynomial degree on the generalization capability of spectral GNNs for the task of node classification. 

Through a uniform transductive stability analysis of spectral GNNs, the authors provide a generic error bound for spectral GNNs, highlighting the impact of expected prediction error and $k$-length walks on the stability and generalization of the GNNs. Then, the authors specify the bound in the case of a graph with 2 classes, showing that in this case high or low homophily correspond to good generalization, while medium homophily hinders generalization, and they prove that, under some assumptions, increasing the order of the polynomial in the spectral GNN leads to worse generalization.

### Strengths
- The paper is well-written and contributions are clearly highlighted
- The theoretical results are interesting, well introduced and clearly explained
- The experiments on synthetic datasets validate the theoretical claims

### Weaknesses
- Throughout the paper, both in the theoretical and in the experimental results, no analysis of the depth and width of these spectral GNNs is conducted. It seems at some that the work focuses on a single layer GNN but this is not explicit. As such, the claims of the paper do not hold for a general setting as GNNs are typically deeper and wider (multiple features), which imply layers of filter banks. The effect of these filter banks and how their combined spectrum affects generalisation abilities is unclear.
- The experiments on real-world datasets show, in some cases, inconsistent results compared to the theoretical claims (e.g., as the authors point out in lines 484-485, the accuracy and loss gaps do not decrease as $H_{edge}$ increases from high to medium heterophily). One possible explanation of this is the varying dataset size, as the authors mention, but a more thorough analysis of this behavior would increase the understanding of how the theoretical results apply in practice. One aspect to consider in this sense might be the fact that the final generalization bound is computed in the case of a graph with 2 classes, whereas the real-world graphs used in experiments contain more. This might have a relevant impact, since homophily itself does not characterize GNNs' behavior well and more sophisticated measures of label distributions provide deeper insights, especially when there are more than 2 classes [1,2]. 
- The above as well as the limitation of the generalisation in terms of transitive stability and node classification, limit drastically the impact of this contribution

### Questions
- Regarding the condition in Theorem 14 "when $\sum_{k=2}^K\theta_k(k-1)!/2^{k-1}$ increases more slowly than $\sum_{k=2}^K\theta_k^2(k-1)!/2^{k}$": it is not clear what this means in practice, e.g., how likely this is to hold? Could the authors provide more details about this?

### Soundness
3

### Presentation
3

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
This work analyzes the generalization ability of spectral GNN in node classification tasks. The authors first bound the generalization error with uniform transductive stability, then analyze the effect of graph homophily/heterophily and polynomial order on stability. Specifically, the authors prove that graphs with strong homophily/heterophily improve model generalization by stabilizing models, while increased polynomial order will increase generalization error when specific conditions hold, leading to varying behavior in real world datasets. Finally, the authors provide experiments to support their findings.

### Strengths
- This work builds the bridge between generalization and uniform transductive stability on spectral GNNs, and explores the effect of graph homophily/heterophily and polynomial order on stability, which is overall original and provides some insight into this area. 
- The paper is and well-organised and easy to follow.

### Weaknesses
This work does provide insight into the effect of graph homophily/heterophily and polynomial order K on generalization. However, I would suggest the following weaknesses:

1. The conditions for K may or may not hold in real-world datasets. As a result, this finding may not offer substantial guidance for future improvements. 
2. The authors do not provide sufficient explanation or intuitive reasoning regarding the conditions for K.
3. Although the authors acknowledge that the work is limited to a specialized cSBM, I would suggest the authors include explanations on why this specialized cSBM is useful when analyzing generalization.

### Questions
Please refer to the weakness

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
2

### Summary
The paper studies the generalization capabilities of spectral graph neural network. The analysis  is based on the theory of uniform transductive stability. The paper theoretically show that generalisation bound is influenced by graph homophily, the polynomial order of the spectral GNN and the cardinality of the training set as follow: 1) a spectral GNN with polynoms of order K performs better on strongly homophilic and strongly heterophilic datasets; 2) if the spectral GNN parameters fulfill certain conditions, the generalisation error bound increases with the order of polynom K (which might explain the poor performance of spectral GNN with high K, despite the increase in expressivity) and 3) increased in the number of training samples improves generalisation. The theoretical results are validated both on synthetic and real-world datasets, for 5 different spectral GNN architectures.

### Strengths
- Analysing the connection between generalization bound, dataset caracteristiques ( degree of homophily/heterophiliy..) and architecture (polynom’s degree) is an interesting topic that can help developing more powerful architectures, better tuned for a specific problem.
- The paper is well-structured, with a smooth progression from one finding to the next one. Significant effort has been made to present the findings in an intuitive and practical manner.
- The design of the experiments is well aligned with the theoretical findings.

### Weaknesses
- While the synthetic experiments clearly match the theoretical results, for the real-world datasets some discrepancies are present. In Figure 2, models with medium degree of heterophily has a similar gap of accuracy compared to strongly homophilic/heterophilic datasets. The authors motivate these results by the influence of training set cardinality, since the datasets not only differ in the homophily level, but also in size.  Indeed, since the generalisation gap is influence by both number of training samples and homophily level, it is hard to disentangle the individual effect of each one of them in the presented real-world experiments. However, this makes it considerably harder  to asses if the theoretical results correlates or not with the practical findings. Does repeating the experiments with a different train-val-split such that the number of training samples remains the same alleviates these discrepancy? Or, given the assumption that n→ infinity it is better to focus on having comparable percentages of nodes in the training set when splitting the data?
- The results in Figure 4 are hard to correlate with the theoretical findings. The theorems show that, under certain assumptions, the gap in accuracy increases with the order of the polynom used by the spectral GNN. However, that is not the case in the experiments provided in Figures 3 and 4. The explanation is that the assumptions might not be fulfilled in some of the scenarios. Is it possible to check or impose that assumption such that the experiments can match the theoretical findings?
- While the paper provide interesting insights into what influence the generalisability of spectral GNNs, it does not provide any guidance towards how can we benefit from these findings.

### Questions
Please see the sections above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces a generalization study of the GNNs based on transductive learning and homophily/heterophily. The paper's results are relevant, and the topic is important. However, the paper's presentation can be improved, and a better discussion of the results should be added. The author's results are not novel per se, given that there were similar results before. Also, the link between homophily/heterophily and generalization is independent of GNNs and only depends on the walk on the graph, which provides little intuition on the implications of them. The numerical experiments are almost uninformative, and the results show a large amount of noise. I believe the topic paper is relevant, but I fail to understand what novelty this paper brings to the community.

### Strengths
1. The paper's topic is relevant. 
2. To the best of my knowledge, the theoretical results of the paper are correct.

### Weaknesses
1. The paper's theory is separated - the generalization results are on one side, and the homophily/heterophily on the other one. This undermines the importance of the results. That is to say, the authors present two relevant results that are disjoint and use different theoretical assumptions. I believe this undermines the importance of the paper given that it should be two different papers, and the novelty and contributions of each should be evaluated separately.  

2. Regarding the generalization results based on transductive learning, these results are not novel, and similar results exist in the literature:

 - Transductive Robust Learning Guarantees, Montasser et al. 
 - On Inductive–Transductive Learning With Graph Neural Networks Ciano et al. 
 - Towards Understanding Generalization of Graph Neural Networks Tang et al.

The authors should explain why their results are important, and how they fit in the crowded landscape. 

3. The previous point is regarding novelty, but there is also something to be said about this transductive analysis in and of itself. The graph is completely and almost entirely ignored. Why is this analysis even relevant to the context of GNNs? Why is it valid to analyze GNN while ignoring the graph completely? Even by looking at the assumptions for Theorem 6, the is not a single mention of the graph. The graph could have no edges (be a set of samples) and the results would still hold. In particular, Theorem 6 is a known result. 

4. The paper uses a very strong assumption regarding uniform transductive stability. In particular, Theorem 9 is valid because of the strong assumption \gamma-uniform transductive stability. When does this assumption hold? The authors should explain why using this assumption is valid. 

4. The graph is introduced in the second part of the paper, and the only relevant features of it are community and label. This largely ignores the progress in graph theory, and graph models that have been used in the last 10+ years. The authors utilize an extremely simple model that does not appear in any real-world problem. What is even worse, in this extremely simple model, the results are impossible to digest; Theorem 13 is unintuitive, and the explanation given with the bullets after it is largely trivial and uninformative. I fail to understand the contribution of these theoretical results. 

5. The numerical results are difficult to process and understand. Given that the authors provide a theoretical framework that is very difficult to follow, the numerical results shed no light on any real-world conclusion. Looking at Figure 2, there is little connection to the theory. Even more so, the theory is so detached from practical implementations, that new metrics have to be introduced (H_{edge}, order K for example). On top of that, the results are noisy and no statistical bands are being provided.

### Questions
See section weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper provides generalisation bounds for a specific class of spectral GNNs and assuming a particular graph model (cSBM) derives the stability of the spectral GNN training in terms of generalisation for differrent homophily and heterophily values.

### Strengths
- Clear presentation
- Significant technical machinery is introduced (though unclear how interesting or relevant it is given the assumptions and results)
- The authors attempt to validate their high-level conclusion empirically, which is nice!

### Weaknesses
TLDR: The assumptions and restrictions of the theoretical analysis are high (full-batch gradient descent and a specific spectral GNN architecture). And it's unclear whether the derived bounds are tight enough to be interesting. The empirical validation is only at a very high-level. If the authors can demonstrate that the bounds are tight, I am willing to raise my score to acceptance.

- Please include proof sketches in the main paper and add links to where the full proof is in the Appendix.
- The work assumes that the GNN is trained with gradient descent and a fixed learning rate. When this is not true in practice, and it is generally known that some optimizers in fact work very poorly. This is fine given the theoretical nature of the paper, but this limitation should be more explicitly addresssed.
- It should be shown that the generalization bound is not vacuous, i.e. that for some realistic values of n and other parameters the the value out of the bound is precise enough to be interesting. In particular it seems to me that assuming a fixed ratio between m and n the generalisation bound quickly gets worse as it relies on the absolute difference of n-m and not the ratio. Even worse in practice the labelled samples m is often quite small compared to n, so much so that O(n-m) approx O(n), which makes for an uninteresting generalisation bound. (I might be missing something here.)
- For the experimental results: it only validates the very high-level conclusion, which is less interesting. Ideally, the generalisation bound and it's tightness would be empirically validated, i.e. plot the generalisation bounds on the graph as a reference else the theory is not very useful.
- Also the presentation of the results could be improved please label the points in Figure 2 for instance to show which datapoint represents which dataset.
- The architecture restriction is quite severe.

### Questions
- Theorem 6 is odd to me, it seems to not be specific to spectral GNNs none of the parameters of the graph (n,f, Pi, Q) seem to affect the transductive stability. In fact nothing except the size of the dataset m seems to affect the result. What am I missing?
- Theorem 6 What is beta_2?
- Have you validated how well you can fit the cSBM model to the real world datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 7

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper explores the generalization capabilities of spectral GNNs, examining how graph homophily and architectural choices affect GNN generalization in node classification tasks. The paper leverages the notion of $\gamma$-uniform transductive stability to obtain stability guarantees and, consequently, generalization bounds for spectral GNNs. From the theoretical analysis, the paper claims that spectral GNNs perform well on graphs with strong homophily or heterophily. Experimental results on synthetic and real-world datasets aim to support the theoretical insights.

### Strengths
- Relevance: Since generalization is one of the most important theoretical aspects of ML, the paper studies a very relevant problem in graph representation learning.

- Novelty: As far as I know, this is the first paper to theoretically investigate connections between generalization and homophily.  

- Another positive aspect is that the empirical assessment considers five popular spectral GNNs.

### Weaknesses
**Generality**: The proposed analysis is valid for SBM-based graphs, which limit the generality of the paper's claims. Overall, the paper fails to discuss the impact of this choice --- how limiting is that choice? How well does such a class capture the distribution of real-world graph data?

**Inconsistent results**: Based on Figures 1 and 2, there is a clear difference in behavior for the experiments using synthetic and real-world datasets. I found the justification based on the number of training samples quite hand-wavy. The authors could strengthen the discussion with additional empirical analysis to confirm the proposed explanation.

**Missing related works**: The paper should acknowledge other prior works on the stability/generalization properties of GNNs, e.g., "Stability properties of GNNs by F. Gama, J. Bruna, and A. Ribeiro".

**Tightness**. The paper lacks a discussion regarding the tightness of the obtained generalization bounds. Also, how well do the obtained bounds correlate with observed generalization gaps? 

**Parallel with prior results**: What parallel can one establish between the paper's findings and existing results regarding the stability/generalization of (spectral) GNNs? While I understand that comparing generalization bounds is challenging (e.g., due to different assumptions), the authors could try to connect their findings with what is known about the generalization capabilities (and stability properties) of GNNs.

**Impact**: The paper does not discuss how the theoretical results could be used to inform practitioners in designing better models. Thus, I found the overall impact of this contribution in practice limited.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
