# Mixture Stochastic Block Model for Multi-Group Community Detection in Multiplex Graphs

- Decision: Reject
- Scores: 5, 3, 3, 5, 6

## Abstract
Multiplex graphs have emerged as a powerful tool for modeling complex data due to their capability to accommodate multi-relation structures. These graphs consist of multiple layers, where each layer represents a specific type of relation. Pillar community detection, a clustering approach that assigns vertices to clusters across all layers, has been employed to identify shared community structures. However, particular layers may possess distinct divisions, deviating from the pillar-based clustering. Consequently, it becomes crucial not to identify individual layer clusters, but a similar cluster for similar layers. In this paper, we propose an approach called the "Mixture Stochastic Block Model," which aims to group similar layers based on shared community structures. A common Stochastic Block Model represents each group's shared community structure. The model is rigorously defined, and an iterative technique is employed for computing the inference. We estimate the layer-to-group assignments using the expectation-maximization technique, while the vertex-to-block assignments within each group are determined using the variational estimation-maximization technique. We assess the identifiability of our proposed model and show the consistency of the maximum likelihood function. The performance of the method is evaluated using both synthetic graphs and real-world datasets, showing its efficacy in identifying consistent community structures across diverse multiplex graphs.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces the "Mixture Stochastic Block Model," a proposed method for grouping similar layers based on shared community structures in multiplex graphs. Instead of focusing on individual layer clusters, the approach aims to identify community structures across different layers. The model is carefully defined, and an iterative technique is utilized for computing the inference. The paper includes analyses of identifiability and consistency, providing theoretical support for the proposed method. The performance of the approach is evaluated using synthetic graphs and real-world datasets, effectively showcasing its ability to identify consistent community structures in diverse multiplex graphs.

### Strengths
S1. The paper demonstrates good writing and logical organization, making it reader-friendly.

S2. The authors provide codes of the proposed method, indicating the reproducibility of the study.

S3. Including identifiability and consistency analyses distinguishes the proposed method from current deep learning approaches, providing theoretical underpinnings.

### Weaknesses
W1. The motivation needs to be strengthened. It is unclear why research on multi-group community detection in multiplex graphs is important. Although some studies on multiplex graphs exist, the scarcity of research on multi-group community detection may be due to the difficulty of the topic or its limited occurrence in the real world. Please clarify the reasons behind the lack of research or provide examples from the real world to support the importance of this study.

W2. The paper lacks novelty. While the authors mention Stanley et al. (2015) as existing literature on multi-group community in multiplex graphs, the paper only highlights the difference between this work and Stanley et al. (2015) in terms of the learning of layer-to-group assignments, with the former using the EM algorithm instead of k-means. What are the challenges in replacing k-means with the EM algorithm? Are there any differences in the modeling assumptions between this paper and Stanley et al. (2015)?

W3. The experimental section is weak. Firstly, the paper does not compare the proposed model, which incorporates layer-to-group assignments, with other models that handle multiplex graphs but do not consider layer-to-group assignments [1,2,3,4]. This comparison is necessary to highlight the importance of layer-to-group assignments. Secondly, the datasets used in the experiments are relatively small, while real-world data often exhibit large-scale characteristics. What is the time complexity of the proposed algorithm? Can it handle large-scale data?

W4. The references are incomplete. For example, [1,2,3,4] are all methods for multilayer graphs, but the authors did not cite them. Although [4] focuses on dynamic networks, its modeling approach for generating multilayer networks is similar to the approach in this paper. I recommend that the authors cite and analyze this reference.

References:
[1] Han Q, Xu K, Airoldi E. Consistent estimation of dynamic and multi-layer block models[C]//International Conference on Machine Learning. PMLR, 2015: 1511-1520.
[2] De Bacco C, Power E A, Larremore D B, et al. Community detection, link prediction, and layer interdependence in multilayer networks[J]. Physical Review E, 2017, 95(4): 042317.
[3] Paul S, Chen Y. Consistent community detection in multi-relational data through restricted multi-layer stochastic blockmodel[J]. 2016.
[4] Corneli M, Latouche P, Rossi F. Exact ICL maximization in a non-stationary temporal extension of the stochastic block model for dynamic networks[J]. Neurocomputing, 2016, 192: 81-91.

### Questions
Q1. What are the reasons behind the scarcity of research on multi-group community detection in multiplex graphs? Are there any challenges or limitations in studying this topic?

Q2. In terms of novelty, are there any differences in the modeling assumptions between this paper and Stanley et al. (2015)? Please clarify.

Q3. What are the reasons the EM algorithm performs better than Kmeans?

Q4. Can you compare the proposed model with other models that handle multiplex graphs but do not consider layer-to-group assignments? This comparison would highlight the importance of layer-to-group assignments.

Q5. What is the time complexity of the proposed algorithm? Can it handle large-scale data?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a mixture stochastic block model (MSBM) for community detection in multiplex graphs. The method is based on the estimation maximization technique according to the underlying MSBM structure. The experimental results on synthetic and real datasets verify the effectiveness of this method.

### Strengths
A new random graph model for multiplex graphs and a new approach for multiplex graph clustering are proposed.

### Weaknesses
To be honest, I cannot continue reading up to Section 3.1. The model definition is quite hard to follow. Due to the pressed review time, I do not have enough time to understand it. The authors defined a large number of parameters with superscripts and subscripts for expression accuracy. However, the basic architecture of the model is not very clearly for me. To my understand, the underlying layers are proposed for the simulation of multiplexity of a graph. Each layer is partitioned into $K$ groups and each group is partitioned further into blocks (a bit weird, probably I have a misunderstanding here). The probability of presence of edges depends on the vertex-to-block assignments and the probability matrices for inter and intra blocks, just like the traditional SBM. But how do we understand that a layer is generated from a group? A layer has several groups, so I wounder how to generate this layer from a single group and how to understand vector $\beta$ and $y_{lk}$. A simple example would be much helpful, but there isn't. Since the basic architecture is not clear, I cannot evaluate the subsequent work.

### Questions
(1) In Section 3.1, what do you mean when you say "let's consider a partition of layer $L$ into $K$ groups and assume that group $k$ comprises $Q^k$ blocks"? What is the relationship between group and block in your definition? What's the purpose of them?

(2) Is there a simple example to help understand the notations in Section 3.1?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The aim of the paper is to compute shared clusters/communities across layers of a multiplex graph that exhibit similar community structures, and separate clusters/communities for layers exhibiting different community structures. It is clearly motivated why having (potentially) different groups of communities across layers may be desirable and how it is not sufficiently captured by existing work on Multiplex Community Detection.  

The method consists of two parts: Grouping layers with expectation-maximization technique and assigning vertices to communities across grouped layers with a variational estimation-maximization technique. Compared to existing variational estimation-maximization technique for community detection, this reduces the complexity of parameters that need to be fitted which is desirable for high-dimensional datasets.

### Strengths
* Good motivation and promising underlying idea.
* Sound theoretical foundation

### Weaknesses
None of the presented state-of-the-art methods from the related work sections are compared against. Only two naive approaches, K-means and the native SBM, are used as competing methods -- even though their drawbacks were made abundantly clear in the paper. Here, some validation on how well the communities in recovered multiplex SBM describes/generate the observed data in comparison to other community detection methods would be useful instead of only considering the two individual steps of the presented approach independently. Such a comparison might also contain synthetic SBMs with only shared groups as assumed by many existing techniques as a baseline and their failure to capture non-shared groups (highlighting the improvement of the new MSBM method). 

The description of task 5.1 is confusing as it is initially refereed to as the vertex-to-block assignment task, but within the same paragraph it is described how the NMI is used to evaluate the layer-to-group association. I am assuming you compared your method which takes all layers into account to 30 independently fitted SBMs which does not seem like a fair comparison; however, this might just be a misunderstanding due to the confusing structure of 5.1 and 5.2. and missing details regarding the actual tasks that are performed. 

For the task 5.2 multiple multiplex SBMs with different parametrizations should be considered to sample synthetic data for the graph clustering, like e.g. varying block or group sizes and varying blocks across layers. It is also not clear to me which different graphs clusters the task 5.2 tries to recover as only a single parametrization of an SBM is given. I am guessing that the clustering is not done on the level of Multiplex graphs, but on the level of their individual layers as graphs which then leads to a very small dataset of just 30 graphs with 3 very distinct inter-block probabilities that might be considered trivial to recover in monolayer graphs. It is not clear to see how recovering these graphs is an improvement to the state of this art. To highlight potential benefits various other graph clustering methods should be considered. However, this might also be a misunderstanding because initially I would have expected a clustering of the actual multiplex graphs based on their MSBMs when hearing “Graph clustering”. 

Furthermore, I would have liked to see the impact of larger graphs have on the computational complexity as currently only small graphs of size 100 are considered. This is especially crucial as you mentioned that other competing approaches like MLSBMs suffer from exponentially large parameter space which drawbacks should be highlighted on large graphs. For this the runtime or some similar measure should be included in the results. Additionally, the impact of varying levels of noise on recovering the clusters would also be an interesting addition to the synthetic experiments.

Regarding the real-world application, it seems as task 1 was to differentiate between real datapoints and randomly generated data points which also seems to not be a suitable benchmark. Task 2 seem to differentiate the article-article graphs from two different datasets which however makes me wonder what the two rows for each dataset in Table 4 refer to as I would expect a single group-to-layer resp. vertex-to-block classification result. Maybe you can clear this up by properly introducing how the mutual information is used in this task. I also would like to ask why the adjusted mutual information (adjusted to random chance) was not used instead.

In summary, despite the well-motivated need for methods such as this, the promising underlying idea and sound theoretical foundation, it is not clear that the presented methods actually achieve this or improves the state-of-the-art. The experimental section does not quantify the quality of the recovered (Mixture) Multiplex SBM as it only shows that it performs better in the two individual steps compared to very naïve approaches.

### Questions
Concrete Suggestions to improve the score:

1.	Describe the actual performed task in the experiment section and make it more clear what you refer to as graph clustering (in the context of Multiplex Graphs).

2.	Include state-of-the-art methods and actually compare the recovered Multiplex SBMs instead of only the results of individual steps of MSBM

3.	(Maybe) include a larger synthetic dataset w.r.t the number of graphs, nodes, blocks and different sets of parameters to seem less exemplary. 

4.	Include more metrics than just NMI and a run time comparison 


Things to improve the paper that did not impact the score:

No statement about reproducibility possible as no code was provided. 

The motivation to leverage Multiplex graphs as multi relational data representation and their benefits are clear. But inherent limitations of Multiplex graphs like the requirement of actors to be identical across types of interactions/layers are not addressed or mentioned. Although being a very powerful tool, it most likely is not the solution to all complex multi-dimensional datasets. I would like the authors to very briefly address and differentiate this to Multilayer Graphs (where inter-layer edges are used to model an n-to-m mapping between actors of nodes) and, potentially, outline how the presented approach could be generalized (or not) to this data representation as part of future work.
The categorization of existing techniques into flattening, consensus and direct approaches and the placement of the new approach overall seems reasonable, however the third category of “direct approaches” seems to be very broad and might benefit from being further subdivided.   

Further questions:
It is not completely clear to me why similar results could not be achieved through consensus-based approaches that support some relaxation of the consensus across all layers to only consensus across groups of layers exhibiting similar community structures and allowing a disagreement between layers of different groups akin to the approach in this paper.

What does “Layers of the same strata present their groups,…” mean? (in section 2)

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes to model the multiplex graphs using a mixture of stochastic block models. Compared to a standard stochastic block model, the use of mixture enables clustering of the graph layers into groups, each group having the same parameters. The model is optimized using Variational EM algorithm, and a spectral initialization approach is used to speed up convergence. Experiments on several synthetic data and a real world data is included.

### Strengths
1. The presentation and writing is clear.
2. Sufficient background is provided.
3. The idea of using spectral initializaiton to speed up convergence is very useful in practical but often left out from papers. It's nice to see an innovative approach included in this paper.
4. Theory of identifiability and consistency of MLE are provided or briefly discussed.

### Weaknesses
1. While the idea is interesting, the contribution seems insufficient. I would be more inclined to give a higher rating if the mixture model is presented more generally for degree corrected mixed membership models (e.g. https://arxiv.org/abs/1708.07852).
2. Most of the experiments are synthetic, and the only real world data is sort of manually constructed. It would be better if this model is application driven by more and larger real datasets.
3. The baseline models used in numerical experiments are just k-means and stochastic block models, which are rather low bars. Other basic baselines include exponential random graphs, latent space models, random dot product graphs. There are also a wide literature of modern models in the past few years.
4. The concept of multiplex graph is closely connected to the terms "population of networks", "replicated networks", and it would be nice to discuss in this more general family of methods.

### Questions
1. The variational EM algorithm only guarantees to converge to a local minimizer, and the optimization objective is seen to be highly non-convex. How different are the minimizers that the algorithm converge to at different runs? And how would you interpret these different minimizers?
2. In the case of graph related modeling, it often appears that a simple spectral initialization can already give a pretty decent result. How much improvement is actually achieved by using variational EM after the spectral initialization? And without using spectral initialization, is the variational EM still capable of numerically finding a good local minimizer given sufficiently many iterations? If yes, how much iterations is saved by using spectral initialization?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper discusses the setting where we are given a multi-layer graph, where each of these layers has been generated by a stochastic block model, and some of these layers are generated by the *same* stochastic block models. Detecting this structure amounts to performing two partitioning tasks at once: on the one hand, the layers need to be partitioned into groups (to determine which layers are generated by the same stochastic block model). On the other hand, for each group of layers, the corresponding block structure (partition of nodes) needs to be detected.
The authors estimate the partition of layers into groups by Estimation Maximization and estimate the partition of nodes into blocks by Variational Estimation Maximization.

### Strengths
The problem setup is really interesting and seems incredibly challenging. It is impressive that the authors managed to develop a theoretically sound estimation technique for this problem that is actually able to detect this subtle structure.

The introduction is well written and pleasant to read!

### Weaknesses
While the introduction is pleasant to read, I felt that it did not properly prepare me for the convoluted problem setup that unfolded in Section 3. Only after reading section 3 several times and going back and forth between section 3 and the introduction, did I finally understand that we are simultaneously clustering layers into groups and nodes into blocks. While I really like the problem setup, it really needs to be explained much better in order to make the reader understand how different it is from the 'normal' clustering problem. In particular, emphasis needs to be put on the difference between groups (sets of layers) and blocks (sets of nodes).

There are some notational errors in Section 3: $\beta_1$ instead of $\beta^1$ below (2), $ln$ instead of $\ln$, $[1,L]$ (continuous interval) instead of $\{1,\dots,L\}.

Section 4 is difficult to understand and the presentation (and notation) could be improved significantly.

The experiments measure the performance by the NMI measure, which is known to be biased towards fine-grained clusterings [1]. I would recommend to either use the Adjusted Mutual Information or the Correlation Coefficient [2].
 
The fact that the method achieves 100% accuracy on the layer-clustering in the experiments is perhaps an indication that one should choose more challenging experiments. It is interesting to see how an imperfect layer-to-group performance impacts the vertex-to-block performance.

[1] Vinh, N. X., Epps, J., & Bailey, J. (2009, June). Information theoretic measures for clusterings comparison: is a correction for chance necessary?. In Proceedings of the 26th annual international conference on machine learning (pp. 1073-1080).
[2] Gösgens, M. M., Tikhonov, A., & Prokhorenkova, L. (2021, July). Systematic analysis of cluster similarity indices: How to validate validation measures. In International Conference on Machine Learning (pp. 3799-3808). PMLR.

### Questions
Does the method assume some posterior distribution for the block/group sizes? In general, I'm interested to know how it relates to Bayesian Blockmodeling [1]

In the while statement of algorithm 1, shouldn't the OR be replaced by an AND? Also, maybe use $\wedge,\vee$ instead of $\|$

[1] Peixoto, T. P. (2019). Bayesian stochastic blockmodeling. Advances in network clustering and blockmodeling, 289-332.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
