# Revisiting and Benchmarking Graph Autoencoders: A Contrastive Learning Perspective

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 5, 5, 3

## Abstract
Graph autoencoders (GAEs) are self-supervised learning models that can learn meaningful representations of graph-structured data by reconstructing the input graph from a low-dimensional latent space. Over the past few years, GAEs have gained significant attention in academia and industry. In particular, the recent advent of GAEs with masked autoencoding schemes marks a significant advancement in graph self-supervised learning research. While numerous GAEs have been proposed, the underlying mechanisms of GAEs are not well understood, and a comprehensive benchmark for GAEs is still lacking. In this work, we bridge the gap between GAEs and contrastive learning by establishing conceptual and methodological connections. We revisit the GAEs studied in previous works and demonstrate how contrastive learning principles can be applied to GAEs. Motivated by these insights, we introduce lrGAE (left-right GAE), a general and powerful GAE framework that leverages contrastive learning principles to learn meaningful representations. Our proposed lrGAE not only facilitates a deeper understanding of GAEs but also sets a new benchmark for GAEs across diverse graph-based learning tasks. The source code for lrGAE, including the baselines and all the code for reproducing the results, is publicly available at https://anonymous.4open.science/r/lrGAE/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents lrGAE (left-right GAE), i.e., a comprehensive benchmark for graph autoencoders (GAEs) in self-supervised learning (SSL). It aims to conceptually and methodologically bridge the gap between GAEs and contrastive learning. The proposed lrGAE enables an extensive comparison of GAEs across eight unique configurations on seventeen real-world datasets, covering a range of scenarios.

### Strengths
1. This paper formalizes a very critical and interesting problem, i.e., exploring the connection between generative and contrastive learning in graph SSL. It provides insight into their relationship through the lens of the loss function and breaks down GAEs based on the main components of contrastive learning.

2. The paper is well-structured and easy to understand, presenting an evaluation across a diverse set of datasets and task types.

3. lrGAE is available as an open-source tool, which can help researchers test their model and keep track of the current research frontier.

### Weaknesses
1. Lack of relevant GAEs baselines/references - Current discussion of generative SSL is mainly contained within masked GAEs, while an important research direction involving spectral GAEs, such as asymmetric spectral filters in GALA [1], low-pass filters in GATE [2, 3], and augmentation-adaptive filters in WGDN [4], is missing. Clarification on how spectral GAEs fit into contrastive-based categories is needed.

2. Inadequate detail on experimental settings and analysis explanations.

    2.1 Critical information on experimental configurations, such as hidden dimensions, training epochs, learning rate ranges, scheduling strategies, and hyperparameter optimization techniques, is not provided. These details are essential for the comprehensiveness and credibility of a benchmark paper.

   2.2 Some baseline methods experience out-of-memory issues for the Physics dataset in Tables 4 and 9. While the authors attribute this to large feature dimensions, reducing the hidden dimension, as suggested by [4], could be a solution. It is advisable to explicitly state and adjust hidden dimensions to facilitate a more balanced and thorough comparison.

   2.3 The authors argue that structure-based GAEs underperform compared to feature-based GAEs due to structural bias. However, for graph-level tasks, attributes are often hand-crafted structural features (e.g., degrees). A more plausible explanation might be that feature-based GAEs extract more nuanced information through the interplay between graph structures and derived attributes.

3. It's recommended to include more experimental results in the main text, as expected of a benchmark paper. Currently, extensive results are deferred to the appendix to adhere to the 9-page limit, but the latest guidelines allow up to 10 pages.

[1] Jiwoong Park, et al. Symmetric Graph Convolutional Autoencoder for Unsupervised Graph Representation Learning. In ICCV 2019.

[2] Amin Salehi and Hasan Davulcu, Graph Attention AutoEncoders. In ICTAI 2020.

[3] Muhammet Balcilar, et al. Analyzing the expressive power of graph neural networks in a spectral perspective. In ICLR 2021.

[4] Jiashun Cheng, et al. Wiener Graph Deconvolutional Network Improves Graph Self-Supervised
Learning. In AAAI 2023.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper re-evaluates Graph Auto-Encoders (GAEs) through the lens of Graph Contrastive Learning (GCL), uncovering the conceptual and methodological links between the two. Furthermore, the authors present a unified framework, lrGAE, and evaluate the performance of various GAEs on different downstream tasks to validate the properties of the proposed methods.

### Strengths
1.	The writing is clear, and the overall structure is easy to understand.
2.	The paper provides a systematic and theoretical analysis of the connection between feature-based GAEs and graph contrastive learning (GCL).

### Weaknesses
1.	The innovation appears limited, as most insights are derived from previous works, including the equivalence between structure-based GAEs and GCLs. The key viewpoint about feature-based methods, two rooted subgraphs are considered positive pairs, is also inspired by another similar work.
2.	The primary contribution of this paper seems to be a new perspective on current graph self-supervised learning (SSL) methods. All variants of lrGAE achieve performance levels similar to existing models, with no improvements in link prediction or node classification tasks. It would be more persuasive if the effectiveness of this new insight were demonstrated through performance enhancement or mitigation of specific problems.
3.	The paper lacks a novel, comprehensive theoretical analysis of the link between GAE and GCL. It relies heavily on theoretical conclusions from other works, such as [1], without thoroughly examining the equivalence between these two unsupervised methods. According to lines 237-247 and 260-261, the training process for structure-based and feature-based GAEs is only considered an alignment-loss minimization procedure, while the uniformity loss (Eq 7) remains unexplored. The statement "GAEs could be regarded as an approximated contrastive learning" is not convincing as a conclusion about the connection between GAE and GCL.
4.	A good benchmark should introduce new metrics/datasets, focusing on specific domain issues or commonly overlooked problems within the field. Relying on common metrics such as "Link Prediction" and "Node Classification," along with merely combining different contrastive views, The current contribution may not produce a proper benchmark.

### Questions
1.	Lack a KL term in Eq5 ?
Please see the above Weaknesses.

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
4

### Summary
This paper focuses on self-supervised representation learning on graphs (especially node-level graph autoencoders). The paper proposes that recent graph mask autoencoder methods can be understood from a contrastive learning perspective. Based on this perspective, the paper presents a design space for graph autoencoders, categorizes current methods, and proposes several novel methods based on this design space. In summary, this paper provides a new perspective for understanding graph autoencoders.

### Strengths
1. This paper provides a unified perspective for understanding graph contrastive learning and various autoencoder methods, which is very meaningful for further understanding graph self-supervised learning.

2. The paper's writing quality and presentation quality are high.

3. The authors provide well-documented code and documentation for reproduction.

### Weaknesses
1. Although there hasn't been a systematic study of (masked) graph autoencoders, this paper isn't the first to propose connecting graph autoencoders with contrastive learning. For example, in [1,2], it was already proposed that adjacent node pairs could be used as positive pairs in contrastive learning. Therefore, in my view, the idea in this paper isn't novel (though I appreciate this paper's contribution as a systematic overview of graph autoencoders).

2. The paper focuses mainly on summarizing previous methods. Although Table 1 proposes three new methods (lrGAE6,7,8), specific details about these methods and their implementations cannot be found in the main text.

3. The proposed methods didn't achieve particularly impressive results.

References:

[1]  Lee, N., Lee, J., and Park, C. Augmentation-free self-supervised learning on graphs. arXiv preprint arXiv:2112.02472, 2021

[2]  Zhang, Hengrui, et al. "Localized contrastive learning on graphs." arXiv preprint arXiv:2212.04604, 2022.

### Questions
1. Besides conceptual and perspective-based contributions, does LrGAE have any empirical contributions compared to current methods?

2. Do the node pairs in Table 1 refer to the objects of reconstruction? For example, when v = u, does it refer to feature reconstruction, and when v ≠ u, does it refer to edge reconstruction?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a novel framework, lrGAE (left-right Graph Autoencoder), which combines principles from contrastive learning with graph autoencoders (GAEs) to enhance representation learning for graph-structured data.

### Strengths
A comprehensive analysis of self-supervised learning on graphs, especially understanding graph autoencoder from the perspective of graph contrastive learning.

### Weaknesses
1. In [1], GAEs has already been connected to GCLs. While the paper introduce a more comprehensive analysis, it is a summary or deep understanding.

2. The core components of lrGAE, including augmentation strategies, contrastive views, encoder-decoder networks, and the use of contrastive loss without negative samples, are extensively covered in prior literature.

3. From the experimental results, the proposed lrGAE (6,7,8) cannot achieve better performance compared to masked graph autoencoder model MaskGAE.

[1] Li, J., Wu, R., Sun, W., Chen, L., Tian, S., Zhu, L., Meng, C., Zheng, Z. and Wang, W., 2023, August. What's Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders. KDD

### Questions
What is the performance of the proposed method on large-scale datasets?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper analyzes the connection between two graph self-supervised learning paradigms, i.e., graph contrastive learning and graph autoencoders. Then, it enhances graph autoencoders with contrastive learning steps, proposing a new model. Finally, it conducts experiments on various graph SSL models on different tasks, including node classification, link prediction, and clustering.

### Strengths
1. The paper provides a comprehensive analysis on different types of graph CL and graph AE methods, identifying the design recipe of graph AE models. Table 1 is good.
2. The experiments are comprehensive, from my perspective.

### Weaknesses
1. Originality: I have doubts about the originality of the work. The abstract states that "the underlying mechanisms of GAEs are not well understood, and a comprehensive benchmark for GAEs is still lacking." However, I believe this is not accurate. For instance, S2GAE already analyzed the connection between graph masked autoencoders and contrastive learning. It is well recognized that "GAEs, whether employing structure or feature reconstruction, with or without masked autoencoding, implicitly perform graph contrastive learning on two paired subgraph views." Moreover, I did not find that this paper proposes a more comprehensive benchmark, since most of the baselines are already covered in recent papers. Contrastive learning, MAE, and graph learning have been long-standing subjects of study. While reading this paper, I felt that many claims were not very novel, suggesting that this work is rather incremental.

2. Clarity: I believe the authors attempted to provide a detailed explanation of the proposed idea in the methodology part, but the writing could be more concise. The narrative is quite extended, and I was lost as to what the KEY contribution of this work is (rather than several marginal contributions put together). Additionally, there seems to be a typo in the explanation of Eq 8.

### Questions
Can you summarize what is THE KEY contribution of this paper?

### Soundness
3

### Presentation
3

### Contribution
2
