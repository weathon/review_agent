# Repelling Random Walks

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
We present a novel quasi-Monte Carlo mechanism to improve graph-based sampling, coined repelling random walks. By inducing correlations between the trajectories of an interacting ensemble such that their marginal transition probabilities are unmodified, we are able  to explore the graph more efficiently, improving the concentration of statistical estimators whilst leaving them unbiased. The mechanism has a trivial drop-in implementation. We showcase the effectiveness of repelling random walks in a range of settings including estimation of graph kernels, the PageRank vector and graphlet concentrations. We provide detailed experimental evaluation and robust theoretical guarantees. To our knowledge, repelling random walks constitute the first rigorously studied quasi-Monte Carlo scheme correlating the directions of walkers on a graph, inviting new research in this exciting nascent domain.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work introduced a repelling mechanism among walkers in a graph when doing MC simulation. The proposed sampling mechanism is easy to understand. And intuitively it makes more sense than iid sampling by considering the graph topological property. Experiments are conducted on three graph-related tasks. Results also verified its better performance than the iid baseline

### Strengths
S1. A more vivid random walk mechanism with considering graph topological property

S2. experiments on three graph tasks to show the advantage of the proposed

S3. Solid theoretical analyses

### Weaknesses
W1. concern on the audience interest

W2. more interesting downstream applications are expected

### Questions
Overall, this is a good paper with both solid theoretical analyses and experiments on various tasks. However, some concerns are: 

C1. Random walk is one of the important research topic in graph, the fundamental research is worthy of applause. Random walk related approaches have also be applied in downstream tasks in the real-world applications, such as graph embedding and community detection. However, the focus of this paper seems to be more fundamental. Random walk theoretical research usually fits better in venues like graph theory (lean more on mathematics) and computing theory (e.g., STOC). So I have concern on audience interest for ICLR. 

C2. More real-world related applications are expected. The authors applied the proposed random walk mechanism in three applications. But the three seem to be more abstract than those driven by real-world applications or hot topics in the current research community. For example, graph kernel approximation covers one major category of approaches for graph embedding. PageRank vector approximation is also one of the fundamental problems for graph embedding and community detection. Graphlet detection is used in subgraph representation. Compared to the three in this paper, graph embedding, subgraph representation, and community detection may be closer to real-world scenario. In recent 10 years, deep learning approaches attract more attentions in almost every research field. If the fundamental research like this paper can show enhancement against recent so-called advanced methods or benefit the recent popular approaches, that would be more exciting.

=================

After reading the authors response where more examples were involved. The response address some of my concerns. But more real-world related applications are expected (C2 in my review) in this paper. 

As a result, I increased my score.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a novel quasi-Monte Carlo mechanism called repelling random walks. The authors demonstrate that the marginal transition probabilities of repelling random walks remain unchanged compared to standard random walks. In particular, the paper proves that the variance of approximate random walk probabilities is suppressed by simulating repelling random walks. The paper showcases the effectiveness of repelling random walks by applying them to three distinct tasks.

### Strengths
S1. The paper introduces a novel quasi-Monte Carlo mechanism, repelling random walks, aimed at enhancing graph-based sampling. This approach could potentially inspire further research in this field.

S2. The marginal transition probabilities of repelling random walks remain unchanged, while the variance of these walks is reduced.

### Weaknesses
W1. The advantage of using repelling random walks over standard random walks appears to be marginal. For instance, the reduction in approximation errors when estimating PageRank using both standard and repelling random walks as shown in Table 2 is relatively minor.

W2.  The validity of certain arguments is heavily dependent on specific assumptions. Take Theorem 4.2, for example: its accuracy hinges on the assumption that the count of random walks is less than the minimum node degree in the provided graph. Nonetheless, in a variety of real-world network structures, the minimum node degree stands at one, rendering repelling random walks virtually indistinguishable from standard random walks. 

W3. The paper's presentation needs improvements. The current manuscript contains ambiguous sentences and unclear notations. For instance: 
- Page 2: the notation $P^{(i)}$ requires clarification, as the paper defines $P$ but not $P^{(i)}$. 
- Page 3: the notation $i_1$ and $\delta_{i_1}$ require clarifications. 
- In the appendix on Page 20: the reasoning behind the statement "only walkers originating from the same node are correlated" requires additional explanation.

--- 
During the rebuttal phase, the authors and I engaged in detailed discussions regarding the novelty and contributions of the paper. I appreciate that the authors have effectively addressed the issue related to the minimum node degree assumption (i.e., W2). We also had a thorough discussion about the fundamental differences between Repelling Random Walk and Radar Push, a closely related work. The authors' responses were not only prompt but also convincing. Consequently, I have decided to revise my initial score from 5 to 6.

### Questions
Q1. The experimental setups presented in Table 2 lack clarity. The paper indicates that 1000 trials are conducted on each graph, with more than two repelling random walks simulated during each trial. Could you specify the exact number of standard and repelling random walks simulated for each graph?

Q2. In Table 2, could you please indicate the minimum node degree for each graph?

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
In this paper, the authors present repelling random walks to sample from a graph. In some examples, theoretical results about the improvement on the concentration of estimators and numerical experiments about efficiency in sampling are given. Both theoretical and numerical results look sound.

### Strengths
S1. A novel quasi-Monte Carlo algorithm called repelling random walks is given. 

S2. Results on typical examples are given to illustrate the new algorithm.

### Weaknesses
W1. Theoretical results only show that the new variance of estimator is less than classical method, but the author can give a more explicit quantitative analysis of how small it can be.

W2. As to the efficiency in sampling, only numerical results are given, which weaken the solidity of the improvement brought by the new algorithm.

### Questions
Q1. Is it possible to give a more explicit relationship comparing the variances of estimators between the classical and the new method?

Q2. Ideally I would like to see some basic properties or results concerning the repelling random walk in Section 2 before diving into the applications.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
