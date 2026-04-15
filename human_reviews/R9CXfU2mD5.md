# Score Propagation as a Catalyst for Graph Out-of-distribution Detection: A Theoretical and Empirical Study

- Decision: Reject
- Scores: 6, 5, 6

## Abstract
The field of graph learning has been substantially advanced by the development of deep learning models, in particular graph neural networks. However, one salient yet largely under-explored challenge is detecting Out-of-Distribution (OOD) nodes on graphs. Prevailing OOD detection techniques developed in other domains like computer vision, do not cater to the interconnected nature of graphs. 
This work aims to fill this gap by exploring the potential of a simple yet effective method -- OOD score propagation, which propagates OOD scores among neighboring nodes along the graph structure. This post hoc solution can be easily integrated with existing OOD scoring functions, showcasing its excellent flexibility and effectiveness in most scenarios. However, the conditions under which score propagation proves beneficial remain not fully elucidated. Our study meticulously derives these conditions and, inspired by this discovery, introduces an innovative edge augmentation strategy with theoretical guarantee. Empirical evaluations affirm the superiority of our proposed method, outperforming strong OOD detection baselines in various scenarios and settings.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a method called GRaph-Augmented Score Propagation (GRASP) for enhancing out-of-distribution (OOD) detection in graphs. It propagates OOD scores among neighboring nodes to leverage graph structure. The paper investigates whether score propagation will always help graph OOD detection. The authors find the ratio of intra-edges (ID-ID and OOD-OOD) to inter-edges (ID-OOD) must be high for propagation to be beneficial. To improve this ratio, GRASP strategically adds edges to a subset G of training nodes that are assured to be in-distribution. This enhances the intra-edge ratio and thus OOD detection performance after propagation. Theoretically, the paper shows that if G connects predominantly to ID data over OOD data, GRASP can provably improve post-propagation OOD detection outcomes. The paper evaluates GRASP on benchmark graph datasets and pre-trained GNNs. It demonstrates GRASP outperforms baselines
Overall, the paper is well-written and provides good insight. However, the experiment was all conducted on small-scale of data, which  limits the ability to fully validate the proposed approach and conclusions. More experiments on large-scale real world data would strength the claims.

### Strengths
1. The paper provides one of the first theoretical analyses of score propagation for graph OOD detection. The theoretical analysis is derived rigorously and proves helpful conditions for when propagation enhances OOD detection. The formulations and proofs are clear.
2. The paper is well-written, the motivation, problem definition, methodology and conclusions are explained clearly throughout the paper. 
3. The proposed GRASP method is original in its strategic augmentation of edges to a subset of training nodes to boost intra-edge ratios.

### Weaknesses
1. However, the experiment was all conducted on small-scale of data, which  limits the ability to fully validate the proposed approach and conclusions. More experiments on large-scale real world data would strength the claims.
2. The time complexity of GRASP compared to baselines is not mentioned in the paper. A thorough accounting of computation/memory demands compared to baselines is important as it relates to practical deployment.

### Questions
1. The theoretical analysis assumes edges follow a Bernoulli distribution. How sensitive are the results to this assumption?
2. How do different propagation mechanisms, like higher-order diffusion, impact the findings? 
3. What is the time complexity of GRASP compared to baselines?

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper studies graph OOD detection through OOD score propagation. The paper theoretically proves that propagation can enhance OOD detection when there are more intra-edges within ID/OOD nodes than inter-edges between ID and OOD nodes. The paper further proves that the efficacy of propagation can be improved by adding edges to nodes that have more connections to ID nodes than to OOD nodes. Then, the paper designs a simple augmentation strategy to boost the performance of propagation.  Experimental results demonstrate the effectiveness of their proposed strategy.

### Strengths
- The paper theoretically analyzed when OOD score propagation will work and how to boost its performance.
- The proposed method is simple and experimental results validate its effectiveness.
- This paper is well-written and easy to understand.

### Weaknesses
- The proposed method is developed based the assumption that intra-edges dominate the graph. However, the challenge in OOD detection arises from the unknown pattern of OOD nodes. This strong assumption makes the analysis less insightful and constrains the practicality of the proposed method.
- The experiments are conducted exclusively with one type of OOD nodes (Label Leave-Out), which limits the comprehensive evaluation of the proposed method under different distribution shifts.
- The selection of Sid/Sood is based on the pre-computed OOD scores (using MSP), rendering the proposed method ineffective when MSP fails, as observed in dataset Squirrel in Table 2.

### Questions
- According to Theorem 3.2, the propagation is deemed effective only when intra-edges dominate. Why does it seem to work well in heterophily dataset Chameleon?
- Why does the improvement of the proposed method over GNNSafe appear to be marginal in homophily datasets like Amazon and Coauthor, while the improvement is more significant in the two heterophily datasets in Table 2?
- How does the proposed method perform when faced with different types of OOD, such as structural manipulation and feature interpolation as addressed in GNNSafe?
- Is the proposed method sensitive to hyperparameter k, $\alpha$, and $\beta$? How to choose the $\alpha$ and $\beta$ when MSP’s performance varies in different datasets?

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper study detecting Out-of-Distribution (OOD) nodes on graphs. The authors first demonstrate through empirical and theoretical evidence that previous OOD detection methods relying on information propagation are only applicable to scenarios where the number of intra-edges is greater than inter-edges. Following this, based on their analytical conclusions, the authors propose an edge augmentation strategy called GRASP to enhance the effectiveness of these methods. Experimental results on several datasets demonstrate the effectiveness of their approach.

### Strengths
A. This paper is well organized and written, with a clear definition of the research problem and detailed introductions of the motivation and methodology. Key conclusions are clearly marked in the paper.
 
B. The proposed method has good generalizability. The post-processing strategy does not require retraining the model, allowing for flexible application to various existing methods and improving their effectiveness on OOD node detection task.
 
C. The experimental results are impressive. The authors have compared their method with baselines on many datasets and conducted a detailed analysis of the experimental results, showing that the proposed method can significantly enhance OOD node detection.

### Weaknesses
A. The data augmentation relies on a robust base model. During the graph augmentation, ID nodes and OOD nodes are sampled based on the base model's OOD prediction scores, which may lead to potential error propagation.
 
B. Analysis of some key hyper-parameters is missing. It appears that the two hyper-parameters \alpha and \beta in the proposed method significantly affect the sampling results, and I would like to know the impact of different \alpha and \beta values on the results.

### Questions
A. According to the authors, the number of intra-edges and inter-edges has a significant impact on OOD detection. What are the respective proportions of these two types of edges in different datasets? And what are their proportions after graph augmentation?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
