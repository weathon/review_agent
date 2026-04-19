# Characterizing Long-Tail Categories on Graphs via A Theory-Driven Framework

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 8

## Abstract
In the context of long-tailed classification on graphs, the vast majority of existing work primarily revolves around the development of model debiasing strategies, with the aim of mitigating class imbalances and enhancing overall performance. Despite the notable success,  there is very limited literature that provides a theoretical tool for characterizing the behaviors of long-tail categories in graphs and gaining insight into generalization performance in real-world scenarios. To bridge this gap, we propose the first generalization bound for long-tail classification on graphs by formulating the problem in the fashion of multi-task learning, i.e., each task corresponds to the prediction of one particular category. Our theoretical results show that the generalization performance of long-tailed classification is dominated by the overall loss range and the total number of tasks. Building upon the theoretical findings, we propose a novel generic framework Tail2Learn for long-tailed classification on graphs. In particular, we start with a hierarchical task grouping module that allows us to assign related tasks into hypertasks and thus control the complexity of task space; then, we further design a balanced contrastive learning module to adaptively balance the gradients of both head and tail classes to control the loss range across all tasks in a unified fashion. Finally, extensive experiments demonstrate the effectiveness of Tail2Learn in characterizing long-tail categories on real graphs. We publish our data and code at https://anonymous.4open.science/r/Tail2Learn-CE08/.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper confronts a significant challenge in long-tailed classification on graphs. While most prior research concentrates on mitigating bias, this paper offers a fresh perspective by introducing a theoretical framework for characterizing long-tail categories and improving generalization in real-world scenarios. The authors present the TAIL2LEARN framework, encompassing hierarchical task grouping and long-tailed balanced contrastive learning. Notably, the experimental results demonstrate promising performance, outperforming state-of-the-art methods.

### Strengths
- The proposed approach is novel and addresses a significant gap in the existing literature by providing a theoretical foundation for long-tail classification on graphs. The motivation for this work is well-defined and highlights the need for a deeper understanding of class imbalances and generalization performance.
- A notable strength of the paper is its comprehensive theoretical analysis, which includes the development of a Generalization Error Bound that substantiates the effectiveness of the proposed method.
- The experimental results effectively illustrate the superiority of the proposed TAIL2LEARN framework. By showcasing its effectiveness in characterizing long-tail categories on real-world graph datasets, the authors provide practical evidence of their method's capabilities.

### Weaknesses
- One potential weakness of the paper is that the hierarchical task grouping approach employed by the authors seems similar with existing techniques like Graph U-Net [1]. Although the authors have extended these prior methods to facilitate multi-task learning and task grouping with theoretical backing, it may require clarification about what sets the TAIL2LEARN framework apart from the existing Graph U-Net. Further clarification and a more detailed comparison between the two would be beneficial to better understand the novelty and differentiation of the proposed framework.
- While the authors have approached long-tailed classification as a multi-task learning problem, they have configured the number of tasks in the second layer to align with the number of categories. It might be worth considering whether the authors have explored the possibility of subdividing the samples into more finely-grained subclasses, which means increasing the number of tasks in the second layer beyond the number of categories.
- The authors claimed that $\mathcal{L}_{BCL} potentially controls the range of losses for different tasks. However, the paper lacks experimental results to support this claim, which could contribute to a more robust evaluation of the method's effectiveness.

[1] Gao H, Ji S. Graph u-nets[C]//international conference on machine learning. PMLR, 2019: 2083-2092.

### Questions
See above.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigate long-tailed classification on graphs by providing a PAC generalization bound in a multi-task learning fashion, which is characterized by the task number and overall loss range. As a solution, the authors propose Tail2Learn, a learning framework for long-tailed node classification, which reduces task complexity by hierarchically grouping tasks and adopting a contrastive loss to adaptively balance the gradients of both head and tail classes to control the loss range.

### Strengths
+ The method presented in this paper is straightforward and easily comprehensible.

+ The approach of addressing the long-tail problem through a multi-task learning perspective appears to be original.

+ The empirical results in the experiment section indicate a promising improvement compared to the baseline methods.

### Weaknesses
- The theoretical aspect of the paper appears to be quite preliminary, lacking in-depth analysis and original contributions. It appears to heavily rely on Theorem 8 from a previous work [1]. Additionally, some statements and derivations are unclear and contain errors, making it challenging to verify their correctness. For specific concerns, please refer to the detailed questions.

- While the shift in perspective towards multi-task learning is novel, the proposed method is essentially a combination of existing, well-known techniques, such as hierarchy graph pooling [2] and contrastive loss [3].

- The discussion and comparison of some relevant work (such as [4][5]) are missing in this paper.

[1] Maurer et al., The Benefit of Multitask Representation Learning, 2016

[2] Ying et al., Hierarchical graph representation learning with differentiable pooling, 2018

[3] Zheng et al., Tackling Oversmoothing of GNNs with Contrastive Learning, 2021

[4] Zhang et al., "Graph-less Neural Networks: Teaching Old MLPs New Tricks Via Distillation, 2022

[5] Zheng et al., Cold Brew: Distilling Graph Node Representations with Incomplete or Missing Neighborhood, 2022

### Questions
1. In Lemma 1, after applying the PCA bound provided by [1], how is the normalization term $1/T$ eliminated? If $1/T$ needs to remain, I would question one of the main claims in the abstract: "generalization performance of long-tailed classification is dominated by the total number of tasks'' (as stated in the abstract). This is because, after accounting for $1/T$, such a bound may no longer scale with the number of tasks. Actually, Theorem 2 in [1] even suggests the generalization error decays with $O(1/\sqrt{T})$ when transferred to new task.

2. In Lemma 3, why is the definition of $R(F)$ (Eq. 11) different from Eq. 4 in [1]? Note that Eq. 4 has $f(y) - f(y')$ in the numerator, while Eq. 11 has $l(f(y) - f(y'))$ in the numerator.

3. In the proof of Corollary 1, it remains unclear to me how the inequality between the last terms (under the square root) is established.

4. Why is this analysis focused specifically on long-tail classification on graphs? Can it be extended to the general long-tail learning problem?

5. Can the authors explain why the proposed approach, Tail2Learn, takes on the form of $f \circ h," resembling a general multi-task learning framework?

6. It is known that task complexity is not always harmful. Instead, improving task diversity can be helpful for multi-task learning [2][3]. Does this contradict to this paper’s claim?

[1] Maurer et al., The Benefit of Multitask Representation Learning, 2016

[2] Tripuraneni et al., On the Theory of Transfer Learning: The Importance of Task Diversity, 2020

[3] Du et al., Few-Shot Learning via Learning the Representation, Provably, 2020

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies long-tail categories in graphs. It proposes a generalization bound for long-tail classification on graphs, as well as a method TAIL2LEARN for long-tailed classification on graphs. The method includes a hierarchical task grouping module to reduce complexity of task space and a contrastive learning module to balance the gradients of head and tail classes. Experiments are conducted to evaluate the method.

### Strengths
1. The paper provides theoretical studies and arrives at a generalization bound.

2. The paper presentation includes rich contents, with tables and figures well organized.

3. The conducted experiments look correct with ablation studies included and code provided.

### Weaknesses
1. Related works not well addressed. The long-tail categories studied in the paper is the same as the node-level imbalanced-class problem in graph. The imbalanced class problem has been studied intensively for graphs, which is closely related to this work but not sufficiently discussed in its related works. The paper lacks a thorough review of related literature. Some missing related works are [1-6].

2. Following the above point, the experiments should include some of the missing imbalanced class baselines.

3. The correctness of Corollary 1 is unclear. Why can contrastive learning guarantee to learn the predictors $f_1^{(l)}, . . . , f_T^{(l)}$ with $Range(f_1^{(l)}, . . . , f_T^{(l)}) < Range(f_1, . . . , f_T)$? In its proof, why do we only need to compare the relationship between $\sum _t 1/(n_t^{(l)})$ and  $\sum _t 1/(n_t)$? And how is the special case of all nodes in one hypertask generalized to prove $\sum _t 1/(n_t^{(l)})\leq \sum _t 1/(n_t)$? The proof should be clearly given step-by-step instead of ambiguously stated.



[1] Imgcl: Revisiting graph contrastive learning on imbalanced node classification

[2] Boosting-GNN: boosting algorithm for graph networks on imbalanced node classification

[3] Graph neural network with curriculum learning for imbalanced node classification

[4] Co-Modality Graph Contrastive Learning for Imbalanced Node Classification

[5] Diving into Unified Data-Model Sparsity for Class-Imbalanced Graph Representation Learning

[6] TAM: topology-aware margin loss for class-imbalanced node classification

### Questions
Please see Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
While current methods focusing on the long-tail problem in graphs have shown notable improvements, this work, Tail2Learn, approaches from a different perspective and formulates the long-tail classification problem into a multi-task learning framework. Built upon theoretical findings, it controls the complexity of task space and the loss range of task-specific classifiers by offering remedies such as hierarchical task grouping and long-tailed balanced contrastive learning. The experiments on the node classification task show the efficacy of Tail2Learn in real-world long-tailed graph datasets.

### Strengths
1. I quite enjoyed reading this paper. Overall, the claims of this paper are well-formulated, and its remedies for the theoretical findings are well-supported.
  
2. The proposed Definition 1, Long-Tailedness Ratio, is intuitive and straightforward. This metric can be generalized to balanced cases, such as 5 classes having 20 training samples each, as it would have a value of 4 in the 80th percentile.  This contribution would further enrich the long-tail GNN community.
   
3. The empirical performance aligns with the theoretical motivation. Also, the paper is well-written and easy to follow.

### Weaknesses
*Major*
1. In M1. Hierarchical Task Grouping, I agree that this approach can reduce label scarcity and task complexity. However, I am concerned whether hierarchical grouping across different classes might compromise the distinctiveness of each class. That is, there could be a trade-off between achieving reduced complexity and maintaining distinctiveness among classes. Although there exists a module for contrastive loss between different classes, its contribution remains unclear. A more detailed discussion of such situations should be provided.
  
2. In M2. Long-Tail Balanced Contrastive Learning, the utilization of supervised contrastive loss seems reasonable. However, given the long-tail situation, there would be very few training samples with labels for tail classes. Consequently, the positive pairs within tail classes would be significantly fewer compared to the head classes. Can you elucidate how Tail2Learn can work effectively in this scenario?
  
3. Although the overall performance of Tail2Learn is effective in current datasets, can you provide more details about the improvements made in tail classes as shown in Figure 4 in LTE4G [1]? This would offer a more comprehensive understanding of Tail2Learn's efficacy in terms of improvement in tail classes without sacrificing performance in head classes.
  
4. Can Tail2Learn generalize well on graph datasets having a relatively small number of classes such as Cora, CiteSeer, and PubMed?
   
*Minor*
1. Although Definition 1, Long-Tailedness Ratio, is well-designed, for clarity, at first glance, I expected the semantic meaning to refer to "how severe the data distribution is long-tailed". However, in actuality, the semantic meaning is "the lower the severity of long-tailedness." Have you considered the reciprocal version of the current long-tailedness ratio?
  
2. The notation (e.g., subscripts) in Equation 6 and Equation 7 appears to be exactly the same, while the underlying meaning is different. For clarity, I suggest differentiating the notations that denote specific classes and specific hypertasks, as they do not necessarily have to be the same value.
  
3. The performance of ImGAGN [2] in Table 1 seems unusually low compared to classical GNN, although it is originally designed to alleviate class long-tailedness. Can you provide further explanations for this?
  
If the above concerns are properly addressed, I would be very happy to raise my score on the current rating.
    
[1] [CIKM 2022] LTE4G: Long-Tail Experts for Graph Neural Networks  
[2] [KDD 2021] ImGAGN:Imbalanced Network Embedding via Generative Adversarial Graph Networks

### Questions
See the Weaknesses.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
