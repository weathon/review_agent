# Approximate Clustering for Extracting Task Relationships in Multi-Instruction Tuning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3

## Abstract
The development of language models involves the evaluation of a broad range of learning tasks. Recent work has shown that by using carefully designed instructions to teach a large transformer model, they can be fine-tuned on a wide range of downstream tasks. However, when the number of instructions increases, they can negatively interfere with each other if trained together. Existing works have relied on domain expertise and manual inspection to construct multi-instruction sets, which can be time-consuming and difficult to scale. To address this challenge, this paper develops a clustering algorithm to find groups of similar tasks based on a given set of task affinity scores. This is an NP-hard problem, and conventional algorithms such as spectral and Llyod's clustering are sensitive to variations in the scale of task losses. Our algorithm instead uses a semidefinite relaxation to maximize the average density of clusters and then rounds the solution with a threshold. We adaptively build the clusters by gradually adding tasks so that the affinities only need to be computed in the existing clusters. Then, we construct an evaluation benchmark to assess task grouping algorithms with verified group structures. The evaluation set includes 63 cases, spanning multitask instruction tuning, multi-instruction tuning, and in-context learning of multiple functions. We validate our algorithm on this evaluation set by showing that it recovers the group structure found by an exhaustive search. We also show that our approach improves performance over multi-instruction and soft-prompt tuning by up to 6\% on several sentence classification and structure-to-text generative tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors are studying the problem of identifying which tasks should be mixed together during training such that there is a positive transfer between these tasks and they help each other to improve. The authors first assume access to a pairwise task affinity matrix which specifies the improvement on task j when task i is trained before it. Assuming access to this task-affinity matrix they formulate the clustering problem in terms of maximizing the average density of task affinity scores across all clusters which is NP-hard. Then the authors use SDP relaxations to arrive at an approximate task clustering algorithm AdaGroup. They discuss and compare spectral and Lloyd clusters as two other approaches.

### Strengths
S1. The problem of task clusters is extremely important and hard to work with because the ground truth clusters are mostly not available. Grouping tasks appropriately can lead to significant improvements in the final model's performance.

S2. The method is well-motivated and can adaptively add new tasks to the existing cluster.

S3. The authors have computed the pairwise affinity of some NLP tasks and released them as a benchmark so that the community can try out different ideas to cluster tasks.

### Weaknesses
W1 The paper is slightly hard to understand, especially the experimental section. Many statements about 

W2. New tasks can be processed in a batched manner to avoid expensive computations, however, the number of clusters are usually predefined and new tasks cannot allocate new clusters. This seems like a problem to me, some non-parametric Bayesian processes like the Chinese restaurant process, and Indian buffet process can potentially be used to solve this issue. Defining the number of clusters can be a challenging as well as computationally expensive process as multiple models are needed to be trained to obtain the clusters for a new set of tasks. 

W3. The paper seems to talk about spectral and LLoyd clusters but in Table 2, I don't see them as a baseline. For example, you could use the clustering obtained from spectral and Lloyd algorithm and then perform training similar to your method. This I feel is an important baseline as it would tell us the impact of identifying wrong task clusters. It might also be the case that the impact of having wrong clusters is not too high, i.e. some decent amount of noise in the clusters does not impact the final downstream performance.

### Questions
**Must do for me to retain my scores**

Q1: Please improve the writing of the in-context learning experiments, I am not able to understand the motivation of, the experimental setup, and the conclusions for it. I am still not sure if I understand where logistic regression and decision trees come into this picture and the implications of the designed experiments. 

Q2: same as W3. W3 needs to be addressed for retaining the score.

 
**Answer these for me to consider increasing my scores**

Q3: On the created benchmark AdaGroup can identify all the clusters correctly. It would be nice to see if you take some heldout tasks, and then use your AdaGroup to cluster these tasks and visualize the cluster/affinity scores. It would be nice to see how much these clusters correlate with human understanding. If this correlation is high then for a small number of tasks doing the clustering manually might be a reasonably good choice. 

Q4: Given that the method needs to train multiple models in order to estimate the task affinity matrix, this method might not be feasible in cases where the datasets are pretty huge. It would be a really nice study to see what is the minimum number of samples from each task that you can use in order to get reliable task affinity scores that can lead to good clusters and improved downstream performance. This would ameliorate the costs associated with this method and make it scale to more number of tasks.

Q5: A solution to the cluster number number problem, see W2 for more details.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel approximate clustering algorithm, along with an evaluation benchmark, that aims to group tasks for language model instruction fine-tuning.

### Strengths
The new clustering algorithm achieves good fine-tuning results, which outperforms multi-instruction and prefix tuning by 3.3% on multiple tasks.

### Weaknesses
1) The paper utilizes SDP relaxation as an approximate clustering technique; however, it does not provide an analysis of the error bound that comes with using this method. To provide a better understanding of its applicability, it is essential to understand the worst-case scenario and where the SDP relaxation clustering algorithm fails.
2) The generalization of the SDP relaxation clustering algorithm is not analyzed in this paper, leaving questions about whether it is a general clustering solution that can approximate spectral clustering or if it only works for instruction fine-tuning.
3) While the paper proposes using an approximate solution, it does not provide clear explanation as to why the SDP relaxation clustering algorithm will outperform spectral clustering. Is it related to the clustering objective function definition issue?
4) It is not clear how to ensure that the output of the convex optimization can follow the ranking constraint. The paper could provide a more detailed exploration of this aspect of the algorithm.
5) The paper does not analyze the impact of different lambda values for different tasks. Is lambda the same in all experiments, or should it be tuned in each experiment?
6) The hyper-parameters, including k, m, s, and alpha, are not thoroughly studied. The paper could explore how to choose these values, such as whether they should be tuned for the particular task or selected according to some rules.
7) The paper proposes an adaptive estimation of task affinities; however, there is no analysis of how different sample sizes affect the final estimation accuracy. Examining how the estimation accuracy impacts the final results could provide deeper insights into the method's performance.
8) It would be helpful if the paper could provide the results of the spectral clustering approach and compare it with the multi-instruction and prefix tuning methods to give a more comprehensive assessment of the new clustering algorithm's performance.

### Questions
1) Please address the above weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper starts from the well-known task grouping problem in Standley et al., 2020 and studies the following formulation: Given n tasks, find a partitioning of them into k groups so that each group of tasks can be best trained together (separately from the other groups).

### Strengths
+ The task grouping problem is indeed an important problem for language models

### Weaknesses
- The approach is missing a rigorous complexity analysis which for this topic seems very important
- The analysis seems incmplete. For xample "Ablation Studies," particularly in the subsection on "Instruction Selection," the authors compare the accuracy of the fine-tuned models resulting from their method and those obtained using other clustering algorithms. Considering providing a more comprehensive evaluation, it would be better if authors could also include a comparison of the time taken by each method.
- The clustering / aggregation approach and overall the prior works are poorly discussed and contrasted, especially recent mathematical approaches  that exploit advanced concepts like curvature and others.

### Questions
1) In the “Task Grouping Setup.” – page 3 and other parts of the paper, it would be useful if the authors could make it clearly whether the tasks are considered independent or there is some form of dependency and how it is captured beyond the task affinity matrix.
2) I have read many vague statements like “The computational cost of these techniques can still be quite high for the scale of instruction finetuning…” but no precise complexity analysis results. Can the authors provide at least in some cases concrete numbers?
3) The authors recognize that the clustering is a well-studied problem, they mention approaches based on SDP and Linear Programming relaxations, but it seems they have missed recent efforts on optimal transport theory like the Ollivier-Ricci curvature in “Ollivier-Ricci Curvature-Based Method to Community Detection in Complex Networks” (Scientific Reports 2019) or “Inferring functional communities from partially observed biological networks exploiting geometric topology and side information” (2022) and others which bear some analytical similitudes to LP relaxations. It would be fair to mention these theoretical works in clustering for weighted graphs as they are very much competitive or potential approaches for the problem studied here.
4) The fact that the “affinity matrix, in our case, can easily violate the triangle inequality” made me wonder if these Ollivier-Ricci curvature approaches that deal with advanced geometric concepts could be useful to this problem.
5) This is a minor issue, SDP is used in section 2 but only defined later.
6) In section 4.2, the authors should give the full name of the “MTL” to make readers better understand the meaning of “MTL performance”.
7) In Figure 2, the authors should include subscripts or labels to distinguish the results obtained from the three different methods. 
8) The authors mention their use of high-order task affinity and adaptive sampling as methods to expedite the calculation of task affinities. In comparison to the direct training of n^2 models, it would be beneficial if the authors could quantify the time savings achieved through their proposed method. Clarification on this matter would greatly enhance the comprehensibility of the paper.
9) In Section 5.4, "Ablation Studies," particularly in the subsection on "Instruction Selection," the authors compare the accuracy of the fine-tuned models resulting from their method and those obtained using other clustering algorithms. Considering providing a more comprehensive evaluation, it would be better if authors could also include a comparison of the time taken by each method.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
