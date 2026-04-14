# On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
We study the discriminative probabilistic modeling on a continuous domain for the data prediction task of (multimodal) self-supervised representation learning. To address the challenge of computing the integral in the partition function for each anchor data, we leverage the multiple importance sampling (MIS) technique for robust Monte Carlo integration, which can recover InfoNCE-based contrastive loss as a special case. Within this probabilistic modeling framework, we conduct generalization error analysis to reveal the limitation of current InfoNCE-based contrastive loss for self-supervised representation learning and derive insights for developing better approaches by reducing the error of Monte Carlo integration. To this end, we propose a novel non-parametric method for approximating the sum of conditional probability densities required by MIS through convex optimization, yielding a new contrastive objective for self-supervised representation learning. Moreover, we design an efficient algorithm for solving the proposed objective. We empirically compare our algorithm to representative baselines on the contrastive image-language pretraining task. Experimental results on the CC3M and CC12M datasets demonstrate the superior overall performance of our algorithm. Our code is available at https://github.com/bokun-wang/NUCLR.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors study the DPM problem on a continuous domain for (multimodal) self-supervised contrastive representation learning by leveraging the MIS method and proposing a novel nonparametric method for approximating the sum of conditional probability densities, establish a new contrastive loss for self-supervised representation learning and optimize it by an efficient algorithm called NUCLR. Experimental results on bimodal pretraining confirm the improvement in performance of their method compared to baseline approaches
on downstream tasks.

### Strengths
1. For the DMP problem on a continuous domain for (multimodal) self-supervised representation learning, the authors first point out a challenge in computing the integral in the partition function for each anchor data, then adopt the MIS method to realize robust Monte
Carlo integration. Particular;y, the authors make generalization error analysis.
2. For current limitation of current InfoNCE-based contrastive loss for self-supervised representation learning, the authors propose a new non-parametric method for approximating the sum of conditional probability densities required by MIS through convex optimization, as a result, yielding a new contrastive objective which is optimized by so-designed an efficient algorithm called NUCLR.
3. So-conducted coparative results with baselines get improved performance.

### Weaknesses
In my opinion, the manuscript is basically complete, just some weaknesses are listed as follows:
1. Lack an analysis of robustness of their results;
2. Assumption 2 requires the same d_L, I want to know if the large difference of x's and y's dimensions impact their corresponding E1 and E2.
3. In Proposition ,its (iii) condition seems difficult to be satisfied.

### Questions
Besides the above weaknesses, other questions are
1. How to determine the involved sizes m and n and is there an optimal relation between them? 
2. Should the m and n be related to dimensions of x and y spaces? What relation is to the bootstrap?
3. Formally, (1) can replace as p_v(x|y)!  When the dimensions of x and y are imbalanced, e.g., y's dimension is far greater than that of x, whether does paired sampling influence the quality of analysis?
4. Assumption 2 requires the same d_L, I still want to know if the large difference of x's and y's dimensions impact their corresponding E1 and E2.
5. (7) has some flavor of the attention mechanism to great extent, it is unclear whether this can be interpreted.
6. Is the objective or loss measure (10) robust? e,g,, x_i or y_i is noisy!
7. In Proposition ,can its (iii) condition satisfied? can the authors give some concrete cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript studies the discriminative probabilistic modeling problem on a continuous domain for (multimodal) self-supervised representation learning. To address the challenge of computing the integral in the partition function for each anchor data, the multiple importance sampling (MIS) technique is considered for robust Monte Carlo integration, which can recover InfoNCE-based contrastive loss as a special case. Typically, the generalization error analysis is conducted to reveal the limitation of current InfoNCE-based contrastive loss for self-supervised representation learning and derive insights for developing better approaches by reducing the error of Monte Carlo integration.

In order to resolve the problem, a novel non-parametric method is proposed for approximating the sum of conditional densities required by MIS through convex optimization, yielding a new contrastive objective for self-supervised representation learning. Moreover,  an efficient algorithm is designed for solving the proposed objective. The algorithm is compared with representative baselines on the contrastive image-language pretraining task. Experimental results on the CC3M and CC12M datasets demonstrate the superior overall performance of the proposed algorithm.

### Strengths
The paper addresses an interesting and important task of discriminative probabilistic modeling problem on a continuous domain for (multimodal) self-supervised representation learning. The idea appears to be novel in this field.

The theoretical analysis is thorough and with a good depth. The generalization error analysis is conducted to show the limitation of the existing algorithm. 

The experimental evaluation and comparison are convincing and sufficient to show the advantages of the proposed algorithm. Overall, it is a solid paper.

### Weaknesses
The weakness of the paper includes:

(1) As it is a theory heavy paper, it will be better to provide more motivation of the proposed method so that the readers can deeply understand why MIS is utilized and how is the convergence of MIS. In particulr, multiple importance sampling is a routine and well-known method. I believe that the uniqueness is probably the application of MIS in the particular framework. However, the author needs to specify and highlight what exact the uniqueness it is. (Is it a different math formulation etc)

(2) It is necessary to highlight the complexity and running time of the proposed method. I suggest to add a subsection for discussion the complexity of the proposed algorithm and comparing the complexity and running time with competing methods.

(3) The scope of the paper seems to be narrow (Can only be applied in  a continuous domain for (multimodal) self-supervised representation learning). For instance, can the proposed method be extend to unsupervised learning or semi-supervised learning ? Moreover, since it is multimodal based solution, can the proposed method be extended to text and image, image and audio etc ?

### Questions
Given the method is to handle discriminative probabilistic modeling problem on a continuous domain for (multimodal) self-supervised representation learning, the scope of the method that can be applied seems to be narrow. I am wondering if the method can be extended to other cases outside of self-supervised representation learning such as unsupervised learning.

It will be also useful to provide examples in the experimental results to show the method can be applied to image and text, audio and image with the connection to the proposed theory to show the advantages. So that the readers can deeply understand how to apply the theory to the real examples.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores discriminative probabilistic modeling in unimodal and multimodal self-supervised learning. By applying multiple importance sampling to Monte Carlo integration, it recovers the contrastive loss as a special case. The paper also identifies a generalization error limitation inherent in contrastive learning frameworks. To address this, it proposes a non-parametric method aimed at mitigating this error. The effectiveness of the approach is validated through experiments on multimodal datasets.

### Strengths
1.	It seems a novel and interesting perspective to improve self-supervised contrastive learning with discriminative probabilistic modeling.
2.	It provides theoretical insights into the generalization error limitation in self-supervised contrastive learning and presents a reasonable solution to address this problem.
3.	Experiments on CC3M and CC12M are conducted to verify the effectiveness of the proposed method.
4.	The paper is generally well-structured and easy to follow.

### Weaknesses
-	This paper claims that the vanilla contrastive learning objective (e.g., GCL) results in non-diminishing error terms due to a misalignment between the uniform distribution and the true data distribution on y. However, it is unclear if these error terms are substantial and whether they might worsen under long-tailed data distributions. Would incorporating long-tailed contrastive learning methods [1,2], which use automatic or prior population distribution estimation, help address this issue?
-	The paper lacks quantitative or qualitative results demonstrating that the proposed method effectively reduces the non-diminishing error terms (as seen in Eq. 6) on real-world datasets like CC3M and CC12M.
-	The proposed method may introduce additional computational overhead. It would be beneficial to discuss and compare the computational cost relative to baseline methods. 
-	According to Figure 4 in the ablation study, the proposed method appears to require specific design tricks for optimal performance, raising questions about its generalizability and applicability across varied scenarios. Clarifying whether substantial hyperparameter tuning is needed for different applications would be helpful.
-	The theoretical framework suggests applicability to unimodal contrastive learning. The paper would be further strengthened by demonstrating its effectiveness on common SSL benchmarks for unimodal data. 
-	The empirical evaluations focus on classification and retrieval tasks across four datasets (MSCOCO, Flickr, CIFAR100, and ImageNet). Including more challenging datasets and a wider variety of tasks would enhance the robustness of the results.

[1] The Hidden Uniform Cluster Prior in Self-Supervised Learning.
[2] Combating Representation Learning Disparity with Geometric Harmonization.

### Questions
please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
3
