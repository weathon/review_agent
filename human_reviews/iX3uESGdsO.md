# Selective Aggregation for Low-Rank Adaptation in Federated Learning

- Decision: Accept (Poster)
- Scores: 6, 5, 8, 5

## Abstract
We investigate LoRA in federated learning through the lens of the asymmetry analysis of the learned $A$ and $B$ matrices. In doing so, we uncover that $A$ matrices are responsible for learning general knowledge, while $B$ matrices focus on capturing client-specific knowledge. Based on this finding, we introduce Federated Share-A Low-Rank Adaptation (FedSA-LoRA), which employs two low-rank trainable matrices $A$ and $B$ to model the weight update, but only $A$ matrices are shared with the server for aggregation. Moreover, we delve into the relationship between the learned $A$ and $B$ matrices in other LoRA variants, such as rsLoRA and VeRA, revealing a consistent pattern. Consequently, we extend our FedSA-LoRA method to these LoRA variants, resulting in FedSA-rsLoRA and FedSA-VeRA. In this way, we establish a general paradigm for integrating LoRA with FL, offering guidance for future work on subsequent LoRA variants combined with FL. Extensive experimental results on natural language understanding and generation tasks demonstrate the effectiveness of the proposed method. Our code is available at https://github.com/Pengxin-Guo/FedSA-LoRA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new way of combining LoRA in the federated setting, called FedSA-LoRA, where the authors found that A matrics are responsible for learning general knowledge, and B matrics focus on client-specific knowledge. Only A matrics are used for server aggregation. Extensive experiment results indicate the effectiveness of the proposed FedSA-LoRA with various LoRA-related variants.

### Strengths
- Observing the roles that matrices A and B play is intuitive and important, and this observation is also prevalent in several LoRA-related variants.
- The experiments are comprehensive, covering different non-iid levels and client numbers and across different tasks. Also, the authors provide many ablation studies.
- Authors provide some theoretical justifications for the convergence of their method.

### Weaknesses
- Motivation for exploring LoRA in federated learning: Why do authors use and investigate Lora-based methods in FL? The authors can expand and explain more about why it is good compared with parameter-efficient methods. In the introduction and related works, the authors state they are efficient, effective, and flexible, which is a bit too general from my perspective. The authors can better motivate the importance of this problem.
- Line 52-71: First, I feel that what kind of aggregation errors is unclear to me: could the authors provide a more accurate and specific illustration of the aggregation errors? Secondly, I wonder in practice, why do people not do the ideal model update? What are the difficulties in doing that?
- Authors can provide some computational analysis (e.g. time and space cost) of their method compared with the baselines.
- The authors could include standard deviations or confidence intervals for their main results tables, particularly for the main results.
- On the natural language generation task, the improvement of the proposed method seems to be a little marginal. Also, it is tested under IID distribution.
- The overall contribution seems to be marginal from my perspective.

### Questions
- The authors mention in the experiment section that the proposed method is more effective when non-IID conditions are more severe. Could the authors provide some potential reasons for that?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper begins by analyzing the roles of matrices A and B in LoRA, concluding that matrix A is responsible for general knowledge while matrix B handles domain-specific knowledge. Based on this analysis, the authors propose FedSA-LoRA, a method that aggregates only the A matrix during federated learning while retaining the edge-side B matrix. This approach aims to integrate general capabilities while preserving personalized knowledge.

### Strengths
(1) The paper's analysis of the distinct roles of matrices A and B clearly demonstrates its insights and motivations.

(2) The proposed method improves upon existing solutions and achieves certain performance enhancements.

### Weaknesses
(1) One of the significant contributions of the paper is the asymmetric analysis of matrices A and B, concluding that matrix A is responsible for general knowledge and matrix B for domain-specific knowledge. However, this perspective has already been proposed in several works, such as HydraLoRA [1], which also designed an asymmetric LoRA structure based on this concept. The paper should properly cite relevant literature, and these existing findings somewhat diminish the paper's contribution.

(2) The experimental results are based on a single run, lacking multiple experiments to calculate averages and perform variance analysis, which may affect the reliability of the results. It is recommended that the authors refer to previous related work [2], conduct multiple experiments, and compute means and variances to more comprehensively evaluate the method's stability and effectiveness.

(3) The paper mainly considers the sharing of edge-side models but lacks a reasonable mechanism to update the server-side model to endow it with comprehensive capabilities.

[1] HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning
[2] Improving LoRA in Privacy-preserving Federated Learning

### Questions
(1) In the experimental results, why does FedSA-LoRA outperform other methods under non-heterogeneous conditions? In the absence of heterogeneity, since both matrices A and B can learn shareable knowledge, sharing information from both matrices—or sharing matrix B's information as in FFA-LoRA—should intuitively achieve equal or even greater gains.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors present FedSA-LoRA, a novel approach for federated learning that shares only the A matrices across clients to leverage the functional differences between A and B matrices in LoRA modules. The method’s effectiveness is validated through both theoretical analysis and empirical results.

### Strengths
1. The paper is generally well-motivated and easy to follow.
2. This work introduces the shared-A LoRA technique into the FL framework, making it promising compared to previous work where the down-projection matrix (A matrix) is kept frozen.
3. Comprehensive experimental results are provided.

### Weaknesses
1. While this work introduces a shared-A LoRA framework into FL, such a technique has already been introduced in the MoE area [1] under similar motivations. Although the novelty of introducing it into FL should be recognized, the introduction of such a framework cannot be solely credited to this work.

2. On page 20, Figure 3, the authors claim that "A matrices are different from the initialized A matrices, indicating that the A matrices are updated." However, the cosine similarity between the learned and initialized down-projection matrices remains quite high (greater than 0.985), which indicates that the A matrices have changed very little compared to the initialization.

3. Compared with [2], the cosine similarity across A matrices for different clients reported in this work is very high. Why does this happen? Is it caused by different levels of data heterogeneity?

4. Assumption 3, especially the last two inequalities, lacks verification and explanation. It appears these are purely for simplifying the proof of Theorem 1 on page 16. The authors need to elaborate more on this uncommon assumption, including under what conditions it could hold and what properties of the A and B matrices this assumption reveals.

5. In the provided code, alongside the B matrices being personalized, a classifier layer is also personalized. Is this necessary? How does this personalized classifier layer influence the experimental results?

References:

[1] Tian, Chunlin, et al. "HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning." arXiv preprint arXiv:2404.19245 (2024).

[2] Zhu, Jiacheng, et al. "Asymmetry in low-rank adapters of foundation models." arXiv preprint arXiv:2402.16842 (2024).

### Questions
See above. I will increase my score if my concerns are well addressed.

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
This paper proposes a low-rank federated fine-tuning algorithm, namely FedSA-LoRA, which regards $A$ as a global adapter and $B$ as a personalized adapter. This algorithm reduces the communication cost by only aggregating matrices $A$.

### Strengths
This paper has the interesting motivation and shows the different roles of low-rank matrices $A$ and $B$ in federated fine-tuning.

### Weaknesses
I have the following concerns:

1. Lemma 1 is very interesting to me and the verified experiments show that all the matrices $A_i$ ($i$ is the client index) seems to be the same and $B_i$ differs with each other. We know that different client has independent initialization of $A$, but finally with the exactly same $A$ since the similarity of $A$ from clients is $1.0$ in Figure 2. As you mentioned in Figure 3 in Appendix that the learned A matrices are different from the initialized A matrices, so how do you guarantee that the learned A across clients are the totally same even in the server non-IID scenario.

2. The implementation details for Figure 2 are not clear to me, especially how do you implement federated lora fine-tuning, do you update/aggregate both $A$ and $B$?

3. In Assumption 1, what is the meaning of $\mathcal{L}_i(W_j)$ and $\nabla \mathcal{L}_i(W_j)$? In your descriptions, both $i$ and $j$ are the client's index, right? Then how can you access the client-$j$'s gradient information in client $i$?  

4. Assumption 2 is too strong, it becomes very easy to handle dissimilarity between clients and stochastic gradients with this assumption. This assumption doesn't appear in smoothness cases in many federated learning papers, so how do you verify this assumption? Can you provide examples to verify the rationality.

5. As I mentioned in concern 4, assumption 2 is a strong assumption, which does not appear in work [1]. Even the same convergence rate as [1] is achieved, that comparison is meaningless to me. 

[1] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective inconsistency problem in heterogeneous federated optimization. Advances in neural information processing systems, 33:7611–7623, 2020.

6. In experiments, they compare with FFA-LoRA, and show better performance than FFA-LoRA. My concern is that, is this a fair comparison? Because the trainable parameters are not the same, FFA-LoRA does not tune the matrices $A$, but FedSA-LoRA does, which means FedSA-LoRA has much more trainable parameters than FFA-LoRA. I hope to see the report of trainable parameters for each algorithm. 

7. Many advanced baselines are not compared, so the performance of the proposed algorithm is not convincing. For example, some related work mentioned in the paper, FedLoRA [2], FedDPA [3], and HetLoRA [4]. 

[2] Liping Yi, Han Yu, Gang Wang, and Xiaoguang Liu. Fedlora: Model-heterogeneous personalized federated learning with lora tuning. arXiv preprint arXiv:2310.13283, 2023. \
[3] Yiyuan Yang, Guodong Long, Tao Shen, Jing Jiang, and Michael Blumenstein. Dual-personalizing adapter for federated foundation models. arXiv preprint arXiv:2403.19211, 2024.\
[4] Yae Jee Cho, Luyang Liu, Zheng Xu, Aldi Fahrezi, Matt Barnes, and Gauri Joshi. Heterogeneous lora for federated fine-tuning of on-device foundation models. In International Workshop on Federated Learning in the Age of Foundation Models in Conjunction with NeurIPS 2023, 2023.

### Questions
Please refer to Weakness.

### Soundness
2

### Presentation
3

### Contribution
2
