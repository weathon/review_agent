# Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Low-Rank Adaptation (LoRA) has become ubiquitous for efficiently fine-tuning foundation models.  However, federated fine-tuning using LoRA is challenging due to suboptimal updates arising from traditional federated averaging of individual adapters. Existing solutions either incur prohibitively high communication cost that scales linearly with the number of clients or suffer from performance degradation due to limited expressivity. We introduce **Federated Silver Bullet (Fed-SB)**, a novel approach for federated fine-tuning of LLMs using LoRA-SB, a recently proposed low-rank adaptation method. LoRA-SB optimally aligns the optimization trajectory with the ideal low-rank full fine-tuning projection by learning a small square matrix ($R$) between adapters $B$ and $A$, keeping other components fixed. Direct averaging of $R$ guarantees exact updates, substantially reducing communication cost, which remains independent of the number of clients, and enables scalability. Fed-SB achieves **state-of-the-art performance** across commonsense reasoning, arithmetic reasoning, and language inference tasks while reducing communication costs by up to **230x**. In private settings, Fed-SB further improves performance by (1) reducing trainable parameters, thereby lowering the noise required for differential privacy and (2) avoiding noise amplification introduced by other methods. Overall, Fed-SB offers a state-of-the-art, efficient, and scalable solution for both private and non-private federated fine-tuning. Our code is available anonymously at: https://anonymous.4open.science/r/fed-sb-anonymous-6F3D.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method, Fed-SB, to achieve exact aggregation in federated LoRA fine-tuning without the high communication costs of prior methods. Fed-SB initializes LoRA adapters (B and A) using an SVD-based approximation of the first full fine-tuning update step and then freezes them, training and communicating only a very small $r \times r$ matrix R between two adaptors. This allows for mathematically exact aggregation through simple averaging of the R matrices, drastically reducing communication cost.

### Strengths
S1: The core idea is intuitive and easy to follow. 
S2: The method demonstrates strong empirical performance across various benchmarks, while using drastically fewer communication parameters per round.

### Weaknesses
W1: The paper lacks clarity on the initialization phase's implementation details. 
W2: Model performance appears highly sensitive to the initial subspace quality, risking a severe performance cap if the approximation phase uses limited or unrepresentative data.
W3: The constrained update space (only matrix R is trained) raises overfitting concerns to the initial subspace, potentially limiting generalization.

### Questions
Q1: Is communication required during initialization? If so, what is its parameter scale, and was this one-time cost included in the communication efficiency analysis?
Q2: How does initialization robustness perfom with more clients and higher data heterogeneity? In large-scale or highly heterogeneous scenarios, does the method require significantly more data samples or resource overhead to obtain a sufficiently representative initial subspace?
Q3: How do alternative SVD-based initializations (e.g., LoRA-GA[1]) for the fixed A and B matrices compare against the proposed method within the same B-R-A architecture?

[1] Wang, S., Yu, L., and Li, J. LoRA-GA: Low-Rank Adaptation with Gradient Approximation. In Advances in Neural Information Processing Systems 38 (NeurIPS 2024), 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Fed-SB, a federated adaptation of LoRA-SB. Fed-SB freezes local low-rank adapters and trains an $r \times r$ matrix $R_i$ for each client. The server aggregates by averaging only $R_i$. This makes aggregation algebraically exact and keeps per-round communication low. The approach is also compatible with Differential Privacy. The experiments show consistent gain across multiple tasks and multiple LLMs.

### Strengths
1. The method is simple and effective.
2. The experiments show consistent gains across multiple tasks and multiple LLMs.

### Weaknesses
1. Despite strong experimental results, the submission lacks the core justification (theoretical or isolating experiments) that would explain why the method works and under what conditions it should be expected to work.
2. Lack of key ablation studies:
 (1) extreme low-rank regimes (rank-1, rank-2, …), to identify the expressivity threshold below which Fed-SB ceases to be effective; (2) different initializations of LoRA-B and LoRA-A, since LoRA-A, LoRA-B are frozen, initialization may be the dominant factor; (3) experiments involving more clients, such as 30 or more. 
3. Table 2-6 should also report standard deviation.

### Questions
See Weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces Federated Silver Bullet (Fed-SB), an approach for federated fine-tuning of LLMs using LoRA-SB, a recently proposed low-rank adaptation method. LoRA-SB optimally aligns the optimization trajectory with the ideal low-rank full fine-tuning projection by learning a small square matrix (R) between adapters B and A, while keeping other components fixed. Fed-SB directly averages R while keeping other components fixed (including A and B), which guarantees exact updates, substantially reduces communication costs, remains independent of the number of clients, and enables scalability.

### Strengths
1. The paper is well-written and easy to understand.
2. The introduced method achieves exact updates for FL with LoRA fine-tuning.

### Weaknesses
1. The novelty of this work is limited. The proposed Fed-SB is built upon LoRA-SB [1], which is not an original contribution of this study. Moreover, such strategies have been applied in [2].
2. The initialization of A and B can significantly impact performance, yet this paper lacks such an analysis, as it directly initializes A and B as orthonormal matrices. Exploring different initializations, such as random initialization or performing SVD decomposition on W_0 and using the decomposed values as the initialization for A and B, is necessary. Similar investigations into how different initializations affect model performance are needed.
3. Some advanced works are missing and should be included for comparison, such as FlexLoRA [3], FedSA-LoRA [4], FRLoRA [5], and others.


[1] Kaustubh Ponkshe, Raghav Singhal, Eduard Gorbunov, Alexey Tumanov, Samuel Horvath, and Praneeth Vepakomma. Initialization using update approximation is a silver bullet for extremely efficient low-rank fine-tuning. arXiv preprint arXiv:2411.19557, 2024. \
[2] Guo, Wei, Siyuan Lu, Yiqi Tong, Zhaojun Hu, Fuzhen Zhuang, Xiao Zhang, Tao Fan, and Jin Dong. "H2Tune: Federated Foundation Model Fine-Tuning with Hybrid Heterogeneity." arXiv preprint arXiv:2507.22633. \
[3] Jiamu Bai, Daoyuan Chen, Bingchen Qian, Liuyi Yao, and Yaliang Li. Federated fine-tuning of large language models under heterogeneous tasks and client resources. In Proceedings of the 38th International Conference on Neural Information Processing Systems, 2024. \
[4] Pengxin Guo, Shuang Zeng, Yanran Wang, Huijie Fan, Feifei Wang, and Liangqiong Qu. Selective aggregation for low-rank adaptation in federated learning. In The Thirteenth International Conference on Learning Representations, 2025. \
[5] Yunlu Yan, Chun-Mei Feng, Wangmeng Zuo, Rick Siow Mong Goh, Yong Liu, and Lei Zhu. Federated residual low-rank adaptation of large language models. In The Thirteenth International Conference on Learning Representations, 2025.

### Questions
In Line 312, "Fed-SB: Pushing the Pareto Frontier," what is the meaning of 'Pareto Frontier'? To my understanding, the Pareto Frontier is a concept in multi-objective optimization. How is it applied here?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Fed-SB, a communication efficient federated LLM fine tuning approach that directly adopts a prior LoRA-SB method. The key point of Fed-SB is that it only requires aggregating small rank by rank matrices among participating clients in the federated learning process. Experimental results have been provided to justify the effectiveness of Fed-SB.

### Strengths
- The paper is well written and well motivated in general.  
- Seeking to improve the communication efficiency of federated LLM fine tuning seems to be an interesting research direction.  
- The proposed Fed-SB method is intuitive and easy to follow.  
- The experimental results of the Fed-SB method seem to be promising.

### Weaknesses
- The contribution of the proposed Fed-SB method seems to be marginal. I find it hard to identify significant algorithmic innovation, except for extending the LoRA-SB method to a federated setting.  
- The base model used in the experimental study seems to be somewhat outdated.  
- I am not sure if federated LLM fine tuning is a practical scenario, as centralized fine tuning appears to be dominating. And I failed to find any real-world adopting for federated LLM fine-tuning.

### Questions
- What is the major motivation for federated fine tuning? If it is for obtaining higher quality SFT data, what would be the most reasonable tasks to work on? And why has it not been adopted in real world LLM fine tuning practices (what are the major barriers)?  
- How would Fed-SB perform, for example, on Qwen 3?  
- Can Fed-SB be extended to support RL workflows for reasoning?

### Soundness
2

### Presentation
3

### Contribution
2
