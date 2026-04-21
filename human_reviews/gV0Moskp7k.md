# Combating the Generalization-Forgetting Trade-off in Continual Learning: A Cautious Passive Low-Rank Approach

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 3, 5, 5, 3

## Abstract
Large Language Models (LLMs) have shown remarkable capabilities through wide-scale pre-training on a wide range of domains. However, they often suffer from catastrophic forgetting when learning sequential tasks. In this paper, we propose a novel parameter-efficient approach for continual learning in LLMs, which empirically explores the role of different effective layerwise ranks, leveraging lower ranks to mitigate catastrophic forgetting of previous tasks and higher ranks to enhance generalization on new tasks. By employing a subspace similarity metric that evaluates the orthogonality of low-rank subspaces between tasks, we gradually increase the rank of layerwise matrices for each new task, minimizing interference with previously learned tasks while enhancing generalization. Experimental results on standard continual learning benchmarks and challenging math benchmarks demonstrate that our method outperforms existing state-of-the-art approaches, effectively mitigating forgetting, improving task performance, and maintaining strong generalization to unseen tasks in a memory-efficient manner.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript focused on continual learning (CL) with Large Language Models (LLM). The authors proposed a parameter-efficient approach based on the low-rank adaptation (LoRA) and explored the role of layerwise ranks in the incremental learning of LoRA between different tasks for CL. Through some empirical results, the authors observed that a trade-off between low ranks and high ranks can be leveraged to balance forgetting mitigation and generalization. Based on this motivation, the authors proposed Cautious Passive Low-Rank (CP-Rank) that gradually increases the rank of layerwise weight matrices during training. Specifically, the similarity of between-task low-rank subspaces is measured to evaluate the orthogonality between subspaces, and then whether to cautiously increase the ranks or passively maintain the current ranks can be determined for each task. Experiments on several benchmarks were conducted to support the effectiveness of the proposed method CP-Rank.

### Strengths
1. The motivation of this manuscript is clear. Determining how to increase the ranks is also novel for the continual learning for LLM with LoRA strategy.
2. Extensive experiments on different benchmarks were conducted to demonstrate the effectiveness. The datasets adopted in this manuscript include a wide range of task types.
3. This paper is well-written and easy to follow.

### Weaknesses
1. The mathematical presentation can be further improved. There are some confusing points that need to be clarified.
2. The paper’s focus is primarily on state-of-the-art low-rank methods, but including a few recent non-low-rank continual learning methods as baselines could provide a broader performance perspective.

### Questions
1. In Line 198, the sentence "Eq. 7 uses the singular values captured by two different task subspaces, which matches our findings in Eq. 5" is hard to follow. It would be better if the authors could provide intuitive explanations about the connection with Eq. 5.
2. In Eq. (2), I guess there is a typo for the subscript before the minus sign. Should it be $\mathbf{\theta}_N$ instead of $\mathbf{\theta}_T$? Besides, in Eq. (2) and Eq. (3), the parenthesis can be added for the different terms of RHS to avoid confusion.
3. In Eq. (7), I wonder if there are some typos about the indices (e.g., $i,s$). This equation is confusing.
4. In Algorithm 1, I noticed that an interval $k$ was set to control the operations within Step 4. However, I didn't see any description of this point. Could the authors explain the role of this interval $k$?
5. Some superscripts in Section 2 are a little messy. For example, in Line 177, the superscript is adopted to indicate the layer $l$. However, in Algorithm 1 and Algorithm 2, the matrices $\mathbf{A}$ and $\mathbf{B}$ were subscripted with a time step $t$. This confusing part needs to be further clarified.
6. I noticed that other studies, such as InfLoRA [1], also considered orthogonality between the subspaces of different tasks during the LoRA adaptation. Could the authors summarize the main differences between their proposed CP-Rank ad InfLoRA?

References;

[1] InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning. CVPR 2024.

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
This paper proposes CP-Rank (Cautious Passive Low-Rank), a continual learning strategy for LLM to balance the trade-off between low ranks and high ranks to balance forgetting mitigation and generalization based on LORA. It is observed that updating a high-rank weight on new task is likely to contribute to better performance on the new-task but greater forgetting previous tasks, and vice versa. CP-Rank progressively and adaptively increase the rank of derivative weight matrix the new task according to its similarity with previous tasks'.
Experiments show that it is effective.

### Strengths
The method is intuitive and basically reasonable. It is shown that the method could result in significant advantage in empirical performance.

### Weaknesses
1. The most concerned weakness is the efficiency problem of CP-Rank. In terms of space complexity, it is required to memorize the derivative weight matrix on every layers of every task, which takes NLd^2. In terms of time complexity, to learn a single new task, there are additionally T/k times to refer to all the NL matrix in memory and respectively perform SVD and evaluate the Grassmann similarity. It is like to be very inefficient in both space and time.
2. The very first experiment that gives rise to the fundamental question is not convincing enough. The results on only one pair of tasks are shown, which is far not enough to support the observed pattern statistically. Though it is intuitive that low-rank updating has advantage in unforgetting and disadvantage in new new loss compared with high-rank.
3. Eqn(5) dos not match the method design well. There are obvious gap between Eqn(5) and the method design in terms of problem setting and similarity metric.
4. There are some redundant contents and detail errors that trouble reading. For example, it seems that eqn(2)(3)(4) is very unnecessary; The $T$ in eqn(2) is likely to be correct to be $N$; Eqn(7) is not likely to has a codomain $(0,1)$.

### Questions
There is one main question that the reviewer is curious about and would appreciate to discuss with the authors: Is there a correlation between the rank of the derivative matrix of the new task and its mean similarity with previous tasks? It is demonstrated that seems similarity and rank are two independent factors. CP-Rank progressively increase the rank under a similarity threshold, which means the similarity corresponding to unforgetting, is prior to the rank corresponding to better fitting. This is reasonable from a practical perspective to just adding untrained rows and columns to AB while not disturbing the existing rows and columns. But it is likely that there is such a correlation which can be exploit. 

For example, a positive correlation can guide the early-stopping during increasing the rank to be more efficient.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents CP-Rank, a parameter-efficient continual learning method that progressively increases the layer-wise rank of LoRAs for new tasks, guided by a low-rank subspace similarity metric between tasks. CP-Rank not only models the low-rank relationships between tasks' incremental LoRAs but also adapts to the unique low-rank dynamics across different layers of the model. This approach effectively prevents forgetting of previous tasks while improving generalization on new tasks, all within a memory-efficient framework. Extensive experiments on natural language processing and complex math reasoning tasks demonstrate that CP-Rank effectively captures rank patterns in continual learning and consistently outperforms existing approaches.

### Strengths
* This paper is easy to follow.

* The proposed approach is simple and easy to understand.

### Weaknesses
* The method would benefit from evaluation on a larger language model, as T5-large may not be sufficiently large for comprehensive assessment.

* The approach for increasing the LoRA rank appears heuristic and lacks sufficient justification. A deeper explanation and analysis of the rank-increasing strategy would strengthen the paper.

* The proposed method introduces several hyperparameters (e.g., $k$, $\epsilon$, and the rank-increasing mechanism), which could complicate its practical application. A simpler or more streamlined approach might improve usability.

* The relationship between low rank and forgetting, and high rank and generalization, is not well-justified. While the introduction offers empirical evidence of this correlation, a formal explanation or theoretical foundation is missing. Additionally, there is no clear discussion of whether the method generalizes to larger models, such as Llama.

* Other:
The font size in Figure 1 is too small, making it difficult to read. Increasing the size would improve clarity.

### Questions
N/A

### Soundness
2

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
5

### Summary
This paper presents a continual learning method called CP-RANK, which achieves a balance between mitigating catastrophic forgetting and enhancing generalization by adjusting the rank across different layers. CP-RANK assigns each task a LoRA module as its adapter, using SVD decomposition to obtain the left singular matrix to assess the similarity between task subspaces. By comparing this similarity to a threshold, it determines whether the tasks are orthogonal. If they are orthogonal, the rank is increased; otherwise, the rank is maintained. Additionally, the paper introduces an improved approach for orthogonal projection. Comparative experiments are conducted to demonstrate the effectiveness of this method, and a necessary discussion on the experiments is provided.

### Strengths
- This paper proposes that low ranks are beneficial for resisting catastrophic forgetting, while high ranks are advantageous for learning new knowledge, with experiments conducted to support this perspective.
- The paper introduces a continual learning method called CP-RANK, which assigns different ranks to matrices at different layers and leverages rank increase to achieve a balance between mitigating catastrophic forgetting and enhancing generalization. Notably, the method incorporates an orthogonal projection algorithm, yielding promising results in experiments.
- It introduces the use of the left singular matrix to calculate the similarity between different subspaces to assess their orthogonality.
- The experimental section specifically evaluates performance on mathematical tasks.
- The paper provides detailed algorithms and formulas relevant to the proposed methods.

### Weaknesses
- The experimental results indicate that CP-RANK without orthogonal decomposition performs worse than O-LoRA, and in testing its resistance to catastrophic forgetting, the original CP-RANK setup was not included. This may be insufficient to demonstrate the effectiveness of rank adjustments for continual learning, as it still primarily relies on the orthogonal projection method used by O-LoRA.
- In Algorithm 2, for situations where rank increase is needed, the newly added matrix elements are initialized randomly. I find the explanation for this part of the algorithm unclear: after increasing the rank, is additional training required? Should orthogonality be reassessed?
- I believe that using "plasticity" (the capacity to learn new knowledge) in place of "generalization" might be more appropriate in this context, as the paper does not include experiments specifically validating generalization.

### Questions
- The paper claims that a major advantage of this model is that it can operate without relying on task IDs. I would like to understand how the model achieves task recognition and adapter selection in this case.

### Soundness
3

### Presentation
2

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
This work addresses the problem of continual learning for large language models (LLMs) and designs a mechanism to gradually increase the rank for the continual fine-tuning of LLMs.

### Strengths
The problem considered in this paper is meaningful in continual learning, and the parameter-efficient fine-tuning method employed is also a mainstream research approach currently.

### Weaknesses
1. Using only T5-large is insufficient to verify the effectiveness of the model on large language models (LLMs). Specifically, this paper claims to address the issue of continual fine-tuning in large models. However, the largest model used is only T5-large, which has only 770M parameters, far fewer than many existing LLMs such as Llama-2-7b, Llama-2-13b. The author should follow existing continual learning works [1, 2] that consider LLMs and perform experiments using Llama-2-7b and Llama-2-13b.

2. According to Equation 5, the motivation of this work is to keep $tr(B_{i}A_{i}(B_{j}A_{j})^{T})$ for ($j<i$) as small as possible when learning the i-th task. However, the algorithm ultimately aims to keep Equation 7 as small as possible. Equation 7 and $tr(B_{i}A_{i}(B_{j}A_{j})^{T})$ are not equivalent, leading to a mismatch between the motivation and the algorithm in this paper.

3. How is the second regularization term in Equation 8 derived? Because the lora branch $B_{i}A_{i}$ needs to be gradually updated during the learning of the $i$-th task, each calculation of the regularization term in Equation 8 requires SVD decomposition of $B_{i}A_{i}$, which will incur significant computational overhead.

[1] Zhao W, Wang S, Hu Y, et al. Sapt: A shared attention framework for parameter-efficient continual learning of large language models. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024: 11641-11661.

[2] Wang X, Chen T, Ge Q, et al. Orthogonal subspace learning for language model continual learning[J]. arXiv preprint arXiv:2310.14152, 2023.

### Questions
Since the motivation of this algorithm is to maintain the orthogonality of $tr(B_{i}A_{i})$ and the lora branches of old tasks when learning a new task $i$, it is unreasonable to use a Gaussian distribution for random initialization of $B_{i}$. In other words, why not adopt some operations to maintain the orthogonality of $B_{i}A_{i}$ and the lora branches of old tasks during the initialization of $A_{i}$ and $B_{i}$?

### Soundness
2

### Presentation
2

### Contribution
2
