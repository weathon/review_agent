# Permutations improve performance in three-dimensional bin packing problem

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3

## Abstract
In recent years, with the advent of deep learning and reinforcement learning, researchers have begun to explore the use of deep reinforcement learning to solve the three-dimensional bin packing problem. However, current innovations in the 3D bin packing problem primarily involve modifications to the network architecture or the incorporation of heuristic rules. Efforts to improve performance from the perspective of function approximation are relatively scarce. As is well known, one of the crucial theoretical foundations of deep learning is the ability of neural networks to approximate many functions. As such, we propose a method based on approximation theory that uses permutations to better approximate policy functions, which we refer to as Permutation Packing. Nonetheless, due to the high memory requirements when the number of permutations is large, we also propose a memory-efficient variation of Permutation Packing, which we call Limited-Memory Permutation Packing. Both methods can be efficiently integrated with existing models. We demonstrate the effectiveness of both Permutation Packing and Limited-Memory Permutation Packing from both theoretical and experimental perspectives. Furthermore, based on our theoretical and experimental results, we find that our methods can effectively improve performance even without retraining the model.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work presents a novel deep learning algorithm designed to tackle the challenging online 3D bin packing problem (BPP), a crucial combinatorial optimization task with applications across various industries. The inherent complexity of this problem stems from its vast combinatorial search space, which can lead to performance limitations in traditional operations research-based approaches, particularly in terms of speed and the need for domain-specific expertise.

Recent advancements in neural combinatorial optimization offer a promising avenue to address these limitations. Notably, deep reinforcement learning (DRL) methods can learn policies as effective combinatorial solvers by iteratively interacting with the problem environment, eliminating the reliance on problem-specific domain knowledge. However, it's worth noting that existing DRL techniques for combinatorial optimization have predominantly concentrated on vehicle routing problems (VRPs), leaving the 3D BPP relatively unexplored. Nonetheless, solving the 3D BPP still demands the development of specific techniques.

In this work, we introduce a permutation packing method and provide a mathematical proof demonstrating its capacity to enhance the approximation capabilities of deep neural networks. Given the potentially vast number of permutations involved, which could lead to increased memory consumption, we also propose a memory-efficient approach for permutation packing. Our experimental results empirically validate the effectiveness of these ideas, underscoring their practical relevance.

### Strengths
1. The idea of permutation packing is simple but novel. 

2. Writing is clear and enjoyable to read. 

3. Theoretical analysis is intuitive and highlights the motivation of the algorithm.

4. Ablations are deeply done and highlight the effectiveness of the method.

### Weaknesses
1. Algorithm pseudo code is too messy. It's really hard to read. Please improve the readability of the algorithm. 

2. Empirical results are not sufficient. Please do more experiments in different scenarios (e.g., distributional shift, scale shift). 

3. Real-world bin packing benchmark results are needed to verify performances.

4. This work is focused on the bin packing problem. The author should care about the extension of this paper's findings into a similar domain.

### Questions
1. Can this algorithm scale up to a larger scale? (e.g., $N>1000).

2. Can neural policy make good generalization capability to the unseen distribution of scales?

3. The reinforcement learning baseline is the rollout baseline. Did you try a more recent baseline for neural combinatorial optimization, such as a shared baseline [1,2]?


[1] Kwon, Yeong-Dae, et al. "Pomo: Policy optimization with multiple optima for reinforcement learning." Advances in Neural Information Processing Systems 33 (2020): 21188-21198.

[2] Kim, Minsu, Junyoung Park, and Jinkyoo Park. "Sym-nco: Leveraging symmetricity for neural combinatorial optimization." Advances in Neural Information Processing Systems 35 (2022): 1936-1949.

### Soundness
4 excellent

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper aims to address the 3D Bin Packing Problem, where the output of an good neural network should remain unchanged when the length, width, and height of the boxes are interchanged. Based on this premise, the authors propose an arrangement packing method to enhance the testing capability of deep reinforcement learning methods.

### Strengths
1. The authors introduce additional perturbations, conducting multiple searches for solutions to the offline bin packing problem. This approach, building upon the existing baseline's ability to output feasible solutions in a single run, enhances the performance of the original packing method.

### Weaknesses
1. The primary idea of the authors, leveraging the symmetry of boxes to conduct multiple searches for solutions, results in excessive packaging and applies too much complex theory. The authors should simplify their presentation.
2. Conducting multiple searches for solutions and comparing them with the baseline increases both time consumption and requires more data utilization. This type of performance comparison is not fair. Additionally, the reviewers noticed that the authors did not report the time taken for various methods in the article, which is misleading.
3. In the supplementary materials, the authors only submitted their own method and did not provide the baseline algorithm. This is really a concern for the reviewer. The reviewer hopes that during the revision, the authors will submit the baseline code, and this will be considered for a potential score increase.

### Questions
The reviewer has no further queries.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes two methods, Permutation Packing and Limited-Memory Permutation Packing, to enhance the performance of solving the three-dimensional bin packing problem using deep reinforcement learning. These methods utilize permutations to approximate policy functions more effectively and can be integrated with existing models. The effectiveness of these methods is demonstrated through theoretical and experimental analysis, showing improved performance without the need for retraining the model.

### Strengths
1. The paper tries to address an important problem in the field of optimization and combinatorial optimization, namely the three-dimensional bin packing problem. The paper proposes a novel method for solving the three-dimensional bin packing problem using permutations to better approximate policy functions. This approach is relatively new and innovative, as most current works in the field involve modifications to the network architecture or the incorporation of heuristic rules.

2. The methods are motivated by theoretical analysis, which is supported by the experiments including sufficient baselines.

### Weaknesses
1. This paper appears to have been hastily prepared and contains numerous typos. E.g., "." on top of Sec. 3.2; Missing "." in the caption of Figure 1.

2. It lacks discussion on the limitations.

3. The bounded function assumption is kind of strict.

### Questions
1. What are the limitations?

2. Can the bounded function assumption be relaxed?

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes Permutation Packing, as an decorator of the policy network in deep reinforcement learning, to introduce a permutation invariant mechanism for 3D bin packing problem. In particular, Permutation packing utilizes the symmetry and generate the policy using permutated replicas of the state representation. To reduce the memory overhead, the paper further proposes using Taylor's series to approximate the replica results. In experiments, the paper shows Permutation Packing is able to improve the performance of existing deep reinforcement learning methods for 3D bin packing in both training and test.

### Strengths
* The paper tries to use symmetry, which has been the core in modern science, to improve the policy in deep reinforcement learning for 3D bin packing. 
* The experiment results show that the proposed method can not only improve the training process, but also works as a plug-in to improve policy quality in inference.

### Weaknesses
* The presentation of the paper is poor.
    *  Missing details in methods. For example, the paper does not have formal definition of the state representation. After reading the paper, I am still not clear how the permutation are performed on states. Also, using the hessian matrix of a deep neural network could be difficult, the paper should elaborate more on limited-memory permutation.
    * Missing details in experiments. The paper only have one experiments, which seems follows the experiment in Attend2Pack in section 3.5. However, the paper does not provide any details like how the boxes are generated and how the training and evaluation are performed. Also, the performance of the baselines are different from the number reported in the original papers, making the experiment results less convincing.
    * The paper spends a lot of space on materials with low relevance. For example, the approximation theory in section 2 and section 3 is necessary to appear in the main text. Using one sentence to introduce the norm $||\cdot||_{\Omega, p}$ and the remaining can be put into appendix. 

* The soundness of the paper need more justification. The machine learning community has devoted significant effort for the symmetry in neural network. Some results, for example, using neural networks with special architectures like deep sets [1], self-attention [2], DSS layers [3]. For me, using an architected neural network admits symmetry is more straightforward than using the method proposed in this work. The paper should add more discussion, or experiments if applicable, about the difference and the advantage of Permutation Packing compared to these methods.


>[1] Zaheer, Manzil, et al. "Deep sets." Advances in neural information processing systems 30 (2017).\
[2] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).\
[3] Maron, Haggai, et al. "On learning sets of symmetric elements." International conference on machine learning. PMLR, 2020.

### Questions
* What is the state representation and how is the permutation performed?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
