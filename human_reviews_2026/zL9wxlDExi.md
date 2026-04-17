# ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning

- Decision: Reject
- Scores: 6, 4, 2, 8, 4

## Abstract
As large language models (LLMs) continue to scale in size, the computational overhead has become a major bottleneck for task-specific fine-tuning. While low-rank adaptation (LoRA) effectively curtails this cost by confining the weight updates to a low-dimensional subspace, such a restriction can hinder effectiveness and slow convergence. This contribution deals with these limitations by accumulating progressively a high-rank weight update from consecutive low-rank increments. Specifically, the per update optimal low-rank matrix is identified to minimize the loss function and closely approximate full fine-tuning. To endow efficient and seamless optimization without restarting, this optimal choice is formed by appropriately scaling the columns of the original low-rank matrix. Rigorous performance guarantees reveal that the optimal scaling can be found analytically. Extensive numerical tests with popular LLMs scaling up to 12 billion parameters demonstrate a consistent performance gain and fast convergence relative to state-of-the-art LoRA variants on diverse tasks including natural language understanding, commonsense reasoning, and mathematical problem solving.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies cumulative LoRA for fine-tuning models, that is the low rank updates are frequently merged into the base model in order to increase the overall rank of the adapters. This problem poses the question of how to initialize the low rank matrices after each merge. This paper provides a theoretical analysis of the initialization. Assuming the smoothness of the loss function, and minimizing the upper bound for the function progress via smoothness as a proxy problem, the paper shows that the optimal initialization is obtained by truncating the SVD of the gradient matrix. The paper then focuses on the initialization by scaling the previous matrices in two cases: scalar and column scaling. In both cases, the authors derive the optimal scaling. In experiment, the paper show the performance gain of ScaLoRA in various tasks as well as the runtime and memory requirements in comparison with existing methods.

### Strengths
- The strength of the paper lies in a rigorous analysis of the methods for initializing A and B matrices when restarting. In all settings the paper studies, the authors are able to derive the optimal strategies.
- In experiment, the proposed methods show a notable performance gain. Specifically, the variant with I=10 performs better than the other high-rank methods (HiRA, MoRA), on the Commonsense reasoning baseline with much better overhead.
- The presentation of the paper is overall easy to follow.

### Weaknesses
- The main weakness of the paper is that the idea of initialization of the A, B matrices using gradient approximation is known from LoRA-GA (Wang et al. 2024). While Wang et al. 2024 doesn't show both the sufficient and necessary conditions, they have started the main idea. That said, to my knowledge, column scaling is novel.
- The vanilla ScaLoRA has a very high time overhead
- Another limitation is the algorithm is tested with small r (the high rank baseline is LoRA with r=32).
- One missing baseline: I think the authors should add the baseline which uses Theorem 1 and does the merging after I iterations, for a suitable I (ie, cumulative LoRA + LoRA-GA for initialization)
- A few minor points: Line 160: what do the authors mean by "directly minimizing the loss function is generally infeasible"; don't we usually optimize the loss function directly? Line192: L is used for smoothness but also the time, which might be confusing.

### Questions
- Please see the weaknesses. I suggest adding experiments with higher rank baselines and the baseline with LoRA-GA for initialization.

### Soundness
3

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
4

### Summary
This paper introduces ScaLoRA, a theoretically grounded extension of LoRA that aims to recover high-rank fine-tuning behavior while preserving parameter efficiency. The key idea is to progressively accumulate high-rank updates by optimally scaling LoRA’s low-rank adapters at each iteration (or periodically, in ScaLoRA-I). The authors derive closed-form optimal scaling rules for scalar and column-wise transformations, ensuring analytical tractability and compatibility with adaptive optimizers like AdamW, avoiding the need for restarts. Experiments across model families and tasks show consistent but modest gains over LoRA, HiRA, and MoRA, with ScaLoRA-I maintaining LoRA level runtime and compute efficiency. The work is mathematically neat and clearly written, though the empirical impact remains relatively limited.

### Strengths
`Strong theoretical foundation`

The derivations are rigorous and provide analytical insight into optimal scaling for low-rank adapters, bridging LoRA’s empirical intuition with optimization theory.


`Practical implementation design`

The method integrates well with adaptive optimizers without restarting gradient statistics.


`Consistent performance and fast convergence`

The approach improves over LoRA and some recent high-rank variants (HiRA, MoRA) and achieves similar performance to higher-rank LoRA models with fewer parameters.


`Clear complexity and efficiency analysis`
The paper transparently reports computational and memory tradeoffs, and ScaLoRA-I effectively achieves LoRA-level runtime and compute efficiency.

### Weaknesses
`W1: Marginal empirical gains relative to complexity`

While ScaLoRA improves over LoRA and its recent high-rank variants (HiRA, MoRA), the differences are consistently minor.In many cases (e.g., Table 2, LLaMA-3-8B results), the rank-8 ScaLoRA matches rank-32 LoRA, but the additional complexity and theoretical machinery do not clearly justify the incremental benefit in using rank-8 vs rank-32 adapters (since the memory usage would be very similar in this case). This weakens the practical motivation: practitioners may prefer slightly higher LoRA ranks rather than adopting a more complicated method.


`W2: Lack of systematic analysis for iteration frequency (I)`

The intermittent scaling variant, ScaLoRA-I, is an important part of the paper’s efficiency argument, yet its behavior is not thoroughly studied. Without understanding how performance varies with I, it is unclear how sensitive the method is or how one should set this hyperparameter in practice.


`W3: Evaluation focuses on tasks with small LoRA–FT gap`

The chosen benchmarks are domains where LoRA already performs very close to full fine-tuning, leaving limited room to showcase ScaLoRA’s benefits. On these tasks, even basic LoRA performs more or less like full FT. The authors’ claim that ScaLoRA better approximates full fine-tuning would be more convincing if tested on harder domains, where LoRA’s performance drop is more substantial.

### Questions
- Q1: Can you please provide experiments comparing ScaLoRA with ReLoRA and ABBA, which also aim to recover high-rank or high-expressive updates under a PEFT setting?


- Q2: Can you please expand the discussion comparing ScaLoRA with LoRA-Pro and LoRA-SB? I would like to understand how the approaches differ conceptually - the 2 mentioned papers aim to minimize deviation between LoRA and full fine-tuning updates.


- Q3: Can you please add ablations on the iteration number I to show how scaling frequency affects accuracy and convergence speed?


- Q4: Can you please evaluate on a task where LoRA performs much worse than full fine-tuning to better demonstrate ScaLoRA’s effectiveness in approximating full FT?

---

I would be happy to increase my score if the authors are able to clarify my questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method for fine-tuning LLMs through iterative merging of low-rank matrices. At each step, an alternative low-rank matrix is computed by scaling the standard LoRA components with either a scalar or a diagonal matrix. The scaling coefficients are determined by minimizing a quadratic upper bound using the standard smoothness assumption. Numerical experiments have been reported on several standard fine-tuning benchmarks, demonstrating its effectiveness.

### Strengths
The paper aims to address the limitations of LoRA and has a clear motivation.

### Weaknesses
We typically have three standard classes of memory-efficient optimization approaches:

1. LoRA style methods: These treat A and B as two fixed learnable parameters and directly optimize them using an optimizer. This approach is memory-efficient during both training and model storage, particularly when fine-tuning multiple task-specific models.

2. Memory-efficient optimizers. Methods that modify the optimization procedure itself to reduce memory usage. A closely related work is Chain of LoRA [1]: At each iteration $t \ge 0$, we aim to approximately solve the subproblem: $\min_{A_t,B_t}\{f(W_t+A_t B_t^T)\}$ where $W_{t+1} = W_t + A_t B_t^T$.

This paper seems to follow the second style. However, the whole logic and the proposed methodology appear to be somewhat confusing and unclear. 

1. Unlike eqn (1), it is unclear why $A_t$ and $B_t$ are updated using eqn (4). 
2. In the equation in the middle between (6) and (7), it is unclear why consider minimizing the upper bound of $\ell (W_t + \Delta \tilde{W}_t)$ instead of the one for $\ell (W_t +  \Delta W_t)$. 
3. Following 2, it seems the paper wants to update $\Delta W_t \approx - \frac{1}{L} \nabla \ell (W_t)$, then the motivation of introducing $\tilde{A}$ and $\tilde{B}$ is unclear to me. Moreover, ideally, one might expect an update of the form $\Delta W_t \approx -\eta_t m_t$ where $m_t$ is the momentum as if we do the full fine-tuning using, e.g., AdamW.  
4. Unlike LoRA, in this scenario, $\tilde{A}$ and $\tilde{B}$ are not fixed parameters to be minimized. The meaning of momentum here is unclear to me. 
5. The setting is different from LoRA since we have to store the full final model instead of $\Delta W$, which is a low-rank matrix as in LoRA. Indeed, we should treat the proposed algorithm as a memory-efficient optimization method. Then, numerically, it appears that some important baseline methods seem missing for comparisons, e.g., Chain of LoRA, GaLoRA[2], etc. 






[1] Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning, ICML 2024.
[2] GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection, ICML 2024.

### Questions
Please see the weakness above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper tries to allow high-rank updates in LoRA by summing a series of low-rank updates at each step within different subspaces, which is achieved by factoring out a "new LoRA" $\tilde{A}_t \tilde{B}^T_t$ from the LoRA update $A_t B_t^T$ at each step, and merging the remnant into the original weights. 

The paper then points out that optimization on LoRA is equivalent to aligning LoRA update with full fine-tuning's weight update. The paper then try to find the optimal $\tilde{A}_t \tilde{B}^T_t$ such that it best matches full fine-tuning updates, and derives the condition of the minimizer $\tilde{A}_t \tilde{B}^T_t$ need to meet. The minimizer is nevertheless non-unique, and the condition requires calculating the SVD of the gradient. Hence the paper consider some minimizers that are specific transformation of the original $A_t B_t^T$, by scalar or column-wise scaling (e.g. $\tilde{A}_t = A_t diag (\alpha), \tilde{B}_t = B_t diag (\beta)$), and derive the optimal scale $\alpha, \beta$ that can be obtained efficiently. In both cases, the new Adam states can be derived. The actual performance is empirically demonstrated, and the efficiency can be further improved by making the updates intermittently.

### Strengths
A novel approach to optimally expand the rank of LoRA updates to best match full fine-tuning is presented. The theoretical analysis is interesting.

### Weaknesses
The presentation is a bit difficult to follow; the empirical study is limited, and the improvement is small; details below.

### Questions
L139: It is a bit confusing when an $\tilde{A}_t \tilde{B}^T_t$ suddenly appears. It will be much easier to understand if the authors mention that this term is something we are going to in the following section, and point out that $\tilde{A}_t$ and $\tilde{B}_t$ are something we create from $A_t$ and $B_t$ at each iteration t.

L148: What is $\Delta \tilde{W}_t$? It appears to be the update to the merged W at step t, i.e., the "actual" update to the weight matrix. It is better to explicitly point this out for easier understanding.

The basic idea looks similar to ReLoRA but use a different formulation of the "new LoRA" at each step to allow much more efficient optimization. The paper will be easier to follow if the authors first introduce ReLoRA in Sec 3 and explicitly discuss how the paper improves over it.

Is the sign of terms in Eq. (7) correct?

L311-314: If it is for the final storage, why can't we still merge the resultant updates into the original weights?

How are the layers where column-wise scaling doesn't work distributed in a model? Is there any pattern?

How can ScaLoRA combine with other recent LoRA improvements, e.g. PiSSA?

I don't think Scalora's advantage in capacity can be demonstrated on GLUE tasks where capacity is not a problem and full fine-tuning leads to worse results compared to standard LoRA. I would suggest more experiments on tasks where standard LoRA fails to match full fine-tuning; check arXiv:2202.07962 for some examples.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
ScaLoRA is proposed as an efficient fine-tuning method for large language models, aiming to overcome the limitations of LoRA. While LoRA is computationally efficient, its restricted representation power can hurt performance. ScaLoRA addresses this by dynamically scaling low-rank adapters in directions that minimize the loss at each iteration, thereby accumulating high-rank updates over time. It analytically computes optimal scalar or column-wise scaling coefficients while keeping optimizer states intact. Experiments show that ScaLoRA consistently outperforms LoRA and other high-rank variants in both accuracy and convergence speed.

### Strengths
- The motivation and methodology are theoretically clear and well-established.
- Although it does not always guarantee high-rank updates, the use of diagonal scaling over A and B to accumulate diverse directions is novel.
- While the method adds computational overhead, the proposal of ScaLoRA-I provides a practical trade-off between performance and efficiency.

### Weaknesses
- W1. Despite the introduction of ScaLoRA-I as a heuristic workaround, the computational cost of ScaLoRA may hinder scalability to larger models.
- W2. There is a lack of strong baselines, and the performance gains are relatively small. LoRA-GA is an important baseline but is not included. Comparisons to other scaling-based methods like LoRA+ are needed.
- W3. Experimental settings, such as the dimension of W in Section 4.1, should be more clearly stated. For example, the fact that W has a dimension of 64 is only mentioned in the appendix.

---
> [1] Wang, Shaowen, Linxi Yu, and Jian Li. "Lora-ga: Low-rank adaptation with gradient approximation." Advances in Neural Information Processing Systems 37 (2024): 54905-54931.
>
> [2] Hayou, Soufiane, Nikhil Ghosh, and Bin Yu. "LoRA+ efficient low rank adaptation of large models." Proceedings of the 41st International Conference on Machine Learning. 2024.

### Questions
- Q1. Why didn't you follow the $r=2$, $r=8$ experimental setup from AdaLoRA, which is known to yield strong performance? Similarly, for LLaMA, the $r=4$ setting seems too small given the 4096 hidden dimension. What about higher ranks like 8 (PoLAR) or 32 (DoRA)? Please compare for larger rank performance and copy the results from baseline in PoLAR or DoRA and also compare the empirical runtime/GPU usage for ScaLoRA and ScaLoRA-I. Please also clarify the hyperparameter search ranges used.
- Q2. Please provide a comparison of effective rank across methods.
- Q3. As in Table 3, how does performance vary when only scalar scaling is used, either fixed or trainable? Like LoRA+, it's possible that scale tuning of adapters or their gradients contributed to the performance improvements.
- Q4. It appears that the method must be applied at every iteration. Is it possible to apply it only once at initialization?
- Q5. Does ScaLoRA-I also maintain high rank? In the setup of Figure 2, what is the final rank achieved by ScaLoRA?
- Q6. Is the method truly producing high-rank updates? For example, in the DeBERTa model with a 768 hidden size, the rank is only around 54. In contrast, in the original results in HiRA paper, it achieves a rank of 2800 on LLaMA-3-8B, covering about 70% of 4096, while ScaLoRA covers only 7% on DeBERTa. Furthermore, Is high-rank achieved on LLaMA as well?

Overall, the motivation and theoretical framework are clear and well-grounded. While scalability remains a concern, the heuristic variant ScaLoRA-I addresses this to some extent. However, the paper would benefit from stronger baselines, clearer experimental setups, and more careful tuning. If these concerns are addressed, I would be happy to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3
