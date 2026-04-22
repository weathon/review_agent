# Balanced Low-Rank Adaptation: Removing Invariance for Fast and Stable Fine-Tuning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Low-Rank Adaptation (LoRA) is the most widely used method for fine-tuning large language models.
LoRA is overparameterized: multiple pairs of low-rank factors correspond to the same adapted weight matrix.
We observe both theoretically and numerically that these pairs can have significantly different condition numbers: converging to different minimizers of the loss affects the convergence rate of LoRA.
Building on this remark, we introduce Balanced Low-Rank Adaptation (BaLoRA), a variant of LoRA that projects iterates onto a balanced manifold where the conditioning of the loss is improved while keeping the same adapted matrix. 
This projection step is computationally inexpensive and integrates seamlessly with existing fine-tuning pipelines. 
Empirically, BaLoRA converges faster than standard LoRA and exhibits greater robustness to hyperparameter choices across a range of fine-tuning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes BaLoRA, a method that enforces a balance constraint between LoRA’s low-rank factors by projecting them onto a hyperbalanced manifold after each optimizer step. This aims to improve conditioning and accelerate convergence. The authors provide theoretical analysis for linear layer and empirical results on GPT-2 and LLaMA-3B, showing moderate improvements.

### Strengths
- This paper provides a novel perspective on the traing dynamics of LoRA from the condition number of minimizers.
- The mathematical derivations are clearly presented and easy to follow.

### Weaknesses
- W1. The toy example is overly simplified and cannot adequately capture how conditioning affects LoRA’s convergence. A single-layer linear network only reflects matrix factorization behavior and fails to model the impact of activation functions or deeper architectures.

 - W2. In Section 3.2, the authors state that BaLoRA with Adam is a heuristic and restricts theoretical analysis to the gradient descent variant. However, the paper still presents BaLoRA (with Adam) as the main method—even in the abstract—thus weakening the theoretical soundness of the overall claims.

 - W3. The empirical improvements in test loss are marginal compared to vanilla LoRA. The paper would benefit from evaluating on more diverse benchmarks (e.g., math or code tasks) and reporting accuracy as well as standard deviation over multiple runs.

### Questions
- The authors make a strong claim in Line 124 that “previous studies in optimization of LoRA do not address fine-tuning of machine learning models.” Could the authors clarify or justify this statement?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Balanced Low-Rank Adaptation (BaLoRA), a variant of LoRA designed to improve conditioning and convergence during fine-tuning of large language models. The core idea is to project LoRA adapters onto a balanced manifold to ensure the optimization dynamics converge toward better-conditioned minima without changing the effective update. The authors provide theoretical analysis of LoRA’s conditioning, prove that balanced minimizers yield optimal condition numbers, and derive efficient projection algorithms. Empirical results on demonstrate faster convergence and the robustness to hyperparameter choices.

### Strengths
Overall, I think this paper provides a novel geometric insight into LoRA conditioning. This contribution is quite significant, as the paper effectively bridges theoretical motivation with simple yet elegant algorithmic design. The empirical results align well with the theoretical findings and further demonstrate the robustness of BaLoRA.

### Weaknesses
* I think the theoretical scope is a little bit limited. The analysis is mostly confined to one-layer linear networks and has limited applicability to transformer layers. Moreover, although the paper aims to improve the convergence, the convergence analysis under Adam and SGD is not fully addressed, especially for those practical nonconvex cases. I think this is not a major drawback of this paper. 

* The experimental settings and results are somewhat limited. For example, the authors could compare with the scale AdamW in paper [1], OLoRA [2], and [3]. Moreover, only the loss is reported; the missing downstream task accuracy should be very important in studying LoRA-based methods.

[1] Zhang, Fangzhao, and Mert Pilanci. "Riemannian preconditioned lora for fine-tuning foundation models." arXiv preprint arXiv:2402.02347 (2024). 

[2] Kerim Buyukakyuz. Olora: Orthonormal low-rank adaptation of large language models. arXiv
preprint arXiv:2406.01775, 2024.

[3] Juneyoung Park, Minjae Kang, Seongbae Lee, Haegang Lee, Seongwan Kim, and Jaeho Lee. Riemannian optimization for lora on the stiefel manifold. arXiv preprint arXiv:2508.17901, 2025.

* Does the balanced projection happen every step? Did the author also consider the less frequent balanced projection case? Moreover, how do the authors choose the learning rates? By simple grid search or some other methods? I am asking this since 1) From Figure 3, I think either training loss or test loss is not that sensitive to the detailed learning rate with a similar scalar. 2) The learning rate adopted in some experiments is like 3.8e-4, which is less common in other LoRA papers. So naturally, my next question is, can you also provide the learning rate sensitivity in downstream tasks?

### Questions
See detailed questions in the weaknesses part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes BaLoRA, a more effective approach to fine-tuning large language models.
LoRA is a popular method that adds small, trainable matrices to a large model; however, it can be unstable to train. BaLoRA addresses this by maintaining the balance of those small matrices after every training step.
This balancing makes training faster, more stable.
In tests on models like GPT-2 and Llama-3, BaLoRA trained more quickly and achieved lower loss than LoRA.

### Strengths
- This paper attempts to mathematically manipulate LoRA to achieve better convergence.
- It demonstrates that using balanced LoRA could improve convergence.
- The theoretical analysis for convergence is quite interesting.
- The explanations and writing are clear and easy to follow.

### Weaknesses
- There is a lack of novelty in this work; many studies have already explored SVD LoRA (TriLoRA, OPLoRA, ...), and I don't see significant improvements or differences here beyond the theoretical aspect.
- Why the only metric used is loss? I think that loss doesn't provide much information about the performance of LLMs. There are several better metrics available to evaluate performance (MTbanch, vicuna, ...).
- Why is the only baseline used here appears to be LoRA? Since there are many variations of LoRA, including those related to SVD (TriLoRA, OPLoRA, ...), it would have been beneficial to compare the results against these other baselines, not just LoRA.

### Questions
- Why is LoRA the only baseline method used? It would be helpful to include additional baseline methods.  
- If you could clarify how this work is novel compared to other SVD-LoRA methods, I would be willing to reconsider my score.  
- The performance improvement would be more evident if better evaluation metrics (such as MTbench or Vincuna) were used, rather than just loss values.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes BaLoRA (Balanced Low-Rank Adaptation), which projects the low-rank adapter onto a super-balanced manifold after each optimization step. This improves the condition number while preserving the adaptation matrix AB, with negligible computational overhead. Validated on text prediction tasks for GPT-2 and Llama-3.2-3B models, BaLoRA converges faster and demonstrates greater robustness to hyperparameters.

### Strengths
* S1: The theoretical analysis in Section 2 provides a clear derivation of the Hessian and condition numbers for LoRA in the matrix factorization and general cases, offering some insight into why balanced minimizers might lead to faster asymptotic convergence.
* S2: BaLoRA's projection step is computationally lightweight, making it easy to integrate into existing LoRA pipelines without significant overhead.
* S3: The empirical evaluation shows robustness to initialization scaling and learning rates, with BaLoRA outperforming LoRA in high-scaling regimes on both synthetic and real LLM fine-tuning tasks.

### Weaknesses
* W1: The theoretical contributions are limited to toy one/two-layer linear networks, which do not capture the complexities of deep transformers or non-linear activations, rendering the "optimal conditioning" claims speculative for practical PEFT scenarios.
* W2: Experiments are narrow: only two models (GPT-2, Llama-3.2-3B) on a single dataset, with no comparisons to recent LoRA variants like DoRA,  AdaLoRA, or OLoRA, and lacking downstream task evaluations to assess generalization.
* W3: The paper overlooks key related works on balanced optimization and Riemannian methods for low-rank manifolds, leading to overstated novelty.
* W4: The conclusions of condition number analysis for linear networks have not been specifically validated in nonlinear Transformers. The experiments did not quantify the specific contribution of condition number optimization to convergence acceleration, and the correlation between theoretical value and practical effectiveness lacks empirical support.

### Questions
* Q1: Insufficient Empirical Validation: The experiments are limited to toy settings (one-layer linear networks) and only two models (GPT-2 and Llama-3.2-3B) on a single dataset (Wikitext), with no comparisons to state-of-the-art PEFT variants like DoRA, AdaLoRA, or O-LoRA. In the absence of evaluations on diverse downstream tasks or larger models, the claims of faster convergence and robustness remain unconvincing.

* Q2: BaLoRA is essentially a minor modification of standard LoRA, relying heavily on existing matrix factorization techniques and balancing concepts from prior works. The projection onto a balanced manifold adds little new insight; thus, the paper's contributions are overstated.

### Soundness
2

### Presentation
2

### Contribution
2
