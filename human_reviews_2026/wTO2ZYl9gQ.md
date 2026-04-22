# AdaMeZO: Adam-Styled Zeroth-Order Optimizer for LLM Fine-tuning Without Memorizing the Moments

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Fine-tuning LLMs is necessary for dedicated downstream uses, but classic backpropagation approaches necessitate a large amount of GPU memory. To this end, a recent work, MeZO, which relies solely on forward passes to fine-tune LLMs, significantly reduces GPU requirements at the cost of slower convergence due to its indifference to loss landscapes. Standard solutions like Adam explore loss landscapes by estimating the first and second-order moments and keeping them in memory to guide the models in moving faster through dimensions with smaller curvature and vice versa. However, directly applying Adam negates MeZO's advantage as it will triple the memory requirement. In light of this, we propose AdaMeZO, a zeroth-order optimizer enhanced by Adam-styled first and second moments estimates, but without keeping them in memory. We present a theoretical analysis of AdaMeZO, corroborated by extensive experiments demonstrating AdaMeZO's performance, showing that AdaMeZO can outperform MeZO while taking up to $70\%$ fewer forward passes. Visualizations of trajectories on toy functions to affirm AdaMeZO's ability to adapt to different loss landscapes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents AdaMeZO, an optimization algorithm which combines the benefits of Adam (good choice of weight updates) and MeZO (foregoing backpropagation using a gradient estimator to save memory). Compared to Adam, AdaMeZO uses much less memory since it doesn't store accumulators for the momentum and variance, and compared to MeZO, AdaMeZO applies momentum and variance normalization which improves the quality of the step taken. AdaMeZO uses a clever gradient/EMA estimator that is structured so that it can be discarded and regenerated very quickly whenever needed without taking much memory or time, and this estimator serves as the basis for reducing the memory requirements for training. Experiments show good performance on fine tuning LLMs at the >=billion parameter scale, with improvement over MeZO.

### Strengths
- A clever gradient regeneration trick is used to compute EMAs without storing the EMAs themselves.
- Block-wise computation of the Adam update using regenerated gradients from PRNG state caching allows for negligible additional memory cost during the step update computation.
- Experiments show that this method is capable of fine tuning models at the billion parameter scale.

### Weaknesses
I would expect that the SPSA gradient estimator to introduce a huge amount of noise into the gradient signal, in such a high dimensional problem. This leaves the good experimental performance unexplained, as I would not ordinarily expect learning to happen at a reasonable rate when the gradients have that much noise in them. There is no explanation offered as to why the noise doesn't slow the learning down to an unreasonable rate. I would increase my score for a good explanation as to how this happens.
- The convergence rate theorem (Theorem 4.1) is not sufficiently described; some variables are not specified. Particularly important is $\sigma^2$, which is the variance of the gradient due to batch stochasticity, serves as a multiplicative constant in the big-O bound on the number of steps needed. Ordinarily I would find this bound to be convincing, but I suspect the variance of the gradient estimates would be massively inflated when we replace backpropagation by SPSA estimation, especially in high dimensional problems. If you can prove that $\sigma^2$ grows sub-linearly with problem dimension with fixed expected gradient magnitude when using SPSA estimation of the gradients, this would offer me some explanation for the above.

### Questions
- The expected variance $v_t$ collected from backpropagation style gradients is not going to be the same as the expected $v_t$ as calculated from the formula on line 251, because the inherent noise introduced by SPSA gradient estimation add to the variance estimate, making it larger. Empirically and/or theoretically, what is the relative scale of this additional variance versus the original $v_t$ that we would have gotten from backpropagation gradients? Hopefully it is a small effect.
- Is there any other explanation you can give on how a zero-order optimizer scales efficiently to the billion parameter regime despite the noise? Anything I am not thinking of or misunderstood. Maybe the loss is relatively insensitive to noise in most of the extraneous directions? Maybe somehow the gradient estimator has a low-rank covariance matrix in practical settings?
- A popular alternative for decreasing memory cost in training is to split the batch into sub-batches and perform gradient accumulation. Do we know how AdaMeZO compares for memory cost and wall time consumed during experiment?
- Please include Adam in Tables 10 and 11, since that's what you're hoping to improve over.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces AdaMeZO, a zeroth-order optimizer for LLM fine-tuning that incorporates Adam-style first and second moment estimation without explicitly storing the moment vectors. The method leverages truncated reconstruction and random state caching to approximate momentum terms efficiently, preserving the low memory footprint of MeZO while improving convergence speed and adaptability.

### Strengths
- Addresses a practical and important challenge: efficient fine-tuning of LLMs with limited memory resources.

- The proposed truncation and reconstruction strategy for Adam-style moments is novel and allows moment estimation without prohibitive memory overhead.

- Provides extensive theoretical analysis supporting the convergence properties of the method.

### Weaknesses
- The experiments mainly compare with MeZO baseline. It would strengthen the claims to include other baselines to validate the generalization ability of the proposed method.

- The writing feels rushed. Important algorithm, experiment reasults on larger LLMs(13B),  ablation studies on hyperparameters are only in the appendix; maybe these should be brought into the main text for clarity and accessibility.

- The paper currently lacks visualizations of the optimization process, similar to Figure 1. It would be very useful to include loss curves for larger-scale models (e.g., 13B) to demonstrate training dynamics more clearly.

### Questions
Seen in weakness

### Soundness
2

### Presentation
1

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
This paper introduces **AdaMeZO**, a new member of the Ada-style optimization family. It extends the recently popular zeroth-order optimizer MeZO, focusing on parameter-efficient fine-tuning*of large language models. AdaMeZO proposes a *truncated estimation* of the first- and second-order moments based on sampled gradients, which reduces memory usage by eliminating two full copies of parameter-sized states. Additionally, the paper introduces a *state caching* mechanism to avoid storing the entire second-order momentum before applying the Adam-like update. The authors provide theoretical analysis supporting convergence and empirical results showing that AdaMeZO outperforms MeZO baselines on language model fine-tuning benchmarks.

### Strengths
1. The main innovation lies in the *truncated-sum estimation* of the moving average momentum, which is both clever and practical for reducing memory usage.
2. The method achieves *strong fine-tuning performance* compared to MeZO, demonstrating clear empirical advantages.  
3. Although unrolling the truncated momentum requires costly $h$ times more gradient samples, the authors mitigate this concern by comparing against an alternate baseline (MeZO-switch), which provides a fairer evaluation.

### Weaknesses
1. The review of related work across all three relevant topics (zeroth-order optimization, memory-efficient fine-tuning, and adaptive methods) is limited and misses many recent advances. Important works such as [1][2][3][4][5] should be included to provide sufficient context.  
2. Despite the introduction of truncated moment estimation, it remains unclear why the algorithm still needs to compute the full $v_t$ before the update. The paper does not clearly discuss this design choice, and Figure 3 does not effectively illustrate the reason of this challenge.  
3. Building on the previous point, the description in Section 3.1.1 is unnecessarily complicated. The key idea seems straightforward: by using different random seeds for each block, one can estimate both $m$ and $v$ without storing an additional copy of state. However, Sections 3.3.1 and 3.3.2 obscure this simplicity with excessive procedural detail.
4. The assumptions underlying Theorem 4.1 are not stated in the main text, making it difficult to interpret the theoretical result. The definitions of $L$, $\sigma$, and $\mu$ should be briefly introduced near the theorem statement for clarity.  
5. Similarly, in Appendix E, several notations are undefined. For example, $\Sigma$ is used without being properly introduced.  
6. Assumption E4 is particularly strong and makes the theoretical setting overly simplified. The authors should at least provide minimal justification or discuss its reasonableness.  
7. In Lemma E7 (line 1000), statements labeled as “E2 to E4” should be marked as *assumptions* rather than *theorems*, likely a typo.  
8. The MeZO baseline used for comparison is weak by current standards in zeroth-order optimization. According to Appendix D.5, AdaMeZO struggles to match to baselines that maintain full second-order momentum, which diminishes the claimed advantage.

## Reference

[1] Zhang, Yihua, et al. *Revisiting zeroth-order optimization for memory-efficient LLM fine-tuning: A benchmark.* arXiv:2402.11592 (2024).  
[2] Chen, Aochuan, et al. *DeepZero: Scaling up zeroth-order optimization for deep model training.* arXiv:2310.02025 (2023).  
[3] Pethick, Thomas, et al. *Training deep learning models with norm-constrained LMOs.* arXiv:2502.07529 (2025).  
[4] Defazio, Aaron, et al. *The road less scheduled.* *Advances in Neural Information Processing Systems*, 37 (2024): 9974–10007.  
[5] Kunstner, Frederik, et al. *Noise is not the main factor behind the gap between SGD and Adam on transformers, but sign descent might be.* arXiv:2304.13960 (2023).

### Questions
1. In Section 5.2, you describe “MeZO-switch—a variant of MeZO where the learning rate is manually adjusted to ensure that its optimization trajectory is longer than that of AdaMeZO.” Could you elaborate on why the *trajectory length* is meaningful in such a very noisy stochastic optimization setting, and how it supports the claim of adaptability rather than underfitting?  
2. The computational cost scales linearly with $h$. How does the choice of $h$ influence final performance? Is there an ablation or sensitivity analysis exploring this relationship?

---

I am willing to raise my score if the authors can address the weaknesses and questions above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AdaMeZO, a memory-efficient zeroth-order optimizer that introduces Adam-style first and second moment estimation without explicitly storing them in memory. The method combines truncated moment approximation and PRNG state caching to mimic Adam-like adaptive preconditioning while retaining MeZO’s forward-only property. The authors present theoretical convergence analysis under a non-convex assumption and empirical evaluations on toy functions and several large language models (RoBERTa-large, OPT-1.3B, and LLaMA-3B).

### Strengths
1.This paper proposes AdaMeZO, which achieves training performance comparable to or surpassing MeZO by significantly reducing the number of forward propagations.

2.This paper provides a very detailed discussion of the methodology in Part Three.

### Weaknesses
1.The experimental section compares AdaMeZO mainly against MeZO and MeZO-switch. However, several concurrent adaptive zeroth-order optimizers (e.g., HiZOO, Helene, ZO-AdaMU) are only discussed but not empirically compared, even though they are conceptually closest. This weakens the empirical validation and makes it hard to quantify the claimed “70% fewer forward passes” advantage.

2.The experiments stop at RoBERTa-large (350M), OPT-1.3B, and LLaMA-3B, with an appendix extension to 7B and 13B models. While informative, results on larger modern-scale LLMs (≥30B) would be crucial to substantiate the claimed scalability and real-world applicability.

3.The proof in Part Four is too brief due to the excessive discussion in Part Three.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
