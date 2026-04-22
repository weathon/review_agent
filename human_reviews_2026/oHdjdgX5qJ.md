# DiHiRA: Diagonal High-Rank Adaption for Large Foundation Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recent advances show that large pre-trained models excel on diverse downstream tasks, driving the popularity of parameter-efficient fine-tuning (PEFT) methods. Among them, Low-Rank Adaptation (LoRA) and its variants approximate weight updates with low-rank matrix products, achieving performance comparable to full fine-tuning while greatly reducing trainable parameters and memory cost. However, low-rank updates inherently constrain model capacity, whereas update weights derived from fully fine-tuning often produces nearly full-rank weight updates. To address this limitation, we propose DiHiRA, a simple yet highly efficient extension of LoRA. By augmenting low-rank updates with a learnable diagonal matrix, DiHiRA enables high-rank adaptation while retaining parameter efficiency. This design improves flexibility and strengthens transferability across diverse tasks. Extensive experiments demonstrate that DiHiRA achieves near full-rank adaptation with LoRA-level efficiency, consistently outperforming baselines on both computer vision and natural language processing benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DiHiRA, a novel parameter-efficient fine-tuning (PEFT) method designed to overcome the inherent low-rank constraint of Low-Rank Adaptation (LoRA). The authors argue that weight updates from full fine-tuning are high-rank, and LoRA's low-rank bottleneck can limit model expressiveness. The core idea of DiHiRA is to augment the standard LoRA update (a product of two low-rank matrices) with a learnable diagonal matrix. The paper presents empirical results across computer vision (VTAB-1K, FGVC) and natural language processing (GLUE, Commonsense Reasoning) benchmarks, demonstrating that DiHiRA consistently outperforms LoRA and other PEFT baselines with a negligible increase in trainable parameters and no additional inference cost.

### Strengths
DiHiRA adopts findings of previous works [1,2,3] on the low rank of LoRA limiting its representation space. 

DiHiRA proposes a simple strategy to increase the rank of DoRA by adding a diagonal matrix to the DoRA factorization.

DiHiRA reports marginal accuracy gains with no notable training speed decrease compared to DoRA. 

The evaluation of DiHiRA is thorough and spans multiple tasks across vision and language.

[1] Albert, Paul, et al. "RandLoRA: Full-rank parameter-efficient fine-tuning of large models." ICLR 2025.

[2] Ji, Yiping, et al. "Efficient learning with sine-activated low-rank matrices."  ICLR 2025.

### Weaknesses
The novelty is limited and DiHiRA can be roughly summarized as DoRA + a diagonal matrix.

Although DiHiRA adds marginal performance gains over DoRA, it also adds a small amount to extra trainable parameters. In this case it is not clear whether performance gains are (1) statistically significant (2) come from the full rank update or the added parameters.

Previous works [1] identify that the rank limitation of LoRA is mostly constraining for larger parameter budgets (LoRA rank 32 and above) yet DiHiRA is only evaluated with very small budgets which probably (as indicated in [1]) suffer more from the parameter constraint than the low rank one.

Although DiHiRA is theoretically full rank, it is not clear whether the effective rank [3] of the weight update is actually large. If the effective rank is still close to LoRA's rank then the diagonal matrix probably only marginally contributes to the improve representation space which would explain the small accuracy gains in practice.

[1] Albert, Paul, et al. "RandLoRA: Full-rank parameter-efficient fine-tuning of large models." ICLR 2025.

[3] Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In European Signal Processing Conference, 2007

### Questions
Have the authors studied DiHiRA for larger ranks (32 and above) ? Have larger accuracy gains over DoRA been reported in this case ?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Diagonal High-Rank Adaptation (DiHiRA), a lightweight modification to LoRA designed to overcome its low-rank bottleneck. The method augments LoRA's low-rank updates ($\Delta W = BA$) with a learnable diagonal matrix $D$, yielding $\Delta W' = BA + D$. This addition is intended to lift the rank of the updates without sacrificing efficiency.

### Strengths
- DiHiRA retains LoRA's mergeability (no inference overhead) and requires negligible extra storage per layer. 
- Evaluation range covers both vision and language domains.

### Weaknesses
- The convergence analysis appears flawed. The mentioned plots do not convincingly demonstrate convergence; in fact, both DiHiRA and LoRA curves appear noisy and far from converging, suggesting neither run truly converged. The claimed faster convergence is unsupported and not convincing. 
- VTAB-1k is a weak benchmark for PEFT tasks. It primarily tests data efficiency, not representational expressiveness. Moreover, the choice of rank 2 is odd; typically, LoRA and its variants are known to underperform in vision at such low ranks, which biases comparisons. 
- The effective rank is ad hoc. The analysis in the paper mixes eigenvalues and singular values interchangeably; for rectangular matrices, the rank must be defined via singular values. The choice of threshold (0.01) is arbitrary and unvalidated. 
- DiHiRA closely parallels recent works such as DoRA, HiRA, MoRA, etc. Yet, the paper does not convincingly argue why adding a diagonal differs materially from these or from identity-skip variants ($\alpha I$ experiment partially covers this). I would also suggest including stronger baselines. 
- Statistical rigor of the improvements is thin. Most experimental results show a minimal gain, and no statistical significance test is given. 
- Adding a diagonal per layer can be non-trivial for large LLMs. The paper does not include memory, activation, and wall-clock overhead analyses. 
- For commonsense reasoning, HiRA uses $10\times$ higher learning rate. Are all methods equally tuned? Such discrepancies can bias results. 
- The ablations seem to be limited. Numerous ablations can be made to probe the mechanism of $D$. It could include: (a) where to place $D$? it can be a selective process; (b) how to initialize $D$? (c) what if you equalize the total parameter budget compared to vanilla LoRA? (d) training stability on all the above points.

### Questions
- Please refer to the above weaknesses. 
- "Diagonal $m\times n$" is non-standard; for $m\neq n$ this means a rectangular matrix with non-zeros on the first $\min(m,n)$ entries. Please clarify this formally in the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DiHiRA (Diagonal High-Rank Adaptation), a simple extension of LoRA for parameter-efficient fine-tuning of large foundation models. The method augments the traditional low-rank update of LoRA with a learnable diagonal matrix, theoretically achieving near full-rank updates while maintaining efficiency. The authors provide a mathematical justification for rank improvement and conduct extensive experiments across vision (VTAB-1K, FGVC) and language benchmarks (GLUE, commonsense reasoning). Empirical results show that DiHiRA consistently outperforms LoRA and comparable PEFT methods with marginal additional parameters.

### Strengths
- The proposed method is conceptually straightforward and clearly described, making it easy to implement and reproduce.
- Extensive experiments across both CV and NLP domains demonstrate consistent, though moderate, performance gains over LoRA.
- The paper provides a simple but valid rank-based rationale explaining why adding a diagonal matrix can improve representational capacity.

### Weaknesses
(1) Limited Novelty: The core idea (adding a diagonal matrix to LoR) is conceptually incremental, offering only a small extension over existing high-rank or hybrid PEFT variants (e.g., HiRA, DoRA).

(2) Lack of Deeper Theoretical Insight: The theoretical section mainly restates rank properties rather than providing rigorous or novel analysis of optimization dynamics or generalization.

(3) Missing Efficiency Discussion: While claiming parameter efficiency, the paper does not deeply analyze training/inference time, memory usage, or deployment implications relative to existing PEFT methods.

### Questions
- Could the authors clarify how the diagonal matrix impacts optimization stability—does it interact with LoRA’s scaling factor or learning rate in non-trivial ways?

- Have the authors compared DiHiRA with adaptive-rank or structured variants (e.g., AdaLoRA[1], PISSA[2]) under the same parameter budget?

- Would the proposed diagonal augmentation still offer benefits when integrated with higher-rank LoRA (e.g., r ≥ 32) or larger-scale LLMs beyond 8B parameters?


**References:**

[1] Zhang, Qingru, et al. "ADAPTIVE BUDGET ALLOCATION FOR PARAMETER-EFFICIENT FINE-TUNING." In 11th International Conference on Learning Representations, ICLR 2023. 

[2] Meng, Fanxu, et al. "Pissa: Principal singular values and singular vectors adaptation of large language models." In Advances in Neural Information Processing Systems 37 (2024): 121038-121072.

### Soundness
2

### Presentation
3

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
This paper operates in the field of Parameter-Efficient Fine-Tuning (PEFT) for large foundation models. It addresses the well-known capacity limitations of Low-Rank Adaptation (LoRA), proposing that increasing the rank of the update matrix is beneficial for adaptation performance. To achieve this, the authors introduce DiHiRA, a simple extension of LoRA that augments the standard low-rank update (BA) with a learnable diagonal matrix (D). The method is evaluated on a comprehensive set of computer vision (VTAB, FGVC) and natural language (GLUE, Commonsense Reasoning) tasks. The experimental results demonstrate that DiHiRA provides notable performance gains over standard LoRA, particularly at very low-rank settings (e.g., r=2, r=4).

### Strengths
1. The paper tackles a relevant and important problem. Improving the expressiveness and capacity of PEFT methods while maintaining high parameter efficiency is a critical area of research for the practical deployment of foundation models.

2. The paper is well-written, clear, and easy to follow. The proposed method (DiHiRA) is simple to understand, and its implementation appears straightforward.

3. The experimental design is a significant strength. The authors have validated their method across both vision and language domains using multiple standard benchmarks, which provides a solid empirical grounding for their claims.

### Weaknesses
1. The primary weakness is the novelty of the core idea. The concept of moving beyond LoRA's low-rank constraint to explore high-rank or full-rank updates is not new. Several existing works have already investigated this, such as methods that merge multiple LoRA modules or re-parameterize the update in other ways [1]. Furthermore, the proposed mechanism—adding a diagonal matrix—only introduces $min(m, n)$ new parameters. While this does technically increase the rank, it is questionable how much additional representational capacity this relatively small number of parameters can truly provide for complex adaptations.

2. The paper positions itself as a high-rank adaptation method, yet its experimental evaluation is missing critical comparisons to other prominent methods in this specific space. Baselines such as MiLoRA [1] and PiSSA [2], which also aim to improve LoRA's capacity, are not compared against. This makes it difficult to contextualize the performance of DiHiRA and understand its advantages, if any, over other high-rank PEFT approaches.

3. The practical utility of the method seems questionable. The paper's own results (e.g., in Table 3 and Figure 6) clearly show that DiHiRA's performance gains are most significant at extremely low ranks ($r=2, r=4$). These settings are rarely used in practical, real-world applications, where ranks of $r=16, r=32$, or higher are common. As the rank increases to these more practical levels, the performance gap between DiHiRA and standard LoRA becomes marginal or even vanishes. This suggests the method does not offer a compelling advantage for typical use cases. Moreover, its parameter efficiency is on par with LoRA, which is less efficient than other recent PEFT families (e.g., ReFT []) that offer different performance-per-parameter trade-offs.

[1] MiLoRA: A Multi-LoRA Framework for Federated Finetuning of Large Language Models. 2024.

[2] PiSSA: Principal Singular values and singular vectors Adaptation of Large Language Models. 2023.

[3] ReFT: Representation Finetuning for Language Models. 2024.

### Questions
Please see my questions above.

### Soundness
2

### Presentation
2

### Contribution
2
