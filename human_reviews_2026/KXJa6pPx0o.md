# Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Low-rank adapters have become standard for efficiently fine-tuning large language models, but they often fall short of achieving the performance of full fine-tuning. We propose a method, **LoRA** **S**ilver **B**ullet or **LoRA-SB**, that approximates full fine-tuning within low-rank subspaces using a carefully designed initialization strategy. We theoretically demonstrate that the architecture of LoRA-XS, which inserts a learnable $r \times r$ matrix between $B$ and $A$ while keeping other matrices fixed, provides the precise conditions needed for this approximation. We leverage its constrained update space to achieve optimal scaling for high-rank gradient updates while removing the need for hyperparameter tuning. We prove that our initialization offers an optimal low-rank approximation of the initial gradient and preserves update directions throughout training. Concretely, LoRA-SB combines this initialization with a constrained low-rank adaptation mechanism, forming a co-designed system where both the update subspace and its optimization dynamics are jointly aligned with full fine-tuning. Extensive experiments across mathematical reasoning, commonsense reasoning, and language understanding tasks demonstrate that our approach exceeds the performance of LoRA (and baselines) while using **27-90** times fewer learnable parameters, and comprehensively outperforms LoRA-XS. Our findings establish that it is possible to simulate full fine-tuning in low-rank subspaces, and achieve significant parameter efficiency gains without sacrificing performance. Anonymous code is available at: https://anonymous.4open.science/r/lora-sb-anonymous-5BEE.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes LoRA-SB, a variant of LoRA-XS that relies on carefully crafted initialization to approximate full fine-tuning. The constrained update space is leveraged for optimal scaling factor, thus avoiding the need for tuning this hyperparameter. Numerical tests are conducted on MetaMathQA (evaluated on GSM8K and MATH), commonsense reasonsing, and GLUE benchmarks with LLMs scaling up to 9B parameters.

### Strengths
1. The limitations of LoRA-XS is clearly illustrated, providing sufficient motivation for LoRA-SB. 
2. Theoretical analysis guarantees the design and correctness of LoRA-SB. 
3. The empirical results demonstrate satisfactory performance compared to other initialization approaches.

### Weaknesses
1. The theoretical contributions in the paper lack novelty and depth. Lemma 1 follows directly from the definition of $\Delta W$, and Lemma 2 is an immediate consequence of the chain rule. These could be better presented as plain text rather than formal lemmas. Furthermore, Theorems 3 and 4 are straightforward extensions and simplifications of results from LoRA-Pro, while Theorem 6 is directly adapted from LoRA-GA. A more thorough discussion and comparison with existing theoretical works are needed. 
2. As a result of weakness 1, LoRA-SB can be viewed as a direct extension of LoRA-Pro and LoRA-GA to the setups of LoRA-XS, which diminishes the novelty and overall contribution of the work. 
3. The scope of the paper is rather limited. The analysis and proposed approach apply exclusively to LoRA-XS. It would strengthen the work to explore whether some of the results can be extended to the general LoRA framework or other PEFT methods. 
4. Several closely related approaches, such as LoRA-GA, are not even included in the numerical experiments.

### Questions
See questions in weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes LoRA-SB, a new initialization method built upon LoRA-XS for PEFT of LLM. It initializes with an approximation of the first full fine-tuning gradient, removing the need for scaling-factor tuning and improving optimization stability. Extensive experiments show that LoRA-SB matches or surpasses other baselines across multiple tasks and models.

### Strengths
1.This paper provides a clear theoretical analysis of why LoRA-XS suffer from limited update capacity and scaling sensitivity, offering strong motivation for the proposed approach.

2.Proposes a new initialization strategy that aligns the low-rank update space with the first full fine-tuning gradient, leading to better optimization stability, scaling independence, and improved convergence.

3.Extensive experiments across diverse models and tasks show that LoRA-SB delivers stable and significant performance gains over LoRA-XS, reinforcing the credibility and robustness of the approach.

### Weaknesses
1.The paper does not evaluate LoRA-SB against dynamic or adaptive rank approaches (e.g., AdaLoRA, Sparse LoRA). Because LoRA-SB uses a static, fixed-rank initialization, it is inherently incompatible with these flexible rank-allocation strategies. This omission leaves an important gap in the analysis, making it unclear how LoRA-SB compares to adaptive methods under similar parameter budgets.

2.LoRA-SB requires estimating the first full fine-tuning gradient on a subset of data, and the paper sets this sample size empirically. This per-dataset calibration may limit practicality and reproducibility.

3.Although the paper provides comprehensive evaluation on text-based reasoning and understanding tasks, its validation is limited to the NLP domain. In contrast, related methods such as LoRA, LoRA-Pro, and DoRA have already demonstrated effectiveness across vision and multimodal tasks. The absence of experiments of other modalities weakens the generality of LoRA-SB.

4.The paper does not evaluate how LoRA-SB performs when the downstream task or data distribution diverges substantially from the pre-training distribution. In such cases, the initial gradient may be misleading, potentially locking the model into a suboptimal low-rank subspace and limiting its ability to adapt effectively.

### Questions
Please see weaknesses.

### Soundness
3

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
3

### Summary
The paper presents LoRA-SB (LoRA Silver Bullet), a novel method for parameter-efficient fine-tuning (PEFT) of large language models that aims to match the performance of full fine-tuning while drastically reducing the number of learnable parameters. The core innovation is an initialization strategy derived from approximating the first gradient update step of full fine-tuning, which is integrated into the low-rank architecture of LoRA-XS, where only the central $r \times r$ matrix is trainable. The authors provide theoretical proofs demonstrating that this approach ensures optimal gradient approximation and eliminates the need for scaling factor tuning, addressing key limitations of prior low-rank methods like standard LoRA and LoRA-XS. Extensive experiments across various reasoning and language understanding tasks show that LoRA-SB significantly outperforms baselines, using up to 90 times fewer parameters than LoRA.

### Strengths
- LoRA-SB achieves high performance while maintaining a severely constrained parameter count, using 27x to 90x fewer trainable parameters compared to LoRA and its variants across various benchmarks
- The method is grounded in theoretical proofs, demonstrating that the SVD-based initialization provides the optimal low-rank approximation of the initial gradient.
- The derived optimal gradient updates, combined with the orthonormal fixed bases B and A, inherently lead to scaling factor independence.

### Weaknesses
- Ablation studies show that the initialization is sensitive to noise levels when performing truncated SVD on the averaged update. This suggests the quality of the initial gradient estimation is critical and any practical implementation must ensure robust gradient computation.
- Although the initialization is optimal for the first step, the fixed nature of A and B inherently restricts the expressivity of the low-rank updates. While the paper claims updates are preserved throughout training, the overall performance gap between LoRA-SB and Full FT remains, suggesting the fixed subspace may limit capacity to learn evolving task-specific nuances over long training runs.
- While AdamW and SGD are mentioned, deeper analysis of optimizer–initialization interactions or potential instability with momentum optimizers is missing.

### Questions
- This paper shows that the initialization provides the best low-rank approximation of the first full FT update direction. The paper claims this preserves update directions throughout training. Can the authors provide more intuition or theoretical backing for why the optimal low-rank subspace identified by the first step remains near-optimal throughout the entire fine-tuning process, especially for tasks requiring extensive adaptation?
- Table 7 compares LoRA at rank $r=8$ (RoBERTa) or $r=32$ (LLMs) against LoRA-SB at higher ranks ($r=16/24$ or $r=64/96$), where LoRA-SB meets or exceeds LoRA performance. While LoRA-SB is significantly more parameter-efficient, it requires higher rank during inference, slightly increasing FLOPs.
- Low‑Rank Adaptation with Gradient Approximation (LoRA-GA) also reports convergence speed and performance improvements. Could the authors clarify whether they considered this method, and if so why it was not included in the empirical comparisons?

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
In this paper, the author proposes a new efficient LoRA method, LoRA-SB, which closes the performance gap between efficient low-rank adaptation and full fine-tuning of large language models. The method uses a novel initialization strategy with the LoRA-XS architecture to optimally approximate full fine-tuning within low-rank subspaces. Experiments show LoRA-SB significantly outperforms standard LoRA across reasoning and language tasks while using 27-90 times fewer parameters, achieving full fine-tuning performance with substantial efficiency gains.

### Strengths
-	The paper is well written, and the formulas are pretty and clear.
-	Extensive experiments over 16 datasets and 4 models verified the effectiveness of LoRA-SB

### Weaknesses
-	My primary concern relates to the novelty of LoRA-SB. The overall framework of LoRA-SB builds upon LoRA-XS; however, LoRA-XS is not a general efficient LoRA framework, which may limit the generalizability of the proposed method. Additionally, the gradient-based initialization approach and the analysis of equivalent gradients bear strong similarities to LoRA-GA and LoRA-Pro, which further diminishes the originality of the contribution.
-	The analysis in lines 266-279 appears somewhat redundant, as these conclusions have already been established through the Eckart-Young theory in equations (7-8).
-	While the scaling factor s is a hyperparameter, it seems to have limited importance since alpha=2r is typically used as a standard setting rather than a parameter requiring tuning. Emphasizing that the method is "scaling factor independent" may not constitute a significant advantage.
-	The experiments in Table 1 are conducted under the condition of rank=32, whereas previous methods (such as LoRA-GA) typically employ rank=8. Would it be possible to provide supplementary results under these more commonly used settings?
-	Previous methods have typically been evaluated on the metamath-100k dataset, while the current experiments only utilize metamath-50k. Could the evaluation be extended to larger datasets to provide more comprehensive results?
-	The comparison lacks LoRA-GA, which employs the same initialization strategy as LoRA-SB within the LoRA framework. I would kindly suggest including relevant experimental comparisons with this method.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2
