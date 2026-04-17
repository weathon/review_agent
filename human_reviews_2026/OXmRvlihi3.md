# LoRA-FA: Efficient and Effective Low Rank Representation Fine-tuning

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Fine-tuning large language models (LLMs) is crucial for improving their performance on downstream tasks, but full-parameter fine-tuning (Full-FT) is computationally expensive and memory-intensive. Parameter-efficient fine-tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), address this by optimizing only a small subset of parameters. However, LoRA may underperform Full-FT in certain scenarios due to the intrinsic limitations of its low-rank gradients.
In this work, we reveal an asymmetric, collapsible structure in LoRA’s update: the low-rank modification to $W$ can be reformulated as a single-layer linear regression, implying that one of the LoRA factors can be frozen without sacrificing expressivity. Leveraging this insight, we introduce LoRA-FA, which freezes the projection-down matrix $A$ and trains only the projection-up matrix $B$. We further close the gap to Full-FT by deriving closed-form gradient corrections that minimize the discrepancy between the induced low-rank gradient and the full gradient.
Through extensive experiments on diverse benchmarks, including GLUE, GSM8K, MT-Bench, and HumanEval, we demonstrate that LoRA-FA consistently achieves comparable performance to existing PEFT methods and Full-FT. Experiments on system efficiency show that LoRA-FA significantly reduces activation memory consumption and computational workload in fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents an improvement over LoRA, in which only the B matrix is trained, while the A matrix is frozen, and also a fix is applied to the gradients of layer B to mimic the gradients as if both AB layers are trained. Freezing the A matrix eliminates the need to store the input activations to layer A, thereby reducing memory footprint.

### Strengths
* A decent number of LoRA variations were compared
* Reduction in activation memory footprint compared to LoRA
* Method is simple and clear
* Theoretical claims are interesting
* Gradient fix is novel and interesting

### Weaknesses
* In the abstract and in the body authors say: “However, LoRA's performance often lags behind Full-FT due to limitations of the low-rank gradient”. This is not always true. In fact, see a quote from the original LoRA paper: “As shown in Table 4, LoRA matches or exceeds the fine-tuning baseline on all three datasets.” . 


* Line 52: “However, these studies generally overlook the fact that A and B contribute
asymmetrically”. Not always true, for example see the PRILoRA paper, which prunes only the A matrix, due to the same reason, that the A matrix ‘matters less’. You may want to address this paper.

* Figure 1. (a) in Full FT maybe the drawing of matrices A,B should be removed? Unless you define full FT as such that includes them frozen.

* THM 3.1 says that A*B* \approx A_0 B’*, but the proof doesn’t prove that. The proof shows that in the general case (when they don’t span the same subspace) the Frobenius norm of the difference is r(m-r), which is not zero, nor necessarily small. I may have missed something?

* Line 188, maybe the authors meant dL?

* Table 2 and the other tables: better to have the same amount of decimal places even for round numbers, to keep numbers aligned: 0. -> 0.0

* Table 2: Results are very different from the ones in the original LoRA paper. E.g. In the RTE column (and others) FT got  77.1, LoRA 74.8 (meaning LoRA inferior), while in the LoRA paper FT got 78.7 and LoRA 86.6. (LoRA superior). This is quite a difference. I assume it is because in the LoRA paper they had unique hyperparams for each benchmark, while the authors set a fixed set of hyperparameters for all benchmarks, which are only optimized for the proposed method, and not for the LoRA baseline. Not sure it is a fair comparison. Two things authors can do: (1) For each benchmark from the GLUE, compare the best LoRA-FA hyper-swept model against best LoRA hyper-swept model (2) Compete against LoRA using LoRA best hyperparameters for each benchmark. If authors succeed in at least one of the two, that would be far more convincing, as these benchmarks are very sensitive to hyperparameters.

* Even on authors table 3, on MT-Bench, LoRA gets 5.6 while full-FT gets 5.3, so how can authors claim that full-FT is always better that LoRA?

* In all tables, a column with a number of trainable parameters should be added. Table 3 has rows with different low ranks, which does not make fair comparison. 

* Line 668 says A*=A_0 C, so why (9) has \approx  and full equality?

* Line 700, I understand that A_0 is N(0,1), but why A*? If this assumption is for the sake of ease of analysis, maybe it should be addressed. In line 753 authors proved the expectancy of the Frobenius norm, but the theorem said nothing about the Frobenium norm… So the reader is left in suspense, seeking to understand what the answer means.


* A very interesting ablation test is missing: What would happen if A is frozen, but without the gradient fix. I am interested to know how crucial it is.

### Questions
See the weaknesses section

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a parameter-efficient fine-tuning algorithm called LoRA-FA, which freezes the A matrix in LoRA and only trains matrix B. To better mimic LoRA's update, the authors provide the optimal closed-form estimation of B's update direction via Theorem 4.1. The authors also conduct experiments to validate LoRA-FA's performance, where LoRA-FA can outperform vanilla LoRA and several variants.

### Strengths
1. I believe the major strengths of this work should be establishing the theoretical relationship between LoRA and algorithms like LoRA-FA which only trains one LoRA component at a time, which corresponds to Theorems 3.1 and 4.1.
2. The proposed algorithm has shown noticeable advantage compared to vanilla LoRA in the experiments.

### Weaknesses
1. I may consider the core mechanism of LoRA-FA, i.e., only training one LoRA component at a time, not novel because I have already seen a lot of algorithms that use this structure. For example, Flora [arXiv:2402.03293] and LoQT [arXiv:2405.16528] directly apply this structure; GaLore [arXiv:2403.03507] projects the gradient into subspaces and is later shown by GoLore [arXiv:2410.11289] that they are essentially equivalent to this structure. However, none of these related works are discussed or compared in this paper, and I may consider that the authors are not aware of them. 
2. Based on my first point, I believe this paper needs stronger theoretical insights or empirical performance to be accepted. Specifically,
i) In theory, the authors only study why LoRA-FA can mimic vanilla LoRA's update via Theorems 3.1 and 4.1. These results, while having their own values, cannot reflect why LoRA-FA can have better performance than vanilla LoRA in practice. As shown in Tables 2 and 3, vanilla LoRA is far worse than full fine-tuning and other variants. Consequently, LoRA-FA's ability to mimic LoRA cannot demonstrate why it can be comparable to stronger baselines.
ii) In practice, stronger baselines, including ReLoRA [arXiv:2307.05695] and other similar approaches that I have mentioned in my first point, are not compared with. This raises a fundamental question: whether LoRA-FA can truly beat previous SoTA algorithms, where some of them even share similar structures?

### Questions
I find the algorithmic description in Appendix B not convenient to read. I suggest replacing long variable names with simpler math symbols, and this description should also be included in the main text rather than in the appendix.

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
The paper introduces LoRA-FA (LoRA by Fixing A), a parameter-efficient fine-tuning method for large language models (LLMs). Standard LoRA adapts model weights using two trainable low-rank matrices, A and B. However, this doubles the number of adaptation parameters and can cause optimization instability. LoRA-FA addresses this by fixing A as a non-trainable matrix (initialized randomly or orthogonally) and only learning B, which effectively reduces trainable parameters by half while maintaining or improving model performance.

### Strengths
- The paper is clearly written and easy to follow, with only a few minor typos.
- The proposed method, LoRA-FA, is conceptually simple and straightforward to implement.
- Experimental results demonstrate that LoRA-FA achieves comparable or slightly better performance than standard LoRA across multiple LLMs (e.g., LLaMA, RoBERTa) and benchmarks.

### Weaknesses
- Limited Related Work Discussion: The related work section focuses mainly on low-rank adaptation methods (e.g., LoRA) and omits many state-of-the-art PEFT approaches from the past two years. The discussion should be expanded to include subset-of-parameter and sparse fine-tuning methods (e.g., [1–8]). These techniques represent a major trend in recent PEFT research and would help strengthen the paper’s contextual foundation.
* Incremental Novelty: The core idea—freezing one of the LoRA matrices (A)—is relatively incremental regards to novelty.
* Marginal Gains: Reported improvements in performance, memory efficiency, and training speed compared to standard LoRA appear modest. The empirical advantages may not be sufficient to justify the new method as a significant advancement.


### Minor Typos
* “fine-tuning pre-trained LLMs has been shown very effective” → missing “to be”
* “Experiments on system efficiency shows” → should be “show”
* “fix to 64” → should be “fixed to 64”
* “while keep the low-rank matrix A frozen” → should be “while keeping …”
* Several table cells missing digits (e.g., Table 2: “83.3 ± 0.”, “90.1 ± 0”). If intended as ±0, remove the decimal point.
* Caption: “Please ref to Appendix B” → should be “Please refer to Appendix B.”


[1] Parameter-Efficient Fine-Tuning without Introducing New Latency

[2] Sparse Matrix in Large Language Model Fine-tuning

[3] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

[4] Parameter-Efficient Transfer Learning with Diff Pruning

[5] Training Neural Networks with Fixed Sparse Masks

[6] Diff prunning: Parameter-Efficient Transfer Learning with Diff Pruning

[7] Scaling Sparse Fine-Tuning to Large Language Models

[8] Composable Sparse Fine-Tuning for Cross-Lingual Transfer

### Questions
- Consider adding evaluations on more recent LLMs beyond the RoBERTa and LLaMA families to strengthen the empirical evidence.
- Include at least one sparse or subset-based PEFT baseline for comparison.
- On Line 408: “LoRA-FA reduces memory usage by more than 27.8 GB (80 GB vs. 52.2 GB)”. On my understanding, when applying LoRA to LLaMA model, the activation memory is trivial since the number of trainable parameters are much less than full fine-tune. I conduct a simple calculation below and the activation memory cost is much smaller than 80GB. Could the author explain more details about how they calculate the activation memory and get the numbers?  

Using the LoRA activation memory formula:

**Activation Memory = 2 × b × s × (d + r)**

where:  
- **b** = batch size = 16  
- **s** = sequence length = 512  
- **d** = hidden size = 4096  
- **r** = LoRA rank = 64  

Substituting the values:

**2 × 16 × 512 × (4096 + 64) = 68,157,440 elements ≈ 130 MB per layer (FP16).**

Since **LLaMA-7B** has **32 transformer layers**, the total activation memory is:

**130 MB × 32 = 4.16 GB.**

Therefore, **LoRA (rank = 64, seq_len = 512, batch = 16)** requires approximately **4.2 GB** of activation memory in total.

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
To improve the efficiency and effectiveness of parameter-efficient fine-tuning (PEFT) for large language models (LLMs), the authors propose LoRA-FA, a simplified yet expressive variant of LoRA. By revealing an asymmetric, collapsible structure in LoRA updates, the authors show that one projection matrix can be frozen without losing expressivity. LoRA-FA freezes the projection-down matrix and trains only the projection-up matrix, further introducing closed-form gradient corrections to better approximate full gradients. Experiments on benchmarks such as GLUE, GSM8K, and HumanEval demonstrate that LoRA-FA matches or exceeds the performance of existing PEFT and Full-FT, while significantly reducing activation memory and computational overhead.

### Strengths
* S1: This paper provides a new perspective on LoRA optimization by interpreting its update as a collapsible single-layer linear regression and proposing a one-sided training method (training only matrix B).
* S2: The paper is well-structured and clearly presented, with theoretical proofs supporting the proposed method.
* S3: Extensive experiments are conducted across various tasks—including NLU (GLUE), math, code, and dialogue—comparing against several recent PEFT/LoRA variants (e.g., LoRA-GA, LoRA-Pro, AdaLoRA), demonstrating the method’s effectiveness and efficiency.

### Weaknesses
* W1: The assumptions in Theorem 3.1 / 4.1 may be too strong, and the paper lacks a discussion of the failure boundaries under practical model settings (e.g., when the rank $r$ is small or $A$ is nearly singular), as well as the consequences when these assumptions are not satisfied.
* W2: There is no comprehensive quantitative analysis of the additional computational or communication overhead introduced by the proposed method.
* W3: The paper lacks sensitivity analyses for some key hyperparameters, such as the scaling factor $\alpha$ and the effect of freezing different proportions of $A$ (e.g., only freezing certain layers instead of “all linear layers”).

### Questions
* Q1: Could the authors further elaborate on whether these assumptions still hold in practical models (e.g., when the rank is very small or $A$ is nearly singular)? How would the closed-form solution and performance of LoRA-FA be affected when $A$ is rank-deficient or poorly conditioned?
* Q2: The paper highlights that LoRA-FA can reduce activation memory, but it does not provide detailed data on the additional computational and communication costs. Could the authors include such quantitative analysis?
* Q3: Have the authors tried applying LoRA-FA only to specific layers (such as attention or MLP layers), or mixing LoRA-FA with standard LoRA across different layers?

### Soundness
3

### Presentation
2

### Contribution
2
