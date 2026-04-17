# BoRA: Towards More Expressive Low-Rank Adaptation with Block Diversity

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Low-rank adaptation (LoRA) is a parameter-efficient fine-tuning (PEFT) method widely used in large language models (LLMs). 
It approximates the update of a pretrained weight matrix $W\in\mathbb{R}^{m\times n}$ by the product of two low-rank matrices, $BA$, where $A \in\mathbb{R}^{r\times n}$ and $B\in\mathbb{R}^{m\times r} (r\ll\min\{m,n\})$.
Increasing the dimension $r$ can raise the rank of LoRA weights (i.e., $BA$), which typically improves fine-tuning performance but also significantly increases the number of trainable parameters.
In this paper, we propose **Block Diversified Low-Rank Adaptation (BoRA)**, which improves the rank of LoRA weights with a small number of additional parameters.
Specifically, BoRA treats the product $BA$ as a block matrix multiplication, where $A$ and $B$ are partitioned into $b$ blocks along the columns and rows, respectively (i.e., $A=[A_1,\dots,A_b]$ and $B=[B_1,\dots,B_b]^\top$).
Consequently, the product $BA$ becomes the concatenation of the block products $B_iA_j$ for $i,j\in[b]$.
To enhance the diversity of different block products, BoRA introduces a unique diagonal matrix $\Sigma_{i,j} \in \mathbb{R}^{r\times r}$ for each block multiplication, resulting in $B_i \Sigma_{i,j} A_j$.
By leveraging these block-wise diagonal matrices, BoRA increases the rank of LoRA weights by a factor of $b$ while only requiring $b^2r$ additional parameters. 
Extensive experiments across multiple datasets and models demonstrate the superiority of BoRA, and ablation studies further validate its scalability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes BoRA, a parameter-efficient fine-tuning method designed to enhance the expressiveness of standard LoRA. The core idea is to partition the LoRA matrices A and B into multiple blocks and introduce a unique learnable diagonal matrix $\Sigma$ for each block pair. Through extensive experiments, the authors demonstrate that BoRA consistently outperforms standard LoRA and its variants under a similar number of parameters.

### Strengths
1. The idea to improve LoRA's power of expressiveness by incorporating blockwise learnable diagonal matrices is novel and makes sense to me.
2. The experimental results consistently show the advantage of the proposed method.
3. The presentation is clear, and the paper is easy to follow.

### Weaknesses
1. I think the comparison with other LoRA-like baseline algorithms is not enough. Beyond LoRA, DoRA, MELoRA and HydraLoRA, there are other similar methods that also aim to improve LoRA's expressive power yet not compared or discussed in this paper. For example, 1) ReLoRA is discussed but not compared with; 2) SLTrain [https://arxiv.org/abs/2406.02214] is neither compared nor discussed, it can also largely improve the model performance; 3) LoLDU [https://arxiv.org/abs/2410.13618] is neither compared nor discussed, I believe its idea has some similarity with this paper.
2. The paper analyzes the FLOPs for the forward pass in Section 3.4, but does not discuss the actual overhead of backward propagation and end-to-end training time. Even if FLOPs are similar, introducing numerous small block operations may lead to increased kernel launch of CUDA, which could potentially result in a large amount of overhead. It is recommended to have a further discussion on the computation overhead.
3. A minor question: I notice that a prior work [https://arxiv.org/abs/2407.15857] in this literature has already used the name *BoRA* for their method. I recommend using another name to avoid ambiguity, or the authors can include some discussions about that BoRA algorithm.

### Questions
1. In practical fine-tuning, what is the percentage increase in time per training step for BoRA compared to standard LoRA or other baselines?
2. Can the authors provide more insights on how to decide an appropriate $b$ when using BoRA in practice, where both model scales and targeted tasks can vary?

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
3

### Summary
The paper presents Block-Diversified Low-Rank Adaptation (BoRA) to enhance rank by introducing block-wise diagonal matrices. 
This increases rank with only a few additional parameters. Theoretical analysis confirms provable rank enhancement with an upper bound.
Empirical results show that BoRA achieves 2-4% accuracy improvements over LoRA across GLUE, mathematical, and commonsense reasoning tasks with similar parameter counts and involves various network architectures.

### Strengths
1. The paper introduces a novel perspective on LoRA by analyzing it through block matrix multiplication, revealing how correlations between block products constrain rank.
2. The paper demonstrates that both standard LoRA and MELoRA are special cases of BoRA, creating a unified theoretical framework.
3. Empirical results are solid, and BoRA is compared with various base models and tasks while achieving strong performance over a range of baselines.
4. The paper provides thorough ablation studies that validate the importance of both the exponential function and normalization function in BoRA. Results also show the consistently superior performance with varying ranks and block numbers.

### Weaknesses
1. In Section 3.2, the authors states
> Assuming $A$ and $B$ are divided into $b$ blocks along columns and rows, respectively, BoRA will additionally learn a set of diagonal matrices $\\{\Sigma_{i, j} \in \mathbb{R}^{r \times r} \mid i, j \in[r]\\}$.

​	This notation shows $i, j \in [r]$ for the indices of $\Sigma_{i,j}$, but this is inconsistent with the block count $b$. Since $A$ and $B$ are divided into $b$ blocks, the indices should be $i, j \in [b]$, not $[r]$.

2. Proposition 1 claims that BoRA requires only $b^{2} r$ additional parameters to increase the weight rank by a factor of $b$. In contrast, LoRA needs $(m+n)(b^{2} r)$ parameters to achieve the same bound, which, however, is incorrect. To achieve a rank of $br$, standard LoRA would need to set its rank parameter to $br$, which would require $r'(m+n)$ parameters where $r'=br$, thus $br(m+n)$ parameters - not $(m+n)(b^{2}r)$. A correct comparison is thus  LoRA (rank $br$) with $(m+n)br$ parameters vs. BoRA (rank bound $br$) with $(m+n)r + b^{2}r$ parameters. The authors should compare these two terms in practice.

3. Proposition 1 claims that the rank of BoRA's weight update $\Delta W$ satisfies $\operatorname{rank}(\Delta W) \leq \min \\{m, n, b r\\}$, but this bound is not properly justified and is too loose. Proposition 1 only gives $\operatorname{rank}(\Delta W) \leq b r$. The key claim that BoRA increases the rank by a factor of $b$ (i.e., up to $b r$) requires a tighter argument that BoRA can actually achieve this bound, which is not provided.

### Questions
1. Could you provide a theoretical justification that the rank of BoRA  can achieve $br$ under some assumptions, rather than merely being bounded by it?
2. In Section 4.5, why is $0.005$ chosen as the threshold?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a parameter-efficient fine-tuning method called **Block Diversified Low-Rank Adaptation (BoRA)**, which improves the rank of LoRA weights through matrix block  diversification. The core idea of BoRA is to divide the LoRA weight matrices $A$ and $B$ into $b$ blocks the columns and rows. To break the correlation between the block products $B_iA_j$ in different rows or columns, BoRA introduces a unique diagonal matrix,  $\Sigma_{i,j}$ , for each block multiplication, resulting in $B_i\Sigma_{i,j}A_j$. This formulation increases the upper bound of the LoRA weight rank by $b$ times while adding only $b^2r$ additional parameters. To demonstrate the effectiveness of BoRA, the authors conduct extensive experiments on natural language understanding, mathematical reasoning, and commonsense reasoning tasks, covering both RoBERTa and various mainstream LLMs. The results show that, compared with LoRA and three of its variants, BoRA consistently achieves stable performance improvements. Under comparable parameter settings, BoRA improves accuracy by about 2-4\% over LoRA.

### Strengths
- The paper analyzes the rank limitation of standard LoRA from the perspective of block matrix multiplication, showing that the correlation between block matrices constrains the expressiveness of LoRA and thus provides a strong theoretical foundation for the proposed solution. 
- It introduces a LoRA variant called BoRA, which eliminates the correlation between block matrices by introducing diagonal matrices, achieving substantial performance improvement with only a small parameter overhead of $b^2r$. 
- The authors also prove that LoRA and MELoRA are special cases of BoRA, thereby establishing a clear connection with previous work. 
- Comprehensive evaluations on models with different architectures and across various task types demonstrate the broad applicability of the BoRA.

### Weaknesses
- Although the authors mention inference latency in the appendix, they lack efficiency comparisons during training (e.g., convergence time, memory usage, training latency), and adding such comparisons would make the work more convincing.
- The authors claim that BoRA raises the rank upper bound, and in Related Work they also mention HiRA and KronA as methods that increase the rank of LoRA weights, yet these are not included as baselines for comparison.
- In the original MELoRA paper, the number of epochs for GLUE tasks is much larger (e.g., 60 for SST-2), whereas this paper uses relatively small epoch counts (e.g., 2 for SST-2). This discrepancy may affect the resulting performance.
- When discussing different tuning granularity, the authors compare BoRA only with LoRA and do not compare it with other baselines.

### Questions
1. Could you provide efficiency comparisons during training (e.g., convergence time, memory usage, training latency)?
2. When dividing the matrices into $b$ blocks, on matrices of what size is this partitioning performed?
3. Why are the ranks of the other baselines set to 8, while HydraLoRA is set to 4? Could you also provide results for HydraLoRA with rank 8?
4. Why did the authors choose relatively small numbers of training epochs?
5. In Fig. 3(b), standard LoRA improves as $r$ increases, whereas BoRA with $b=16$ or $64$ shows performance drops as $r$ increases (e.g., for $b=16, r=8 \ vs. r=16$). Could you explain the reason for this behavior?
6. For Different Tuning Granularity, why are comparisons made only between BoRA and LoRA, without including other baselines?
7. Is the notation $i, j \in [r]$ in Section 3.2 a typo by the authors? Should $r$ be $b$ here?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes BoRA, which addresses the rank-constrained problem of LoRA by enhancing diversity through matrix partitioning and the introduction of a block diagonal matrix, thereby increasing the weight rank of LoRA by a factor of b with only a few additional parameters. Multi-model, multi-task experiments show that it outperforms LoRA and its variants, achieving a 2-4% accuracy improvement with similar parameters.

### Strengths
- The methodology is clearly and comprehensively described, and Figure 1 clearly explains BoRA and its differences from other methods.

- The experiments in the paper are comprehensive, validating the effectiveness of the method on different tasks and models.

### Weaknesses
- The "Results of Different Tuning Granularity" section in Section 4.4 does not guarantee that the number of parameters or computational costs will be consistent (or as consistent as possible).

- Although the authors analyzed the efficiency of BoRA, experiments comparing the efficiency of different methods are lacking.

### Questions
- Why must the diagonal matrix $\Sigma $ be non-negative?

- Is it feasible to freeze the existing trained LoRA structure and train only the diagonal matrix $\Sigma $?

- Will the matrix rank change with different layers and training periods?

### Soundness
3

### Presentation
3

### Contribution
3
