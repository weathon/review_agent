# IO-Adam: Rethinking Memory-Efficient Adaptive Optimizers from Gradient Computation

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Adaptive Moment Estimation (Adam) is one of the most popular stochastic optimizers for deep neural network training and has become the default optimizer in many scenarios, especially on language tasks. With the first and second moment estimation, Adam provides adaptive learning rates for each parameter, significantly outperforming Stochastic Gradient Descent (SGD). However, as the deep neural networks become larger, the estimation of the first and second moments takes up substantial memory, motivating methods to reduce the memory usage for adaptive optimizers. In this paper, we propose to rethink the first and second moment estimation from a gradient computation perspective. The gradient of the weight matrix is the multiplication of the input and the gradient of the output. Instead of trying to find a low-rank approximation for the first and second moment estimation as in previous works, we propose to track the input and the output gradient for efficient moment estimation. We provide analyses on the connection and difference between our proposed method, the widely used Adam optimizer, and previous memory-efficient optimizers proposed to reduce the memory usage. We conduct experiments to verify the effectiveness of our method, where our method reduces the memory usage by up to $30$% while preserving similar performance or even improving the performance of Adam.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a memory-efficient variant of the popular Adam optimizer that exploits the structure of gradients computed through backpropagation. The proposed method uses a second-moment estimate that is lower bounded by the second-moment estimate of Adam. The authors show a regret bound for the proposed optimizer, patterned off the analysis of Adam. Empirical results on fine-tuning, pretraining, and vision tasks demonstrate a 30-40% reduction in memory usage while maintaining comparing performance to Adam. A generalization of the main method is also proposed and explored.

### Strengths
1. To my knowledge, the proposed idea is novel and simple to understand, being well-motivated in the context of backpropagation.
2. The memory reduction is significant, even compared to similar work on memory-efficient Adam variants.
3. The generalization via Holder's inequality is quite interesting and promising. I found the intuition that $p<2$ should work better to be very insightful.

### Weaknesses
1. The theoretical contribution is extremely limited, essentially consisting of only a straightforward extension of the analyses in [1, 2]. A regret bound is common in online learning or reinforcement learning contexts but not necessarily standard for modern optimizer papers. It would be helpful to provide e.g. objective value or gradient norm bounds in order to strengthen the theoretical value of this work.
2. Given the primarily empirical nature of the contributions, the experiments should be held to the same standards as other empirical papers on optimizers, e.g. [3, 4, 5]. In particular, no evaluation is provided for larger models (7B+ parameters), making it difficult to conclude that the proposed method is scalable.
3. It is hidden in the appendix that the proposed method takes ~7% more time per step on average than AdamW, and it is merely mentioned that this implementation could possibly be improved. This may or may not have a significant practical impact, but I would have like to see more discussion in the main text.

[1] Diederik P. Kingma, Jimmy Ba. Adam: A Method for Stochastic Optimization, ICLR 2015.
[2] Sashank J. Reddi, Satyen Kale, Sanjiv Kumar. On the Convergence of Adam and Beyond, ICLR 2018.
[3] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian. GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection, ICML 2024.
[4] Qijun Luo, Hengxu Yu, Xiao Li. BAdam: A Memory Efficient Full Parameter Optimization Method for Large Language Models, NeurIPS 2024.
[5] Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, Shiwei Liu. SPAM: Spike-aware Adam with Momentum Reset for Stable LLM Training, ICLR 2025.

### Questions
1. In Line 772, it is mentioned that the analysis requires an additional assumption that was pointed out by Reddi et al. Could you clarify how this assumption is satisfied for the proposed method?
2. See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes IO-Adam, a “memory-efficient” variant of Adam that separately tracks the input activations $X$ and output gradients $\nabla_Y \mathcal{L}$ for each linear layer instead of storing full-size moment estimates. They re-express the weight gradient $\nabla_W \mathcal{L}=(\nabla_Y \mathcal{L})X^T$ and construct the second-moment estimate from the outer product of exponential moving averages of $X^2$ and $(\nabla_Y \mathcal{L})^2$, reducing memory costs.

### Strengths
1. This paper introduces an explicit dual-buffer implementation that separately stores input and output gradients, which can slightly reduce memory usage in small-batch settings.

2. The presentation of this paper is good.

### Weaknesses
1. The method fails to save memory when applied to attention layers, since both $\nabla_Y \mathcal{L}$ and $X$ are high-dimensional matrices rather than vectors. In this case, storing the input and output gradient buffers actually requires more memory than storing the weight gradients themselves. Moreover, in most large language models, attention layers constitute the majority of parameters and computation, making this limitation particularly critical. In addition, they works well only if the batch size is small.

2. If the method can only be applied to linear layers, it is unclear how the attention layers are handled in the LLaMA and GPT-2 experiments or how any memory savings are achieved in those cases.

3. The buffer for the first momentum stores the input and output gradients from the most recent $c$ batches, while the buffer for the second momentum maintains a “ring buffer” of size $b$. Consequently, the first momentum and the second momentum are computed over different batches, resulting in a mismatch between the two moment estimates.

4. The second momentum estimation can be highly biased, potentially far larger than the ground truth. There is no theoretical guarantee or mechanism to control this bias.

5. The modified bias correction term $(1-\beta_2^t)^2$ is introduced only because the authors multiply two EMAs. There is no theoretical analysis or empirical evidence supporting that this modification leads to unbiased or stable estimates.

6. The experiment results are not convincing. Although IO-Adam exhibits a large second-moment bias, it still outperforms AdamW in all reported cases. The authors attribute this improvement to larger learning rates, which is an unconvincing and insufficient explanation.

### Questions
1. Please define the variables $m$ and $n$ in Table 1.

2. From Eq.2, the statement of “the rank of the weight gradient is equal to the batch size $bs$” is not right. The rank should be less and equal to batch size.

3. No experimental details are given for the first-moment buffer. Please clarify.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a memory-efficient variant of the Adam optimizer, called IO-Adam. Specifically, for the second-moment estimation in Adam, this paper tracks the input vector and the output gradient for each layer. Since these are smaller than the full weight matrix, this approach significantly reduces memory cost. Experiments show that the proposed method is more memory-efficient than Adam, achieving around 30% reduction in memory usage. The paper also demonstrates that this tracking scheme estimates the second moment as an upper bound of Adam’s original second-moment estimate.

### Strengths
1) The paper studies an important challenge—the high memory cost of training large models using second-order moment estimates—and effectively reduces memory usage compared to the original Adam.
2) The motivation is clear, and the proposed method is reasonable. By separately tracking the input vector and output gradient, it efficiently estimates the momentum of weight gradients.
3) Theoretical analysis and experimental results are clearly presented. The proposed method successfully reduces memory usage and achieves comparable or better performance than related baselines.

### Weaknesses
1) According to the paper, memory storage depends on the batch size, meaning different batch sizes may lead to very different memory usage (larger batches may cause higher memory consumption compared to baselines). It is important to conduct a study evaluating performance and memory usage versus batch size.
2) I have concerns that in standard batch training, the input distribution may change over time due to sampling bias. Since this method stores input vectors, for earlier layers of the network, such estimation may lead to large deviations as the input changes drastically. Could this cause unstable training in the early stages? Or is a larger batch size required to stabilize it?
3) Another potential weakness is training time. This method may take longer per iteration compared to related works. It would be better to include total training convergence time and comparisons with baselines. If training time is significantly slower, the method’s practical usefulness may be limited.
4) Minor parts:  
4.1) The paper mainly focuses on training affine blocks; it would be useful to expand the scope to 2D CNNs.  
4.2) The paper notes that the optimal learning rate of IO-Adam may be larger than that of Adam. It would be valuable to study the optimal learning rate in broader experiments and, if possible, suggest a scaling guideline (e.g., a factor like 1.5x) for practical application.

### Questions
1) Could the authors conduct a study to evaluate performance and memory usage versus batch size?
2) Given that input distributions change during training, especially in earlier layers, could this cause instability? Would a larger batch size mitigate it?
3) What is the training time compared to baselines?
4) How is IO-Adam implemented for CNNs, and what is the performance impact?
5) Could the authors clarify how to determine or scale the learning rate?

Please see the [Weakness] section for more details.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents IO-Adam, a memory-efficient variant of Adam that redefines how second moments are stored and updated. Instead of maintaining a full per-parameter second-moment matrix, IO-Adam tracks moving averages of squared input activations and squared output gradients, reconstructing an approximate second moment from their outer product. This change significantly reduces optimizer memory use while keeping similar convergence guarantees. The authors also introduce a buffer mechanism to stabilize estimates and a Hölder-based generalization to control the tightness of the approximation. Experiments on NLP and vision tasks show that IO-Adam matches or slightly improves AdamW’s performance while cutting optimizer memory by about 30–40 percent.

### Strengths
* **Novel and well-grounded idea.** The approach builds on the natural structure of gradient computation, offering a fresh and simple way to reduce memory without complex matrix factorization.
* **Solid theory.** The paper shows that IO-Adam’s second-moment estimate upper bounds Adam’s and proves a matching regret bound, giving the method theoretical credibility.
* **Comprehensive experiments.** Evaluations across GLUE, ViT, and large-scale LLaMA pretraining show consistent results with significant memory savings.
* **Flexibility and extensions.** The buffer mechanism and the Hölder variant make the framework adaptable and potentially useful in broader settings.
* **Clear practical value.** The optimizer delivers meaningful savings in large-scale training while preserving accuracy.

### Weaknesses
* **Limited applicability.** The method only directly applies to linear layers. It is unclear how IO-Adam handles other parameter types such as convolutions, embeddings, or normalization layers. This limits general usability.
* **Unclear memory accounting.** The paper’s claims of 30–40 percent savings lack a complete breakdown including the cost of buffers, first moments, and parameters that might fall back to Adam.
* **Runtime overhead.** Tracking inputs and output gradients increases implementation complexity and can slow training. The paper does not quantify this overhead clearly.
* **Hyperparameter sensitivity.** The buffer size and Hölder exponent are new and sensitive settings. Their tuning rules are not well explained.
* **Limited statistical reporting.** Some performance gains appear small, with no standard deviation or multi-seed results to confirm significance.

### Questions
1. How does IO-Adam handle non-linear or non-matrix parameters such as embeddings or normalization weights?
2. Could the authors provide a full accounting of memory, including buffers, first moments, and fallback parameters?
3. How much runtime overhead does IO-Adam introduce compared with AdamW in large-scale training?
4. Is there a practical heuristic for choosing buffer size or Hölder exponent across tasks?
5. Could this approach be extended to convolutional layers or combined with other low-rank or quantized optimizers?

### Soundness
3

### Presentation
3

### Contribution
3
