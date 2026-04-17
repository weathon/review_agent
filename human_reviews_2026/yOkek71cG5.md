# Clapping: Removing Per-sample Storage for Pipeline Parallel Distributed Optimization with Communication Compression

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Pipeline-parallel distributed optimization is essential for large-scale machine learning but is challenged by significant communication overhead from transmitting high-dimensional activations and gradients between workers. Existing approaches often depend on impractical unbiased gradient assumptions or incur sample-size memory overhead. This paper introduces **Clapping**, a **C**ommunication compression algorithm with **LA**zy sam**P**ling for **P**ipeline-parallel learn**ING**. Clapping adopts a lazy sampling strategy that reuses data samples across steps, breaking sample-wise memory barrier and supporting convergence in few-epoch or even online training regimes. Clapping comprises two variants including **Clapping-FC** and **Clapping-FU**, both of which achieve convergence without relying on unbiased gradient assumption, effectively addressing compression error propagation in multi-worker settings. Among them, Clapping-FU has an asymptotic convergence rate of $\mathcal{O}(1/\sqrt{T})$. Numerical experiments validate the performance of \ours across different learning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Clapping, a pipeline-parallel training method that compresses activations and gradients without storing per-sample states. Using a lazy-sampling error-feedback mechanism, it maintains stability and convergence under compression. Clapping cuts communication by up to 95% with minimal accuracy loss in low-bandwidth settings.

### Strengths
1. Convergence analysis provides certain level of guarantee, though I have not been able to fully check the proofs in the appendix.

### Weaknesses
1. No wall-clock speed up has been shown against baseline. The theoretical speedup might not translate into actual advantage after all.
2. Intra-node PP with ultra-fast interconnect (NVLink/NVSwitch) where comm ≪ compute and microbatching already saturates the pipe; compression overhead could be wasted or even slow. A single datapoint of 100Mbps is also not a realistic assumption and not representative. It would be more informative to see speedup v.s. bandwidth pareto frontier.
3. There is no study on how batch size and device map of the model will impact the outcome. 
4. Only a single compression baseline is provided. It will be more convincing if more recent methods are included.
5. Table 2 results seem consistently worse than no compression in majority of the benchmarks. That indicates there is not really free lunch.

### Questions
Was the baseline no compression run with properly implemented with overlapping to reduce bubbles?
How does it change the throughput as batch size and number of workers grow for different methods?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Clapping, a communication-efficient pipeline-parallel optimization framework that eliminates per-sample storage through lazy sampling mechanisms. The idea of retaining data samples across steps significantly reduces memory usage while maintaining convergence. The paper is well-motivated, clearly written, and demonstrates solid theoretical grounding. The authors make an important contribution to the efficient training of large-scale models, and the proposed ideas are both innovative and practical for real-world distributed learning scenarios.

### Strengths
- Provides rigorous convergence proofs demonstrating theoretical soundness.
- Introduces a novel lazy sampling mechanism that effectively reduces memory consumption.
- Compares with several classical algorithms and shows significant performance improvements, which demonstrate the method’s effectiveness within a reasonable range of settings.

### Weaknesses
- Retaining previous samples may distort label distributions in classification tasks; this is not discussed or empirically examined.
- Compressing forward activations may add extra computation and bias, the paper lacks justification for its necessity.
- Experiments involve only two models (7B and 8B) with similar parameter sizes, limiting evidence of general applicability.

### Questions
The paper does not provide details or equations about the compression of gradients and forward activations. Could the authors clarify how these are compressed in the proposed method?

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
4

### Summary
This paper introduces Clapping, a communication compression framework for pipeline-parallel distributed optimization. By integrating error feedback and lazy sampling, Clapping-FU—achieve a state-of-the-art convergence rate. This result holds without relying on the unbiased gradient assumption and avoids the sample-wise memory overhead present in existing algorithms.

### Strengths
- Clapping is the first communication compression algorithm for pipeline parallelism that achieves a convergence rate of $O(1/\sqrt{T})$ without requiring the unbiased gradient assumption.

- Experimental results demonstrate that Clapping-FC achieves significantly better performance compared to the baseline, AQ-SGD.

### Weaknesses
- Limited Applicability to LLM Training: Clapping relies heavily on the assumption that training is conducted over multiple epochs. This contrasts with modern LLM pre-training, which often uses only one or a few epochs, and fine-tuning, which typically uses around four epochs. Furthermore, training on sequentially ordered data often yields weaker performance than training on fully shuffled data. It is unclear whether the no-compression baseline experiments in this work used completely shuffled data, a standard practice for achieving optimal performance. Consequently, Clapping's applicability to LLM tasks may be limited.

- Questionable Practical Efficiency: Clapping-FC converges at a rate of $O(1/\sqrt[3]{T})$, which is significantly slower than the $O(1/\sqrt{T})$ rate of non-compressed SGD. While Clapping-FU restores the $O(1/\sqrt{T})$ rate, it does so by forgoing compression with a probability of $1-p$. This limits the maximum communication reduction to a factor of $p$, and the paper suggests $p \approx 0.5$ is needed to maintain performance. Additionally, Clapping requires extra memory to store previous activations. This memory could otherwise be used to increase the batch size, potentially accelerating total training time. In summary, the net reduction in overall training time achieved by Clapping-FU may be minimal compared to a no-compression baseline.

### Questions
- Convergence with $p=0$: According to Algorithm 1, setting $p=0$ implies repeatedly training on the same data sample, a scenario under which $\frac{1}{T}\sum_{t=1}^T{\mathbb{E}} \Vert \nabla {\cal l(w_t) }\Vert^2$ would not be expected to converge. However, Lemma 7 and Lemma 13 state that convergence at a rate of $O(1/\sqrt{T})$ is still guaranteed in this case. This appears counterintuitive and may indicate an issue in the theoretical derivation.

- Inconsistency with Vanilla SGD: When the momentum parameter is set to zero ($m=0$), the algorithm reduces to vanilla SGD. The convergence bound given in Lemma 7 and Lemma 13 for this case is $O(1/\sqrt{T}) + \sigma^2$, which does not align with the established theoretical result that vanilla SGD can converge to a stationary point without such an additive noise term. Could the authors clarify this discrepancy?

- Error Accumulation in Deep Pipelines: Theorem 1 suggests that compression errors in Clapping accumulate exponentially with the pipeline depth $K$. For models with large $K$ (e.g., modern LLMs), this property would be a significant practical drawback. Could the authors comment on the practical implications of this for scaling to very deep models?

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
3

### Summary
The paper proposes the clapping framework, which addresses the key barrier of sample-scale memory overhead(L2) in communication compression for pipeline-paralllel distributed optimizaiton via the Lazy Sampling strategy.  Moreover, the framework provides theoretical convergentce guarantees without the unrealistic unbiased gradient assumption (L1) and extends its analysis to multi-worker setups(L4).

### Strengths
The paper resolves core memory bottleneces via the lazy sampling strategy.
The paper makes theoretical progress by providing convergence guarantees for pipeline-parallle compression algorithms.
The paper provides compatibility with modern optimizers. The convergence analysis is extended to the Adam optimizer and large-batch scenarios.

### Weaknesses
1. The lazy sampling strategy  sacrifices model generalization.
2.  The experimental scale and validation is insufficient. The largest pre-trainded model in experiments is 1B and fine-tuning task cannot fully represent pre-training challenges.

### Questions
NONE

### Soundness
2

### Presentation
3

### Contribution
2
