# DeMo: Decoupled Momentum Optimization

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Scaling neural network training increasingly depends on synchronous data-parallelism, yet full-precision gradient all-reduce imposes a severe communication bottleneck. We propose Decoupled Momentum Optimization, a drop-in replacement for any momentum-based optimizers that significantly reduces the communication bandwidth while maintaining convergence. DeMo (i) decouples local momentum updates, (ii) applies a fast orthonormal transform (e.g., DCT) followed by top-$k$ sparsification, and (iii) reuses the momentum buffer for error feedback via momentum subtraction. This design reduces per-step communication by up to two orders of magnitude with minimal computational overhead.  Experiments on 300M- and 1B-parameter DeMo language models show DeMo transmits up to 85× less data per GPU than AdamW-DDP while achieving comparable loss and accuracy. DeMo is topology-agnostic and enables training across multi-datacenter or Ethernet-based setups.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DeMO, a new algorithm for communication efficient distributed training. DeMo works by compressing the communication by transmitting a compressed version of a local momentum from each worker instead of the gradient. The compression is performed by performing a discrete cosine transform and only transmitting the top coefficients. They also explore using error feedback on top of the local momentum at each worker. The paper compares DeMo to an AdamW baseline for LLM training at a decent scale ~330M and ~1B scale.

### Strengths
* Paper is well written, easy to follow with clear figures
* Method is interesting and has practical relevance
* Experiments are performed on a convincing scale
* Code provided

### Weaknesses
* Experimental comparison is lacking in crucial ways:
    * There is no comparison with other communication focused distributed training algorithms
    * The AdamW baseline does not seem to be tuned
    * There are no experiments with Muon even if this is discussed in Section 2.1

### Questions
Overall I would like to see the experimental comparison significantly strengthened to change my opinion.

* What learning rate values were used for the experiments, how were these tuned? Note that changing the momentum coefficient for certain optimizers can change the "effective learning rate" and shift the optimal LR value.
* Why not compare against some distributed baseline? It feels a bit odd not too since this is not the first paper to focus on communication efficient optimization.
* To what extent is this algorithm compatible with different reduction techniques, e.g. without a central server?
* Why does the AdamW baseline perform so poorly in Figure 2 but well in Figure 3?
* Comparing training curves directly like in Figure 2 is not very informative. Unless each line is fully tuned, what is the takeaway here? Changing the LR or other important HPs slightly could also significantly change any given curve.
* L251-L254 are not very clear. Why these values for the momentum?
* It would also be nice to clarify the computational overhead. What is a typical flop overhead?

Minor pointers:
* There is a missing reference in L438

### Soundness
2

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
5

### Summary
This paper introduces Decoupled Momentum Optimization (DeMo) to reduce communication overhead. DeMo integrates three key techniques: local momentum decoupling, a fast orthonormal (e.g., DCT) transform with top-k sparsification, and momentum subtraction for error feedback.

### Strengths
The primary objective of this paper is to democratize the training and fine-tuning of large models by minimizing communication volume, which obviates the requirement for expensive, high-performance networking infrastructure.

### Weaknesses
Limited Perceived Novelty: The claimed novelties of DeMo appear to build upon existing work. For instance, [1] also compresses the momentum term rather than the raw gradient. Even earlier literature has demonstrated that compressing the total model update, rather than the intermediate gradient, can maintain competitive performance while dramatically reducing communication overhead. Furthermore, the technique of using the momentum buffer to store error feedback for memory efficiency is also employed in [1].

Questionable Practical Efficiency: The net reduction in overall training time achieved by DeMo may be minimal, despite its apparent reduction in communication volume. DeMo requires computationally expensive operations, including a sorting algorithm to identify the Top-k elements and two orthonormal transforms per step. The overhead from these operations is non-trivial and may offset the gains from reduced communication time.
Moreover, as a variant of the Top-K algorithm, DeMo's communication savings are primarily effective in parameter-server architectures. In a decentralized architecture, aggregating Top-K elements requires a AllGatther operation instead of a more efficient AllReduce. The communication cost of AllGather is $O(n^2K)$ while AllReduce is $O(nK)$, where $n$ is the number of nodes. This scaling implies that DeMo may not yield net communication savings with a large number of nodes. This concern is supported by [3], which experimentally demonstrated that Top-K methods can be slower than non-compressed optimizers, even when network bandwidth is the bottleneck.

References

[1] Ahn, Kwangjun, et al. "Dion: Distributed orthonormalized updates." arXiv:2504.05295, 2025.

[2] Peng, Hanyang, et al. "Birder: communication-efficient 1-bit adaptive optimizer for practical distributed DNN training." NeurIPS, 2023.

[3] Agarwal, Saurabh, et al. "On the utility of gradient compression in distributed training systems." Proceedings of Machine Learning and Systems, 2022.

### Questions
- Counterintuitive Performance: As shown in Figure 2, DeMo with various Top-K sparsity levels appears to outperform AdamW. This is counterintuitive, as Signum (the uncompressed version of the update direction used in DeMo) is generally known to underperform compared to AdamW. Could the authors provide an explanation for this phenomenon?

- Clarification on Parameter $\alpha$: The parameter $\alpha$ introduced in Subsection 3.4 is not mentioned in the preceding sections. Could the authors clarify its role and meaning within the DeMo framework?

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
This paper proposes Decoupled Momentum Optimization (DeMo), a communication-efficient framework for distributed training designed to replace standard gradient-based All-Reduce in data-parallel setups. The core idea is to synchronize a compressed version of the optimizer's momentum state instead of the full gradient. DeMo works by (i) allowing each worker to maintain and update a local momentum buffer, (ii) applying a blockwise fast orthonormal transform (e.g., DCT) to this momentum, (iii) communicating only the top-k sparsified coefficients, and (iv) performing momentum subtraction to reuse the local momentum buffer as an implicit error-feedback mechanism. The authors demonstrate that this approach drastically reduces communication overhead while achieving comparable pre-training loss and downstream accuracy to a standard AdamW baseline on 300M and 1B models.

### Strengths
1. The paper's core idea of decoupling momentum into dominant components (for communication) and non-dominant components (for local updates) presents an effective and intuitive paradigm for reducing communication overhead.

2. The application of DCT as the mechanism for sparsification, rather than operating in the raw gradient or momentum space, is a novel approach that provides a new perspective on structured compression.

### Weaknesses
## 1. The theoretical analysis, particularly the convergence rate presented in Theorem 1, warrants further discussion as it does not appear to be optimal. 
For compressed stochastic optimization algorithms in the non-convex setting, convergence rates can achieve $\mathcal{O}(\omega/\sqrt{NT})$ (where $\omega$ is the variance bound of the compression estimator, see [1]). Notably, this rate is dimension-free ($D$ is not in the numerator). The dimension $D$ typically appears when analyzing the total communication complexity (convergence iterations $\times$ per-step cost), which might be $\mathcal{O}(D/\sqrt{NT})$. A dependency on $D$ in the convergence rate itself is common in analyses of Adam-based optimizers, which often require an $l_1$-norm analysis due to per-element operations. However, SGD-based algorithms typically yield an $l_2$-norm result and a dimension-free convergence rate [2]. Although Theorem 1 is applied to SGD/Signum-based methods, the proof relies on an $l_1$-norm analysis, resulting in a dimension-dependent convergence rate of $\mathcal{O}(D/\sqrt{T})$. When factoring in the per-step communication cost $k$, the overall complexity does not seem to surpass existing state-of-the-art results.

## 2. The paper provides insufficient comparisons, both theoretically and experimentally. 

2.1 Theoretically: As noted above, the authors do not compare their theoretical bounds against other established compression algorithms, making it difficult to ascertain a clear advantage from the provided convergence rate. Intuitively, DeMo acts as a hybrid between local SGD and standard global-all-reduce SGD, where dominant momentum directions are communicated precisely while non-dominant directions are updated locally. This bears similarity to methods that perform communication and optimization on a subspace [2, 3], yet the theoretical results for DeMo do not appear to go beyond this existing work.

2.2 Experimentally: The paper only compares DeMo against a standard AdamW-DDP baseline. It lacks comparisons against other state-of-the-art communication compression methods. Without this, it is difficult to conclude that the proposed paradigm of compressing momentum instead of the gradient offers any essential advantage.

## 3. The paper focuses on communication compression, but the experimental scale (300M and 1B models) is relatively small for assessing this bottleneck. 
In pre-training models of this size, communication overhead often accounts for only 20% or less of the total training time. Therefore, even with a significant reduction in data transmission, the actual wall-clock time savings might be limited, especially if the compression scheme introduces any convergence instability or computational overhead. Reporting end-to-end wall-clock time comparisons is crucial to demonstrate the method's practical benefit, but this is missing from the experiments. Further experiments in larger-scale settings (e.g., those involving model sharding and hybrid parallelism) would be valuable to demonstrate its practical impact where communication is a more significant bottleneck.

[1] MARINA: Faster Non-Convex Distributed Learning with Compression

[2] SEPARATE: A Simple Low-rank Projection for Gradient Compression in Modern Large-scale Model Training Process

[3] Greedy Low-Rank Gradient Compression for Distributed Learning with Convergence Guarantees

### Questions
1. Communicating the principal components of the momentum, for instance by using SVD or a random projection onto a subspace, could also achieve the goal of extracting the dominant information. What are the specific advantages of using a fixed DCT basis followed by top-k sparsification compared to these adaptive low-rank projection methods?

2. In Figure 2, the DeMo training curves converge significantly faster and to a lower loss than the AdamW-DDP baseline. Intuitively, compression is lossy and introduces an information bottleneck, which might be expected to slow down convergence, not accelerate it. How do the authors explain this phenomenon where a compressed, lossy update outperforms the full-precision baseline?

### Soundness
3

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
This paper proposes DeMo (Decoupled Momentum Optimization), a communication-efficient distributed training method that addresses the gradient synchronization bottleneck in data-parallel training. 
Key innovation: decouple local momentum updates across workers, apply blockwise DCT transformation followed by top-k sparsification, and reuse the momentum buffer for error feedback through momentum subtraction. 
Experimental results: the authors demonstrate up to 85× reduction in communication while maintaining comparable performance to AdamW on 300M and 1B parameter language models trained on 100B tokens.

### Strengths
- Practical Impact: Achieves dramatic communication reduction (up to 85×) while maintaining model quality, which is highly valuable for distributed training scenarios with limited bandwidth.
- Clean Design: The three-component approach (decoupled momentum, DCT+top-k, momentum subtraction) is conceptually clear and builds naturally on existing optimization principles.

### Weaknesses
- Limited Scalability Analysis: The download bandwidth scales with the number of workers, which could become prohibitive at large scale; but all experiments use 64 GPUs; scaling behavior to hundreds or thousands of GPUs remains unclear.
- Multi-datacenter: The method is positioned for "multi-datacenter" training but only tested within single datacenters.
- Baseline Comparisons: No comparison with other communication-efficient methods (e.g., gradient quantization, other sparsification approaches like Deep Gradient Compression, or recent methods like DiLoCo beyond brief mention).
- Baseline: Missing comparison with gradient compression baselines that would contextualize the 85× improvement.
- Baseline HParams: AdamW baseline uses standard hyperparameters, but DeMo requires β=0.999 (vs standard 0.9) - this hyperparameter sensitivity isn't fully explored.
- Scale: Models are relatively small (300M, 1B) - behavior at 7B+ parameter scales is unknown.
- Theoretical: No analysis of how DCT specifically helps beyond empirical observation. Why is DCT the correct basis to project onto?

### Questions
Weaknesses section leaves a lot of questions.

### Soundness
2

### Presentation
2

### Contribution
2
