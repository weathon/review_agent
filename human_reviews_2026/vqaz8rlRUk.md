# MetaCluster: Enabling Deep Compression of Kolmogorov-Arnold Network

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Kolmogorov-Arnold Networks (KANs) replace scalar weights with per-edge vectors of basis coefficients, thereby boosting expressivity and accuracy but at the same time resulting in a multiplicative increase in parameters and memory. We propose MetaCluster, a framework that makes KANs highly compressible without sacrificing accuracy. Specifically, a lightweight meta‑learner, trained jointly with the KAN, is used to map low‑dimensional embedding to coefficient vectors, shaping them to lie on a low‑dimensional manifold that is amenable to clustering. We then run K-means in coefficient space and replace per‑edge vectors with shared centroids.  Afterwards, the meta‑learner can be discarded, and a brief fine‑tuning of the centroid codebook recovers any residual accuracy loss. The resulting model stores only a small codebook and per-edge indices, exploiting the vector nature of KAN parameters to amortize storage across multiple coefficients. On MNIST, CIFAR-10, and CIFAR-100, across standard KANs and ConvKANs using multiple basis functions, MetaCluster achieves a reduction of up to $80\times$ in parameter storage, with no loss in accuracy. Similarly, on high-dimensional equation modeling tasks, MetaCluster achieves a parameter reduction of $124.1\times$, without impacting performance. Code will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the memory overhead of Kolmogorov–Arnold Networks (KANs), where each edge carries a vector of basis coefficients rather than a scalar. It proposes MetaCluster, a three-stage pipeline: (1) train a lightweight meta-learner that maps low-dimensional embeddings to coefficient vectors so that coefficients lie on a clusterable low-dimensional manifold; (2) perform K-means in coefficient space and replace per-edge vectors with shared centroids plus indices; (3) discard the meta-learner and briefly fine-tune the centroid codebook. On MNIST, CIFAR-10/100, and both fully-connected KANs and ConvKANs with several basis families, MetaCluster reports up to 80× parameter-storage reduction without accuracy loss relative to the uncompressed KAN.

### Strengths
- Leveraging the vectorized per-edge structure of KANs makes codebook amortization particularly effective compared to scalar-weight MLPs. The method is simple to integrate: the meta-learner is used only during training and removed at inference.
- Clear articulation of why naïve weight sharing fails on KANs  and how manifold shaping mitigates this.
- Solid empirical coverage for small/medium scale: Results span FC-KAN and ConvKAN, with B-spline/RBF/Gram bases on MNIST/CIFAR-10/100.

### Weaknesses
- Experiments are limited to MNIST/CIFAR and relatively small KAN/ConvKANs; there is no large-scale vision or transformer-style model demonstration.
- The paper emphasizes zero inference overhead from removing the meta-learner, but provides no wall-clock or FLOPs comparison of training cost vs. baselines for the meta-learner + clustering + fine-tuning stages.
- While related work mentions Hessian-weighted K-means and differentiable K-means (DKM), the paper does not evaluate these variants or other clustering families (e.g., hierarchical or agglomerative).
- The paper uses a single global K per model family (e.g., FC-KAN K=16; ConvKAN K=256 per hyperparameter tables) and does not explore layer-wise varying K.

### Questions
1. Recent post-training, clustering-based methods (e.g., model folding[1], IFM[2] ) share the theme of parameter sharing/tying and low-dimensional structure. How does MetaCluster compare conceptually and empirically? Would it be possible to perform post-training clustering on a KAN model trained without metaclustering?

2. The authors state that quantization is complementary. Do you anticipate non-trivial accuracy loss when combining MetaCluster with 8-bit / 4-bit quantization of centroids and/or indices? Any preliminary data?

[1] Wang, Dong, et al. "Forget the data and fine-tuning! just fold the network to compress." arXiv preprint arXiv:2502.10216 (2025).
[2] Chen, Yiting, Zhanpeng Zhou, and Junchi Yan. "Going beyond neural network feature similarity: The network feature complexity and its interpretation using category theory." arXiv preprint arXiv:2310.06756 (2023).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose MetaCluster, a compression framework for Kolmogorov–Arnold Networks (KANs). Although KANs have demonstrated stronger performance than MLPs, they require significantly more parameters. This motivates a compression strategy based on weight sharing, where parameters are clustered into a small codebook, and only compact indices are stored. However, applying standard clustering directly to KANs is not straightforward. To address this, the authors train a small meta-learner that maps each edge’s coefficient vector onto a low-dimensional manifold, after which the vectors are clustered using K-means. The per-edge coefficients are then replaced by shared centroids (plus indices), followed by brief centroid fine-tuning. MetaCluster achieves up to an 80× reduction in parameter storage with no loss in accuracy.

### Strengths
The paper is well written, and its motivation is clear. The main strength lies in the impressive experimental results, as demonstrated in Tables 1 and 2. Additionally, the authors provide thorough ablation studies to validate their design choices, as shown in Section 4.3.

### Weaknesses
- The authors do not benchmark against non-KAN compression baselines. Given the extensive literature on model compression, it would be valuable to compare MetaCluster with common techniques (e.g., pruning, quantization, or weight sharing) applied to MLPs or CNNs. This would help clarify whether MetaCluster is state-of-the-art relative to general compression methods. If those methods are not easily extendable to KANs, a discussion explaining why would strengthen the paper.

- The evaluation is conducted only on relatively simple datasets (MNIST and CIFAR). It remains unclear how MetaCluster performs on more challenging or large-scale datasets.

- The meta-learner does induce a bit more of training complexity, since this adds engineering steps and hyperparameters (e.g., number of clusters, embedding dimnesions, etc) which can complicate the process. Can we also not jointly train but perhaps find a way to do post-training compression, i.e., learn a meta-learner afterwards? Perhaps decoupling these can ease the process a bit.

- The ablation results indicate that performance is sensitive to the meta-embedding dimension: clustering becomes more difficult as the embedding dimension increases. This suggests that finding a suitable dimension may require tuning and makes the method less plug-and-play.

### Questions
I have a few simple questions:

- Why choose K-means instead of other potential alternatives? Perhaps there are other choices that could boost the performance of MetaCluster. 

- Does your compression ratio / memory account for codebook overhead + index storage + bit packing? 

I apologize in advance if these were already answered in the manuscript and I missed them.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MetaCluster, a three-stage compression framework for Kolmogorov-Arnold Networks (KANs) that combines meta-learning with weight sharing. In the first stage, a small meta-learner maps low-dimensional embeddings into coefficient vectors, constraining them to lie on a manifold that is highly clusterable. Then, K-means clustering is applied to replace per-edge coefficient vectors with shared codebook centroids. Finally, the centroids are fine-tuned to recover accuracy. The authors report up to 80× reduction in parameter storage with minimal accuracy loss across multiple architectures and datasets.

### Strengths
- The paper clearly identifies KAN’s memory inefficiency and proposes a targeted solution.
- This is the first successful application of weight sharing specifically designed for KANs.
- The paper is well organized and easy to follow.

### Weaknesses
- Complete absence of vector quantization literature. The proposed method is fundamentally vector quantization (VQ): mapping high-dimensional vectors to discrete codebook entries via clustering. However, the paper never mentions "vector quantization" and ignores relevant research.
- Lack of comparison with established vector quantization methods. The paper employs standard K-means with Euclidean distance but provides no comparison against advanced VQ techniques. For example, Product Quantization [1], which decomposes vectors into subvectors quantized independently, could achieve superior compression ratios. The choice of Euclidean distance over alternatives (cosine similarity, learned metrics) is also unjustified—for coefficient vectors representing basis functions, cosine similarity might better preserve functional shape. Without these comparisons, we cannot assess whether the meta-learner genuinely adds value over simpler VQ baselines.
- The paper motivates KAN compression with the references of KNN’s advantages in scientific tasks. However, all experiments are conducted on computer vision tasks (MNIST, CIFAR-10, CIFAR-100). This is largely different from the scientific tasks where KNN is explored. It would strengthen the paper to include evaluations on domains that better reflect the stated motivation, such as scientific or physical modeling tasks.

[1] Jégou et al. (2010) proposed product quantization for efficient nearest neighbor search.

### Questions
- Can you provide results on scientific computing tasks (equation modeling, PDE solving) where KANs have demonstrated their primary advantages, rather than only vision tasks?
- How does your method compare against Product Quantization, which decomposes vectors into independently quantized subvectors and could achieve smaller codebook sizes?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a three stage weight sharing method for KAN. The first stage involves mapping low dimensional embeddings to per-edge coefficient vectors. The second stage involves k-means clustering to replace per-edge vectors with centroids. The final stage involves  finetuning centroid codebook to recover accuracy loss.

### Strengths
The proposed weight sharing method greatly reduces the amount of trainable parameters in KAN.

### Weaknesses
The proposed method crucially relies on k-means clustering to provide reasonable good centroids. However, k-means clustering assumes the data is spherically shaped, which may not be true in practice. Could the authors replace the K-means clustering by other clustering methods (e.g. gaussian mixture model) to illustrate the proposed method can be used together with different clustering algorithms?

### Questions
Could the authors theoretically quantify the accuracy loss from weight sharing, compared to vanilla KAN?
What is the complexity of the proposed 3-stage approach and how does it compare to vanilla KAN?
Could the authors report the standard error in experiments (table 1-3)? The standard error can potentially be obtained by using different train/validation splits.

### Soundness
3

### Presentation
2

### Contribution
2
