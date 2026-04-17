# Reshape-then-Factorize: Communication-Efficient FL via Model-Agnostic Projection Optimization

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Federated learning (FL) enables collaborative model training across distributed clients without sharing sensitive data. However, communication overhead remains a significant bottleneck, particularly for large-scale models. Low-rank decomposition techniques address this by approximating each layer’s weights or gradients with a product of low-rank matrices, thereby reducing the communication cost in FL. While effective, these methods are constrained by the layer's architecture and shapes, limiting their flexibility and performance.
We propose *Model-Agnostic Projection Optimization* (MAPO), a novel method that reshapes and factorizes the full model gradient into a *fixed reconstruction matrix* and a *trainable projection vector*, avoiding layer-wise decomposition and architecture constraints. MAPO directly optimizes the projection in a randomly sampled subspace, with all clients generating the reconstruction matrix via a shared random seed, incurring no additional communication overhead for synchronization.
By decoupling the gradient from architectural constraints through reshaping and enabling communication-free exploration of dynamic subspaces via seed sharing, MAPO provides a more flexible and efficient low-rank representation.
Empirical results demonstrate the effectiveness of MAPO in various FL settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a model-agnostic factorization method MAPO. The method first flattens the whole model’s gradient so that it ignores the model architecture, and reshapes the flattened model gradient into a matrix, then factorize it as BA with a small trainable projection vector B and a shared reconstruction vector A.
The paper also provides theoretical analysis on factorization properties of MAPO, which states that MAPO achieves computation and communication efficiency. Empirically, MAPO is evaluated on vision, text, and GLUE fine-tuning, showing good empirical performance.

### Strengths
The idea is creative: it’s model-agnostic and applies a factorization to the flattened whole-model gradient, not layer by layer, which makes it broadly usable without architecture-specific tweaks. Thus, the method is easy to plug into different models.
The theory is straightforward and MAPO preserves reconstruction error and communication overhead compared to model-dependent factorizations.
The experiments make a practical point: accuracy stays close to FedAvg.

### Weaknesses
The discussion of the “low-rank averaging problem” is still an open question. Exact aggregation may not be important since the aggregation error can be larger than the error induced by low-rank averaging. It needs a further justification. Thus, the paper argues MAPO avoids certain pitfalls, but it doesn’t give a scientific underpinning like a theorem. A direct comparison between frozen-A and trainable-A would make the argument more convincing.
MAPO doesn’t naturally accommodate architecture-dependent adaptations, which are becoming a big deal in LoRA-style methods. For instance, LoRA variants often assign different ranks per layer based on task relevance; MAPO’s model-agnostic design makes that kind of per-layer customization harder.
The approach could also be valuable in centralized training, but that angle isn’t explored. (In line 136 the text says Figure 2 shows centralized MNIST training, yet the setup still uses federated algorithms.)

### Questions
Is there any reason MAPO uses whole flatten gradient for factorization? For MAPO, we can also divide whole flatten gradient into several parts and apply factorization for each part, and it would be a good direction to make flexibility.
What is the exact algorithm for MAPO fine-tuning? Is MAPO fine-tuning adding vec(BA) at each round? If so, does MAPO have a benefit over LoRA regarding flexibility (LoRA can be used as plug-in module)?

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
This paper proposes a novel federated learning method called Model Independent Projection Optimization (MAPO) to address the bottleneck problem of excessive communication overhead in federated learning. Unlike traditional layer-by-layer low-rank decomposition methods, MAPO reshapes and decomposes the gradient of the entire model into a fixed reconstruction matrix and a trainable projection vector, thereby freeing it from model architecture constraints and achieving a more flexible low-rank representation.

### Strengths
1. The proposed MAPO method fundamentally changes the paradigm of traditional low-rank decomposition, avoiding architectural constraints through model-level decomposition and providing a novel approach to communication efficiency optimization.

2. The paper provides rigorous theoretical analysis, including definitions of communication overhead rate and reconstruction error rate, as well as mathematical proofs and convergence analyses of MAPO decomposition, enhancing the credibility of the method.

### Weaknesses
1. Insufficient computational complexity analysis: While the paper mentions that MAPO is more efficient than layer-by-layer methods, the specific analysis of actual computational overhead is not in-depth enough, and there is a lack of practical performance evaluation on resource-constrained devices.

2. Inadequate discussion of hyperparameter sensitivity: The choice of parameter k has a significant impact on performance, but the paper does not provide guidelines for choosing the system's k value, which may increase the difficulty of tuning in practical deployments.

3. Insufficient security considerations: The paper does not discuss the potential security risks associated with the generation and synchronization of the dynamic reconstruction matrix in privacy-sensitive federated learning scenarios.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes MAPO (Model-Agnostic Projection Optimization), a method for improving communication efficiency in federated learning (FL). It aims to replace layer-wise low-rank or LoRA-style decompositions with a model-wide projection. Instead of constraining each layer’s gradients separately, MAPO reshapes the entire model gradient into a single matrix and factorizes it. The paper provides some FL based convergence analysis, and empirical evaluations on multiple datasets. The results show that MAPO achieves comparable accuracy to FedAvg while reducing communication cost.

### Strengths
1. The problem motivation is valid: communication efficiency is a critical bottleneck in large-scale FL.
2. The idea of global gradient projection using shared random seeds is simple and elegant; it generalizes many existing gradient compression approaches under a unified formulation.
3. The paper provides a formal convergence bound under standard smoothness and bounded-variance assumptions.
4. The experiments are broad, spanning image and text datasets with different architectures.

### Weaknesses
1. The novelty is extremely limited. MAPO essentially repackages global random projection with seed synchronization, ideas existing in one form or another in earlier methods. The contribution lies mainly in notation and aggregation formulation rather than a new algorithmic principle.
2. The comparison set is largely incomplete: several recent or highly relevant works are missing, such as FedEx-LoRA [1], SparseFL [2], FedRPCA [3], PRISM [4], Heterogeneous LoRA for Federated Fine-tuning [5], etc. These should have been discussed and compared empirically.
3. There is no rigorous analysis of computational cost. I understand that communication is an important bottleneck, but one cannot ignore the limited computational resources at the clients, and potentially at the server. It will be ideal to provide a rigorous analysis on the pareto front for computation-communication burden of the proposed scheme, with clear comparisons to prior works. The claim of efficiency is only on communication, while the proposed reshaping and projection steps add nontrivial local computation and memory overheads (e.g., matrix inversion or vector reshaping costs are ignored). 
4. The convergence theory is largely boilerplate, relying on standard FL proofs without any MAPO-specific insight (the same result could hold for many gradient projection schemes). If there are three layers in a neural network, how does MAPO converge differently from doing a projection for each layer separately?
5. The paper’s writing is verbose, and the presentation sometimes obscures what is genuinely new. For instance, the reshaping step is more of a mathematical convenience than a conceptual innovation.

[1] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Foundation Models. (ACL 2025)
[2] Revisiting Sparsity Hunting in Federated Learning: Why does Sparsity Consensus Matter? (TMLR 2023)
[3] FedRPCA: Enhancing Federated LoRA Aggregation Using Robust PCA. (arXiv)
[4] Overcoming Resource Constraints in Federated Learning: Large Models Can Be Trained with only Weak Clients. (TMLR 2023)
[5] Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models. (ACL 2024)

### Questions
Please address the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Model-Agnostic Projection Optimization (MAPO), a novel method for communication-efficient federated learning (FL). The core idea is to circumvent the limitations of traditional layer-wise low-rank decomposition. Instead of factorizing each layer's gradient, MAPO reshapes the entire model's gradient vector into a single matrix. This matrix is then factorized into a small, trainable projection vector ($B$) and a larger, non-trainable reconstruction matrix ($A$). The matrix $A$ is not transmitted; instead, it is regenerated on all clients and the server in each round using a shared random seed. The authors claim this approach is model-agnostic, offers more flexible control over the communication budget, and improves performance by exploring new random subspaces in each round. The paper provides a theoretical convergence analysis and extensive empirical results across various datasets and models to support its claims.

### Strengths
1. The central idea of reshaping the entire model gradient into a single matrix and applying a rank-1-like factorization is a conceptually simple yet novel departure from the prevalent layer-wise decomposition methods in FL (like LoRA variants for gradients). The use of a shared seed to synchronize a dynamically changing projection basis ($A$) without communication cost is a clever and effective mechanism.

2. The empirical results, if taken at face value, are highly significant. The method demonstrates substantial communication savings, reportedly achieving performance close to uncompressed FedAvg with orders of magnitude less communication (e.g., 98.9% of FedAvg's accuracy on CIFAR-10 with only 1.2% of the communication cost). This could represent a major step forward for practical, large-scale FL deployments.

3. The paper is well-written and the core idea is presented clearly, particularly with the aid of Figure 1 and Algorithm 1. The authors have conducted a comprehensive empirical evaluation, spanning a wide variety of datasets (vision, NLP), model architectures (CNN, Transformer, RoBERTa), and data distributions (non-IID). The inclusion of numerous appendices covering fine-tuning, complexity analysis, and ablation studies indicates a thorough investigation. The ablation study in Figure 7, which highlights the benefit of a "Fresh A" over a "Frozen A," provides strong support for the subspace exploration argument.

### Weaknesses
1. The "model-agnostic" reshaping, while simple, is heuristically motivated and lacks a strong theoretical foundation. It concatenates gradients from disparate layers (e.g., convolutional filters, attention heads, normalization parameters) which have fundamentally different structures, scales, and sensitivities. The paper provides no theoretical justification or empirical analysis as to why this aggressive homogenization is a principled approach. It is plausible that this could harm optimization by mixing unrelated gradient information, and the method's success might be more attributable to the sheer reduction in variance from projecting onto a random subspace rather than any inherent benefit of the reshaping itself.

2. The paper focuses heavily on communication efficiency but downplays the computational overhead introduced on the client side. During local training, each forward and backward pass requires reconstructing the model update from the projection, i.e., computing $\Delta W' = \text{vec}(B A^T)[0:d]$. This involves a matrix multiplication and reshaping operation within the local training loop for every batch. While Appendix K provides a theoretical complexity analysis, it is abstract. A practical wall-clock time comparison against baselines is conspicuously absent. For large models where $d$ is massive, even if $k$ is small, the dimension of $A$ ($\approx d/k$) can be very large, making the BA product a potential bottleneck that could negate the benefits in scenarios where computation, not just communication, is constrained.

3.The theoretical claims, particularly Proposition 3.6 regarding a memory reduction of $k^2$, are potentially misleading. The analysis in Appendix K correctly shows that the memory to store the factorization components ($A$ and $B$) is reduced. However, it ignores the dominant memory cost of the full-rank model parameters ($W$) and activations, which MAPO must maintain on the client. The method is communication-efficient, not parameter-efficient (unlike LoRA for fine-tuning). This crucial distinction is not made sufficiently clear in the main text, and the claim of significant memory reduction could be misinterpreted by readers.

### Questions
1. Could you provide wall-clock time comparisons for local client training (per round) between MAPO and the main baselines (FedAvg, FedLoRU, EvoFed)? This would clarify the practical computational trade-offs of your method.

2. The core "reshaping" mechanism treats all parameters equally. Have you investigated the effect of this homogenization? For instance, does normalizing gradients from different layers before concatenation affect performance? Is there any intuition for why mixing gradients from a shallow convolutional layer and a deep classification layer is beneficial?

3. Could you clarify the claim of $k^2$ memory reduction in Proposition 3.6 in the context of the overall client memory footprint? Given that the full model $W$ of size $d$ must be stored, what is the practical memory saving as a percentage of the total memory required to train the model on a client?

4. How sensitive is MAPO's performance to the initial random seed $r_t$? Does the choice of a "lucky" or "unlucky" random projection matrix $A_t$ in a given round lead to high variance in the training trajectory?

### Soundness
3

### Presentation
3

### Contribution
3
