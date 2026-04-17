# Towards Atoms of Large Language Models

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
The fundamental units of internal representations in large language models (LLMs) remain undefined, limiting further understanding of their mechanisms. Neurons or features are often regarded as such units, yet neurons suffer from polysemy, while features face concerns of unreliable reconstruction and instability. To address this issue, we propose the ***Atoms Theory***, which defines such units as atoms. We introduce the atomic inner product (AIP) to correct representation shifting, formally define atoms, and prove the conditions that atoms satisfy the Restricted Isometry Property (RIP), ensuring stable sparse representations over atom set and linking to compressed sensing. Under stronger conditions, we further establish the uniqueness and exact $\ell_1$ recoverability of the sparse representations, and provide guarantees that single-layer sparse autoencoders (SAEs) with threshold activations can reliably identify the atoms. To validate the Atoms Theory, we train threshold-activated SAEs on Gemma2-2B, Gemma2-9B, and Llama3.1-8B, achieving 99.9\% sparse reconstruction across layers on average, and more than 99.8\% of atoms satisfy the uniqueness condition, compared to 0.5\% for neurons and 68.2\% for features, showing that atoms more faithfully capture intrinsic representations of LLMs. Scaling experiments further reveal the link between SAEs size and recovery capacity. Overall, this work systematically introduces and validates Atoms Theory of LLMs, providing both a theoretical framework for understanding internal representations and a foundation for mechanistic interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes the Atom Theory, a theoretical framework defines atoms as the fundamental units of LLMs. Authors present atomic inner product as a replacement of Euclidean inner product in examining orthogonality to correct representation shifting. They prove the conditions under which atoms satisfy the Restricted Isometry Property to ensure the uniqueness and recoverability of sparse representations. They also prove threshold-activation SAEs can effectively achieve near perfect sparse reconstruction, and scale the method on up to 9B language models.

### Strengths
1. The paper is well-written and the theoretical framework and proofs are rigorous. Intuition behind the theoretical statements are clear.
2. The Atom Theory gives rigorous definition of fundamental units of LLMs and various properties they should fulfill, supplementing previous SAE studies with solid theoretical groundings.

### Weaknesses
1. The explicit form of the Atomic Inner Product in Theorem 2 $d_i^TSd_j=c^2d_i^T(DD^T)d_j$ is a solution to the linear least square problem $\min_S||c^4I-D^TSD||$, and will naturally produce near zero values in off-diagonal elements. Thus, the significance of the results in Figure 3 (correcting representation shifting) is doubtful.
2. The idea of SAEs with threshold activations (i.e. JumpReLU SAEs) is not novel and proposed by Rajamanoharan et al. before. The results of high sparsity and near perfect reconstructions contradict the previous studies.

### Questions
1. (As in Weaknesses 1) Can random vectors and ReLU SAE features serve as atoms that fulfill the Atomic Inner Product or the $\epsilon$-Approximately Orthogonal Atoms?
2. (As in Weaknesses 2) Can the authors explain the difference of performance between the original JumpReLU paper and this work? Does the difference come from different activation collecting method (natural text corpus versus knowledge-atomization tasks)?

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
This paper proposes the Atoms Theory as a formal framework for identifying fundamental representational units—termed “atoms”—within large language models (LLMs). The authors introduce the concept of the Atomic Inner Product (AIP) to correct geometric distortions in representation space and prove that under certain conditions, these atoms satisfy the Restricted Isometry Property (RIP), ensuring stable sparse recovery. They further show that single-layer sparse autoencoders (SAEs) with threshold activations can theoretically identify such atoms. Empirical experiments are conducted on Gemma2 and Llama3.1 models to validate reconstruction fidelity and atomicity, claiming that atoms provide a more stable and unique representation than neurons or features.

### Strengths
- The theoretical part (Section 3) is systematically developed, with clear connections to compressed sensing and RIP theory.
- The proofs are complete and rigorous, giving the work a solid mathematical foundation.
- The motivation—defining fundamental representational units of LLMs—is important and aligns with ongoing efforts in mechanistic interpretability.

### Weaknesses
- The threshold-activated SAE is functionally equivalent to JumpReLU SAE, offering no methodological novelty.
- The reconstruction experiments are conducted on a dataset orders of magnitude smaller than those used in GemmaScope and LlamaScope as baseline  (tens of thousands vs. hundreds of millions).
- The comparison of reconstruction scores is therefore unfair and misleading, as performance differences can be entirely attributed to dataset scale rather than model quality.
- The claimed empirical validation of “Atoms Theory” is thus not credible given the limited and mismatched setup. The work overstates its practical impact despite weak experimental support.

### Questions
- How is the proposed threshold-activated SAE fundamentally different from the JumpReLU SAE? If they are equivalent, what justifies reintroducing it under a new name?
- Why were the experiments conducted on such a small dataset (only tens of thousands of activations)? Can the authors demonstrate that their method scales to hundreds of millions or billions of activations like the compared baselines?
- Would the conclusions about reconstruction fidelity and atomicity still hold if the models were trained and evaluated on the same-scale datasets as GemmaScope or LlamaScope?
- Can the authors provide ablation or control experiments to verify that the improvements are not due to scale or preprocessing differences?

### Soundness
1

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
The paper proposes the Atoms Theory and studies the fundamental units of internal representations of LLMs. Previously, neurons or features were regarded as such units, but the paper argues that they are insufficient since (1) neurons often exhibit substantial polysemy, growing doubt on their validity for analysis and (2) features face concerns of unreliable reconstruction and instability. This paper introduces methodologies to reliably identify indivisible, fundamental units, "atoms", of LLMs, to provide a rigorous theoretical foundation for mechanistic interpretability.

### Strengths
- The paper studies a well-known and important problem in the literature: the polysemy of neurons and the reconstruction unreliability of standard SAE features.

- The claims are theoretically supported with rigorous analysis. 

- Empirically, the proposal framework is better than baselines on Gemma 2B, Gemma 9B, and Llama3 8B.

### Weaknesses
- What is the computational cost of obtaining the empirical results in the paper? Training specialized SAEs for every layer of LLMs would be too costly. Are there any computational tricks the authors are using? I suggest discussing the computational cost in more detail in the paper.

- The theoretical guarantees rely on the assumption that LLM representations are inherently extremely sparse. Do the authors have an understanding of how much of these results hold if the model employs dense representations, e.g., for complex tasks?

### Questions
See Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3
