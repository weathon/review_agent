# FLoRG: Federated Fine-tuning with Low-rank Gram Matrices and Procrustes Alignment

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Parameter-efficient fine-tuning techniques such as low-rank adaptation (LoRA) enable large language models (LLMs) to adapt to downstream tasks efficiently. Federated learning (FL) further facilitates this process by enabling collaborative fine-tuning across distributed clients without sharing private data. However, the use of two separate low-rank matrices in LoRA for federated fine-tuning introduces two types of challenges. First, aggregation error can arise from separately aggregating the two low-rank matrices.
Second, even if the server aggregates the product of two low-rank matrices, it needs to decompose the aggregated matrix back into low-rank matrices. Since the decomposition is not unique, it can lead to decomposition drift. To tackle the aforementioned challenges, we propose federated low-rank Gram-matrix aggregation (FLoRG), a federated fine-tuning framework which employs a single low-rank matrix for fine-tuning and aggregates its Gram matrix (i.e., the matrix of inner products of its column vectors). FLoRG can eliminate the aggregation error and reduce the communication overhead. It also minimizes the decomposition drift by introducing a Procrustes alignment approach which aligns the decomposed matrix between consecutive fine-tuning rounds for consistent updates. We theoretically analyze the convergence of FLoRG and prove that adopting the Procrustes alignment results in a tighter convergence bound. Experimental results across multiple LLM fine-tuning benchmarks demonstrate that FLoRG outperforms five state-of-the-art baseline schemes by providing higher downstream task accuracy and can reduce the communication overhead by up to 2041$\times$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes FLoRG, a novel federated fine-tuning framework for large language models (LLMs) that improves upon the conventional LoRA-based approach. Traditional federated LoRA uses two separate low-rank matrices, which can lead to aggregation errors and decomposition drift during global model updates. To address these issues, FLoRG employs a single low-rank matrix and aggregates its Gram matrix to eliminate aggregation errors and reduce communication costs. Additionally, the authors introduce a Procrustes alignment mechanism to align decomposed matrices between rounds, thereby mitigating decomposition drift. Theoretical analysis demonstrates that this design achieves a tighter convergence bound. Extensive experiments on multiple LLM fine-tuning benchmarks show that FLoRG achieves higher downstream task accuracy and reduces communication overhead by up to 82% compared to state-of-the-art baselines.

### Strengths
1. The authors provide a theoretical convergence analysis and rigorously show that the proposed Procrustes alignment leads to a tighter convergence bound, which enhances the credibility of the method.
2. Experimental results on multiple LLM fine-tuning benchmarks demonstrate consistent improvements over several state-of-the-art baselines in both accuracy and communication efficiency.
3. The paper is well written and easy to follow.

### Weaknesses
1. The experiments are limited to natural language understanding tasks with relatively small models. It would strengthen the paper to include evaluations on larger models or additional task types to demonstrate broader applicability.
2. Updating only the low-rank matrix  $A$ may limit the model’s representation capability, potentially constraining its ability to adapt to more complex tasks.
3. The paper lacks an analysis of efficiency in terms of computational cost, memory usage, or communication overhead.

### Questions
1. How does the proposed method perform on more general tasks such as question answering and dialogue?
2. The performance of the proposed method when applied to more popular and larger language models, such as LLaMA?
3. The local training stage of the proposed method is interesting. It seems applicable to centralized learning as well. How does it perform under centralized learning? Compared with centralized learning, what advantages does it provide in the FL setting?
4. It is unclear whether updating only module A is sufficient to learn good representations of the client data. Could the authors clarify or provide evidence?
5. Could the authors provide details on how the dataset was used?

If the authors can adequately address my concerns, I am willing to increase my rating.

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
3

### Summary
FLoRG replaces the usual LoRA two-factor update BA with a single low-rank matrix A and updates the model via a Gram matrix aggregation. Clients locally SGD-update A; the server linearly aggregates n∑An⊤An, then eigendecomposes the aggregated Gram and applies a Procrustes alignment to pick a decomposition closest to the previous A, which stabilizes directions and enforces a target rank. The authors prove a non-convex convergence bound in which Procrustes alignment cancels a “drift” term, and empirically report higher GLUE accuracy vs. FedIT/FeDeRA/FFA-LoRA/FedSA-LoRA with up to 82% fewer transmitted parameters to reach target accuracy.

### Strengths
1. Bias-free aggregation with one matrix.
2. Convergence bound tightens when alignment is used; ablations show sizeable accuracy gains from Procrustes; headline comms savings to target accuracy.

### Weaknesses
1. The approach relies on semi-orthogonal L,R that never update; performance is sensitive to their initialization.
2. Each round per layer requires eigendecomposition of Q and an SVD for Procrustes; scalability or latency with many layers or clients isn’t benchmarked.

### Questions
1. How robust is FLoRG if L,R are learned (slowly) or adapted per layer/round? Can you provide theory/ablation for updating L,R versus keeping them fixed?
2. What are per-round costs of eigendecomposition + Procrustes across all LoRA layers at N>100 clients?

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
5

### Summary
This paper proposes FLoRG, a framework for federated fine-tuning of LLMs using Low-rank Adaptation. The authors point out two primary challenges with existing federated LoRA methods:  Aggregation Error (caused by naively aggregating the LoRA matrices B and A separately), Decomposition Drift (caused by the fact that there is not unique decomposition matrix) 
The authors tackle these challenges with a two-part solution:

* Gram Matrix Aggregation: Instead of LoRA's two matrices (B,A), FLoRG uses a single trainable low-rank matrix A and utilizes existing linear algebra techniques to convert the A matrix to the original dimension of the $\Delta W$. 

* Procrustes Alignment: The server decomposes the aggregated weights, it performs a Procrustes alignment step. This solves an optimization problem to find an orthogonal matrix that best aligns the new A~t+1 with the matrix from the previous round At, thereby minimizing the "decomposition drift".

The paper provides a theoretical convergence analysis showing that the Procrustes alignment step results in a tighter convergence bound. The authors also empirically show that their method outperform four baselines on GLUE benchmarks.

### Strengths
* The paper is well-written. The authors did a good job categorising and explaining the existing problems. 
* The algorithm performs better than the mentioned baselines. 
* The authors provide a convergence analysis for FLoRG.

### Weaknesses
* Clarity on Communication Saving. I would appreciate it if the authors explained the communication saving part of their claim. Did they measure the communication compared to full matrix communication or other Federated LoRA methods?

* Server-Side Computational Overhead: The paper does not discuss the server-side computational cost, which appears to be substantial, especially doing matrix decomposition and solving optimization. 

* The baselines are considerably basic. By just checking recent ACL and ICML conferences, I found recently accepted papers on Federated LoRA. The merits of the paper is not clear for me considering it is missing several works.

### Questions
* Information about the setting is missing. For example, what is the parameter for different levels of heterogeneity? How did you do LDA for datasets without labels? 

* What is the federated learning setting, how many clients participate each round?

* Did you do hyperparameter search? 

* Are the results averaged for different random seeds or they are done only for one seed?

Please also check the weakness section.

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
This paper addresses the issue that the federated aggregation of LoRA updates may not accurately reflect the intended global aggregation result. To this end, the authors propose FLoRG, which employs a single low-rank matrix for fine-tuning and aggregates its Gram matrix. Extensive experiments demonstrate its superiority over the existing works.

### Strengths
1. The paper presents a well-structured theoretical analysis with formal proofs, offering strong theoretical soundness and clear convergence guarantees.
2. It explores an interesting and under-studied problem—eliminating aggregation bias and decomposition drift in federated LoRA fine-tuning—introducing new insights into parameter-efficient federated learning.
3. The paper is clearly written, correctly annotated, and provides a thorough description of the proposed FLoRG framework, making it easy to follow and reproducible.

### Weaknesses
1. The paper does not address the partial client participation scenario, which is common in practical federated learning settings. Evaluating FLoRG under varying client availability would strengthen its applicability.
2. The experiments are conducted only on OPT-125M and RoBERTa-large, which are relatively dated compared to current state-of-the-art LLMs such as LLaMA-3 and Qwen-2.5. Using more recent backbones would better demonstrate the scalability and relevance of FLoRG.
3. The paper reports final accuracy and communication cost but does not include convergence curves showing performance versus communication rounds. Such a figure would provide clearer insights into the training dynamics and stability of FLoRG compared with baselines.

### Questions
**See weaknesses.**

### Soundness
3

### Presentation
3

### Contribution
3
