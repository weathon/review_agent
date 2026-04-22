# SEQR: Secure and Efficient QR-based LoRA Routing

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Low-Rank Adaptation (LoRA) has become a standard technique for parameter-efficient fine-tuning of large language models, enabling large libraries of LoRAs, each for a specific task or domain. Efficiently selecting the correct LoRA adapter for a given input remains a challenge, particularly in secure environments where supervised training of routers may raise privacy concerns. Motivated by previous approaches, we formalize the goal of unsupervised LoRA routing in terms of activation norm maximization, providing a theoretical framework for analysis. We demonstrate the discriminative power of activation norms and introduce SEQR, an unsupervised LoRA routing algorithm designed to maximize efficiency while providing strict routing guarantees. SEQR provably identifies the norm-maximizing adapter with significantly greater efficiency, making it a highly scalable and effective solution for dynamic LoRA composition. We validate our results through experiments that demonstrate improved multi-task performance and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SEQR, an unsupervised LoRA routing method designed for efficiency and security. The authors formalize the goal of unsupervised routing as activation norm-maximization, providing a theoretical framework to analyze existing methods and their proposed solution. SEQR leverages a shared, frozen A matrix and QR decomposition to provably select the norm-maximizing adapter with significantly lower computational cost. The claims are supported by theoretical proofs and empirical experiments demonstrating competitive multi-task performance and superior efficiency.

### Strengths
1. The paper's formalization of unsupervised routing as an activation norm-maximization problem provides a clear and valuable framework. The subsequent theoretical analysis, including proofs that SPECTR and SEQR are norm-maximizing while ARROW is not, adds significant rigor and clarifies the behaviors of these methods.

2. SEQR demonstrates a significant improvement in computational and storage efficiency over existing methods that offer similar routing guarantees, as shown in the complexity analysis and empirical measurements. This makes the method highly practical for real-world scenarios involving large libraries of LoRA adapters, directly addressing a key challenge in the field.

### Weaknesses
1. The entire SEQR framework is built upon the assumption that a single, frozen, randomly initialized A matrix is sufficient for training diverse adapters. While the paper shows this holds for the tested tasks, this constraint might limit performance when fine-tuning for highly dissimilar or complex tasks where a unique A matrix could capture important task-specific input transformations.

2. The paper posits activation norm-maximization as the primary goal for unsupervised routing. While intuitive and empirically supported, this objective may not be universally optimal. It is possible for an adapter to produce the highest activation norm without yielding the best task-specific performance, especially in cases of model miscalibration or overfitting.

3. SEQR requires an offline calibration step that computes and stores the mean (µ_i) and standard deviation (σ_i) of activation norms from each adapter's training data. While this avoids training a cross-silo router, these statistics themselves are derivatives of the private data and could potentially leak information about the data distribution.

4. According to Table 3 and footnote 5, the average performance improvement of SEQR over LAG (k=3) is small (93.5 vs 92.9), and the difference is not statistically significant (p=0.096). This suggests that SEQR's primary advantage is its efficiency, not necessarily superior task performance over a well-configured LAG.

5. The paper compares SEQR (which requires a shared A) against baselines like ARROW and LAG. It is unclear if these baselines were evaluated using adapters with unique A matrices (their standard formulation) or shared A matrices in the final performance table (Table 3), which impacts the fairness of the storage and computational comparisons.

### Questions
Please see weaknesses.

### Soundness
2

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
5

### Summary
In secure environments, supervised router training may lead to privacy leakage. To address this, we propose SEQR, an efficient unsupervised LoRA routing algorithm. Theoretically, SEQR can accurately identify the norm-maximizing adapter with high efficiency, making it a scalable and effective solution for dynamic LoRA composition. Extensive experiments verify the rationality and effectiveness of the proposed method.

### Strengths
This paper is theoretically sound, well-motivated, and proposes a simple yet effective method.

### Weaknesses
Although the theoretical motivation of this method is reasonable, several questions remain:

1. While this is an unsupervised LoRA routing algorithm, supervised routing methods might serve as an upper performance bound — it would be helpful to include such results for reference.
2. The datasets used in the experiments appear somewhat outdated; testing on more recent datasets commonly used for LLM evaluation and comparing with results from LoRA-only fine-tuning would strengthen the work.
3. It is recommended to provide results on a wider range of large language models (LLMs) to further demonstrate the generality of the proposed method.

### Questions
This is not a critical issue to address, and adding new experiments might be difficult during the rebuttal stage. However, providing relevant results would make the work more convincing.

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
This paper formalizes unsupervised LoRA routing as an activation norm maximization problem and proves that SPECTR is equivalent to directly maximizing this norm. Under the setting of a shared and frozen A, the paper proposes SEQR, which applies QR decomposition and spatial comparison to each B_i. This approach maintains the strict optimal routing guarantee while reducing computational complexity.

### Strengths
1.This paper explicitly defines unsupervised routing as an optimization problem of maximizing the activation norm within a LoRA library, providing a unified analytical baseline for different methods.
2.The proposed algorithm offers significant efficiency advantages, requiring only r*r multiplication for inference, thereby lowering algorithmic complexity. Experiments demonstrate that under typical scales, routing FLOPs are reduced from the million-level to 64K.

### Weaknesses
1.The equivalence and efficiency of SEQR depend on the assumption that "all adapters share the same frozen A matrix." Does the method become inapplicable in scenarios where A is trained independently for each task or subsequently fine-tuned?
2.The paper's experimental scope is limited, evaluating only on a relatively small base model (Llama-3.2-3B) and classification tasks. Evaluations on other types or scales of LLMs are missing.
3.The paper's motivation emphasizes security and privacy, but SEQR only associates this with being unsupervised and sharing matrix A. Its core algorithmic innovation (i.e., QR decomposition) does not intrinsically provide additional security or privacy guarantees, which does not seem to fully support this claim.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SEQR, an unsupervised routing algorithm for selecting LoRA adapters based on activation norm maximization. The authors formalize unsupervised LoRA routing as the problem of efficiently identifying the adapter that maximizes the activation norm. SEQR targets the shared-A setting and precomputes a reduced QR decomposition, achieving the same norm-maximization guarantee as SpectR but with substantially lower complexity. Experiments show the benefits of the proposed method.

### Strengths
1.	Clear problem formalization: Casting unsupervised LoRA routing as activation norm maximization provides a coherent lens to compare methods and reason about guarantees. 
2.	The method is simple yet effective to follow.

### Weaknesses
1.	Incremental novelty: Both building blocks—SVD/QR factorization of LoRA variants and the Shared A regime—are established ideas in the LoRA literature and in modular LoRA routing. SEQR primarily combines “known” linear-algebraic decompositions with the shared-A assumption and applies them to norm-maximizing zero-shot routing. While the formalization is useful and the efficiency result is practical, the conceptual leap beyond these ingredients is modest.
2.	Narrow baselines and single-family backbone: This paper evaluates primarily on a single Llama-3.2-3B-Instruct model and compares with works from the same series. This limits the external validity and makes it hard to assess robustness across model families, sizes, and tasks beyond the included classification suite.
3.	Limited related work: The related work section lacks a broader overview of LoRA variants and composition methods, which would better situate SEQR within the ecosystem.
4.	Ablations are insufficient: The offline calibration step is central under shared-A but lacks robustness ablations (distribution shift, noisy μ/σ, per-layer vs aggregated statistics). The paper validates how hidden dimension, number of adapters, and LoRA rank affect routing efficiency, but does not show how these factors impact task performance.

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
