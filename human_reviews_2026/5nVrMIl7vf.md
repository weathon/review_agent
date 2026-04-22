# Continual Fine-Tuning with Provably Accurate and Parameter-Free Task Retrieval

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
Continual fine-tuning aims to adapt a pre-trained backbone to new tasks sequentially while preserving performance on earlier tasks whose data are no longer available. Existing approaches fall into two categories which include input- and parameter-adaptation. Input-adaptation methods rely on retrieving the most relevant prompts at test time, but require continuously learning a retrieval function that is prone to forgetting. Parameter-adaptation methods instead use a fixed input embedding function to enable retrieval-free prediction and avoid forgetting, but sacrifice representation adaptability. To combine their best strengths, we propose a new parameter-adaptation method that enables adaptive use of input embeddings during test time with parameter-free retrieval. We derive task-retrieval error bounds for a clustering-based, parameter-free paradigm, providing theoretical guarantees that link low retrieval error to structural properties of task-specific representation clusters, revealing a fresh insight into how well-organized clustering structure will enable reliable retrieval. Motivated by this insight, our method is designed with two key components: (i) an adaptive module composition strategy that learns informative task-specific updates to preserve and complement prior knowledge, and (ii) a clustering-based retrieval mechanism that captures distinct representation signatures for each task, enabling adaptive representation use at test time. Extensive experiments show that these components work synergistically to improve retrieval and predictive performance under large shifts in task semantics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces PROTEUS, a continual fine-tuning framework that achieves provably accurate, parameter-free task retrieval. It bridges the gap between input-adaptation and parameter-adaptation methods by combining adaptability in test-time representation with retrieval-free prediction. The method derives theoretical bounds linking retrieval accuracy to the clustering structure of task-specific representations, ensuring low retrieval error through well-separated clusters. PROTEUS incorporates an adaptive fine-tuning mechanism that promotes orthogonal and informative updates across tasks, preserving prior knowledge while improving discrimination among task embeddings.

### Strengths
1. The paper introduces a parameter-adaptation method that uniquely combines adaptive use of input embeddings with parameter-free task retrieval, bridging the gap between input-adaptation and parameter-adaptation paradigms.

2. The authors provide a theoretical analysis that formally connects retrieval accuracy to structural properties of task-specific representation clusters. This foundation is concretely realized through the adaptive module composition strategy, which enforces orthogonality and selective knowledge transfer, ensuring both preservation and complementarity of prior knowledge.

3. The clustering-based retrieval mechanism offers a clear, interpretable, and scalable solution for adaptive test-time inference, improving retrieval stability and reducing forgetting.

### Weaknesses
1. As noted, the method generates a distinct LoRA update for each task, which can become memory-intensive when scaling to long sequences or high-dimensional backbones. Although the authors argue that LoRA units are lightweight, the cumulative storage of adaptation modules and their associated signature distributions may still pose practical challenges.

2. The Gaussian clustering assumption is a somewhat strong theoretical simplification in the paper. In reality, feature distributions might be non-Gaussian or overlap between tasks, violating the conditions under which PROTEUS can provably bound the retrieval error.

### Questions
1. The paper presents results primarily on 10-task sequences (e.g., Split CIFAR-100, ImageNet-R, and VTAB benchmarks), but continual fine-tuning is especially challenging when the number of tasks grows larger. It would be helpful if the authors could provide or discuss results for shorter (e.g., 5) and longer (e.g., 20 or 50) task sequences, to better understand how PROTEUS scales in performance, retrieval accuracy, and memory usage as the task sequence length increases.

2. Could the authors consider introducing a fixed memory budget and examining how performance degrades under such constraints?

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
4

### Summary
This paper addresses the challenge of catastrophic forgetting in continual fine-tuning (CFT), specifically tackling the limitations of existing input-adaptation and parameter-adaptation methods. The authors argue that input-adaptation methods (e.g., prompt retrieval) suffer from "retriever forgetting" , while parameter-adaptation methods (e.g., RanPAC) lack representation adaptability at test time

### Strengths
1. Clearly identifies the problem of retriever forgetting in prompt/parameter-pool based CFT methods and proposes a novel parameter-free retrieval mechanism as a direct solution. 

2. Theoretical Foundation: Provides a non-trivial theoretical analysis linking the retrieval error rate to geometric properties (cluster separation factor $\delta$) of the learned representation signatures. This offers valuable insight and principled guidance for the algorithmic design. Theorem 3.4 and 3.5 are significant.

3. Achieves SOTA performance on several challenging CL benchmarks, including those with large semantic gaps like VTAB , significantly outperforming both prompt-based and other LoRA-based CL methods.

### Weaknesses
1. High System Complexity: The overall PROTEUS framework is quite intricate, involving adaptive LoRA with orthogonality constraints, non-parametric GMM fitting (DP-GMM) for potentially many components per task, storing these GMM parameters as signatures, computing likelihoods against all signatures during retrieval, and finally performing LDA prediction. This complexity raises concerns about implementation difficulty, computational overhead (especially GMM fitting and retrieval), and potential fragility.

2. Insufficient Ablation of Design Choices: While key components (retrieval, adaptive FT) are ablated, the paper could benefit from more fine-grained ablations. For example, why is DP-GMM necessary? Would simpler clustering (e.g., k-means for centroids) or simpler signatures (e.g., just means) suffice? Is the combination of $l_1/l_2$ regularization 95and orthogonality 96 both necessary in the LoRA update? Justifying the necessity of each complex piece versus simpler alternatives would strengthen the paper.

3. Reliance on Distributional Assumptions: The method relies on the assumption that task-specific embeddings $h_k(x)$ can be well-modeled by GMMs. The theoretical bounds (Assumption A.1) also depend on this. While GMMs are flexible, the quality of fit and the resulting cluster separation might degrade if the true embedding distributions are highly non-Gaussian or have complex structures, potentially impacting retrieval accuracy and invalidating the theoretical guarantees

### Questions
* Can the authors comment on the practical computational overhead of the DP-GMM fitting process during training and, more importantly, the signature matching process during inference, especially as the number of tasks $m$ grows large (e.g., hundreds or thousands)?

* How robust is the GMM fitting and retrieval performance if the underlying embedding distributions deviate significantly from Gaussian mixtures? Have the authors explored alternative, potentially more robust, signature representations?

* The adaptive LoRA update combines knowledge transfer ($S_\tau$) and orthogonal new directions ($\Delta\omega_{k+1}^\perp$). Could the authors provide ablation studies isolating the effects of just the orthogonality constraint versus just the knowledge transfer component (compared to the baseline without either)?

* Theorem 3.5 suggests error can be made arbitrarily low if $\delta$ is large enough. How large does $\delta$ typically become in practice with the proposed adaptive fine-tuning? Does it continue to grow sufficiently as $m$ increases, or does it saturate, potentially limiting performance on very long sequences? (Figure 2 provides some insight for CIFAR-100, but more analysis would be helpful).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces PROTEUS, a method for parameter-based continual fine-tuning (CFT) that improves parameter-free retrieval. The method has two synergistic parts. First, an "Adaptive Knowledge Composition" strategy trains new LoRA adapters to be a selective combination of past adapters plus a new, orthogonal component for task-specific knowledge. Second, this orthogonal training creates highly separated representation clusters, which are captured by a "Parameter-Free Retrieval" mechanism (a DP-GMM). At test time, an input is assigned to the task with the highest likelihood—a non-learnable lookup. The paper provides theoretical bounds linking this cluster separation to low retrieval error and shows SOTA empirical results, demonstrating superior retrieval and predictive performance.

### Strengths
1. The paper provides a novel application of Gaussian Mlixutre Models on CF. Improving prior state of the art in CFT with LoRA-based adapters. 
2. The paper provides a solid theoretical backing for its approach.
3. The paper provides empirical evidence of the consistently higher performance of their method compared to previous state of the art across a diverse set of datasets.

### Weaknesses
1. The paper claims to alleviate key issues with lack of representation adaptability and forgetting in retrieval-based methods. However, the latter is a prominent problem in prompt-based methods, not in parametric CFT methods (as PROTEUS) and previous parametric CFT approaches (RanPAC, InfLoRA, SD-LoRA) already address this with parameter-free retrieval methods.

### Questions
1. "Provably Accurate" is a term consistently mentioned in the paper but not properly introduced. Adding reference when mentioned for the first time would strengthen the readability of the paper.
2. I understand that at test time the input embedding for unseen samples has to go through each of the LoRA-augmented architectures in contrast to previous methods that used the first or last adaptation. How does the inference latency scale as we increase the number of tasks compared to the other parametric CFT baselines ?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors examine the strengths and weaknesses of input-adaptation (prompt-based) and parameter-adaptation (LoRA-based) approaches for continual fine-tuning of pre-trained models. Input-adaptation enables flexible test-time behavior but suffers from retriever forgetting, while parameter-adaptation is stable but lacks adaptability. To bridge this gap, they propose PROTEUS, a parameter-free, provably accurate task-retrieval framework that learns orthogonal low-rank adapters and retrieves the most relevant one via statistical matching of task-specific signature distributions. Theoretical analysis provides error bounds linking retrieval accuracy to cluster separation, and experiments on benchmarks such as CIFAR-100, ImageNet-R/A, and VTAB-5T show that PROTEUS achieves state-of-the-art performance with improved adaptability and minimal forgetting.

### Strengths
- The authors try to mathematically justify the proposed approach
- The proposed method outperforms the baselines

### Weaknesses
- The paper is poorly written and difficult to follow, even after multiple readings. It would benefit from substantial restructuring, rewriting, and clearer explanations throughout.

	- The distinction between input-adaptation and parameter-adaptation is unclear. From the paper, it seems that input-adaptation corresponds to prompt-tuning and parameter-adaptation to LoRA-based fine-tuning, but these categories are only briefly mentioned in the abstract and never clearly defined or elaborated upon in the main text.
	- The authors state that input-adaptation relies on retrieving relevant prompts at test time, whereas parameter-adaptation uses a fixed embedding function, is retrieval-free, and avoids forgetting. However, it is not well explained why these approaches necessarily have these characteristics. It seems the intended distinction is that one requires identifying task-specific parameters at test time while the other does not. If that is the case, it would be clearer to use standard terminology such as task ID and task-specific parameters instead.
	- Moreover, the claim that only input-adaptation requires test-time task retrieval while parameter-adaptation does not is too rigid. Whether retrieval is required depends on how task-specific parameters are constructed, not inherently on whether the method is input- or parameter-based.
	- The paper’s terminology is often confusing and inconsistent, making it hard to understand the authors’ intent. For example, the phrase "Solution Vision" in the "Existing Literature" section is ambiguous. It appears to mean "proposed solution." Similarly, terms like "signature pattern," "nearest signature," and "signature distribution" are introduced without clear definitions. Such vague or unconventional terminology significantly reduces the readability and clarity of the paper.

### Questions
Refer to my comments in weaknesses

### Soundness
2

### Presentation
1

### Contribution
2
