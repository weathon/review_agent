# Accurate and Efficient Singular Value Decomposition For LLMs via Decay-aware Rank Allocation and Feature-Preserved Weight Update

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Singular Value Decomposition (SVD) provides a hardware-agnostic and effective paradigm for compressing and accelerating Large Language Models (LLMs) by decomposing and truncating weight matrices, followed by weight updates to restore accuracy. However, SVD-based compression faces two major challenges:**(1) Rank Selection Problem:** Optimizing truncation and update ranks constitutes a high-dimensional combinatorial problem. Existing solutions rely on computationally expensive search, leading to both suboptimal performance and diminished efficiency. **(2) Limited Accuracy Restoration:** The sequential weight update strategy employed by state-of-the-art approaches (e.g., SVD-LLM) results in Hessian anisotropic, which hampers accuracy recovery and slows convergence. To overcome these, we introduce DF-SVD, which integrates: **(1) Decay-Aware Rank Allocation:** We derive and validate a correlation between decay characteristics of each weight's singular value spectrum and its importance. This enables dynamic, layer- and weight-specific rank allocation, ensuring high fidelity without costly search. **(2) Feature-Preserved Weight Update:** We introduce a theoretically grounded update strategy that fixes the truncated weight matrix $V^{\top}S^{-1}$ along with the principal components of $U\Sigma$,  while updating only the minor components. This design ensures Hessian isotropic, achieving superior accuracy restoration and faster convergence. DF-SVD not only significantly outperforms baselines in accuracy, but also completing compression in just 30 minutes, achieving speedups of $7\times$, $11\times$ and $16\times$ compared to SVD-LLM, ASVD and Dobi-SVD respectively. DF-SVD directly correlates the singular spectrum with training-free rank selection and boosts Hessian isotropy, paving the way for a new paradigm in accurate and efficient SVD-based LLM compression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DF-SVD, a method for compressing LLM using SVD with two main contributions: (1) decay-aware rank allocation that dynamically determines truncation and update ranks based on singular value decay, and (2) feature-preserved weight updates that achieve isotropic Hessian by fixing V^TS^{-1} and selectively updating only minor components of UΣ. The method achieves much speedup over existing SVD-based methods while maintaining or improving accuracy.

### Strengths
1. Strong exp results and practical speedup. 
2. Sound theoretical analysis: The Hessian conditioning analysis (section 3.2) is mathematically sound and provides clear intuition for why the proposed reformulation can achieve better convergence properties

### Weaknesses
1. Limited novelty relative to SVD-LLM: The paper heavily builds on SVD-LLM's foundation (Cholesky whitening, sequential optimization framework, experimental setup). Much of the methodology is inherited, making this more of an incremental improvement.
2. Missing critical comparisons:
A. No comparison with AdaLoRA: The paper cites AdaLoRA for importance-based rank allocation but never compares against it
B. No comparison with other methods: Methods like "Dynamic Low-rank Estimation for Transformer-based Language Models" (Hua et al., EMNLP 2023 findings) are highly relevant but not discussed or compared
3. No empirical validation that decay coefficient actually correlates with ground-truth importance (e.g., gradient magnitudes, ablation impact)

### Questions
1. Can you show via experiments that Hessian isotropy causes the speedup (e.g., via iteration counts, convergence curves)?
2. Why not compare with AdaLoRA, which you cite as inspiration?
3. What is the correlation between λ_norm and ground-truth importance metrics (gradients, sensitivity)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the DF-SVD framework, which aims to simultaneously improve accuracy recovery and compression efficiency in large language models. The authors first identify two key issues with conventional SVD-based compression methods: difficulty in selecting appropriate truncation and update ranks, and limited fine-tuning stability. To address these challenges, the paper proposes two core modules. The first, Decay-Aware Rank Allocation, models the singular value decay rate of each layer’s weight matrix to dynamically determine both truncation and update ranks, achieving adaptive compression across layers and matrices. The second, Feature-Preserved Weight Update, freezes the dominant components and only updates the minor subspace, thereby preserving critical pretrained features while improving the isotropy of the Hessian for faster convergence. Experimental results show that DF-SVD consistently outperforms existing methods such as SVD-LLM, ASVD, and Dobi-SVD on LLaMA, LLaMA2, LLaMA3, and OPT models under 30–60% compression, achieving comparable accuracy with 7–16× faster end-to-end compression.

### Strengths
1. Clear and practical implementation design.
The paper follows the SVD-LLM pipeline with whitening (via Cholesky decomposition) and SVD pre-processing, while confining its innovation to rank allocation and the update subspace. This design choice makes the method easy to reproduce, integrate, and deploy in real-world model compression workflows.

2. Comprehensive experimental evaluation.
The experiments cover multiple model families (LLaMA and OPT) and diverse datasets, and report both accuracy and end-to-end compression time. The study also compares DF-SVD against pruning and quantization methods, demonstrating its compatibility and potential for combined use.

### Weaknesses
1. Limited novelty.
The paper’s motivation—improving rank selection and reducing update time—targets a well-studied problem. While the proposed approach is practical, it appears relatively straightforward and lacks deeper theoretical innovation. For example, using singular value decay as a heuristic for rank allocation is intuitive but overlooks inter-layer importance differences; in practice, some critical layers may still require higher ranks even with rapid singular value decay.

2. Insufficient validation of the exponential decay assumption.
The core rank allocation mechanism hinges on the assumption that singular values follow an approximately exponential decay pattern and can be modeled by a single parameter λ. Although the paper provides preliminary theoretical reasoning and empirical evidence, it lacks sensitivity analyses showing how deviations from this assumption affect model performance, as well as more rigorous theoretical justification.

3. Under-examined assumptions in the optimization analysis.
The theoretical claim that the Hessian becomes isotropic (𝐻=2𝐼) depends on the assumption of nearly orthogonal, whitened inputs. However, it remains unclear whether this assumption holds under small-sample calibration or distributional shift, and whether it is consistent across different layers or batches. While freezing principal components may preserve pretrained knowledge, it could hinder adaptation in cases of aggressive compression or significant domain shift.

4. Modest empirical gains.
In the reported results, DF-SVD achieves only marginal improvements over SVD-LLM in accuracy, which may not be sufficient to demonstrate a strong advantage given that DF-SVD employs mixed-rank allocation whereas SVD-LLM uses a fixed rank. Although DF-SVD shows faster compression compared with Ada-SVD and Dobi-SVD, the performance comparisons are not exhaustive, leaving some uncertainty about its overall effectiveness.

### Questions
Refer to the weakness section

### Soundness
3

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
2

### Summary
This paper proposes DF-SVD, a SVD-based compression framework for large language models (LLMs). It addresses two key challenges in existing SVD compression:
1. Rank Selection Problem – current methods rely on costly search or uniform rank allocation. DF-SVD introduces decay-aware rank allocation, which leverages the singular value spectrum’s decay rate to assign truncation and update ranks per weight matrix dynamically.

2. Limited Accuracy Restoration – sequential weight updates in prior work (e.g., SVD-LLM) lead to Hessian anisotropy and slow convergence. DF-SVD proposes a feature-preserved weight update strategy that freezes principal components and only updates minor components, ensuring Hessian isotropy and preserving pretrained knowledge.

### Strengths
1. Clear motivation and problem definition: Identifies two fundamental bottlenecks in SVD compression (rank allocation and update inefficiency).
2. Theoretical contribution: Provides analysis showing Hessian isotropy under the proposed update scheme, linking spectral properties to convergence guarantees.

### Weaknesses
1. Generalization to larger models: Experiments are on 7B–8B scale models; it remains uncertain how well DF-SVD scales to 30B+ models.

### Questions
Have you tested DF-SVD on huge models (e.g., Qwen3-30B-A3B-Instruct-2507)? Does the efficiency advantage hold at that scale?

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
4

### Summary
This paper proposes DF-SVD to compress Large Language Models using Singular Value Decomposition. It solves two key challenges: the Rank Selection Problem and Limited Accuracy Restoration. There are two innovations: 1. Decay-Aware Rank Allocation, which dynamically assigns truncation and update ranks to each weight based on its singular value decay characteristics, eliminating the need for costly search; 2. Feature-Preserved Weight Update, a theoretically-grounded strategy that freezes key matrix components while only updating minor ones. This update strategy ensures an isotropic Hessian, leading to superior accuracy and faster convergence. The results show that DF-SVD outperforms existing methods.

### Strengths
1. The paper validates DF-SVD across four different models (LLaMA 1/2/3 and OPT) and eight datasets, consistently demonstrating superior performance.
2. The authors provide a detailed ablation study that confirms the positive impact of both the Decay-Aware Rank Allocation and the Feature-Preserved Weight Update components.
3. The method is efficient, completing the entire compression process 7-16 times faster than competing SVD baselines.

### Weaknesses
1.The Decay-Aware Rank Allocation method relies on an original truncation position ($ra_{old}$) and update rank ($rank_{old}$). It’s not clear how these critical baseline values are chosen, which makes the results difficult to reproduce.

2. Lack of theoretical proof for the assumptions (such as the reason of the singular value spectrums follow an exponential decay model should be justified). 

3.I was wondering whether the reported wall-clock time includes the LoRA fine-tuning stage, or only the SVD and calibration steps.

4.The update procedure using LoRA in section 3.2 looks quite similar to SVD-LLM .  Could your please articulate the key differences/novelty

5.The analysis of the Hessian (convergence) is based on minimizing reconstruction error, not the model's final task loss. This optimality for the reconstruction objective may not hold for the task objective. Is this a negative impact to the task performance?

### Questions
Please see the weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
