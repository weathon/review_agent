# LeSTD: LLM Compression via Learning-based Sparse Tensor Decomposition

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large Language Models (LLMs) achieve remarkable success, but their massive parameter counts present significant deployment challenges. Post-training tensor decomposition offers a promising, data-free compression strategy by exploiting structural redundancies within the model weights. However, existing tensor methods face a critical limitation: the dense core tensor bottleneck. While these methods find a shared low-rank basis, the resulting dense core tensor grows polynomially with the chosen ranks, becoming a new storage bottleneck and capping the maximum achievable compression. To overcome this fundamental barrier, we introduce LeSTD (\textbf{Le}arning-based \textbf{S}parse \textbf{T}ensor \textbf{D}ecomposition), a novel two-stage framework for the high-ratio compression of Multi-Head Attention (MHA) blocks. LeSTD first employs an iterative algorithm to identify a high-quality, and shared orthogonal basis that jointly represents all attention heads. Subsequently, it introduces a principled, importance-based pruning algorithm that learns an ultra-sparse core tensor by systematically removing the least salient elements and refitting the remaining ones to preserve model fidelity. By decoupling basis optimization from core sparsification, LeSTD breaks the compression ceiling imposed by the dense core, enabling significantly higher compression ratios than prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LeSTD (Learning-based Sparse Tensor Decomposition), a data-free, post-training compression framework for large language models. It addresses the dense core bottleneck in tensor decomposition methods by learning a shared basis across attention heads and then applying a theoretically grounded pruning mechanism to create an ultra-sparse core tensor. This yields higher compression ratios without accuracy loss. LeSTD performs inference directly in the compressed domain, providing throughput gains without custom kernels.

### Strengths
1. If I understand correctly, prior methods such as SVD-LLM and ASVD require calibration data, whereas LeSTD operates entirely data-free—yet still outperforms them. This is quite impressive.  
2. The idea of tensorizing all weights into a unified 4D structure and applying Tucker decomposition** is both intuitive and novel, offering a principled way to exploit inter-layer correlations that matrix-based methods overlook.

### Weaknesses
Check in questions part.

### Questions
1. The authors should clarify the distinction between LeSTD and TensorLLM. If my understanding is correct, Section 3.1 is identical with TensorLLM, while Section 3.2 is the main difference. It would be helpful to make this relationship explicit.

2. As I understand, LeSTD does not require calibration data, which is a key advantage. Would it be possible to incorporate activation information into this framework to further improve performance? I suspect this might be challenging under Tucker decomposition, but a discussion on its feasibility would be valuable.

3. The paper could include comparisons with more advanced SVD-based pruning methods, such as Basis Sharing [1] and Pivoting Factorization [2]. The concept of Basis Sharing appears somewhat related, since Tucker decomposition also captures inter-weight similarity. 

4. It would strengthen the presentation to include an inference algorithm, similar to Algorithm 1 for pruning. Section 3.3 is currently somewhat difficult to follow, and a concise pseudocode description would improve clarity.

5. Will the code be released? If I understand correctly, the original linear layer requires one matrix multiplication, low-rank layers require two, and the proposed Tucker-based structure requires four. This may introduce additional I/O overhead. The authors claim that the method achieves speedup without custom kernels—i.e., purely using PyTorch—which is an impressive claim but also raises concerns. Providing the implementation during the rebuttal period would greatly improve credibility and reproducibility.

[1] Wang, Jingcun, et al. "Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression." The Thirteenth International Conference on Learning Representations.  
[2] Zhao, Jialin, Yingtao Zhang, and Carlo Vittorio Cannistraci. "Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models." Forty-second International Conference on Machine Learning.

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
3

### Summary
This paper proposes a two-stage large language model compression framework based on Tucker decomposition. The first stage performs subspace decomposition to obtain low-rank latent representations, while the second stage compresses the core tensor via a closed-form sparse pruning method. The paper is well-organized, with sound theoretical analysis, detailed mathematical derivations, and extensive experimental validation.

### Strengths
1.	The paper introduces a concise and elegant closed-form sparsification method in the decomposed latent space.
2.	The mathematical derivations are thorough and rigorous, providing solid theoretical support for amplitude pruning in the Tucker latent space.
3.	The motivation is clearly articulated, and the background is well presented.

### Weaknesses
1.	There are some formatting issues in the manuscript (e.g., line 100 references Figure 4 located at line 227).
2.	The figures and their explanations are somewhat unclear. For example, in Figure 3, the components are scattered without clear annotation or explanation of what each parameter represents.
3.	The introduction of the core tensor compression method—the paper’s key innovation—is somewhat disorganized. Although the mathematical derivations are complete, their presentation could be improved with more structured figures and explanations.

### Questions
In the throughput analysis, the paper claims inference acceleration using standard PyTorch functions on sparse cores. However, it is uncertain whether PyTorch natively accelerates unstructured sparsity. Have the authors considered other possible factors that might contribute to the observed speedup?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this submission a novel post-training compression framework for LLM called LeSTD is proposed. The approach is two-step: in the first step Tucker decomposition is applied to learn shared orthonormal factors across attention heads for each layer of the LLM. During the second step the ultra-sparse core tensor is created using importance-based iterative pruning. The importance score is derived from a reconstruction error.

### Strengths
1.  The two-step approach is well-motivated and intuitive: first we learn a basis and then we sparsify within that basis
2.  Theoretical justification for magnitude-based pruning is solid.
3.  Extensive experiments with different models and different datasets.
4.  LeSTD shows improvement over the competing methods across different compression ratios.
5.  Paper is well-written, the presentation of the method is good with neat illustration;
6.  The limitations are acknowledged and discussed (unstructured sparsity and MHA-only compression)

### Weaknesses
0.  The overall novelty of the submission is limited: Stage I is a known combination of the Tucker decomposition and HOOI, Stage II reduces to magnitude pruning (which is known optimal for orthonormal bases). The contribution is primarily in combining these for LLMs compression rather than theoretical/methodological novelty
 
1.  Other sparse tensor methods are not considered (CP decomposition, Tensor-Train Decomposition, structured/block-sparse Tucker variants, etc)
2.  Experiments do not include statistical significance (which is crucial in such works): no error bars, confidence intervals, or multiple runs reported;
 
3.  Limited ablation studies: only 6 rank configurations tested (Table 2), no ablation on pruning rate α, refitting frequency, or HOOI convergence criteria
 
4.  For the throughput, LeSTD sometimes loses to SVD-LLM (e.g., for OPT-30B on MathQA at 0.8 compression rate), but analysis of when/why is insufficient

### Questions
1.  Can you provide error bars across multiple experimental runs to assess statistical significance of the improvements over other methods?
2.  How sensitive is performance to the pruning rate α? Only α=0.1 is mentioned; was this tuned?
3.  What is the actual sparse indexing overhead at different sparsity levels? How does this affect real compression ratios?
4.  Why does LeSTD sometimes lose to SVD-LLM on throughput (e.g., OPT-30B)? Can you characterize when your method wins vs. loses?
5.  Can you provide results extending LeSTD to FFN layers, even if preliminary?
6.  How important is the refitting step (Eq. 6)? How does skipping this step affect the performance?
7.  What are the wall-clock compression times for Stage I and Stage II compared to baselines?

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
The paper introduces LeSTD, a two-stage, post-training compression framework for LLMs. Stage I performs a shared-subspace Tucker decomposition of the tensorized multi-head attention (MHA) weights: all heads in a layer share three orthonormal factor matrices, while each head retains its own small core, estimated via HOOI to minimize reconstruction error. Stage II sparsifies the per-head Tucker cores using an importance score equal to the coefficient magnitude, followed by a closed-form refit of the remaining coefficients. The method supports inference directly in the compressed domain where it reuses the shared projection and contracts the (sparse) per-head cores—without reconstructing dense weights. Empirical results demonstrate the effectiveness and efficiency of the proposed method across various tasks and models.

### Strengths
* The paper is clearly written, with the methodology and experimental setup presented with detail in a organized way.
* The pruning step is well justified.
* Exploring tensor decomposition for post-training compression is an interesting and relatively underexplored direction in the domain.

### Weaknesses
* Figure 1 is not strong enough to justify the paper's motivation. The “shared subspace across heads within the same layer” claim is not well supported given the low explained energy, and the intra-layer explained energy is only marginally higher than the inter-layer case.
* As the paper laid out in the limitation section, the current method does not handle FFN layers which constitute a large fraction of LLM parameters. Additionally, because the pruning is unstructured, actual storage and speed benefits would depend on the chosen sparse format and kernel support, so the practical gains may be smaller than the reported parameter reduction.

### Questions
* Can you report wall-clock for Stage I optimization?

* Is it possible to consider structured pruning? Would this affect the performance greatly?

### Soundness
3

### Presentation
3

### Contribution
2
