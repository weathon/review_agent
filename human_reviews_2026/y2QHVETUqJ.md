# UniPruning: Unifying Local Metric and Global Feedback for Scalable Sparse LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Large Language Models (LLMs) achieve strong performance across diverse tasks but face prohibitive computational and memory costs. Pruning offers a promising path by inducing sparsity while preserving architectural flexibility. However, existing methods struggle to balance efficiency and robustness: local metric approaches prune layer by layer but often collapse under high sparsity, whereas global feedback methods enforce consistency at the cost of expensive weight updates or restrictive semi-structured formats. We present \textbf{UniPruning}, a unified post-training pruning framework that combines the speed of local saliency metrics with the stability of global coordination, enabled by a mirror descent based optimization, all \textbf{without updating model weights}. UniPruning leverages fast layer-wise scoring and a lightweight global controller to allocate a single sparsity budget, supporting both unstructured and semi-structured $N{:}M$ pruning within one framework. After a brief calibration, it can generate pruning masks for arbitrary sparsity levels in one shot, and adapts seamlessly to hardware-aware constraints. Extensive experiments on multiple pretrained LLM families and standard benchmarks show that UniPruning consistently delivers competitive or superior perplexity and zero-shot accuracy. Ablation studies further highlight the importance of mirror descent and local saliency anchoring. Overall, UniPruning provides an efficient, principled, and scalable solution for sparsifying large-scale LLMs.We will release the code in the future.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
UniPruning introduces a new algorithmic variant to identify promising sparsity masks when pruning pretrained LLMs. They adopt a setting where the final weights are just a masked variant of the original weights -- without any further updates on the non-zero weights. The loss that is optimized to identify the pruning mask optimizes over the weights and a saliency variable $\Gamma$ (same dimensionality as the weights). The loss is composite of three terms  a/ the *global* language modelling objective of the weights under some calibration data b/  a *local* term aligning $\Gamma$ with the layerwise importance of the weights and c/ a regularizer on $\Gamma$ to induce sparsity.

The optimization problem is solved iteratively via a Proximal Operator / Mirror Descent approach (supported by theory) and the final pruning mask maintains the weights with largest corresponding value of $\Gamma$.

The paper evaluates the approach on LLMs of size up to 14B. At 60% unstructured sparsity the pruned models fall behind the original dense model by 7%-14% averaged over downstream tasks. While this marks a drastic drop in model capabilities, the paper reports 1-2% better results than competing pruning methods.

### Strengths
- The paper proposes a new pruning algorithm and provides conditions and a theoretical proof when the algorithm converges.
- The authors provide code to reproduce their experiments
- The method identifies scores for each weight and can then be efficiently used to extract masks a different sparsity levels without pruning again.
- The authors include ablations motivating the individual components of their loss.

### Weaknesses
- The paper focuses on a setting without weight updates after pruning and compares mainly to methods that do the same. This seems a very unnatural constraint and leaves a lot of potential of the expressivity of the model unused. For example [1] found that a simple closed-form local masked-gradient update after pruning can significantly improve Wanda's downstream task performance. Why would you not do that (for all methods). Furthermore, the paper does not provide results for other sparsity ratios than 60%. Why not, given that at 60% the loss is quite large already?

- The paper claims "pruning can achieve strong efficiency-accuracy trade-offs for LLMs". Honestly, think this is plainly wrong because the models have large accuracy drops and further compensation is needed. I am not aware that such models are really used beyond 
research settings -- as opposed to for example quantization.

- The paper is much more complicated than previous works like Wanda which has no hyperparameters. Can you provide runtime numbers of the algorithm? While the paper slightly improves over wanda, it does not relevantly close the accuracy drop against the base model. Also how does the algorithm quality and the runtime scale to models larger than 14B. I don't think that for such small models pruning is particularly relevant. Furthermore, amongst the more involved methods it seems barely better than ProxSparse on 2:4 sparsity.



[1] Kubler et al, arXiv 2501.18015,  *A Proximal Operator for Inducing 2:4-Sparsity*  
[2] Shen et al, ICML 2025,  *Targeted Low-rank Refinement: Enhancing Sparse Language Models with Precision*

### Questions
- What exactly is $\Omega$? I it just the L1 norm of the entire weights?
- In the algorithm it seems that $X$ used for the local scores is only computed once (line 1). Why not update it in every iteration, given that for the task loss one anyways needs to run a full forward pass through the model.
- Why do you not do weight updates?
- For unstructured sparsity, is the sparsity budget fixed at the matrix level or globally? If globally, do you also adjust this for Wanda?
- Please check the AVG in table 1. For example for Llama-3.2-1B the Dense model's average is clearly wrong.
- The regularizer for 2:4 sparsity seems to come from [1] and should be attributed correctly.
- line 862 has a broken link "By (??) ..."
- I think there are some inconsistencies in algorithm 1. In case of 2:4 sparsity do you really use both $R_{2:4}$ and $\Omega$?

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
3

### Summary
The authors propose UniPruning, a centralized post-training pruning framework that combines the performance of local saliency metrics and the stability of global coordination using a Mirror Descent optimization scheme.

### Strengths
- Introducing the saliency variable $\Gamma$ beautifully bridges the gap between local heuristic ratings and global optimization.
- One-time mask generation via global sorting of $|\Gamma^\star|$ improves hardware scalability and adaptability. 
- Includes detailed evaporation on local metric options (Magnitude, Wanda, RIA, stochRIA) and the role of Mirror Descent.

### Weaknesses
- All experiments focus on LLaMA and Qwen-family transformers, although this method claims to be architecturally agnostic. But there is no evidence of the performance of models with different architectures (such as Mistral, OPT, or encoder-only architectures).

- The Lipschitz convergence theorem assumes continuity of $\nabla L_{\text{task}}$ and smoothness of $S(W)$, which may not strictly hold in deep transformer architectures with layer-norm and non-linearity.

- The empirical advantages are less clear compared to previous dynamic approaches such as BESA, while the methods are much more complex.

### Questions
- How sensitive is UniPruning to the hyperparameters $\lambda$, $\rho$ and $\kappa$?
- How does it work under very high fragmentation (>80%) or mixed pruning?
- Can UniPruning support online or incremental pruning?

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
This paper introduces **UniPruning**, a unified post-training pruning framework for large language models (LLMs) that aims to combine the efficiency of local metric methods with the robustness of global feedback approaches. The key idea is to employ a **mirror descent–based** optimization that jointly learns a saliency variable $\Gamma$ alongside a frozen copy of the pretrained weights. UniPruning integrates two complementary signals: **local saliency alignment**, which leverages activation-aware statistics collected from a small calibration set, and **global sparsity feedback**, which enforces a single, model-wide sparsity budget through a proximal update on $\Gamma$. After a brief calibration phase, the method can generate one-shot pruning masks for arbitrary sparsity levels without retraining or weight updates, and the same framework supports both unstructured and semi-structured (N:M) sparsity patterns.

### Strengths
- The paper proposes a unified pruning strategy that bridges the gap between _local metric_ and _global feedback_ methods. Unlike prior approaches that focus solely on either local or global sparsity criteria, UniPruning jointly leverages both through a principled mirror-descent formulation. This hybrid view is conceptually elegant and practically useful for achieving balanced sparsity allocation.
    
- The use of mirror descent to couple local saliency and global sparsity is mathematically well-grounded. The paper provides convergence analysis and a  interpretation of how the auxiliary saliency variable $\Gamma$ stabilizes pruning under high sparsity. Theoretical insights are consistent with empirical behavior, showing good alignment between analysis and experiments.

### Weaknesses
1. **Unclear mathematical presentation.**  The theoretical exposition is difficult to follow and lacks rigor in notation and symbol definition. For instance, 
	* Eq. (1) ambiguously reuses $C$ to denote both the calibration dataset and a cost term; 
	* Eq. (3) introduces an undefined function $g_\ell$; 
	* Eq. (4) employs the hyperparameter $\rho$ without prior explanation; and the sparsity regularizer $\Omega(\cdot)$ — a key innovation in this work — is not clearly defined or intuitively described. 
	* Variable $V$ appears in Eq. (6) without definition. 
	These inconsistencies make it hard to reproduce or even interpret the mirror-descent dynamics. As a result, roughly half of the equations require readers to infer meaning, reducing overall clarity and technical credibility.  

2. **Conceptual disconnection between figures and method.** The only process diagram (Figure 1) does not map cleanly to the described algorithm. It omits the role of the global vs. local components, the function of $V$ in the optimization loop, and the action of the sparsity regularizer $\Omega$. Moreover, while the figure distinguishes between “search” and “pruning” stages, this distinction is not reflected in the algorithmic formulation or pseudocode. The resulting mismatch makes the framework harder to follow and even misleading about the underlying computation flow.  

3. **Unbalanced and incomplete baseline selection.**  The experiments focus mainly on lightweight post-training pruning baselines (e.g., Wanda, RIA), which are heuristic and efficient but methodologically less aligned with UniPruning’s optimization-based nature. In contrast, more comparable dynamic approaches such as **SparseGPT** [1] (ICML 2023) and **BESA** [2] (ICLR 2024) are omitted, despite sharing similar mask-optimization goals. This limits the fairness of the empirical claims. Moreover, the paper repeatedly asserts efficiency but does not provide quantitative comparisons of **actual pruning runtime** or **computation overhead** from the mirror-descent search stage. Without such measurements, the “efficiency” claim remains unsubstantiated.  

[1]: Frantar, Elias, and Dan Alistarh. "Sparsegpt: Massive language models can be accurately pruned in one-shot." _International conference on machine learning_. PMLR, 2023.

[2]: Xu, Peng, et al. "BESA: Pruning large language models with blockwise parameter-efficient sparsity allocation." _arXiv preprint arXiv:2402.16880_ (2024).

4. **Ambiguity in N:M sparsity and proximal operator implementation.**  Algorithm 1 introduces the update $W^{n+1} \leftarrow \text{Prox}_{R_{2:4}}(W^{n+1})$ with a heuristic definition of $R_{2:4}(\cdot)$, alongside $\Gamma^{n+1} \leftarrow \text{Prox}_{\Omega}(V^{n+1})$. However, the mechanism by which these proximal mappings enforce semi-structured (N:M) sparsity is unclear.  The paper should clarify whether $R_{2:4}$ acts as a projection operator, a regularizer, or a surrogate loss, and how $\text{Prox}_{\Omega}(\cdot)$ concretely encodes the block-sparsity constraint. This is a key conceptual step that currently lacks transparency.  

5. **Insufficient ablation depth and hyperparameter analysis.**  The study does not explain the selection or sensitivity of critical hyperparameters such as $\lambda$ and $\rho$, which are fixed to 0.001 and $10^{-5}$ without justification. The number of mirror-descent iterations $N$ and its influence on convergence or performance are also not analyzed. The use of *stochRIA* as the default local metric is mentioned but never formally defined, further obscuring its contribution.  

6. **Limited practical gains relative to methodological complexity.**  Although UniPruning improves slightly over static local baselines, the empirical advantage is modest compared to prior dynamic approaches such as BESA. Considering the additional complexity of introducing mirror descent and auxiliary variables ($\Gamma, V, \Omega$), the marginal gains do not convincingly justify the added computational and conceptual burden.

### Questions
1. **Efficiency–performance trade-off.**  
   The paper emphasizes efficiency, yet no quantitative pruning-time comparison is reported. Could the authors provide the wall-clock pruning time or FLOPs cost for UniPruning versus Wanda, RIA, SparseGPT, and BESA on the same model scale? 

   In the post-training compression literature, a key criterion is achieving a favorable trade-off between **efficiency** and **performance** — i.e., either outperforming more efficient baselines by a large performance margin, or matching stronger performance-oriented methods with significantly better efficiency. Without explicit runtime and computational cost analysis, it is difficult to assess where UniPruning lies along this trade-off spectrum.
   
2. **Mirror-descent iteration analysis.**  
   How many mirror-descent iterations are typically required for convergence, and what is the relationship between the number of iterations, pruning quality, and resource cost (time or FLOPs)? Could the authors identify whether there exists an optimal iteration point that balances performance and efficiency?

### Soundness
3

### Presentation
2

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
This work presents a LLM pruning framework that unifies local saliency metrics and global coordination using mirror-descent optimization. By doing so, this paper combines the benefit of both local and global pruning:  the pruning masks are computed in one shot and pruning is done without any retraining. The proposed method is evaluated on a set of LLMs (LLaMA2, Qwen2.5, Llama-3.2, DeepSeek-R1) and against multiple baseline methods (e.g., magnitude-based pruning, Wanda and RIA).

### Strengths
+This paper offers a principled approach for pruning: it uses mirror-descent optimization that couples local saliency with global feedback and sparsity projection. Theory 1 justifies extracting pruning masks directly from the limit $\Gamma^{\star}$ via global thresholding, enabling one-shot mask generation without retraining.
+ The proposed work is evaluated on a wide range of LLMs  (LLaMA2, Qwen2.5, Llama-3.2, DeepSeek-R1), showing superior performance over the selected basline methods.

### Weaknesses
The reviewer indeed saw a few weakness:
1. Pruning is a very well studied and crowded topic at this moment. Many local and global pruning methods (and their hybrid combinations) have been reported. As a result, the overall idea of this work still sounds incremental.
2. Some other important basline methods are missing in the result evaluation. For instance, optimal brain surgeon, Woodfisher, etc. This is somehow unavoidable when working in a very crowded field like LLM pruning, since so many similar ideas are published per week. 
3. Without retraining, this pruning method may lose some accuracy.

### Questions
This work has compared with Wanda. I'm just curious, how would this method compare with some improved work of Wanda (e.g., Wanda++ by Yang in 2025 and M-Wanda by Choenni in May 2025)? I understand that these works were released only a few months before the ICLR'2026 deadline, so it's hard to provide a comprehensive evaluation, so a brief comparision with some key quantative measure should be enough.

### Soundness
3

### Presentation
3

### Contribution
2
