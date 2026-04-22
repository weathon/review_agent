# LOTFormer: Doubly-Stochastic Linear Attention via Low-Rank Optimal Transport

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Transformers have proven highly effective across a wide range of modalities. However, the quadratic complexity of the standard softmax attention mechanism poses a fundamental barrier to scaling them to long context windows. A large body of work addresses this with linear attention, which reformulates attention as a kernel function and approximates it with finite feature maps to achieve linear-time computation. Orthogonal to computational scaling, most attention mechanisms—both quadratic and linear—produce row-normalized maps that can over-focus on a few tokens, degrading robustness and information flow. Enforcing doubly-stochastic attention alleviates this by balancing token participation across rows and columns, but existing doubly-stochastic attention mechanisms typically introduce substantial overhead, undermining scalability. We propose LOTFormer, a principled attention mechanism that is simultaneously linear-time and doubly-stochastic. Our approach exploits the connection between attention maps and transportation plans between query and key measures. The central idea is to constrain the transport plan to be low-rank by conditioning it on a learnable pivot measure with small support. Concretely, we solve two entropic optimal transport problems (queries$\to$pivot and pivot$\to$keys) and compose them into a conditional (glued) coupling. This yields an attention matrix that is provably doubly-stochastic, has rank at most $r \ll n$, and applies to values in $O(nr)$ time without forming the full $n\times n$ map. The pivot locations and masses are learned end-to-end. Empirically, LOTFormer achieves state-of-the-art results on the Long Range Arena benchmark, surpassing prior linear and transport-based attention methods in both accuracy and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LOTFormer, an attention mechanism that achieves both linear-time complexity and doubly-stochastic structure. The authors leverage optimal transport theory to factorize the attention map into two entropic transport plans—one from queries to a learnable pivot measure and another from the pivot to keys. This results in a low-rank, doubly-stochastic attention matrix that avoids the computational overhead of traditional Sinkhorn-based methods. LOTFormer is evaluated across multiple benchmarks including Long Range Arena (LRA), ImageNet-1K, and IWSLT’14 translation tasks, demonstrating strong performance and scalability.

### Strengths
* The use of low-rank optimal transport with a learnable pivot measure is a mathematically elegant solution to enforce doubly-stochasticity while maintaining linear complexity.

* Evaluations are conducted on both vision and language tasks.

### Weaknesses
* This paper has several writing issues, making it difficult to follow:
  - It is heavily based on optimal transport theory but does not provide sufficient background on it. 
  - Section 3 introduces many symbols and variables (e.g., $\delta_{z_t}$, $\delta_{k_j}$, $\delta_{q_i}$, $C$, $\Gamma$ and so on) without adequate explanation of their meanings, shapes, or roles, making it hard to follow the derivation.

* The proposed approach is sensitive to hyperparameters. According to Table 6, performance appears sensitive to choices like entropy regularization and pivot size, which may require extensive tuning.

### Questions
1. Is LOTFormer compatible with causal attention settings, such as autoregressive generation?  
2. Does LOTFormer support hardware-efficient implementations like FlashAttention?

### Soundness
2

### Presentation
2

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
This paper proposes LOTFormer, a novel attention mechanism that achieves both linear-time complexity and doubly-stochastic normalization by reformulating attention as a low-rank optimal transport (OT) problem. The key idea is to introduce a learnable pivot measure with a small support size , which acts as an intermediate latent distribution connecting queries and keys. Two entropic OT problems are solved and composed via a glued coupling, yielding a provably doubly-stochastic and low-rank attention matrix.
Empirically, LOTFormer achieves state-of-the-art results on the Long Range Arena (LRA) benchmark and competitive performance on ImageNet-1K and IWSLT’14 translation tasks, outperforming prior linear and transport-based attention methods.

### Strengths
**Theoretical Foundation**. The paper provides a principled derivation of attention as an entropic optimal transport plan. The proposed “glued coupling” formulation ensures provable double stochasticity and rank <=r. The mathematical connection between transport plans and normalized attention is clearly established.

**Complexity Efficiency**. Achieves linear complexity, O(nr), without constructing the full n×n matrix.
Sinkhorn iterations are applied in factorized form, which improves scalability.

**Empirical Effectiveness**. LOTFormer sets new state-of-the-art results on LRA with the best average accuracy. On ImageNet-1K , it surpasses the baseline with negligible overhead. Consistently outperforms the baseline in the machine translation task.

**Compatibility and Extensibility**. The method can be combined wit existing improvements, such as polarity decomposition (Pol-[cls]). Conceptually unifies linear attention and transport-based normalization.

**Clear Visualization and writing**. This paper plot sufficient visualizations and present clearly.

### Weaknesses
**Missing Speed and Memory Benchmarks**.
The paper lacks quantitative comparisons of training/inference speed and memory usage against other baselines, which linear attention is proposed to solve. Although theoretical complexity is discussed, practical runtime and memory improvements are not empirically demonstrated. Without these results, claims of linear-time scalability remain partly unverified.

**Inadequate Long-Sequence Evaluation**.
Empirically, the performance of linear attention will degrade as the sequence length increasing comparing with softmax attention, which is the main issue recently. Only LRA benchmark is not convincing because it tests the model under only two transformer layers.

**Limited Model Size and Scope**.
The experiments are conducted on small-scale backbones and there is no evaluation on large-scale or high-capacity models (Base/Large in CV or 1B+ LLM), which limits the evidence of scalability. This makes it unclear whether the proposed method scales favorably in real-world high-throughput settings.

### Questions
**Q1. Causal Mask Compatibility:**
The proposed method enforces doubly-stochastic normalization, assuming balanced bidirectional attention. How would this formulation behave under causal masking, where attention is asymmetric and upper-triangular? Do the theoretical guarantees still hold in that setting, or would the normalization need to be modified? Additionally, how is the decoder implemented in Section 4.4 under this formulation?

**Q2. LOTFormer with Pola:**
Considering that the polarity decomposition in PolaFormer [1] was originally proposed to mitigate the loss of negative qk values in linear attention with ReLU feature maps, the motivation for adopting the Pola mechanism in LOTFormer appears unclear. Could the authors elaborate on why polarity decomposition is necessary in this framework, and why a simpler global Pola + Softmax formulation would not suffice?

**Q3. Runtime and Memory Efficiency:**
Could the authors provide runtime and GPU memory benchmarks to empirically validate the claimed linear-time complexity?

**Q4. More Extensive Experiments:**
It would strengthen the paper to include evaluations on longer-sequence benchmarks and larger backbones (e.g., DeiT-B, ViT-B, or GPT-like architectures with ≥340 M parameters) to better demonstrate scalability.

**Suggestion – Convergence and Stability of Sinkhorn Iterations:**
The paper relies on Sinkhorn iterations to compute entropic OT couplings, yet a more detailed analysis of their convergence behavior and numerical stability would be valuable. Since Sinkhorn iterations can be sensitive to the regularization parameter ε and may suffer from instability in low-precision or deep-training settings, including empirical convergence diagnostics or stability experiments would help clarify the robustness of the proposed approach.

This is an interesting work, and I would consider raising my score if the authors adequately address the concerns above.

[1] Meng, Weikang, et al. "PolaFormer: Polarity-aware Linear Attention for Vision Transformers." The Thirteenth International Conference on Learning Representations.

### Soundness
3

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
3

### Summary
The paper introduced LOTFormer, a transformer which changes the softmax attention into a doubly stochastic attention which is still linear in time. The theoretical framework based on the optimal transport is solid an visualisation on the effect of the query/key transport plan and attention heat maps by increasing the $r$ factor. The experimental setup is vast and covers Tiny scale image classification using ViT family, LRA benchmark for long-range analysis of the model, and plug-and-play and fine tuning results on machine translation task.

### Strengths
The paper has the following strengths:

1) The theoretical framework for the design of **LOTFormer** is sound, and the connection to optimal transport is well-established.

2) The experiments cover multiple modalities, including image classification on **ImageNet-1k** using the ViT codebase, as well as **LRA** and machine translation tasks.

3) The visualizations presented in the paper are clear and insightful, particularly **Figures 1 and 2**.

### Weaknesses
However, while the theoretical insights of the paper are sound, the empirical and experimental setup have several shortcomings:

1) **Learning of $Z_t$ in LOTFormer**  
   In the attention mechanism of LOTFormer, the model needs to learn $Z_t$, which appears to be an element with the same length as the sequence (if understood correctly). This introduces a significant bottleneck, as the model size will grow with sequence length — a critical limitation that prevents scaling beyond the model’s fixed context. This constraint contradicts one of the key advantages of linear and softmax transformers, which can extend context length flexibly.  
   If $Z_t$ is instead derived directly from the input, this is not clearly stated in the manuscript, particularly in *Section “Learning”* on page 5. How exactly is $Z_t$ learned?

2) **Role of Convolution in LRA Experiments**  
   The role of the convolution component in the LRA experiments is unclear. Convolution is a central feature in many linear transformers such as **Mamba-2** and **GDNet**, where it has been shown to significantly improve performance. It is likely that the observed performance gain of LOTFormer on LRA tasks primarily stems from this convolutional component rather than from its attention mechanism.  
   Therefore, an ablation study — removing the convolution from LOTFormer and adding it to other baselines — is essential to isolate its effect.

3) **Limitations of the LRA Benchmark**  
   The LRA benchmark is no longer a strong indicator of model quality. Models such as **S4** can achieve up to 84% accuracy on LRA tasks but perform poorly on language modeling. Moreover, pre-trained transformers (as shown in [1]) can reach around 85% accuracy on LRA, effectively solving these tasks. Thus, reporting approximately 60% accuracy for LOTFormer does not convincingly demonstrate its effectiveness or good design.

4) **Scaling in Image Classification**  
   The image classification experiments are conducted only on the *Tiny* scale, which is generally easier to handle. Even models like **DeiT** perform modestly at this scale but show strong improvements when scaled up. The paper should evaluate how LOTFormer performs at larger scales, as this would better reflect its scalability and robustness in vision tasks.

5) **Efficiency and Comparison with FlashAttention**  
   In Table 3, FlashAttention is reported as a benchmark. FlashAttention achieves efficiency through fused CUDA kernels and chunkwise computation for fast and memory-efficient training.  
   Does LOTFormer also implement similar optimized kernels (e.g., in **Triton** or **CUDA**) or any form of chunkwise parallel computation to support efficient training?  
   If not, it is unclear how LOTFormer can be faster or more efficient during training compared to FlashAttention. Additionally, even if such efficiency exists, how does LOTFormer scale with sequence length — particularly in comparison to the scaling trends shown in **Figures 2 and 6 of GLA [2]**?


------


### References

[1] Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors Ido Amos, Jonathan Berant, Ankit Gupta


[2]  Gated Linear Attention Transformers with Hardware-Efficient Training: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim

### Questions
1) How does the convolution component affect other linear transformer baselines on the LRA task, or how does removing the convolution affect LOTFormer?  
   This is particularly important since short convolution is a key component in many linear transformers. It would be valuable to analyze its impact, as I believe most of the performance gain may come from the convolution rather than the attention mechanism in LOTFormer.

2) What does $\delta_{k_j}$ represent on page 4?  
   For readers who are not very familiar with optimal transport notation, this term is unclear and should be better explained.

3) What is the closed-form expression for the attention mechanism in LOTFormer based on $Q, K, V, Z,$ and $\sigma$?

4) How does LOTFormer perform at a larger scale in the image domain?  
   While improvements at the Tiny scale are significant, it is typically much harder to achieve gains at the Base scale, as demonstrated in [1, 2].

------

### References


[1] Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers
Sukjun Hwang, Aakash Lahoti, Tri Dao, Albert Gu


[2] Linear Attention for Efficient Bidirectional Sequence Modeling
Arshia Afzal, Elias Abad Rocamora, Leyla Naz Candogan, Pol Puigdemont, Francesco Tonin, Yongtao Wu, Mahsa Shoaran, Volkan Cevher

### Soundness
2

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
The paper introduces **LOTFormer**, an attention mechanism that factors the full \(n\times n\) attention into two Sinkhorn-regularized optimal transport (OT) steps via a learned pivot distribution (queries→pivot, pivot→keys). This yields a low-rank, doubly-stochastic attention matrix without explicitly forming the full matrix, targeting near-linear time and memory \(O(nr)\) with small rank \(r\). The method keeps a tunable accuracy–efficiency trade-off through \(r\), the entropic OT temperature \(\varepsilon\), and the number of Sinkhorn iterations. Experiments span LRA tasks, ImageNet-1K (DeiT-Tiny setting), and IWSLT14 De-En, where the model is competitive or slightly better than recent linear/DS attention baselines. Ablations explore sensitivity to \(r\) and OT hyperparameters, and visualizations show controllable sparsity/structure in the learned attention.

### Strengths
- Clear formulation with a tunable rank; solid empirical coverage and visualizations.
- Competitive results on standard benchmarks.

### Weaknesses
- Contribution feels incremental vs. prior DS/linear attention and Sinkhorn-based methods.
- Gains on ImageNet seem to rely on extra tricks (e.g., CLS/polarization, DWC), making the core benefit hard to isolate.
- Limited evidence of end-to-end speed/memory wins and missing long-context causal LM tests.

### Questions
- Can you provide apples-to-apples baselines without the auxiliary tricks to isolate LOTFormer’s impact?
- How robust are ε, iteration count, and rank r for longer contexts/larger models, and how well does it handle causal masking in practice?

### Soundness
3

### Presentation
3

### Contribution
2
