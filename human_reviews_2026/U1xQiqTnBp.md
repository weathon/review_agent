# Positional Encoding via Token-Aware Phase Attention

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We prove under practical assumptions that Rotary Positional Embedding (RoPE) introduces an intrinsic distance-dependent bias in attention scores that limits RoPE's ability to model long-context. RoPE extension methods may alleviate this issue, but they typically require post-hoc adjustments after pretraining, such as rescaling or hyperparameters retuning. This paper introduces Token-Aware Phase Attention (TAPA), a new positional encoding method that incorporates a learnable phase function into the attention mechanism. TAPA preserves token interactions over long range, extends to longer contexts with direct and light fine-tuning, extrapolates to unseen lengths, and attains significantly lower perplexity on long-context than RoPE families.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Token-Aware Phase Attention (TAPA), a novel positional encoding method to deal with the limitations of Rotary Positional Embedding (RoPE) in long-context language modeling. The authors first provide a theoretical proof that RoPE has an intrinsic distance-dependent bias, which limits its ability to long-range modeling. TAPA overcomes this by incorporating a learnable phase function into the attention mechanism, which eliminates this bias. The paper demonstrates theoretically that TAPA's distance bias vanishes as context grows while maintaining non-degenerate attention. Empirically, the authors pretrain a LLaMA3 7B model and show that TAPA significantly outperforms RoPE and its variants (like PI and YaRN) on long-context perplexity benchmarks, especially when extrapolating to unseen context lengths. When pretrained on 8k context without finetuning, TAPA remains effective up to 32k context length, while other methods see a collapse in performance.

### Strengths
* The paper presents a mathematical proof of the inherent distance-dependent bias in RoPE. It also provides theoretical guarantees for TAPA, proving its decaying distance bias and non-degeneracy for long contexts.

* TAPA is a well-motivated idea to the prevalent RoPE-based methods. Unlike many RoPE extensions that rely on heuristic adjustments such as position rescaling, TAPA is presented as a more fundamental solution that can be extended to longer contexts with minimal fine-tuning and without manual hyperparameter changes.

* The head-to-head comparison with RoPE, PI, and YaRN on a LLaMA3 7B architecture demonstrates TAPA's superior performance in long-context scenarios. In particular, under a 8k context pretrain setting, TAPA remains stable up to 32k context length while the other methods collapse.The zero-shot evaluation also highlights TAPA's generalization ability.

### Weaknesses
* TAPA's formulation requires two QK dot products, which is computationally more expensive than the standard attention mechanism. While they mention the feasibility of a custom implementation, the practical overhead in terms of training and inference speed compared to highly optimized RoPE implementations is a potential limitation.

* The experiments are conducted on a 7B LLaMA3 architecture. Demonstrating the effectiveness of TAPA on a wider range of model sizes and architectures would strengthen the generalizability of the claims.

* The evaluation is focused on perplexity. Although this is a standard metric, it may not fully capture all aspects of long-context understanding, such as reasoning over long documents or complex instruction following.

### Questions
* Could you elaborate on the practical implications of the increased computational cost of TAPA? Have you performed any analysis on the slowdown compared to a flash-attention-based RoPE implementation during training and inference?

* The paper focuses on a quadratic phase function for TAPA. While the ablation shows it outperforms a linear phase, have you explored other families of functions for the learnable phase?

* Do you have insights into how pre-training with TAPA from scratch might differ from pre-training with RoPE in terms of training dynamics or final model capabilities?

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
This article re examines the shortcomings of remote attenuation in Rotary Positional Embedding (RoPE) and proposes a new position encoding called s Token-Aware Phase Attention (TAPA) based on this, which is beneficial for length extrapolation.

### Strengths
1.  Sufficient theoretical analysis. There is theoretical analysis on the weaknesses of ROPE's distance-dependent bias (Theorem 2.1, 2.2) and the relative advantages of TAPA (Theorem 3.2) in this regard.

2. Innovative solutions. A new solution (TAPA) for length extrapolation has been proposed and be verify in Section 4.3.

### Weaknesses
1.**Evaluation is limited**, without a long window benchmark, such as  a needle in a haystack [1] or ruler [2].

2.The assumption 2.1 **has not been verified**.  For example, the assumption of 2.1 may not be satisfied for any $d$.  As shown in [3][4], there is significant anisotropy between different dimensions. 

3.Lack of comparison. Perhaps we can compare alibi (with different slopes) and sliding window attention (with different window size). There are other methods that do not rely on RoPEs and can be extrapolated, e.g. CAPE [5], antmax[6]. 


[1] Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation. https://aclanthology.org/2023.eacl-main.75/

[2] RULER: What’s the Real Context Size of Your Long-Context Language Models? https://openreview.net/forum?id=kIoBbc76Sy

[3]The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval https://aclanthology.org/2025.findings-acl.697.pdf

[4]On the token distance modeling ability of higher RoPE attention dimension https://aclanthology.org/2024.findings-emnlp.338.pdf

[5]CAPE: Context-Adaptive Positional Encoding for Length Extrapolation https://arxiv.org/html/2405.14722v1

[6] Long-Context Generalization with Sparse Attention https://arxiv.org/pdf/2506.16640

### Questions
1.In my opinion, there is a **logical incompleteness** in this article, namely the unclear correlation between intrinsic distance dependent bias in attention scores and length extrapolation. In fact, when there is strong decay bias, the model can actually achieve better extrapolation on PPL, such as using swa [5], or even the a larger $\theta$ in RoPE [6].  **However**, this will present a superficial contextual ability, where the model's PPL can extrapolate, but in reality, it cannot perform tasks such as finding a needle in a haystack. 

2.At the same time, even the a larger $\theta$ in RoPE [6] can lead to better length extrapolation. But in line 130 “Fortunately, the next Theorem says that one can reduce such gap between any given position-pairs by further decreasing RoPE’s $\theta_0$".  Can the contradiction presented by these two be explained using the theorem presented in this article?

3.There is no code. This article does not provide code for implementing TAPA, especially for efficiently utilizing the Triton/Telelang. I am currently unclear about its speed difference compared to RoPE.

4.Will partial ROPE[10], as a recently popular setting [11,12], affect the conclusions of this article?

5.In [13], the author of RoPE get context length $L$ scale with the $2 \theta_0^{-1}$  by using $Ci(x)$ under a sufficiently large $d$. This is consistent with line 126's statement that 'context length is $O(\theta_0^{-1})$ Is there a connection？

[7] Dissecting transformer length extrapolation via the lens of receptive field analysis. https://aclanthology.org/2023.acl-long.756.pdf

[8] Scaling laws of rope-based extrapolation. https://openreview.net/forum?id=JO7k0SJ5V6

[9] Base of RoPE Bounds Context Length. https://openreview.net/pdf?id=EiIelh2t7S

[10] https://github.com/lucidrains/x-transformers/issues/40

[11] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model https://arxiv.org/pdf/2405.04434

[12] Kimi K2: Open Agentic Intelligence https://arxiv.org/pdf/2507.20534

[13] https://kexue.fm/archives/10122

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
This paper provides an analysis of the shortcomings of Rotary Position Embedding (RoPE), demonstrating the distance bias of RoPE, and further presents a new method, namely Token-Aware Phase Attention, which utilizes the inner product of the query and key to determine a phase function that encodes position information. The paper presents theoretical analyses and empirical evidence demonstrating that the new method outperforms RoPE in scenarios with long context lengths.

### Strengths
* The proposed method seems novel, which addresses a vital problem of distance-dependent bias that exists on RoPE.

* Overall, the analyses on the distance bias of RoPE and how the proposed method would improve it are clear and interesting.

* The experiments demonstrate that models with the proposed method are more stable when the context length increases, compared to models with RoPE.

### Weaknesses
* The proposed method splits the query and the key of each token into two parts and uses part of the query and key for the learnable phase function, which may restrict the ability of the model. As demonstrated in the experiment, under regular context lengths, the model with TAPA demonstrates worse performance than the model with RoPE.

* While the proposed method adopts a learnable phase function where $\cos (2\pi |m-n|^{\alpha} \phi(q,k))$ affects the attention score, the function $\phi(q, k)$ has an unstable impact on the attention score. For example, at some distances, larger $\phi(q,k)$ leads to a larger attention score, while at other distances, larger $\phi(q,k)$ leads to a smaller attention score. It also raises concerns about the expressivity and training stability of models with the proposed TAPA.

* When $\alpha<1$ (e.g., the $\alpha$ is set to $0.1$ in the experiments), $|m-n|^{\alpha}$ rarely changes at long distance, leaving the phase function (approximately) solely determined by the phase function $\phi(q,k)$, which is similar to a scenario where no position embedding is used. The community has witnessed methods like Multi-Head Latent Attention (MLA) that only adopt RoPE for part of the query and key. However, no experiments have been conducted to compare the proposed method with these previous methods.

* Lack of an ablation study on the hyperparameter setting. For example, how would $\alpha$ and $\theta$ and the choice of $\mathcal{N}$ affect the performance of the proposed method?

### Questions
As mentioned above in the weakness section, I have concerns about how the proposed method will affect the expressivity and training stability of the model, as well as the relationship between the proposed method and previous methods, such as MLA. In the following, I have further questions about the proposed method:
* While attention has a quadratic complexity regarding the input length, we have witnessed techniques like sliding window attention that try to reduce the computational complexity. Is it really necessary to pursue the ability to process extremely long text while sacrificing the ability in scenarios of regular length?

* Is the $\mathcal{N}$ in Eq. 10 a learnable parameter or initialized and fixed for the entire training? If it is learnable, it appears to introduce a large number of parameters.  If it is fixed, would the initialization have a dramatic influence on the model's performance? A more detailed ablation study would be appreciated.

Generally, I appreciate the theoretical analyses in the paper and think it would be an excellent paper with further justification on the proposed method and more empirical results. However, with the concerns I mentioned above, I am leaning towards rejection and look forward to the authors' replies.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a theoretical analysis of Rotary Positional Embedding (RoPE), proving it suffers from an intrinsic distance-dependent bias that hinders long-context modeling. To address this, it introduces Token-Aware Phase Attention (TAPA), a novel positional encoding that incorporates a learnable, token-dependent phase into the attention mechanism. The core contributions are: (1) A rigorous proof of RoPE's long-context instability and an explanation for the effectiveness of extension methods like PI. (2) The proposal of TAPA, which theoretically eliminates this bias and maintains long-range interactions. (3) Strong empirical results on a LLaMA3-7B scale model, where TAPA significantly outperforms RoPE variants at context lengths up to 64k, remaining stable where others collapse.

### Strengths
1. The paper provides a rare and rigorous theoretical dissection of RoPE's failure modes in long-context scenarios. The proofs regarding distance bias (Theorems 2.1, 2.2) are insightful and fill a crucial gap in community understanding.
2。 TAPA is a fundamentally new approach to positional encoding. Moving the positional dependency into a token-aware phase is an elegant idea, backed by sound theoretical motivations from harmonic analysis (e.g., decaying bias in Thm 3.2).
3. The experiments are large-scale, well-designed, and convincing. TAPA's ability to remain stable and performant at 49k-64k context lengths, where strong baselines like YaRN completely fail, is a very strong result.

### Weaknesses
1. The proposed TAPA formulation (Eq. 12) requires two separate `QK^T` dot products (`q_A^T k_A` and `q_P^T k_P`), effectively doubling the FLOPs of the most expensive part of the attention mechanism. The paper acknowledges this but understates the practical implications for training and deployment at scale.
2. The theoretical analysis of RoPE heavily relies on `Assumption 2.1`, which posits that expectations of certain bilinear forms of token embeddings are constant. This is a strong simplification that may not hold in practice, where representations are highly context-dependent. The paper does not discuss the gap between this assumption and the reality of trained models.
3. The token-aware phase term `q_P^T k_P` appears in the denominator of the argument to a cosine function. If this term approaches zero for certain token pairs, it could cause extremely high-frequency oscillations and numerical instability. The paper does not address this potential failure mode or discuss any regularization techniques (e.g., adding an epsilon) to prevent it.
4.  **Insufficient Analysis of Hyperparameters (Insufficient Analysis):** The method introduces new, critical hyperparameters (`θ` and `α`) 4. The model's performance dependency on the amplitude/phase split (`θ`) and the phase scaling (`α`) remains unexplored, hindering reproducibility and practical application.
5. The specific implementation of TAPA uses a hard, disjoint split of embedding dimensions for amplitude and phase. This design is somewhat ad-hoc. It is unclear why this is preferable to a "soft" approach where all dimensions could potentially contribute to both, for example, through separate linear projections.
6. While mathematically motivated, the paper offers no insight into what the learned phase `q_P^T k_P` captures. Is it learning syntactic relationships, semantic distance, or something else? Without this analysis, the "token-aware" aspect remains a black box.

### Questions
1.  Regarding computational cost: Could you provide wall-clock time benchmarks for training/inference throughput of TAPA versus a standard (and ideally, fused) RoPE implementation? How significant is the slowdown in practice on modern hardware?
2.  Regarding numerical stability: Did you observe instances of `q_P^T k_P` becoming very small during training, and did it cause any instability? Have you considered adding a small constant `ε` to the denominator as a safeguard, and if so, how does it affect performance?
3.  Regarding the TAPA formulation: Have you experimented with alternative formulations that don't rely on a hard split of dimensions? For instance, one could use the full query/key vectors and project them differently for the amplitude and phase components (e.g., `(qW_A)^T(kW_A)` and `(qW_P)^T(kW_P)`).
4.  Regarding hyperparameters: How was the phase scaling factor `α=0.1` chosen? This term seems to control the "sensitivity" of the attention score to the phase. How does performance change as `α` varies, for instance, from `1.0` down to `0.01`?
5.  Regarding the theoretical assumptions: How well does `Assumption 2.1` hold in your trained models? Could you empirically measure the statistics of the `A_d` and `B_d` terms from Eq. (4) to show how much they vary from a constant `µ0` and `ν0` across different positions and head dimensions?

### Soundness
2

### Presentation
1

### Contribution
2
