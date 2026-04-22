# DimPO: Dimensionality Reduction for Attention using Preference Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2, 4

## Abstract
Large Language Models (LLMs) require increasing memory and computational resources to process long context, motivating methods that reduce the dimensionality of key and query representations while preserving the resulting attention patterns. A previous triplet-based method for LSH-style projections optimizes only local pairwise relations and therefore struggles to capture the global ranking structure inherent to attention. We reframe dimensionality reduction of attention as a listwise learning problem based on preference optimization and ranking losses. Our analysis shows that listwise approaches better preserve full attention distributions in low-dimensional spaces, reducing KL divergence to less than 25\% of the value achieved by triplet-based and other pairwise methods. Building on this, we directly apply listwise-learned projections to the attention layers of  LLaMA3-[1B, 3B, 8B], Qwen2.5-7B and Qwen3-4B instruct models. On general benchmark tasks, listwise projection reduces KV cache memory by 10–15% while maintaining 95% of model performance. Larger models using this projection on long-context tasks exhibit only a marginal performance drop and demonstrate compatibility with existing KV-cache reduction techniques such as SnapKV, improving throughput without appreciable accuracy degradation. These results establish listwise learning as a principled and effective approach for dimensionality reduction of attention, directly applicable to attention layers at inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces DimPO, a novel reference-free listwise preference optimization loss designed to reduce the dimensionality of attention key and query vectors in large language models, thereby lowering KV cache memory and computational costs.

### Strengths
N/A

### Weaknesses
1. Poor performance. The authors need to sacrifice 5% performance when KV cache memory is reduced only by 10%-15%. If we take a look at a contemporary work, native sparse attention [1], they even achieve better performance than standard attention while reducing KV cache memory by at least 50%.
2. Poor performance, especially on longer context. If my understanding is correct, the motivation of saving KV cache is to handle longer context. However, in Table 5, we can see that the longer the sequence is, the more performance the authors have to sacrifice.
3. Limited evaluation. I think the authors could compare their methods with other KV cache saving methods other than preference optimization in their experiments, as they listed in their related work.
4. Efficiency. The authors' method cannot reduce required computation, which is commonly achieved by other KV cache compression methods.

[1]. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. ACL 2025.

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the memory bottleneck of the Key-Value (KV) cache in Large Language Models (LLMs). The authors propose to reduce the memory footprint by learning a projection to map key and query vectors into a "learned lower-dimensional space" , while leaving value vectors unchanged. The core contribution is twofold: (1) framing this projection task as a "preference optimization problem" , and (2) introducing DimPO, a "novel reference-model-free, listwise preference optimization loss". The paper argues that existing pairwise losses are computationally infeasible and existing listwise losses require a reference model, which is not available in this setting. The DimPO loss function is derived by adapting listwise losses (like LiPO) and removing the reference model by assuming it is a "uniform distribution".

### Strengths
1. The paper makes a valuable contribution to an important and active area of research: LLM inference efficiency. The primary contribution is the novel framing of attention dimensionality reduction "as a preference optimization problem". This opens a new avenue for this task. 

2. The second contribution, the DimPO loss, is a practical tool derived to fit this new problem formulation. It effectively fills the gap left by existing PO losses. The empirical results, showing a "10-15% reduction in KV cache memory with only about a 5% performance drop on generic tasks", represent a meaningful and practical trade-off for model deployment.

### Weaknesses
1. **Training Cost**: The proposed method introduces a significant training overhead for the projection layers. Appendix A reports that DimPO requires "\~5 min/layer" for training, whereas baseline methods like SimPO, Triplet, and ORPO are "substantially faster (\~10 s/layer)". This 30x increase in training time is a significant practical drawback, even if it is a one-time cost.

2. **Weak Long-Context Performance:** The method is strongly motivated by the challenges of "long-context tasks". However, the experimental results on the RULER benchmark (Table 4) are weak for smaller models. The paper notes "Llama 1B... quickly experienced substantial performance degradation". For instance, at 9.38% memory savings, the Llama3.2-1B model's average 4K score drops from 79.35 to 17.72. This severe drop suggests the method, while effective on general tasks, "are more sensitive"  and may not be a viable solution for its primary motivating use case, especially in smaller models.

3. **Issues with Other Methods**: The paper investigates combining DimPO with MagicPIG . The results in Table 5 show a non-trivial drop in performance. The base MagicPIG model achieves 92.63 on RULER 4K, while the "MagicPIG 9.38% (d' = 64, l = 12)" variant scores 87.59. A similar drop is seen on LongBench (35.84 to 32.58). The paper's claim that performance "remains reasonably high"  seems to understate this performance loss. This suggests that the errors from DimPO's projection and MagicPIG's LSH approximation may compound.

4. **Inconsistencies Between the Paper and the Code**: 

* 4.1. Definition of the Lambda Weight ($\Delta_{i,j}$)

  * **Paper Description:** The paper, when introducing the LiPO loss framework (Section 3.2), explicitly defines the lambda weight $\Delta_{i,j}$ as:
    $\Delta_{i,j}=|\frac{1}{D(\tau(i))}-\frac{1}{D(\tau(j))}|$
    where $D(\tau(s_{i}))=\log(1+\tau(s_{i}))$.

  * **Code Implementation:** The code implementation in `DimPO/src/models/lowdim_models.py` (inside the `_dimpo_loss` method of the `LowDimDimPO` class) calculates this weight (named `delta`) differently. It includes an additional gain term, $|G_i - G_j|$:

    ```python
    # Gains G_i = 2^{phi_i} - 1
    G = torch.pow(2.0, psi) - 1.0               # [B, K]

    # Discounts D(tau(i)) = log(1 + tau(i))
    D = torch.log1p(ranks.float())              # [B, K]
    
    # ... (omitted code) ...

    # Pairwise differences
    # ...
    G_i = G.unsqueeze(2)
    G_j = G.unsqueeze(1)
    delta_G = torch.abs(G_i - G_j) # [B, K, K]
    
    D_i = D.unsqueeze(2)
    D_j = D.unsqueeze(1)
    delta = delta_G * (1.0 / D_i - 1.0 / D_j) # [B, K, K]
    ```

  * **Discrepancy:** The code implements the weight as $\Delta_{i,j} = |G_i - G_j| \cdot |\frac{1}{D(\tau(i))} - \frac{1}{D(\tau(j))}|$. The gain term $|G_i - G_j|$ (where $G_i = 2^{\psi_i} - 1$ is defined in the paper) is present in the code but is omitted from the explicit definition of $\Delta_{i,j}$ provided in the paper.

---

* 4.2. Sign of the Final Loss Function (Equation 1)

  * **Paper Description:** Equation 1 defines the DimPO loss with a leading negative sign:
    $\mathcal{L}_{DimPO}(\pi_{\theta})=-\mathbb{E}_{(x,y,\psi)\sim\mathcal{D}}[\sum_{\psi_{i}>\psi_{j}}\Delta_{i,j}\log(1+e^{-(s_{i}-s_{j}-\gamma)})]$

  * **Code Implementation:** The code implementation in `DimPO/src/models/lowdim_models.py` (in the `_dimpo_loss` method) computes a positive loss value, which is then minimized by the optimizer:

    ```python
    # pair_loss = log(1 + exp(-(s_i - s_j - gamma)))
    pair_loss = F.softplus(-s_diff)             # [B, K, K]

    # Apply mask and weight
    weighted = (delta * pair_loss) * mask       # [B, K, K]
    loss_per_list = weighted.sum(dim=(1,2)) / mask.sum(dim=(1,2)).clamp_min(1)     #[B]
    
    return loss_per_list.mean()
    ```

  * **Discrepancy:** The code calculates a positive loss (the mean of the listwise softplus terms, weighted by $\Delta_{i,j}$) and returns this positive value for minimization. However, the paper's Equation 1 includes a leading negative sign. Minimizing the paper's formula as written would be equivalent to *maximizing* the positive loss term calculated in the code.

    This strongly suggests the negative sign in Equation 1 is a typo, as the code's implementation (minimizing a positive loss) represents the standard and mathematically correct optimization objective.

### Questions
1. The performance collapse of Llama3.2-1B on RULER (Table 4) is severe (e.g., 79.35 -> 17.72). The paper states larger models are "more robust". Can the authors provide insight into why this degradation is so extreme for smaller models? Does this imply that the learned projection fails to capture essential long-range attention patterns in models with lower capacity, and does this limit the method's applicability to only very large (>7B) models for long-context tasks?

2. Regarding the 30x training time increase for DimPO ("\~5 min/layer") compared to baselines ("\~10 s/layer") (Appendix A) : Is this cost primarily from the listwise loss computation, which must consider $O(m^2)$ pairs (where $m=4096$ in training)? Given this high cost, do the authors believe the improved KL/MSE (Table 1) over the much-faster ORPO/SimPO baselines justifies this trade-off?

3. In the MagicPIG combination experiment (Table 5), the authors applied DimPO "to the last 12 layers", resulting in a significant performance drop (e.g., 92.63 -> 87.59 on RULER 4K). Why were 12 layers chosen for this experiment? Could the observed performance drop be due to compounding approximation errors (LSH from MagicPIG and projection from DimPO)? Would applying DimPO to fewer layers (e.g., $l=4$ or $l=8$) have achieved a better balance of memory savings and performance preservation?

4. The derivation in Section 3.2 removes the reference model $\pi_\text{ref}$ by "treat[ing] $\pi_{ref}$ as a uniform distribution". The paper also cites SimPO, which "argues that using a reference model... is inconsistent with inference" , and adopts its margin $\gamma$. Can the authors please clarify the precise novelty of DimPO's formulation compared to what might be termed a "Listwise SimPO" (i.e., applying SimPO's reference-free reward $s_i = \beta \log \pi_\theta(y_i|x)$ directly to the LiPO listwise loss)?

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
4

### Summary
The paper proposes DimPO, a reference-free list-wise preference-optimization loss that learns a single linear projection to compress query/key vectors while keeping the attention distribution essentially unchanged. Applied to the last few layers of LLaMA-3 (1\~8 B) and Qwen-2.5/3 (4\~7 B) models, DimPO yields 10–15 % KV-cache memory reduction with ≤5 % average drop on five short-context benchmarks and comparable or slightly larger degradation on long-context suites. It outperforms triplet, PCA, random-projection, and existing pairwise preference losses (SimPO, ORPO, CPO) by 2–3× in KL divergence between original and projected attention weights.

### Strengths
1. Novel framing: first to treat attention dimensionality reduction as a preference-ranking problem rather than hashing or reconstruction, motivating a natural list-wise objective.

2. DimPO loss is elegantly reference-free, avoids the quadratic explosion of pairwise comparisons, and consistently beats strong baselines.

3. Thorough experimental sweep: ablates dimension, number of projected layers, and model scale; reports both attention-quality metrics (KL, MSE) and downstream task scores; includes long-context benchmarks and throughput measurements.

4. Practical impact: a 10% cache cut translates directly into larger batch sizes or longer contexts for serving; the projection adds only a small linear layer with negligible latency.

### Weaknesses
1. Scope of evaluation: all experiments compress only the last few layers; no systematic study of how layer depth interacts with sensitivity (only a top-down heuristic).

2. Task coverage: long-context evaluation is restricted to 4k and 8k RULER tasks and up to 65 k LongBench; no examination of >100 k tokens or code/documentation datasets where attention patterns differ.

3. Baseline omissions: no comparison against simultaneous KV-quantization (e.g., 4-bit KIVI) or low-rank factorization of the full attention matrix leaving the combined settings unclear.

### Questions
1. Why restrict the projection to a single shared linear matrix per layer? Did you try head-specific or unshared version that might yield higher compression with similar parameter count?

2. How does DimPO behave when combined with 4-bit KV quantization—do the errors compound multiplicatively?

3. Did you explore non-uniform layer-wise allocation of dimensions (e.g., d'=32 for early layers, d'=8 for final ones) to push compression further where attention is most robust?

4. Are there attention-pattern failure modes (e.g., many-shot retrieval, long code files) where DimPO degrades catastrophically?

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
This paper introduces a learned dimensionality reduction technique for keys in the KV cache of attention layers, aiming to reduce KV cache memory usage.

### Strengths
The proposed approach achieves a 10–15% reduction in KV cache memory.

### Weaknesses
1.	Limited evaluation: Reducing KV cache memory is a well-studied topic. Prior work has explored both reducing the number of keys and reducing key dimensions. The paper primarily compares against projection-based key dimension reduction methods, which makes the evaluation less comprehensive and less convincing.
2.	Missing key objectives: Most KV cache optimization techniques ultimately aim to improve inference speed or throughput. The paper does not report comparison results on these critical metrics, leaving the practical impact unclear.
3.	Evaluation metrics: Using KL divergence and MSE to compare projection methods (Table 1) is questionable because the ultimate goal is to maintain end-to-end model performance. Direct comparisons on end-to-end performance would be more informative.
4.	Incomplete comparison of pairwise losses: There are several established pairwise ranking losses (e.g., RankNet, ListNet) designed to preserve ranking. Including these in the comparison would provide a clearer picture of the proposed method’s advantages.
5.	Performance concern: Table 5 shows that applying DimPO on top of existing KV cache optimization methods introduces a noticeable performance drop, which raises concerns about its practical utility.

### Questions
1.	In line 181, should F(K_{k,q}) actually be F(K_{l,q})?
2.	What is the additional inference cost introduced by DimPO?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DimPO (Dimensionality Reduction for Attention using Preference Optimization), a post-training method aimed at reducing the KV cache size. The core idea is to learn a linear projection that maps the key (K) and query (Q) vectors into a lower-dimensional space. To achieve this, the authors propose a DimPO frames the task as a preference optimization problem, introducing a reference-model-free, listwise loss function that optimizes the projection matrices to ensure the compressed attention distribution closely matches the original, full-dimensional distribution. Experiments show that DimPO successfully reduces KV cache memory by 10-15% while retaining high performance (over 95%) on general benchmarks and maintaining competitive results on demanding long-context tasks.

### Strengths
- The proposed method poses the problem of dimensionality reduction in query and key to the preference optimization (PO) problem. The authors consider the original attention scores as the "preferred" ranking and the compressed attention scores as the "current" ranking. They propose the DimPO loss which directly optimizes the relative order of attention weights. This is fundamentally more effective for preserving the critical attention distribution than simply minimizing the absolute distance between vectors.

- The paper compares DimOP to other preference optimizations including Triplet, CPO, SimPO and ORPO, and shows that DimPO outperforms in terms of KL and MSE on attention weights. 

- The method provides memory reduction on KV cache. When the dimensionality reduces from 128 to 64, the paper shows that DimPO can achieve a 10-15% memory reduction.

### Weaknesses
- DimPO requires an additional post-training stage to learn the optimal low-dimensional projection. This introduces an additional cost and time compared to post-hoc optimization methods that can be applied directly to a pre-trained model without further training. In addition, learning the projection depends on the quality and diversity of the training dataset used for fine-tuning, potentially requiring high-quality, long-context-specific data to generalize well.

- The primary goal of the proposed method is to reduce the KV cache memory size for efficient token generation in LLM. However, the paper does not provide a direct head-to-head benchmark against other common KV cache optimization techniques like H2O (heavy-hitter eviction), SnapKV (token selection), or KIVI (quantization), which are mentioned in the related works section. Beyond the token eviction method, quantization is a dimensionality reduction approach, and there is a recent work [1] that applies a signed random projection on QK to reduce KV cache. It would be good to compare those works to the proposed method.

[1] Zandieh, Amir, Majid Daliri, and Insu Han. "Qjl: 1-bit quantized jl transform for kv cache quantization with zero overhead." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 24. 2025.

### Questions
How the proposed method performs compared to other KV cache compression methods, including H2O, SnapKV, KIVI and QJL [1] (above)?

Minor issues:
1. in line 160, does ``q ∈ Q_l`` mean a row vector in $Q_l$?
2. in line 180, is the equation should include as Softmax(F(q) F(K_{l,q})^T)? , not K_{k,q}?

### Soundness
2

### Presentation
3

### Contribution
2
