# CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling Multi-Head Latent Attention

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Converting pretrained attention modules such as *grouped-query attention* (GQA) into *multi-head latent attention* (MLA) can improve expressivity without increasing KV-cache cost, making it attractive for efficient inference. However, many practical conversion baselines rely on weight-only low-rank approximations (e.g., SVD-style initializations) and uniform rank allocation. They focus on minimizing the difference between weight matrices rather than on how those weights affect input activations, ignore the covariance structure of activations, and enforce uniform rank across layers—causing activation drift and degraded attention fidelity. To address these issues, we propose CARE, a ***C**ovariance-**A**ware, **R**ank-**E**nhanced* MLA conversion pipeline under a fixed KV width. CARE introduces three key steps: (i) *activation-preserving factorization*, which aligns the approximation with the actual input activations rather than just the weights; (ii) *adjusted-rank allocation*, which spreads a fixed KV budget across layers by giving more capacity to layers that need it most; and (iii) *KV-parity mapping*, which reparameterizes the converted K and V to fit the MLA format while keeping the KV-cache size unchanged. Our method outperforms a uniform-rank SVD baseline on Qwen3-4B/30B-A3B-Instruct-2507 and Llama-3.1-8B/70B-Instruct, reducing one-shot perplexity by up to 215× and improving mean accuracy by up to 1.70× at matched KV budgets. With a brief post-SVD "healing" fine-tune, we fully recover the original model's accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a Covariance-Aware, Rank-Enhanced MLA conversion pipeline under a fixed KV width for converting Grouped Query Attention (GQA) architectures into Multi-Linear Attention (MLA). The goal is to improve the efficiency of large language model (LLM) inference while maintaining strong performance.

The method introduces three main innovations:
	1.	Activation-preserving factorization — a decomposition scheme that minimizes error in activation space (∥XW − X̂W∥) rather than in weight space, aligning the approximation with the actual activations encountered during decoding.
	2.	Adjusted-rank allocation — a rank-adaptive scheduling strategy that redistributes a fixed KV cache budget across layers, allocating more capacity to layers with higher representational importance.
	3.	KV-parity mapping — a reparameterization technique that reformulates the converted K and V projections to match the MLA format while preserving covariance structure.

Empirically, the paper reports that this approach achieves superior zero-shot performance compared to TransMLA, suggesting better preservation of representational fidelity under constrained KV width.

However, the fairness of the comparisons and the novelty relative to TransMLA warrant further clarification—particularly the relationship between the proposed activation-preserving factorization and KV-parity mapping versus TransMLA’s RoRoPE and BKV-PCA components.

### Strengths
The paper is well-motivated, addressing a meaningful and practical problem in efficient LLM inference under KV cache compression. The proposed Covariance-Aware, Rank-Enhanced approach presents a sound and intuitive idea for improving the quality of MLA conversion under a fixed KV width.

Moreover, the ablation studies are well-designed and effectively demonstrate the effectiveness and contribution of each proposed component, providing convincing empirical support for the overall method.

### Weaknesses
1. Comments on Methodology and Comparative Analysis. 
In Lines 077–092, the manuscript states that TransMLA employs a direct SVD initialization that minimizes the error in weight space (∥W − Ŵ∥) rather than in activation space (∥XW − X̂W∥), thereby overlooking how the projection actually operates during decoding. However, it should be noted that the original TransMLA paper introduces three key innovations—RoRoPE, FreqFold, and BKV-PCA—all of which are explicitly designed to perform activation-aware decomposition. The manuscript would benefit from a clearer discussion of this distinction and its implications.

2. Concerns on Rank-Adaptive Scheduling and Efficiency.
The proposed rank-adaptive scheduling under a fixed KV width raises concerns regarding the variability of KV cache sizes across layers and attention heads. Such heterogeneity may lead to inconsistent memory allocation and suboptimal utilization of hardware resources, potentially diminishing the actual inference speedup. It is therefore recommended that the authors conduct empirical speed comparison experiments based on established efficient attention frameworks such as FlashMLA and FlashAttention to validate the claimed efficiency gains.

3. Comparative Evaluation and Missing Baselines.
Table 1 currently reports results only in comparison with TransMLA. In their experiments, compressing the Llama-3-8B KV cache to 1−576/2048 = 71.875% yields a perplexity of 25.8047 [1], which is substantially lower than that reported in this manuscript. This discrepancy warrants further investigation. In addition, the evaluation should be expanded to include other relevant GQA-to-MLA conversion methods, notably Palu [2] and MHA2MLA [3], to ensure a fair and comprehensive comparison.

4. Clarification on RoRoPE and Absorb Operations.
The operation described in Line 291 appears conceptually related to the RoRoPE mechanism proposed in TransMLA. The manuscript should more explicitly delineate the differences and connections between these two approaches. Furthermore, it remains unclear how the paper achieves the conversion from GQA’s RoPE to NoPE. The authors should also clarify whether their method supports the MLA Absorb operation, which is a crucial design feature in TransMLA enabling the transition between efficient training and efficient inference.

References

[1] TransMLA: https://github.com/MuLabPKU/TransMLA

[2] Palu, https://arxiv.org/pdf/2502.14837

[3] MHA2MLA, https://arxiv.org/abs/2502.14837

### Questions
See Weaknesses.

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
The authors propose CARE, a post-training adaptation method to transform trained MHA/GQA module to MLA for improved efficiency and performance. The CARE method essentially performs SVD as best low-rank approximation not over weight matrices but on the product of weights and inputs. They also propose to distribute rank budgets to different layers based on the observation of heterogeneous KV rank across different layers of a Transformer model. The model after CARE conversion is trained with a teacher-student knowledge distillation loss along with additional SFT to heal / recover lost performance due to the conversion. The resulting method achieves better performances compared to other baseline in this domain on various language modeling task.

### Strengths
- The proposed method is a clear improvement over prior works, and the insight on performing SVD over activations is indeed a good one.
- The paper is well-written and easy to understand.
- I really like observation 1 & 2 and it’s an interesting read.

### Weaknesses
- Line 068-069: I disagree that this is a whitening operation. Whitening is a very well-defined operation. If X is a matrix of shape (B, D), where B is batch size and D is dimension. Then whitening is X * (1/(B-1) * X^T X)^{-1/2}. Also we assume that X is centered. So I suggest avoiding using the very specific term of whitening with a mathematically precise definition.
- In general, all the text in the figures are too small and impossible to read if you print out the paper.
- Line 205-211: I don’t think this is precisely the covariance matrix. You are missing a 1/(T-1) multiplier and a centering operation.

### Questions
- It’s good to see that the CARE method is better compared to lots of baseline in terms of perplexities / accuracies etc, but how expensive is it compared to other baseline? It just seems like optimizing equation 2 is very costly.
- I’m happy to raise my score if authors address my concerns & questions.

### Soundness
2

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
CARE proposes a post-hoc conversion method to transform pretrained GQA/MHA attention modules into Multi-Head Latent Attention (MLA) while maintaining fixed KV-cache budgets. CARE replaces the common joint‑SVD initialization with an activation‑aware factorization, enabling post-hoc conversion of pretrained attention models to memory-efficient MLA.

### Strengths
1. While covariance-weighted factorization exists in prior work (FWSVD, SVD-LLM), the specific formulation of SVD(CW) followed by C^(-1) unwhitening for MLA conversion is fairly novel.

2. Adaptive rank allocation under fixed budget: The water-filling algorithm for distributing rank across layers based on weighted singular spectra is a good solution, with the empirical observation (Figure 2) that layers have heterogeneous sensitivity to rank reduction motivating this approach. The paper also does a good job of combining activation-aware factorization, layer-wise rank scheduling, and KV-parity constraints into a cohesive pipeline.

3. Evals: Experiments across 8+ benchmarks (WikiText, ARC, HellaSwag, MMLU, etc.) with both zero-shot and fine-tuned evaluations are reported. The paper studies calibration dataset sensitivity (Table 2), rank profiles across corpora (Figure 3), and layer-wise heterogeneity (Figure 2a-b), providing empirical support for design choices. Some ablations are well-designed: Section 4.2.1 shows consistency of rank profiles across four calibration corpora, supporting the claim that rank distribution is model-intrinsic. Section 4.2.2 demonstrates robustness to calibration corpus choice and Figure 2's layer-wise sensitivity analysis supports the heterogeneous rank allocation.

### Weaknesses
1. The paper's central claim uses ||C(W - Wc)||²F as a proxy for ||√C(W - Wc)||²F (page 5, Section 3.4), justified by a brief eigenspace argument that both are left-multiplied by the same eigenspaces of C with different weightings (λ²ᵢ vs λᵢ).  This essentially squares the importance weights, which could over-emphasize dominant directions and under-represent moderate-variance directions that still matter for downstream tasks. The paper's claim that this "tends to preserve ordering of dominant components" is unsubstantiated.

2. Results focus almost entirely on Llama-3-8B, with Qwen models mentioned only in Appendix D hyperparameters but absent from main results tables. Scalability to larger models (13B, 34B, 70B) where KV-cache can be more critical remains unvalidated. The paper is also missing evaluation on models with different attention patterns (e.g., Mixtral MoE, Qwen with different GQA ratios) and would benefit from analysis of whether the depth-dependent rank profile (Figure 3) generalizes across architectures.

3. MHA2MLA is dismissed for zero-shot evaluation due to partial-RoPE, but no fair comparison is provided after fine-tuning (Table 3 only compares TransMLA). Other activation-aware SVD methods (SVD-LLM V2, FWSVD) are cited in related work are not empirically compared.

4. Missing long-context evaluation: Despite KV-cache being important for long sequences, no experiments on long-context benchmarks (e.g., LongBench, ZeroSCROLLS) or needle-in-haystack tasks are provided to validate usecases where MLA/CARE matter most.

5. Fine-tuning details: Table 3 shows "healed" results but doesn't specify exact token budgets, learning rates, or whether hyperparameters were tuned per method. The claim that CARE "requires less data" isn't quantitatively supported with ablations over data budgets.

6. Computing covariance (O(ND²)) and SVD on CW requires non-trivial cost, but there's no comparison of conversion time vs. naive SVD is provided. The paper also states CARE preserves MLA's "comparable throughput" but provides no actual latency/memory measurements or throughput benchmarks on realistic workloads. The paper claims CARE is practical, but doesn't provides runtime/memory analysis of the conversion process itself to validate this.

### Questions
1. Can the authors provide ablation results for: CARE decomposition (activation-aware SVD) with uniform rank allocation and standard SVD with CARE's adaptive rank allocation? This would clarify whether improvements come primarily from covariance-weighting or adaptive scheduling.

2. Model scale and architecture diversity: Can the paper provide results on larger models (Llama-3-70B, Qwen-72B) where KV-cache is more critical? How does CARE perform on MHA to MLA conversion? It would also be helpful to see Table1-style results for other model families (Qwen, Mistral, Gemma) at comparable scales? For instance, for MoE models (e.g., Mixtral), do different experts need different rank allocations? Without broader architectural validation, it would be unclear if CARE is a general solution or specifically tuned to Llama-style architectures. 

3. Can the authors also evaluate on long-context benchmarks (LongBench, InfiniteBench, Needle-in-Haystack)? This would help see CARE's advantage scale with sequence length (1K, 4K, 8K, 16K, 32K tokens) and confirm if the covariance estimated on 2048-token windows (Appendix D) generalizes to longer contexts?

4.  Shrinkage parameter α: C_λ = (1-α)C + αλI was chosen for invertibility, it would help to have more details around how were α and λ chosen and include ensitivity analysis over α ∈ {0.001, 0.01, 0.1}.

### Soundness
2

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
4

### Summary
This paper presents a practical pipeline, called CARE (Covariance-Aware and Rank-Enhanced), to convert already-trained MHA/GQA attention in LLMs into MLA, while keeping the same KV-cache budget. The motivation is that many current methods only minimize weight-space error by a joint SVD on (W_K, W_V), and they use a uniform rank for all layers, which does not consider the real activation distribution at inference and makes some sensitive layers over-compressed. Experiments on Llama-3-8B and several benchmarks show that, under the same KV budget, CARE can keep lower PPL and better task scores than uniform-SVD style baselines.

### Strengths
- The problem is realistic and currently unsolved: we have many deployed MHA/GQA models, but we want the MLA advantages (smaller KV, more efficient decoding) without re-training from scratch.
- The paper does not only say “data-aware compression is better”, but it derives the form by considering the activation-space objective, which is more solid.
- The global and non-uniform rank allocation is motivated by data (different layers have different sensitivity) and the solution is standard and optimal under the budget.
- Experiments cover multiple tasks and multiple KV budgets, and they include the correct baselines, so the message is quite consistent.
- The design of KV-parity shows the authors think about real deployment and real KV bandwidth limitation.
- Reproducibility seems good, with anonymized repo and scripts mentioned.

### Weaknesses
1. The robustness of the covariance estimation is not fully shown. All calibration corpora are relatively similar; it is not clear whether for code or instruction-heavy data the same rank allocation is still good.
2. The relation to other data-/curvature-aware compression methods could be compared more directly on at least one setting.

### Questions
1. How robust is the covariance-aware step if the calibration set is distributionally far from the deployment set (e.g., code LMs but text-only calibration)? Do you recommend streaming/online covariance updates, or is healing enough?
2. Different tasks seemed to benefit differently at low KV ratios — could you expose task weights to the rank allocator to get multi-objective water-filling?

### Soundness
3

### Presentation
3

### Contribution
3
