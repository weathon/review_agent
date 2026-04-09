## Summary
AdaSVD proposes two complementary techniques for improving SVD-based LLM compression: **adaComp**, which compensates for truncation error via alternating Moore-Penrose pseudoinverse updates of the truncated singular matrices using calibration data, and **adaCR**, which assigns layer-specific compression ratios based on input-output cosine similarity. The method consistently outperforms prior SVD-based approaches (particularly SVD-LLM) across multiple LLM families and compression ratios, with the largest gains at high compression (60%+).

## Strengths
- **Principled truncation error compensation**: The adaComp mechanism (Eqs. 6–13) formulates a concrete optimization objective (∥ŨV̂⊤X − WX∥²_F) and solves it via stabilized alternating least squares using Moore-Penrose pseudoinverse, with empirical evidence (Figure 3a) showing smooth, monotonic error reduction compared to naive matrix inversion. This directly addresses a real gap—prior SVD methods perform truncation but do not adjust the retained factors.
- **Consistent and substantial improvements at high compression**: At 60% compression on LLaMA2-7B, AdaSVD achieves 50.33 PPL vs. SVD-LLM's 89.90 on WikiText-2 (Table 1), a 44% relative reduction. The gap widens precisely where deployment matters most (resource-constrained settings), which is where prior methods degrade catastrophically.
- **Demonstrated orthogonality with quantization**: Table 4 shows AdaSVD + GPTQ-INT4 outperforms SVD-LLM + GPTQ-INT4 across all compression ratios, and at 60–80% compression, AdaSVD+GPTQ even outperforms SVD-LLM without quantization. This is practically valuable for composing compression pipelines.
- **Stack-of-batch strategy**: A practical engineering contribution that enables use of more calibration data under fixed GPU memory (Figure 3b shows consistent error reduction), addressing a real implementation constraint.

## Weaknesses

### Major:
- **No inference efficiency measurements despite deployment claims**: The introduction explicitly claims SVD "can effectively accelerate model inference by reducing the memory requirements" (Page 2). Yet no latency, throughput, peak memory, or wall-clock inference measurements appear anywhere in the experiments. For a paper positioning itself around deployment on resource-constrained devices, this is a significant gap. SVD factorization replaces one large matmul with two smaller ones, which can *increase* latency on GPUs due to kernel launch overhead and reduced tensor core utilization. The paper must substantiate or retract the acceleration claim with actual benchmarks.

### Minor:
- **Potential mathematical issue in adaCR formula (Eq. 19)**: The retention ratio CR(W) = mrr + In(W)·(trr − mrr) can exceed 1.0 for layers with high normalized importance. Figure 4 shows max/min importance ratios up to 9.49 for Llama-7B, meaning In(W) could be well above 1 for the most important layers. For instance, with mrr=0.4, trr=0.6, and In(W)=5, the formula yields CR=1.4—an impossible retention ratio. The paper does not mention clipping or how this is handled. If clipping is applied, it changes the effective global compression ratio, violating the "fixed target compression ratio" guarantee; if not, the formula is invalid for extreme importance values. This needs explicit clarification.
- **Counter-intuitive importance metric with weak justification**: adaCR defines layer importance as cosine similarity between input X and output WX (Eq. 17), assigning higher retention to layers that preserve their input. However, a near-identity layer (high similarity) could be argued to be doing *less* transformative work and thus more redundant—contradicting some pruning literature that identifies highly transformative layers as critical. The paper provides no theoretical or empirical analysis justifying why high similarity implies high importance. While the method works empirically (Figure 4 shows the first layer is important, which is plausible), the metric's logic needs stronger defense.
- **Stack-of-batch is not equivalent to using more data**: Averaging calibration samples into buckets (Eq. 15) produces X'_k = mean of a batch, which yields a different second-moment structure than using all samples individually. The resulting covariance estimate discards within-batch variance. The paper claims this enables "more calibration data," but it actually uses a *summary* of the data that acts as implicit regularization. This distinction should be acknowledged rather than presenting the strategies as equivalent.
- **VLM evaluation is purely qualitative**: Figure 5 shows captioning examples but provides no quantitative metrics (CIDEr, BLEU, ROUGE) on the COCO dataset. The claim of "better image captioning results" is unsupported by standard evaluation.

### Trivial:
- **Compression time not reported**: adaComp requires iterative alternating updates (Table 3c tests up to 15 iterations). The wall-clock time for the compression phase itself is never reported, making it unclear how practical the method is relative to a single-pass SVD.

## Nice-to-Haves
- Inference latency/throughput benchmarks on target hardware (even a single GPU), which would substantiate the deployment narrative.
- Evaluation on at least one model at 13B+ scale to demonstrate scalability of adaCR's importance metric and adaComp's convergence behavior.
- Convergence analysis (theoretical or empirical loss curves) for the alternating update scheme to characterize overfitting risk with increasing iterations.
- Comparison with non-SVD compression methods (e.g., AWQ, GPTQ alone) at equivalent bit-budgets to contextualize where SVD-based methods stand in the broader compression landscape.
- Fine-tuning recovery experiments to assess whether AdaSVD-compressed models can regain performance with minimal additional training.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Comparison with non-SVD compression at equivalent bit-rates"** (from Spark Finder) — The paper's stated scope is improving SVD-based compression methods. Demanding comparison with quantization or pruning methods is scope creep. The paper already shows orthogonality with GPTQ (Table 4).
- **Weakness: "Ablation isolating adaComp vs. adaCR contribution is missing"** (from Spark Finder) — Factually incorrect. Table 3a shows AdaSVD with adaComp alone (adaCR=✗), and Table 3b shows the incremental benefit of adding adaCR. The individual contributions are explicitly isolated.
- **Weakness: "Missing related works"** — Not verifiable without external sources; removed per hard rules.
- **Weakness: "Formatting artifacts / notation inconsistency"** — Per instructions, parser artifacts are not paper problems. Minor notation issues are trivial and removed.
- **Weakness: "Low-rank weight compensation claim is too strong given LoRA"** (from Harsh Critic) — The claim is explicitly scoped to "SVD-based LLM compression" context; LoRA addresses a different problem (adaptation, not post-training compression). The claim is reasonable within its scope.
- **Weakness: "Missing Limitations section"** — Formatting/style requirement, not a substantive weakness.
- **Weakness: "Reproducibility concerns about calibration data"** — Per hard rules, reproducibility nitpicks about implementation details are removed.
- **Strength: "The paper is well-written / topic is important / experiments are extensive"** — Generic strengths weakened per soft rules.

## Novel Insights
The most interesting observation across the reviews is the tension in adaCR's importance metric: empirically it works (first layers are rated important, which aligns with intuition), yet theoretically the logic is inverted relative to some pruning perspectives (high input-output similarity could indicate redundancy rather than importance). This suggests the metric may be capturing something different from "transformative importance"—perhaps it measures *information preservation criticality*, where layers that faithfully pass information through the residual stream are more fragile to compression because any error propagates unattenuated. Disentangling these two interpretations could lead to a more principled importance metric and is a fruitful direction for future work.

## Suggestions
- Add a small table (even 3-4 rows) reporting inference latency and peak GPU memory for the original model vs. SVD-LLM vs. AdaSVD at matched compression ratios on a single GPU. This directly addresses the deployment narrative.
- Explicitly address the Eq. 19 overflow issue: either show that In(W) values stay within bounds for all tested models, add a clipping mechanism with analysis of its impact on the effective global compression ratio, or reformulate the equation.
- Provide quantitative VLM results (CIDEr or BLEU scores on COCO) instead of or in addition to the qualitative Figure 5.
- Report wall-clock compression time for adaComp with different iteration counts to help practitioners choose k.