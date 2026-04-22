Now I have all the information needed. Let me compile the final review.

## Summary

The paper introduces Weight-Activation Subspace Iteration (WASI), a method for efficient on-device fine-tuning of transformer models that jointly compresses model weights and activation maps using low-rank subspace iteration. By exploiting the stability of parameter subspaces during fine-tuning, WASI replaces expensive per-iteration SVD with warm-started subspace iteration, enabling both training and inference to operate entirely in a compressed low-rank representation. Experiments on ViT, SwinT, and TinyLlama demonstrate memory reductions of up to 62× and FLOPs reductions of up to 1.5× at accuracy-matching settings (ε=0.9), with a 1.4× wall-clock speedup on a Raspberry Pi 5.

## Strengths

- **Joint weight-activation compression is a genuine contribution.** Unlike LoRA (weights only, no inference savings) and ASI (activations only, no inference savings), WASI jointly compresses both in a unified low-rank framework, yielding compounding savings for both training and inference. Evidence: Eq. 8–10 show forward and backward passes operating entirely in the compressed subspace; Fig. 5 shows WASI achieves up to 100× higher memory efficiency than SVD-LLM at similar accuracy on ViT/CIFAR-10.

- **Subspace iteration provides concrete gains over naive re-SVD.** Fig. 3b demonstrates WSI requires 1.36× fewer FLOPs than per-iteration SVD at the same accuracy, and outperforms SVD by ~35% accuracy at the same FLOP budget, validating that subspace reuse does not harm convergence.

- **Real-device evaluation on Raspberry Pi 5.** Many efficiency papers report only theoretical FLOPs; this paper measures wall-clock time on actual edge hardware (Fig. 8), showing ~1.4× faster training and inference at ε=0.9.

- **Consistent accuracy-efficiency trade-offs across multiple architectures and datasets.** At ε=0.9, WASI matches vanilla accuracy on SwinT across five datasets while reducing memory by up to 62× and FLOPs by 1.5× (Fig. 6). Experiments span ViT, SwinT (vision), and TinyLlama (LLM).

- **Dynamic programming for rank selection** replaces ASI's exponential brute-force search with linear cost (Sec. 3.3, improvement (i)), a practical algorithmic improvement.

## Weaknesses

### Fatal
None.

### Major

- **The weight update mechanism (Eq. 11) is underspecified for a method paper.** Equation 11 states L_i R_i = L_i R_i + η · ∂L̂/∂W_i, which implies adding a gradient to the product L_i R_i. Since the gradient generally does not maintain the low-rank structure of L_i R_i (even with f_LR projection, the sum of two rank-K matrices can have rank up to 2K), it is unclear how the low-rank constraint is restored after each step. The operator f_LR in Eq. 9, which is critical for understanding this, is described only as "a linear operator applied in the low-rank space" and deferred to Appendix A.1. For a method paper whose core contribution is operating in the low-rank space, the main text must clearly explain how gradients are computed and applied while maintaining the rank constraint — i.e., whether updates are applied separately to L_i and R_i, whether f_LR projects the gradient to rank K_i, and whether the full weight matrix is ever materialized. Without this clarity, readers cannot verify that WASI actually achieves the memory savings it claims.

- **The TinyLlama experiment does not reliably support the claimed generality.** The experiment uses ε=0.1 (retaining only 10% of explained variance), which is extraordinarily aggressive and demands strong evidence. The paper claims "no accuracy loss" (line 190) but provides no actual accuracy number for this experiment in the main text. More critically, resource consumption is measured "only at the layers that are fine-tuned" (the last 5 layers), selectively excluding the frozen majority of the model. This inflates compression ratios by orders of magnitude — the 953.86× activation memory figure is not a meaningful claim about practical on-device memory savings when most of the model is unmeasured. The experiment as presented cannot substantiate the paper's claim that WASI generalizes to LLMs.

### Minor

- **Headline numbers reflect only MLP blocks, excluding attention layers.** The paper acknowledges this and defers attention-layer results to Appendix B.3, but the abstract and conclusions quote memory/FLOPs figures based solely on MLP measurements. For transformers, attention layers are major memory consumers; the practical memory savings will be substantially lower than the quoted 62× when the full model is considered.

- **Speedup is measured per-iteration, not per converged model.** If WASI requires more iterations to converge (a possibility with aggressive compression), total training time could differ. The paper does not provide convergence curves or total-epoch-to-target-accuracy comparisons. This is an addressable gap — even a simple plot of accuracy vs. epoch would clarify whether convergence rate is affected.

- **Stability assumption validated narrowly.** The key hypothesis that layer ranks remain stable is validated on a single configuration: ViT on Pets with ε=0.8 (Fig. 3a). While the main experiments implicitly validate that the method works across settings, explicit sensitivity analysis across learning rates, architectures, and datasets would strengthen the foundation.

- **Abstract "up to" claims are not simultaneously achievable.** The abstract states "up to 62× memory reduction and 2× FLOPs reduction." At ε=0.9 (accuracy-matching), the FLOPs reduction is 1.5×, not 2×. The 2× FLOPs figure comes from more aggressive settings where accuracy degrades. These two "up to" figures cherry-pick from different operating points and are not jointly realizable at the accuracy-matching setting.

### Trivial
None.

## Nice-to-Haves

- Comparison with LoRA or LoRA-FA on training memory and FLOPs. The paper argues LoRA is not directly comparable (it doesn't compress inference or activations), which is reasonable for the joint-compression claim. However, since the paper's title emphasizes "resource-constrained training," a LoRA comparison on training-specific metrics would be informative and strengthen the paper's positioning.

- Reporting absolute accuracy numbers for the TinyLlama experiment rather than claiming "no accuracy loss" without evidence.

- Convergence curves (accuracy vs. epoch) comparing WASI to vanilla training.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **SVD-LLM comparison is unfair**: The harsh critic argues that forcing SVD-LLM to use WASI's compression ratios is unfair. However, comparing at the same compression level is standard controlled experimental practice. Furthermore, SVD-LLM is designed for LLMs and the paper explicitly acknowledges this limitation; applying it to ViT already disadvantages SVD-LLM. Per the hard rule on unfair comparisons where the asymmetry favors the baseline, this criticism is removed.

- **WASI surpassing vanilla on CUB is "almost certainly" implicit regularization**: This is speculation and a common, well-understood phenomenon in compression literature. Low-rank constraints can act as regularizers — this is not a weakness but a well-known property.

- **Complexity analysis assumes same optimal rank for A_i and W_i**: The paper labels this a simplification explicitly ("For simplicity, we assume..."). The analysis is meant to provide intuition (Fig. 2), not exact prediction. The actual experiments use different ranks.

- **Subspace iteration overhead not analyzed**: This is a minor implementation detail; the paper empirically validates that WSI is efficient (Fig. 3b).

- **Grammar/formatting issues**: Removed per hard rules on parser artifacts.

- **Missing related works**: Removed per hard rules (cannot verify existence of suggested references).

- **Missing appendix proofs/details**: Removed per hard rules (parser strips appendices).

- **Reproducibility concerns about hyperparameters**: Removed per hard rules on nitpicks about undisclosed hyperparameters.

## Novel Insights

The joint weight-activation subspace iteration framework is a genuine structural advance over prior work that addresses only one or the other. The empirical evidence that subspace iteration (rather than full SVD per iteration) does not degrade convergence (Fig. 3b: WSI matches SVD accuracy with 1.36× fewer FLOPs) is the strongest specific result in the paper and directly validates the core stability hypothesis. The finding that WASI surpasses vanilla accuracy on CUB at ε=0.9, while not novel as a phenomenon (low-rank regularization), demonstrates that WASI's compression is aggressive enough to provide regularization benefits while still achieving major efficiency gains.

## Suggestions

- In the main text, replace Eq. 11 with explicit update rules for L_i and R_i separately (e.g., ∂L/∂L_i = ∂L̂/∂W_i · R_i^T and ∂L/∂R_i = L_i^T · ∂L̂/∂W_i), and explain f_LR concretely rather than deferring it. This single change would resolve the most significant reviewer concern.

- For the TinyLlama experiment, either (a) report full-model memory (including frozen layers) alongside fine-tuned-layer-only numbers, clearly distinguishing the two, or (b) state compression ratios only for the fine-tuned layers with an explicit caveat that practical memory savings depend on the fraction of fine-tuned layers. Include the actual accuracy number.

- Qualify the abstract's "up to 2× FLOPs" claim to indicate the ε setting at which this occurs, or rephrase to note that the 62× memory and 2× FLOPs figures correspond to different operating points.

## Score and Decision

**Calibration anchors:**

- **SubTrack-Grad** (avg 4.75, Reject): Similar topic (subspace tracking for efficient LLM training), similar weaknesses (underspecified method details, limited experimental validation). WASI has stronger empirical results (real device evaluation, multiple architectures) but similar clarity issues in the core algorithm.

- **Dobi-SVD** (avg 6.2, Accept Poster): SVD-based compression with differentiable truncation. More clearly specified method and theoretical analysis. WASI is weaker in method clarity but has real-device evaluation.

- **LDAdam** (avg 7.0, Accept Poster): Low-rank adaptive optimizer with convergence proofs and clear method specification. WASI lacks this theoretical depth and methodological clarity.

- **"Few Heads are Enough"** (avg 5.67, Reject): Underspecified method section, small-scale experiments. WASI has better empirical grounding but similarly unclear core algorithm description.

- **Tensor-GaLore** (avg 5.25, Reject): Extension of prior work to tensor setting, somewhat straightforward. WASI has more novelty but similar concerns about method specification.

- **LoRAM** (avg 6.2, Accept Poster): Memory-efficient LoRA training with clear presentation and practical results.

WASI sits between SubTrack-Grad (4.75) and Dobi-SVD/LoRAM (6.2). It has genuinely novel contributions (joint weight-activation subspace iteration) and strong empirical evidence (real device, multiple architectures), but the underspecified core update rule (Eq. 11) and unreliable TinyLlama experiment are significant enough to weigh against acceptance. The paper is above SubTrack-Grad because of its real-device evaluation and more extensive experiments, but below Dobi-SVD/LoRAM because of the methodological clarity gap. A score of 5 reflects this position.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>