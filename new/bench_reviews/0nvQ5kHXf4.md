Let me look at the paper content provided in the user message more carefully. I'll check the key claims made by the harsh reviewer about the algorithm being underspecified.

Looking at the paper text provided:

1. **Algorithm 1 (WSI)**: The text says:
   - At t=0: `L_{i,(t)}, R_{i,(t)} = SVD(W_{i,(t)}, ε)` (see Eq. 5, 6, 7)
   - At t>0: `R_{i,(t)}^T = W_{i,(t)}^T · L_{i,(t-1)}` and `L_{i,(t)} = Orthogonalize(W_{i,(t)} · R_{i,(t)}^T)` 
   
2. **Eq. (11)**: "L_i R_i = L_i R_i + η · ∂L̂/∂W_i" — this is indeed written in a way that looks like W = W + η·∂L/∂W, which is a standard gradient update equation. It's not dimensionally inconsistent if ∂L/∂W_i is computed in the low-rank space (i.e., it's the gradient with respect to the low-rank factors).

3. **f_LR**: "f_LR(·) denotes a linear operator applied in the low-rank space (see Appendix A.1)" — referenced to Appendix A.1, which is not provided in the main text.

4. **"Pre-tuning perplexity"**: Sec. 3.3 mentions "a dynamic-programming strategy that determines r_i by minimizing memory usage under a target pre-tuning perplexity, rather than relying on a fixed budget B". This is referenced to Appendix A.2.

Now let me construct the final review.

Key observations:

**On the harsh critic's claims:**

1. **Ambiguity of ε control**: The critic raises valid points about the gap between ε and actual memory bounds. The paper does use ε as a proxy for information loss but doesn't provide a principled mapping from ε to memory usage. However, the paper does provide empirical plots (Figs. 5, 6) showing the trade-off. The claim about "no worst-case memory guarantee" is valid but may be overly strict for an empirical systems paper — many practical on-device methods use heuristic knobs. I'll keep this as a moderate weakness.

2. **Incomplete algorithmic specification**: The algorithm description has some issues — the OCR artifacts make Algorithm 1 harder to read, but the key question about whether W is ever reassembled is legitimate. Eq. (11) is written ambiguously. The f_LR operator is deferred to the appendix. However, I should note that this is a common pattern in ML papers — appendix details are normal. The core question of whether the full W is ever materialized is partially addressed by the memory savings claim (if W were materialized, there would be no savings). I'll keep a moderate version of this concern.

3. **Fairness of baseline comparisons**: The paper compares with ASI, SVD-LLM, and vanilla training. The absence of LoRA is notable since the paper specifically discusses LoRA's limitations. However, LoRA and WASI serve different purposes — LoRA adds parameters and doesn't reduce inference cost, while WASI reduces both. The comparison might not be fully apples-to-apples but LoRA is a natural baseline. I'll flag LoRA's absence as a moderate weakness.

4. **Generalization beyond ViT**: Valid concern — the TinyLlama experiment is very limited. However, the paper's main claim is about enabling on-device training of transformers, and doing so on ViT/SwinT for vision + a small LLM probe is reasonable for a first paper. I'll keep this as a minor-moderate weakness.

**On the neutral reviewer's claims:**

1. **Limited LLM evaluation**: Valid, but the paper acknowledges this is an extension experiment.
2. **Missing LoRA baselines**: Valid point.
3. **Narrow validation of subspace stability**: Valid but the paper provides some empirical evidence.

**On the spark reviewer's claims:**

1. **No direct LoRA comparison**: Valid.
2. **Attention layers in appendix**: Valid — but the paper says they're included in Appendix B.3, not omitted.
3. **TinyLlama limitations**: Valid.
4. **No convergence analysis**: Valid concern.

**On the human finder patterns:**

- Several papers in this space were rejected (LLRC: 3,5,5,5; Fira: 6,6,6,5; WeLore: 5,3,6,6; SubTrack-Grad: 5,5,6,3; ASVD: 6,8,6,5 → rejected)
- Some were accepted (LORO: 6,6,6,6; SVD-LLM: 8,5,6,6; Dobi-SVD: 8,5,6,6,6; AdaRankGrad: 8,6,8,5,8)
- ReLoRA: 5,6,6,6 → Accept (poster)

The WASI paper is somewhat similar to these. Its novelty is moderate — subspace iteration for weights + activations is a clear contribution building on ASI. The empirical results are solid on ViT/SwinT but limited for LLMs. The algorithm description has some gaps. Missing LoRA comparison is a notable gap.

Overall, I'd place this paper in the low-to-mid range — it has a good idea and solid ViT results, but the algorithm isn't fully specified, baselines are limited, and LLM claims are overstated. Similar to where Fira (6,6,6,5 → Reject) or WeLore (5,3,6,6 → Reject) landed, but perhaps slightly better due to the on-device real-hardware results.</think>Now I have a thorough understanding of both the paper and the reviewer landscape. Let me synthesize the final review.

## Summary

WASI proposes Weight-Activation Subspace Iteration, a method that simultaneously compresses transformer model weights and activation maps during fine-tuning by exploiting the hypothesis that essential parameter information lies in a stable low-rank subspace. By applying subspace iteration rather than repeated full SVDs and jointly compressing both weights and activations, WASI achieves up to 62× training memory reduction and 1.4× speedup on a Raspberry Pi 5 over vanilla training, with accuracy comparable to vanilla fine-tuning on vision transformer benchmarks.

## Strengths

- **Addresses a genuine gap**: Prior activation-compression methods (ASI, AMC) leave full-rank weights untouched, while LoRA-style methods don't reduce inference cost. WASI jointly compresses both weights and activations and maintains the compressed representation for inference — this is a legitimate and important contribution for the on-device learning setting (Sec. 3.3, Eqs. 8–10).

- **Real hardware evaluation**: The Raspberry Pi 5 experiment (Sec. 4.4, Fig. 8) showing 1.4× speedup over vanilla even at ε=0.9 provides concrete evidence of practical benefits. Many efficiency papers only report theoretical FLOPs; this paper validates on actual edge hardware.

- **Substantial empirical results on ViT/SwinT**: The trade-off curves in Figs. 5–6 demonstrate that WASI matches vanilla accuracy (e.g., on CUB it surpasses vanilla) while cutting memory by up to 62× and FLOPs by up to 2×. The consistent improvement across five datasets on SwinT (Sec. 4.3) is compelling.

- **Efficient subspace tracking**: The WSI method (Algorithm 1) replaces full SVD at every iteration with warm-started subspace iteration. The validation in Fig. 3b showing WSI uses 1.36× fewer FLOPs than repeated SVD at equal accuracy is a meaningful algorithmic efficiency result.

## Weaknesses

### Major:

- **Incomplete algorithmic specification for the core update rules**: The paper never fully specifies how gradients propagate through the low-rank factorization in a way that avoids materializing full-rank tensors. Eq. (11) reads "L_i R_i = L_i R_i + η · ∂L̂/∂W_i" — but this is ambiguous: are the gradients computed with respect to L and R separately (preserving low-rank structure), or is W reconstructed in full for the backward pass? The operator f_LR(·) in Eq. (9) is the key to this question but is only described as "a linear operator applied in the low-rank space (see Appendix A.1)" with no concrete definition in the main text. Since the claimed memory and FLOPs savings depend on never reassembling W or A in full, this is not a cosmetic gap — it is central to verifying the method's correctness and efficiency claims.

- **Limited baseline comparisons — no LoRA or GaLore-type methods**: The paper extensively discusses LoRA's limitations (added memory for adapters, unchanged inference cost) in Sec. 1–2, yet never benchmarks against LoRA, QLoRA, GaLore, or recent gradient-projection methods. For on-device fine-tuning, LoRA (with activation checkpointing) is the standard baseline, and the reader cannot assess whether WASI's architecture-level compression actually outperforms the dominant parameter-efficient approach. SVD-LLM is compared, but it is designed for post-training LLM compression, not on-device training — making it an incomplete comparison for the claimed use case.

- **Subspace stability assumption is only weakly validated**: The paper's WSI method depends on the claim that weight subspaces remain stable during fine-tuning, yet Fig. 3a only shows rank stability for a single model (ViT), a single dataset (Pets), and a single ε value (0.8). Rank stability is weaker than subspace stability — the same rank can span a different subspace. No experiment tests whether the subspace (singular vectors) drifts over training, nor whether the method is sensitive to different fine-tuning regimes (larger learning rates, distribution shifts, longer training).

### Minor:

- **TinyLlama experiment is too limited to support generality claims**: Only the last 5 layers of TinyLlama are fine-tuned, at a single ε=0.1, on one dataset (BoolQ). No comparison with any LLM-specific method is provided. The paper claims WASI applies broadly to "any neural network trained with backpropagation" based on this minimal probe, which is an overstatement.

- **No convergence curves or training dynamics**: The paper only reports final accuracy, never showing training loss curves for WASI vs. vanilla or other methods. If WASI converges more slowly (requiring more epochs to reach comparable accuracy), the FLOPs-per-epoch savings could be offset by more total epochs needed — and the reader cannot assess this.

- **No standard deviation or multiple-seed reporting**: All experimental results appear to be single-run. For small fine-tuning tasks where random seeds can shift accuracy by 1-2%, this makes it harder to distinguish genuine improvements from noise.

- **Metric scope for on-device claims**: Memory and FLOPs are reported only for linear layers in MLP blocks, with attention-layer results deferred to the appendix. For on-device deployment, total memory and compute matter; reporting only MLP-block costs gives an optimistic picture of savings.

## Nice-to-Haves

- Comparison with LoRA + activation checkpointing on the same hardware, reporting both training memory and inference latency, to clarify when WASI is preferable
- Attention-layer results in the main paper, since attention projections are a core source of memory/compute in transformers
- Convergence dynamics (training loss curves) for WASI vs. vanilla vs. LoRA
- Sensitivity analysis of ε across models and tasks, with guidance for practitioners on choosing ε

## Removed Points

- **"The method lacks a rigorous mapping from ε to a hard memory budget guarantee"**: While true, many practical on-device methods use heuristic knobs. The paper explicitly sweeps ε and shows empirical trade-off curves. Demanding a hard worst-case guarantee is scope creep for an empirical systems paper — the ε sweep is informative enough for practitioners.

- **"SVD repeated at each step is a strawman baseline"**: The paper compares WSI against per-step SVD to validate the subspace iteration approximation specifically, which is the methodological contribution. This is a targeted ablation, not a claim that practitioners would run per-step SVD.

- **"Pre-tuning perplexity is undefined for vision tasks"**: The paper references the details to Appendix A.2. The concept of using a proxy metric to set rank budgets is a reasonable engineering choice even if the specific definition is in the appendix.

- **"Attention layers excluded from main results"**: The paper explicitly states that attention layer results are in Appendix B.3, not omitted. Including them in the main paper would be preferable, but their absence from the main body is not a methodological flaw.

- **"No comparison with quantization-based methods (QLoRA, INT8)"**: WASI is a complementary low-rank approach. Whether it can be combined with quantization is an interesting future direction but not within the paper's scope. The paper evaluates low-rank decomposition methods, and mixing paradigms without careful co-design would not be a fair comparison.

- **"LoRA is plug-and-play while WASI modifies the architecture"**: This is not a weakness — it is the explicit design choice of WASI. The paper's contribution is precisely that inference runs on the compressed model, which LoRA cannot do. This asymmetry favors the authors' method.

- **"Paper claims 953× memory reduction for TinyLlama activations at ε=0.1 without reporting accuracy"**: While the ε=0.1 setting is aggressive, the paper states "without accuracy loss" (Sec. 4.3). The figure is in the paper, and the claim should be evaluated on its merits rather than dismissed for the extremity of the compression ratio.

## Novel Insights

The most interesting insight in the paper is the empirical observation that even when fine-tuning modifies weights, the subspace (rank) needed to meet a fixed explained-variance threshold remains remarkably stable (Fig. 3a). This motivates the use of subspace iteration — initialized from a single SVD and refined via Gram-Schmidt at each step — rather than recomputing full decompositions. While the theoretical grounding is thin, the engineering intuition that fine-tuning with small learning rates produces weight updates that are approximately low-rank perturbations is well-motivated and worth formalizing further.

## Suggestions

- Provide a complete, self-contained algorithmic description of the WASI update (including how f_LR is computed and how gradients flow through L and R without materializing W) in the main text — this is the single most important revision needed
- Add LoRA (with activation checkpointing) as a baseline on at least one ViT/SwinT experiment, comparing training memory, training FLOPs, and inference latency side-by-side
- Show training loss curves (WASI vs. vanilla vs. LoRA) to verify that convergence speed is not significantly degraded
- Report results with multiple random seeds to establish that accuracy differences are meaningful

## Score and Decision

**Calibration**: I compared this paper against several low-rank training/compression papers reviewed at ICLR:
- **WeLore** (subspace stability for LLM compression): Scores 5,3,6,6 → Reject — similar subspace stability motivation but limited empirical validation
- **Fira** (low-rank training of LLMs): Scores 6,6,6,5 → Reject — similar scope (efficient low-rank training), stronger LLM experiments but with theoretical concerns
- **SubTrack-Grad** (gradient subspace tracking): Scores 5,5,6,3 → Reject — similar idea of avoiding repeated SVDs, with similar concerns about missing memory measurements
- **LORO** (Riemannian optimization for low-rank pretraining): Scores 6,6,6,6 → Accept (Poster) — stronger algorithmic specification and comparable experiments on LLMs
- **SVD-LLM** (trained SVD for LLM compression): Scores 8,5,6,6 → Accept (Poster) — more thorough evaluation across model families

WASI is weaker than LORO and SVD-LLM (which had better algorithmic clarity and broader evaluation) but comparable to or slightly stronger than WeLore and Fira (which were rejected). WASI has real hardware results (Raspberry Pi) and solid ViT/SwinT experiments, but the algorithm specification is incomplete, baselines are narrow, and LLM generality claims are unsupported. I place it below the accept threshold — roughly where Fira and SubTrack-Grad landed.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>