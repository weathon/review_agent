Now let me read some of the calibration papers referenced by the Human Finder reviewer.Now I have enough context to write the consolidated final review. Let me synthesize everything.

---

## Summary

Cut Cross-Entropy (CCE) proposes a custom GPU kernel that computes the cross-entropy loss for large-vocabulary language models without materializing the full logits matrix (of size N×|V|) in global memory. By decomposing cross-entropy into an indexed matrix multiplication and a blockwise log-sum-exp, both computed in SRAM, CCE reduces loss-layer memory from 24 GB to 1 MB for Gemma 2 (2B), enabling 1.5×–10× larger batch sizes in realistic multi-GPU setups. The paper additionally introduces gradient filtering (skipping softmax entries below bf16 precision) and vocabulary sorting to improve backward-pass throughput, and validates convergence for both fine-tuning and limited pretraining experiments.

---

## Strengths

- **Eliminates the dominant training memory bottleneck for large-vocabulary LLMs with near-zero overhead.** Table 1 shows CCE achieves 1 MB loss memory vs. 24 GB baseline, with total head memory of 1.16 GB vs. 28 GB — hitting within rounding error of the theoretical lower bound. This directly addresses a real and worsening problem as vocabulary sizes grow beyond 256K.

- **Runtime is competitive with the fastest existing baseline (`torch.compile`) rather than sacrificing speed for memory.** CCE computes loss slightly faster (46ms vs. 49ms) and loss+gradient within 2ms (145ms vs. 143ms) of `torch.compile`, while using orders of magnitude less memory. This distinguishes CCE from chunking-based alternatives like Liger (304ms) and TorchTune (169ms).

- **Principled and well-analyzed algorithmic decomposition.** The reformulation of CCE into two independently optimizable sub-operations (Eq. 4), plus the detailed blockwise algorithms (Algs. 1–3) with access-pattern diagrams (Fig. 2), make the contribution technically transparent and reproducible.

- **Gradient filtering insight is empirically grounded.** Fig. 3 demonstrates convincingly that average softmax probabilities fall below bf16 precision by the ~50th most likely token, explaining the 3.5× speedup from filtering. The ablation in Table 1 rows 1 vs. 7 isolates this effect clearly.

- **Exemplary candor about pretraining failure modes.** Section 5.3 explicitly reports that the default CCE degrades pretraining perplexity due to (i) gradient filtering suppressing rare-token gradients via ∇C, and (ii) bf16 summation precision loss. The authors then fix both issues with CCE-Kahan-FullC and re-validate. This transparency strengthens trust in the remaining empirical claims.

- **Open-source release and concrete end-to-end benefit demonstrated.** The Mistral NeMo experiment shows CCE-Kahan-FullC enabling a 2× batch size increase that reduced total training time by 16%, providing one concrete end-to-end throughput benefit beyond isolated kernel timings.

---

## Weaknesses

### Fatal
*(None identified.)*

### Major

- **The abstract's "without sacrificing convergence" claim is too strong and is partially contradicted by the paper's own Section 5.3.** The abstract reads: *"Experiments demonstrate that the dramatic reduction in memory consumption is accomplished without sacrificing training speed or convergence."* Yet Section 5.3 states explicitly: *"In our initial experiments using CCE for pretraining, we found that validation perplexity suffered due to two sources of error."* The default CCE variant (the one benchmarked in Table 1) is not safe for pretraining. Only CCE-Kahan-FullC is shown to be stable — a slower variant that does not appear in the headline abstract claim. The abstract and introduction create a misleading impression that a single drop-in method works universally; the paper should clearly communicate that pretraining requires a separate, slower, and slightly more memory-hungry variant.

- **Pretraining validation scale is insufficient for the scope of the claims.** The pretraining experiment in Section 5.3 covers only 5% of OpenWebText for ~1500 gradient steps. This is enough to detect obvious optimization pathology, but the paper's identified failure modes — rare-token gradient suppression and summation precision — are exactly the kind of issues that can have subtle long-horizon effects. Given that the paper explicitly acknowledges these failure modes for the default variant, demonstrating that CCE-Kahan-FullC is truly safe requires more than a short run on a 5% data subset. The claim that CCE-Kahan-FullC *"produces identical curves as `torch.compile`"* (Section 5.3) should be scoped to the evaluated run length and data scale.

### Minor

- **Gradient filtering threshold ε = 2⁻¹² lacks sensitivity analysis.** The threshold is motivated heuristically via bf16 precision (Section 4.3, footnote 1), but there is no systematic sweep over threshold values (e.g., 2⁻¹⁰ to 2⁻¹⁵) reporting both runtime and convergence effects. Practitioners using different precisions or model families need to know whether this choice is robust or sensitive.

- **End-to-end multi-GPU training throughput is not systematically demonstrated.** All kernel timings are isolated single-GPU measurements on an A100-SXM4. The Mistral NeMo end-to-end example is a single data point. The paper makes broad claims about FSDP batch-size increases (Fig. 1), but these are modeled from memory accounting, not measured. A systematic multi-GPU throughput comparison would directly support the "no sacrifice in speed" claim at the training system level.

- **Vocabulary sorting overhead is not fully characterized.** Section 4.3 describes using an O(|V|) atomic-addition buffer during the forward pass to track average logits, then sorting for the backward pass. Table 1 rows 1 vs. 6 isolates the effect on *backward* time (100ms vs. 115ms), but the forward-pass overhead of accumulating average logits during training is not separately reported. The ~1 MB buffer is small, but the compute cost of this heuristic over many training steps is unquantified.

- **Rare-token gradient suppression from ∇E filtering in CCE-Kahan-FullC is unanalyzed.** Section 5.3 correctly identifies that filtering ∇C hurts pretraining and disables it in CCE-Kahan-FullC. However, gradient filtering on ∇E is still active in CCE-Kahan-FullC (Table 1 rows 9 vs. 8, FullC removes filtering only from ∇C). The paper does not analyze whether rare tokens that appear infrequently in a training batch could still lose embedding gradient signal through ∇E filtering over long pretraining runs.

### Trivial

- **The merged backward algorithm (Algorithm 4) is only referenced, not shown.** The main text (end of Section 4.3) notes that the indexed matrix-multiplication backward is merged with the log-sum-exp backward and defers to *"Algorithm 4"* in the appendix, which is not included in the reviewed text. Since this merged pass is a key implementation detail for the claimed runtime efficiency, at minimum a brief inline description in the main text would help readers.

---

## Nice-to-Haves

- Downstream task evaluation (e.g., MMLU, HellaSwag) for models trained with CCE-Kahan-FullC would provide stronger evidence of functional equivalence beyond perplexity curves.
- A heatmap or visualization of softmax sparsity patterns before and after vocabulary sorting would validate the mechanistic claim for that optimization.
- Evaluation on H100 or consumer-grade GPUs (RTX 4090) would help practitioners on non-A100 hardware assess whether the SRAM-heavy kernel strategy transfers.
- A characterization of the |V|/D crossover point below which CCE's speed advantage disappears (mentioned in Appendix C but not in the main paper) would guide practitioners choosing whether to adopt CCE.

---

## Removed Points

*These points are flagged as removed; treat them with caution.*

**"Incomplete comparison with Liger Kernels [as a weakness]"** — The paper explicitly states (Section 2) that Liger's fused design requires user-defined loss transformations to be implemented inside the kernel, a genuine design constraint. This is a legitimate design tradeoff, not a missing comparison. The runtime comparison in Table 1 is fair: Liger trades flexibility for a fused pass. REMOVED as a weakness; the paper adequately acknowledges the distinction.

**"Single GPU type evaluation (A100 only)"** — This is standard practice for custom kernel papers at ICLR. FlashRNN, ThunderKittens, and FlashMask all evaluate primarily on a single GPU family. MOVED to nice-to-have.

**"Fig. 1 batch-size estimates are modeled, not measured"** — The paper describes this as a memory breakdown for a specific FSDP setup (Figure 1 caption) and provides exact values in Table A4. The modeling is transparent. REMOVED as a standalone weakness; it is partially addressed in the Minor section on end-to-end throughput.

---

## Novel Insights

The most genuinely novel observation is the **gradient filtering insight applied to cross-entropy**: because the softmax sums to one and bf16 has a 7-bit fraction, no more than 4096 entries per token can have non-negligible gradient contributions, and in practice far fewer (≪0.02%) do. This turns the backward pass of cross-entropy from a dense into a highly sparse operation, yielding a 3.5× speedup. Critically, the paper further shows that the sparsity of this set **grows** as vocabulary size increases, meaning CCE's computational advantage will compound as vocabularies expand — an anti-scaling property for the baseline that CCE specifically exploits.

---

## Suggestions

1. **Revise the abstract and introduction** to clearly distinguish default CCE (fine-tuning/memory experiments) from CCE-Kahan-FullC (pretraining), and scope the convergence claim to the respective settings.
2. **Provide sensitivity analysis for ε**, reporting convergence and runtime for at least three threshold values to demonstrate robustness.
3. **Extend pretraining validation** to at least one run covering the full OpenWebText dataset or equivalent (~35B tokens), or clearly limit the convergence claim to the short-run regime.
4. **Report multi-GPU end-to-end training throughput** (tokens/sec) for at least one model in an FSDP setup to validate the claimed batch-size improvements translate to real-world speedups.
5. **Analyze ∇E filtering impact on rare tokens** in CCE-Kahan-FullC, or disable ∇E filtering as well and report the cost, to fully close the numerical gap analysis.

---

## Score Calibration and Decision

**Calibration anchors:**
- **FlashMask** (FlashAttention mask extension, ICLR 2025): scores 8,8,6,6 → Accept Poster. Broader scope than CCE but similar IO-aware kernel engineering quality and similar training validation scale.
- **FlashRNN** (IO-aware RNN kernels, ICLR 2025): scores 6,8,6,6 → Accept Poster. Similar technical approach (Triton, single operator, SRAM-based), but addressing a more niche task than the universally present cross-entropy layer.
- **ThunderKittens** (kernel framework, ICLR 2025): scores 6,8,8,8 → Accept Spotlight. Much broader scope and stronger performance claims across multiple operators; CCE is more targeted.
- **Memory-Efficient Backprop through Large Linear Layers** (ICLR 2025): scores 3,3,6,3 → Reject. Lacked scale in experiments, limited novelty; CCE is clearly stronger on all axes.

**Positioning:** CCE's core contribution is more targeted than ThunderKittens but its practical significance is arguably higher (cross-entropy is a universal bottleneck for all LLMs; RNN kernels serve a niche). The memory savings are more dramatic and better demonstrated than FlashRNN. The main weakness — limited pretraining scale and the framing overclaim — is real but does not undermine the core systems result. The paper is honest about its limitations in a way that strengthens rather than undermines confidence in the results. This places CCE solidly in the Accept range, consistent with FlashMask/FlashRNN, with the pretraining limitations preventing a Spotlight-level assessment.

**Axis evaluations:**
- *Novelty*: Moderate-to-high — the blockwise log-sum-exp kernel is inspired by FlashAttention but the gradient filtering insight and its combination with vocabulary sorting for sparse backward passes is a genuinely new contribution to the cross-entropy computation literature.
- *Technical soundness*: High — the algorithmic decomposition is mathematically clean and the implementation strategy is carefully engineered.
- *Empirical support*: Good for the core memory/speed claims; adequate-but-limited for convergence, especially pretraining.
- *Significance*: High — cross-entropy memory is a universal bottleneck in all LLM training, and the savings are dramatic.
- *Clarity*: Good overall, with the main clarity gap being the insufficient distinction between CCE variants in the abstract and introduction.

**Final Score: 7.0 — Accept (Poster)**

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>