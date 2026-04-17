Now let me look at some calibration papers:Now I have enough information to write the final review. Let me synthesize all inputs carefully.---

## Summary

QZO (Quantized Zeroth-Order Optimization) proposes a method that combines model quantization with zeroth-order optimization by perturbing continuous quantization scales rather than discrete weights, avoiding the incompatibility between ZO's continuous gradient estimates and quantized integers. A companion technique, Directional Derivative Clipping (DDC), is proposed to stabilize training. The method achieves an 18× memory reduction over full 16-bit fine-tuning, reportedly enabling fine-tuning of Llama-2-13B on a single 24GB GPU, and is shown to work with both scalar-based (GPTQ) and codebook-based (AQLM) quantization schemes.

---

## Strengths

- **Elegant and practical core idea**: Perturbing continuous quantization scales instead of discrete weights is a non-obvious but well-motivated solution to a real incompatibility. The parameterization `θ = Δ ⊙ θ̄` is leveraged cleanly and extends naturally to multiple PTQ families without modification to quantized weights.

- **Substantial memory reduction with competitive performance**: QZO achieves 4.8–6.3 GB peak memory for 7B-class models (vs. 14.8 GB for MeZO and 26–32 GB for full fine-tuning), and on most tasks performs on par with or better than MeZO despite using 3× less memory and only ~1% of trainable parameters.

- **Breadth of compatibility**: Demonstrated on OPT-6.7B, Llama-2-7B, Llama-3.1-8B, and Llama-2-13B; compatible with GPTQ (4-bit) and AQLM (2-bit); even includes a diffusion model application (SD 3.5 Large in the Appendix), showing genuine generality.

- **DDC ablation is clear and convincing**: Figure 2 directly demonstrates training collapse without DDC, providing strong empirical motivation for the component.

- **Reproducibility**: Code is publicly released, datasets are public, and hyperparameters are clearly reported.

---

## Weaknesses

### Fatal
*(None: the paper has real contributions and is not fundamentally broken, but it does have major gaps.)*

### Major

- **Missing QLoRA comparison** — QLoRA (Dettmers et al., 2023, cited in the paper's own references) is the de-facto standard for memory-efficient fine-tuning of quantized LLMs, combining 4-bit weights with a small set of high-precision LoRA adapters while requiring backpropagation only through the low-rank adapters. QZO explicitly targets a similar use case and similar GPU memory budget, yet there is no comparison against QLoRA in Tables 1–3. Without this, readers cannot judge whether QZO's trade-offs (no backprop, scale-only updates, lower accuracy on some tasks) are worthwhile relative to the most widely used alternative. The omission is particularly glaring because QLoRA is already cited, meaning the authors are aware of it.

- **Questionable theoretical foundation of DDC (Theorem 1 unbiasedness)**: Theorem 1 claims that the *clipped* gradient estimate `∇̂_Δ L'(Δ⊙θ̄; B) = d'·z` is an *unbiased* estimate of the full gradient `∇_Δ L`. This claim is non-trivial: since `d = [L(Δ+ϵz) − L(Δ−ϵz)] / (2ϵ)` depends on z, `d` and `z` are not independent, and clipping a scalar that depends on the direction vector z does not generally preserve unbiasedness. The variance-reduction argument (Eq. 8) critically uses this unbiasedness in its final step — if E[`∇̂_Δk L'`] ≠ `∇_Δk L`, then the inequality in Eq. 8 does not simplify as claimed, and the theoretical justification for DDC collapses. The proof is delegated to Appendix A but is not presented in the main text, making it impossible to verify. DDC is empirically effective and well-motivated, but the theoretical justification as written is at risk.

- **No comparison against any gradient-based PEFT baseline for quantized models**: The paper frames QZO as an advance in *memory-efficient fine-tuning of quantized LLMs*. Yet the primary comparisons are full 16-bit fine-tuning (an explicitly unreasonable memory bound), MeZO on full 16-bit models, and zero-shot lower bounds. The absence of PEFT alternatives (QLoRA, LoRA with gradient checkpointing) with matched memory budgets leaves the Pareto trade-off between performance and memory completely uncharacterized. Practitioners need this to make informed choices.

- **Limited benchmark scope**: All five tasks (SST-2, RTE, CB, BoolQ, SQuAD) are relatively simple classification/extraction benchmarks from the pre-LLM era and directly mirror MeZO's evaluation setup. Tasks that genuinely stress the model's reasoning capabilities — e.g., MMLU, GSM8K, or instruction-following — are absent. For a method whose practical value is adapting large LLMs to downstream tasks, the scope of the evaluation understates the extent to which scale-only updates may be insufficient for harder adaptations. As noted in reviews of comparable ZO papers, these benchmarks can be handled by much smaller models (e.g., RoBERTa-large) and do not demonstrate the full practical utility of large-model adaptation.

### Minor

- **No multi-seed evaluation or variance reporting**: Results are single-run numbers on datasets of 1,000 training examples. ZO methods are high-variance, and some reported differences (e.g., QZO 85.5 vs MeZO 80.7 on SQuAD for Llama-2-7B) could plausibly be attributable to run-to-run variation. Following standard practice in this sub-field, mean ± std over at least 3 seeds should be reported.

- **No training curves or wall-clock time**: The paper reports total FLOPs and per-step cost (Table 2), but never shows training loss curves or wall-clock time to a target performance level. Since ZO methods converge slowly, the per-step FLOP advantage of QZO over MeZO may be offset by requiring more steps. Convergence curves are necessary to evaluate real-world efficiency.

- **2-bit experiments (Table 3) have no trained baseline**: Only Zero-Shot-Q is compared against QZO for Llama-2-13B in the 2-bit setting. Without any trained baseline (even MeZO applied to the same model after de-quantization, or a QLoRA-style adapter), it is impossible to contextualize QZO's gains — they may be trivially achievable.

- **Expressiveness of scale-only updates is unanalyzed**: QZO updates ~1% of parameters (the quantization scales). The paper does not analyze which types of adaptation this supports or where it systematically fails. Some tasks show large gaps to full fine-tuning (e.g., Llama-3.1-8B CB: 69.6 vs MeZO 91.1; RTE Llama-2-7B: 66.8 vs fine-tuning 71.5). Understanding whether these gaps reflect the limits of scale-only expressiveness versus ZO estimation quality would be valuable for practitioners.

### Trivial

- **Typo in ablation text (Section 4.3)**: *"QZO w/ DDC can be seen as setting C to an infinitely large value"* should read *"QZO w/o DDC"* — with DDC active, C is finite; without DDC, C is effectively infinite.

---

## Nice-to-Haves

- **Perturbation scale ε sensitivity analysis**: The paper ablates clipping threshold C but never studies ε, which also controls gradient approximation quality. A brief ablation would reassure readers of robustness.

- **Visualization of scale parameter evolution during training**: Showing how Δ values change across layers and training steps would illuminate whether QZO is making distributed, semantically meaningful updates or primarily adjusting a few outlier scales.

- **Harder generation benchmarks**: Testing on instruction-following or summarization with ROUGE/BLEURT metrics would strengthen the case for broader practical utility.

- **Wall-clock efficiency analysis**: Given QZO's 2× forward passes per step with smaller perturbation space, a time-to-convergence plot comparing QZO vs MeZO would complete the computational efficiency picture.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic — "Fine-tuning quantized weights is misleading/structural flaw"**: The paper is transparent throughout: it states "perturbs the continuous quantization scale parameter(s) rather than the discrete weights, which are kept fixed throughout training" (Section 3.2.1) and explicitly calls QZO "fine-tuning quantized neural networks" in the sense that scales are genuine parameters of a quantized model. The naming is defensible, and the paper never claims to update discrete integer codes. The critic's characterization of this as "not conceptually new" or "very close to LoRA" conflates parameterization with optimization mechanism; the ZO-without-backprop property is the key novelty. **REMOVED** as a misread.

**Harsh Critic — "18× figure not backed by fair memory accounting"**: The paper follows MeZO's own memory-profiling protocol (batch size 1, 100 steps, SST-2). This is the established standard for this sub-field and explicitly noted. Batch size 1 is used to report *minimum* VRAM requirements, which is a legitimate practical claim. The concern about activations at larger batch sizes is noted as a nice-to-have (the paper acknowledges this in Section 3.2.3) but does not invalidate the reported minimum. **WEAKENED** to the QLoRA comparison gap, which is kept as a major weakness.

**Harsh Critic — "Comparison unfair due to different trainable parameter counts"**: The paper explicitly reports and discusses the ~100× difference in trainable parameters (Table 2) and frames it as a virtue. No claim of equivalent expressiveness is made. This is transparent methodology, not a structural flaw. **REMOVED** as a strawman that misreads the paper's framing.

**Harsh Critic — "ZOQO/concurrent ZO+quantization not empirically compared"**: The paper acknowledges these methods qualitatively in Related Work, noting QZO is "inherently more efficient and flexible" and "does not require quantization of perturbation noises or re-quantization." A full empirical comparison would be valuable, but per the rules, this falls under the soft "missing baselines" category rather than a fatal flaw, especially since the paper provides a clear rationale. **DOWNGRADED** to nice-to-have.

**Harsh Critic — Missing hyperparameter sweeps, undisclosed sensitivity to ε**: Hyperparameters for QZO are clearly stated (lr=1e−7, ϵ=1e−3, C=100, batch=16, 20k steps). This is not an undisclosed detail. **REMOVED** as a reproducibility nitpick.

**Harsh Critic — "Per-layer vs per-channel vs per-group scales not specified"**: The paper states "quantization group in GPTQ is set to 128" (Section 4.1), which specifies the parameterization. **REMOVED** as factually incorrect criticism.

**Human Finder — Weaknesses from other papers not applicable to QZO** (e.g., gradient accumulation memory analysis, activation outlier similarity, sparse mask transferability): These concern SensZOQ's design and do not apply to QZO's scale-perturbation approach. **REMOVED**.

---

## Novel Insights

The most genuinely novel observation surfaced across all reviews is the asymmetry between *what QZO actually optimizes* and *what typically gets optimized* in memory-efficient LLM fine-tuning. By targeting the quantization scale parameters exclusively, QZO discovers that the information needed to adapt a quantized model to a new task is partially encoded in the continuous scale factors — not in the discrete codebook indices or integer weights. This is a non-obvious claim: it implies that task-specific adaptation can be carried out in a parameter space that is naturally continuous, dramatically smaller than the weight space, and tightly integrated into the quantization arithmetic, enabling genuine gradient-free training. Whether this scale-space expressiveness is sufficient for harder tasks remains open and is arguably the most important empirical question the paper does not yet answer.

---

## Suggestions

1. **Add QLoRA at a matched memory budget as the primary comparison baseline** — this is the single highest-priority revision. Show the performance-vs-memory Pareto curve for both methods.
2. **Rigorously verify Theorem 1** in the main text, not just the appendix. If the proof relies on specific conditions (e.g., symmetric distribution of d, small-ϵ approximation), state them explicitly. If bias exists, frame DDC as a bias-variance tradeoff and adjust the variance inequality accordingly.
3. **Report multi-seed results** (mean ± std over ≥3 seeds) given the small dataset sizes.
4. **Add training loss curves and wall-clock time plots** for QZO vs MeZO to clarify convergence behavior.
5. **Include at least one challenging benchmark** (MMLU 5-shot, GSM8K, or a generation task with reference-based metrics) to assess whether scale-only adaptation generalizes to harder settings.
6. **For the 2-bit Table 3**, add either MeZO on the 2-bit model or QLoRA as a trained baseline to contextualize QZO's improvements over zero-shot.

---

## Score and Decision

**Calibration anchors used:**
- *SensZOQ* (ZO + quantization, sparse params, Accepted Poster): scores 6,6,6,6,5 — similar domain, similar missing baselines (no QLoRA), comparable evaluation scope. QZO has a more elegant core idea but a similarly weak theoretical contribution and missing QLoRA comparison.
- *SparseMeZO* (ZO sparsity for LLMs, Rejected): scores 5,6,6,5 — rejected in part due to soundness concerns and missing motivation. QZO's empirical results are cleaner and the idea is more defensible, but shares the issue of limited expressiveness analysis.
- *LOZO* (low-rank ZO, Accepted Poster): scores 8,6,8,6 — substantially stronger theory and convergence analysis; QZO does not reach this bar.
- *SubZero* (subspace ZO, Withdrawn/Reject): scores 3,3,5,6 — weaker in both theory and experiment; QZO is clearly above this level.

QZO sits closest to SensZOQ in quality but is pulled slightly below by (a) the questionable Theorem 1 unbiasedness claim and (b) the absent QLoRA comparison, which is the most pressing gap for a paper positioning itself as a memory-efficient fine-tuning method for quantized LLMs. The core idea is solid and the empirical results are real, but the paper is not yet ready for publication without these revisions.

**Originality**: Moderate-to-high — the scale-perturbation insight is genuinely novel.  
**Importance of research question**: High — memory-efficient fine-tuning of quantized LLMs is practically pressing.  
**Claim support**: Moderate — results are plausible but baseline comparisons are incomplete and the main theorem is questionable.  
**Soundness**: Moderate — DDC works empirically; the unbiasedness claim requires scrutiny.  
**Clarity**: Good — the paper is well-written and easy to follow.  
**Value to community**: Moderate — the method is useful if it stands up to comparison with QLoRA.

**Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>