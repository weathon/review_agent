## Summary
This paper proposes Quantized Zeroth-order Optimization (QZO), a method for fine-tuning post-training-quantized models by applying zeroth-order optimization to continuous quantization scales rather than discrete quantized weights. The key practical claim is that this removes gradients and optimizer states while also reducing weight memory through quantization, enabling very low-memory adaptation of large models, including 4-bit 7B LLMs and a 2-bit 13B LLM on a single 24GB GPU. The paper also introduces directional derivative clipping (DDC) as a stabilization mechanism for zeroth-order training.

## Strengths
- **The core technical idea is specific and nontrivial:** perturbing continuous quantization scales instead of discrete quantized weights is a clean way to bridge zeroth-order optimization with post-training quantization. This directly addresses the paper’s stated obstacle that discrete weights cannot be perturbed continuously and continuous ZO updates cannot be directly applied to quantized integers.
- **The method is demonstrated across two distinct quantization regimes:** the paper does not only test standard 4-bit scalar quantization (GPTQ), but also a 2-bit codebook-based setting (AQLM), which strengthens the claim that the approach is compatible with multiple PTQ styles rather than tied to a single quantizer.
- **The low-memory operating point is practically compelling:** the paper presents an end-to-end recipe that reportedly fits adaptation of large models into consumer hardware, including 4-bit 7B-class models and a 2-bit Llama-2-13B on a 24GB GPU. Even allowing for caveats in attribution, this is a concrete and useful systems outcome.
- **The stabilization ablation is materially informative:** Figure 2 and Figure 3 show that training without DDC is unstable and that clipping threshold meaningfully affects optimization behavior. This is more useful than a generic ablation because it exposes an actual failure mode and a corresponding control knob.
- **The appendix discussion with QLoRA is helpful and revealing:** Table 5 clarifies that QZO is not competitive with strong first-order PEFT in raw quality, but can be combined with PEFT while retaining a very small memory footprint. This makes the paper’s practical niche clearer.

## Weaknesses

###: Fatal
- **The theoretical justification for DDC is not sound as written.** Theorem 1 claims that the clipped estimator remains unbiased, but the paper does not establish conditions under which this would hold, and the Appendix proof is not convincing. Since clipping is a nonlinear operation applied to the directional derivative, a general unbiasedness claim is highly suspect without much more careful assumptions and derivation. The subsequent variance argument in Eq. 8 then relies on this theorem. This does **not** invalidate the empirical usefulness of DDC, but it does invalidate the paper’s current theoretical claim that clipping is an unbiased variance-reduction device.

### Major:
- **The FLOPs claim in Table 2 appears seriously misleading or miscomputed.** The paper states that QZO uses “about 1% of the FLOPs of MeZO,” but both methods are zeroth-order and require two forward passes per step to estimate a directional derivative. While QZO updates many fewer trainable variables, that does not by itself reduce the dominant cost of running the full model forward twice. The paper’s own explanation—“This is because QZO only fine-tunes the continuous quantization scale while leaving most weights fixed”—does not justify a 100× FLOP reduction for a full-model ZO method. This weakens the paper’s computation-efficiency claim substantially.
- **The memory-efficiency presentation over-attributes gains to the optimizer rather than to the optimizer-plus-quantization combination.** The paper’s framing is partly fair—its stated goal is a unified framework that minimizes weights, gradients, and optimizer states jointly—but the comparison to MeZO can read as if QZO itself yields a 3× memory reduction over MeZO through optimization alone. In fact, relative to MeZO, the memory difference is primarily from using quantized weights rather than from a new zeroth-order memory trick. The contribution is still meaningful as a combined recipe, but the paper should isolate what comes from ZO and what comes from PTQ more carefully.
- **The evaluation regime is narrow for supporting broad fine-tuning claims.** The main LLM experiments use 1,000 training examples per dataset and only five relatively small NLP tasks. This setup is sufficient to show feasibility, but it is not enough to establish that QZO is broadly effective for realistic fine-tuning workloads. This matters especially because zeroth-order methods can behave differently in larger-data regimes.
- **The empirical positioning against stronger memory-efficient first-order baselines is incomplete in the main paper.** The appendix includes QLoRA comparisons, and those results are informative: first-order PEFT is clearly stronger in task quality, while QZO occupies a different memory/quality tradeoff. But because this comparison is relegated to the appendix and not integrated into the main experimental narrative, the paper does not fully establish where QZO sits relative to standard practical alternatives under matched resource budgets.
- **The 2-bit 13B evaluation is under-contextualized.** Table 3 compares against zero-shot baselines, which demonstrates improvement over no adaptation, but does not sufficiently benchmark the method against stronger alternatives in that extreme setting. As a result, the 2-bit result is impressive as a feasibility demonstration but weaker as evidence of competitive learning quality.

### Minor
- **QZO has a substantial quality gap to first-order tuning/PEFT, and the paper’s practical claim should be framed more explicitly as a memory-first tradeoff.** This gap is visible in Table 1 and Appendix Table 5. The paper acknowledges some of this, but the practical takeaway should be sharper: QZO is appealing when memory is the dominant constraint, not when best downstream accuracy is the goal.
- **DDC appears important but somewhat sensitive.** Figure 3 suggests nontrivial dependence on the clipping threshold: too small underfits, too large destabilizes. The paper does provide a useful ablation, so this is not unaddressed, but it does indicate that the method is not fully plug-and-play.
- **The diffusion-model extension is interesting but not yet fully validated.** The appendix presents only qualitative results and the paper itself acknowledges a noticeable gap to the target distribution. This is better viewed as a promising extension than a strong empirical pillar of the paper.

### Trivial
- None.

## Nice-to-Haves
- Report gradient-alignment diagnostics, e.g., cosine similarity between QZO’s scale-direction estimates and first-order gradients with respect to scales on smaller models, to clarify whether scale perturbations recover meaningful optimization directions.
- Recompute and clarify FLOPs accounting so that forward-pass cost dominates the analysis appropriately for ZO methods.
- Add memory profiling at actual training batch sizes in addition to batch size 1, since the current profiling is mainly a minimum-VRAM demonstration.
- Expand larger-scale or larger-data fine-tuning experiments to show whether the method remains useful beyond 1k-example settings.
- Promote the QLoRA comparison from the appendix into the main paper, ideally under matched memory/computation budgets.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Unfair comparison with baselines because settings differ from QLoRA.”** The paper’s appendix explicitly frames QZO and PEFT as orthogonal methods and provides a comparison plus a hybrid. This is not a straightforward unfair-comparison issue; the more valid concern is incomplete positioning against practical first-order alternatives, which is kept above.
- **“The paper lacks an ablation on clipping threshold C.”** This is factually incorrect. Figure 3 is precisely an ablation on the clipping threshold.
- **“Q-SPSA is theoretically sound.”** This should not be retained as a strength in that form, because while the Q-SPSA construction is intuitive and technically reasonable, the paper’s main theoretical claim around clipping is not convincingly established.
- **Pure reproducibility complaints about omitted details or release status.** The paper provides code, implementation details, and cites all tools/models used; no concern of this form should be treated as a substantive weakness.
- **Formatting/parser artifacts in tables/figures.** These are extraction issues and not paper issues.

## Novel Insights
The paper’s real contribution is strongest when interpreted as a **systems recipe for the most memory-constrained end of adaptation**, not as a new theoretically grounded optimizer. The experiments and appendix together suggest a clearer positioning than the paper itself emphasizes: QZO is valuable because it exploits the tiny trainable surface offered by quantization scales to make zeroth-order adaptation feasible under very low VRAM, but this same restriction also explains why it tends to trail first-order PEFT methods in quality. In other words, the paper is most convincing as an argument that **quantization scales are a surprisingly capable optimization interface for ultra-low-memory adaptation**, rather than as evidence that QZO is a generally competitive replacement for gradient-based fine-tuning.

## Suggestions
- Remove or substantially weaken Theorem 1 and the associated unbiasedness/variance claims unless a correct proof under explicit assumptions can be provided.
- Rework the efficiency section to decompose gains into:
  - savings from zeroth-order optimization (no gradients / optimizer states), and
  - savings from quantization (reduced weight storage).
- Recompute Table 2 FLOPs with a transparent accounting that reflects the cost of two full forward passes per ZO step.
- Move the QLoRA comparison into the main paper and explicitly position QZO as a memory-first alternative rather than a generally competitive fine-tuning method.
- Strengthen the 2-bit section with stronger baselines or, at minimum, frame it clearly as a feasibility result rather than a competitive benchmark.
- If theoretical analysis is retained, focus on a bias-variance tradeoff statement for clipping rather than claiming unbiasedness.