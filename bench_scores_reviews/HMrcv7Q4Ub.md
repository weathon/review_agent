## Summary
VL-Cache is a KV cache compression method tailored for Vision-Language Models (VLMs). It makes two core contributions: (1) a sparsity-aware, prompt-adaptive layer-wise cache budget allocation that departs from fixed or monotonically-decaying schedules, and (2) a modality-aware token scoring policy based on "post-vision attention" — attention from the language tokens following the visual tokens in the prompt — rather than the full-prompt attention used by LLM-derived baselines. Experiments on two LLaVA variants across three benchmarks show that 10% KV cache retention mostly preserves accuracy while delivering up to 7.08× decoding speedup.

---

## Strengths

- **Novel and well-grounded modality-boundary observation.** The paper's central empirical finding — that decoded language token attention aligns more closely with the post-vision language prompt segment than with the visual tokens themselves — is specific, non-obvious, and clearly demonstrated via Figure 1. This distinguishes VL-Cache from a straightforward migration of LLM compression techniques. The CacheHitRate metric is a principled formalization of this insight.

- **Non-monotonic, prompt-adaptive layer budget allocation.** Unlike PyramidKV's fixed monotonically-decaying schedule, VL-Cache computes sparsity per-prompt and allocates budgets accordingly. Figure 2 shows that layers do not exhibit a simple monotonic sparsity pattern, making this adaptive allocation genuinely useful. The computational complexity advantage O(τm) vs O(m²) for sparsity computation (when τ ≪ m) is an additional practical benefit.

- **Post-vision scoring as a dynamic sliding window.** Framing post-vision attention as a "prompt-specific dynamic sliding window" (Section 4.2) whose length equals τ (the post-vision language token count) rather than a fixed w is an elegant and concrete design improvement over the sliding window baseline, with Figure 3 quantitatively validating higher CacheHitRate across all layers.

- **Strong low-budget accuracy results, especially on the 34B model.** On the more demanding LLaVA-1.6-34B, VL-Cache at 5–10% budget substantially outperforms all baselines on Coco-Caption (120.11 / 137.35 CIDEr vs. H2O's 20.87 / 58.14 at 5%) and DocVQA, making the advantage convincing in the practically motivated low-budget regime.

- **Detailed efficiency benchmarking.** The paper reports kernel-level prefill and decoding latencies under multiple (batch size, prompt length) combinations, honestly showing that end-to-end speedup is bounded by prefill and that the method's advantage grows with output length. The throughput–latency curve (Figure 6) is a practically useful artifact.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No ablation isolating the two contributions.** The paper bundles sparsity-aware budget allocation and post-vision modality scoring but never evaluates either component independently (e.g., "uniform budget + post-vision scoring" vs. "sparsity-aware budget + standard accumulated attention"). Without this, it is impossible to determine how much of the observed accuracy gain at low budgets comes from each component. This is a core scientific gap: a reader cannot tell whether the budget allocation or the scoring policy is carrying the result.

- **Critical model diversity gap.** All accuracy experiments use only two models from the LLaVA family (Mistral-7B and Yi-34B), both sharing the same visual encoder (openai/clip-vit-large-patch14-336). The modality boundary and attention sparsity patterns are discovered on and validated only against CLIP-ViT-based VLMs. Whether these patterns hold for models using different visual encoders (e.g., InternVL, Qwen-VL, models using Q-Former or cross-attention) is entirely untested. The paper's claim that it provides a general VLM-specific solution is therefore unsubstantiated — the observations could be partially CLIP-ViT-L artifacts.

- **Budget constraint potentially violated after per-layer clipping.** Algorithm 1 Line 12 clips each per-layer budget β^(l) to [0.01, 1]. After clipping, the sum ∑_l β^(l) no longer necessarily equals αL (the global budget target). For example, if many layers are dense and their unclipped budgets exceed 1.0 and are clamped, total allocated cache overshoots α. Conversely, if many layers are very sparse and are clamped at 0.01, total cache undershoots α. There is no re-normalization step in the algorithm, and the paper does not report empirical measurements of realized vs. target budget, nor does it discuss this boundary behavior. For a contribution centered on principled budget allocation, this is a correctness concern that needs to be addressed.

### Minor

- **"Consistently outperforms" is overstated at higher budgets.** The abstract and Section 5.1 claim VL-Cache "consistently outperforms" baselines. However, Table 1 shows H2O beats VL-Cache on Coco-Caption (Mistral-7B) at the 20%, 40%, 60%, and 80% budget points (e.g., 104.04 vs. 102.06 at 20%). The genuine advantage is in low-budget regimes (≤10%), where the claims are well supported and the advantage is sizable. The paper should scope its claims more precisely.

- **Speed benchmark uses a simplified proxy for τ.** The latency benchmarks use synthetic prompts and fix "the last 50 tokens" to stand in for the actual post-vision language tokens τ (Section 5.2). This is an acknowledged approximation, but it means the reported speedups do not correspond to exactly the same configuration as the accuracy experiments. For workloads where τ is much shorter or longer than 50, the overhead and speedup would differ.

- **Abstract/headline speedup is cherry-picked.** The "up to 2.33×" end-to-end speedup comes from batch size 64, prompt length 2K — a single favorable configuration. At batch=1 / 2K (the common single-user scenario), end-to-end speedup is only 1.16×. The abstract does not qualify this, which may mislead readers. (The paper does explain the batch-size and output-length dependence transparently in Section 5.2, which is appreciated.)

- **Post-vision window size τ sensitivity not analyzed.** The method's scoring quality depends on the length τ of the post-vision language tokens. Prompts with very short post-vision text (e.g., "Describe the image.") provide very few query vectors for computing attention statistics, potentially producing noisy token importance estimates. No analysis of method behavior across a range of τ values is presented.

### Tiny

- **"Post-vision attention" terminology is mildly confusing.** The term sounds like attention occurring *after* vision, but actually denotes attention *from* post-vision language tokens. A brief disambiguating note at first introduction would prevent a re-read.

- **MathVista anomaly (1% occasionally beats 5%) is noted but unexplained.** The paper attributes it to "regularization effect" (caption of Figure 5), but this is speculative. The instability at very low budgets (1%) is visible across several methods and deserves a short discussion.

---

## Nice-to-Haves

- **Video and multi-image accuracy benchmarks.** The introduction and conclusion both explicitly motivate video as a primary target (largest KV cache pressure), but no video QA benchmark (e.g., Video-MME, MVBench) is evaluated. Adding even one video benchmark would directly validate the most strongly motivated use case and would substantially support the paper's framing.

- **Sensitivity analysis for p = 1%.** The sparsity threshold p controls which attention entries count as non-sparse, and directly shapes the layer budget allocation. A brief ablation over p ∈ {0.1%, 1%, 5%} in terms of downstream accuracy and budget distribution would demonstrate robustness and guide practitioners.

- **Visualization of retained token distributions.** A heatmap or histogram showing which visual vs. text tokens are retained across layers at 10% budget, for a concrete example input, would provide intuitive validation of the modality-aware eviction claim and distinguish VL-Cache visually from token-uniform baselines.

- **Layer budget allocation profile plot.** Plotting actual per-layer allocated budgets β^(l) for a few representative inputs vs. PyramidKV's monotonic schedule would concretely illustrate the non-monotonic allocation that the paper claims is important. This visualization would strengthen Section 4.1's argument.

- **Clarify multi-turn / interleaved image-text handling.** Section 4 assumes a single-image VLM prompt structure (language → vision → language). The paper could briefly note whether and how the definition of "post-vision" tokens extends to multi-turn or interleaved contexts, even if handling these is left to future work.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"First work" claim is invalid (harsh critic).** The critic argues FastV and HiRED preempt the novelty claim. However, FastV prunes visual tokens from the attention forward pass (a different mechanism), and HiRED focuses on offline token selection. The paper's specific claim — "first work that specifically optimizes KV cache compression for VLMs" — is more precisely scoped and likely still holds. REMOVED.

- **Notation inconsistency between Section 3.2 sliding window formula and Algorithm 1 (harsh critic).** The critic claims Q_{m-w:m}K_{m-w:m} in the sliding window definition is inconsistent with Algorithm 1's full K. But the sliding window scoring is defined over the windowed K by design (it only scores tokens in that window), while post-vision scoring ranks all tokens using scores from the post-vision query. These are different policies with different intended computation. REMOVED as misreading.

- **Baseline configuration fairness (balanced reviewer).** The balanced reviewer questions whether LLM-originated baselines (H2O, PyramidKV, etc.) were optimally tuned for VLM inputs. The paper applies a consistent and reasonable adaptation (proportional budget scaling), and an asymmetric tuning advantage would favor the baselines, not VL-Cache — making VL-Cache's gains stronger evidence, not weaker. REMOVED.

- **Strength: "The paper is well-written" (generic).** Removed as it applies to any paper. REMOVED.

- **Strength: "The topic is important."** Removed as generic. REMOVED.

- **Requesting confidence intervals / multiple-run statistics (harsh critic, MathVista).** Single-run evaluation is standard practice for VLM benchmarks at this scale. REMOVED per policy.

---

## Novel Insights

The most substantive insight beyond the paper's own stated contributions (surfaced by cross-reading the three reviews) is the *asymmetric utility of the two components*: the post-vision scoring policy appears to provide the larger accuracy benefit (since it fundamentally changes which tokens are considered important, especially for visual content), while the sparsity-aware budget allocation provides an additional memory-efficiency advantage by preventing wasted slots in already-sparse layers. However, because no ablation exists, this decomposition is speculative — and precisely why the missing ablation is the most consequential gap in the paper. If the budget allocation is the weaker component, the paper could be simplified; if it is equally important, the paper would be stronger. Either way, knowing the answer would meaningfully advance the field's understanding of what makes VLM-specific KV compression work.

---

## Suggestions

1. **Add a 2×2 ablation table** (post-vision scoring × sparsity-aware budget, each toggled on/off) for at least one model and dataset. This is the single most important addition — without it, the contribution of each component cannot be assessed.

2. **Address the post-clipping budget constraint.** Either add a re-normalization step after clipping, or empirically measure and report the realized vs. target total budget deviation across the evaluation set to show it is negligible in practice.

3. **Evaluate on at least one architecturally distinct VLM** (e.g., InternVL or Qwen-VL, which use different visual encoders and projection mechanisms). Even a qualitative check of whether the modality boundary exists in those models' attention patterns would substantially strengthen the generalization claim.

4. **Scope the "consistently outperforms" language** to the low-budget regime (≤10–20%), and qualify the abstract's speedup claim with the batch-size and output-length context.

5. **Add a τ-sensitivity analysis** covering at least three prompt types with very short (< 10 tokens), medium (~50 tokens), and long (> 100 tokens) post-vision text to show robustness of the scoring policy across realistic prompt diversity.

---

**Axes evaluation:**
- **Novelty:** Moderate-to-high. The modality-boundary observation and its application to both scoring and budget allocation are genuinely novel within the KV cache compression literature, even if the building blocks (attention-based scoring, layer-wise allocation) are known.
- **Technical soundness:** Moderate. The post-clipping budget violation is an unaddressed algorithmic issue in the central contribution, and the absence of ablations prevents attribution of gains to specific components.
- **Empirical support:** Moderate. Results on two models within the same family, three benchmarks, no video, and no ablations represent a narrower experimental scope than the broad claims warrant.
- **Significance:** High. Efficient VLM inference is a pressing practical problem, and the core insight (use post-vision attention as the scoring proxy) is actionable and transferable.
- **Clarity:** Good. The paper is logically structured and the key ideas are presented accessibly, though the "post-vision attention" terminology benefits from careful reading.