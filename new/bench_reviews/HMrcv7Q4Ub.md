Now let me look at the human reviews for calibration.Now I have all the information I need to produce a final consolidated review.## Summary
VL-Cache proposes a KV cache compression recipe for Vision-Language Model (VLM) inference. The paper identifies two characteristics of VLM attention that differentiate it from LLM attention: (1) non-monotonic layer-wise sparsity variation that makes static budget schedules suboptimal, and (2) a clear modality boundary where "post-vision" language attention better predicts decoding-phase token importance than accumulated full-prompt attention. Based on these observations, the method introduces a sparsity-aware layer-adaptive budget allocation and a post-vision attention scoring policy, demonstrating that 10% KV cache largely preserves accuracy while achieving decoding speedups up to 7.08×.

---

## Strengths

- **Well-motivated, VLM-specific insight.** The observation that VLM attention has a clear modality boundary (Figure 1b) and that post-vision language attention is a stronger predictor of decoding-phase importance than full accumulated attention is a concrete, original insight with direct algorithmic implications. The CacheHitRate metric (Definition 3.1) quantifies this cleanly.

- **Strong low-budget accuracy results.** Table 1 shows consistent, often large improvements over baselines at 5–10% budget. For example, at 10% budget on Coco-Caption 34B, VL-Cache scores 137.35 CIDEr vs. 58.14 (H2O) and 116.91 (PyramidKV), while full cache is 135.07. These are practically meaningful margins.

- **Dynamic, per-prompt budget allocation.** Unlike PyramidKV's static monotone schedule, Algorithm 1 customizes layer-wise budgets to each prompt's measured sparsity, providing greater flexibility and explaining empirical gains at low budgets.

- **Comprehensive baseline comparison** across two model scales (7B and 34B), three diverse tasks (OCR, reasoning, captioning), and four contemporary KV compression methods.

- **Clearly written.** The problem setup, the analytical observations in Section 3, and the algorithmic design in Section 4 flow logically and are easy to follow.

---

## Weaknesses

### Fatal
*None that invalidate the core accuracy contribution.*

### Major

- **Speed evaluation is confounded by a custom kernel implementation.** Section 5.2 states: *"For both prefill and decoding in the baseline, we used default settings from the HuggingFace implementation, including CUDA-based FlashAttention-v2. To optimize performance in our VL-Cache, we applied our Triton-based solution for self-attention forward pass, layer-wise sparsity evaluation, and modality-aware token scoring."* The reported speedups (up to 7.08× decoding, 2.33× end-to-end) thus confound two distinct effects: (a) the algorithmic benefit of KV compression and (b) the systems benefit of replacing HuggingFace+FlashAttention-v2 with a custom Triton kernel. Without an implementation-controlled ablation—e.g., measuring VL-Cache's compression benefit atop the same Triton kernel—the paper cannot attribute the reported speedups to the compression algorithm per se. Since efficiency is a headline contribution, this is a significant methodological gap.

- **Missing ablation study for the two proposed components.** The paper presents two innovations—sparsity-aware layer budget allocation and post-vision attention scoring—but never evaluates them separately. A 2×2 ablation (equal/sparsity-aware budget × accumulated/post-vision scoring) is the minimum needed to show that both contributions matter. Without it, the observed gains could be entirely attributable to one component.

- **Evidence base too narrow for the "VLMs broadly" framing.** The core analysis (Section 3) is conducted on a single model, LLaVA-Mistral-7B. All main experiments use two LLaVA-family models sharing the same CLIP visual encoder. The paper frames contributions as "VLM attention sparsity patterns" and a "recipe tailored for VLMs," but the evidence supports conclusions about this LLaVA variant, not VLMs as a class. Given that the post-vision attention concept depends on a specific prompt structure assumption (contiguous vision block followed by language), it is genuinely unclear whether the insight generalizes to architectures with interleaved modalities, different visual encoders, or different template formats.

- **Concurrency claim is unsubstantiated and acknowledged as such.** The abstract advertises "up to 10x higher concurrency," but Section 5.2 explicitly states: *"In our implementation of both the baseline and VL-Cache, maximum batch size is limited by peak memory usage during prefill instead of KV cache size, so compression of KV cache does not lead to higher batch size."* The claim is purely theoretical and contingent on implementing continuous batching and chunked prefill. This should be removed from the abstract or clearly labeled as an upper-bound theoretical projection.

### Minor

- **"Consistently outperforms" overstates Table 1.** At moderate-to-high budgets (40–80%), ZipCache and PyramidKV match or exceed VL-Cache in several cells (e.g., Coco-Caption 7B at 40%: VL-Cache 99.93 vs. H2O 102.64; DocVQA 7B at 60%: VL-Cache 67 vs. ZipCache 68). The advantage is clear at low budgets (5–10%), and the text should be scoped accordingly rather than claiming universal superiority.

- **Speed benchmark uses a simplified post-vision window.** Section 5.2 uses *"the last 50 tokens of the prompt"* to determine eviction, while the method is defined in Section 4 as using the actual post-vision segment (dynamically determined by τ). This disconnect makes the latency measurements not directly representative of the algorithm as evaluated for accuracy.

- **"Regularization effect" claim is speculative.** The Figure 5 caption attributes occasional improvements over full cache to *"the regularization effect of KV cache compression"* without any controlled experiment. This should be either investigated or qualified as a hypothesis.

- **No multi-image/video evaluation despite explicit motivational use case.** Section 1 motivates the problem with multi-image and video inputs, and the memory estimate in the introduction uses a five-image scenario. All experiments use single-image prompts. At minimum, the scope gap between the motivation and the evaluation should be acknowledged.

### Trivial

- **Algorithm 1 budget clipping may cause total budget mismatch.** Line 12 of Algorithm 1 clips layer budgets to [0.01, 1], but the global budget α is not renormalized after clipping. At extreme compression ratios where many layers hit their floor, the realized total budget could diverge from α. The paper does not discuss this.

- **p=1% threshold is fixed without sensitivity analysis.** Since this threshold drives both sparsity measurement and budget allocation, a brief sensitivity check would increase confidence in the method's robustness.

---

## Nice-to-Haves

- Evaluate on at least one additional VLM architecture (e.g., Qwen-VL, InternVL) to test generality of the modality-boundary observation beyond LLaVA-family models.
- Discuss or extend the "post-vision" boundary identification for prompts with multiple interleaved image-text blocks, since this is the use case the introduction motivates.
- Add overhead analysis of the dynamic budget allocation as a function of context length and batch size, beyond the single "1–4% of prefill" figure.
- Report absolute latency in Table 2 alongside speedup ratios; a 2× speedup on 10ms is very different from 2× on 500ms.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Reviewer claim: "First work to investigate VLM attention sparsity" novelty statements should be questioned.** The harsh reviewer raises this but explicitly notes they cannot verify priority claims from the manuscript alone. Per hard rules, this is not a legitimate reviewable weakness.

- **Reviewer claim: confidence intervals / statistical testing should be required.** Single-run evaluation without confidence intervals is standard practice in VLM benchmarking communities (lmms-eval follows this norm). Moved to nice-to-have would even be generous; removed entirely.

- **Reviewer claim: abstract's "10% budget achieves comparable accuracy" is too broad.** Upon inspection of Table 1, the claim is substantially supported: on most task×model combinations, the 10% numbers are within ~5% relative of full cache. DocVQA 7B (62 vs. 68) is the most prominent exception, but the abstract says "majority of vision-language tasks," which is defensible. This is a minor precision issue, not a structural problem.

- **Reviewer claim: missing long-context benchmarks like RULER/InfiniteBench.** VL-Cache targets VLM tasks (image understanding) rather than long-form LLM tasks (document QA over thousands of words). The benchmarks selected (DocVQA, Coco-Caption, MathVista) are the standard lmms-eval suite for this setting. Demanding RULER or InfiniteBench is scope creep.

- **Reviewer claim: missing comparisons with SnapKV.** Cannot verify SnapKV is within scope or a fair comparison target from manuscript alone; per hard rules, missing related works claims are removed.

---

## Novel Insights

The most genuinely insightful contribution is the formalization of the *modality boundary* as an algorithmic lever: by restricting the scoring window to the post-vision language segment rather than the full prompt, the method implements a dynamic, semantically-grounded sliding window whose size automatically tracks the question length. The CacheHitRate metric (Definition 3.1) is an elegant compression-agnostic proxy that enables policy comparison before committing to full task evaluation. These observations together suggest that the standard LLM practice of treating all prefill tokens uniformly for eviction scoring may be suboptimal for any architecture where a structured non-textual prefix precedes the actual instruction—a broadly applicable principle beyond just LLaVA models.

---

## Suggestions

1. **Run an implementation-controlled speed experiment.** Use the same custom Triton attention kernel for both VL-Cache and a "full-cache" baseline; this isolates the speedup attributable to KV compression from the implementation change.
2. **Add a 2×2 ablation** (equal vs. sparsity-aware budget) × (accumulated vs. post-vision scoring) to show each component's independent contribution.
3. **Remove or clearly qualify the 10× concurrency claim** in the abstract; replace with "could theoretically enable..." or defer to future work given the prefill bottleneck.
4. **Test on at least one non-LLaVA VLM** (e.g., Qwen-VL or InternVL) to validate the modality-boundary observation's generality.
5. **Clarify the speed benchmark** by either aligning it to the dynamic τ procedure used for accuracy or explaining why the fixed 50-token approximation is representative.

---

## Score and Decision

**Calibration:**

| Anchor paper | Score | Comparison |
|---|---|---|
| Dynamic-LLaVA (hzVpZDrW73) | 6,6,6,6 (Accept) | Similar VLM efficiency scope; has ablations, comprehensive benchmarks, stronger evaluation rigor |
| D2O (HzBfoUdjHt) | 6,6,6,5,6 (Accept) | Layer+token adaptive KV for LLMs; has ablations and broader model testing |
| ZipVL (bx0IbCcBvO) | 5,3,3,5 (Reject) | VLM adaptive token sparsity; rejected for limited evaluation and overhead analysis |
| DynamicKV (uHkfU4TaPh) | 5,3,5,6,3 (Reject) | Dynamic layer-wise KV for LLMs; rejected for FlashAttention compatibility and missing efficiency evaluation |
| SparseVLM (1xG3MN1RRW) | 3,5,6,6,6 (Reject) | VLM visual token sparsification; mixed scores due to accuracy drops |

VL-Cache lands above ZipVL and SparseVLM in terms of empirical strength and clarity of motivation. However, it shares ZipVL's problem of not isolating the efficiency gain (implementation change vs. algorithm) and DynamicKV's missing FlashAttention/efficiency rigor, and lacks the ablation studies that differentiate D2O and Dynamic-LLaVA from the rejected papers. The accuracy results at 5–10% budget are the paper's strongest asset. The missing ablation study and confounded speed evaluation prevent acceptance at this stage.

**Axes:**
- *Originality*: Moderate-good. Post-vision attention as a prompt-structure-aware scoring policy is genuinely novel; budget allocation is an incremental extension of layer-wise ideas.
- *Importance of research question*: High. KV cache pressure in VLMs is a real deployment bottleneck.
- *Claim support*: Accuracy claims are well-supported. Efficiency/concurrency claims are inadequately supported.
- *Soundness of experiments*: Adequate for accuracy; flawed for efficiency.
- *Clarity of writing*: Good.
- *Value to community*: Meaningful if efficiency claims are properly established.

**Final Score: 5.0 — Weak Reject**

The paper has a valuable core idea and strong low-budget accuracy results, but the key efficiency claim is not cleanly established (confounded by kernel implementation), the missing ablation prevents attribution of gains to individual components, and the broad "for VLMs" framing is not supported by the single-family evidence base.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>