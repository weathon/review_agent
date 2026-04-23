Now I have enough information to synthesize the final review. Let me compose it carefully.

## Summary

VL-Cache proposes a KV cache compression method tailored for Vision-Language Models (VLMs), leveraging two VLM-specific observations: (1) a modality boundary in attention patterns where post-vision language tokens' attention closely approximates decoding attention, and (2) non-monotonic, layer-wise sparsity variation. The method combines sparsity-aware layer-adaptive cache budget allocation with a modality-aware token scoring policy (post-vision attention), achieving near full-cache accuracy at 10% KV cache budget on several benchmarks and up to 2.33× end-to-end speedup.

## Strengths

- **Novel and well-motivated identification of VLM-specific attention patterns.** The observation of a clear modality boundary between visual and language tokens in VLM attention (Figure 1) is a genuine and useful insight that distinguishes VLM KV cache compression from LLM KV cache compression, and directly motivates both algorithmic contributions.

- **Strong accuracy at aggressively low cache budgets.** Table 1 shows VL-Cache at 10% KV cache budget closely matches or exceeds full-cache accuracy across most settings—for instance, DocVQA with LLaVA-1.6-34B achieves 84 ANLS at 10% vs. 85 at 100%, while H2O drops to 75 and StreamingLLM collapses to 34 at the same budget. On Coco-Caption 34B, VL-Cache actually exceeds full-cache performance (137.35 vs. 135.07 CIDEr) at 10%.

- **CacheHitRate provides a clean, interpretable proxy for scoring policy evaluation.** Figure 3 demonstrates that accumulated post-vision attention achieves consistently higher CacheHitRate across all layers compared to alternatives, with especially large gaps in deeper layers (above 0.8 vs. below 0.4 for alternatives).

- **Minimal prefill overhead and practical deployment analysis.** Table 2 shows prefill speedup of 0.96–0.99 (only 1–4% overhead from sparsity computation). Figure 6 provides a throughput-latency Pareto curve showing VL-Cache achieves both higher peak throughput and lower latency at any desired throughput level—directly useful for practitioners.

- **Honest about limitations.** The paper transparently acknowledges that end-to-end speedup is bounded by prefill latency (e.g., 53% of e2e time at 128K), and that maximum batch size is limited by prefill memory rather than KV cache size in their implementation.

## Weaknesses

### Fatal
None.

### Major

- **No ablation isolating the two proposed components.** The paper proposes two distinct mechanisms—sparsity-aware budget allocation (Section 4.1) and modality-aware token scoring via post-vision attention (Section 4.2)—but both are always applied together in all accuracy evaluations (Table 1, Figure 5). No experiment isolates their individual contributions (e.g., post-vision scoring with uniform budget, or sparsity-aware allocation with accumulated attention scoring). This is especially important because PyramidKV (a budget-allocation-only method) already achieves competitive results on some settings (DocVQA 34B: 83 vs. VL-Cache's 84 at 10% budget), raising the question of whether the scoring policy contributes meaningfully beyond the budget allocation. Without this ablation, the reader cannot determine whether both components are necessary or whether one dominates.

- **The "comparable to full cache" and "98%" claims are overclaimed for some settings.** The abstract states "retaining only 10% of KV cache achieves accuracy comparable to that with full cache," and the introduction claims "retains 98% of the original task-level accuracy." However, on DocVQA with LLaVA-Mistral-7B (Table 1), 10% budget yields ANLS of 62 versus full cache 68—approximately 91% retention, not "comparable" or "98%." The "majority" qualifier in the introduction is present but easy to miss relative to the prominent headline claims. The claim should be more precisely scoped.

### Minor

- **CacheHitRate metric only validates against the first decoding token.** Definition 3.1 defines the ground-truth scoring function ψ* using A_{m+1}—attention scores from the first decoding token only. Since VLMs generate dozens to hundreds of tokens, importance patterns from later decoding steps may differ from the first step. While the end-to-end accuracy results partially compensate, the theoretical motivation for the scoring policy rests on a proxy that captures only one step of a multi-step generation process.

- **Speed benchmarks use synthetic prompts, not real workloads.** Table 2's headline speedup numbers (7.08× decoding, 2.33× end-to-end) are measured with synthetic prompts rather than actual benchmark data. The paper explicitly states this (Section 5.2), and it is standard practice for kernel-level benchmarking, but the eviction behavior may differ on real multi-modal inputs where attention sparsity patterns could vary.

- **The method assumes a specific prompt ordering (instruction → images → question).** Post-vision attention is defined as the language tokens that follow visual tokens in the prompt. While this matches the standard LLaVA format, the paper does not discuss generality for other prompt orderings or VLMs with interleaved image-text inputs. This limits the claimed scope of the contribution.

- **The 10× concurrency claim is theoretical, not empirically demonstrated.** The paper states "In inference scenarios where KV cache size is the limiting factor to higher concurrency, VL-Cache enables up to 10x higher concurrency" (line 69), but also acknowledges that in their implementation, "maximum batch size is limited by peak memory usage during prefill instead of KV cache size." The concurrency claim is a theoretical projection that is not validated in the experimental setup, which could mislead practitioners.

### Trivial
None.

## Nice-to-Haves

- Ablation separating the budget allocation from the scoring policy (e.g., VL-Cache scoring with uniform budget, and sparsity-aware budget with accumulated attention scoring). This would significantly strengthen the contribution claims.
- Evaluation on additional VLM architectures beyond LLaVA (e.g., Qwen-VL, InternVL) to test whether the modality boundary insight generalizes.
- Sensitivity analysis of the sparsity threshold p=1%, which controls budget allocation.
- Quantification of the correlation between prefill and decoding sparsity, which underpins the budget allocation mechanism.
- Qualitative case studies showing which visual tokens are retained/evicted to assess semantic sensibility.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **No variance bars on sparsity measurements (Figure 2):** Trivial presentation nitpick; the key insight (non-monotonic sparsity variation) is clearly visible.
- **Algorithm 1 clip operation could cause total budget to deviate from target:** This is a minor implementation detail that is unlikely to cause significant deviation in practice; the clip bounds of [0.01, 1] are safety margins.
- **Recent token window size fixed at 10% disadvantages baselines like StreamingLLM:** The paper states this is for "fair comparison" and applies it uniformly to all methods. If anything, this standardizes a design choice that baselines might optimize differently, but it does not clearly advantage VL-Cache over them.
- **Missing related works:** Per hard rules, cannot verify existence of uncited works.
- **Formatting/typos:** Per hard rules, these are parser artifacts.
- **Missing appendix/reproducibility details:** Per hard rules, the parser strips appendices.
- **MathVista being an easy discriminator:** This is an observation about the dataset, not a weakness of the paper. The paper itself notes this is due to 54% multiple-choice questions.

## Novel Insights

The paper's most valuable insight is the modality boundary in VLM attention—language tokens after visual tokens attend selectively to a few critical visual tokens, while visual tokens attend uniformly to each other. This creates a natural "post-vision attention" window that is both dynamically sized per prompt and far more informative for token importance scoring than full-prompt attention. However, the two-component structure (budget allocation + scoring policy) without ablation leaves open the possibility that the simpler budget allocation alone drives most of the gains, particularly on tasks like DocVQA where PyramidKV is already competitive.

## Suggestions

- Run a 2×2 ablation (sparsity-aware vs. uniform budget × post-vision vs. accumulated attention scoring) to clarify individual component contributions—this is the single most impactful addition for strengthening the paper.
- Qualify the "98%" and "comparable" claims more prominently, e.g., "retains ≥95% of full-cache accuracy on 5 of 6 evaluated settings."
- Add a correlation analysis (scatter plot or coefficient) between prefill and decoding sparsity to quantitatively support the budget allocation mechanism.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| FastGen (Adaptive KV Cache) | /home/wg25r/review_agent/human_reviews/uNrFpDPMyo.md | 8.0 | Stronger ablation and broader experiments, but less VLM-specific; VL-Cache has a clearer VLM-specific motivation but weaker evaluation rigor |
| SqueezeAttention | /home/wg25r/review_agent/human_reviews/9HK2rHNAhd.md | 5.5 | Similar layer-wise budget allocation concept but for LLMs; VL-Cache has stronger VLM-specific insight and better low-budget accuracy |
| DynamicKV | /home/wg25r/review_agent/human_reviews/uHkfU4TaPh.md | 4.4 | Task-aware adaptive KV cache; rejected for lack of practical deployment analysis; VL-Cache has better deployment analysis but similar evaluation gaps |
| MixAttention | /home/wg25r/review_agent/human_reviews/2DD4AXOAZ8.md | 2.0 | Zero novelty, just reproduces a blog; VL-Cache is clearly far above this |
| NAMMs | /home/wg25r/review_agent/human_reviews/s1kyHkdTmi.md | 7.0 | Learned memory management for KV cache including VLMs; stronger methodology but different approach |
| HeadKV | /home/wg25r/review_agent/human_reviews/FJFVmeXusW.md | 6.5 | Head-level KV cache compression; comparable scope and evaluation quality |

VL-Cache sits above SqueezeAttention (5.5) and DynamicKV (4.4) due to its genuine VLM-specific insight and strong empirical results, but below FastGen (8.0) and NAMMs (7.0) due to the missing ablation and evaluation gaps. The closest anchor is HeadKV (6.5), which has a similarly focused contribution with solid but not overwhelming evaluation.

## Score and Decision

**Originality:** The modality boundary observation is novel and well-motivated. The two-component method builds on this insight reasonably, though sparsity-aware budget allocation has been explored in LLM contexts (PyramidKV, SqueezeAttention). The post-vision attention scoring is the more distinctive contribution.

**Importance of research question:** KV cache compression for VLMs is practically important given the growing deployment of VLMs and the disproportionate memory cost of visual tokens.

**Claim support:** The main accuracy claims are largely supported except for DocVQA 7B where they are overclaimed. The speed claims are well-characterized with honest limitations. The missing ablation is the most significant gap in claim support.

**Soundness of experiments:** Accuracy evaluation is thorough across budgets and models. Speed evaluation is adequate but uses synthetic prompts. The CacheHitRate proxy is informative but limited to first-token validation.

**Clarity:** The paper is well-organized with clear algorithmic specification (Algorithm 1) and good figure quality.

**Value to community:** Practical for VLM deployment; the modality boundary insight will likely inform future VLM-specific optimization work.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>