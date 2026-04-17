The paper content is in the user message. Let me check key points against it directly.

The paper states on the matter of limitations: "We do not train CLIP or SigLIP models from scratch or use larger batch sizes due to the limitation in computing resources, so the conclusion on the effects of different factors is restricted."

On text-only controls: Let me check if there's any mention of text-only baselines. I don't see any explicit text-only baseline in the paper's methodology.

On the relaxed evaluation: The paper says in Section 5.2 that they calculate "the model perplexity of two possible options" and normalize them, then compare relative preferences. This requires both images simultaneously, which is different from the original evaluation.

Now I have enough information to write my review. Let me synthesize across all three reviewers' inputs.

---

## Summary

This paper challenges the claim that CLIP's "erroneous agreements" (high cosine similarity between visually distinct images) imply information loss or blindness in the encoder. The authors demonstrate that LLaVA-1.5-7B, which uses the same frozen CLIP encoder, achieves near-perfect accuracy on What'sUp spatial reasoning benchmarks despite embedding cosine similarities above 0.99, and outperforms CLIP-like models on multiple benchmarks. Through controlled ablations of evaluation method, training data, and text encoder, they argue the performance gap is primarily driven by the VLM paradigm rather than the encoder. They further show that alternative decoding (M3ID) and a relaxed pair-comparison evaluation can reveal additional preserved visual information, suggesting under-utilization rather than information loss.

## Strengths

- **Strong empirical counterpoint to the "encoder blindness" narrative.** The striking contrast between CLIP (~49% individual accuracy) and LLaVA-1.5 (~99-100%) on What'sUp subsets with embedding cosine similarity >0.99 (Table 1) directly and convincingly demonstrates that erroneous agreements do not inevitably cause VLM failure. This is an important empirical finding that the community needs.

- **Systematic and well-structured ablation design.** The paper methodically controls for evaluation method (Section 4.1), training data including hard negative captions (Section 4.2), and text encoder quality via llm2vec ablation (Section 4.3), isolating paradigm as a residual factor. The negative results from these ablations are themselves valuable data points.

- **Constructive reframing with actionable insights.** The M3ID decoding experiment yielding +6% pair accuracy on MMVP (Table 6) is a concrete, non-trivial finding—comparable to vision-modification methods like Libra and I-MoF—suggesting practical improvements are possible without changing the encoder.

- **The paper honestly acknowledges its limitations**, noting computational constraints that prevented training from scratch and explicitly stating that conclusions on factor effects are "restricted."

## Weaknesses

### Major:

- **Over-interpretation of correlational evidence for the "paradigm is key" claim.** The paper moves from "LLaVA can sometimes extract more information than CLIP given the same encoder" to the much stronger conclusion that "erroneous agreements do not contribute to VLM failures on their own" and "the key factor is the paradigm." This overreaches in two ways: (a) On MMVP, LLaVA-1.5 also performs near random chance (~25% pairs accuracy), consistent with genuine information loss or severe degradation in the encoder—the same encoder the paper claims is not the bottleneck. (b) The ablations do not cleanly isolate paradigm effects: all experiments use the same frozen CLIP encoder, and the llm2vec variant still uses cosine-similarity-based alignment, so it tests "stronger text encoder under CLIP objective" rather than a fair paradigm comparison. The conclusion that paradigm is the primary cause rests on a process of elimination that is incomplete and confounded by practical constraints.

- **The relaxed-constraints evaluation (Section 5.2) is methodologically problematic as evidence for "preserved visual nuances."** This evaluation gives the model both images simultaneously and tests relative perplexity ordering, which is a fundamentally different task from the original single-image evaluation. No controls are run to rule out text priors or systematic biases—e.g., a text-only baseline, random image permutation, or adversarial label flipping. Without such controls, the 73.3% "accuracy" cannot be confidently attributed to preserved visual nuances rather than subtle text-based ordering preferences. Moreover, this metric changes the task and scoring rule, so it is not an upper bound on the original task accuracy as implied.

- **Limited generalization across MLLM architectures and task types.** The core analysis focuses heavily on LLaVA-1.5-7B (a single model architecture). While Appendix B.5 extends to "some other MLLMs," the main conclusions about paradigm are drawn from one model. All primary benchmarks (What'sUp, COCO-spatial, GQA-spatial, MMVP) focus on spatial understanding, leaving unclear whether findings generalize to other visual reasoning dimensions (counting, attribute binding, object state).

### Minor:

- **The 3D toy example (Section 3.2) is rhetorically misleading.** Vectors [10,11,12]^T and [12,11,10]^T with Spearman's ρ=-1 have high cosine similarity. But in realistic 768+ dimensional CLIP embeddings, angular differences near zero leave vanishingly small room for rank-reversal patterns. The example gives the impression of large reserves of exploitable nonlinear information, but no evidence is provided that such structure exists in real embeddings at the high-dimensional scale.

- **Modest gains from M3ID (+6% pair accuracy, 25.3%→31.3%) on MMVP.** While meaningful, this is a single benchmark experiment with no variance estimates or multiple runs reported. The absolute performance remains low, making it difficult to assess whether the improvement indicates under-attention or merely marginal utilization gains.

- **The computational resource limitation acknowledged in the conclusion significantly weakens the paradigm claim.** The paper admits not training CLIP/SigLIP from scratch or with larger batch sizes. Given CLIP's well-known sensitivity to batch size, the data ablation (Table 4-5) may underestimate what properly trained contrastive models could achieve with comparable data.

### Trivial:
- No statistical significance tests or variance estimates are reported across any experiments.

## Nice-to-Haves

- A simple linear or MLP probe on frozen CLIP embeddings for the spatial tasks would directly test whether the relevant information is linearly recoverable, cleanly separating embedding-level information from extraction strategy. This would substantially strengthen or clarify the paper's core thesis.

- Testing on non-spatial benchmarks (e.g., Winoground for compositionality, POPE for hallucination) to assess generalizability of claims beyond spatial reasoning.

- Per-instance analysis correlating CLIP cosine similarity with LLaVA accuracy margins to reveal whether higher similarity systematically predicts difficulty for MLLMs, rather than relying on aggregate statistics.

- Controls for the relaxed evaluation (text-only baseline, random image permutation) to validate that the observed relative ordering is genuinely image-driven.

## Removed Points

- **"Limited generalization beyond LLaVA-1.5-7B" (from human finder)** — While this is a valid concern and is kept as a minor weakness, the harsh critic's framing of this as requiring experiments on Qwen-VL, InternVL, InstructBLIP goes too far. The paper does extend to other models in Appendix B.5, and this is a known limitation of any focused study. The concern is real but does not invalidate the findings.

- **"Relaxed evaluation is not practical for real VLM usage" (from neutral reviewer)** — This conflates diagnostic utility with deployment scenarios. The relaxed evaluation is explicitly presented as a diagnostic, and its impractical nature does not undermine the finding that visual information is preserved but underutilized.

- **"Modest improvements and lack of novelty in M3ID" (from human finder, citing similar papers)** — The M3ID result is presented as supporting evidence, not as a core contribution. The paper appropriately cites prior work (Favero et al., 2024). Criticizing it for not being novel misses the paper's framing.

- **"The paper does not propose concrete novel solutions"** — The paper explicitly frames itself as a diagnostic/analysis contribution, not a methods paper. Criticizing it for not proposing novel techniques is scope creep.

- **"Some findings are somewhat expected" (from human finder)** — While it may seem intuitive that a generative MLLM with billions of parameters can extract more from embeddings than a simple dot product, the systematic evidence and the specific findings (e.g., that erroneous agreements don't necessarily imply blindness) are non-trivial contributions.

## Novel Insights

The paper makes an important observation that the gap between CLIP-like models and MLLMs using the same encoder is primarily driven by information extraction and utilization strategy rather than necessarily by information loss in the encoder. The relaxed evaluation insight—that models may preserve visual nuances in their internal representations that fail to influence final outputs due to language prior dominance—is a genuinely useful diagnostic perspective. However, the conceptual leap from "sometimes extractable" to "paradigm is the key factor" overstates what the experiments establish; the MMVP results, where the same model still fails badly, are more consistent with genuine information loss (or at least severe degradation) in the hardest cases, suggesting both encoder limitations and extraction deficits play interconnected roles.

## Suggestions

1. **Add a linear/MLP probe** on frozen CLIP embeddings for What'sUp and MMVP pair discrimination—this is cheap to run and would directly quantify how much spatial information is linearly vs. nonlinearly recoverable.

2. **Add controls to the relaxed evaluation**: run text-only (no image) and random-image-pair baselines to verify the 73.3% is image-driven.

3. **Temper the main claims** to acknowledge that: (a) the paradigm is *a* factor, not necessarily *the* factor; (b) MMVP results show both encoder and paradigm limitations; (c) the ablations are constrained by compute and use frozen encoders. Phrasing like "our evidence suggests paradigm is an important factor" would be more appropriate than "the key factor in their performance discrepancy might lie in their paradigms."

4. **Report per-instance analyses** (e.g., correlation between embedding cosine similarity and LLaVA accuracy) to provide a more nuanced picture than aggregate statistics.

## Score and Decision

Calibration comparison:
- **"From CLIP to DINO" (scores 3,3,3,6)**: Similar topic (visual encoders in MLLMs), but significantly weaker methodology and novelty. This paper is clearly stronger.
- **"Intriguing Properties of LLVMs" (scores 6,6,5,5)**: Similar empirical study of VLM properties, limited to LLaVA family, with loosely connected findings. This paper has a clearer central thesis and stronger systematic ablation, though it also overclaims.
- **"Towards Interpreting Visual Info Processing in VLMs" (scores 6,8,6,8)**: Accepted poster. Similar empirical depth on LLaVA, well-structured analysis. Our paper has comparable empirical contribution but with more overclaim in its conclusions.
- **"Fading Focus" (scores 3,6,5,3,6)**: VLM decoding improvement with modest gains. Weaker empirical contribution than our paper.
- **"Open Eyes Then Reason" (scores 5,5,5,5)**: VLM visual reasoning improvement with modest results. Similar pattern of overclaiming from limited evidence.

This paper makes a real and important empirical observation (LLaVA outperforming CLIP on spatial benchmarks with the same encoder) but overinterprets it into a paradigm-level causal claim that the ablations cannot fully support. The relaxed evaluation is methodologically insufficient as evidence for "preserved visual nuances." However, the core empirical finding is genuine, the systematic ablation is useful, and the M3ID result is a real contribution. The paper is above average for the venue but needs significant tempering of its claims.

**Final assessment**: The paper has real empirical contributions but structurally overclaims. The core observation is important, but the conceptual conclusion that "erroneous agreements do not contribute to VLM failure on their own" and "paradigm is the key factor" goes beyond what the evidence supports, particularly given the MMVP results and the confounded ablation design. This is a borderline paper—worthwhile findings marred by overclaiming.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>