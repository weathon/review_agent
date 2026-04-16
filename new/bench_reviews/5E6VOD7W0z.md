## Summary

This paper challenges the claim from prior work (Tong et al., 2024c) that "erroneous agreements" (high cosine similarity between visually distinct CLIP image embeddings) indicate information loss causing VLM failures. The authors demonstrate that LLaVA-1.5-7B, using the same frozen CLIP image encoder, achieves near-perfect accuracy on What'sUp subsets where CLIP embeddings have cosine similarity >0.99 and CLIP itself performs at random chance. Through ablation studies varying evaluation method, training data, and text encoder, they attribute the performance gap to VLM paradigm differences. They further show that M3ID decoding and a novel relaxed-constraint evaluation suggest that LLaVA-1.5 retains more visual information than its original evaluation indicates.

## Strengths

- **Striking and important empirical finding:** The core observation that LLaVA-1.5-7B achieves near-100% accuracy on What'sUp Subset A (Left/Right) where CLIP image embeddings have cosine similarity >0.995 and CLIP itself is at random chance (Table 1) is a genuinely surprising result that meaningfully nuances the "erroneous agreements = blindness" narrative.

- **Well-structured ablation design:** The paper systematically controls for evaluation method (Section 4.1), training data (Section 4.2), and text encoder quality (Section 4.3) to narrow down the cause of the performance gap. Replacing CLIP's text encoder with an LLM-derived encoder (llm2vec) while keeping the image encoder the same (Table 5) and still failing to close the gap is a telling negative result.

- **Constructive analysis beyond diagnosis:** The M3ID decoding experiment (Table 6) and the relaxed-constraint evaluation (Table 7) go beyond identifying the problem to suggest that visual information is present but underutilized, offering actionable directions for future model improvement.

- **Unified evaluation methodology:** Section 4.1's unified multiple-choice evaluation for both CLIP and LLaVA on MMVP/MMVP-VLM (Table 3) eliminates an obvious evaluation confound, strengthening the validity of performance comparisons.

## Weaknesses

### Major:

- **Overclaiming relative to evidence on the "erroneous agreements" thesis.** The paper's conclusion states "Our study questions the use of erroneous agreements to reflect CLIP image encoders' information loss or blindness." What the evidence actually shows is more limited: a *different* model paradigm (7B-parameter autoregressive LLM with MLP adaptor, instruction-tuned) can extract some information from embeddings where CLIP's *linear* dot-product alignment fails. This challenges only the strongest version of the "blindness" claim—that *no* useful information is preserved—and does not demonstrate that erroneous agreements are an unreliable indicator of information loss *for the CLIP paradigm itself*. The paper's own analysis in Section 4.3 acknowledges that CLIP's dot-product alignment "might not effectively capture all correspondences," which is essentially consistent with the original framing. The claim should be reframed to accurately reflect what is shown: erroneous agreements indicate failure under linear extraction, not that the representations are completely uninformative.

- **LLaVA vs. CLIP comparison is heavily confounded, and the ablation is underpowered to isolate "paradigm" as the cause.** LLaVA-1.5-7B differs from CLIP in model capacity (7B vs. much smaller), architecture (autoregressive LLM + MLP adaptor vs. dual encoder), training objective (generative instruction-following vs. contrastive), and training data (curated multimodal instructions vs. web-crawled captions). The ablation studies in Sections 4.2–4.3 only show that small-scale finetuning of CLIP on LLaVA's data and replacing its text encoder with llm2vec do not close the gap. But the paper acknowledges (Section 6) that they "do not train CLIP or SigLIP models from scratch or use larger batch sizes due to the limitation in computing resources." Null results under these constrained conditions do not robustly support the strong attribution to "paradigm"—they only show that these particular interventions with these particular resources were insufficient. The conclusion that "differences in VLM paradigms may largely explain the performance gap" (Section 4.3) is conjecture rather than established fact.

- **The relaxed-constraint evaluation (Section 5.2) redefines the task and overstates the amount of usable information.** The proposed metric evaluates whether perplexity preferences are *relatively* ordered across both images in a pair, rather than requiring each image to be correctly classified independently. This necessarily inflates measured accuracy by allowing weak, inconsistent directional signals to count as successes, and doubles the random baseline from 25% to 50%. The paper interprets the 73.3% accuracy as evidence that "visual nuances are often extracted and aligned with the correct semantics," but this conflates the existence of residual relative signal with the ability to form correct per-image decisions. The finding is informative as a diagnostic (some signal exists), but the framing as "accuracy significantly increases" overstates what the metric measures.

### Minor:

- **The 3-dimensional toy example (Section 3.2) is an oversimplification.** The example showing two vectors with cosine >0.989 but opposing rank order is mathematically correct for 3D vectors but does not directly apply to 768/1024-dimensional learned CLIP embeddings. Whether such opposing-rank structure actually exists in real erroneous-agreement pairs is not empirically verified, leaving the intuition unvalidated at scale.

- **M3ID improvement is modest and lacks analysis of what it fixes.** The +6% gain on MMVP pair accuracy (25.3% → 31.3%) is on a small benchmark (~150 pairs) and could partly reflect generic hallucination reduction rather than specifically improved visual grounding for erroneous-agreement pairs. No per-category analysis is provided to distinguish these explanations.

- **Narrow model and benchmark scope.** The main experimental narrative involves one MLLM (LLaVA-1.5-7B) and primarily spatial reasoning benchmarks. While Appendix B.5 mentions extending to other models, the core claims about "paradigm" differences would be substantially strengthened by testing additional MLLM architectures (e.g., InternVL, Qwen-VL) under the same protocols.

- **The paper does not adequately explain why LLaVA-1.5 succeeds on What'sUp (98%) but fails on MMVP (25%).** Both involve spatial reasoning with high-cosine-similarity pairs. The "mystery" is acknowledged in Section 1 ("its poor performance on the MMVP benchmark remains a mystery") and Section 5, but no analysis is provided into what differs between these tasks (visual complexity, question format, type of spatial relation), which would clarify the boundary conditions of the paper's claims.

## Nice-to-Haves

- A linear probing experiment on CLIP image embeddings for spatial labels to directly quantify how much spatial information is linearly decodable, which would ground the "extractable information" claim more rigorously than indirect arguments.

- Testing M3ID on What'sUp to assess whether the "language prior forgetting" issue is consistent across spatial reasoning benchmarks or specific to MMVP's harder visual patterns.

- Per-category breakdown on MMVP to identify which visual patterns benefit from M3ID vs. relaxed evaluation, clarifying whether the extraction deficit is uniform or pattern-specific.

## Removed Points

- **Formatting and presentation nitpicks** (removed per instructions on style nitpicks).

- **Reproducibility concerns about hyperparameters and training details** in ablations (removed per instructions—these are standard implementation details).

- **Demands for training CLIP from scratch or with large batches** — the paper already acknowledges this as a limitation. Demanding experiments the authors state they cannot run is not a substantive criticism of what is presented.

- **Claims that referenced models or datasets don't exist or haven't been released** (removed per hard rules).

- **Criticism that the paper is "not novel" because MLLMs outperforming CLIP on spatial tasks is "expected"** — while prior work showed MLLMs beat CLIP on spatial benchmarks, the specific finding about high-cosine-similarity pairs and the systematic ablation isolating paradigm factors adds genuine value beyond the obvious.

- **Demand for confidence intervals on small benchmarks** — single-run evaluation is standard for these benchmarks; per soft rules, this is not a community-standard requirement.

- **Strawman criticism that the paper claims erroneous agreements are "not real" or "don't matter"** — the paper's actual claim is more nuanced ("not the sole issue"), and the evidence does support that erroneous agreements alone don't fully explain VLM failures.

## Novel Insights

The paper's most distinctive contribution is the empirical demonstration that the same frozen CLIP image encoder can support near-perfect spatial reasoning in one paradigm (LLaVA-1.5) while being at random chance in another (CLIP), even on image pairs with cosine similarity >0.99. This creates a productive reframing: rather than asking "is the encoder broken?", the community should ask "what extraction strategies can recover information that linear alignment cannot?" The M3ID and relaxed-evaluation results, while individually modest, together suggest that the gap between "information present in embeddings" and "information usable by current generation" is wider than previously assumed.

## Suggestions

- Reframe the conclusion to accurately state what is supported: erroneous agreements indicate failure under CLIP's *linear* alignment paradigm, but some discriminative information may still be recoverable by non-linear architectures. Remove claims that erroneous agreements don't reflect "information loss" or "blindness" without this qualification.

- Add a brief analysis of why What'sUp (success) and MMVP (failure) differ for LLaVA-1.5—this is a natural next diagnostic given the paper's framing and would clarify the scope of the findings.

- Report per-pattern or per-category results on MMVP to reveal whether M3ID and relaxed evaluation help uniformly or selectively, adding mechanistic insight.

- When presenting the relaxed-constraint evaluation, explicitly acknowledge that it measures relative rank preservation rather than per-image decision accuracy, and discuss its limitations as a practical evaluation protocol (it requires access to both images and captions simultaneously).

## Score and Decision

I calibrated against several related papers reviewing VLM analysis and CLIP spatial reasoning:

- Papers examining VLM limitations with empirical ablations but overclaimed conclusions (similar to `Pj4Aid3XqL.md` — "Should VLMs be Pre-trained with Image Data?") received scores of 5–6.
- Papers with novel framing but limited experimental scope confirming somewhat intuitive findings (similar to `skHPtDnYGa.md` — "Understanding Role of LLMs in Evaluation") received scores of 3–5.
- Papers with strong empirical findings on CLIP/VLM analysis that were well-executed (similar to `tLFWU6izoA.md` — DIVA/CLIP visual shortcomings analysis) received scores of 6–8.
- A well-executed analysis of two factors affecting CLIP performance (`uAFHCZRmXk.md`) received scores around 8.

This paper sits in between: it has a genuinely important and surprising empirical observation (LLaVA-1.5 solving What'sUp Subset A at near-perfect accuracy despite >0.99 cosine similarity), but the interpretive claims overreach what the evidence supports, and key ablations are underpowered. The relaxed-constraint evaluation is innovative but methodologically questionable as evidence for the stated claims. The paper makes a meaningful contribution to the VLM literature by challenging a prevailing narrative, but the gap between what is demonstrated and what is claimed is too wide for a top-venue acceptance as-is.

MY FINAL SCORE: 5.5
MY FINAL DECISION: Reject