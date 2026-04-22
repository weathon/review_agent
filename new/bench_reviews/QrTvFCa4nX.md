Now I have enough information to synthesize my final review. Let me compile all verified observations.

## Summary

The paper proposes INTP, a training-free method to extend Video-LLMs (specifically Video-LLaVA) from processing 8 frames to 32 frames by combining two techniques: (1) an interleaved video token rearrangement strategy that splits frames into alternating subsequences to maintain temporal consistency when passing through a frozen encoder/projector designed for fewer frames, and (2) NTK-aware RoPE interpolation to extend the LLM's context window. It also analyzes inference costs and introduces basic KV-cache quantization to reduce memory overhead.

## Strengths

- **Training-free extension from 8→32 frames with consistent gains**: Tables 2 and 3 show INTP improves over the 8-frame Video-LLaVA baseline across all five benchmarks (e.g., +3.6 on ActivityNet-QA, +4.4 on NExT-QA temporal, +1.6 on EgoSchema) without any retraining — a practically useful result. No prior Video-LLM method achieves this in a training-free manner as noted in Sec. 2.1.

- **Creative and well-motivated interleaved subsequence idea**: The token rearrangement technique (Fig. 2, Sec. 3.2) — pairing Frame #1 with #3, Frame #2 with #4, etc. — is a simple but thoughtful solution to the temporal inconsistency that would arise from naive concatenation of separately-encoded frame groups. The two-challenge decomposition (❶ fixed encoder/projector, ❷ limited LLM context) in Sec. 3.1 cleanly structures the problem.

- **Practical efficiency analysis identifying KV-cache bottleneck**: Table 1 systematically quantifies that KV cache grows from 1.1 GB (8 frames) to 17.2 GB (128 frames FP16), and INT2 quantization reduces it to 2.1 GB. This identifies a real deployment constraint rather than a hypothetical one, and is a useful practical finding.

- **Transparent reporting of performance degradation at 64 frames**: Table 4 shows performance declining at 64 frames across all three benchmarks, which the paper acknowledges. This honest reporting is valuable, even if the language ("plateau") understates the decline.

- **Plug-and-play applicability with minimal overhead**: The method requires no GPU training hours and only code changes, lowering the barrier to adoption (Sec. 4.1.2).

## Weaknesses

### Fatal
None.

### Major

- **No ablation isolating the token rearrangement from NTK-aware scaling**: The paper proposes two distinct contributions (token rearrangement in Sec. 3.2 and context window interpolation in Sec. 3.3) but explicitly refuses to ablate them independently: "We consider Alternative Video Token Rearrangement and Interpolating Video-LLM Backbone as one unit" (Sec. 4.3). Without testing NTK scaling + naive concatenation vs. NTK scaling + interleaved rearrangement, it is impossible to determine whether the rearrangement — the paper's main claimed novelty — contributes anything beyond simply processing more frames. This is the single most critical ablation the paper is missing, and its absence leaves the core technical contribution unsubstantiated.

- **Unexplained 8-frame anomaly undermining evaluation credibility**: At 8 frames, INTP should be a no-op: m=1 (single subsequence, rearrangement is identity), and the NTK scaling ratio s=L'/L=1 (scaling is identity). Yet Table 4 shows results *different* from baseline: ActivityNet-QA jumps +10.0 points (45.3→55.3), MSVD-QA drops -1.2, and MSRVTT-QA drops -1.0. A method that should be identical to the baseline producing a ±10 point swing on one benchmark is deeply concerning — it suggests either an implementation bug, a hidden pipeline change (prompt format, frame sampling strategy), or highly unreliable GPT-3.5 scoring. The paper does not acknowledge or explain this discrepancy.

- **Evaluation primarily on short-video benchmarks that do not test the claimed long-video capability**: The paper's framing is about understanding *long* videos ("Longer-Sequence LMMs"), but the primary benchmarks (MSVD-QA, MSRVTT-QA) average ~10-30 second videos where 8 frames is arguably sufficient. While EgoSchema and ActivityNet-QA involve longer videos, EgoSchema is multiple-choice only, and ActivityNet-QA shows only modest improvement at 32 frames (+3.6). The paper does not evaluate on established long-video benchmarks requiring temporal reasoning over minutes (e.g., MLVU, LongVideoBench, Video-MME longue). Demonstrating that the method works on short-to-medium videos is necessary but not sufficient to support the claim of enabling long-video understanding.

### Minor

- **KV-cache quantization is introduced as a contribution but never evaluated for quality impact**: The paper introduces INT2 KV-cache quantization (Sec. 3.4, Eq. 3.7) and claims INTP "optimizes memory usage during inference," but all accuracy results in Tables 2–4 use FP16. Without reporting accuracy under INT2 quantization, it is impossible to assess whether the memory savings come at acceptable quality cost. The contribution remains unevaluated.

- **Performance *decline* at 64 frames is understated as a "plateau"**: Table 4 shows INTP at 64 frames *degrading* below the 8-frame baseline on all three benchmarks (MSVD-QA: -3.2, MSRVTT-QA: -4.0, ActivityNet-QA: -3.8). The paper describes this as a "performance plateau" (Sec. 4.3) when it is actually a performance decline, which misrepresents the method's limitations.

- **RoPE context window scaling is a direct application of existing techniques**: Both context window scaling (Eq. 3.5, from Chen et al. 2023) and NTK-aware interpolation (Eq. 3.6, from Roziere et al. 2023) are applied to Video-LLMs without adaptation. While applying existing techniques to a new domain has value, the novelty of this component is limited. The paper is transparent about this lineage but should more clearly delineate what is novel vs. what is applied.

- **Evaluation limited to a single Video-LLM architecture (Video-LLaVA)**: Generalizability of the token rearrangement and NTK scaling pipeline to other Video-LLMs (e.g., LLaMA-VID, VideoChat2) is not demonstrated, limiting the scope of the claims about universal applicability.

### Trivial
- The roofline model numbers in Table 1 show some inconsistencies (32-frame INT2 decode time 22.9ms equals FP16; 64-frame INT2 decode 18.8ms is less than 32-frame's), though these are theoretical estimates, not measured latencies.

## Nice-to-Haves

- Ablation comparing naive concatenation + NTK scaling vs. interleaved rearrangement + NTK scaling at 16/32 frames to validate the rearrangement's contribution.
- Evaluation on a dedicated long-video benchmark (e.g., MLVU, LongVideoBench) to directly test the claimed capability.
- Accuracy results under INT2 quantization to validate the KV-cache compression contribution.
- Testing on a second Video-LLM architecture to demonstrate generalizability.
- An explanation for the 8-frame anomaly or at minimum an acknowledgment of evaluation noise from GPT-3.5 scoring.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"INT2 decode times in Table 1 are unreliable"**: The harsh critic questions the roofline model numbers, but these are clearly stated as theoretical estimates from a roofline model ("time estimated by roofline model represents the theoretical performance that the hardware can achieve"), not measured values. The inconsistencies may reflect roofline model limitations rather than errors. This is a soft criticism at best.

- **"Computation cost analysis is based on roofline model, not real inference"**: The paper clearly discloses this. Using a roofline model for cost analysis is a standard practice in systems work, and the paper is transparent about the methodology.

- **"Cherry-picked qualitative examples"**: All papers use selected qualitative examples. The harsh critic's objection is generic and not specific to anything misleading in this paper's examples.

- **"Calibration dataset for KV-cache quantization contradicts training-free claim"**: The paper states "no traditional data training processes" (Sec. 4.1.2), and calibration for PTQ (a small set of forward passes to determine quantization parameters) is standard and distinctly different from training. This is a scope-creep criticism.

- **"Missing related works"**: Removed per instructions (cannot verify existence of uncited works).

- **"Fig. 3 baseline appears to miss key frames (sampling issue)"**: This is a valid observation but it's about the baseline's frame sampling, not about INTP's methodology. The paper's contribution is about processing more frames, which addresses this sampling limitation.

- **Strength removed: "Ablation in Table 4 validates rearrangement's effectiveness"** (from Strength Finder): This conflicts with the verified Major weakness that the rearrangement is never ablated in isolation. Table 4 only varies frame count, which confounds rearrangement with NTK scaling and more frames.

## Novel Insights

The 8-frame anomaly in Table 4, where INTP at 8 frames (theoretically an identity operation) produces markedly different results from the baseline — particularly the +10 point swing on ActivityNet-QA — potentially reveals more about the unreliability of GPT-3.5-based evaluation than about the method itself. This has implications beyond this paper: the field's reliance on LLM-based scoring for open-ended VQA may be masking genuine methodological gaps. Additionally, the fact that the paper's claimed novelty (token rearrangement) cannot be disentangled from simply scaling up frames with existing RoPE interpolation raises a broader question: in training-free extensions, how much improvement comes from architectural tricks versus simply giving the model more visual evidence?

## Suggestions

- Run the critical missing ablation: test NTK-aware RoPE scaling + naive frame concatenation (no interleaving) at 16 and 32 frames, and compare against the full INTP pipeline. This single experiment would either validate or falsify the rearrangement's contribution.
- Investigate and explain the 8-frame anomaly. If it stems from GPT-3.5 evaluation noise, report confidence intervals or switch to a more reliable evaluation protocol. If it stems from an implementation artifact, document and fix it.
- Evaluate on at least one dedicated long-video benchmark to substantiate the "longer-sequence" claim.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Norton (long-term temporal video) | 9Cu8MRmhq2 | 8.0 | Far more novel, thorough ablations, diverse evaluation — this paper is well below |
| LLaVA-Interleave (multi-frame LMM) | oSQiao9GqB | 7.33 | Comprehensive dataset + model + ablations — this paper is well below |
| Coarse Correspondences (training-free MLLM boosting) | 8ibaVk4mU8 | 4.67 | Similar training-free angle but had better ablations; this paper is roughly comparable |
| StreamChat (training-free streaming video) | JbPb6RieNC | 5.8 | Similar training-free concept, had benchmark concerns; this paper has ablation gap but clearer improvement numbers |
| PPLLaVA (pooling for video LLM) | qUZY7ymDPr | 4.80 | Similar novelty concerns about applying known techniques — this paper is comparable |
| VAP (training-free key frame selection) | KtqZrNjvjd | 5.8 | Similar training-free video method, but VAP has better evaluation scope |
| MoE-VideoLLM summarization | ujNe7sybJu | 2.5 | Weak novelty, no ablations — this paper is above this |
| Grounded-VideoLLM | YCwN7wQA6W | 4.25 | Similar insufficient-ablation and limited-novelty issues — comparable |

This paper sits in the 4–5 range. It has a clear practical contribution (training-free 8→32 frame extension with consistent improvements) but is significantly weakened by: (1) the inability to attribute improvements to the claimed novel component (token rearrangement), (2) the unexplained 8-frame anomaly that undermines evaluation credibility, and (3) evaluation that doesn't directly test the claimed long-video understanding capability. It is below papers scoring 5.5+ (which typically have either thorough ablations or stronger novelty), and above papers scoring ≤3 (which have fundamental flaws or near-zero novelty). It is closest to the 4.0–4.8 band of papers with real but insufficiently validated contributions.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>