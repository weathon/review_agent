Now I have all the information needed. Let me compose the final review.

## Summary

This paper proposes INTP, a training-free method to extend existing Video-LLMs (specifically Video-LLaVA) from processing 8 frames to 32 frames without retraining. The method combines three components: (1) a video token rearrangement technique that interleaves tokens from separately-encoded frame groups to bypass the fixed encoder/projector frame limit, (2) NTK-aware RoPE interpolation to extend the LLM backbone's context window, and (3) post-training KV-cache quantization to mitigate the resulting memory bottleneck. Experiments across five video QA benchmarks show modest but consistent improvements at 32 frames.

## Strengths

- **Practical and well-motivated problem framing.** Extending Video-LLMs to more frames without retraining is a genuinely valuable goal given the prohibitive training costs (Sec. 3.1 notes ~200 A100-hours even for Video-LLaVA fine-tuning). The training-free angle is clearly motivated.
- **Consistent performance improvements at 32 frames across all five benchmarks.** Tables 2–3 show gains of +1.3 (MSVD-QA), +2.2 (MSRVTT-QA), +3.6 (ActivityNet-QA), +4.4 (NExT-QA Temporal), and +1.6 (EgoSchema) when going from 8 to 32 frames, supporting the claim that more frames provide useful information.
- **Informative efficiency analysis.** Table 1 systematically profiles OPs, decode time, total memory, and KV-cache storage across frame counts (8–128) and quantization levels (FP16 vs. INT2), correctly identifying KV-cache as the dominant memory bottleneck (17.2GB of 30.1GB at 128 frames).
- **Honest reporting of scaling limitations.** Table 4 transparently shows performance degradation at 64 frames, and the paper acknowledges this as a limitation of the NTK-based extension method rather than hiding it.

## Weaknesses

### Fatal

None.

### Major

- **Token rearrangement advantage over naive concatenation is never experimentally validated.** The paper's first claimed contribution (Sec. 3.2) is that rearrangement "preserves temporal consistency" whereas naive concatenation produces "distorted temporal representations" (p. 4, lines 77–89). Yet no experiment compares rearrangement against naive concatenation of encoded subsequences. If both produce similar results, this claimed contribution collapses. The rearrangement is presented as a core technical novelty but is asserted rather than demonstrated. Tables 2–4 only show results with the full INTP pipeline; there is no ablation isolating the rearrangement's contribution.

- **Evaluation does not specifically test the core claim of enabling "longer video understanding."** The paper's title and abstract frame the contribution as enabling Video-LLMs to "understand longer video content." Yet all five benchmarks (MSVD-QA, MSRVTT-QA, ActivityNet-QA, NExT-QA, EgoSchema) are standard video QA benchmarks where 8 frames already provide reasonable coverage for many questions. None are specifically designed to require long-range temporal reasoning across distant events—i.e., where 8 frames are demonstrably insufficient. Without evaluation on benchmarks explicitly designed for long-video understanding (e.g., LongVideoBench, or targeted evaluation where temporal coverage is the bottleneck), the headline claim remains untested. The marginal improvements observed (+1.3 to +3.6 points) could result from denser sampling of already-adequately-covered videos rather than genuine long-video comprehension.

- **Unexplained anomalous results at 8 frames undermine internal consistency and the "plug-and-play" claim.** Table 4 shows that INTP at 8 frames (where m=1, meaning no rearrangement occurs) produces contradictory effects: MSVD-QA drops (70.7→69.5), MSRVTT-QA drops (59.2→58.2), but ActivityNet-QA jumps dramatically (45.3→55.3, +10.0). Since at 8 frames the only change is NTK-aware RoPE scaling at the original context length, (a) the +10.0 swing on ActivityNet from just modifying RoPE is suspicious and unexplained, (b) the degradation on MSVD-QA and MSRVTT-QA contradicts the claim in Sec. 4.2 that INTP "acts as a plug-and-play enhancement," and (c) ActivityNet performance under INTP *decreases* when going from 8→16 frames (55.3→46.9), then partially recovers at 32 frames (48.9)—a non-monotonic pattern the paper ignores entirely.

### Minor

- **KV-cache quantization impact on accuracy is not reported.** Section 3.4 introduces INT2 KV-cache quantization as part of the INTP system, and Table 1 shows its efficiency gains. However, no accuracy numbers are reported for the quantized model, making it impossible to assess whether INT2 quantization degrades the quality of the answers. Without this, the quality-efficiency tradeoff claimed in Section 3.4 is incomplete.

- **Performance degradation at 64 frames is attributed to NTK-based extension limitations without evidence.** Section 4.3 states "a limitation in the NTK-based LLM backbone extension method" but provides no diagnostic evidence. The degradation could alternatively stem from the token rearrangement producing worse features for temporally sparse subsequences, the visual encoder struggling with more distant frame pairings, or simply the LLM struggling with more tokens regardless of positional encoding. Ablating rearrangement vs. scaling separately at 64 frames would clarify this.

## Nice-to-Haves

- Evaluate on benchmarks specifically designed for long-video understanding where 8 frames are demonstrably insufficient and 32 frames provide genuinely new temporal information. This is the single most important gap.
- Compare token rearrangement against naive concatenation to validate (or falsify) the rearrangement's claimed advantage.
- Report accuracy under INT2 KV-cache quantization to complete the efficiency analysis.
- Diagnose and explain the anomalous 8-frame INTP results, especially the +10.0 on ActivityNet-QA.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"No error bars / statistical significance tests"** — Demanding confidence intervals for GPT-based evaluation is not standard practice in this community for single-run benchmark evaluations. While the improvements are modest, this is a generic critique that doesn't threaten the core claim specifically.

- **"Section 3.3 is essentially a recap of prior work" / "Section 3.4 is standard post-training quantization"** — The paper's novelty is explicitly in the *combination and application* of existing techniques to the Video-LLM setting. While the individual components are borrowed, criticizing them individually misses the paper's stated framing. This is already partially captured by the more substantive concern about unvalidated rearrangement.

- **"Qualitative results are cherry-picked"** — Standard practice; all qualitative examples in papers are selected to illustrate differences. The paper does show both a correct and incorrect case from the baseline, which is typical.

- **"The abstract's '32 frames' claim is cherry-picked"** — The paper transparently shows results at 8, 16, 32, and 64 frames in Table 4. Reporting the best-performing configuration in the abstract is standard practice, not cherry-picking.

- **"No comparison with encoding each frame independently as images"** — This would test a different approach entirely and is outside the paper's scope. The paper is about extending *existing* Video-LLMs, not proposing an alternative architecture.

- **Missing related works** — Per instructions, I do not flag missing citations.

## Novel Insights

The unexplained +10.0 accuracy jump on ActivityNet-QA when applying NTK-aware RoPE scaling at the *original* context length (8 frames, where no rearrangement occurs) may reveal something important about how RoPE scaling interacts with visual tokens specifically. If confirmed, this could suggest that RoPE frequency adjustments have an outsized effect on attention patterns for visual token sequences compared to pure text—a hypothesis worth investigating in future work, as it might imply that the "right" scaling factor for multimodal models differs from that of pure LLMs.

## Suggestions

- Add a direct comparison between token rearrangement and naive concatenation of encoded subsequences (e.g., running the same 32-frame setup but concatenating tokens in encoding order rather than absolute position order). This single experiment would validate or invalidate the paper's first technical contribution.
- At minimum, add a brief discussion of the anomalous 8-frame results in Table 4. If the +10.0 on ActivityNet-QA is reproducible, explain why NTK-aware scaling at the original context length helps ActivityNet but hurts MSVD/MSRVTT. If it's an artifact, acknowledge it.
- Test on at least one benchmark where video length is the explicit bottleneck (e.g., a long-video subset of an existing benchmark) to substantiate the "longer video understanding" claim.

## Score and Decision

**Calibration anchors used:**

| Paper | Score | Relation to this paper |
|-------|-------|----------------------|
| Norton (Multi-granularity Correspondence) | 8.0 | High anchor: novel OT framework for long-video, extensive ablations — far above this paper in novelty and validation |
| StreamChat | 5.8 | Medium-high anchor: training-free video framework with memory system, more comprehensive but with benchmark quality issues — above this paper due to broader evaluation |
| VAP (Video Active Perception) | 5.8 | Medium anchor: training-free long-form video QA with novel keyframe selection — above due to novel mechanism and explicit long-video evaluation |
| MVU | 5.67 | Medium anchor: off-the-shelf tools for video understanding, novel framework — similar in using existing components but more novel in integration |
| PLLaVA | 5.25 | Medium-low anchor: limited novelty pooling strategy but more comprehensive evaluation — slightly above this paper |
| Hybrid SSM MLLM | 3.4 | Low anchor: claims long-video but doesn't test on long-video benchmarks, "just follows common practice" — below this paper since this paper at least shows consistent improvements |
| MixAttention | 2.0 | Very low anchor: no novelty at all, just evaluates a blog post — far below this paper |

This paper sits between the Hybrid SSM MLLM (3.4, similarly fails to validate on long-video benchmarks and has limited novelty) and the medium-scoring training-free video papers (5.3–5.8). It shares the Hybrid SSM paper's weakness of not testing on long-video benchmarks, but shows more concrete improvements. However, the unvalidated rearrangement claim and the anomalous 8-frame results place it below PLLaVA and MVU. The combination of (a) unvalidated core technical contribution, (b) evaluation mismatch with stated claim, and (c) unexplained anomalous results keeps it firmly in the reject range.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>