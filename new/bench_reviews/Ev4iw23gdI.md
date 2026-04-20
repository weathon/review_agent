## Summary

EMMA introduces two complementary training-only modules — a Pixel-wise Alignment Loss (PAL) and a Multi-scale Feature Fusion (MFF) — to address visual feature degradation within Mamba-based MLLMs. Both modules operate exclusively during training, preserving the sub-quadratic inference speed of the Mamba backbone. The approach delivers consistent improvements over the Mamba MLLM baseline (Cobra) across most benchmarks, with particularly large gains on MME, HallusionBench, and TextVQA, alongside ablation studies that clearly validate each component's contribution.

## Strengths

- **Clear, well-isolated improvements over the direct Mamba baseline.** EMMA-V1 uses the identical MambaV1-2.8B backbone and 1.2M dataset as Cobra, differing only in the addition of PAL and MFF. Under these controlled conditions (Table 1, rows 188–189), EMMA-V1 improves on every benchmark, with headline gains of +278.5 on MME and +4.8 on TextVQA. This directly attributes the improvement to the proposed modules.
- **Clean ablation study isolating each component.** Table 4 (MambaV1 version only, eliminating backbone confounds) shows that removing MFF drops MME from 1572.8 to 1294.1 (nearly matching Cobra), and removing PAL collapses HallusionBench from 51.0 to 41.4 and TextVQA from 57.2 to 52.4. This provides strong evidence that both structural (PAL) and hierarchical (MFF) alignment contribute independently to the gains.
- **Inference efficiency preserved by design.** PAL and MFF (cross-attention blocks and image decoder) are used only during training (Sec. 3.3: "poses no additional computational overhead in inference"). Table 3 confirms EMMA-V1 matches Cobra's inference throughput (138.95 tokens/sec), while EMMA-V2 on the MambaV2 backbone achieves 149.96 tokens/sec — approximately 3.6× faster than similarly-scaled transformer MLLMs (~40 tokens/sec).
- **Motivation grounded in visual evidence.** Figure 1 qualitatively illustrates progressive blurring of intermediate visual features across deep layers in Cobra, and shows EMMA's features remain structurally coherent. The hallucination analysis (Table 2) further links this to concrete performance: EMMA achieves 51.0 on HallusionBench vs Cobra's 41.4, demonstrating reduced visual hallucination.

## Weaknesses

### Fatal

None

### Major

### Minor

- **Latency claims conflate backbone upgrades with proposed method.** Table 3 shows EMMA-V1 (MambaV1-2.8B) has *identical* latency to Cobra (138.95 tokens/sec, 1.84s). The ~8% speedup comes solely from EMMA-V2's MambaV2-2.7B backbone, not from PAL or MFF. While the paper body honestly acknowledges this ("Our model achieves even better runtime than Cobra due to more efficient processing in the MambaV2 LLM backbone," Sec. 4.3, line 219), the Abstract and Introduction repeatedly claim the *method itself* yields lower latency, which overstates the contribution. The speed gains should be attributed to the backbone choice, not to EMMA's modules, which add zero inference-time benefit.

- **Pixel-wise alignment via L2 regression is an imperfect proxy for structural similarity.** The PAL minimizes raw L2 pixel distance ($\|f_{dec}(\hat{X}_v) - X_v\|_2^2$, Eq. 6), which is known to prioritize low-frequency statistics and produce blurred reconstructions rather than preserving high-frequency structural detail. The authors claim this enforces "structural alignment" (Sec. 1, line 44), but L2 distance measures neither SSIM nor perceptual similarity. The ablation (+AVF) compares pixel-level alignment against feature-level alignment, finding the former superior — but this does not prove L2 is optimal, only that it is better than the tested alternative (aligning on encoder-extracted features). Comparing against perceptual or contrastive objectives would strengthen the claim.

- **Comparisons to larger transformer baselines are confounded by data and pretraining scale.** Table 1 compares EMMA (1.2M data, 300B–627B token LLM pretraining) against models trained on 1.3T–3.4B tokens (Table 1, rows for MobileLLaMA, Phi-2, EMU). The paper explicitly acknowledges these disparities (Sec. 4.2, lines 196–197), which mitigates but does not eliminate the concern. The most fair comparison is within the Mamba group (same 1.2M data, same backbones), and there the gains are clear. The transformer comparisons provide useful context but should not carry too much weight in evaluating the contribution.

### Trivial

None identified.

## Nice-to-Haves

- Report training throughput (iterations/sec), peak VRAM, and total training FLOPs for EMMA vs. Cobra, to quantify the training-time compute cost of the cross-attention blocks and autoregressive decoding path. This would help future practitioners assess the trade-off.
- Include a layer-wise quantitative analysis (e.g., CKA or reconstruction error across depth) to complement Figure 1's qualitative visualization and empirically measure the extent of feature degradation and MFF's mitigation across layers.
- A controlled baseline training Cobra under the identical 1.2M data, 2-epoch, no-pretrain regime with the same seeds would further corroborate that the PAL+MFF modules alone account for the gains, removing any residual uncertainty around training stochasticity.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Harsh Critic #1: "L2 pixel loss is fundamentally mismatched with structural alignment and invalidates the core claim."** The ablation (Table 4) shows PAL produces strong results, and feature-level alignment (+AVF) degrades severely. While L2 is not a perfect structural metric, the paper demonstrates its empirical effectiveness. The critic's claim that it "invalidates" the contribution overreaches — the method works in practice even if the metric is imperfect. Downgraded to a minor weakness.
- **Harsh Critic #2: "Latency gains are confounded with backbone upgrades, invalidating efficiency claims."** This is partially valid — the speedup does come from the V2 backbone — but the paper body is transparent about this (Sec. 4.3 explicitly states so). Table 1's EMMA-V1 vs Cobra comparison is fair (same backbone, same data, clear gains). The claim of "invalidation" is too strong; this is a presentation overclaim, not a fundamental flaw.
- **Harsh Critic #3: "Training protocol deviation makes Table 1 comparisons unfair."** Within the Mamba MLLM group (Cobra, VL-Mamba, EMMA-V1), all models use the same 1.2M data and MambaV1-2.8B backbone. The comparison is controlled at this level. The author also transparent discusses differences with transformer groups (Sec. 4.2). The criticism conflates cross-family comparisons with the core baseline comparison, which is fair.
- **Harsh Critic: "Mamba inherently encodes positional information; the claim about lack of positional embeddings is wrong/uncited."** Section 1 (line 41) says Mamba LLM layers "lack structural constraints on the visual features," not that they lack *any* positional encoding. This is about the absence of spatial priors specific to visual data within the Mamba backbone — a different claim. Misread by the harsh critic.
- **Harsh Critic: "EMU comparison is inappropriate — EMU uses diffusion decoders and trains on billions of tokens."** The paper actually acknowledges this disparity (Sec. 4.2, lines 198–199: "Both EMU and EMU2 require a sizable stable diffusion decoder... demand substantially more data and significantly larger parameters"). The authors position EMMA as competitive *despite* these disadvantages, not equivalent. The harsh critic ignored this acknowledgment.
- **Harsh Critic: "Decoder architecture severely under-specified."** Sec. 3.1 (line 151) states: "The image decoder consists of a combination of 4 Mamba and linear layers." Sec. 4.1 notes details are in the Appendix. This is sufficient for a conference paper.
- **Strength Finder: "Competitive with models trained on vastly more data."** While true, this is less a strength of the method and more contextual framing. Kept as supporting context, not promoted.
- **Strength Finder: "Competitive with EMU despite less data."** Similar to above — useful context but not direct evidence of the method's contribution. Included as supporting strength.

## Novel Insights

The paper makes a compelling case for applying reconstruction-based visual supervision *within* MAMA LLMs, where prior work has mostly focused on preserving visual features before they enter the LLM. The key insight — that autoregressive visual token prediction can complement textual next-token prediction to create a structurally grounded multimodal representation — is conceptually neat. The hierarchical fusion of intermediate layer features via cross-attention and Mamba blocks is a practical engineering contribution that addresses information vanishing in deep state-space sequences. However, the approach is incremental in the broader MLLM literature (similar ideas exist using diffusion decoders or perceptual losses), and the gains, while consistent, do not dramatically shift the Mamba-MLLM competitive frontier.

## Suggestions

- Clearly attribute the latency advantage in the Abstract to the MambaV2 backbone upgrade rather than to PAL/MFF, which add no inference-time benefit.
- Consider adding a comparison of L2 pixel loss against at least one perceptual or feature-space objective (e.g., LPIPS or SSIM loss) to justify the design choice beyond the +AVF ablation.
- In the latency table, explicitly separate EMMA-V1 (same backbone as Cobra) from EMMA-V2 to avoid conflating backbone efficiency with method contribution.

## Score and Decision

I compared this paper against several calibration anchors:

- **bIlnpVM4bc.md (Samba, scores 8,6,6, Accept Poster):** Simple idea, well-executed, solid experiments with strong ablation. Similar profile but Samba is an architecture-level contribution with broader impact.
- **he6mX9LTyE.md (Kosmos-G, scores 6,6,6, Accept Poster):** Modest novelty, consistent engineering improvement, solid ablation — closely matches EMMA's profile.
- **EdMb9TqqDY.md (scores 6,6,6,6, Accept Poster):** Consistent improvements over baselines, solid empirical results.
- **XKQ2qzajbU.md (GlobalMamba, scores 5,5,5,5, Withdrawn):** Consistent improvement but reviewers questioned whether gains came from the core idea or confounding factors — similar concern about backbone upgrade for latency.
- **VYRT8ajHRr.md (Mono-InternVL, scores 5,5,5,6, Withdrawn):** Unfair baseline comparison concerns confound the evaluation — similar to the latency confound in EMMA.

EMMA sits in the range of a solid, incremental contribution with clear empirical validation. The controlled Mamba-group comparison (EMMA-V1 vs Cobra, same backbone and data) is genuinely fair and shows consistent gains with clean ablation. The weaknesses (latency attribution, L2 loss simplicity, transformer comparison confounds) are real but manageable — they do not invalidate the core contribution. Compared to papers accepted as posters (6s) with similar profiles, EMMA performs comparably or better on the fairness of its core comparison. It falls short of higher-scoring papers (7-8) where the contribution is more fundamental or broadly impactful.

<pineapple>6.0</pineapple>