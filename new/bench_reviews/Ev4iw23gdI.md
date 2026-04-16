Now let me search for calibration papers to anchor the score.Now I have enough to write the final review. Let me synthesize everything.

---

## Summary

EMMA introduces two training-time auxiliary modules for Mamba-based MLLMs: (1) a **Pixel-wise Alignment Loss (PAL)** that applies an L2 reconstruction objective on a generated visual feature decoded via a small Mamba decoder, and (2) a **Multi-scale Feature Fusion (MFF)** module that hierarchically fuses intermediate LLM hidden states through cross-attention and Mamba projections before applying the reconstruction loss. The method adds zero inference overhead and shows improvements over the Cobra baseline across most benchmarks, with notable gains in HallusionBench and MME. The paper frames these as enabling "autoregressive visual-token generation" alongside text generation.

---

## Strengths

- **Real and consistent gains over the matched Mamba baseline (Cobra):** EMMA-V1 surpasses Cobra on nearly all evaluated benchmarks using the same backbone, data, and scale. The TextVQA gain (+4.8 points) and HallusionBench gain (+9.6 points, 51.0 vs. 41.4) are substantial and not trivially explainable by noise, as confirmed by the ablation in Table 4.

- **Zero-inference-overhead design is practically valuable:** The paper clearly demonstrates (Sec. 3.3, Table 3) that MFF and the visual decoder are only active during training, making the method drop-in compatible for deployment on Cobra-equivalent infrastructure.

- **Informative ablation study:** Table 4 clearly decomposes contributions of PAL vs. MFF vs. alternatives (+CSM, +AVF), showing that: PAL primarily drives TextVQA and HallusionBench gains; MFF is critical for MME; feature-level alignment (+AVF) degrades nearly all metrics. This is one of the stronger parts of the submission.

- **Well-motivated problem identification:** The observation that Mamba LLMs gradually lose spatial structure of visual features through depth (Fig. 1, visualization) is compelling, and the proposed solutions are direct responses to this identified problem.

- **Efficiency story is real:** EMMA-V2 achieves ~150 tok/s vs. ~40 tok/s for transformer models of similar scale—a ~3.7× speedup—which is practically meaningful for deployment.

---

## Weaknesses

### Fatal
*None triggered.*

### Major

- **Speed claim is misattributed to the method rather than the backbone.** Table 3 shows EMMA-V1 and Cobra having *identical* latency (138.95 tok/s, 1.84s), since they share the same MambaV1-2.8B backbone. The only latency improvement over Cobra is in EMMA-V2, which uses the MambaV2 backbone—a backbone-level change unrelated to PAL or MFF. The abstract's claim that "our model shows lower latency than other Mamba-based MLLMs" is technically supported only by EMMA-V2, but it misattributes the speed gain to EMMA's proposed method rather than to the MambaV2 backbone. The conclusion repeats this framing. This is a structural misattribution, not just a presentation nitpick, because it inflates the apparent value of the proposed contributions.

- **The "autoregressive visual generation" framing does not match the actual implementation.** Eq. (5) introduces a factorized probability over a visual token sequence $\hat{X}_v$, implying sequential autoregressive generation. But the implemented training objective (Eq. 6/9) is simply a pixel-level L2 loss between the decoder output and the original image—an auxiliary reconstruction loss. The paper never specifies: whether $\hat{X}_v$ is generated autoregressively or in parallel, what spatial ordering is used, whether teacher forcing is applied, or whether $f_{dec}$ has autoregressive dependencies. As written, the core mechanism is standard reconstruction regularization, not autoregressive visual generation. The mismatch between the narrative and the implementation is significant because it overstates novelty relative to concurrent works like ROSS (which also uses image reconstruction as auxiliary supervision but with a more carefully described denoising objective).

- **The MME concentration problem is under-analyzed.** The ablation (Table 4) shows that removing MFF causes MME to collapse from 1572.8 to 1294.1 (nearly exactly back to the Cobra baseline), while most other metrics change by less than 1 point. This strongly suggests MFF is disproportionately affecting whatever MME measures—but the paper never breaks MME into subcategories or investigates what is driving this. If MFF is genuinely helping, the paper should explain *why* its effect is so concentrated in MME. This raises questions about whether the results reflect genuine cross-modal alignment improvement or a benchmark-specific artifact.

### Minor

- **Weak hallucination evidence on POPE.** The POPE gain is only 0.8 points (88.0 vs. 87.2 for Cobra), which is marginal. The paper's broad claim that EMMA "exhibits lower degrees of hallucination" leans too heavily on the much stronger HallusionBench gain. The two benchmarks measure different phenomena (object hallucination vs. visual illusion/knowledge hallucination), and the POPE result alone does not support the general claim.

- **Training cost not reported.** The paper emphasizes zero inference overhead but never quantifies the additional training cost (time, memory, FLOPs) from the MFF module and visual decoder. For a method whose value proposition is efficiency, this is a notable omission.

- **Ablations do not sweep MFF design choices.** The paper uses three intermediate layers for MFF but does not justify or ablate this choice (which layers, how many). Similarly, the decoder capacity (4 Mamba + linear layers) is fixed without sensitivity analysis. These are relevant for understanding whether the gains are robust.

- **Scale and generalization not tested.** All experiments are at ~2.8B parameters on 1.2M samples. Whether the proposed alignment strategies remain effective at larger scales (e.g., 7B backbone, more training data) is unknown. Since the paper's motivating claim is about a fundamental bottleneck in Mamba MLLMs, scale invariance matters.

### Trivial

- **VQAv2 still underperforms VL-Mamba** (76.3 vs. 76.6 for V1, 75.7 for V2), though the gap is small (<1%) and the paper acknowledges it.
- VSR is marginally *lower* for EMMA-V1 vs Cobra (51.5 vs. 51.7), consistent with the paper's explanation that coarse binary tasks don't benefit as much.

---

## Nice-to-Haves

- **Qualitative reconstruction examples from $f_{dec}$.** Showing what the pixel decoder actually reconstructs would clarify whether PAL enforces meaningful structural preservation or primarily captures low-frequency statistics. This would also address the implicit concern that L2 loss encourages blurry reconstructions.
- **Alternative reconstruction losses.** Comparing L2 against perceptual loss (LPIPS) or CLIP-feature-space alignment would validate the L2 design choice and strengthen or revise the "fine-grained" narrative.
- **Quantitative feature degradation analysis.** The core motivation (Mamba layers progressively lose visual feature quality) is supported only by a single qualitative visualization (Fig. 1). A probing analysis (e.g., feature similarity to input across layers for both Cobra and EMMA) would make the motivation rigorous.
- **Deeper MFF layer-selection ablation.** Sweeping over which intermediate layers are fused and how many fusion blocks are used would clarify the sensitivity of the MFF design.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Architecture description is ambiguous—$\psi$ integration during training vs. inference."** The paper clearly states in Sec. 3.3: "the feature fusion and visual decoding stage only occurs during training." The inference concern is addressed.

- **Harsh Critic: "Section 3.3 conflates encoder features with LLM features."** Reading Sec. 3.3 carefully, the paper consistently refers to "hidden visual features $\{\bar{X}_i, \bar{X}_j, \bar{X}_k\}$ from layers $i, j, k$" of the Mamba LLM (not the vision encoder). The paper's use of "intermediate features of the pretrained visual encoder" in the introduction is informal but the formalism in Sec. 3.3 is clear.

- **Harsh Critic: "Evaluation is unfair because EMMA uses SigLIP + DINOV2 dual encoders."** EMMA explicitly follows and is compared to Cobra (same dual encoders), making this comparison matched. The Human Finder raised the same point, but it only applies when comparing against models with single encoders—the primary comparison is against Cobra, which uses the identical encoder setup.

- **Harsh Critic: "The paper never shows how visual tokens are generated sequentially."** While this is a conceptual issue (addressed in Major Weaknesses above as overclaiming), removing it as a separate point avoids double-counting.

- **Neutral/Spark: "Comparison with EMU is incomplete."** EMU uses a 13B LLaMA with 3.4B training samples and a Stable Diffusion decoder. The architectural/scale gap makes deep comparison impractical; EMMA already acknowledges this difference in Sec. 4.2.

- **Spark: "Evaluate on multi-image or long-context benchmarks."** This is outside EMMA's stated scope. The paper does not claim to target long-context; evaluating on short-context benchmarks is appropriate for its stated contribution of better visual feature alignment.

---

## Novel Insights

The most genuinely novel observation surfaced by the reviews—and verified against the paper—is the MME concentration phenomenon: removing MFF collapses MME almost exactly back to Cobra's baseline (1572.8 → 1294.1 vs. Cobra's 1294.3) while barely affecting other metrics. This suggests MFF's contribution is highly benchmark-specific and poorly understood. The paper should either decompose MME subcategories to explain this or acknowledge it as an open question. The related finding that feature-alignment (+AVF) catastrophically degrades performance while still achieving a reasonable HallusionBench score is also noteworthy—it implies that visual self-supervision helps hallucination even when overall quality degrades, pointing toward disentangled mechanisms worth investigating.

---

## Suggestions

1. **Disambiguate the speed claim:** In the abstract and conclusion, explicitly state that EMMA-V1 matches Cobra's latency (as training-time modules add no overhead) and that the latency advantage is due to the MambaV2 backbone in EMMA-V2. Do not attribute the speed advantage to the proposed method.
2. **Revise the autoregressive framing:** Rename and re-describe the PAL mechanism as "auxiliary pixel-level reconstruction supervision" rather than "autoregressive visual generation." Eq. (5)'s sequential factorization should be either implemented as written (with ordering and teacher forcing specified) or removed in favor of a direct description of the L2 objective.
3. **Analyze MME subcategories:** Break MME into perception and cognition sub-scores and report how MFF affects each subtype to explain the +278 point jump.
4. **Report training overhead:** Add a table comparing training time/memory for Cobra vs. EMMA-V1/V2 to provide a complete efficiency picture.
5. **Ablate MFF layer selection:** Show results for 1, 2, and 3 intermediate layers and different layer combinations to establish robustness of the hierarchical alignment.

---

## Score and Decision

**Calibration anchor papers:**

| Paper | Topic | Score | Decision |
|---|---|---|---|
| ROSS (`8q9NOMzRDg`) | Image reconstruction supervision for VLMs | 5,6,6,6,6 avg≈5.8 | Accept (Poster) |
| MambaVLM (`0A6f1b66pE`) | Mamba-based MLLM with visual scanning | 6,3,3,5,6 avg≈4.6 | Withdraw (Reject) |
| Hybrid SSM MLLM (`cagNCwQEEN`) | Mamba+Transformer hybrid MLLM | 3,3,5,3,3 avg≈3.4 | Reject |
| SeTok (`n64NYyc6rQ`) | Vision tokenization for semantic alignment | 6,6,5,8,6 avg≈6.2 | Accept (Poster) |

**Reasoning:** EMMA is most directly comparable to ROSS—both apply auxiliary image reconstruction supervision to improve visual feature quality in MLLMs. ROSS got accepted as a poster (~5.8 avg). EMMA is *weaker* than ROSS in key ways: (a) ROSS applies denoising reconstruction in latent space with careful design rationale, while EMMA uses simpler pixel L2 without exploring alternatives; (b) EMMA has a significant autoregressive overclaiming issue absent in ROSS; (c) EMMA's speed claim misattribution is a notable presentation problem. However, EMMA is *stronger* than the rejected Mamba MLLM papers (MambaVLM, Hybrid SSM) because it has a clearer contribution (training-time visual supervision), concrete improvements over a matched baseline, and an informative ablation. The paper sits below ROSS's level but clearly above the rejected Mamba-MLLM group. The major weaknesses (speed misattribution, autoregressive overclaiming, MME concentration) are real and substantive but do not invalidate the core contribution of improving visual feature quality in Mamba MLLMs through auxiliary reconstruction. A score of **5.0** (borderline reject) is appropriate—the paper has a real contribution, but the overclaiming issues and analytic gaps require revision before the work can be accepted.

**Originality:** Moderate. Training-time auxiliary image reconstruction is not new (ROSS, EMU), but applying it to address Mamba MLLM-specific visual degradation is well-motivated.  
**Importance:** Moderate. Mamba-based MLLMs are a real research direction, and showing training-time visual supervision helps is useful.  
**Claim support:** Weak-to-moderate. Benchmark gains are real but speed claims are misattributed and autoregressive framing is unsupported.  
**Experimental soundness:** Moderate. Ablations are directionally helpful but many design choices are not isolated.  
**Writing clarity:** Acceptable but the abstract overreaches.  
**Community value:** Modest, as a practical recipe for Mamba MLLM practitioners.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>