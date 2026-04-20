Now let me run calibration searches to properly anchor my score.Now I have sufficient calibration information to write the final review.

---

## Summary

EMMA introduces two training-time auxiliary modules for Mamba-based MLLMs: (1) a Pixel-wise Alignment Loss (PAL) that supervises visual features extracted from the Mamba LLM to reconstruct the input image, and (2) a Multi-scale Feature Fusion (MFF) module that aggregates intermediate LLM layer features to prevent gradual loss of fine-grained visual details. Both modules are active only during training, adding zero inference overhead. Experiments on diverse benchmarks show improvements over the Cobra baseline, most notably a ~279-point gain on MME and notable hallucination reduction on HallusionBench.

---

## Strengths

- **Identification of a genuine and well-characterized problem**: The observation that Mamba LLMs (unlike transformer-based LLMs with positional embeddings) gradually lose fine-grained visual detail in intermediate layers is a valid architectural insight, and the motivation distinguishes EMMA from prior work that improves visual representations *before* the LLM (Section 2.3).
- **Training-only design adds zero inference cost**: The explicit design choice to run MFF and the pixel decoder only during training (Section 3.3) is architecturally clean and practically important. Inference latency is unaffected.
- **Hallucination reduction is convincingly demonstrated**: Table 4 shows that removing PAL causes HallusionBench to drop from 51.0 → 41.4, one of the clearest and most task-interpretable ablation signals in the paper. Table 2 situates EMMA's HallusionBench score (51.0) ahead of the much larger LLaVA-1.5 (47.1) and BLIP2-T5 (48.1).
- **More thorough ablation than typical MLLM papers**: Table 4 covers five conditions including two design alternatives (CSM, AVF), and evaluating both POPE (language-side hallucination) and HallusionBench (vision-side hallucination) is a thoughtful choice given the alignment framing.
- **MFF formulation is clearly specified and implementable**: Equations 7–8 give a complete, self-contained definition.

---

## Weaknesses

### Fatal
None.

### Major

- **Unexplained super-additive interaction on the headline result (MME)**: Table 4 shows that -MFF alone gives MME = 1294.1 and -PAL alone gives MME = 1294.3 — both identical to Cobra's baseline (1294.3). Only with both components combined does MME jump to 1572.8 (+278 points). Neither component individually moves the needle on MME at all, yet together they produce the paper's most dramatic result. This super-additive interaction — where the sum is far greater than either part — is given zero mechanistic explanation. The authors do not consider whether this reflects measurement variance in MME (a benchmark known to be sensitive to prompt formatting), a training synergy, or a run-specific fluctuation. Without at minimum multi-seed variance estimates on this result, the 278-point MME gain cannot be taken as a reliable finding. Notably, cleaner signals *do* exist (HallusionBench, TextVQA), but the abstract and introduction foreground MME as the headline result.

- **Overclaimed "autoregressive" framing of PAL**: The paper presents Eq. 5 as an autoregressive factored probability over visual tokens — directly paralleling next-token text prediction — and the abstract and contributions frame this as "autoregressively optimizing the learning and processing of spatial image-level features." However, the actual implemented loss (Eq. 6) is simply `‖f_dec(X̂_v) − X_v‖₂²`: a standard L2 reconstruction loss on decoded visual features. There is no sequential generation of image patches, no causal chain between visual positions, and no evidence that $\hat{X}_v$ is produced as an autoregressive sequence in the visual sense. The actual operation is: extract visual hidden states at LLM visual positions, decode to pixel space, compute L2 loss. This is a perfectly valid auxiliary reconstruction loss and a genuine contribution — but calling it "autoregressive visual alignment" overstates the novelty and is the description of the paper's core novelty pillar. The paper should describe what it actually implements.

- **Training recipe difference partially confounds the baseline comparison**: Section 4.1 explicitly states EMMA discards the pretrain phase (finetune-only), while Cobra uses pretrain + finetune. The ablation claims that "-PAL is equivalent to training the plain Cobra model," yet the numbers in the -PAL row are exactly identical to Cobra's published scores (e.g., MME 1294.3, VQAv2 74.9, GQA 59.1 — all matching to the decimal). This exact coincidence is suspicious: either the authors used Cobra's published results as their -PAL ablation without re-training, or the pretrain stage genuinely has no effect on any metric whatsoever. In either case, the claimed equivalence is not justified, and a Cobra-without-pretrain control is needed to cleanly attribute gains to PAL and MFF versus training recipe.

### Minor

- **Fig. 1 is partly circular evidence**: The "magnitude of reconstructed spatial activations" metric used to visualize EMMA's intermediate features is directly related to what PAL/MFF are explicitly trained to optimize (pixel reconstructability). Showing that EMMA's features reconstruct images better than Cobra's is tautological — it demonstrates the model learned its training loss, not that it independently preserves semantic structure. The Cobra half of Fig. 1 (showing degradation) remains valid and informative; the comparison to EMMA is problematic. An independent probe (e.g., spatial relationship accuracy from intermediate representations) would be more convincing.

- **Speed claim misattributed to EMMA**: Table 3 shows EMMA-V1 and Cobra have *identical* throughput (138.95 tok/s, 1.84 s). The faster EMMA-V2 speed comes from the MambaV2 backbone, not from any EMMA-specific design. The "nearly four times faster than transformer-based MLLMs" claim in the abstract properly belongs to Mamba architectures generally.

- **Catastrophic +AVF degradation unexplained**: The feature-alignment alternative (+AVF) causes performance to collapse (VQAv2 76.25 → 52.8, MMB 53.2 → 25.0). The paper attributes this to "robust structural information inherent in pixel-level images" but a degradation of this magnitude strongly suggests training instability (gradient scale mismatch, loss weight imbalance, normalization failure) rather than a clean comparison between alignment targets. The paper presents it as conclusive evidence without any diagnostic analysis.

### Trivial

- Table 2 presents a "−" for GPT-4V and Claude3 on POPE without noting why scores are unavailable, which is slightly confusing to readers.

---

## Nice-to-Haves

- Multi-seed variance estimates on MME for the full model and ablations (–MFF, –PAL) would substantially strengthen the credibility of the headline result.
- A Cobra-without-pretrain control condition would cleanly separate the effect of training recipe from the effect of PAL/MFF.
- Replacing "reconstructed activation magnitude" in Fig. 1 with a task-agnostic probe (semantic category or spatial relationship accuracy at each layer) would independently validate the visual feature preservation claim.
- Qualitative reconstruction examples from the PAL decoder would let readers assess whether the L2 loss captures meaningful structure versus blurry averages.
- A brief perceptual or SSIM loss comparison against L2 in Eq. 6 would strengthen the claim that structural (not just low-frequency) information is preserved.
- Scaling test on a 7B Mamba backbone, even on a subset of benchmarks, would directly address the paper's stated motivation that Mamba visual processing degrades "in the billion-parameter scale."

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: Comparison with EMU/EMU2 as misleading "scale framing"**: The paper explicitly contextualizes these comparisons ("utilizing nearly one-fourth of its parameters"), and the comparison serves as a reference point for demonstrating that the Mamba-based approach is competitive despite vastly smaller scale. Not a meaningful weakness.

- **Harsh Critic: Section 3.1 — how visual positions are "separated" from textual positions**: In a standard causal LLM, visual tokens occupy known positions in the input sequence; the corresponding output hidden states are naturally addressed. This is standard in all LLaVA-style models and requires no special explanation.

- **Harsh Critic: Demanding Fig. 1 to show MFF at inference clarification**: The paper explicitly states (Section 3.3) that MFF runs only during training. Any ambiguity about what Fig. 1 depicts is a minor presentation note, not a substantive flaw.

- **Strength Finder: "Superior inference speed with competitive performance" as a core strength attributed to EMMA**: Removed per verified finding that EMMA-V1 matches Cobra's speed exactly; EMMA-V2's speed is due to MambaV2, not EMMA-specific design. This remains a valid system-level property but should not be counted as an EMMA contribution.

---

## Novel Insights

The paper's most genuinely novel observation is that applying visual reconstruction supervision *inside* the Mamba LLM (rather than at the encoder or projection stage) reduces hallucinations more effectively than expected given the lightweight nature of the auxiliary loss — with HallusionBench improving by nearly 10 points (41.4 → 51.0) relative to Cobra, outperforming much larger models (LLaVA-1.5 at 47.1). This suggests that Mamba's weakness in visual feature preservation is not a fundamental scaling limitation but a supervision deficit that can be addressed cheaply. The extreme failure of feature-level alignment (+AVF collapsing VQAv2 from 76.25 → 52.8) versus the success of pixel-level alignment is also a non-obvious empirical finding, even if its mechanistic explanation is inadequate in this submission.

---

## Suggestions

1. Replace the autoregressive framing of PAL with an accurate description (auxiliary pixel reconstruction loss) and focus the novelty claim on the *where* (inside the LLM) rather than the *what* (L2 reconstruction).
2. Run 3 seeds for the full EMMA model and both single-component ablations (–MFF, –PAL); report mean ± std on MME. If the super-additive interaction holds across seeds, that demands mechanistic analysis. If it vanishes, the authors need to revise their main result accordingly.
3. Add a Cobra-without-pretrain control to disambiguate training recipe effects from method effects.
4. Revise Fig. 1 caption to note that visualized features are computed using a reconstruction-optimized metric, and either complement with a task-independent probe or clearly caveat the interpretability of the EMMA comparison.

---

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Avg Score | Decision |
|---|---|---|---|
| 0A6f1b66pE (MambaVLM) | Mamba MLLM, Cobra-based | 4.6 | Withdrawn/Reject |
| QPDbIFumQ8 (EMMA-fusion) | Multimodal fusion in LLMs | 5.3 | Reject |
| RLhEGWt94S (modality alignment in MLLMs) | MLLM alignment | 4.5 | Reject |
| cagNCwQEEN (hybrid SSM for MLLM) | Mamba MLLM | 3.4 | Reject |
| Y2RW9EVwhT (Eagle) | Multimodal LLM, vision encoders | 7.2 | Accept (Spotlight) |

**Positioning**: This paper is most directly comparable to MambaVLM (0A6f1b66pE), which also builds on Cobra with incremental improvements and was rejected at ~4.6. EMMA has somewhat more principled contributions (PAL + MFF with a clear ablation study) and stronger individual results (HallusionBench improvement is unambiguous), placing it modestly above MambaVLM. However, the unexplained super-additive MME interaction — the paper's headline number — is a substantive reliability concern that MambaVLM did not face in the same way. The training recipe confound also reduces confidence in the causal claims. The autoregressive framing overclaim makes the paper read as stronger than the actual implementation.

Compared to the anchor cluster of rejected Mamba MLLM papers (3.4–4.6), EMMA belongs in the lower half of the 5-range: the core ideas are sound and ablations partially convincing, but the paper's reliability hinges on an unexplained interaction term with no variance analysis. I place this at **4.5**, consistent with the neighboring anchor cluster.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>