Now let me do calibration searches to anchor my score.Now I have sufficient information to write the final review. Let me synthesize everything.---

## Summary
EMMA proposes two training-time modules for Mamba-based multi-modal large language models (MLLMs): (1) a pixel-wise alignment loss (PAL) that uses an MSE-based image reconstruction objective as auxiliary supervision on visual features inside the Mamba LLM, and (2) a multi-scale feature fusion (MFF) module that hierarchically combines intermediate Mamba LLM layer features before decoding them to pixel space. Both modules incur zero inference overhead. The system outperforms the Cobra baseline on all evaluated benchmarks, with a notable 9.6-point HallusionBench improvement attributed to PAL, and a 278-point MME improvement attributed to MFF.

---

## Strengths

- **Clear empirical gains with well-structured ablation**: Table 4 isolates each component's contribution cleanly. Removing PAL drops HallusionBench from 51.0 to 41.4 and TextVQA from 57.2 to 52.4; removing MFF collapses MME from 1572.8 to 1294.1. Both numbers are reproducible (–PAL matches the reproduced Cobra numbers exactly, providing an internal consistency check).

- **Practical zero-overhead design**: Section 3.3 explicitly confirms that the feature fusion and visual decoding stages are training-only, imposing no inference cost. This is a significant engineering virtue — performance gains at no deployment cost.

- **Meaningful hallucination reduction**: The HallusionBench gain of 9.6 points (41.4 → 51.0) over the Cobra baseline is substantial for a method that merely adds a reconstruction auxiliary loss without changing the inference graph. POPE also improves (87.2 → 88.0), and EMMA-V1 surpasses LLaVA-1.5 and BLIP2-T5 (7B and 12.4B models respectively) on HallusionBench (Table 2), which is a pointed and honest comparison.

- **Pixel-vs-feature ablation is informative**: The +AVF experiment reveals that aligning processed visual features rather than raw pixels causes catastrophic collapse across most metrics, while PAL (pixel-level) works. This is a concrete, actionable finding for the Mamba MLLM community.

---

## Weaknesses

### Fatal
None.

### Major

- **Central diagnostic claim rests on a single, methodologically vague example.** The entire paper motivation — that Mamba LLMs progressively destroy fine-grained visual features across layers — is substantiated by exactly one pizza image in Figure 1. The visualization is described as "magnitude of reconstructed spatial activations on each image," but no details are given about how these reconstructions are computed (PCA components? patch reconstruction values? magnitude of hidden states?), preventing reproduction. More critically, there is no quantitative measurement of feature degradation across layers on a population of images. The downstream task improvements are consistent with the claim, but do not constitute mechanistic evidence. A proper layer-wise feature quality analysis across even 10–20 images with a quantitative metric (e.g., reconstruction quality or feature-image correspondence) is needed to elevate this from anecdote to evidence.

- **MFF's effect is almost entirely concentrated in a single benchmark (MME) without explanation.** Table 4 shows: removing MFF drops MME by 278 points but changes all other metrics by at most 1.5% absolute (VQAv2: 76.25→75.9, GQA: 60.5→59.3, POPE: 88.0→87.1, HallusionBench: 51.0→50.7). If MFF genuinely preserves fine-grained visual information broadly, uniform gains should appear across TextVQA and other detail-sensitive benchmarks. The selective gain on MME specifically suggests MFF may be exploiting something particular to MME's evaluation protocol, and the paper offers no explanation for this pattern. This undermines the generality of the "hierarchical alignment" claim.

- **Abstract's latency claim is misleading for the primary model variant.** The abstract states "Our model shows lower latency than other Mamba-based MLLMs," but Table 3 shows EMMA-V1 (MambaV1-2.8B) runs at exactly 138.95 tokens/second — identical to Cobra (138.95). The latency advantage over Cobra belongs entirely to EMMA-V2 (MambaV2-2.7B, 149.96 tok/s), which is attributable to the MambaV2 backbone architecture, not to EMMA's PAL or MFF contributions. The paper does correctly note in Section 4.3 that "Our model achieves even better runtime than Cobra due to more efficient processing in the MambaV2 LLM backbone," but the abstract-level framing remains inaccurate for the primary (V1) variant. This is not a contribution of the paper's methods.

### Minor

- **L2/MSE reconstruction loss is theoretically misaligned with the stated goal of fine-grained structural preservation.** The paper claims PAL enforces "sensitivity to visual details" and "preservation of structural features," and Section 3.2 describes L2 as focusing on "overall similarity that helps in preserving structures and shapes." MSE is well known to reward blurry, globally averaged outputs and to be insensitive to high-frequency spatial detail. The empirical improvement on TextVQA (requires text reading in images) suggests PAL provides some useful signal, but the paper never explains *why* MSE works here or compares it against a perceptual loss (e.g., LPIPS, VGG feature matching), which would directly test whether the gain comes from any visual supervision vs. structure-preserving reconstruction specifically.

- **The +AVF collapse is under-analyzed.** The catastrophic failure of feature-level alignment (VQAv2: 76.25→52.8, MME: 1572.8→984.8) is explained in one sentence attributing it to "robust structural information inherent in pixel-level images." This does not address whether the collapse is a training instability (e.g., gradient scale mismatch), a loss-weighting issue, or a fundamental incompatibility. Without this analysis, the pixel-vs-feature finding, while empirically real, is not fully understood.

- **Pretraining status of the Cobra baseline deserves more explicit disclosure.** The –PAL ablation row in Table 4 is numerically identical to the Cobra row in Table 1 (all values match to reported precision), confirming the reproduced Cobra also skips pretraining. The paper states "discarding the pretrain phase" for EMMA but does not explicitly say this also applies to its Cobra reproduction, leaving the reader to infer it. This should be stated explicitly to confirm baseline comparability.

### Trivial

- The formulation of Eq. (5) uses the autoregressive probability factorization $p(\hat{X}_v \mid X_v, X_t) = \prod_i p_\phi(\hat{x}_{v,i} \mid \{X_{v,j} \mid j < i\}, X_t)$, implying sequential visual token *generation*. What actually occurs is extraction of LLM hidden states at visual token positions followed by MFF and a small decoder — not sequential generation in the conventional sense. The notation is borrowed from text generation and is imprecise for what the model does. This is a presentation issue, not a fundamental flaw, but it should be clarified.

---

## Nice-to-Haves

- A quantitative layer-wise feature quality analysis on a population of images (e.g., nearest-neighbor retrieval accuracy or patch reconstruction MSE per layer) would transform the core motivation from a single illustration into a properly supported empirical finding.
- A comparison of MSE-based PAL against a perceptual loss (LPIPS, VGG features) would clarify whether fine-grained structural gains require structure-specific supervision or whether any reconstruction objective suffices.
- Testing whether PAL generalizes to transformer-based MLLMs (e.g., a LLaVA backbone) would clarify whether the benefit is Mamba-specific (consistent with the stated diagnosis) or general.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **Harsh Critic — Introduction framing of transformer context window limits**: The critic argues that "transformers struggle with long-ranged dependencies due to limited context windows" is outdated. While modern transformers do support long contexts, the cited motivation is still a valid general positioning statement and does not affect the paper's contributions. Removed as scope nitpick.

- **Harsh Critic — Eq. (2) formalism misrepresents computational flow**: The critic argues `f_φ(X_LLM, ψ)` suggests MFF is an argument during the forward pass, whereas it extracts intermediate features. This is a minor notational imprecision and does not affect reproducibility. Retained as Trivial.

- **Harsh Critic — Claim that Mamba MLLMs "degrade at large scale" is imprecisely characterized**: At 2.7-2.8B parameters, EMMA operates in a regime the paper explicitly targets. The scale characterization issue is presentation-level and does not invalidate the paper's specific experimental setting. Removed.

- **Strength Finder — Near 4× inference speed advantage is a core strength**: The speed advantage over transformers is entirely a property of the Mamba backbone and is not attributable to EMMA's innovations. Removed as a strength claim. (Retained as a context note: Mamba is fast; EMMA does not slow it down.)

- **Strength Finder — Pixel-level alignment shown superior to feature-level**: The +AVF ablation is real and informative. Retained but downgraded to supporting evidence only, since the explanation for the failure is thin.

---

## Novel Insights

EMMA's most transferable insight is that auxiliary pixel-reconstruction supervision — applied inside the Mamba LLM via a training-only decoder, with no inference overhead — significantly reduces visual hallucination (9.6-point HallusionBench gain) in a setting where text-only supervision is the norm. This is consistent with concurrent work such as ROSS on transformer LMMs, and lends weight to the emerging finding that visual self-supervision inside the LLM backbone is a generally effective strategy. The MFF finding — that hierarchical intermediate feature fusion primarily benefits coarse-grained holistic evaluations (MME) rather than fine-grained benchmarks — is an unexpected empirical observation that, if properly explained, could inform future design choices about where multi-scale feature aggregation yields returns.

---

## Suggestions

1. Replace the single pizza visualization with a quantitative layer-wise degradation plot averaged over ≥10 images (e.g., mean squared reconstruction error per layer between visual token features and input image patches, comparing Cobra vs. EMMA).
2. Ablate the PAL loss function: run EMMA with an LPIPS or VGG perceptual loss in place of L2 to determine whether structure-specific supervision outperforms global pixel matching.
3. Correct or qualify the abstract's latency claim to specify that lower latency compared to Cobra is achieved only via the MambaV2 backbone in EMMA-V2.
4. Explicitly state that the Cobra baseline in ablations (Table 4) discards the pretrain phase, matching EMMA's training recipe.
5. Provide a mechanistic analysis or ablation of why MFF's gains are concentrated in MME — at minimum, report MME sub-score breakdowns (Perception vs. Cognition sub-tasks) for EMMA vs. –MFF.

---

## Score and Decision

**Calibration Anchors:**

| Paper | Similarity | Human Scores | Avg | Decision |
|---|---|---|---|---|
| ROSS (8q9NOMzRDg) | Pixel reconstruction auxiliary supervision for LMMs (transformer) | 5,6,6,6,6 | 5.8 | Accept Poster |
| VLSA (RLhEGWt94S) | Modality alignment + visual reconstruction in MLLMs | 5,3,5,5 | 4.5 | Reject |
| MambaVLM (0A6f1b66pE) | Mamba-based MLLM, similar scope | 6,3,3,5,6 | 4.6 | Reject |

EMMA is most comparable to ROSS in terms of its core idea (auxiliary visual reconstruction supervision for multimodal models). ROSS was accepted at average 5.8. However, EMMA falls below ROSS on several dimensions: (1) ROSS uses a more principled denoising reconstruction in latent space rather than raw MSE; (2) ROSS applies to the broader transformer-LMM setting and demonstrates cross-architecture generality; (3) EMMA's central motivation is supported by a single visualization, while ROSS provides more systematic evidence. On the other hand, EMMA has a more focused target (Mamba-specific visual feature degradation) and clean ablations demonstrating both PAL and MFF contribute independently.

EMMA is stronger than the rejected MambaVLM and VLSA papers: the empirical gains are real and consistent, ablations are clean, and the paper does not overclaim. But it sits below ROSS in methodological rigor.

**Positioning:** ~5.0 — below the ROSS acceptance threshold, above the VLSA/MambaVLM rejection cluster.

**Evaluation on key axes:**
- *Originality*: Moderate — the combination of Mamba + pixel reconstruction supervision is novel in this setting, but the individual ideas are not new.
- *Importance*: Moderate — Mamba MLLMs are a live and active research direction, so solving visual feature degradation therein is meaningful.
- *Claims vs. support*: Weak — the central diagnostic claim lacks quantitative cross-image evidence; the latency claim for the primary variant is misleading.
- *Soundness*: Adequate — empirical results are consistent, ablations are clean, no fundamental methodological errors.
- *Clarity*: Adequate — readable, but has notation imprecision and thin explanations for key phenomena (MFF's selective MME impact, AVF collapse).
- *Value to community*: Moderate — provides a practical and reproducible approach for Mamba MLLM practitioners.

**Final Score: 5.0 — Weak Reject.** The paper makes real contributions but the core motivation lacks rigorous evidential support, the primary efficiency claim in the abstract is misleading, and the selective impact of MFF on a single benchmark is unexplained. These issues should be addressable in a revision, but in the current form the paper does not meet the bar for acceptance.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>