## Summary

This paper introduces Head Relevance Vectors (HRVs), a lightweight mechanistic interpretability tool for text-to-image (T2I) diffusion models that assigns relevance scores to individual cross-attention (CA) heads with respect to human-specified visual concepts. HRVs are constructed by running competitive argmax-based attribution over 2,100 random generations and are validated through an "ordered weakening analysis" (MoRHF vs. LeRHF). The paper then applies HRV-based concept strengthening and concept adjusting to three practical tasks: polysemous-word disambiguation, attribute-based image editing (P2P-HRV), and multi-concept generation (A&E-HRV).

---

## Strengths

- **Head-level interpretability grounded in intervention, not just correlation.** The ordered weakening analysis (MoRHF vs. LeRHF) goes beyond showing attention correlations and provides both qualitative image evidence and quantitative CLIP-score trajectories showing that weakening the most-relevant heads for a concept removes that concept faster than weakening the least-relevant heads. This is a concrete, if imperfect, causal test, and the pattern holds across 34 concepts and is extended to SDXL.

- **No fine-tuning required and model-agnostic construction.** HRVs are constructed entirely from forward passes over random images and are applicable to different model sizes (SD v1.4: 128 heads; SDXL: 1,300 heads) without any parameter updates, making the approach practically accessible and reducing overfitting risk.

- **Strong and diverse empirical gains across three task families.** The improvements are not confined to one application. P2P-HRV achieves Pareto-optimal CLIP + BG-DINO performance on all three object-attribute editing benchmarks and receives 2.39–2.53× more human preference votes over the next-best method for image-attribute editing. A&E-HRV improves consistently over A&E across all six metric–prompt-type combinations. Polysemy misinterpretation drops from 63.0% to 15.9%. These gains span multiple evaluation protocols (automated metrics + human evaluation), reducing the chance of metric gaming.

- **Directly addresses well-known diffusion model failure modes.** The paper targets polysemy misinterpretation, difficult attribute edits (material, geometric patterns, weather, style), and catastrophic neglect—all recognized problems with existing SD-based pipelines—and provides a unified head-level framework that improves on separate task-specific baselines.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No ablation on HRV construction robustness (concept-words and sample count).** HRVs are built using GPT-4o-generated concept-word lists (10 words per concept). There is no experiment showing that (i) the HRVs remain stable if different synonyms or a different LLM are used, (ii) the HRVs converge with far fewer than 2,100 images, or (iii) manually curated word lists yield similar patterns to GPT-4o-generated ones. Because the entire downstream pipeline inherits from these choices, the absence of such ablations makes it impossible to judge whether HRVs reflect genuine model-internal concept specialization or are artifacts of the specific lexical proxy set. This is the most critical missing experiment.

- **No head-level vs. layer-level ablation.** The core claim is that *head-level* granularity is meaningful and useful. However, the paper never compares against a layer-level baseline (e.g., weakening entire CA layers ordered by aggregate relevance, or applying the rescaling vector at the layer level rather than per-head). If operating at the layer level achieves comparable results, the head-level framing loses its main justification. This ablation is essential given the paper's positioning.

- **Heuristic and unjustified intervention design.** Two central design choices lack ablations or theoretical motivation: (1) the ordered weakening multiplier is −2, which flips and magnifies the CA map rather than merely attenuating it — the paper does not test zeroing, soft attenuation, or replacement with baseline activations; (2) the concept adjusting formula uses coefficients 2 and −1 with no sensitivity analysis. Since these choices directly affect all three task results, readers cannot tell how robust the gains are to these hyperparameters.

- **Polysemy evaluation uses only 10 manually chosen cases.** The headline result — misinterpretation dropping from 63.0% to 15.9% — is computed over 10 cases × 10 seeds = 100 total images. This is a small and potentially cherry-picked set. At ICLR, a claim of this magnitude warrants a broader, ideally randomly sampled evaluation. The human evaluation protocol details (number of raters, inter-rater agreement) are deferred entirely to the appendix.

### Minor

- **SDXL generalization claim is only partially supported.** Section 6.1 shows ordered weakening patterns for SDXL and provides supporting results in Appendix G, but does not apply P2P-HRV or A&E-HRV to SDXL. The generalization claim in the abstract and conclusion is therefore only partially substantiated — the analysis transfers, but it is unknown whether the downstream steering benefits transfer.

- **Timestep-invariance claim is too strong for the main-paper evidence.** Section 6.2 states that "generation timesteps do not significantly affect the head relevance patterns" based primarily on a t-SNE visualization. t-SNE is not a reliable tool for establishing the absence of an effect. The paper mentions cosine similarity plots in Appendix I, but the main-paper claim should be hedged or the quantitative evidence brought forward.

- **CLIP-based validation of HRVs is partially circular.** HRVs are constructed using concept-word proxies derived from the CLIP embedding space, and the ordered weakening analysis is then evaluated using CLIP image-text similarity to those same concept-words. The signal is real — image content does change — but the metric is not fully independent of the construction procedure. A complementary evaluation (e.g., human annotation of concept presence during ordered weakening) would break this circularity.

- **No analysis of head-assignment quality (entropy/coverage of HRVs).** The argmax assignment forces every head into exactly one concept per update. The paper does not report what fraction of heads have high-confidence, clean assignments vs. near-uniform relevance distributions. If most heads are diffusely assigned, the "alignment with human concepts" claim applies only to a subset that is never characterized.

### Tiny

- **The notation** in Sections 3.2–3.3 uses *R* for spatial resolution in the CA map and *H* for the number of heads in the query shape, but *H* is also the number of heads globally. The paper should use distinct symbols (e.g., *S* for spatial dimension) to avoid ambiguity.

- **The exact U-Net indexing** of the 128 global head positions across multiple layers and resolutions is not spelled out in the main paper, making independent reimplementation harder than necessary.

---

## Nice-to-Haves

- A prompt-engineering or simple-rephrasing baseline for the polysemy task (e.g., replacing "lavender" with "light purple") would contextualize how much of the gain requires HRVs vs. how much can be trivially recovered by rewording. This would not undermine the contribution but would make the claim more precise.

- An analysis of whether a subset of timesteps (e.g., only early denoising steps) is sufficient for HRV construction, given that Section 6.2 suggests timestep variation is small. This could substantially reduce the computational cost of HRV construction.

- Reporting raw vote percentages and inter-rater agreement statistics alongside the normalized HP-scores in Table 1 would make the human evaluation easier to interpret.

- Extension of at least one downstream task (e.g., image editing) to SDXL to demonstrate that the practical benefits of HRVs also scale to larger models.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"Missing related works" criticisms** (from harsh critic): Per review policy, we do not flag missing related works without access to external sources. Removed entirely.

- **Complaints about missing baselines (Direct Inversion, InfEdit, StyleDiffusion, etc.)** (from spark finder): Per review policy, criticisms claiming a cited or available method was omitted cannot be validated without external sources. The existing comparison set (SDEdit, P2P, PnP, MasaCtrl, FPE) is already reasonably broad. Moved to a soft nice-to-have at most.

- **The -2 weakening "flips sign and is nonlocal, inducing artifacts"** (from harsh critic): This is partially addressed by the paper itself, which notes the weakening is inspired by P2P's rescaling technique and is tested across 34 concepts with consistent MoRHF/LeRHF separation. The concern about the specific choice is kept as a methodological weakness (ablation missing), but the framing that it necessarily induces unrelated artifacts is speculative and removed.

- **Fairness concern about SDEdit being tested at only 0.5 and 0.7 noise strength** (from harsh critic): SDEdit's noise strength is its main hyperparameter; 0.5 and 0.7 bracket the typical useful range. There is no reason to believe additional values would flip the comparison, especially given the Pareto-optimal scatter plots. Removed.

- **Concern that comparison with methods of "unfavorable" baseline tuning benefits the baseline** (from harsh critic re: P2P compatibility): The paper's method is P2P + HRV vs. standalone baselines. This is intentional — adding HRV on top of P2P is itself the contribution. The comparison is not unfair to competing methods. Removed.

- **Broader impact / dual-use discussion absence** (from harsh critic): While a more thorough discussion would be welcome, this is not a substantive weakness for ICLR in the absence of unusual risk. Removed as a standalone weakness; can be a tiny nice-to-have.

- **Claim that the paper's framing as "mechanistic interpretability" is misleading** (from harsh critic): The paper explicitly situates its contribution as a head-level attribution and intervention framework. The interpretability literature has a broad spectrum, and the ordered weakening analysis is a legitimate (if imperfect) intervention-based validation. The framing is defensible; overstating this as a fatal framing error is inappropriate. Removed.

---

## Novel Insights

The most genuinely novel observation across the three reviews is the possibility that the argmax-based competitive assignment — rather than being merely an engineering convenience — may implicitly implement a kind of winner-take-all specialization test: heads are assigned to the concept whose key-projected embedding they most strongly correlate with, under conditions of diverse concurrent competition. Whether this actually produces sparse, semantically pure HRVs (as the downstream results suggest) or merely reflects prompt co-occurrence statistics is a deep open question that the paper raises but does not fully resolve. A related insight is that the timestep-invariance pattern (Section 6.2), if confirmed quantitatively, would suggest that concept-relevant CA heads maintain stable specialization throughout the denoising trajectory — a non-obvious property that has implications for understanding how semantic information flows through the diffusion U-Net over time.

---

## Suggestions

1. **Run the concept-word sensitivity ablation.** Swap GPT-4o-generated words for manually curated ones or for an alternative LLM, and report HRV cosine similarity between the two construction runs. Even a single concept tested across 3–4 different word-list variants would substantially increase confidence in robustness.

2. **Add a layer-level control baseline.** Replicate the ordered weakening and at least one task (e.g., image editing) using layer-level relevance aggregation instead of head-level. This is the single most important ablation for the paper's core claim of head-level granularity.

3. **Ablate the −2 weakening coefficient and the {2, −1} concept-adjusting coefficients.** A simple sweep over {−1, −2, 0 (zeroing), 0.5} for weakening and over a few coefficient pairs for concept adjusting would directly inform both methodological rigor and practical reproducibility.

4. **Expand the polysemy benchmark.** Sample 30–50 polysemous-word prompts semi-randomly (e.g., from a curated list of known CLIP ambiguities) rather than 10 manually chosen cases, and report human agreement statistics alongside the misinterpretation rate.

5. **Bring the cosine similarity timestep analysis from Appendix I into the main paper** (or at minimum replace the strong claim "do not significantly affect" with a hedged statement supported by the quantitative evidence in the appendix).

---

## Paper Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderate-to-high. Head-level attribution vectors for T2I diffusion models constructed via competitive argmax over random generations are not a direct extension of prior work. The connection to ordered intervention (MoRHF/LeRHF) as a validation strategy is also a meaningful contribution. |
| **Importance of research question** | High. Understanding which parts of cross-attention encode which concepts is a foundational question for controllable and trustworthy generation. |
| **Claims well-supported** | Partially. The downstream task improvements are well-evidenced by multiple metrics and human evaluation. The core interpretability claim (heads genuinely align with concepts) is supported directionally but would benefit from ablations on construction robustness and a head-level vs. layer-level comparison. |
| **Soundness of experiments** | Moderate. Baselines are appropriate and comparisons are reasonably fair. Key methodological choices (−2 coefficient, concept-adjusting formula) are unjustified empirically. The polysemy sample size is small. |
| **Clarity of writing** | Good. The paper is accessible and the figures are informative. Some notation ambiguity exists (dual use of H), and key design decisions are insufficiently motivated in the main text. |
| **Value to the research community** | High for practitioners working with SD-family models. HRVs are simple to compute, require no training, and yield meaningful control improvements. The code is released. |
| **Contextualization relative to prior work** | Adequate. The distinction from prior whole-layer attention manipulation is drawn clearly. The paper could more sharply position HRVs relative to attention head importance estimation in the NLP and vision transformer literature. |