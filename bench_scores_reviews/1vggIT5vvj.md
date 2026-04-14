## Summary
This paper introduces Head Relevance Vectors (HRVs), a method for assigning concept-level relevance scores to individual cross-attention heads in text-to-image (T2I) diffusion models. HRVs are constructed by running inference over 2,100 random prompts and accumulating argmax-based "wins" per head, per concept, across all timesteps. The paper validates these vectors via an ordered weakening analysis (MoRHF vs. LeRHF), and applies them to three tasks: disambiguating polysemous words in generation, enhancing Prompt-to-Prompt (P2P) for image editing, and reducing catastrophic neglect in multi-concept generation via Attend-and-Excite (A&E).

---

## Strengths

- **Head-level mechanistic interpretability for T2I diffusion models is genuinely novel.** Prior cross-attention control methods (P2P, PnP, A&E) operate at the layer or token-map level. This paper constructs explicit concept-aligned vectors over individual CA heads — a specific and previously absent granularity for diffusion model interpretability.

- **The ordered weakening analysis (MoRHF vs. LeRHF) is a principled evaluation tool.** Rather than passive correlation, the paper uses an intervention-based test: concepts aligned with an HRV should disappear faster when the most-relevant heads are weakened first. The CLIP-score divergence plots across nine concepts (3 in main text, 6 in appendix) and 33+ qualitative examples provide meaningful, if not exhaustive, causal evidence for the HRV's concept-specificity.

- **Pareto-optimal performance in image editing, including very large human preference margins.** P2P-HRV achieves Pareto-dominance over all five baselines across all three object-attribute editing tasks (Color, Material, Geometric Patterns). For image attributes (Image Style, Weather), it receives 2.39× and 2.53× the votes of the second-best method respectively — not a marginal gain. This is the strongest empirical contribution.

- **Scalability to SDXL (1300 CA heads) is demonstrated.** Showing that the ordered weakening behavior persists in a 10× larger head space, with 45 qualitative examples in the appendix, provides meaningful evidence that the approach is not specific to a single architecture.

- **Training-free, plug-and-play design.** HRV construction requires only inference-time forward passes, and concept steering is applied via simple CA map rescaling with no per-image optimization.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Core HRV construction design choices are unjustified and unablated.** The method uses argmax (not soft weighting), spatial average pooling (not max), and count-based updates (not margin-weighted). None of these choices have theoretical justification or empirical ablation. Why is argmax over average spatial attention the right relevance estimator? Could a single head contribute to multiple concepts in a single update? Are the resulting HRVs sensitive to whether argmax or soft assignments are used? Without any ablation on these fundamental choices, the HRV construction is a plausible heuristic without validated design.

- **The −2 multiplication "weakening" is not simply suppression — it flips sign and amplifies.** The ordered weakening analysis is the primary tool used to validate the interpretability claim for the entire paper. The authors apply this operation to CA maps across all timesteps. However, multiplying by −2 does not just remove a head's contribution; it inverts it and doubles its magnitude, which can destabilize the denoising process in ways unrelated to the targeted concept. The paper gives no ablation over alternative weakening strengths (e.g., 0, 0.5, −1) or alternative zeroing approaches. If the dramatic CLIP score drops under MoRHF are partly due to general denoising destabilization rather than concept-specific erasure, the core interpretability claim is weakened.

- **No comparison to simpler head-selection baselines in any downstream task.** The fundamental claim is that HRV-guided, concept-specific head weighting provides meaningful control. However, none of the three applied tasks compare against uniform scaling of all heads for the target token, random head selection, or top-K binary masking. Without this, it is impossible to determine whether the continuous HRV structure matters or whether any head-level intervention of similar magnitude would produce equivalent results. This is the single most important missing experiment.

- **Polysemy task (Section 5.1) lacks any competitive baseline.** The reported improvement from 63.0% to 15.9% misinterpretation rate is striking, but the only comparison is SD vs. SD-HRV. Obvious alternatives — prompt engineering (e.g., "a vase in the color lavender, not the plant"), CLIP text-encoder disambiguation, or alternative CA control methods — are not tested. Without these, the magnitude of the gain cannot be properly contextualized, and it is unclear whether a lightweight textual fix would achieve similar results.

- **Missing ablation on the steering coefficients (2 and −1) in concept adjusting.** These constants directly drive the downstream results in polysemy and editing tasks. The paper provides no sensitivity analysis or theoretical justification for why 2·HRV_desired − 1·HRV_undesired is the right formulation. This matters especially because the rescaling vector can go negative for some heads, and the behavior in that regime is not discussed.

- **No systematic quantitative summary of the ordered weakening validation across all 34 concepts.** The main text shows 3 concepts; the appendix shows 9. A summary metric — e.g., area between MoRHF and LeRHF CLIP-score curves across all 34 concepts — would strongly substantiate the interpretability claim. As written, the reader cannot tell whether the shown concepts are representative or cherry-picked.

### Minor

- **Concept vocabulary sensitivity is not evaluated.** HRVs depend entirely on GPT-4o's selection of 10 concept-words per concept. The paper shows in Appendix J that new concepts can be added, but does not test whether using different words (or fewer words) for existing concepts would change the HRVs substantially. This is important because HRVs may capture lexical statistics of the concept-word set as much as visual model structure.

- **SDXL downstream tasks are absent.** The scalability claim in Section 6.1 rests only on the ordered weakening analysis. None of the three applied tasks are demonstrated on SDXL. At minimum, one task should be shown on SDXL to substantiate the generalization claim.

- **Head overlap across concepts is not analyzed.** If many of the 34 concepts share the same high-relevance heads, the concept adjusting formulation could produce near-zero or sign-flipped rescaling vectors, undermining the method. A 128×34 HRV heatmap or inter-concept correlation matrix is conspicuously absent and would directly reveal whether heads specialize or are broadly shared.

- **Human evaluation methodology is underspecified in the main text.** Annotator count, inter-annotator agreement, and evaluation instructions are deferred to appendices for both the polysemy and image editing studies. Given that the polysemy result (63%→15.9%) is one of the headline claims, these details should appear in the main text.

- **Computational cost of HRV construction is not reported.** The HRV construction iterates over 2,100 images × 128 heads × 50 timesteps. GPU hours and memory requirements are never reported, making it unclear whether this is practical for new model releases or different hardware settings.

### Tiny

- The notation uses H ambiguously for both the number of CA heads (128) and the spatial height of the latent (e.g., "Q^(h) ∈ R^{H²×F}"), which creates unnecessary confusion when reading the method section.

- The main text defers too many experimental details to appendices, including method specifics for P2P-HRV and A&E-HRV; a self-contained description of these modifications would aid reproducibility.

---

## Nice-to-Haves

- A layer-wise breakdown of HRVs would be informative: are high-relevance heads concentrated in specific U-Net layers or resolution levels, or spread uniformly? Given that SD's U-Net has cross-attention at multiple resolutions, this structural analysis could yield interesting mechanistic insights.

- The t-SNE analysis in Section 6.2 (claiming timesteps do not significantly affect HRV patterns) is only weakly supported. A quantitative analysis — within-concept across-timestep cosine similarity vs. between-concept similarity — would make this finding credible and publishable on its own.

- Downstream experiments on SDXL (at least one task), to fully substantiate the generalization claim.

- A brief discussion of potential misuse (e.g., covert concept steering for content manipulation) would be appropriate for a paper about fine-grained semantic control.

---

## Removed Points
*These points are flagged for removal — treat them with caution.*

- **"Insufficient engagement with attribution/probing literature"** (Reviewer 1): The paper's contribution is primarily empirical and applied; demanding deeper engagement with representation probing or causal mediation literature is scope creep for a paper positioned as a systems/application contribution to T2I control.

- **"Notation/conceptual inconsistency in spatial dimensions"** (Reviewer 1): The critic conflates H (number of heads, 128) with H (spatial height of latent, R). The paper is clear in context that R is the spatial dimension; the apparent inconsistency is a surface-level notation ambiguity, not a methodological error.

- **"Human preference scores normalized to P2P-HRV = 100 hides absolute rates"** (Reviewer 1): This is a formatting/presentation style choice, not a methodological flaw. The relative gaps and vote ratios (2.39×, 2.53×) are clearly reported.

- **"Prior T2I methods manipulate finer granularity than claimed"** (Reviewer 1): This is a related-work positioning dispute. The paper's specific claim is about head-level control as opposed to token-map or layer-level control; no cited prior work constructs concept-aligned head vectors in this way.

- **"Leakage through concept labels in polysemy section"** (Reviewer 1): The method intentionally requires identifying desired/undesired concepts, which is part of the user-control design. Calling this "leakage" mischaracterizes the intended use case.

- **"HRVs must be reconstructed for each model version, limiting universality"** (Reviewer 2): The paper never claims cross-model HRV transfer. Re-constructing HRVs for a new model is a one-time cost and is not presented as a limitation by the authors. Criticizing a scope not claimed is not a valid weakness.

- **"Comparison to nonce-word substitutions to distinguish concept grounding from lexical memorization"** (Reviewer 1): This is a sophisticated mechanistic interpretability control that goes well beyond the paper's stated scope of practical utility. Moving to nice-to-have would be appropriate, but this is a deep theoretical validation the paper does not claim to provide.

---

## Novel Insights

The most genuinely novel conceptual insight surfaced by synthesizing the three reviews is the **head overlap problem** identified by Spark Finder: if the 34 concepts share substantial head overlap (i.e., many concepts have similarly high relevance scores for the same heads), then the concept adjusting formula (2·HRV_desired − 1·HRV_undesired) could systematically produce vectors with many near-zero or negative entries, potentially degrading rather than steering generation. The paper never analyzes inter-concept head overlap, which is both a significant methodological gap and an interesting scientific question — if heads are highly specialized per concept, that is strong evidence for the mechanistic claim; if they are highly shared, the steering mechanism's success would need a different explanation. A 128×34 HRV heatmap could resolve this and would constitute a substantive addition to the mechanistic interpretability literature.

---

## Suggestions

1. **Add uniform-head-scaling and random-head-selection baselines to all three downstream tasks.** This is the highest-priority experiment — it directly validates whether HRV structure is necessary or whether any head-level intervention of similar magnitude produces the same effect.

2. **Ablate the −2 weakening multiplier** with values in {0, 0.5, 1.0, −1.0, −2.0} and show whether MoRHF/LeRHF divergence is specific to sign-flipping or holds for simpler suppression (multiplying by 0 or 0.5). If the result holds under zeroing, the interpretability validation is substantially more convincing.

3. **Add at least one competitive baseline to the polysemy experiment** — e.g., prompt engineering by appending an explicit concept clarifier. The current setup only demonstrates SD-HRV > SD, which leaves open whether any prompt modification would achieve the same result.

4. **Report a summary metric for ordered weakening across all 34 concepts** — e.g., area under the MoRHF curve divided by area under the LeRHF curve — rather than displaying only 9 of 34 CLIP-score plots.

5. **Include a 128×34 HRV heatmap** showing concept–head relevance structure. This would reveal whether heads specialize per concept (supporting the mechanistic claim) or are broadly shared (requiring alternative explanation).

6. **Conduct a sensitivity analysis on the concept-word vocabulary**: test whether using different GPT-4o–generated words or varying the number of words per concept (e.g., 5 vs. 10 vs. 20) significantly changes downstream HRV quality.

7. **Report GPU hours for HRV construction** and test whether a smaller image set (e.g., 500) produces HRVs of comparable quality, to assess practical adoption cost for new model versions.

---

**Evaluation summary:**

- *Novelty*: Moderate-to-strong. HRVs as concept-aligned head-level interpretability objects for T2I diffusion are a novel contribution; the downstream applications build on well-established baselines (P2P, A&E) in a principled way.
- *Technical soundness*: Moderate. The HRV construction and the validation via ordered weakening both contain unjustified design choices that substantially weaken the mechanistic claims. The method works empirically but is under-analyzed.
- *Empirical support*: Moderate-to-strong for image editing (Pareto-optimal, large human preference margins); weaker for polysemy (no competitive baseline) and multi-concept generation (modest 2–6% gains).
- *Significance*: Moderate-to-strong. Bridging interpretability and controllability at the head level is a meaningful contribution to both communities.
- *Clarity*: Generally clear at a high level; notation ambiguities and heavy reliance on appendices for key experimental details reduce reproducibility from the main text alone.