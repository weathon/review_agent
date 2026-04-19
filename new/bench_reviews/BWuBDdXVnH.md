Now I have enough information to write the final review. Let me synthesize everything.---

## Summary
ControlAR introduces *conditional decoding* as a mechanism to add spatial conditioning (edges, depth maps, segmentation masks) to next-token-prediction autoregressive image generation models (LlamaGen, AiM). Rather than prefilling control tokens before image generation (which inflates sequence length and degrades quality), control tokens are added elementwise to image tokens at selected intermediate layers. The paper demonstrates competitive performance with ControlNet++-based diffusion systems across multiple control types and introduces a multi-resolution extension that requires no additional parameters.

---

## Strengths

- **Conditional decoding clearly outperforms prefilling in both quality and efficiency (Figure 2c/d):** Conditional decoding achieves better FID convergence and reduces GPU memory by ~37% (21.0 GB vs. 33.4 GB) and training time by ~49% (27 min vs. 53 min/epoch), directly validating the paper's core design choice.

- **Strong FID results vs. ControlNet++ (Table 3):** ControlAR outperforms ControlNet++ on all six FID metrics—including segmentation (27.15 vs. 29.49 on ADE20K), hed edge (10.53 vs. 15.01), and depth (14.61 vs. 16.66)—demonstrating that the approach yields higher image quality alongside its efficiency gains.

- **Architecture-agnostic framework (Table 1):** ControlAR works with both Transformer-based (LlamaGen-B and LlamaGen-L) and Mamba-based (AiM-L) backbones, confirming the method generalizes beyond a specific sequence model family.

- **Arbitrary-resolution generation emerges naturally (Figure 6b):** Because the count of decoded image tokens matches the control token count, multi-resolution generation requires no extra modules. MR-ControlAR maintains SSIM ~85.5 across aspect ratios from 1:1 to 2:1, while standard ControlAR degrades to ~80.5 at 2:1.

- **Well-executed ablation on control encoder design (Table 4):** The analysis across CNN, ViT-S, DINOv2-S, and DINOv2-B establishes that pre-training domain alignment (ImageNet-supervised ViT for C2I; DINOv2 for T2I) explains the performance differences—a practically useful and grounded finding.

---

## Weaknesses

### Fatal
None.

### Major

- **Abstract overclaims vs. actual results.** The abstract states ControlAR "surpasses previous state-of-the-art controllable diffusion models, e.g., ControlNet++." Verified from the paper's own tables: on conditional consistency (Table 2), ControlNet++ outperforms ControlAR on ADE20K mIoU (43.64 vs. 39.95), lineart SSIM (83.99 vs. 79.22), and depth RMSE (28.32 vs. 29.01). ControlAR wins on COCOStuff mIoU, hed SSIM, and ties on canny F1. On FID (Table 3), ControlAR does win all six metrics vs. ControlNet++. The body text correctly qualifies this as "comparable or even better" (§4.2), but the abstract's "surpasses" conflicts with the mixed conditional consistency results. This is not a subtle miscalibration—it is a factual misstatement in the highest-visibility part of the paper that will be caught immediately by readers comparing the abstract to the tables.

- **C2I comparison against ControlVAR rests on an estimated value at a narrow margin.** Table 1 explicitly footnotes that ControlVAR's FID values are "estimated from its histograms." The headline C2I win at the smallest model (LlamaGen-L 343M vs. ControlVAR-d30 2B) on canny FID is 7.69 vs. 7.85—a 0.16-point margin on a graphically read number. The depth comparison is more convincing (4.19 vs. 6.50), and the paper's overall efficiency argument (17% of ControlVAR-d30's parameters) is compelling even without the canny win. Nevertheless, presenting the 0.16 canny margin as a clear victory over a 2B-parameter model without direct evaluation under identical conditions is not defensible. The paper should either run ControlVAR directly or more prominently flag the uncertainty.

### Minor

- **Cross-paradigm comparison conflates control mechanism with base model quality.** ControlAR is built on LlamaGen-XL (775M, trained on LAION-Aesthetics with T5 text encoder), while all diffusion baselines use SD1.5. The FID improvements in Table 3 may partly reflect LlamaGen-XL's base generation quality rather than the control mechanism per se. This does not invalidate the paper's contribution—the paper presents a practical framework and the FID wins are real—but the claim that conditional decoding is the source of the FID improvements is not cleanly isolated.

- **Ablations conducted on a different model than main results.** Tables 4–6 use LlamaGen-B (111M, C2I on ImageNet 256×256). All headline T2I results use LlamaGen-XL (775M). Whether the optimal encoder and fusion strategy choices transfer across a ~7× parameter increase and a different conditioning modality is assumed, not demonstrated.

- **Which layers are used for LlamaGen-XL (36 layers) is not stated.** The paper specifies layers 1, 5, 9 for LlamaGen-B (12 layers) but does not state which three layers are used for LlamaGen-XL. Since all T2I headline results use LlamaGen-XL, this is a reproducibility gap.

- **MR-ControlAR evaluation is thin.** The only quantitative evaluation for the multi-resolution variant is SSIM on a single control type (hed edge) in Figure 6b. No FID comparison between ControlAR and MR-ControlAR at non-square resolutions is provided, and no comparison with other resolution-flexible methods is included.

### Trivial

- **Figure 2(c) description is confusing.** The parsed alt text describes the F1-Score curve for conditional decoding as *decreasing* from ~35 to ~18, while the caption claims decoding outperforms prefilling. This appears to be a figure description artifact (possibly mislabeled axes or curves in the original figure or in PDF parsing), but it should be clarified to prevent reader confusion.

- **No failure cases shown.** Figure 5 shows cherry-picked successes and uses red boxes to highlight competitor failures. Given that ControlAR loses to ControlNet++ on lineart SSIM and ADE20K mIoU, representative failure cases for ControlAR on these tasks would give a more honest picture.

---

## Nice-to-Haves
- An ablation comparing offset=0 (aligning $C_t$ with $I_t$) vs. the offset=1 scheme in Eq. 4 would directly test whether the specific one-position-ahead alignment is load-bearing or incidental.
- An analysis of *why* dense control injection at all 12 layers increases FID (11.75 vs. 10.64 at 3 layers, Table 5) despite marginally improving F1-Score (34.21 vs. 34.15) would strengthen the theoretical grounding of the design.
- FID comparison for MR-ControlAR vs. ControlAR at multiple resolutions would substantiate the quality claim for arbitrary-resolution generation.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh critic concern about Figure 2(c) being a "major failure."** The figure description (from the PDF parser's auto-generated alt text) shows conditional decoding F1 decreasing, which the critic treats as evidence the model is failing. However, the paper caption explicitly states conditional decoding outperforms prefilling, and the final numbers (FID ~10 for decoding vs. ~35 for prefilling) confirm this. This is almost certainly a parser/alt-text artifact, not a real figure error. Retained only as a trivial note about figure clarity.

- **Diffusion comparison is "unfair" because base models differ.** The harsh critic argues this invalidates the T2I FID comparisons. Moved to Minor with softened framing because cross-paradigm comparison is standard in this field, the paper does not claim to isolate the control mechanism from the base model, and the practical contribution (a working controllable AR framework competitive with diffusion) is the actual claim.

- **Generic request for theoretical analysis.** The harsh critic asks why over-injection hurts FID. This is a valid empirical observation but the paper makes no theoretical claims; requesting a formal analysis goes beyond community standards for a systems paper. Retained as a Nice-to-Have.

- **Missing failure-case visualizations.** Moved to Trivial since their absence doesn't invalidate any quantitative claim.

---

## Novel Insights
The most practically underappreciated finding is that simple elementwise addition—not cross-attention—is the superior fusion strategy for injecting spatial control into AR models (Table 5: addition achieves 34.01/11.02 F1/FID vs. cross-attention's 30.86/15.34). The paper's explanation—that cross-attention must learn positional correspondences the model doesn't already know, slowing convergence—is plausible and if confirmed more rigorously, would have broader implications for how spatial structure is incorporated into sequential generative models. The emergent multi-resolution capability is also genuinely elegant: the paper doesn't design for it explicitly, but conditional decoding makes the number of decoded tokens a free parameter that tracks the control sequence length, giving arbitrary resolution essentially for free.

---

## Suggestions
1. **Rewrite the abstract** to replace "surpasses" with "achieves performance competitive with or superior to" ControlNet++, consistent with the body text's accurate framing.
2. **Report ControlVAR canny FID under direct evaluation conditions** (or explicitly bound the uncertainty), and re-frame the efficiency argument (17% of parameters, competitive FID on depth) as the primary C2I headline rather than a marginal canny win.
3. **State explicitly which layers are used for LlamaGen-XL** in the implementation details section.
4. **Add at least one FID metric for MR-ControlAR** at non-standard resolutions to make the multi-resolution claim quantitatively grounded.

---

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Human avg. score | Decision |
|---|---|---|---|
| HART (q5sOv4xQe4) | Hybrid AR transformer, 1024px generation | 6.8 | Accept Poster |
| Show-o (o6Ynz6OIQ6) | Unified AR+diffusion transformer | 6.5 | Accept Poster |
| SlotAdapt (kZvor5aaz7) | Adapter-conditioning for diffusion | 6.25 | Accept Poster |
| NOVA (JE9tCwe3lp) | AR video generation | 6.5 | Accept Poster |

ControlAR is positioned similarly to these papers: it fills a real gap in a competitive subfield, the core method is clean and well-motivated, results are generally competitive, but the contribution is narrower than HART (which achieved a more fundamental advance in AR generation quality) and has the abstract overclaim issue and thin ControlVAR evaluation that keep it from a clear 7. The paper is most comparable to Show-o and NOVA, which were accepted at 6.5. The abstract overclaim and marginal ControlVAR comparison are real but correctable issues that don't undermine the methodology. The overall evidence supports a borderline-accept judgment.

**Axis evaluation:**
- *Originality*: Moderate — conditional decoding is a clean idea and the AR controllable generation framing is genuinely novel in this specific form, but the mechanism itself (elementwise token fusion) is not deeply surprising.
- *Importance of research question*: High — bridging controllable generation from diffusion to AR models is timely and consequential.
- *Claims supported by evidence*: Mostly yes, with one overclaim in the abstract.
- *Soundness of experiments*: Good, with the caveat of the ControlVAR estimation and cross-paradigm comparison.
- *Clarity*: Good, with the figure 2(c) confusion as a minor exception.
- *Value to research community*: High — the method is practical, efficient, and likely to be widely adopted.

**Final score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>