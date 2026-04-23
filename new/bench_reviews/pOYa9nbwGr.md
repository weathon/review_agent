Now I have all the information needed to write the final review. Let me compile my analysis.

## Summary

RTD proposes a post-processing approach for projection-based zero-shot composed image retrieval (ZS-CIR) that updates the CLIP text encoder using cheap text triplets and target-anchored contrastive learning, aiming to reduce the "task discrepancy" between CLIP's image-text alignment pre-training and the composed retrieval task. The method freezes the target caption embeddings from the original CLIP text encoder as anchors while training the learnable text encoder to align concatenated caption embeddings with these anchors, requiring only text data and adding ~0.5 hours of training on 8 A100 GPUs.

## Strengths

- **Consistent, significant empirical improvements across all tested configurations.** Tables 2–4 show RTD improves every baseline (Pic2Word, SEARLE, LinCIR) on every dataset (FashionIQ, CIRR, CIRCO) across ViT-B/32 and ViT-L/14 backbones, with gains ranging from +1.5 to +10.8 average points. This monotonic improvement across 6 base configurations is rare and convincing.

- **Practical efficiency with concrete measurements.** Section 3.2 quantifies the advantages: rule-based triplet generation is 570× faster than CIR triplet generation; text triplets require 100MB vs. ~400GB for equivalent image data; total RTD training adds only 0.5 hours on 8 A100s for ViT-L/14, comparable to LinCIR's original training time. Pre-extracted visual embeddings are preserved, avoiding retrieval database updates.

- **Rule-based triplet generation achieves competitive performance with LLM-generated triplets (Table 7).** Rule-based triplets yield +2.85 avg improvement, close to CompoDiff (+3.34) and in-context learning (+3.53), significantly lowering the barrier to adoption and improving reproducibility.

- **The anchoring mechanism is well-motivated and its necessity is clearly demonstrated.** Table 6 row 5 shows removing the anchor (using the learnable encoder for both composed and target captions) drops average performance from 39.64 to 36.25, even below the baseline of 37.38, confirming that naive fine-tuning without anchoring actively harms performance.

- **Comprehensive ablation study (Table 6).** Each component (TCL loss, refined batch sampling, refined concatenation, anchoring) is shown to contribute meaningfully, with anchoring being the most critical.

## Weaknesses

### Fatal
None.

### Major

- **Table 8's "naïve tuning" baseline is a strawman that does not fully support the conceptual claim about "task discrepancy."** The paper's central explanatory claim is that RTD works by specifically reducing task discrepancy, not merely by updating the text encoder. Table 8 provides the key evidence: "naïve tuning" (updating the text encoder with the original projection-method loss) severely degrades Pic2Word (Avg 32.15→10.86) and slightly degrades LinCIR (36.86→35.52). However, these original losses were designed to train φ, not ψ_T — applying them to the text encoder is architecturally inappropriate, making catastrophic degradation unsurprising. Table 6 row 5 partially compensates by showing that even the TCL loss without anchoring drops below baseline (36.25 vs 37.38), demonstrating that careful design matters. But the gap remains: there is no control using a *reasonable* alternative fine-tuning approach (e.g., standard text contrastive learning on (T_{r+c}, T_t) pairs with a single encoder and regularization, or LoRA-based fine-tuning). Without this, the paper cannot fully separate whether RTD's gains come from specifically addressing "task discrepancy" versus being a well-designed fine-tuning recipe. This matters because if any reasonable text-encoder fine-tuning on relevant data would suffice, the conceptual framing around "task discrepancy" is substantially overclaimed.

- **No evaluation of whether the updated text encoder degrades general CLIP capabilities.** RTD replaces the frozen CLIP text encoder at inference. The anchoring mechanism is designed to preserve original alignment, but the paper provides no direct evidence that it does. Section 3.3 shows composed-query similarity to target images increases (0.10→0.29), but this is expected since RTD was explicitly trained to optimize this. Missing is any measurement on standard text-to-image retrieval or zero-shot classification (e.g., ImageNet, COCO retrieval) to verify that the updated encoder maintains general CLIP alignment. If the updated encoder trades some general alignment for CIR-specific performance, users should know. A single zero-shot classification or text-to-image retrieval experiment would suffice.

### Minor

- **The self-contrastive terms in Eq. 1 may push semantically similar embeddings apart.** The symmetric InfoNCE loss includes terms like ∑ exp(c(t_i^k, t_i^j)) that push different target embeddings apart from each other, even when they may be semantically similar (e.g., multiple captions describing dogs in the same batch). The paper does not discuss this potential issue.

- **The refined concatenation scheme introduces a training-inference distribution mismatch handled heuristically.** During training, φ(t_r) with Gaussian noise (σ=0.5, manually tuned) replaces φ(ψ_V(I_r)) used at inference. The paper acknowledges this follows Gu et al. (2024), but does not analyze the magnitude of this mismatch. The ablation (Table 6, rows 4 vs 6) shows RC contributes only ~1.1 avg points, so this is a modest concern, but the RC component is underspecified relative to its description.

### Trivial
None.

## Nice-to-Haves

- Test whether RTD can be combined with non-projection methods (e.g., CoVR, CASE) to further demonstrate generality.
- Add a "reasonable but simpler" fine-tuning control (e.g., LoRA on text encoder with standard contrastive loss on text pairs) to better isolate the contribution of the "task discrepancy" framing.
- Analyze per-dataset gain variance to understand whether RTD's improvement correlates with dataset characteristics, which would strengthen the "task discrepancy" narrative.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Critic's claim that the "task discrepancy" term inflates the conceptual contribution.** While the evidence for the specific mechanism is incomplete (see Major weakness above), the framing itself is reasonable — the text encoder is indeed being asked to process concatenated captions with modification instructions at inference, which differs from its pre-training. The issue is insufficient experimental support, not the framing itself.

- **Critic's concern about model selection on CIRR dev then reporting CIRR test.** This is standard practice in the field and not a meaningful weakness; all baselines use the same evaluation protocol.

- **Critic's concern about non-apples-to-apples comparison in Table 5.** The paper explicitly acknowledges this: "Note that this comparison is not entirely fair due to differences in backbone models and training data across categories."

- **Strength Finder's claim that Table 8 "provides decisive evidence" that gains stem from task discrepancy reduction.** Table 8 uses a strawman baseline; the evidence is suggestive but not decisive. This strength is weakened to reflect the actual level of support.

- **Critic's concern about captioning noise in the toy experiment (Table 1).** The improvement from 10.12 to 15.12 mAP@5 is meaningful regardless of caption noise, and the gap to the ideal caption (18.96) is itself informative. This is not a weakness.

## Novel Insights

The finding that rule-based text triplets achieve near-competitive performance with LLM-generated triplets for CIR-specific fine-tuning is practically significant and somewhat surprising — it suggests that the text encoder update benefits primarily from exposure to the *structure* of concatenated captions (reference + modification → target) rather than from semantically rich or diverse modifications. This contrasts with the general trend in VL research where LLM-generated data is assumed to be superior.

## Suggestions

- Add a single zero-shot classification experiment (e.g., ImageNet top-1 accuracy) with the updated text encoder to confirm that general CLIP alignment is preserved — this would significantly strengthen the paper.
- Add a reasonable fine-tuning control (e.g., standard text contrastive learning on (T_{r+c}, T_t) pairs with weight decay or LoRA regularization) to isolate whether RTD's specific design choices are necessary or whether any principled fine-tuning approach would suffice.

## Calibration

**Anchors compared:**
- **ISA (5BXAXOpaWu)**, ZS-CIR with adaptive token learner, avg score 7.50: Same task domain, but ISA has cleaner conceptual contribution. RTD has broader empirical validation across more baselines but weaker conceptual support.
- **TCR (BmG88rONaU)**, plug-in module for cross-modal retrieval, avg score 7.50: Similar plug-in design philosophy with consistent improvements. RTD is comparable in empirical breadth.
- **Data-free KD from CLIP (1aF2D2CPHi)**, avg score 8.00: Stronger conceptual contribution with comprehensive experiments. RTD is below this level.
- **ColCLIP (7F4ioiKQFT)**, fine-tuning CLIP for retrieval, avg score 4.00: Much weaker — limited novelty, unclear problem definition. RTD is clearly above.
- **Semi-supervised CLIP adaptation (97D725GJt)**, avg score 5.80: Addresses domain gap with moderate results. RTD has stronger and more consistent empirical gains.
- **Harry Potter OOD (3ZdGSTxKuy)**, avg score 2.00: Fundamentally flawed experiments and overclaimed contribution. RTD's strawman issue is much less severe.
- **End-to-end RL (eM5dar35Ys)**, avg score 2.60: Strawman baselines and overclaimed novelty. RTD has real, consistent improvements across proper baselines.

RTD sits above the medium-scoring anchors (ColCLIP at 4.00, semi-supervised CLIP at 5.80) due to its strong, consistent empirical results and practical advantages, but below the high-scoring anchors (ISA at 7.50, TCR at 7.50) due to the strawman control in Table 8 and missing general CLIP capability evaluation. The paper is a solid contribution with notable but not fatal gaps.

## Score and Decision

**Originality:** Moderate — the "task discrepancy" framing is intuitive but the experimental support is incomplete; the method itself is well-designed but builds incrementally on existing contrastive learning and projection-based approaches.

**Importance of research question:** High — ZS-CIR is an active and practically important research direction, and improving projection-based methods with minimal overhead addresses a real need.

**Claims well supported:** Mostly — the empirical improvements are robust and consistent, but the specific claim about "task discrepancy reduction" versus "well-designed fine-tuning" lacks a proper control, and preservation of general CLIP alignment is unverified.

**Soundness of experiments:** Good — comprehensive evaluation across baselines, backbones, datasets, and ablations, with one notable gap (strawman control in Table 8) and one missing evaluation (general CLIP capability).

**Clarity of writing:** Good — the paper is well-organized and the method is clearly described.

**Value to research community:** High — RTD is practical, reproducible, and can be seamlessly integrated with existing methods.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>