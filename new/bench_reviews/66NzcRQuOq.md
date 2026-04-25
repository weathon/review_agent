Now let me search for calibration papers.Now I have enough to write the final review.

---

## Summary

Pyramidal Flow Matching introduces a unified video generation framework that reinterprets the denoising trajectory as a sequence of resolution stages (spatial pyramid) and compresses autoregressive temporal history (temporal pyramid). Both pyramids are jointly optimized by a single flow matching objective on a standard MM-DiT backbone initialized from SD3 Medium, enabling end-to-end training without separate generation and super-resolution models. The resulting system trains a 768p, 24fps, 10-second video model in 20.7k A100 GPU hours and achieves the highest VBench quality score (84.74) among all compared models, including commercial ones.

---

## Strengths

- **Principled, unified flow matching objective (Eq. 11) across all pyramid stages**: Unlike cascaded approaches (Ho et al. 2022b; Pernias et al. 2024) that train separate generation and super-resolution models, the paper formulates a single velocity-regression target that simultaneously handles generation and decompression at all resolutions, with coupled noise sampling (Eqs. 9–10) to ensure trajectory straightness. This eliminates the need for separate per-stage models.

- **Theoretically grounded renoising scheme (Eqs. 12–15)**: The corrective noise at pyramid stage boundaries is not heuristic — the paper derives the blockwise covariance structure from first principles under nearest-neighbor upsampling (Eq. 14), and the choice of γ = −1/3 is justified as minimizing corrective noise amplitude while achieving full decorrelation. This provides a clean theoretical basis for the jump-point handling problem.

- **Convergence acceleration substantiated quantitatively (Fig. 7)**: Under identical data, tokens-per-batch, architecture, and hyperparameters, the pyramidal variant achieves roughly 3× faster FID convergence on MS-COCO compared to standard flow matching. This directly supports the training efficiency claim.

- **State-of-the-art VBench quality score (84.74) using only public data and 2B parameters (Table 1)**: The paper surpasses Gen-3 Alpha (84.11) and Kling (83.38) on quality score while using only publicly available training data — a meaningful result given the data asymmetry.

- **Zero-shot image-to-video emerges naturally**: Due to the autoregressive design with causal attention where the first frame acts as an image condition (Section 3.4), the model supports text-conditioned image-to-video without any additional fine-tuning (Fig. 6), a practical bonus over methods requiring separate I2V training.

- **Full open-source release of code and models**: The project page at https://pyramid-flow.github.io supports reproducibility and community use.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Ablations run only at early training on simplified settings — the method's contribution to final video quality is undemonstrated**: Fig. 7 ablates the spatial pyramid at 60k image generation steps on MS-COCO FID; Fig. 8 ablates the temporal pyramid qualitatively at 100k low-resolution video steps. Neither ablation runs the baseline variant to full training and evaluates it on VBench or EvalCrafter. The convergence curve in Fig. 7 has not flattened at 60k steps for either variant, so the long-run quality comparison is absent. More critically, the final system is initialized from the SD3 Medium text-to-image checkpoint (stated explicitly in Section 4.1), which contributes substantially to visual quality. Without a full-training video ablation comparing pyramidal vs. standard flow at matched compute, it is impossible to attribute the strong VBench performance specifically to pyramidal flow matching rather than to the SD3 initialization. This is the core empirical gap in the paper.

### Minor

- **Training efficiency claim conflates incompatible hardware; SD3 initialization cost excluded**: Section 4.2 compares "20.7k A100 hours" against "4.8k Ascend + 37.8k H100 hours" for Open-Sora 1.2. A100, H100, and Ascend chips have substantially different FLOP profiles and the combined Ascend+H100 figure is not straightforwardly comparable to A100 hours. Additionally, the 20.7k figure excludes the cost of the SD3 Medium pre-training. The efficiency claim would be more defensible if scoped as "video-specific fine-tuning cost starting from a text-to-image foundation model." The efficiency improvement is real, but the headline framing overstates it.

- **User study confounded by resolution and frame rate differences**: The paper explicitly acknowledges that user preferences for motion smoothness are partly explained by the frame rate advantage: *"This is due to the substantial token savings achieved by pyramidal flow matching, enabling generation of 5-second (up to 10-second) 768p videos at 24 fps, while the baselines usually support video synthesis of similar length only at 8 fps."* Preferences measured against baselines operating at lower resolution and lower frame rate cannot be attributed to generation quality alone. The 32.5% motion preference *loss* against Kling (which presumably generates at high fps) is consistent with this confound. This doesn't undermine the automatic metric results, but the user study as presented does not support quality claims beyond what VBench/EvalCrafter already show.

- **Selective baseline set in EvalCrafter (Table 2) inflates apparent result**: Table 1 includes CogVideoX-2B and CogVideoX-5B as strong public-data baselines, yet Table 2 omits CogVideoX entirely, comparing instead against ModelScope, Show-1, LaVie, and VideoCrafter2 — substantially weaker models. The margin over the best public-data baseline on EvalCrafter is only 1 point (244 vs. 243 for VideoCrafter2), which is within measurement noise. Adding CogVideoX to Table 2 is necessary to make the EvalCrafter state-of-the-art claim meaningful.

- **Semantic score is notably weak with unverified explanation**: The method scores 69.62 on VBench Semantic Score — lower than Open-Sora 1.2 (73.39), VideoCrafter2 (73.42), and substantially below CogVideoX-5B (77.04). Section 4.3 attributes this to "coarse-grained synthetic captions" but provides no supporting experiment (e.g., varying caption quality on a held-out set). Given that the paper otherwise claims SOTA among public-data models, this gap is a noteworthy limitation that deserves empirical support rather than assertion.

### Trivial

- **Inference time reported only at 384p**: Section 4.2 states "56 seconds to create a 5-second, 384p video clip." The headline resolution of the paper is 768p, and inference time at this resolution is not provided.

---

## Nice-to-Haves

- **Ablation of pyramid stages K**: K=3 is fixed throughout, and no experiment explores the quality-efficiency tradeoff as K varies. Understanding this tradeoff would substantially improve the paper's practical usefulness.
- **Full video ablation at convergence on VBench**: Running the standard flow matching baseline and the pyramidal variant to full training on video, then comparing VBench scores, would directly demonstrate the method's contribution beyond the SD3 initialization.
- **Controlled comparison at matched resolution and fps**: Side-by-side comparisons against CogVideoX-5B at the same resolution and fps would provide a clean assessment of quality independent of the frame-rate confound.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic Issue on γ = -1/3 not ablated**: While no ablation is provided for the γ value, the paper offers a clear theoretical justification (γ = -1/3 minimizes added noise amplitude subject to the semidefiniteness constraint). Demanding an ablation of a theoretically motivated constant is a nitpick for an empirical paper.

- **Harsh Critic claim on full sequence attention vs. factorized attention**: The critic notes lack of ablation comparing full-sequence attention to factorized attention. However, the paper does report ablation results for the causal attention design in Appendix C.2. Demanding additional architectural ablations at every design choice is scope creep for an empirical systems paper.

- **Strength Finder Strength on user preference over Kling (63.6% aesthetic)**: Selectively reports the aesthetic win while omitting the 32.5% motion loss. The user study is confounded as discussed; presenting half the results as a strength is misleading.

- **Strength Finder Strength on 20.7k GPU hours vs. Open-Sora (quantified efficiency)**: The comparison involves mismatched hardware. Retained in a weakened form in the main efficiency discussion.

- **Harsh Critic inference time claim as "structural" issue**: The missing 768p inference time is a minor omission, not a structural problem.

---

## Novel Insights

The most genuinely novel insight in this paper is the treatment of the video denoising trajectory not as a single high-dimensional flow but as a piecewise flow across resolution stages, where each stage transitions between a compressed, noisier distribution and a finer, cleaner one. The key mathematical contribution — deriving the corrective noise covariance at stage boundaries (Eq. 14) to ensure probability path continuity — is non-trivial and provides a principled alternative to the heuristic upsampling typically used in cascaded diffusion. The temporal pyramid further extends this principle to autoregressive history, compressing historical context without losing semantic coherence, and the joint training objective (Eq. 11) that unifies generation and decompression in a single model is cleaner than any prior cascade approach.

---

## Score and Decision

**Calibration anchors consulted:**

| Paper | Avg Score | Comparison |
|---|---|---|
| `lvgsPjRtLM` VideoDiT | **2.5** (Reject) | Very weak low anchor: limited novelty, unclear method, poor results. The paper under review is clearly far stronger. |
| `6rydymz1Qg` Efficient Video Flow | **4.0** (Reject) | Low anchor: similar efficiency motivation, but limited contribution and no SOTA results. |
| `tOzCcDdH9O` Matryoshka Diffusion | **6.25** (Accept/Poster) | Closest topically: multi-resolution joint diffusion, unified model for image/video. Key weakness was quantitative gap vs. baselines. Our paper achieves SOTA quality and stronger theoretical grounding but has weaker ablations than MDM. |
| `8pusxkLEQO` ARLON | **6.25** (Accept/Poster) | T2V with novel AR+DiT combination. Weaker VBench results than ours; comparable scope. |
| `lTrrnNdkOX` PT-T2I/V | **6.4** (Accept/Poster) | Efficient DiT for video, solid ablations, competitive results. |
| `LQzN6TRFg9` CogVideoX | **6.8** (Accept/Poster) | Direct competitor with similar T2V scope and strong results, slightly more comprehensive ablations. |
| `N8Oj1XhtYZ` SANA | **8.5** (Accept/Oral) | High anchor: efficient image generation with comprehensive per-component ablations, 20× smaller than Flux-12B. Significantly stronger ablation methodology than this paper. |

**Assessment relative to anchors**: This paper's core contribution is more principled than Matryoshka Diffusion (6.25) and its empirical results are stronger than CogVideoX (6.8) on the same benchmark, while the two papers share comparable scope and rigor. The major gap versus SANA (8.5) is the ablation methodology — SANA's components are each individually ablated to convergence, while this paper's ablations are early-training snapshots on simplified tasks. This gap justifies staying below 7. The paper comfortably exceeds the 6–6.4 cluster due to its principled theory, SOTA quality scores, and open-source release. I place this at **6.5**.

**Axes summary**: Originality — above average (novel flow formulation, clean theory); Research question — important (efficient video generation is a central challenge); Claims support — moderate (quality claims well-supported; efficiency and attribution claims need nuance); Experimental soundness — adequate but not rigorous (early-training ablations, baseline gaps); Clarity — good; Community value — high (open-source, competitive results, elegant framework).

**Decision: Accept (Poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>