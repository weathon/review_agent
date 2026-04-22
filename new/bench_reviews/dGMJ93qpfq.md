## Summary

The paper proposes Patch-Aware Prompting (PAP), a modular extension to CLIP prompt-tuning that incorporates patch-level information into vision features, text prompts, and prediction consistency via intra-/inter-view patch losses, view-tailored patch-conditioned text prompts, and an inter-logit consistency loss. PAP is evaluated on base-to-novel generalization, cross-dataset transfer, and domain generalization, showing consistent but modest gains over strong baselines such as PromptSRC and DePT.

## Strengths

- **Well-motivated problem and coherent overall design.** The paper clearly targets an important issue in CLIP prompt tuning—preserving generalization while adapting to specific tasks—and proposes a conceptually unified framework that uses patch-level information in the vision, text, and prediction branches (Sec. 3, Fig. 1).
- **Broad empirical evaluation on standard benchmarks.** Results span 11 base-to-novel datasets (Table 1), cross-dataset transfer (Table 2), and domain generalization on ImageNet variants (Table 3), with consistent improvements over PromptSRC and DePT in all three regimes.
- **Demonstrated modularity across methods.** PAP is applied on top of several different prompt-tuning baselines (PromptSRC, DePT, CoCoOp, CoPrompt), with Table 1, Table 2, Table 3, and especially Table 11 showing that adding PAP yields systematic performance gains across diverse underlying methods.
- **Nontrivial design choices are empirically probed.** The paper ablates clustering strategies (Table 8), conditioning mechanisms for prompts (Table 7), projection/adapter configurations (Table 9), and view/augmentation choices (Tables 10 and 12), giving some insight into why specific design decisions (e.g., Voronoi clustering, combined text adapter + ConvProj) were adopted.

## Weaknesses

### Fatal

- **Incorrect intra-view patch loss (Eq. 5) as written.**  
  Equation (5) defines
  \[
  \mathcal{L}_{\text{intra-view}} = \sum_{i=1}^M (1 - \text{sim}(\tilde{\mathbf{P}}_{\text{an}}^i - \tilde{\mathbf{P}}_{\text{an}}^i)),
  \]
  which uses a *difference* instead of a pair of arguments to `sim`. As written, each term is `sim(0)` and is independent of the model parameters; this does not “keep the patches aligned with the original CLIP model” as claimed in the text (lines 141–145) and provides no meaningful gradient signal toward zero-shot patches. Given that intra-view patch consistency is central to the claimed mechanism and appears again in ablations (Table 6), this is a specification error in one of the core losses. The reader cannot reconstruct a correct implementation from the paper alone, and the contribution relying on precise intra-view regularization is therefore not properly supported.

### Major

- **Ambiguity and inconsistency about zero-shot vs prompted patch features in the patch losses.**  
  In Sec. 3.2 the notation conflates zero-shot and prompted quantities. Line 139 states “Let \(V_{\text{an}}, P_{\text{an}}\) represent the zero-shot class and patch features,” but immediately after: “The prompted class and patch features for the anchor view are \(V_{\text{an}}, P_{\text{an}}\)”—reusing the same symbols. Later, \(\tilde{P}_{\text{an}}\) and \(\tilde{P}_{\text{aug}}\) are defined as ConvProj of *prompted* patches (line 139), yet Eq. (6) and the accompanying sentence (lines 147–151) say “we identify the closest zero-shot patch in \(X_{\text{an}}\)” and “using zero-shot outputs to calculate similarity,” written in terms of these \(\tilde{P}\) variables. This makes it unclear at each step whether \(\tilde{P}\) denotes frozen zero-shot features, prompted features, or ConvProj outputs of one or the other. Since the main conceptual claim is that PAP regularizes prompted patches toward *frozen zero-shot* CLIP at the patch level, this ambiguity undermines the ability to verify that the losses in Eqs. (5)–(8) actually implement that mechanism rather than prompt-to-prompt matching.

- **Ablation tables for components and losses are logically inconsistent, so component contributions are not evidenced.**  
  Table 4 (“Different Components of our framework”) has all rows marked with ✓ for Patch Loss, T.Text, and V.Feat, with different results (lines 296–305); no row corresponds to disabling any component. Similarly, Table 5 (“Different Losses”) has all rows with ✓ for \(\lambda_p, \lambda_t, \lambda_l\) (lines 306–314). Table 6 (“Patch Loss Comparison”) has all rows with ✓ for both “Intra” and “Inter” (lines 316–323), despite the text (lines 333–337) explicitly claiming to analyze “individual effects of intra- and inter-view patch losses” and “cases where no losses are applied.” As extracted, these tables do not actually show any on/off ablations, and thus they cannot substantiate statements like “each component improves performance” or “utilizing both intra- and inter-view losses boosts base accuracy while enhancing generalization.” Even if there was a typesetting issue, the current text+tables are self-contradictory from the reader’s perspective.

- **Very small, incremental empirical gains without variance reporting, relative to nontrivial added complexity.**  
  On base-to-novel (Table 1), average gains over PromptSRC are +0.89/+1.31/+1.08 (base/novel/HM), and over DePT +0.50/+1.56/+1.09. Cross-dataset improvements are ~0.5–0.6 points in average accuracy (Table 2). Domain generalization gains over PromptSRC/DePT are 0.31 and 0.53 average (Table 3). PAP roughly doubles training time and adds ~4.5–5M parameters and extra GPU memory (Table 13, lines 396–398). No standard deviations, multiple seeds, or confidence intervals are reported; single-run improvements of ~0.5–1.5 points in these benchmarks are plausibly within run-to-run variation. At ICLR level, such marginal gains for a considerably more complex training pipeline, without any robustness/variance analysis, fall short of clearly establishing that PAP yields reliably better performance rather than noise or tuning advantages.

- **Novelty and positioning relative to closely related patch-based prompt work remain under-specified.**  
  The paper acknowledges Long et al. (2024) as “independently developed” and “uses clustered patch tokens for text prompts but lacks inter-view consistency and patch integration into predictions” (lines 91–93), but there is no experimental comparison and only a brief qualitative distinction. Given that PAP’s main conceptual idea is also to use patch tokens for text conditioning and patch-informed consistency, the paper does not clearly delineate the incremental novelty beyond Long et al. and other patch-based CLIP works (e.g., FILIP). Without head-to-head results or ablations that directly isolate the added value of inter-view consistency and logit-level integration over such contemporaneous methods, the claimed conceptual advance is difficult to assess.

### Minor

- **Overstated claims in abstract and conclusion compared to evidence.**  
  The abstract claims PAP “represents the first integration of such [patch-level] semantics in this context” (line 15) and that results “mark a step forward in foundation model tuning.” Given the existence of prior patch-based CLIP extensions (FILIP; patch-level self-supervised CLIP variants) and acknowledged contemporaneous work (Long et al.), this “first” claim is overstated. Similarly, the conclusion and runtime section emphasize “significantly superior performance” and a “well-justified” resource trade-off (lines 396–398), which is not commensurate with the observed ~0.5–1% average gains without variance analysis.

- **Insufficient detail about some nonstandard components (e.g., Voronoi clustering).**  
  View-tailored text prompts rely on “Voronoi_Clustering(\bar{P})” (Eq. 9, lines 163–171), but the paper does not specify the distance metric, how sites/centers are initialized, or whether clustering is per-image vs global. This matters because per-image Voronoi diagrams over high-dimensional patch features are unusual and could be noisy or expensive; more detail is needed to allow faithful reimplementation and to understand stability.

- **Hyperparameter tuning transparency is limited.**  
  The paper states that \(\lambda_p, \lambda_t, \lambda_l\) are set to (1.0, 0.1, 1.0) “as default but modify it for individual dataset when required” (line 235), and that “global loss scaling factors mostly follow PromptSRC,” but does not detail which datasets deviate or how aggressively \(\lambda\)s are tuned relative to baselines. While this is not a fatal reproducibility problem, it increases uncertainty about whether small reported gains could partly stem from more tailored tuning.

### Trivial

- Occasional wording issues (e.g., slightly misleading phrase “our approach impressively achieves better results across all settings when paired with DePT” at line 264, despite noting a small average base-class drop) could be toned down for precision.

## Nice-to-Haves

- Multi-seed (or at least 3-seed) results with mean ± std for the main summary metrics (average HM in Table 1, average cross-dataset accuracy in Table 2, and average domain-generalization accuracy in Table 3) to demonstrate that improvements are statistically reliable.
- Clear, correctly specified ablation tables where each row toggles a single component (patch loss, view-tailored text, enhanced vision features, inter-logit loss, etc.) to substantiate component-level claims.
- Direct experimental comparison to Long et al. on at least base-to-novel and cross-dataset settings, with analysis of where PAP’s extra mechanisms help or fail.
- Qualitative visualizations of patch correspondences (e.g., nearest-patch maps across views) to show that inter-view patch consistency captures meaningful structure rather than trivial matches.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **“Ablation tables are unreadable because of PDF extraction”** — The critique that Tables 4–6 are “unreadable” due to extraction artifacts is not appropriate; we should judge the tables as they appear in the text we see. However, the substantive issue that they do not show on/off ablations *given their current content* is retained above as a Major weakness. Only the part attributing this to PDF issues is removed.
- **“Existing methods do not correspond to currently available systems / reproducibility doubts about existence of Long et al. or FILIP”** — Any skepticism about the existence or release status of cited models/methods is disallowed by the reviewing policy; the paper cites these works, so they are treated as existing.
- **Generic missing-related-work complaints** — Criticisms about not citing every possible adapter/self-supervision paper are removed, because we cannot confirm specific omissions without external search, and such complaints risk being speculative. The more grounded novelty-positioning issue relative to Long et al. and patch-based CLIP extensions is kept as a Major weakness.

## Novel Insights

The central methodological idea—multi-view, patch-level consistency combined with patch-conditioned prompts—is appealing, but as currently specified, the mathematical inconsistency in Eq. (5) and the ambiguous role of zero-shot vs prompted patch features prevent a reader from confidently tying the observed small empirical gains to the intended mechanism. This gap between conceptual story and precisely defined loss functions is, in this case, more damaging than the modest size of the gains themselves.

## Suggestions

- Correct Eq. (5) to a meaningful intra-view alignment between prompted and zero-shot patch features (e.g., `sim(prompted_patch, zero_shot_patch)`), and thoroughly audit Eqs. (5)–(8) to ensure arguments and symbols consistently reflect zero-shot vs prompted quantities.
- Rewrite Sec. 3.2 with disambiguated notation: use distinct symbols for zero-shot vs prompted features (and for pre-/post-ConvProj variants) and explicitly state which encoder is used where. This should make it possible to reimplement PAP without guessing.
- Redesign and re-render Tables 4–6 so each row corresponds to a distinct, clearly described configuration (components toggled; specific losses present/absent). Ensure the text in Sec. 4.4 matches the actual contents of the tables.
- Add multi-seed experiments for primary metrics and report variance. If gains remain ~1 point but are consistently above baseline with tight error bars, this will substantially strengthen the empirical case; if not, that itself is informative.
- Provide a focused comparison to Long et al. (2024), at least on a subset of datasets, and possibly an ablation emulating Long’s design inside the PAP framework to isolate the effect of inter-view consistency and prediction-level integration.
- Expand methodological details for Voronoi clustering and any other nonstandard components to enable faithful reproduction (distance metric, per-image vs global clustering, initialization, etc.).
- Consider simplifying PAP if possible—e.g., test whether some losses or adapters can be removed without harming performance—to yield a more favorable complexity/benefit trade-off.

## Score and Decision

### Calibration anchors

- **Medium-score anchors (4–6):**
  - `/home/wg25r/review_agent/human_reviews/wsRXwlwx4w.md` (CoPrompt, consistency-guided CLIP prompt learning), avg ≈5.75, Accept (poster). Strong empirical results and clear method; some novelty concerns but no core-specification errors. Compared to this, PAP has similarly broad experiments but weaker methodological clarity (incorrect loss, ambiguous notation) and smaller, less clearly substantiated gains.
  - `/home/wg25r/review_agent/human_reviews/ZPTHI3X9y8.md` (PATCH tuning strategy for LVLM hallucinations), avg ≈6.0, Reject. Clear, well-explained method with meaningful empirical improvements but some methodological and comparison gaps. PAP appears weaker on methodological clarity (core loss miswritten, ambiguous features) and has more marginal gains, so should be scored below this.
  - `/home/wg25r/review_agent/human_reviews/dsiwLm8yjz.md` and `/home/wg25r/review_agent/human_reviews/YG01CZDpCq.md` (other VLM prompt-learning works, avg ≈5.0–5.5, mostly rejected). These have clearer method descriptions and acceptable experiments but issues around novelty or limited gains. PAP is comparable or weaker because of the core loss mis-specification, so should not exceed this band.

- **High-score anchors (>7):**
  - `/home/wg25r/review_agent/human_reviews/bJx4iOIOxn.md` (visual prompt tuning/finetuning analysis), avg 7.5, Accept. Strong methodological soundness and empirical backing.
  - `/home/wg25r/review_agent/human_reviews/a745RnSFLT.md` and `/home/wg25r/review_agent/human_reviews/kC5nZDU5zf.md` (prompted VLMs with solid novelty and experiments), avg 7.0–7.5. PAP is clearly below these in both novelty clarity and evidential strength.
  - Given the specification error in a central loss and unclear ablations, PAP cannot reasonably be placed near this band.

- **Low-score anchors (<3):**
  - `/home/wg25r/review_agent/human_reviews/2VAi5F9BOJ.md` (PLPP, perplexity-based prompt learning), avg 2.5, Withdrawn. Main issues: very limited novelty, weak empirical evidence, and goals (comprehensible prompts) not achieved.
  - `/home/wg25r/review_agent/human_reviews/pdzHpQbGrn.md`, `/home/wg25r/review_agent/human_reviews/j1FLTvgyAh.md`, `/home/wg25r/review_agent/human_reviews/ZaudLwn0Hm.md` (few-shot prompt methods with small gains and over-claims), avg ≈2.5–3.0, Reject. These typically have modest or inconsistent improvements and sometimes unclear positioning or weak experiments.
  - Compared to these, PAP has a stronger conceptual story and broader experiments, so it is *above* the 2–3 band, but the incorrect core loss and ambiguous implementation keep it below the mid-5 range exemplified by CoPrompt.

Positioning relative to these anchors, PAP lies between the very weak PLPP-like works (~2.5) and the marginal-accept CoPrompt (~5.75), but meaningfully closer to the low band because of the specification error in Eq. (5) and unreliable ablations. A fair calibrated score is around 3.5.

**Final score:** 3.5  
**Decision:** Reject

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>