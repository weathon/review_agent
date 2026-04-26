Now I have a thorough understanding of the paper and relevant calibration anchors. Let me compose the final review.

## Summary

The paper demonstrates that commonly used auto-interpretability metrics (e.g., fuzzing AUROC from the Bills/Paulo pipelines) fail to distinguish SAEs trained on genuinely trained transformers from SAEs trained on randomly initialized ones, across multiple Pythia model sizes and multiple randomization schemes. The authors further propose token distribution entropy as a proof-of-concept metric that does capture the qualitatively different feature structure (trained models develop increasingly abstract features across layers; randomized models remain token-specific), and provide toy model experiments investigating why random networks yield sparse, reconstruction-friendly activations.

## Strengths

- **Important empirical finding with well-designed experiments.** Figure 1 (and the full Figure 2) convincingly shows that auto-interpretability AUROC curves for trained and randomized transformers overlap substantially, while the Gaussian-input control is near chance. The use of four randomization variants (Step-0, re-randomized incl./excl. embeddings, and Gaussian control) isolates different confounds, and results hold across Pythia 70M–6.9B and multiple SAE configurations (Figure 18). This is a result the mechanistic interpretability community needs to engage with.

- **Constructive counterpoint via entropy analysis.** The last row of Figure 2 shows that token distribution entropy cleanly separates trained from randomized models: trained models show monotonically increasing entropy across layers (consistent with increasingly abstract features), while randomized models remain stuck at low entropy. This transforms the finding from a purely negative result into a path forward, suggesting that measuring feature abstractness is a missing ingredient in SAE evaluation.

- **Honest and careful discussion of scope.** Section 5 explicitly states: "we do not claim that SAEs fail to capture information from trained Transformers above and beyond randomly initialized transformers; only that aggregate auto-interpretability measures do not necessarily indicate the existence of interesting underlying features." The paper also correctly notes that CE loss score distinguishes trained from random (since randomized models have near-random loss regardless), and recommends routine randomized baselines as a concrete, actionable sanity check for practitioners.

## Weaknesses

### Fatal
None.

### Major

- **Title overclaims in a way that the paper's own evidence partially refutes.** The title states "Automated Interpretability Metrics Do Not Distinguish Trained And Random Transformers," but the paper's own entropy analysis (Section 3, last row of Figure 2) clearly distinguishes the two settings. The correct claim — which the abstract and conclusion approximate better — is that *commonly-used aggregate auto-interpretability scores* (fuzzing, detection AUROC) fail to make this distinction, but that other automated metrics (entropy, CE loss score) can succeed. The absolute framing of the title undermines the more nuanced reality presented in the paper and could mislead readers who only see the title. This isn't just cosmetic: it reframes what kind of contribution this is, from "metrics fundamentally fail" to "a specific class of metrics is insufficient, and complementary metrics help."

- **The toy model (Section 4) explains reconstruction quality, not auto-interpretability scores.** The main empirical finding is that *LLM-generated explanations score well on random-model latents* — a phenomenon about interpretability, not just sparsity. The toy models demonstrate that random MLPs produce sparse/reconstructable activations (Pareto frontier analysis), but they do not explain why an LLM would assign high interpretability scores to features that are semantically vacuous. Sparsity and reconstructability are necessary but not sufficient for high interpretability scores; the gap between these concepts is the more puzzling and arguably more important phenomenon, and it remains unexplained. The paper acknowledges this ("we defer conclusions as to the mechanism responsible to future work"), but the presentation could more clearly signal that Section 4 addresses a different question than the main finding.

### Minor

- **No variance reporting or statistical testing for auto-interpretability scores.** The paper reports results for 100 randomly sampled latents per SAE, without confidence intervals, standard errors, or bootstrap resampling. For large SAEs (R=64 on Pythia-6.9B has hundreds of thousands of latents), 100 is a small fraction. Auto-interpretability scores are known to have high per-latent variance. Without distributional information, the reader cannot assess whether the observed overlap between trained and random AUROC curves is robust or artifact of sampling. The paper mentions "Appendix E for multiple random seeds" but does not present variance quantification in the main text.

- **The re-randomized-vs-Step-0 comparison suggests activation scale confounds, which is discussed but not tested.** The paper notes that re-randomized variants (preserving parameter norms) are more similar to trained models than Step-0, and speculates this is due to "parameter norms." A simple experiment — normalizing activations before SAE training — could test whether reconstruction metric similarities between re-randomized and trained models are driven by scale rather than learned structure. This is mentioned as speculation but not investigated.

### Trivial
None significant.

## Nice-to-Haves

- Case studies comparing the *content* of high-scoring explanations for random vs. trained model latents (beyond Appendix J examples) to diagnose whether "similar scores" reflect genuinely similar features or different features that happen to score similarly.
- Systematic analysis of how the trained-vs-random gap scales with model parameters beyond the four Pythia sizes already shown.
- Formal validation of the entropy metric beyond this proof-of-concept — does it also capture other notions of feature quality beyond token-specificity?

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Harsh critic's claim that the entropy result "contradicts" the headline claim makes it sound like a fatal flaw.** In reality, the paper presents this as a constructive finding within its own framework, and the claim is about *commonly-used auto-interpretability metrics*, not all possible automated metrics. The title is overclaiming, but not to the point of invalidating the paper — it's a framing issue, not a contradiction.

- **Harsh critic's demand for probing whether high-scoring latents in random models are "qualitatively different."** The paper does provide Appendix J with examples; further case studies would strengthen but are not required for the central claim.

- **Strength Finder's claim that "Figure 1 directly undermines the validity of the most widely used SAE evaluation metric."** This is overclaiming strength. The finding shows auto-interpretability scores *fail a particular sanity check* — they don't distinguish trained from random. This is evidence of insufficiency, not invalidity. The scores may still capture other useful properties. Removed as it overclaims the paper's implications.

- **Harsh critic's claim that the Karvonen et al. (2024c) chess comparison "raises an important question the paper doesn't address."** This is a scope-creep criticism. The paper explicitly discusses this comparison and notes the difference between language and board game data. Demanding further investigation into why language specifically causes this is a nice-to-have, not a weakness.

## Novel Insights

The most novel insight emerging from the reviews is that the failure mode is specific to language (and likely other sparse, structured data domains) but not universal — Bricken et al. (2023) found that auto-interpretability *did* discriminate for small one-layer transformers, and Karvonen et al. (2024c) found clear discrimination for chess transformers. This suggests the problem scales with both model size and data complexity: as models get larger and their input distributions more structured, random networks increasingly inherit sufficient sparsity from the data for SAEs to find "interpretable-looking" features. The gap between the toy model's explanation (sparsity/reconstruction) and the unexplained puzzle (why LLMs assign high interpretability *scores* to these features) points to an important open question about what the auto-interpretability pipeline is actually measuring.

## Suggestions

- Revise the title to specify "Common Auto-Interpretability Metrics" rather than "Automated Interpretability Metrics" broadly, or soften the claim to "Can Fail to Distinguish" to reflect that metrics like entropy and CE loss do succeed.
- Add bootstrap confidence intervals or at minimum per-latent score distributions (not just aggregate curves) for the auto-interpretability results.
- Include a brief analysis or case study examining what the LLM actually describes for high-scoring random-model latents, to shed light on *why* the auto-interpretability pipeline gives false positives.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Scaling SAEs (TopK) | tcsZt9ZNKD | 8.2 | Much stronger — introduces architecture, scaling laws, trains on GPT-4. This paper is narrower. |
| Rethinking SAE Evaluation | HpUs2EXjOl | 5.75 | Similar scope (evaluating SAE metrics), similar quality. This paper has a more striking empirical finding. |
| SAGE Ground Truth | sknUS8X9q0 | 4.0 | Weaker — philosophical issues, less clean experiments. This paper is clearly better. |
| Gradient Interpretation Generalization | EwAGztBkJ6 | 4.0 | Weaker — limited significance, restricted theoretical contribution. This paper has stronger empirical work. |
| Blind Baselines Beat MI | BXMoS69LLR | 4.5 | Similar theme (evaluations are flawed), but that paper had significant methodological criticisms. This paper is somewhat stronger. |
| Multi-layer SAEs | XAjfjizaKs | 6.5 | Stronger methodological contribution. This paper's finding is more novel but less complete. |

This paper sits between the 5.75 anchor (SAE polysemy evaluation) and the 6.5 anchor (multi-layer SAEs). Its empirical finding is more striking than the polysemy paper's, and the experimental design is thorough, but the title overclaim and the gap between toy model explanation and the actual interpretability finding hold it back. Relative to the low anchors (4.0–4.5), this paper is clearly stronger — the experimental design is careful, the null comparison is principled, and the finding is important.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>