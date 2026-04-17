Now I have enough context to write the final consolidated review. Let me synthesize all the information carefully.

Key issues to verify against the paper:

1. **"Two orders of magnitude" claim**: The actual numbers in Table 1 show improvements of ~6-15× for early metrics, not 100×. This is a legitimate concern.

2. **Missing baselines**: The paper compares against D-SCRIPT, Topsy-Turvy, and xCAPT5 used as standalone pairwise classifiers, but does not include simple baselines like: (a) aggregation of model scores over known partners, (b) full-embedding cosine similarity without active-region masking, (c) network-based methods (e.g., random walk with restart). These are legitimate concerns.

3. **Circularity/leakage in evaluation**: The re-ranking uses UniProt annotations, GO terms, biomedical text corpora, and AlphaFold-based structure — all of which may encode information about the same interactions that appear as "novel" in STRING v12. The paper partially acknowledges this ("it is uncertain if their gains reflect not only semantic generalization but also latent knowledge of interactions from the training data"), but still frames the contribution as "prospective prediction." This is a legitimate concern.

4. **Re-ranking only on top-10**: The paper's re-ranking is limited to 10 candidates per protein. This is very narrow.

5. **No statistical significance**: No variance, confidence intervals, or significance tests.

6. **"Interpretability" terminology**: The paper explicitly acknowledges that it does not use interpretability for explanations but for feature engineering. This is partially a framing issue.

7. **No ablation of the core mechanism**: The paper never compares active-region similarity against full-embedding similarity.

Let me now write the review.</think>## Summary

This paper introduces the problem of PPI Candidate Ranking—prioritizing which novel protein-protein interactions should be experimentally tested—and proposes a two-stage framework: (1) interpretability-guided retrieval that exploits predicted contact map activations from D-SCRIPT/Topsy-Turvy to identify active embedding regions of known partners, then ranks candidates by sliding-window cosine similarity in those regions; (2) a re-ranking module integrating multiple biological signals (interaction scores, AlphaFold2-derived structural plausibility, GO/pathway annotation overlaps, and LLM-based semantic similarity). Evaluation on STRING v11→v12 transitions shows large improvements over using raw model interaction probabilities, with PubMedBERT-based re-ranking providing the most consistent gains.

## Strengths

- **Well-motivated and practically important problem formulation.** Casting PPI discovery as per-target candidate ranking conditioned on known interactors is genuinely useful—it mirrors how experimentalists actually approach interactome extension for a protein of interest. The prospective evaluation using successive STRING releases (v11→v12) is a step forward from purely static benchmarks.

- **Substantial empirical improvements at early ranks.** The gains are not marginal: for D-SCRIPT, Recall@10 rises from ~1.2% to ~26%, and MRR improves by 4–6×. These are step-change improvements that would matter for practical candidate screening, regardless of how they are achieved.

- **Comprehensive multi-signal re-ranking analysis.** Table 2 provides a thorough pairwise rank-shift comparison across 9 re-ranking signals (cosine, IS, pDockQ, TF-IDF, Token/Location/KeyTerm overlaps, BioBERT, BioMedRoBERTa, PubMedBERT), revealing that semantic signals are surprisingly strong while structural plausibility (pDockQ) is better suited for filtering than ordering. This analysis of complementarity is a useful practical contribution.

- **Honest discussion of select limitations.** The authors explicitly acknowledge: (a) reliance on having known partners (failing for underexplored proteins); (b) rankings remain non-interpretable despite "interpretability-guided" terminology; and (c) uncertainty about LLM gains possibly reflecting memorized interaction knowledge.

## Weaknesses

### Major:

- **Missing baselines that would isolate the contribution of the contact-map-guided mechanism.** The core technical claim is that selecting "active regions" via predicted contact maps improves ranking over raw model scores. However, the paper never compares against straightforward alternatives that could close much of the gap: (a) aggregating the same models' interaction scores over known partners (e.g., for target p, rank candidates by max predicted probability among p's known interactors); (b) full-embedding cosine similarity to known partners without the contact-map masking step; (c) network-based baselines leveraging STRING v11 topology (e.g., random walk with restart, common neighbors). Without these, it is impossible to determine whether the improvement comes from the specific interpretability mechanism or simply from using *any* per-target aggregation over known partners—which the baselines are denied. This is the most critical gap in the experimental design.

- **The "prospective prediction" framing is undermined by information circularity.** The re-ranking module uses UniProt/GO/Reactome/ComplexPortal annotations, biomedical LLMs pretrained on PubMed, and AlphaFold2-derived structural scores. These sources are themselves downstream of the *same* experimental literature and curation pipelines that feed STRING v12's new edges. The paper acknowledges this concern in passing ("it is uncertain if their gains reflect not only semantic generalization but also latent knowledge of interactions from the training data"), yet still frames the contribution as "anticipating novel interactions" and "prospective prediction" throughout. A more careful audit (e.g., analyzing how many v12 interactions have pre-existing textual or annotation evidence from before the v12 cutoff) would substantially clarify what the evaluation actually measures. As designed, it likely measures "STRING-consistency with external curated knowledge" more than genuine extrapolation.

- **"Two orders of magnitude" improvement claim is numerically unsupported.** The abstract and conclusion assert "two orders of magnitude" improvement, but Table 1 numbers show D-SCRIPT Recall@5 from 0.0071→0.0452 (~6×), Recall@10 from ~1.2%→~26% (~15×), and MRR from 0.0074→0.0340 (~5×). These are substantial, but nowhere near 100×. The claim should be revised to match the actual metric values.

- **No ablation of the core interpretability-guided mechanism.** The central methodological idea—using contact-map-activated regions to select embedding sub-regions for similarity computation—lacks direct evidence of its necessity. An ablation comparing active-region cosine similarity against full-embedding cosine similarity (with the same anchor/aggregation strategy) is essential to justify this as the key contribution rather than a byproduct of the anchor-based aggregation.

### Minor:

- **Re-ranking evaluated only on top-10 candidates per protein (2,280 pairs).** This is a very narrow re-ranking window. The rationale (computational cost) is mentioned but no sensitivity analysis is provided for other cutoffs (k=20, 50, etc.). Additionally, it is unclear how representative this 2,280-pair subset is of the full ~279k v12 positives; no distributional characterization is given.

- **No reporting of variance or statistical significance.** All metrics (Table 1, Table 2) are single point estimates with no standard deviations, confidence intervals, or hypothesis testing. Given the large-scale dataset, this is partially understandable for ranking metrics, but the re-ranking analysis on 2,280 pairs would benefit from such measures.

- **The "interpretability-guided" terminology is somewhat misleading.** The paper itself acknowledges that it does not produce interpretable explanations—"we do not use it as a means of generating explanations." The contribution is more accurately described as contact-map-guided or activation-guided retrieval. While the authors explain this distinction in the text, the repeated use of "interpretability-guided" in the title and abstract creates expectations the method does not meet.

- **No analysis of performance stratified by number of known partners.** The method critically depends on KP(p), yet no analysis shows how performance degrades as |KP(p)| decreases. This is particularly important because proteins with few known interactors are precisely where new PPI discovery is most needed.

### Trivial:

- The negative sampling procedure (10:1 ratio) is described for training classifiers but the paper does not state clearly whether these synthetic negatives form the candidate pool for ranking evaluation, or whether all non-KP proteins are candidates. This affects the interpretation of precision numbers.

## Nice-to-Haves

- A combined/learned re-ranker that integrates multiple signals rather than evaluating each independently; this would demonstrate whether signals are complementary or redundant.
- Evaluation on a second database transition (e.g., IntAct or an earlier STRING version) or organism to test generalizability beyond a single v11→v12 transition.
- Qualitative case studies showing specific proteins where the method successfully promotes/incorrectly demotes true novel partners, with the active regions and semantic features responsible.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The method fundamentally relies on the assumption that novel interactions follow patterns of known ones."** (Harsh Critic, echoed in others) — The paper explicitly states this assumption and acknowledges it as a limitation in Section 6. The authors are transparent about the cold-start problem. Criticizing the assumption as a "structural flaw" ignores that the authors have already addressed it.

- **"Underexplored proteins with few known partners cannot benefit from the method."** — Again, the paper explicitly acknowledges this limitation. Raising it as a fatal flaw overstates it; it is a scope limitation, not a methodological error.

- **"xDAPT5/xCAPT5 availability concerns."** — Any model cited in the paper is assumed to exist per the rules. Removed.

- **"No comparison with ESM-2 or other modern PLM methods."** (Human Finder) — The paper's framework operates on top of D-SCRIPT and Topsy-Turvy, which use the Bepler & Berger encoder. Using ESM-2-based methods would require retraining the underlying PPI models, which is outside the paper's scope. The paper is about how to refine rankings from existing PPI predictors, not about building new ones.

- **"Formatting and writing issues."** — Removed per rules on style nitpicks.

- **"Complete training logs or hyperparameter details missing."** (implied reproducibility concern) — Removed per rules; the paper provides code in supplementary materials.

## Novel Insights

The pairwise rank-shift analysis (Table 2) reveals a counterintuitive finding: lightweight annotation-based heuristics (TF-IDF, Jaccard overlaps on GO terms, localization, key terms) achieve maintain-or-improve rates of 63–75%, rivaling heavyweight structure-based AlphaFold2 re-ranking (pDockQ at ~47–63%). This suggests that much of the "missing signal" in sequence-only PPI prediction comes from basic functional and contextual coherence (shared pathways, colocalization), not from fine-grained structural complementarity. This finding, if confirmed with controls for circularity, would have practical implications for PPI prioritization pipeline design—simple, scalable annotation signals may offer more bang-for-buck than expensive structural modeling.

## Suggestions

1. **Add a full-embedding cosine similarity baseline (same anchors, same aggregation, no contact-map masking)** and a **max-score-aggregation baseline** (for target p, rank candidates by max D-SCRIPT/Topsy-Turvy interaction score over KP(p)). These are the minimum needed to isolate the contribution of the contact-map activation mechanism.

2. **Revise the "two orders of magnitude" claim** to match the actual numbers (~6–15× on early metrics), and add a brief discussion of where the most dramatic improvements occur and why.

3. **Audit the circularity between re-ranking signals and evaluation labels:** Even a rough estimate of what fraction of v12 interactions have pre-existing UniProt/GO/PubMed evidence (dated before the v12 release) would significantly clarify the true "prospective" component of the evaluation.

4. **Add stratified analysis by number of known partners** |KP(p)|, showing where the method works and where it degenerates.

5. **Clarify the role of synthetic negatives** in the candidate pool for ranking evaluation, so that reported precision/recall numbers can be properly interpreted.

## Score and Decision Calibration

I compared this paper against the following calibration anchors:

- **LLaPA (eh1fL0zw8o)**: PPI prediction with data leakage concerns and limited novelty; scores 5–6, rejected. Similar circularity concerns, but LLaPA had a stronger methodological contribution (novel multimodal architecture) whereas this paper is more engineering-focused.

- **MAPE-PPI (itGkF993gz)**: PPI prediction with genuine architectural novelty (microenvironment codebook + masked codebook modeling); scores 3–8, accepted as spotlight. Much higher novelty bar with similarly strong results, but also received a score of 3 from one reviewer for limited clarity.

- **Structural interactomics (0JOhLEf2bX)**: Combines existing tools (GNNs + structures) with little methodological innovation; scores 3–5, withdrawn/rejected. Similar pattern of combining existing methods without sufficient novelty.

- **PPIformer (xcMmebCT7s)**: PPI design with strong architectural contribution; scores 3–8, accepted as poster.

This paper lies between the rejected papers (limited novelty, circularity issues) and the accepted ones (genuine methodological contribution). The problem formulation and empirical magnitude are genuine strengths, but the missing baselines are a critical gap that prevents confidence in the claimed mechanism, and the overclaiming on "two orders of magnitude" and "prospective prediction" is substantial. Comparable to LLaPA in the circularity/missing-baseline dimension, but with a novel problem formulation advantage.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>