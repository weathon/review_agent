Now I have good calibration anchors. Let me write the final review.

## Summary

HyperDAS proposes a transformer-based hypernetwork architecture that automates two key steps in distributed interchange interventions: (1) dynamically selecting token-positions in the residual stream where a concept is realized, and (2) dynamically identifying a linear subspace for intervention via Householder transformations of a base orthogonal matrix. Evaluated on the RAVEL benchmark with Llama3-8B, the asymmetric variant achieves state-of-the-art average Disentangle scores (84.7% vs. 76.0% for MDAS), while the symmetric variant (76.9%) barely improves on the MDAS baseline.

## Strengths

- **Addresses a genuine bottleneck in mechanistic interpretability**: Automating token-position search for interchange interventions is a real and important problem — prior methods like DAS/MDAS rely on manually specified (typically last-entity-token) intervention positions. HyperDAS's attention-based dynamic token selection (Section 3.2, Eqs. 6–9) is a principled architectural response to this bottleneck. The gains on City (Causal: 70.8 vs. 55.8) and Verb (Causal: 93.0 vs. 74.3) in Table 3a demonstrate the value of moving beyond fixed-position heuristics.

- **Householder transformation is a principled architectural choice**: Using Householder reflections (Eq. 10, Section 3.3) to modify a fixed orthogonal base matrix $\mathbf{R}^l$ conditional on concept encoding guarantees that $\mathbf{R} = \mathbf{R}^l\mathbf{H}$ maintains orthogonal columns, a requirement for distributed interchange interventions. This is a genuine improvement over DAS where $\mathbf{R}$ is fixed per concept.

- **Careful design to prevent evaluation hacking**: The base-prompt masking (Section 4) prevents a concrete failure mode where the hypernetwork could learn to condition intervention location on whether the base and target attributes match (a trivial solution that bypasses concept localization). The sparsity loss (Eq. 13) and its analysis in Figure 7 similarly address the risk of many-to-one alignments that achieve high weighted-intervention scores without meaningful one-to-one correspondences.

- **Layer-wise analysis of intervention positions reveals meaningful patterns**: Figure 4 shows how HyperDAS's selected positions evolve across layers — entity tokens at middle layers align with prior findings, while BOS-token and syntax-token targeting at shallow/deep layers provides new empirical observations about information flow in Llama3-8B.

- **Honest reporting of symmetric/asymmetric comparison**: The paper reports both variants despite the symmetric variant performing much worse, which is intellectually honest and enables the community to assess the faithfulness implications (Figure 8).

## Weaknesses

### Fatal
None.

### Major

- **The headline result relies on an asymmetric variant that violates a necessary property of genuine causal mediators, while the principled symmetric variant barely improves on the baseline.** The paper itself states the intuition: "if we have localized a concept, then 'get' operations and 'set' operations should both target the same features and hidden representations" (Section 4). Yet the best-performing model (Asymmetric per-domain, 84.7%) selects *different* token positions depending on whether the same input serves as base or counterfactual (Figure 8). The symmetric variant, which enforces this necessary causal property, achieves only 76.9% — barely above MDAS at 76.0%. The Symmetric All Domains variant collapses to 54.8%, *worse* than MDAS. The paper acknowledges the asymmetry observation but treats it as an interesting finding rather than as evidence that the method may be finding role-specific shortcuts rather than genuine causal structure. This matters because it directly threatens the paper's claim to be doing interpretability (recovering real model structure) rather than model steering.

- **The improvement over MDAS conflates token-position search with subspace identification; no ablation isolates the hypernetwork's contribution.** MDAS is constrained to intervene on the manually selected last entity token. HyperDAS searches over all token positions, which provides a substantial additional degree of freedom. The improvement from 76.0 → 84.7 could be largely attributable to this broader search space rather than the hypernetwork's subspace-identification component. A simple baseline — any reasonable token-search procedure (e.g., attention-based or gradient-based selection) combined with standard DAS — would isolate the contribution of the dynamic subspace. Without such an ablation, the core architectural contribution (Householder-conditioned dynamic subspaces) remains unvalidated. Note that even the symmetric variant (which also searches tokens) only reaches 76.9 vs. 76.0, suggesting token search alone accounts for most of the modest symmetric improvement.

### Minor

- **Disentangle improvements are primarily driven by Iso score increases, with Causal scores stagnant or declining for some domains.** In Table 3a, the Asymmetric per-domain variant *decreases* Causal for Occupation (50.7→50.4) and Nobel Laureate (56.0→55.4). The large Disentangle improvements are driven by Iso jumps (e.g., Occupation Iso: 88.1→99.1). Low-rank interventions (128/4096 dimensions) naturally have minimal effect on non-target attributes regardless of subspace quality, potentially inflating Iso scores. A random-subspace baseline at the selected token would clarify how much Iso improvement is trivially expected from low-rank structure.

- **The Householder transformation constrains the dynamic subspace to a single reflection of the base matrix $\mathbf{R}^l$.** This is a strong inductive bias — it may not be possible to reach all relevant subspaces via a single Householder reflection. The high cosine similarities between Householder vectors for related attributes (Country-Continent: 0.87, Longitude-Latitude: 0.97, Figure 6) suggest the learned subspaces may overlap substantially. If subspaces are this similar, it is unclear how disentanglement is achieved without cross-contamination. The paper does not discuss this tension or analyze effective subspace overlap directly.

- **Layer selection is reported as "best layer between 10 and 15."** While this applies equally to MDAS, the restricted range 10–15 and the "best" selection inflate reported numbers. Figure 3b shows substantial performance variation across layers. Reporting average performance across the layer range would be more informative and avoid selection bias.

- **Deep-layer targeting of JSON syntax tokens lacks disambiguation.** Figure 4 shows that at deep layers, HyperDAS targets non-entity positions including JSON syntax tokens. The paper frames this as discovering "previously unknown" attribute storage, but an equally plausible interpretation is that the model is finding shortcuts at layers where entity information has already been processed. No evidence distinguishes these interpretations.

### Trivial
None.

## Nice-to-Haves

- **Token-search ablation**: Combine a simple token selection procedure (e.g., attention-weighted or gradient-based) with standard DAS to isolate the hypernetwork's contribution to subspace identification.
- **Generalization test**: Evaluate whether a HyperDAS model trained on a subset of RAVEL attributes can localize a held-out attribute, testing whether the method discovers general structure vs. overfits to training concepts.
- **Random-subspace Iso baseline**: Intervene on a random 128-dimensional subspace at selected token positions to quantify how much of the Iso score is trivially achieved by low-rank structure.
- **Multi-layer intervention selection**: Currently limited to single-layer interventions; allowing cross-layer selection would substantially increase practical value.

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Unfair baseline comparison" (as originally phrased)**: The harsh critic claimed the comparison was "unfair" because HyperDAS searches tokens while MDAS doesn't. I've recharacterized this as a *confound* rather than an "unfair" comparison — the problem isn't that the deck is stacked against MDAS, but that the two contributions (token search vs. subspace identification) are conflated. The asymmetric improvement is real; the question is what causes it, not whether the comparison is unfair per the hard rule about asymmetries favoring the baseline.

- **Demand for "complete training logs" or reproducibility details**: Nitpicks about hyperparameter disclosure fall under the hard rule against trivial reproducibility concerns. The paper reports the key hyperparameters (Section 4).

- **Formatting/presentation complaints**: Any issues about section ordering or where masking discussion is placed are removed as formatting nitpicks.

- **Missing appendix/proofs**: Removed per hard rule — the parser strips appendices.

## Novel Insights

The asymmetric/symmetric comparison reveals a previously underappreciated tension in interpretability: the "get" and "set" operations of causal interventions may target genuinely different loci in the residual stream, and forcing symmetry (as causal interpretability ideals demand) may sacrifice significant practical performance. This raises a fundamental question about whether current interpretability benchmarks measure faithful causal recovery or reward methods that find effective steering shortcuts — a question the paper's own data gestures toward but does not resolve.

## Suggestions

- Add a token-search ablation (simple heuristic + DAS) as a baseline. This is the single most impactful addition that would strengthen the paper.
- Explicitly discuss why Iso scores improve more than Causal scores, and consider a random-subspace baseline for Iso.
- Report average performance across layers 10–15 rather than best-layer results.
- Discuss the tension between high Householder vector similarity and claimed disentanglement; compute effective subspace overlap directly.

## Calibration Summary

**High anchors (avg >7):**
- `/home/wg25r/review_agent/human_reviews/I4e82CIDxv.md` (Sparse Feature Circuits, avg 8.0, Accept Oral): Strong empirical results with scalable automated circuit discovery. HyperDAS has similar automated-search ambition but weaker faithfulness guarantees and less comprehensive downstream evaluation.
- `/home/wg25r/review_agent/human_reviews/3cuJwmPxXj.md` (Intervention Extrapolation, avg 8.0, Accept Poster): Strong theoretical identifiability results with practical validation. HyperDAS lacks comparable theoretical grounding.

**Medium anchors (avg 4–6):**
- `/home/wg25r/review_agent/human_reviews/sZq3lDDETp.md` (Circuit Probing, avg 4.2, Withdrawn): Good empirical results but reviewers questioned whether discovered circuits reflect true causal structure — directly analogous concern to HyperDAS's faithfulness issue. HyperDAS has stronger benchmark results but a more severe faithfulness concern (symmetric collapse).
- `/home/wg25r/review_agent/human_reviews/dsd04MYKax.md` (Sum-of-Parts, avg 4.8, Reject): Faithful-by-construction attributions with limited evaluation scope. HyperDAS has a similar "faithfulness" concern but a more novel architectural contribution.
- `/home/wg25r/review_agent/human_reviews/v675Iyu0ta.md` (Interpretability Illusions, avg 5.6, Reject): Shows MI methods can produce illusory results that fail OOD — directly relevant to HyperDAS's faithfulness concern. HyperDAS contributes a new method rather than a critique.
- `/home/wg25r/review_agent/human_reviews/GdbQyFOUlJ.md` (NeurFlow, avg 6.5, Accept Poster): Automated neuron-group interpretation framework with faithfulness concerns but sufficient empirical validation. HyperDAS is at the edge of this tier — similar automated search ambition, more severe faithfulness issues.

**Low anchors (avg <3):**
- `/home/wg25r/review_agent/human_reviews/fSbPwHjdDG.md` (Llamas Think in English, avg 3.0, Reject): Causal intervention paper with problematic erasure method and narrow experiments. HyperDAS has a more methodologically sound architecture but deeper faithfulness concerns.
- `/home/wg25r/review_agent/human_reviews/Wxl0JMgDoU.md` (SAE chess interventions, avg 2.5, Reject): Limited evaluation scope and missing ecological validation. HyperDAS is clearly above this tier in contribution and experimental rigor.

**Score reasoning**: HyperDAS identifies a real bottleneck and proposes a genuinely novel architecture, but the core faithfulness concern (symmetric variant barely beats MDAS; asymmetric variant violates necessary causal property) and the confound between token search and subspace identification prevent strong confidence in the claimed contribution. This places it below NeurFlow (6.5) which had similar automated-interpretability ambition with less severe faithfulness issues, and below the identifiability/SAE papers at 7-8, but above the low-scoring causal intervention papers (2.5-3.0) that had fundamentally flawed methodologies. The paper's honesty in reporting symmetric results partially mitigates the concern. A score around 4.5-5 reflects a paper with genuine innovation and solid engineering but where the central interpretability claim is not convincingly supported by the evidence.

## Score and Decision

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>