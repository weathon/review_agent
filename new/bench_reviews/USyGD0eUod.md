Now I have all the information I need. Let me compose the final review.

## Summary

This paper investigates whether commonly used SAE quality metrics and auto-interpretability pipelines can distinguish trained transformers from randomly initialized ones. Across Pythia models (70M–6.9B) and multiple randomization schemes (Re-randomized with/without embeddings, Step-0, Gaussian-input Control), the authors find that auto-interpretability AUROC scores and reconstruction metrics are more similar between trained and randomized models than between either and the Gaussian control, suggesting these aggregate metrics are insufficient proxies for mechanistic interpretability. The paper also proposes token distribution entropy as a proof-of-concept metric that captures the "abstractness" distinction that auto-interpretability misses, and provides toy model analysis for why random networks might preserve or amplify superposition.

## Strengths

- **Important and timely research question**: Whether SAE evaluation metrics actually measure what they claim is a foundational concern for the mechanistic interpretability community. The sanity-check approach (comparing to random baselines) directly parallels influential work in other fields (e.g., Adebayo et al., 2020 for saliency maps).

- **Well-designed multi-scheme randomization**: The four variants—Re-randomized incl./excl. embeddings, Step-0, and Gaussian-input Control—allow disentangling contributions from learned weights, pretrained embeddings, initialization-scale effects, and input structure. The Gaussian-input Control serves as a true negative, validating that the auto-interpretability pipeline *can* detect unstructured activations (Figure 1, black line near chance), which strengthens the finding that it fails to distinguish trained from random.

- **Constructive diagnostic proposal**: The token distribution entropy metric (Section 3, Figure 2 last row) demonstrates that trained and randomized models produce qualitatively different features—trained models show increasing entropy across layers (features become abstract), while randomized models remain token-specific. This goes beyond a negative result by offering a proof-of-concept for what current metrics miss.

- **Robustness across SAE hyperparameters**: Figure 18 confirms similar patterns across expansion factors 16–128 and sparsities 16 and 32 on Pythia-160m, reducing concerns that the finding is an artifact of specific SAE configurations.

- **Thoughtful discussion of norm-preservation confound**: The paper explicitly notes that the re-randomization procedure preserves parameter norms from the trained model, and shows that the Step-0 variant (which lacks this property) exhibits different L1 norm behavior (Section 3, Figure 2). This identifies parameter scale as an important variable and demonstrates awareness of this confound.

## Weaknesses

### Fatal
None.

### Major

- **Overgeneralization from auto-interpretability AUROC to "commonly used SAE quality metrics"**: The abstract claims that "commonly used SAE quality metrics and automatic explanation pipelines" fail to distinguish trained from random models, and that "auto-interpretability scores and reconstruction metrics" are "similar" between the two. However, the paper's own evidence shows this is most clearly true for auto-interpretability AUROC on larger models. The entropy metric the authors themselves propose cleanly separates the conditions (Figure 2, last row). The CE loss score is inapplicable to random models by construction (Section 3: "CE loss score only makes sense for the trained variant"). For reconstruction metrics (cosine similarity, explained variance), the paper argues they are "more similar to trained than to control"—but being closer to each other than to an extreme outlier does not establish indistinguishability. The title is actually narrower ("Automated Interpretability Metrics") than the abstract's broad claim about "commonly used SAE quality metrics." The conclusion tempers this with "particularly aggregate auto-interpretability scores," which is more accurate. This overgeneralization matters because it misrepresents the scope of the finding and could lead to misinformed conclusions about the utility of reconstruction metrics.

- **No statistical quantification of "similar"**: The paper's central empirical claim—that metrics are "similar" between trained and random models—rests on visual comparison of overlaid curves (Figures 1, 2). No statistical tests, confidence intervals, or quantitative similarity measures are provided. Without defining what constitutes "similar enough" to be problematic, the core finding remains an impressionistic observation rather than a falsifiable, quantified result. Looking at Figure 1, the trained model's AUROC curve appears consistently above the randomized variants—whether this difference is statistically significant is impossible to assess from the current presentation. This is not a minor presentation concern; it directly bears on whether the paper's headline result actually holds.

### Minor

- **Model-size dependence acknowledged but not reflected in framing**: The paper notes (Section 2) that "auto-interpretability scores for randomized models were relatively low for smaller models (e.g., Pythia-70m) but that the gap was narrowed for larger models (e.g., Pythia-6.9b)." This is an important qualification: the metric may work for small models. The title and abstract do not reflect this size-dependent scope, and the paper does not analyze what changes across scale. If the failure is size-dependent, the contribution should be framed accordingly.

- **Norm-preserving randomization as underexplored confound**: The paper shows that the Step-0 variant (which doesn't preserve trained norms) diverges more from trained models than the Re-randomized variants (which do preserve norms). The paper acknowledges this and speculates about why, but does not systematically test whether the apparent metric "failure" is partially an artifact of norm-matching in the randomization scheme. An ablation with standard initialization (e.g., Kaiming init without matching trained norms) would clarify this.

- **100-latent sample for auto-interpretability**: With expansion factor 64 on a 4096-dim residual stream, the total latents are ~262K; sampling 100 is a small fraction. The paper does not provide justification or analysis of whether this sample is representative. While auto-interpretability is computationally expensive, this sample size limitation should be discussed.

- **Toy model section provides plausibility but not explanation**: Section 4 demonstrates that random matrices preserve or slightly amplify superposition, but has no analog of the auto-interpretability pipeline. The paper explicitly defers conclusions, which is honest, but the section adds limited insight into why auto-interpretability specifically gives similar scores.

### Trivial
None significant.

## Nice-to-Haves

- Statistical tests or bootstrap confidence bands on the per-layer AUROC curves to substantiate the "similar" claim
- Histograms or density plots of per-latent AUROC scores (trained vs. random) rather than just mean AUROC per layer, to reveal whether distributions overlap completely or whether trained models have a heavier right tail
- Ablation with non-norm-matching randomization (e.g., standard Kaiming initialization) to test whether metric similarity is an artifact of the norm-preserving scheme
- Scale-dependent analysis disentangling whether AUROC failure is driven by SAE size, model width, or residual stream dimensionality

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "CE loss score distinguishes trained from random"**: The harsh critic claims CE loss score is a metric that "does distinguish the conditions." However, the paper explicitly states "CE loss score only makes sense for the trained variant"—it cannot be computed for random models because they lack meaningful loss. This is not "distinguishing" in any useful sense; it is inapplicability by construction. The harsh critic's framing misrepresents the paper's own explanation.

- **Harsh critic: "Control comparison is a non-sequitur"**: The harsh critic argues the paper's logic ("trained and random are more similar to each other than to control, therefore metrics fail") is a non-sequitur. However, for the AUROC claim specifically, the paper shows actual curve overlap in Figure 1—the claim is not based solely on the comparison-to-control logic. The "more similar to trained than to control" framing is used for reconstruction metrics in the Evaluation section, which is a weaker claim than indistinguishability. The harsh critic conflates these two different lines of evidence.

- **Harsh critic: "Toy model section has single random seed"**: This is a minor observation about Section 4.3, but the toy model section is explicitly exploratory and deferred ("we leave the question of which predominates...to future work"). Demanding rigorous multi-seed experiments for a plausibility demonstration is scope creep.

- **Harsh critic: "Sampling only 100 latents is a small fraction"**: While this is a valid minor concern (moved to Minor weaknesses), the harsh critic's framing as a "critical issue" is disproportionate. Auto-interpretability is computationally expensive, and the paper provides multiple random seeds in Appendix E.

- **Strength finder: "Explicit limitation acknowledgment that strengthens credibility"**: This strength is generic and doesn't cite a specific section. Dropped for lack of specificity.

- **Strength finder: "Comprehensive multi-metric, multi-scale figure"**: While Figure 2 is comprehensive, this is a presentation quality observation rather than a substantive strength. Dropped as generic.

## Novel Insights

The paper's most insightful observation is the distinction between "interpretability" and "abstractness" as properties of SAE features: auto-interpretability AUROC can be high for both trained and random models because both can produce features that are *locally* interpretable (activating on specific tokens), but only trained models produce features whose activations *spread* across many tokens (high entropy), indicating abstract, computationally relevant features. This suggests the field may need to shift from evaluating "is this feature interpretable?" to "what computational level does this feature operate at?"—a qualitative shift in how SAE quality should be conceptualized.

## Suggestions

- Reframe the abstract and title-adjacent text to center the finding around auto-interpretability AUROC specifically, and acknowledge that reconstruction metrics show smaller differences (rather than no differences). The more nuanced framing in the conclusion ("particularly aggregate auto-interpretability scores") should be the paper's front-and-center claim.
- Add statistical quantification (even simple ones: mean ± std across latents, or bootstrap CIs) for the AUROC comparison at key layers. This would transform the visual impression into a falsifiable finding.
- Discuss model-size dependence as a core finding rather than a footnote: the fact that the problem emerges at scale is itself an important observation that merits analysis.

## Evaluation Axis Summary

- **Originality**: High. Applying sanity-check methodology from saliency maps (Adebayo et al.) to SAE evaluation is novel, and the finding that auto-interpretability fails to distinguish trained from random is a genuinely new and important negative result.
- **Importance of research question**: High. If SAE metrics don't measure what they claim, this has broad implications for mechanistic interpretability research.
- **Claim support**: Moderate. The visual evidence is compelling but unquantified, and the generalization from AUROC to "commonly used metrics" is overstated relative to the evidence presented.
- **Soundness of experiments**: Moderate. Good experimental design (multiple randomization schemes, control condition, multiple model sizes), but weakened by lack of statistical tests and the norm-preservation confound.
- **Clarity of writing**: Good. The paper is well-structured and the multi-metric Figure 2 is informative, though the framing inconsistency between title/abstract and body is confusing.
- **Value to community**: High. Even with its limitations, this paper raises a critical concern that the community needs to address, and the entropy metric proposal offers a constructive path forward.

## Calibration Anchors

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| tcsZt9ZNKD.md (Scaling and evaluating SAEs) | 8.2 | Accept (Oral) | Much stronger: introduces TopK SAE architecture, scaling laws, GPT-4 scale, multiple new metrics. This paper under review is far less comprehensive. |
| LC2KxRwC3n.md (Feature absorption in SAEs) | 7.5 | Reject | Identified a specific, well-documented SAE failure mode with ground-truth evaluation. This paper's finding is broader but less precisely documented. |
| HpUs2EXjOl.md (Rethinking SAE evaluation via polysemy) | 5.75 | Accept (Poster) | Most comparable: also finds traditional SAE metrics fail semantic sanity checks and proposes new evaluation. This paper has a more striking finding but weaker quantification and more overgeneralization. |
| TIjBKgLyPN.md (Limitations of TopK SAEs) | 5.50 | Reject | Similar: identifies limitations of current SAE approaches. This paper's finding is more surprising and impactful. |
| uOrfve3prk.md (Intervention-based eval) | 5.25 | Reject | Shows interpretability metrics are inconsistent. This paper has a cleaner experimental design but similar issues with overclaiming. |
| 3ZdGSTxKuy.md (Overclaimed OOD) | 2.0 | Withdrawn | Much weaker: fundamentally overclaimed with tiny dataset and narrow setting. This paper's overgeneralization is less severe and its core finding is real. |

**Score reasoning**: The paper is most comparable to HpUs2EXjOl.md (5.75) and TIjBKgLyPN.md (5.50). Like those papers, it identifies a genuine limitation in SAE evaluation. It has a more striking central finding than either (auto-interpretability truly failing to distinguish trained from random), which pushes it upward. However, the overgeneralization in the abstract and lack of statistical rigor push it downward. The paper is slightly below the HpUs2EXjOl anchor because that paper's claims are more precisely matched to its evidence, while this paper overclaims. I place it at 5.0.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>