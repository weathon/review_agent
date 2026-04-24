## Summary

This paper introduces HyperDAS, a transformer-based hypernetwork that automates the token-position selection and feature subspace construction steps of Distributed Alignment Search (DAS), eliminating the brute-force search bottleneck in mechanistic interpretability. The method achieves state-of-the-art performance (84.7% average Disentangle) on the RAVEL benchmark with Llama3-8B, and provides novel empirical observations about where concepts are localized across different layers.

## Strengths

- **Novel automated DAS formulation:** The cross-attention decoder hypernetwork that jointly parameterizes soft token alignment (Eq. 8–9) and learnable Householder-rotated subspaces (Eq. 10–11) provides a clean, end-to-end differentiable relaxation of the discrete interchange intervention search. This is a genuine architectural contribution to the interpretability toolkit.
- **SOTA empirical performance on a standard benchmark:** HyperDAS achieves 84.7% average Disentangle across five RAVEL entity domains (Table 3a), outperforming the previous SOTA (MDAS at 76.0%) in a single unified model. The claim is factual and backed by the results table.
- **Systematic layer-wise analysis yielding novel findings:** The sweep across layers (Fig. 3b) and the token-type categorization (Fig. 4) reveal that HyperDAS learns to target entity tokens in middle layers and previously unreported JSON syntax tokens in deeper layers. These observations provide community value beyond the hypernetwork architecture itself.
- **Transparent analysis of failure modes:** Figure 7 and Sparsity Loss (Sec 3.5, Eq. 13) clearly diagnose what happens with too much or too little regularization, showing awareness of the optimization pitfalls inherent in discrete neural routing.

## Weaknesses

### Major

- **Missing brute-force DAS baseline undermines the automation claim:** The paper positions HyperDAS as solving DAS's brute-force search bottleneck (Sec. 1, "the field has developed a variety of methods for learning such interventions, but all of them require a brute-force search"), yet the only baseline reported is MDAS, which uses *manually fixed* token positions (Sec. 4: "MDAS relies on manually selected token position for intervention, which in our case is the final token of the entity"). Because MDAS's weakness is precisely the manual search that HyperDAS automates, the performance gains in Table 3a cannot be cleanly attributed to better search automation versus a crippled baseline. A brute-force DAS baseline with exhaustive layer/token search would establish that the hypernetwork truly automates the search better than an unlearned exhaustive approach. Without it, the SOTA claim is conditional on the choice of a weak comparison point.

- **Training objective conflates behavioral steering with mechanistic interpretation:** The RAVEL loss (Eq. 12) directly minimizes cross-entropy on the counterfactual label, optimizing for downstream output correctness. The paper itself acknowledges this risk in the introduction ("we run the risk of false-positive signals") and the limitations section, but the current experimental design offers no decoupled evaluation to verify that the learned token alignments and Householder subspaces correspond to the model's native causal circuitry rather than learned steering shortcuts. Because the Disentangle score is the exact quantity the training optimizes, high scores prove behavioral control capability but do not distinguish it from faithful causal discovery. The constraints (attention masking, sparsity loss) mitigate but do not resolve this fundamental gap.

### Minor

- **Fixed subspace basis $R^l$ is underspecified:** The Householder transformation applies to a fixed orthogonal matrix $R^l$ (Sec. 3.3: "In DAS, there is a fixed low-rank matrix with orthogonal columns $R^l$ representing a fixed subspace targeted for intervention"), but the paper does not specify how $R^l$ is derived for each layer—whether it comes from PCA of activations, a random orthogonal initialization, or some other procedure. This matters for reproducibility and for understanding how much flexibility the subspace search actually has.

- **Single-run results without variance estimates:** Table 3a reports point estimates for all HyperDAS and MDAS scores. For learned neural methods, reporting standard deviations or multi-seed results is minimal to assess whether HyperDAS's 8.7-point average improvement over MDAS is statistically stable rather than within optimization variance.

### Trivial

- **Deep-layer "JSON syntax" findings are underspecified:** The paper notes that ~32% of deep-layer base interventions target non-entity tokens (Fig. 4), including "JSON syntax tokens" and "Others" like the word "is," but does not provide concrete qualitative examples showing the actual token strings and their interventions. Providing a few explicit examples would clarify whether the model is exploiting formatting artifacts or genuinely using syntactic cues for mediation.

## Nice-to-Haves

- **OOD generalization test for learned subspaces:** Applying the learned token alignments to rephrased prompts or out-of-distribution entity types would strengthen the interpretability claim by showing the features transfer beyond the training distribution rather than memorizing prompt-specific routing shortcuts.
- **Quantitative analysis of masking robustness:** The attention masking of the base prompt (Sec. 4) is motivated well but not quantified. Measuring information leakage would strengthen the argument that the trivial solutions are effectively blocked.

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **Asymmetric token routing "contradicting invariant feature localization"** — The paper already reports and extensively discusses this phenomenon in Sec. 4.2 and Figure 8, presenting it as an empirical finding rather than hiding it. The reviewer treats this as a flaw when the paper already transparently addresses it. The paper shows symmetric variant collapses to 54.8% and asymmetric achieves 84.7%, documenting the trade-off honestly.
2. **Soft-to-hard discretization "optimization gap"** — The paper explicitly introduces the sparsity loss (Eq. 13) to handle the soft-to-hard transition and provides Figure 7 showing what happens with too little or too much regularization. The reviewer ignores this design choice, which directly addresses the concern.
3. **Symmetric variant collapse "contradicting theoretical framing"** — The symmetric collapse is reported as an empirical finding, and the paper does not claim symmetry should *theoretically* work—only that it is "intuitive" to consider. The paper presents both variants and lets the data speak; this is not a contradiction but a result.
4. **Deep-layer JSON syntax "conflicts with interpretability narrative"** — The paper presents these deep-layer findings as a discovery of "intuitive positions" and "previously unknown" intervention sites. Dismissing this as negative conflicts with the paper's honest reporting of novel findings.
5. **"Not even an interpretability tool, just a steering function"** — This phrasing overreaches. The paper acknowledges the steering risk and constrains it; this is a known tension in the field, not a paper-specific defect.

## Novel Insights

The paper's most interesting finding is arguably the layer-dependent behavior of the intervention sites: beginning-of-sentence tokens in shallow layers, entity tokens in middle layers, and JSON syntax/non-entity tokens in deeper layers. This layered progression suggests the model's internal representation of entity attributes evolves from syntactic/formatting cues to semantic entity representations and then back to formatting-aware locations—a pattern worth investigating independently of the hypernetwork architecture.

## Suggestions

- Add a brute-force DAS baseline (exhaustive token-pair search across layers) to establish that HyperDAS truly automates the search rather than simply benefiting from a floating intervention site compared to MDAS's fixed positions.
- Specify how $R^l$ is initialized/derived for each layer in the experiments section.
- Report mean ± standard deviation (or at least min-max across seeds) for the RAVEL scores in Table 3a to establish statistical stability.
- Provide 2–3 concrete examples of deep-layer interventions showing the exact token selected and the model's behavior to support the "JSON syntax" observations.

## Score and Decision

**Calibration anchors compared:**
- **High (≥6):** I4e82CIDxv (avg 8.00, Oral) — sparse feature circuits with clear downstream application and strong human-interpretable results. PBjCTeDL6o (avg 8.00, Oral) — UNI, novel baseline-free attribution method with thorough evaluation. Both papers have cleaner baseline comparisons and stronger justification of their core claims.
- **Medium (~5):** vsU2veUpiR (avg 5.25, Reject) — mechanistic unlearning with strong empirical results but concerns about interpretability overclaims. uOrfve3prk (avg 5.25, Reject) — intervention evaluation framework with good framing but limited scope.
- **Low (≤4):** JZjW3k4Kyc (avg 3.75) — automated circuit discovery rejected for unclear merits and missing comparisons. v675Iyu0ta (avg 5.60, but had lower individual scores) — interpretability illusions, rejected for narrow scope but rigorous analysis.

HyperDAS sits between the low and high anchors. It is substantially better than the low anchors (JZjW3k4Kyc) because of genuine SOTA results on a real benchmark, a novel architecture, and transparent reporting of limitations. It falls below the high anchors (I4e82CIDxv, PBjCTeDL6o) because of the missing brute-force DAS baseline, the interpretability-vs-steering overclaim, and the lack of variance estimates. It is comparable to or slightly better than the medium anchors (vsU2veUpiR, uOrfve3prk) because the empirical results are stronger and the architectural contribution is cleaner, though it shares similar concerns about whether the evidence fully supports the mechanistic interpretability framing.

**Score: 5.5** — The paper presents a genuinely novel method with solid empirical SOTA results and transparent discussion of limitations. The missing baseline and interpretability-over-steering gap are real concerns but do not invalidate the contribution. It is borderline for acceptance: a useful methodological advance that the community could build upon, but with claims that slightly outpace the evidence.

<pineapple>5.5</pineapple>
<orange>Reject</orange>