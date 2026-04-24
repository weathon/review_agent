## Summary

HyperDAS proposes a hypernetwork architecture to automate two key steps in Distributed Alignment Search (DAS): dynamic selection of token positions for interventions and construction of linear subspaces for concept mediation. Trained on the RAVEL benchmark with Llama3-8B, it achieves higher Disentangle scores than a manually-tuned MDAS baseline. The paper also discusses design decisions (masking, sparsity loss, symmetric/asymmetric variants) aimed at mitigating faithfulness concerns common to supervised interpretability methods.

## Strengths

- **Novel architecture for automation**: HyperDAS replaces brute-force token/layer search with a transformer-based hypernetwork that learns to align base and counterfactual tokens via an intervention score matrix (Eqs. 6–9) and to generate attribute-specific subspaces via Householder transformations (Eq. 10). This is a creative and technically sound contribution to scalable interpretability.
- **Strong empirical results on RAVEL**: Table 3a shows HyperDAS achieving an average Disentangle score of 84.7% vs. 76.0% for the MDAS baseline. Figure 3b confirms consistent gains across layers for the “city” domain.
- **Insightful analysis of intervention patterns**: Figure 4 reveals that HyperDAS consistently selects entity tokens in counterfactual inputs across all layers, while base-input selections vary (e.g., syntax tokens in deep layers). This provides novel, hypothesis-generating observations about where attributes reside in transformers.
- **Clear presentation**: The framework diagram (Fig. 1) effectively breaks down the four-step pipeline, and the intervention score heatmaps (Fig. 2) make the soft–hard discretization mechanism tangible.
- **Honest limitations**: The paper explicitly acknowledges risks of false positives from supervised interpretability and the possibility of non-linear mediators, showing appropriate scholarly caution.

## Weaknesses

### Fatal
None

### Major
- **Baseline comparison is underspecified and potentially unfair**: MDAS is reported to use a “manually selected token position … the final token of the entity,” but the paper does not describe how MDAS was implemented (which layer, training hyperparameters, whether it was re-trained, or how token selection was decided for each domain). Without these details, the claimed state-of-the-art performance cannot be verified. A fair comparison would either use MDAS with an exhaustive token/layer sweep or at least report all configuration choices transparently.
- **Evaluation scope is extremely narrow**: All experiments use a single model (Llama3-8B) and a single benchmark (RAVEL). No results on other models (e.g., Mistral, GPT-2) or other causal interpretability benchmarks (e.g., Counterfact) are provided. This limits confidence that HyperDAS generalizes beyond the tested setting.
- **Critical design choices lack ablation and reproducibility**: The “masking of the base prompt” is cited as essential to prevent trivial solutions (Sec. 4), yet its implementation is not described. The sparsity loss schedule (“linearly increase from 0 to 1.5, starting at 50% of steps”) is stated as crucial but no sensitivity analysis or justification is provided. Without ablations, it is unclear how necessary these choices are for performance.
- **Symmetry intuition is not reconciled with empirical findings**: The symmetric variant is motivated by the principle that “get” and “set” operations should target the same features, yet asymmetric models often select different base/counterfactual tokens (Fig. 8) and achieve higher average Disentangle (Table 3a). The paper neither explains why symmetry fails nor analyzes when the symmetric variant helps (it is better on Causal for city domain). This leaves a key design rationale ungrounded.

### Minor
- **No quantification of result variability**: Table 3a reports single numbers per domain; no means ± standard deviations over random seeds are given. While the gaps appear large, the stability of HyperDAS gains is unknown.
- **Failure modes not investigated**: Figure 4 shows HyperDAS selects “Others” tokens (e.g., “is”) in deep layers for base prompts (~32% of cases). The paper notes this is “previously unknown” but does not check whether intervening on these tokens actually yields correct outputs or whether they represent artifacts. Without such analysis, the causal validity of those selections is uncertain.
- **Faithfulness discussion remains conceptual**: Section 4.2 lists architectural constraints (masking, sparsity loss) as safeguards against steering, but offers no independent validation (e.g., out-of-distribution generalization, robustness to ablations, alignment with known circuits). Performance on RAVEL is taken as indirect evidence, which is insufficient given known risks of “hacking” causal benchmarks.

### Trivial
- Minor figure caption duplication (e.g., “Figure 2” appears twice) and an inconsistent label (“Year1” in Fig. 1) are formatting artifacts that do not affect understanding.
- Notational redundancy in Eqs. 6–9 (e.g., repeated indexing) could be streamlined.

## Nice-to-Haves
- **Oracle token selection ablation**: With a fixed learned subspace, evaluate performance using ground-truth token pairs to separate the contributions of token localization vs. feature quality.
- **Cross-dataset validation**: Run HyperDAS on another causal benchmark (e.g., Counterfact) and/or another model family to test transferability.
- **Sensitivity analysis of sparsity loss weight λ**: Sweep over λ and its schedule to demonstrate robustness or identify a narrow effective range.
- **Qualitative case studies**: Pick 5–10 RAVEL examples and show base/counterfactual prompts, HyperDAS-selected tokens, intervention outputs, and MDAS outputs side-by-side.

## Removed Points
These points are flagged to be removed, treat them with caution:

- The harsh critic’s phrase “post‑hoc rationalization” for the faithfulness discussion was overly harsh; the paper does discuss design choices intended to mitigate concerns. The underlying issue—lack of independent validation of faithfulness—remains and is captured in the Minor weakness above.
- A comment about “MDAS baseline unfairly configured” is valid and kept; the concern is the lack of implementation details, not that MDAS is nonexistent.
- The observation that Householder vector similarities (e.g., Longitude/Latitude 0.97) might indicate entanglement was deemed speculative but not false; it is subsumed under the broader need for more analysis of subspace geometry.

## Novel Insights

The layer-specific token selection analysis (Fig. 4) reveals a pattern not emphasized in prior work: base‑input intervention locations shift dramatically from early (random/BOS) to middle (entity token) to deep (syntax tokens) layers, while counterfactual‑input locations remain stable on entity tokens. This suggests that the mechanism for retrieving an attribute may differ from the mechanism for setting it, which could explain why asymmetric variants outperform symmetric ones—a potentially important insight for future causal interpretability designs.

## Suggestions

1. **Provide full MDAS implementation details** (layer, hyperparameters, training regime) and, ideally, results with a token/layer sweep. If the sweep is computationally prohibitive, acknowledge this limitation explicitly.
2. **Add ablations for the sparsity loss schedule and base-prompt masking** (e.g., no masking, different λ ramps) to justify these “crucial hyperparameters.”
3. **Report variance**: Include mean ± std over ≥3 seeds for all main numbers and a statistical test for HyperDAS vs. MDAS differences.
4. **Test generalization**: Run HyperDAS on at least one additional model (e.g., Mistral-7B) and/or benchmark (e.g., Counterfact) to show results are not RAVEL-specific.
5. **Analyze the “Others” token selections**: Categorize these cases and measure whether intervening on those tokens still produces correct outputs; if not, report failure rates.

## Score and Decision

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>