## Summary

This paper introduces Influence-Aware Time Series Forecasting (IATSF), a paradigm that reframes time series forecasting from predicting future values based solely on history ("self-stimulation") to modeling the external influences that drive system dynamics. The authors provide a control-theoretic analysis proving that ignoring influences imposes an irreducible error floor (Proposition 2.1), introduce a leak-free benchmark with temporally-synced textual influences, and propose FIATS—a lightweight model with channel-aware mechanisms (CASM, CAPS) for integrating text signals. Experiments across synthetic, physics, electricity, and market datasets show substantial gains over self-stimulated baselines.

## Strengths

- **Novel theoretical framing with formal proofs:** The control-theoretic analysis formalizing the "self-stimulation error floor" (Proposition 2.1) and the influence-efficacy result (Proposition 3.1) provide a principled foundation for why external influences matter. The proofs covering both linear and nonlinear systems (Appendix B) are thorough and give the paper conceptual depth rare in empirical TSF work.

- **Strong empirical gains across diverse settings:** FIATS achieves consistent improvements over strong baselines including foundation models (Chronos-L, MOIRAI-L, Time-MoE-U) and multimodal methods (TimeLLM). The 36–44% MSE reductions on Atmospheric Physics and NYC Traffic are substantial, and the near-zero error on the FM Toy dataset validates the theoretical predictions.

- **Thoughtful benchmark design:** The IATSF benchmark explicitly addresses information leakage, temporal synchronization, and independence of influences (Section 4.1)—addressing documented flaws in prior datasets like Time-MMD (Appendix N). The inclusion of channel descriptors and multiple dataset categories (synthetic, physics, market) is well-designed for evaluating influence-aware methods.

- **Architectural interpretability:** The CASM attention maps (Fig. 5, 10) demonstrate that FIATS learns meaningful channel-specific sensitivity patterns (e.g., atmospheric pressure channel attending to pressure-related text), providing evidence that the model captures genuine influence dynamics rather than spurious correlations.

## Weaknesses

### Major:

- **The independence assumption ($U_t \perp X_h$) limits theoretical generality and practical applicability of the "hard barrier" claim.** Proposition 2.1 derives the error bound assuming influences are independent of historical observations. In many real-world systems—influences are endogenous or correlated with past states (e.g., weather exhibits strong autocorrelation; economic indicators are path-dependent). When $U_t$ correlates with $X_h$, a sufficiently expressive self-stimulated model can partially infer influences from history, potentially reducing the error floor substantially. The paper frames the barrier as "hard" and "mathematical," but this holds only under an assumption that many real systems violate. The theoretical and practical implications of this assumption need more honest discussion, particularly since the paper's own Atmospheric Physics dataset involves weather—highly autocorrelated and predictable from history.

- **The Atmospheric Physics dataset may violate the paper's own independence requirement, raising leakage concerns.** Section 4.1 requires influences to be "independently evolving—external factors that influence the system but are not themselves outcomes of it." Yet the flagship dataset uses weather reports (describing "clear skies," "high pressure," "humidity level") to predict atmospheric physics variables that ARE weather. These text descriptions are essentially compressed observations of the target system state, not truly external influences. This creates a circularity: the "influence" text describes the same physical system being predicted, which both violates the independence assumption and inflates performance relative to genuinely exogenous settings (e.g., predicting traffic from weather, where the influence is truly external). The paper should explicitly acknowledge this tension and validate on a domain where the text is genuinely exogenous to the target.

- **Main experiments use ground-truth future influences (oracle setting), creating a significant gap to real-world deployment.** The results in Table 1 assume perfect knowledge of $U_f$ at test time. The paper's own analysis (Appendix B.3) demonstrates that influence forecaster error ($B\Sigma_{\hat{U}}B^\top$) can dominate model error, yet no experiments evaluate performance under realistic forecast noise for $U_f$. Figure 6 shows noise robustness but only for semantic perturbations to text embeddings, not for temporal or predictive errors in the influence itself. This oracle evaluation makes the "breaking the barrier" claim difficult to assess in practical terms.

- **Insufficient ablation to separate modality benefit from architectural contribution.** All baselines in Table 1 are self-stimulated (no text input). Table 3 ablates CASM and CAPS within FIATS, but no experiment gives a standard model (e.g., PatchTST) access to the same text embeddings via a simple fusion mechanism (e.g., concatenation or standard cross-attention). Without this, it remains unclear whether FIATS's gains stem from the principled CASM/CAPS design or simply from having text as an input modality. The ablation in Table 6 (training with zero/random text reducing to PatchTST-level) is suggestive but doesn't address this directly—a simple text-concatenation baseline would.

### Minor:

- **Nonlinear extension relies on first-order approximation.** The error bound for general nonlinear systems (Proposition B.1) uses a first-order Taylor expansion, discarding higher-order terms. For strongly nonlinear or chaotic systems, these terms can be significant. The paper acknowledges this as a limitation (Section 7) but states the barrier as universally applicable in the introduction and abstract, which overclaims.

- **Practical brittleness to misleading text.** Appendix I (Table 6) shows that training with good text but testing with incorrect text yields poor performance (MSE 0.724 vs. 0.186). This suggests the model trusts textual influence signals heavily, which could be problematic in deployment scenarios where text sources are unreliable or adversarial.

- **Weather as a validation domain has inherent circularity.** The NYC Traffic Speed dataset, where weather is genuinely exogenous to traffic, would be a stronger test of the paradigm. The paper's results on this dataset are strong (44.3% MSE reduction), but the primary validation (Atmospheric Physics) uses a domain where the "influence" and "target" are essentially the same system.

### Trivial:

- The proof notation switches between linear and nonlinear cases in ways that require careful reading; a unified presentation might improve accessibility.

## Nice-to-Haves

- Evaluate FIATS under realistic influence forecasting noise (e.g., use actual weather forecast errors rather than ground-truth reports) to characterize the practical performance envelope.
- Add a simple text-concatenation baseline (PatchTST + text embeddings via cross-attention or concatenation) to isolate the architectural contribution of CASM/CAPS.
- Test on a domain where the textual influence is truly exogenous to all target variables (e.g., macroeconomic news predicting individual stock prices, or policy text predicting health outcomes) to validate the paradigm where the independence assumption genuinely holds.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **GAUD reproducibility (only releasing embeddings):** The critic raised that releasing only pre-computed embeddings prevents verifying the "leak-free" nature of influences. However, the paper cites intellectual property constraints (Appendix O.5), which is a practical limitation, not a methodological flaw. The benchmark design principles are clearly documented.

- **Statistical significance concerns:** The critic suggested some improvements may not be statistically significant. The paper provides standard deviations (Table 10), a critical difference diagram (Fig. 12), and runs experiments 3–5 times. This exceeds standard reporting for the field.

- **Missing exogenous-variable baselines (TimeXer, ChronosX):** The spark finder suggested including these. The paper already benchmarks against TimeLLM (a text-informed method) and Chronos-L. While adding more baselines would strengthen comparisons, the existing set covers the key categories (linear, transformer, foundation, multimodal).

- **GAUD "cold-start" terminology:** The critic noted that games with developer logs aren't truly "cold-start." This is a minor terminology quibble; the paper's actual claim (Section 6.3) is about "cold-start problems for new games, where historical data is sparse but influence information is available," which is accurate.

- **Formatting artifacts in Table 1:** These are parser issues, not paper issues, and are explicitly excluded by the rules.

## Novel Insights

The paper's most insightful contribution is the formalization of the "self-stimulation barrier" as an information-theoretic impossibility result: no matter how expressive the model, if the external influence is independent of history, the model can only learn the conditional expectation $E_U[F(X_h, U)]$, not the actual realization. This reframes the TSF plateau as a *task formulation* problem rather than a *model capacity* problem—a genuinely different perspective from the prevailing scaling-oriented narrative. The channel-aware sensitivity analysis (showing that error reduction from influence is proportional to $\nabla_U F \cdot \Sigma \cdot \nabla_U F^\top$, meaning high-sensitivity channels benefit most) provides a principled explanation for why weather text helps pressure channels more than temperature channels—something the attention visualizations empirically confirm. However, the gap between the theoretical framework's assumptions (independence, perfect forecaster) and the empirical setup (weather-as-own-influence, oracle text) remains the paper's central tension.

## Suggestions

- Add a "PatchTST + simple text fusion" baseline to Table 1 to isolate the architectural contribution of CASM/CAPS from the modality benefit of having text input at all.
- Evaluate on at least one dataset where the text influence is genuinely exogenous and temporally forecasted (not oracle ground-truth), reporting performance degradation curves as a function of influence prediction error.
- Discuss the independence assumption's practical scope more explicitly: clarify which real-world systems approximately satisfy $U_t \perp X_h$ and which violate it, and provide empirical analysis of how much historical predictability of $U_t$ reduces the self-stimulation barrier in practice.

## Quality Assessment

- **Novelty:** High — the control-theoretic reframing of TSF limitations and the formal error bounds are genuinely novel contributions that shift the discourse from model capacity to task formulation.
- **Technical Soundness:** Moderate — the theory is correct under its stated assumptions but those assumptions (independence, perfect forecaster, first-order approximation) limit practical applicability more than the paper acknowledges.
- **Empirical Support:** Moderate — results are strong in the oracle setting, but the conflation of genuinely exogenous influences (traffic+weather) with self-descriptive influences (atmospheric physics+weather reports) and the absence of realistic influence forecasting experiments weaken the practical claims.
- **Significance:** High if the theoretical framework holds up — the "barrier" framing could redirect the field's efforts. But the practical significance depends on domains where genuinely independent influences are available as text.
- **Clarity:** Good — the paper is well-structured with clear separation between theory, benchmark, model, and experiments. The appendices are extensive but the notation is consistent (Table 4).