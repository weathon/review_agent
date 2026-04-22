Now I have a thorough understanding. Let me compose the final review.

## Summary

GEFA extends gradient-estimation-based feature attribution (previously limited to continuous inputs via GEEX) to arbitrary input types by introducing Bernoulli-parameterized proxy variables that enable path integration in a continuous proxy space. The paper proves that this proxy-space path method computes exactly Shapley Values (Theorem 2), establishes equivalence to Integrated Gradients under aligned edge paths (Theorem 4), introduces a control variate for variance reduction (Theorem 3), and demonstrates the method's effectiveness on text and image classification tasks.

## Strengths

- **Generalization to discrete features via proxy variables (Section 4.1, Equations 5–6):** The Bernoulli mask construction for proxy variables cleanly solves the genuine problem of extending gradient-estimation-based explanation to discrete/categorical inputs, making GEFA applicable where GEEX cannot operate. This is a concrete practical advance, demonstrated in the text classification experiment (Table 1).

- **Rigorous theoretical framework connecting multiple approaches:** The paper establishes that the proxy-space path integral yields Shapley Values (Theorem 2) and that GEFA becomes equivalent to IG under aligned edge paths (Theorem 4). These formal connections, regardless of their novelty relative to known results, provide a clearer unified understanding of how gradient-estimation-based, gradient-based, and game-theoretic methods relate.

- **Effective control variate design (Section 4.3, Theorem 3):** The control variate $h(|\epsilon|) = |\epsilon|/p$ is simple, provably preserves unbiasedness, and produces consistent empirical improvements (Tables 1–2), with a clear theoretical justification based on correlated output.

- **Informative qualitative comparison with GEEX (Section 5.3, Figure 2):** The demonstration that GEFA differentiates class-specific contributions (conflicting red/blue pixels for "dog" vs. "cat") while GEEX highlights generic low-level features is a genuinely insightful observation about the qualitative differences between binary-mask and Gaussian-perturbation query strategies.

- **Clean algorithmic presentation (Algorithm 1):** The 8-line pseudocode for GEFA is concise and implementable, lowering the barrier for practical adoption.

## Weaknesses

### Major

- **Theorem 2 is presented as a "surprising" discovery but is mathematically equivalent to Owen's 1972 multilinear extension representation of Shapley Values.** The paper's Equation 7 integrates the score-function gradient estimator along the unit interval parameterized by γ, which is exactly the Owen integral $\phi_i = \int_0^1 \sum_S \gamma^{|S|}(1-\gamma)^{p-1-|S|}[v(S\cup\{i\})-v(S)]\,d\gamma$ rewritten via the REINFORCE identity. The paper cites Owen (1972) indirectly through Okhrati & Lipani (2021) in the related work but never acknowledges in Sections 4.1–4.2 that this integral *is* the Owen representation, framing it instead as a "surprisingly" discovered connection (Section 4.2: "we surprisingly find that GEFA…is an alternative to compute Shapley Values"). The discovery that this integral equals Shapley Values was made in 1972; what is genuinely new here is the specific REINFORCE/score-function Monte Carlo estimator applied to it (Equation 8). This misrepresentation inflates the perceived novelty of the core theoretical contribution. The actual novelty — applying the REINFORCE trick to the Owen integral and the proxy variable construction for handling discrete features — is real but modest, and should be clearly framed as such.

- **The "information waste reduction" claim lacks variance analysis or query-budget scaling experiments.** The paper's central practical claim is that GEFA avoids "potential information waste" (Sections 1, 4.2) because each query contributes to all features simultaneously. However, REINFORCE/score-function estimators are well known to suffer high variance, particularly when terms like $\epsilon_i/\gamma$ and $\bar{\epsilon}_i/(1-\gamma)$ in Equation 8 diverge near $\gamma=0$ or $\gamma=1$. Whether GEFA is net more efficient than standard Owen sampling (which uses low-variance paired marginal contribution queries) depends entirely on the variance tradeoff. The paper provides: (a) no theoretical variance bounds, (b) no per-feature variance comparison with competing methods at matched query budgets, and (c) no query-budget scaling curves. Without any of these, the "information waste" claim is an assertion about per-query information content, not a substantiated claim about estimator efficiency.

### Minor

- **The "surpasses white-box approaches" claim in the abstract and conclusion is overstated.** The paper states GEFA "even surpasses white-box approaches" (Conclusion) and "surpasses the white-box explainer when tested with token removal" (Section 5.2). However, under the deletion metric aligned with IG's feature absence representation (embedding reset), GĒFA (0.6482) < IG (0.6622) in Table 1, and GĒFA (0.8747) < IG (0.8805) in Table 2. GEFA only beats IG under token removal, a setting architecturally more favorable to GEFA since IG operates in embedding space by design. The claim should be qualified as "surpasses IG under token-removal deletion" rather than generalized. Additionally, no error bars or statistical tests are reported for any Monte Carlo method, making it impossible to assess whether the small numerical differences are meaningful.

- **Antithetic Owen sampling (Okhrati & Lipani, 2021) is cited but never experimentally compared.** This is the most directly comparable baseline: it uses the same Owen multilinear extension with antithetic sampling for variance reduction. Without this comparison, it is unclear whether GEFA's specific estimation strategy (REINFORCE) offers a practical advantage over the existing antithetic Owen approach.

- **Assumption 1 (correlation between model output and number of present features) is unjustified for models with non-monotonic responses.** The paper claims (Section 4.3) this "generally holds for any properly trained model," but models can exhibit non-monotonic responses to feature inclusion (e.g., negation in text, adversarial patterns), making this assumption fragile. The paper provides no empirical validation of Assumption 1 other than the downstream attribution quality, which is a circular measure.

### Trivial

- The $\beta$ hyperparameter values for the control variate are not reported, making it difficult to assess practical sensitivity.

## Nice-to-Haves

- Query-budget scaling curves (attribution quality vs. number of queries for GEFA, PSHAP, and Owen sampling) would substantiate the "information waste reduction" claim and help practitioners choose appropriate budgets.
- Stratified or importance sampling over $\gamma$ (sampling more densely away from 0 and 1) to reduce variance from the $1/\gamma$ and $1/(1-\gamma)$ terms — this is a natural and potentially impactful extension.
- Attribution variance maps or per-feature standard deviations across runs, which would directly illuminate whether the score-function estimator's high variance in certain regimes creates practical problems.

## Removed Points

- **Formatting/typo nitpicks** (e.g., inconsistent notation between $\xi$ and $\tilde{\xi}$, Unicode characters in GĒFA): Parser artifacts, not author errors — removed per rules.
- **Demand for larger datasets or more models**: The paper tests on both text (BERT) and image (InceptionV3) with reasonable coverage for a methods paper — this is a generic request that wouldn't change core conclusions.
- **Unfair comparison favoring baselines (IG under embedding reset vs. token removal)**: The paper tests under both deletion settings and is transparent about which favors which method. The actual issue is the overclaiming in the framing, not the experimental design itself — the asymmetry is honestly presented even if the narrative overreaches.
- **Missing proofs in appendix**: Parser strips appendices; they exist in the original submission.
- **Claim that GEFA's Bernoulli mask construction is "exactly standard coalition sampling relabeled as proxy space"**: While technically accurate in the sense that Bernoulli masks with parameter γ produce the same coalition distribution as Owen sampling, this overlooks the genuine contribution of constructing a continuous proxy space that enables gradient estimation and path integration for discrete features. The relabeling serves a real functional purpose.

## Novel Insights

The most interesting observation emerging from the reviews is the tension between GEFA's genuine practical contribution — the proxy variable formulation that enables gradient estimation for discrete features without embedding-level access — and its theoretical framing, which substantially overstates novelty by presenting the Owen integral connection as "surprising" when it is mathematically equivalent to a 1972 result. The proxy variable insight (that you can parameterize feature presence through Bernoulli probabilities and then apply REINFORCE-style gradient estimation in this continuous space) is itself a valuable engineering contribution that enables a genuinely new class of black-box explanations for discrete features. Had the paper framed Theorem 2 as "we show that applying a REINFORCE estimator to the Owen multilinear extension yields an efficient black-box Shapley value estimator," the contribution would have been accurately represented. Instead, the disconnect between what is actually new and what is claimed to be new weakens the paper's credibility on its strongest point.

## Suggestions

- Reframe Theorem 2: explicitly acknowledge that Equation 7 *is* the Owen (1972) multilinear extension integral, and position GEFA's contribution as applying the REINFORCE estimator to this integral. This would strengthen rather than weaken the paper by showing clear awareness of the mathematical landscape while highlighting what is genuinely novel (the specific Monte Carlo estimator and proxy variable construction).
- Add query-budget scaling experiments (nAOPC vs. number of queries) comparing GEFA against PSHAP and Owen/Antithetic Owen sampling. This is the most direct way to substantiate the "information efficiency" advantage.
- Report error bars or confidence intervals across multiple runs for all Monte Carlo-based methods, and qualify the "surpasses white-box" claim to specify the conditions under which it holds.

## Evaluation

**Originality:** The proxy variable formulation for extending gradient estimation to discrete features is a genuine and useful contribution. However, the theoretical novelty is overstated — Theorem 2 recovers a known 1972 result, not a new discovery. The REINFORCE application to the Owen integral and the proxy construction represent moderate, incremental novelty.

**Importance of research question:** Feature attribution under black-box access is an important and practical problem. Extending to discrete features (text) is a real gap.

**Claims support:** The theoretical claims are mathematically correct but misframed in novelty. The "surpasses white-box" empirical claim is overreaching — under comparable deletion metrics, GEFA underperforms IG. The "information waste reduction" claim lacks variance analysis.

**Soundness of experiments:** Experiments cover both text and image settings with appropriate baselines, but lack error bars, statistical tests, budget scaling analysis, and comparison with the most directly related sampling baseline (Antithetic Owen).

**Clarity:** The paper is well-written with clear notation and good organization. The proxy variable introduction is elegant and easy to follow.

**Value to community:** GEFA provides a practical method and a clean framework, but the contribution is primarily an engineering adaptation (applying REINFORCE to Owen's integral via proxy variables for discrete features) rather than a deep conceptual advance.

## Calibration Anchors

- **Leverage SHAP** (avg 7.33, Accept Spotlight): Provides provably accurate Shapley value estimation with theoretical convergence guarantees. Compared to GEFA, Leverage SHAP has stronger theoretical novelty (new algorithm with provable guarantees) and better-framed contributions. GEFA's theoretical contribution is less novel (Owen integral restated).
  
- **Greedy PIG** (avg 4.25, Reject): A modification of Integrated Gradients with adaptive baselines. Similar to GEFA in being an incremental improvement over existing methods. Greedy PIG was judged as having insufficient motivation and evaluation. GEFA has cleaner theory but overclaims.

- **Old Dog New Tricks** (avg 5.0, Reject): Applies known techniques to an old architecture with overclaimed novelty. Similar pattern to GEFA: real but incremental contribution obscured by overclaiming.

- **Differentiation of Multi-objective Pipeline** (avg 2.33, Reject): Weak novelty with derivations largely from source material — more severe than GEFA, which has a genuine (if incremental) contribution.

- **On Formal Feature Attribution** (avg 5.25, Withdrawn/Reject): Proposes formal feature attribution with provable properties but limited empirical backing. Similar to GEFA in having a theory-practice gap.

GEFA sits in the 4–5 range: the proxy variable formulation and REINFORCE-to-Owen connection are real contributions, but they are incremental, and the paper inflates their significance. This is below the borderline set by papers like the Formal Feature Attribution paper (5.25) and the Old Dog paper (5.0), both of which were rejected, but GEFA has a more practical orientation and cleaner engineering. It sits approximately at 4.5 — a contribution that exists but is modest and overstated.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>