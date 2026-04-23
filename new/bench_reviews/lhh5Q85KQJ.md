Now let me do calibration searches to anchor my score.Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary

SubDisMO proposes combining SAM-style perturbations with resource-heterogeneous federated submodel training. Each client trains an adaptively masked submodel and applies a gradient-ascent perturbation step before the descent update, mitigating "arbitrary submodel sharpness." The paper provides convergence analysis yielding a rate O(1/√(QTC*)) parametrized by a minimum covering number C*, and a PAC-Bayesian generalization bound incorporating per-layer parameter remaining rates.

---

## Strengths

- **Unified convergence framework via C* (Corollary 1, Remarks 1–3):** The minimum covering number C* cleanly nests FedAvg (C*=N, δ=0), FedSAM (C*=N), RAM-Fed (δ=0), and OAP (C*=1, δ=0) as special cases. This parameterization is a useful contribution that organizes prior work into a single framework.

- **First generalization bound for resource-limited federated training (Theorem 2):** The PAC-Bayesian bound in Eq. (17) explicitly incorporates per-layer remaining rates s_j and reduces to the FedSAM bound when s_j=1 (Remark 4). No prior work provides such a bound for submodel-training scenarios.

- **Comprehensive empirical evaluation (Table 1):** SubDisMO outperforms all 19 resource-limited baselines across all six experimental settings (three CIFAR-10 + three CIFAR-100 splits), with consistent gains of 1.52–2.97% on CIFAR-10 and 0.55–1.26% on CIFAR-100 over the second-best method.

- **Loss landscape visualization directly validates motivation (Figure 3):** The comparison to RAM-Fed shows a visibly flatter loss landscape for SubDisMO, concretely demonstrating that perturbations reduce sharpness even in the submodel regime.

- **C* sensitivity analysis empirically confirms theory (Figure 4):** Higher C* yields faster convergence and higher accuracy, directly validating the theoretical prediction in Corollary 1.

---

## Weaknesses

### Fatal
None.

### Major

- **Misleading "generalized minimax" framing unsupported by content or experiments.** The paper's title, abstract, introduction, and related work position SubDisMO as solving general distributed minimax optimization—explicitly citing AUC maximization (SGDAM-PEF, LocalSCGDAM) and DRO as the motivating class of problems. However, the actual problem formulation in Eq. (2) reduces the inner maximization to a first-order Taylor perturbation over an ε-ball—precisely SAM—with no semantic content to the δ variable. The paper's own related-work section notes that "FedSAM… focus on provably optimizing… distributed minimax," and SubDisMO's update rule is identical to FedSAM applied to submodels. Not a single experiment evaluates any AUC maximization, DRO, or other genuine minimax objective. The "generalized" in the title is unsupported; the honest scope is "SAM applied to federated submodel training." This framing creates a false expectation of what the method can do and misrepresents the contribution's scope relative to the extensive minimax FL literature cited.

- **Missing RAM-Fed + SAM ablation isolating the perturbation contribution.** SubDisMO differs from RAM-Fed in two ways: (1) SAM-style perturbations; (2) the specific mixed-resource submodel assignment (50% of clients train 25% of model, 50% train 50%). Without a baseline of RAM-Fed augmented with SAM perturbations at the same δ, it is impossible to determine whether Table 1's improvements come from the minimax perturbation or from the different submodel structure. Since the perturbation is the primary claimed innovation, this is the most critical missing experiment.

### Minor

- **Assumption 4 justification inconsistent with experimental setup.** Remark 2 justifies the small-l regime by citing "existing model adaptive pruning works (Ma et al., 2021) which focused on masking insignificant parameters." However, the experimental setup in Section 5.1 uses random parameter assignment ("Low-resource clients randomly choose one submodel to train"), not importance-based masking. Random masking does not preferentially exclude small-magnitude parameters, so the cited justification for small l does not apply to the paper's own experiments. The assumption may still hold empirically, but it needs a more appropriate justification or empirical verification (e.g., measuring ||θ_q − θ_q ⊙ m||² / ||θ_q||² in the actual experimental runs).

- **Convergence bound over trained parameters only.** Theorem 1 and Corollary 1 explicitly bound `(1/Q) Σ_q Σ_{i∈K_q} E[||∇f^i(θ_q)||²]`, summing only over parameters in K_q (trained in round q). Untrained parameters are excluded. The paper acknowledges this in Remark 2 ("so we give a rigorous bound of the averaged gradient of the trained parameters"), but the claim of "asymptotically optimal convergence rate" in the abstract implicitly compares to FedAvg/FedSAM bounds that cover all parameters. When C*=N (full model), the comparison is valid, but the general submodel case leaves full-model convergence unaddressed.

- **Theorem 2's counterintuitive tighter bound for submodels unexplained.** The generalization bound in Eq. (17) has `∏_j (s_j + 1/r)²` in the numerator. Since s_j ≤ 1, this product is smaller than the full-model case (s_j=1), meaning the generalization bound is tighter when fewer parameters are trained. Remark 4 notes the s_j=1 case recovers FedSAM but does not explain why smaller submodels yield formally tighter bounds—this appears to be a consequence of PAC-Bayesian hypothesis class reduction and deserves a brief explanation.

- **Loss landscape comparison missing FedSAM.** Figure 3 only compares SubDisMO vs. RAM-Fed. Since FedSAM also applies SAM perturbations (to the full model), a comparison between SubDisMO and FedSAM loss landscapes would more directly validate the claim that submodel perturbations preserve the flatness benefits of full-model SAM.

### Trivial

- Both Eq. (5) and Eq. (6) use the same mini-batch ξ_{n,t-1} for the perturbation and descent steps, which is standard SAM practice. This is worth noting in the text but is not a defect.

---

## Nice-to-Haves

- Evaluate on at least one genuine minimax task (e.g., AUC maximization) to support the paper's positioning, or explicitly rescope the introduction to remove those claims.
- Experiment with importance-based submodel assignment (consistent with Assumption 4) vs. random assignment, since the theoretical justification favors the former.
- Scale to at least one medium-scale task (e.g., ImageNet or a language task) to support the "large-scale model" motivation cited in the introduction.
- Add a RAM-Fed+SAM control in Table 1 as an ablation (see Major weakness above).

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Harsh critic: "shared-batch SAM introduces bias"** — This is standard SAM implementation (Foret et al., 2021; Qu et al., 2022 also use shared batch). Not a paper-specific flaw; removed.
- **Harsh critic: "Assumption 3 upper bound of π² is trivially true"** — The normalization assumption IS tighter than raw bounded variance in practical FL; the observation that π² is trivial is a precision nitpick rather than a meaningful critique. Downgraded to trivial/removed.
- **Harsh critic: "EQ(2) conflates two different uses of δ"** — In Eq. (1), δ is not a free optimization variable; it is a scalar bound on the perturbation norm in both (1) and (2). The critic misread the problem formulation; Eq. (1) and (2) use δ consistently as a perturbation radius.
- **Harsh critic: "small experimental scale (10 clients)"** — Single-digit to low double-digit client counts are standard in FL benchmarking papers (FedSAM, RAM-Fed, FedP3 all use similar scales). This is a scope scoping issue, not a flaw.
- **Strength finder: "Tighter stochastic gradient variance assumption"** — While the normalized-gradient framing is more specific, the practical constraint (σ_l² ≤ π²) is indeed trivially satisfied and adds minimal rigor. Dropped as a strength.

---

## Novel Insights

The paper's most genuinely novel observation is the "arbitrary submodel sharpness" phenomenon: when clients train random submodels under data heterogeneity, aggregated submodel parameters fall into sharp local minima that degrade global model quality. The C* framework (minimum covering number) is an elegant parameterization that unifies a family of FL algorithms—including FedAvg, FedSAM, OAP, and RAM-Fed—under a single theoretical umbrella. The PAC-Bayesian bound in Theorem 2 incorporating per-layer remaining rates s_j is a concrete step toward generalization theory for partial-model federated training. The somewhat surprising mathematical consequence that smaller s_j formally tightens the bound (due to hypothesis class reduction) is worth exploring more deeply as a standalone insight.

---

## Score and Decision

**Axes evaluation:**
- *Originality*: Moderate. SAM in FL is explored (FedSAM, FedTOGA), but combining it with resource-heterogeneous submodel training and deriving C*-parametrized theory is a new combination.
- *Importance of research question*: Moderate-high. Resource-heterogeneous FL with large models is a relevant practical challenge.
- *Claims well-supported*: Partially. Empirical claims are well-supported; theoretical claims are sound but narrower than presented (trained-parameter convergence, not full model).
- *Soundness of experiments*: Moderate. Comprehensive baselines but small scale, missing ablation.
- *Clarity of writing*: Moderate. The "generalized minimax" framing is misleading; the paper's actual contribution is narrower.
- *Value to community*: Moderate. Useful C* framework and generalization bound, but framing limits impact.

**Calibration:**
- *Low anchor* (<3): FedPeWS (2.5, Withdrawn) — poor baseline comparison and no novel framing; this paper is clearly stronger.
- *Medium anchors* (4–6): FedTOGA (4.0, Withdrawn) — SAM+FL with incomplete theory; kWsJkH1tNi (5.0, Reject) — PAC-Bayes FL generalization with limited scope; FedP3 (6.0, Accept poster) — model heterogeneity + FL + theory, accepted.
- *High anchors* (>7): PAdaMFed (7.6), LoCoDL (7.5) — significantly more rigorous theory with cleaner claims.

SubDisMO is comparable to FedP3 (6.0, accepted) in structure and ambition but falls short because: (a) the framing mismatch is more serious than FedP3's "absence of theoretical insights," (b) the missing ablation weakens the core empirical claim, and (c) the Assumption 4 inconsistency is unaddressed. It is meaningfully above FedTOGA (4.0) due to cleaner theory, unified framework, and better experiments. Placing at **4.5** — borderline reject, below FedP3 but above the 4.0 cluster, given the meaningful but imperfectly supported contribution.

**Recommendation: Reject** — The paper contains a real contribution (SAM for resource-heterogeneous FL with C*-parametrized convergence and generalization bounds), but the misleading "generalized minimax" framing, missing key ablation, and internal inconsistency in Assumption 4 justification collectively make the paper inadequate for acceptance in its current form. A revision that honestly rescopes the contribution to SAM-on-federated-submodels, adds the RAM-Fed+SAM ablation, and fixes the Assumption 4 justification would constitute a credible submission.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>