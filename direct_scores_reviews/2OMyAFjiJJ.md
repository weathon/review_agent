## Summary

This paper establishes the first almost minimax optimal convergence rate for Flow Matching (FM) under the *p*-Wasserstein distance (*1 ≤ p ≤ 2*). Building on the nonparametric estimation framework of Oko et al. (2023) for diffusion models but adapted to the ODE setting, the authors prove that FM achieves the rate *Õ(n^{−(s+1)/(2s+d)})* — matching the minimax lower bound up to poly(log n) and an arbitrarily small δ > 0 — when the variance parameter satisfies σ_t ~ √t (κ = 1/2). The analysis introduces an Alekseev-Gröbner lemma–based technique (replacing the Girsanov/KL approach used for SDEs) and covers a broad parametric family of mean/variance schedules, identifying κ = 1/2 as the critical threshold for near-optimality.

---

## Strengths

- **First convergence *rate* for FM, not merely consistency.** Prior work (Albergo and Vanden-Eijnden 2023, Benton et al. 2023b) showed consistency but not rates. Theorem 9 gives the first explicit minimax-matching rate, directly paralleling Oko et al. (2023) for diffusion models and providing a rigorous theoretical basis for FM's empirical competitiveness.

- **Genuine technical novelty: W_2 bounds via Alekseev-Gröbner lemma (Theorem 3).** The ODE setting forecloses Girsanov's theorem and KL-based techniques. The authors replace these with a novel application of the Alekseev-Gröbner lemma (Eq. 13) to obtain a W_2 bound from the integrated L_2-risk of the vector field, extending Oko et al.'s W_1 result to W_2. This is a non-trivial technical contribution that is specific to the FM/ODE regime.

- **Broader parameter coverage and concrete guidance on variance scheduling.** The paper analyzes the full one-parameter family σ_t ~ t^κ for κ ≥ 1/2 and shows conclusively that only κ = 1/2 achieves optimal rates (Theorem 9, Eq. 24). For κ > 1/2 the rate degrades to *n^{−(s+(2κ)^{−1}−1−δ)/(2s+d)}, which is strictly sub-optimal. This offers concrete theoretical justification for the σ_t ~ √t convention that is common in diffusion models and FM practice.

- **Insightful connection between FM at τ = 1 and KDE (Section 3.1).** The paper cleanly shows that solving the ODE all the way to τ = 1 recovers a Gaussian KDE with bandwidth σ_min, which achieves only the KDE rate *O(n^{−4/(4+d)})*. Early stopping at τ = 1 − T_0 is thus necessary to escape the KDE regime, and the paper rigorously characterizes how T_0 = N^{−R_0} must be chosen.

- **General coverage of joint distribution construction.** The result explicitly handles both independent coupling and OT-coupled (x_{[0]}, x_{[1]}) within the same theoretical framework, since the marginal *P_{[τ]}(x|x_{[1]}) = N_d(m_{[τ]}x_{[1]}, σ²_{[τ]}I_d)* depends only on the target sample x_{[1]}.

---

## Weaknesses

- **Time-divided architecture (K = O(log n) separate networks) is a significant gap between theory and practice.** The main result (Theorem 9) requires training one neural network per dyadic time interval. The authors themselves state in Section 4.4 that without this partition, the analysis yields only *Õ(n^{−1/(2s+d)})* for W_2, which fails to match the minimax lower bound. Standard FM practice uses a single network over [0, 1]. It is therefore unclear whether the almost optimal rate is achievable by any practically deployed FM implementation. The paper honestly acknowledges this limitation and notes the ODE-vs-SDE distinction makes it hard to close the gap (unlike the diffusion model TV bound of Oko et al., which avoids time division), but the severity of this gap for the overall claim deserves stronger emphasis.

- **Unusual and strong boundary smoothness assumption (A1): ñ > max{6s − 1, 1}.** For moderate smoothness s = 1, Assumption (A1) requires ñ > 5 on the boundary strip *I^d \ I^d_N*. The authors attribute this to compensating for the non-differentiability of p_0 at the boundary under (A2), but they do not discuss whether any natural class of practical distributions satisfies this joint requirement, nor do they analyze whether this assumption could be relaxed. This is a genuine concern about the tightness and reach of the result.

- **Sub-optimality for κ > 1/2 is unresolved.** For κ > 1/2, Theorem 9 gives a rate strictly worse than the minimax lower bound. The paper does not discuss whether a matching lower bound for κ > 1/2 exists (which would confirm the slower rate is a fundamental statistical phenomenon), or whether an improved proof technique could recover optimality. This gap weakens the completeness of the analysis and should be discussed explicitly.

- **Typographical/notation error in informal Theorem 1, Eq. (10).** As rendered, the exponent reads `s + (2κ)κ − 1 − δ`, but the formal Theorem 9 and Eq. (24) make clear it should be `s + (2κ)^{−1} − 1 − δ`. This discrepancy between the informal and formal statements is likely a PDF extraction artifact, but it could mislead readers relying on the informal version and should be corrected.

- **Assumption (A2) — density bounded away from zero on a compact cube — excludes much of modern generative modeling practice.** Combined with (A1)'s compact support, this excludes distributions with light or heavy tails, lower-dimensional concentration, or multimodal structures with near-zero density regions. This is standard in nonparametric minimax theory but limits the practical scope of the guarantees more than the paper acknowledges.

---

## Nice-to-Haves

- **Empirical illustration of the convergence rate.** Even a simple 1D or 2D numerical experiment verifying that the *n^{−(s+1)/(2s+d)}* rate is achieved in a setting with known Besov smoothness would significantly strengthen the paper's impact for readers who want to connect abstract rates to observable behavior.

- **Discussion of how κ = 1/2 maps to specific FM variants.** The paper identifies σ_{[τ]} = (1 − τ)^{1/2} as optimal but does not discuss whether the affine path (σ_{[τ]} = 1 − τ, κ = 1), OT-CFM, or rectified flow satisfy or violate this condition, nor what schedule modification κ > 1/2 variants would require for near-optimality.

- **Discuss T_0 selection in practice.** The optimal stopping time T_0 = N^{−R_0} requires knowledge of s and d. A brief remark on data-adaptive or conservative choices would be helpful.

- **Extension to Total Variation or KL divergence.** A TV or KL bound (analogous to Oko et al.'s optimal TV result without time division for diffusion models) would provide a closer theoretical comparison; the authors acknowledge this is an important open direction, and even a negative result or conjecture would be useful.

- **ODE discretization.** The paper assumes exact ODE integration. Euler or Runge-Kutta errors accumulate near t = 0 where the vector field is ill-conditioned, and this could dominate the statistical error in practice. A brief discussion of how discretization error scales relative to the derived statistical rate is desirable.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Alekseev-Gröbner Theorem 3 has a √t redundancy."** (Harsh Critic) The √t factor in Eq. (13) is outside the integral over [0, t] and is a standard Gronwall-type factor arising in ODE perturbation bounds; it is not redundant or erroneous.

- **"The paper is dismissive of Jiao et al. (2024)."** (Harsh Critic) The paper's characterization that Jiao et al. did not include the degree of smoothness in their rate is accurate per the paper's description. That Jiao et al. handles ODE discretization (which this paper does not) is a scope difference, not a mischaracterization. The comparison is fair given the stated aims.

- **"Incremental relative to Oko et al."** (Harsh Critic, general) The adaptation from SDE to ODE requires fundamentally different proof techniques (Alekseev-Gröbner vs. Girsanov), and extending the result from W_1 to W_2 is a genuine contribution. Labeling the overall contribution as merely incremental is not accurate.

- **"The relationship to joint distribution construction is not verified."** (Harsh Critic) The paper explicitly states in Section 2.2: "This case is covered by our result, which does not depend on the construction of joint distribution," immediately following the affine path discussion. The marginal P_{[τ]}(x|x_{[1]}) = N_d(m_{[τ]}x_{[1]}, σ²_{[τ]}I_d) is a conditional given x_{[1]} only, independent of how (x_{[0]}, x_{[1]}) are jointly coupled.

- **Requesting confidence intervals or multi-run statistics for large-scale benchmarks.** Not applicable (this is a theory paper with no benchmarks).

- **Heavy reliance on Oko et al. as a standalone weakness.** Acknowledged as context, but the technical contributions are distinct enough that this should not be a scored weakness.

---

## Novel Insights

The most genuinely novel observation, both in the paper itself and emerging from synthesis of the reviews, is the identification of κ = 1/2 (i.e., σ_t ~ √t) as a phase transition for optimality rather than merely a conventional choice. The paper shows rigorously that for κ > 1/2 the FM rate strictly degrades — suggesting that the "diffusion-like" variance schedule is not just traditional but theoretically necessary. A deeper open question not fully addressed is whether this κ = 1/2 threshold also corresponds to a matching minimax *lower bound* specific to FM (i.e., whether κ > 1/2 is fundamentally sub-optimal for FM, or whether better proof techniques could recover the optimal rate). Resolving this would clarify whether the paper's result is tight or whether the sub-optimality for κ > 1/2 is a proof artifact — a question that has implications for FM architecture design beyond what the paper currently discusses.

---

## Suggestions

1. **Fix the exponent in Informal Theorem 1, Eq. (10):** Correct `(2κ)κ` to `(2κ)^{−1}` and ensure the informal and formal statements are consistent.
2. **Strengthen Section 4.4 discussion of the time-division limitation:** Explicitly frame the single-network result as an open problem and discuss whether the W_2 bound gap (between the optimal and the non-partitioned Õ(n^{−1/(2s+d)}) result) could be closed via an ODE-analogue of Girsanov's theorem or a different metric (e.g., KL divergence).
3. **Discuss whether Assumption (A1)'s boundary smoothness condition can be relaxed:** At minimum, provide an example class of distributions satisfying A1 and A2 simultaneously, or state explicitly that it is an open question whether this assumption is necessary.
4. **Address the κ > 1/2 case more thoroughly:** Either provide a lower bound showing the slower rate is fundamental, or conjecture a better proof for the optimal rate, so readers understand whether sub-optimality for κ > 1/2 is structural.
5. **Add a remark connecting σ_t ~ (1−τ)^κ to concrete popular FM variants** (affine path, rectified flow, diffusion path) to immediately orient practitioners as to which existing schedules satisfy or violate the κ = 1/2 condition.

---

**Evaluation across axes:**

- **Novelty:** Moderate-to-high. The first convergence rate for FM and the Alekseev-Gröbner approach for W_2 are genuine contributions; the framework inherits significant structure from Oko et al. but the ODE-vs-SDE transition requires non-trivial new ideas.
- **Technical soundness:** Good overall; the main arguments are well-structured and the proof strategy is clear. The typographical error in the informal theorem and the undiscussed tightness of A1 are the main blemishes.
- **Empirical support:** Absent, as expected for a pure theory paper; the theoretical claims are internally consistent with the formal framework.
- **Significance:** Meaningful for the theoretical foundations of generative models; the practical impact is tempered by the time-division requirement and restrictive distributional assumptions.
- **Clarity:** Good for a theory paper; the time-reversal notation change is flagged and the discussion sections are honest about limitations.

MY FINAL SCORE: <pineapple>6.5</pineapple>