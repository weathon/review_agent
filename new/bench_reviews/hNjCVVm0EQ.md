Now I have all the information I need. Let me compose the final review.

## Summary

MamKO proposes adapting Mamba's selection mechanism to generate time-varying Koopman operator matrices from historical data, enabling a linear time-varying state-space model for nonlinear systems. The generated matrices preserve convexity in a subsequent MPC formulation, yielding a computationally efficient predictive control framework. Experiments on five benchmark systems (time-invariant and time-varying) demonstrate improvements in modeling accuracy and control performance over Deep Koopman (DKO), MLP, and SAC baselines.

## Strengths

- **Core design is well-motivated and technically sound**: Generating time-varying Koopman matrices from historical data (rather than current inputs) is a principled choice that avoids the bilinear term $B_k u_k$, preserving convexity of the MPC optimization (Section 3.2: "As a substitute, we generate the matrices from the historical data"). This is the paper's most insightful contribution.

- **MPC formulation correctly leverages convexity**: Once matrices are generated, the resulting QP is convex (Section 4, Eq. 10). The computational advantage is clearly demonstrated: MamKO-based MPC reduces computation time by 83–99% over MLP-based MPC across all systems (Table 2), while remaining within sampling period constraints (Table 3).

- **CELU activation ablation validates a key adaptation**: Replacing Mamba's negative exponential with negative CELU to allow positive continuous-time eigenvalues is a reasonable modification, and Figure 4 demonstrates its benefit across three systems, with particularly clear gains on GRN and RSCP.

- **ZOH discretization ablation is clean and informative**: Table 1 shows that proper ZOH discretization (Eq. 6) consistently outperforms the naive "Multiplication" approach, with especially large gaps on RSCP (2.92e-3 vs 7.60e-3) and time-varying RSCP (9.67e-3 vs 1.19e-2).

- **Strong empirical signal on time-varying systems**: Figure 2f is compelling — MamKO's modeling advantage over DKO and MLP grows substantially as the angular frequency of parameter variation increases, consistent with the claim that generated operators adapt to changing dynamics.

## Weaknesses

### Fatal

None.

### Major

- **Comparison on time-varying systems is confounded by asymmetric information access**: MamKO conditions its matrix generation on the historical data sequence $[z_{k-H:k-1}^T, u_{k-H:k-1}^T]^T$ (Eq. 8, Figure 1), while DKO uses a constant (time-invariant) Koopman operator with no historical context, and the MLP baseline is described only as "based on multilayer fully connected NNs" (Section 5) without specifying whether it receives historical data. For time-varying systems — the paper's central motivation — the performance gap may partially reflect MamKO's access to more temporal information rather than the specific architectural innovation of generating operator matrices. A history-augmented MLP baseline receiving $[x_{k-H:k-1}, u_{k-H:k-1}]$ as input would isolate the contribution of the generative architecture from the contribution of simply having more temporal context. Without this control, the headline claims of "superiority" on time-varying systems (Abstract, Section 5.1, Section 5.2) are not convincingly attributed to the proposed method.

- **Diagonal $\bar{A}$ constraint is a core architectural restriction that goes unexamined**: The matrix $A$ is constrained to be diagonal (Section 3.2: "The matrix $A$ is set as a diagonal matrix, facilitating the discretization process"), meaning each lifted state component evolves independently ($z_{k+1}^{(i)}$ depends only on $z_k^{(i)}$, not on other components). This is a significant restriction on the expressiveness of the linear dynamics in the lifted space. Standard Koopman frameworks use full matrices where cross-coupling between lifted states is possible. While the observable function $\psi$ and time-varying $C_k$ provide some coupling, the paper provides no ablation comparing diagonal vs. full $A$ within the MamKO framework, nor any theoretical analysis of how much expressiveness is lost. The justification ("facilitating the discretization process") is incomplete — ZOH discretization of full matrices is standard and straightforward. This matters because if a full $A$ significantly improves performance, the diagonal constraint becomes a liability rather than a feature, and the comparison with DKO (which likely uses full $A$) confounds diagonal-vs-full with time-varying-vs-constant.

### Minor

- **Time-varying evaluation for CartPole covers only sinusoidal parameter variations**: The time-varying CartPole adds sinusoidal friction variations at known frequencies (0.1, 1, 10 rad/s, Section 5.1). Sinusoidal variations are the most predictable class of time variation — they are periodic and thus highly amenable to history-based prediction. The time-varying RSCP system (referencing Nikravesh et al., 2000) may provide some diversity, but the nature of its time variation is not described in the main text. Whether MamKO handles non-periodic, abrupt, or stochastic parameter changes remains unknown. This is a scope limitation on the generality of the central claim.

- **Abstract overclaims "integrating large language models with control frameworks"**: The abstract states the approach "unlocks new possibilities for integrating large language models with control frameworks." What is actually integrated is Mamba's matrix generation mechanism — a small neural network producing SSM parameters from inputs. No LLM capabilities (pre-training, language understanding, scale, attention) are leveraged. The conclusion similarly claims "seamlessly integrating the Koopman operator with the large language model Mamba." While Mamba is an LLM architecture, this phrasing overstates the connection.

- **Time-varying $C_k$ departs from standard Koopman framework without discussion**: In standard Koopman theory, the observable function $\psi$ (and hence the decoder $C$) is fixed, giving the lifted state $z$ a consistent physical meaning across time. Having $C_k$ vary (Eq. 5) means the interpretation of the lifted state changes over time, which is a notable departure from the Koopman framework's conceptual structure. The paper does not discuss this choice or its implications.

- **MPC weight tuning per method reduces comparison fairness**: The paper states "The weighting parameters $Q$, $R$, and $P$ for the MPCs are carefully adjusted to reach the best performance of each baseline method" (Section 5.2). Separate tuning per method can introduce a favorable bias toward whichever method benefits most from tuning effort. Reporting results under common weights or across multiple weight settings would strengthen the control comparison.

### Trivial

- The one-step lag between when matrices are generated (from data up to $k-1$) and when they are applied (at time $k$) is not discussed, though it is implicitly handled by the experiments showing good performance even at 10 rad/s.

## Nice-to-Haves

- A history-augmented MLP baseline receiving the same temporal context as MamKO would directly address the comparison fairness concern and is the single most impactful addition the authors could make.
- An ablation comparing diagonal vs. full $\bar{A}$ within the MamKO framework would clarify the expressiveness cost of the diagonal constraint.
- Reporting parameter counts for all methods would clarify whether MamKO's improvements stem from architectural innovation or model capacity.
- Testing on non-sinusoidal time variations (step changes, random drift) would strengthen claims of general applicability.
- Comparing against an online-updating Koopman method (e.g., Hao et al., 2022 or Liu et al., 2023, both cited in Related Work) would assess whether the generative approach truly outperforms existing time-varying Koopman methods — the current baselines all use time-invariant operator structures.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Critic's CELU instability claim (incorrect)**: The critic argued that $\bar{A} = -\text{CELU}(A)$ "produces eigenvalues in $(-\infty, 1]$, meaning eigenvalues above 1 (which indicate discrete-time instability via exponential growth) are impossible." This confuses continuous-time eigenvalues of $\tilde{A}$ with discrete-time eigenvalues of $\bar{A} = e^{\tilde{A}T}$. For $\tilde{A}$ eigenvalue $\lambda \in (0, 1)$, the ZOH discretization gives $\bar{A}$ eigenvalue $e^{\lambda T} > 1$, so discrete-time instability IS possible. The paper's claim that CELU "facilitates the representation of unstable systems" is correct.

- **Computation time ambiguity (speculative)**: The critic speculated that Table 2 might only report QP solve time, missing the matrix generation overhead. The paper states these are "computation times of the MamKO-based MPC and baselines during control," which naturally includes all online computation steps. The concern about MamKO's 10.2ms on CartPole vs. 20ms sampling period is worth noting but the margin appears adequate (2x headroom).

- **Parameter count as a major issue**: The critic treated missing parameter counts as evidence that improvements might come from larger model capacity. While parameter counts would be informative, the claim that improvements on time-invariant systems "may simply reflect MamKO's larger parameter count" is speculative — the observable function architecture and matrix generation mechanism provide a qualitatively different inductive bias, not just more parameters.

- **Missing related works**: Per rules, criticisms about missing related works are removed.

- **Formatting and style issues**: Per rules, formatting nitpicks are removed.

## Novel Insights

The most insightful observation across the reviews is that the paper's core contribution is not just "using history to improve prediction" but specifically the architectural move of generating operator matrices from history while avoiding input-dependent operators (which would create bilinear terms). This distinction — conditioning on $[z_{k-H:k-1}, u_{k-H:k-1}]$ rather than on $u_k$ — is what preserves convexity and is the paper's most elegant design choice. However, the experimental evaluation does not disentangle this architectural contribution from the simpler effect of having access to historical data, which is the central gap in the paper's evidence.

## Suggestions

- Add a history-augmented MLP baseline that receives $[x_{k-H:k-1}, u_{k-H:k-1}]$ as input — this is the single most important experiment to validate that the generative architecture provides value beyond simply having temporal context.
- Run a diagonal vs. full $\bar{A}$ ablation within MamKO to quantify the expressiveness cost of the diagonal constraint.
- Moderate the "integrating large language models" language to more precisely describe what is borrowed from Mamba (the selection/matrix-generation mechanism, not LLM-scale capabilities).
- Add at least one non-sinusoidal time variation experiment (e.g., step changes in parameters) to test generalization beyond periodic variations.

## Score and Decision

**Calibration anchors used:**

- `/home/wg25r/review_agent/human_reviews/fkrYDQaHOJ.md` — Koopman in interactive environments, diagonal A, poster, avg 5.5. Directly comparable: similar diagonal Koopman approach, similar concerns about comparison fairness and theoretical looseness, but accepted with mixed reviews (3, 5, 6, 8). MamKO has a more novel time-varying mechanism but weaker experimental controls.
- `/home/wg25r/review_agent/human_reviews/yIdCQFvbYe.md` — Bayesian Adaptive Koopman (BLAK), avg 5.83, Rejected. Similar paper addressing time-varying Koopman operators with online adaptation. MamKO is comparable in novelty but has confounded comparisons on its main claim.
- `/home/wg25r/review_agent/human_reviews/UvTo3tVBk2.md` — Unlocking State-Tracking in LRNNs via Negative Eigenvalues, avg 8.0, Oral. Also modifies Mamba's eigenvalue constraints (negative values), but with rigorous theoretical backing. MamKO has a similar modification (CELU) but without theoretical depth.
- `/home/wg25r/review_agent/human_reviews/vOfDGYGVyj.md` — Sparse Mamba with controllability, avg 2.5, Withdrawn. Poorly executed Mamba adaptation. MamKO is clearly superior in execution and empirical validation.
- `/home/wg25r/review_agent/human_reviews/53gU1BASrd.md` — Financial time series with unfair baselines, avg 4.5, Rejected. Shares confounded-comparison weakness with MamKO.

MamKO sits between the rejected BLAK (5.83, with similar comparison fairness issues) and the accepted diagonal Koopman paper (5.5, with similar methodological concerns but better-controlled experiments). The core idea is sound and well-motivated, but the experimental validation for the central claim (time-varying system superiority) is confounded, and the diagonal A constraint is unexamined. These are not fatal flaws but they substantially weaken the evidence for the paper's main contribution. Below the accepted Koopman anchor (5.5) because that paper had more controlled comparisons; above the unfair-baseline papers (~4.5) because MamKO's core idea is more novel and the MPC formulation is a genuine contribution.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>