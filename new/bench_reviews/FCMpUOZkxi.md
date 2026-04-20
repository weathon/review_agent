Now let me do calibration searches to properly score the paper.Now I have enough information to write the final consolidated review.

---

## Summary

This paper proposes AUPD (Adaptive and Universal Primal-Dual), a single-stage algorithm for stochastic contextual bandits with knapsacks (CBwK) in the small budget regime $B = \Omega(\sqrt{T})$. AUPD achieves two results: (1) $\tilde{O}((1 + \nu^*/\delta b)\sqrt{T})$ regret under strict feasibility, matching the best-known bounds without requiring prior knowledge of the safety margin $\delta$ or a doubling trick; and (2) $\tilde{O}(\sqrt{T} + (\nu^*/\sqrt{b})T^{3/4})$ regret without strict feasibility — the first such result in the literature. The key technical contribution is a budget-aware virtual queue design with $V = b\sqrt{T}$, analyzed via a Lyapunov drift framework that avoids the two-stage structure of all prior work.

---

## Strengths

- **First regret guarantee for CBwK without strict feasibility assumption** (Theorem 1, first result): The paper achieves $\tilde{O}(\sqrt{T} + (\nu^*/\sqrt{b})T^{3/4})$ regret without Assumption 3, a result that — as the authors note and Table 1 confirms — has no prior analog even in the large-budget regime $B = \Omega(T)$. This is a genuine advance.

- **Tight bound under strict feasibility without prior knowledge** (Theorem 1, second result; Remark 2): AUPD achieves $\tilde{O}((1 + \nu^*/\delta b)\sqrt{T})$, matching the lower bound from Chzhen et al. (2024), without requiring knowledge of the safety margin $\delta b$ and without a doubling trick. Prior work either required known $\delta$ (Chzhen et al., 2024) or $B = \Omega(T^{3/4})$ (Han et al., 2023; Agrawal & Devanur, 2016).

- **Novel Lyapunov drift analytical framework** (Lemma 2 and Section 5): The paper introduces a per-step regret–Lyapunov drift bridge (Lemma 2) that is independent of strict feasibility. This unifies analysis of both regret-before-stopping and regret-after-stopping under a single framework, and yields meaningfully different virtual queue bounds depending on whether Assumption 3 holds (Lemma 4: $O(\sqrt{KVT})$ vs. $O(\sqrt{KV/\delta b})$), directly driving the two regret regimes.

- **Single-stage, fully adaptive algorithm** (Algorithm 1, Remark 1): The budget-aware parameter $V = b\sqrt{T}$ encodes budget scarcity directly into decision-making without an explicit search phase or conservative dual-update bounds, distinguishing AUPD from all prior approaches.

- **Experimental validation in small-budget regime** (Figure 1a): AUPD substantially outperforms SquareCBwK and PGD Adaptive when $B = \Theta(\sqrt{T})$, the regime where competing algorithms theoretically degrade.

---

## Weaknesses

### Fatal
None.

### Major

- **Internal inconsistency in Table 1 regarding Han et al. (2023) budget requirement**: Table 1 lists SquareCBwK (Han et al., 2023) with a budget requirement of $\Omega(T^{1/2})$. However, the Related Work section (Section 1) states explicitly: *"the work Han et al. (2023) still assumes $B = \Omega(T^{3/4})$ as it also utilizes the two-stage method."* The Introduction also contains both numbers — lines 30–31 claim $\Omega(T^{1/2})$ while the Related Work later says $\Omega(T^{3/4})$. The distinction (minimum budget to not terminate early vs. minimum budget for the regret to hold) is never made explicit. This inconsistency directly affects how AUPD's contribution is positioned: the regime in which AUPD genuinely advances over Han et al. depends on whether Han et al. needs $T^{1/2}$ or $T^{3/4}$. This needs resolution.

- **Experiments do not measure cumulative regret**: The entire experimental section reports "Averaged Reward" on the y-axis (Figure 1). While this shows AUPD depletes budget more slowly (staying active longer), it does not validate the paper's central theoretical claims — specifically the $T^{3/4}$ vs. $\sqrt{T}$ regret scaling. An algorithm that makes poor per-round decisions but conserves budget will appear favorable in an averaged-reward plot. The $T^{3/4}$ vs. $\sqrt{T}$ distinction, which is AUPD's theoretical headline, is invisible in Figure 1. Cumulative regret curves are standard in bandit papers and needed to support the theorems.

### Minor

- **Proof sketch in Section 5.3 elides non-trivial steps**: After deriving the intermediate bound under strict feasibility ($\tilde{O}(\sqrt{KV\nu^*/\delta b^2} + \sqrt{T}\nu^*/b + \sqrt{T} + K(Tb+Tb^2)/V)$), the paper states "Let $V = b\sqrt{T}$ and we prove Theorem 1" without showing the reduction. The first term after substitution is $(K\nu^*/\delta b)^{1/2} T^{1/4}$, which requires $\nu^*/\delta b = \Omega(K/\sqrt{T})$ to be absorbed into $(\nu^*/\delta b)\sqrt{T}$. This condition is not stated. While the full proof is in the appendix, this gap in the main paper's sketch is non-trivial enough to deserve a sentence of justification.

- **Lemma 5 notation $\sqrt{KV/\delta b}^2$ is ambiguous**: It is unclear whether $^2$ applies to the $\sqrt{\cdot}$, giving $KV/\delta b$, or is a formatting artifact. The interpretation changes the bound. The intended form should be stated clearly.

- **Discrepancy between theory and experiment for cost estimation**: The theoretical framework (Assumption 2) requires general online learning oracles for cost estimation under contextual inputs. The experiments use "empirical mean" for costs, which ignores context-dependence. This discrepancy is not acknowledged.

### Trivial
None.

---

## Nice-to-Haves

- **Optimality of $T^{3/4}$**: Remark 2 acknowledges the lower bound for the no-strict-feasibility case is open. Even a heuristic hard-instance argument suggesting $T^{3/4}$ is unavoidable would substantially strengthen the contribution and clarify whether this is a tight bound or a proof artifact.

- **Formal corollary for the "typical" setting**: Remark 1 notes that when $T\nu^* = \Theta(B)$, AUPD recovers $\tilde{O}(\sqrt{T})$. A short corollary would make this more accessible to readers.

- **Regret-vs.-$T$ curves on synthetic instances**: A synthetic experiment where the strict feasibility condition is exactly controlled (e.g., $\delta = 0$ vs. $\delta > 0$) would directly demonstrate the two theoretical regimes and validate the scaling claims in a way Figure 1 cannot.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Table 1 shows $T^{1/4}$ for AUPD without strict feasibility (Harsh Critic)**: Table 1 line 44 reads $\tilde{O}(\sqrt{T} + \frac{\nu^*}{\sqrt{b}}T^{1/4})$ while Theorem 1 and the abstract consistently state $T^{3/4}$. Under the hard rule that parsing artifacts should not be held against authors, this is likely a PDF-rendering error (misrendering $\frac{3}{4}$ as $\frac{1}{4}$). The actual claim is unambiguously $T^{3/4}$ in both the abstract and Theorem 1. Removed as a likely parser artifact rather than an author error.

- **Reproducibility/hyperparameter concerns**: The harsh critic mentions hyperparameter undisclosure; these are standard reproducibility nitpicks removed per policy.

- **Generic "larger experiments" request**: Requests for additional datasets or larger scale are generic and do not target a specific methodological gap in the paper's claims.

---

## Novel Insights

The paper's most important observation — that omitting the Slater's condition forces an additional $T^{3/4}$ penalty in the regret, which no prior CBwK work had even attempted to characterize — opens a new axis of hardness in the CBwK literature. The Lyapunov drift perspective is technically interesting: by treating the virtual queue as a Markov chain and deriving its stationary behavior, the authors sidestep the explicit dual-variable learning that required prior works to impose either the strict feasibility assumption or a two-stage structure. The quantitative gap between $O(\sqrt{KV/\delta b})$ (stable queue, with feasibility) and $O(\sqrt{KVT})$ (growing queue, without feasibility) in Lemma 4 directly explains — rather than just upper-bounds — why strict feasibility matters for regret. The paper does not, however, establish whether $T^{3/4}$ is optimal in the absence of strict feasibility, leaving open a fundamental question about the problem's intrinsic difficulty.

---

## Suggestions

1. **Resolve the Han et al. budget inconsistency**: Clarify in Table 1 and Section 1 whether the listed budget requirement is the minimum for the algorithm to not terminate prematurely, or the minimum for the regret bound to hold. State explicitly that Han et al. needs $B = \Omega(T^{3/4})$ for regret purposes, and update Table 1 accordingly.
2. **Add cumulative regret plots**: Replace or supplement Figure 1's "Averaged Reward" metric with plots of cumulative regret vs. $T$. Use synthetic data where the strict feasibility condition is controlled, so the two theoretical regimes can be empirically distinguished.
3. **Explicit intermediate derivation in Section 5.3**: Add one or two sentences showing how $(K\nu^*/\delta b)^{1/2} T^{1/4}$ is absorbed into $(\nu^*/\delta b)\sqrt{T}$ after substituting $V = b\sqrt{T}$, or state the mild parameter condition under which this holds.
4. **Clarify $\sqrt{KV/\delta b}^2$ notation in Lemma 5**: Write this as $KV/\delta b$ explicitly to remove ambiguity.

---

## Score and Decision

**Calibration anchors consulted:**

| Paper | Topic | Avg Score | Decision |
|---|---|---|---|
| `ilbxbOHk7a.md` | Constrained MDPs with bandit feedback; heavily combines prior work | ~3.6 | Reject |
| `of25Zg4AdM.md` | Contextual bandits with budget constraints; limited novelty, relies on Agrawal & Devanur | ~4.25 | Reject |
| `z0B7A6Dh1H.md` | Contextual bandits with stage-wise constraints; genuine novel angle (high-prob safety) | ~6.0 | Reject |
| `h6ktwCPYxE.md` | Contextual bandits with function approximation; solid theory, some gaps | ~6.0 | Accept (Poster) |
| `5t57omGVMw.md` | Online learning with regret bounds; very clean, tight analysis, all 8s | ~8.0 | Accept (Spotlight) |

**Positioning**: The paper under review is clearly above `ilbxbOHk7a.md` and `of25Zg4AdM.md` — it makes a genuine, previously-unknown theoretical contribution (first CBwK result without strict feasibility; tight result matching lower bound under strict feasibility), introduces a novel analytical framework (Lyapunov drift for virtual queue stability), and presents a clean algorithm. This puts it in the range of `z0B7A6Dh1H.md` and `h6ktwCPYxE.md` (~6.0 avg). The paper falls short of the ~8.0 tier (e.g., `5t57omGVMw.md`) primarily because of: the unresolved Table 1 inconsistency about Han et al.'s budget requirement (a genuine presentation problem that affects the positioning of contributions), and experiments that cannot empirically validate the key theoretical claims. These are meaningful but correctable issues. The core theory is sound, the contributions are real, and the result is the first of its kind in an established and well-studied problem.

**Final score: 6.0**

**Originality**: High — first result without strict feasibility; novel Lyapunov drift analysis avoiding two-stage structure.
**Importance**: High — addresses a fundamental assumption gap in a well-studied problem.
**Claim support**: Moderate — theory is rigorous, but experimental evaluation does not directly support the scaling claims.
**Experimental soundness**: Weak — averaged reward metric cannot validate regret-rate claims.
**Clarity**: Good — algorithm and proof structure are well-explained; Table 1 inconsistency is a notable exception.
**Community value**: Moderate-high — opens a new direction (CBwK without strict feasibility) and closes a gap in the existing rate landscape.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>