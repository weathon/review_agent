Now I have sufficient calibration data to write the final review. Let me compose it.

---

## Summary

FedMPDD encodes each FL client's gradient as $m$ scalar inner products with random Rademacher vectors (directional derivatives), transmitting only those $m$ scalars plus a random seed per round. The server reconstructs a gradient estimate via the same seed. This simultaneously reduces uplink communication from $O(d)$ to $O(m)$ and exploits the rank-$(d-m)$ nullspace of the projection to impede gradient inversion. Theorem 2 proves $O(1/\sqrt{K})$ convergence (matching FedSGD) when $m = O(\ln(d/\delta)/\epsilon^2)$, and Lemmas 1–2 bound the adversary's gradient and data reconstruction error under this scheme.

---

## Strengths

- **Clean, novel mechanism with a direct convergence fix.** The paper correctly identifies that a single-projection FedPDD suffers from a $\sqrt{d}$ variance blow-up leading to $O(d/\sqrt{K})$ convergence, and rigorously shows that averaging $m = O(\log d)$ projections via the JL lemma restores $O(1/\sqrt{K})$. The narrative arc from FedPDD to FedMPDD is coherent and the convergence proof is technically sound.

- **Magnitude-independent privacy bound.** Lemma 1 establishes a relative reconstruction error of $(d-1)/m$ independent of gradient norm, which is a genuine property absent in additive-noise methods where protection degrades as gradients shrink.

- **Large, concrete communication savings backed by correct arithmetic.** Table 2 shows FedMPDD ($m=600$, 0.2% of $d$) achieving a 356× total-byte reduction vs. FedSGD to reach target accuracy, and Figure 3's accuracy-vs.-bits curve consistently demonstrates per-bit efficiency advantages over QSGD, Top-k, lp-proj, and SA-FedLora.

- **Dual GIA evaluation.** The paper evaluates privacy under both the recent Yu et al. (2025) attack and the classical DLG attack (Zhu et al., 2019) across multiple architectures, providing broader empirical coverage than typical single-attack evaluations.

- **Practical seed trick.** Transmitting only the random seed rather than the $d$-dimensional projection vectors — letting the server regenerate them — is an efficient and correctly implemented design choice that eliminates the overhead of vector transmission entirely.

---

## Weaknesses

### Fatal
None.

### Major

- **Abstract states O(1/K) convergence; the paper proves O(1/√K).** The abstract reads: *"establishing that FedMPDD converges at a rate of O(1/K), matching the performance of FedSGD."* However, the contributions bullet on the same page correctly says O(1/√K), and Theorem 2 (Eq. 5) proves O(1/√K) for smooth non-convex objectives. O(1/K) is the rate for strongly convex problems, which FedMPDD does not address. FedSGD itself converges at O(1/√K) on non-convex problems, so the "match" is only accurate if the claim is O(1/√K). This misstatement in the abstract — the first thing readers and reviewers evaluate — is a meaningful inconsistency that must be corrected.

- **Privacy superiority over LDP is asserted but not formally established.** The paper repeatedly claims FedMPDD provides "stronger" or "more consistent" privacy than LDP (e.g., Section 2, paragraph after Lemma 2: *"Achieving consistent privacy with LDP would require large, performance-degrading noise values"*). However, the paper's privacy guarantee is *geometric*, not information-theoretic: (a) Lemma 2's lower bound depends on $L_v(\mathbf{x})$, the Lipschitz constant of the gradient w.r.t. the input, which is uncontrolled and potentially arbitrarily large for deep networks; (b) Remark 2's multi-round bound ($T \times m < d$) rules out only *exact* algebraic reconstruction, not approximate reconstruction by a statistically sophisticated adversary; (c) no (ε, δ)-DP guarantee is derived for FedMPDD. LDP provides (ε, δ)-DP — information-theoretically composable, adversary-agnostic, input-distribution-agnostic. FedMPDD provides a geometric reconstruction-error bound valid for one class of loss-minimization attacks. These are incommensurable notions; claiming one is superior to the other is not justified by the existing theory. The SSIM evidence in Figure 2 is empirically valuable but measures a specific attack, not all possible adversaries.

- **No baseline combining compression with formal privacy.** The paper's stated goal is *joint* communication efficiency and privacy. Yet cpSGD (Agarwal et al., 2018) — which combines DP with compressed SGD and is explicitly cited in Related Work — is never included in experiments. Amiri et al. (2021), also cited, combines Gaussian DP noise with gradient quantization. Without at least one properly tuned joint compression+privacy baseline, the paper cannot substantiate that its *combination* outperforms prior joint approaches; the two benefits are only compared in isolation.

### Minor

- **FedAvg not compared.** All convergence and accuracy experiments use FedSGD (single local step per round) as the baseline, but FedAvg (multiple local steps) is the dominant practical FL algorithm and is substantially more communication-efficient per unit computation. The paper's compression claims should be evaluated against FedAvg to demonstrate practical relevance.

- **Theorem 2 not concretely linked to experimental configurations.** The convergence bound (Eq. 5) includes a residual term $O(\epsilon G^2/\sqrt{K})$ controlled by the JL distortion parameter $\epsilon$, which depends on the number of projections $m = O(\ln(d/\delta)/\epsilon^2)$. For the experimental choices (e.g., $m=400$ for a $d \approx 20{,}000$ MNIST/LeNet model), the implied $\epsilon$ is not computed or reported, so it is unclear how large the residual term is or whether the $O(1/\sqrt{K})$ decay dominates in practice.

### Trivial

- Figure 2 caption references "FedSGD + Lag (m=X)" — "Lag" appears to abbreviate "Laplace" noise but is not defined in the caption or the surrounding figure legend. The parameterization in the line plot (e.g., "Ours (m=1.0)") also conflicts with the image grid's "FedMPDD (m=0.01, 0.001)," suggesting these may be relative ratios $m/d$ vs. absolute values. This notation should be harmonized.

---

## Nice-to-Haves

- Derive the best achievable (ε, δ)-DP guarantee for FedMPDD as a function of $m$, $d$, $T$, and compare it numerically to the LDP noise level achieving equivalent DP. This would transform the LDP comparison from informal to rigorous.
- Show SSIM as a function of training rounds past the $T \times m = d$ threshold to empirically quantify multi-round privacy degradation.
- Provide accuracy-vs.-bits curves for all baselines in Table 2 (not just endpoint measurements), to reveal whether FedMPDD's per-bit advantage holds throughout training.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"JVP approach deferred to appendix" as a flaw.** The harsh critic argues the main algorithm computes the full gradient, creating a gap with the computational claim. However, Remark 1 (lines 124–125) explicitly addresses this: it explains the JVP approach, states when it reduces computation ($m < hpT/(h+p)$), reports empirical timing in Table A.10, and defers the full study to Appendix F. The main algorithm is presented with the full gradient for generality; the computational optimization is a bonus. Flagged as a strawman — the paper addresses this directly.

- **"Multi-round privacy model is brittle."** The paper explicitly describes the $T \times m < d$ bound as a "worst-case" guarantee and notes that gradient evolution provides stronger practical protection. Criticizing it as "brittle" because it differs from DP composition is criticizing the paper for not being a DP paper, which is scope creep.

- **"Sketched updates are often biased" overreaches.** The harsh critic flags that some sketching methods are also unbiased. This is a minor imprecision that does not affect any proof or experimental result.

- **QSGD performance is suspiciously low.** The critic suggests QSGD may be misconfigured. Under a 0.09 GB budget, 8-bit QSGD is only ~4× smaller than FedSGD (float32), so both are budget-starved; FedMPDD at 0.2% of $d$ is ~500× smaller. The performance gap is consistent with this arithmetic and does not indicate misconfiguration.

---

## Novel Insights

The core insight that projecting gradients onto $m$ random Rademacher vectors provides a *magnitude-independent* relative reconstruction error of $(d-1)/m$ — contrasted with additive-noise methods whose privacy protection degrades when gradients are large — is a genuinely novel geometric observation for the FL setting. The paper also clearly articulates why the single-projection variant (FedPDD) fails due to $\sqrt{d}$ variance scaling and why averaging $m = O(\log d)$ projections resolves this at essentially no cost in convergence rate. These are clean, verifiable, and reproducible insights. The claim that this geometric protection is *formally superior* to LDP is where the paper overreaches, but the underlying mechanism is original and practically useful.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Human Score | Comparison |
|---|---|---|
| `jj5ZjZsWJe.md` (SCALLION/SCAFCOM) | 8.0 | FL + compression, tight rates under heterogeneity, multiple strong baselines — far more rigorous than FedMPDD |
| `PpYy0dR3Qw.md` (LoCoDL) | 7.5 | FL local training + compression, doubly accelerated — stronger theoretical contribution |
| `TCJbcjS0c2.md` (LASER) | 5.83 | Low-rank gradient compression, wireless, rejected at borderline — comparable scope but cleaner analysis |
| `ogIFNo2bQw.md` (BiCompFL) | 4.80 | FL bi-directional compression, missing convergence analysis, rejected — similar strength to FedMPDD |
| `ER1VDuwWvB.md` (CORE) | 3.67 | Random projection compression, flawed DP comparison, unsubstantiated claims, rejected — similar approach, weaker execution |
| `WYEEWScbaM.md` (Gradient Distillation FL) | 3.0 | Communication-efficient FL, gradient distillation, weak theoretical grounding, rejected — weaker than FedMPDD |

FedMPDD is clearly stronger than CORE (3.67) and Gradient Distillation (3.0): the algorithm is cleaner, the convergence analysis is correct, and the empirical study is more thorough. It is comparable to BiCompFL (4.80), which was rejected for missing convergence analysis. FedMPDD has convergence analysis but has the abstract O(1/K) error, an overreaching LDP superiority claim, and a missing joint-baseline comparison — all of which are real but correctable weaknesses. It falls short of LASER (5.83), which despite borderline rejection had a technically tighter analysis. Anchoring between BiCompFL (4.80) and LASER (5.83), with the abstract error and LDP comparison issue pulling the score down, I place FedMPDD at **5.0** — a weak reject that could be addressed with targeted revisions.

**Overall axis assessment:**
- *Originality:* Medium-high. The projected directional derivative as a FL primitive is novel; the multi-projection averaging to achieve JL-based convergence is elegant.
- *Importance of research question:* High. Joint communication efficiency and privacy in FL is a real and pressing problem.
- *Claims vs. support:* Weak on privacy. The abstract convergence claim is incorrect (O(1/K) vs O(1/√K)), and the "superior to LDP" privacy claim is formally unsubstantiated.
- *Soundness of experiments:* Moderate. Communication efficiency experiments are solid; the privacy comparison lacks a joint baseline.
- *Clarity:* Good overall, with minor notation inconsistencies.
- *Value to community:* Moderate. The mechanism is useful and the empirical results are concrete; but the theoretical privacy framing needs significant tightening before the paper can claim its full headline contribution.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>