Now I have a thorough understanding of the paper. Let me run calibration searches.Now I have enough information for calibration. Let me write the final consolidated review.

---

## Summary

This paper introduces **Markovian compressors** for distributed optimization — a novel class of sparsification operators where coordinate selection follows a Markov chain rather than being i.i.d. Two concrete examples are developed: **BanLast(K,m)**, which prohibits re-transmitting coordinates used in the last K iterations, and **KAWASAKI(K,b,π_Δ,m)**, a softer probabilistic generalization. The compressors are embedded into QSGD (MQSGD, Algorithm 1) and an accelerated variant (AMQSGD, Algorithm 2), with convergence analysis under non-convex/PL and strongly convex settings. Experiments across logistic regression and ResNet-18/CIFAR-10 training show consistent empirical gains over Rand sparsification.

---

## Strengths

- **Novel compressor family with formal ergodicity characterization (Theorem 1, Definitions 5–6).** BanLast and KAWASAKI are provably ergodic Markov chains with uniform stationary distributions. Theorem 1 provides explicit, quantitative formulas for the contraction rate ρ and constant C, rather than mere existence results. This is genuine groundwork that validates the core assumption (Assumption 5) used in all convergence proofs.

- **New "stepping back" proof technique (Equations 4–5).** The paper develops a non-trivial analytical tool to handle Markovian bias by relating step-t quantities to step-(t−τ) quantities via ε-approximate unbiasedness, enabling convergence theorems for a class of compressors where standard unbiasedness arguments fail. This technique appears original and could have broader use.

- **Genuine momentum acceleration relative to the base Markovian method.** Comparing Corollary 1 (PL/non-convex MQSGD: linear dependence on L/μ) and Corollary 2 (strongly convex AMQSGD: (L/μ)^{2/3} dependence), the accelerated method provides a provable improvement in condition number dependence over the unaccelerated Markovian baseline. This is the acceleration the paper actually delivers.

- **Exemplary intellectual honesty in Section 2.4.** The three-bullet discussion explicitly identifies and explains each gap between the theory and existing baselines (d/m vs d²/m², mixing-time paradox, (L/μ)^{2/3} vs √(L/μ)), attributing each to structural constraints of Markovian analysis shared by the broader literature. This level of candor is uncommon and increases confidence in the work's integrity.

- **Consistent empirical improvements across multiple settings.** Figures 1 and 3, and Table 1 show gains across MQSGD, AMQSGD, DIANA, and SGD with momentum on logistic regression and neural network benchmarks. Table 1 in particular reports mean ± std over 5 seeds: KAWASAKI achieves 89.05±0.29% vs. Rand's 87.9±0.18% test accuracy on CIFAR-10 with ResNet-18.

- **Compatibility with existing compressors.** Figure 3 demonstrates that BanLast and KAWASAKI can be composed with Natural compression, making the proposed compressors modular additions to existing pipelines.

---

## Weaknesses

### Fatal
None.

### Major

- **Convergence theory establishes strictly worse rates than the Rand baseline in every quantitative dimension.** Section 2.4 and Theorem 2 are explicit: comparing against vanilla Rand+QSGD (Beznosikov et al., 2023a), MQSGD degrades the contraction constant by 12×, grows the noise term by an extra (d/m)·τ factor (from d/m to d²/m²·τ), and shrinks the admissible step size by m²/(d²τ). The paper openly acknowledges this, yet the title ("Accelerate the Future"), abstract ("demonstrate superiority"), and introductory claims do not clearly communicate that the proposed algorithms are provably slower than the simplest baseline in every theoretical metric. This gap between the framing and the content is the central tension of the paper. The Markovian stochasticity literature does impose these constraints on all authors in the area (the paper correctly attributes them to the literature), but that observation makes the contribution primarily empirical, while the paper presents itself as a theoretical-plus-empirical advance over baselines.

- **Empirical comparison uses best-run selection for the proposed methods without comparable validation.** Figure 1 caption: "All hyperparameters are fine-tuned, and best runs are selected." Figure 2 caption: "Best runs for each method are displayed." Markovian compressors (BanLast, KAWASAKI) carry extra hyperparameters (K for BanLast; K, b, π_Δ for KAWASAKI) that were tuned, while the Rand baseline has essentially no compressor-specific parameters. Only Table 1 reports mean ± std. The main training-curve figures should show averages over seeds to credibly support the claimed superiority, especially since the gaps in Figure 1 are substantial and seed-selection effects are plausible.

### Minor

- **BanLast shows almost no empirical improvement over Rand despite identical theoretical motivation.** Table 1: BanLast achieves 88.0±0.12% vs. Rand 87.9±0.18% — an overlap within one standard deviation. Meanwhile KAWASAKI achieves 89.05±0.29%. The paper briefly attributes this to KAWASAKI's "smoother history accumulation" being needed for complex tasks, but no ablation or analysis is provided to support this claim. Since BanLast is presented as the primary intuitive example, this inconsistency weakens the case.

- **The theory-practice contradiction regarding K is unresolved.** Section 2.4 explicitly notes that larger K worsens the theoretical mixing time τ but improves empirical performance. The paper treats this as a "logical contradiction" and leaves it open. An ablation varying K would help clarify when and why larger K helps empirically, either confirming or refuting the theory's qualitative predictions.

- **Example 1's "3x speedup" does not transfer cleanly to global convergence.** The example computes expected steps to exit a single sparse-gradient point, which is an informative but narrow motivator. Real gradients in neural networks are not coordinate-sparse in the same structured way, and the paper does not connect this argument to the actual convergence behavior analyzed in Theorems 2–3.

### Trivial

None (parser artifacts are excluded per policy).

---

## Nice-to-Haves

- An extension of MQSGD/AMQSGD to variance-reduction settings (DIANA, MARINA) with theoretical guarantees would substantially increase practical impact, since without variance reduction the algorithms converge only to a σ²-neighborhood of the solution. The paper acknowledges this as future work.
- Reporting mean ± confidence intervals for Figures 1 and 3 (not just Table 1) would make the empirical claims fully credible.
- A formal comparison or relationship between BanLast and PermK (Szlendak et al., 2021) would clarify when BanLast offers something genuinely distinct, since both are cycling/exclusion sparsification schemes.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **KAWASAKI definition garbled/ambiguous (Harsh Critic).** The formula in Definition 6 with "t^# of choices j" is a PDF parser artifact. The text surrounding the definition — including explicit reference to b as the "forgetting rate" and b > 1 — is consistent with a clean formula in the original. Per policy, parser artifacts are not author errors. REMOVED.

- **"Strongly convex and non-convex cases are rarely present in the field" is an unjustifiable claim (Harsh Critic).** This was flagged as a claim that may irritate reviewers. However, it is a minor framing issue, not a methodological flaw. REMOVED as trivial claim rather than a substantive weakness.

- **Claim about reproduced KAWASAKI formula being ambiguous (Harsh Critic).** Tied to the parser artifact issue above. REMOVED.

- **Generic strength: "important problem" (Strength Finder).** Dropped as non-specific.

- **Strength about AMQSGD proving acceleration in the strongest sense (Strength Finder).** The Strength Finder claimed "momentum acceleration in the strongly convex case" as a strength vs. standard Nesterov. While true relative to MQSGD, AMQSGD achieves only (L/μ)^{2/3} vs. Nesterov's √(L/μ) — a verified weaker rate. This strength has been contextualized and partially retained above with appropriate qualification.

---

## Novel Insights

The most genuinely novel observation — surfaced across both reviewers but not fully developed in the paper — is that there may exist a **structural incompatibility between memory-based compressors and classical SGD convergence analysis**. The d/m → d²/m² degradation and the τ-penalty in Theorem 2 arise because Markovian stochasticity forces a uniform bound on compressor noise in place of a variance bound. This is not a failing of the specific analysis technique: as the paper notes, *every existing paper on Markovian stochasticity* (Beznosikov 2023b, Dorfman & Levy 2023, Doan 2020a) carries the same constraint. This suggests that the theory-practice gap observed for BanLast/KAWASAKI may be inherent to the entire class of Markovian compressors at current proof-technique resolution, not a gap that better analysis of *these specific compressors* would close. The practical success of the compressors may thus depend on properties of real gradient sequences (e.g., temporal redundancy) that the worst-case theory cannot capture.

---

## Calibration

**Anchors used:**

| Paper | Path | Avg Human Score | Comparison |
|---|---|---|---|
| CORE (low) | `ER1VDuwWvB.md` | 3.67 (Reject) | Distributed compression paper with unsupported claims and weak theory; this paper is more honest and more conceptually novel |
| BiCompFL (low-medium) | `ogIFNo2bQw.md` | 4.80 (Reject) | Communication compression FL paper; comparable scope and theoretical weakness |
| LASER (medium) | `TCJbcjS0c2.md` | 5.83 (Reject) | Novel compression scheme for wireless; achieves rates *comparable* to SGD baseline (stronger theory) but rejected for novelty concerns. This paper has more novel concept but strictly weaker theory than baseline |
| MoTEF (high) | `CMMpcs9prj.md` | 6.6 (Accept) | Decentralized optimization with compression; achieves rates *matching* distributed SGD — the theoretical bar this paper does not reach |
| LoCoDL (high) | `PpYy0dR3Qw.md` | 7.5 (Accept Spotlight) | Doubly-accelerated distributed learning; a clear theoretical advance over baselines |

**Positioning:** This paper sits between CORE (3.67) and LASER (5.83). It is more novel and honest than CORE, but its theory is explicitly weaker than the Rand baseline — a more severe issue than LASER's (which achieves parity with SGD). The empirical evaluation concern (best-run selection) further prevents this paper from reaching LASER's level. I anchor at ~4.5.

---

## Score and Decision

**Originality:** Moderate-high. The Markovian compressor concept is genuinely novel; the analytical technique (stepping back) is a fresh tool.

**Importance:** The research question is important and understudied.

**Claim support:** Weak on the theoretical side. The central claims of "acceleration" and "superiority" are not supported by the convergence analysis, which proves the opposite. The empirical claims are supported but with a methodological concern (best-run selection).

**Soundness of experiments:** Partially sound. Table 1 is properly statistically validated; Figures 1–2 are not.

**Clarity:** Good. Section 2.4's honest discussion is commendable.

**Value to community:** Moderate. Introduces a novel compressor family and proof technique, but does not resolve whether these compressors can match or beat simple Rand baselines theoretically.

The paper introduces a genuinely novel compressor family with solid formal characterization (Theorem 1) and a new proof technique, earning real credit. However, the gap between the title/abstract framing and the theoretical content is substantial and not fully acknowledged; the empirical evaluation protocol needs strengthening; and the core contribution of Markovian compression does not yet achieve its stated goal of "accelerating" beyond existing baselines in any theoretical sense. This positions the paper below the acceptance threshold for a venue like ICLR, where theory-empiric consistency is expected.

**Final Score: 4.5 / 10 — Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>