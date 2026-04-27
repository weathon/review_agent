Now let me search for calibration anchors.Now I have enough to write the final review. Let me verify the Table 2 two-row issue before writing.Now I have everything I need to write the final consolidated review.

---

## Summary

This paper decouples the class label from the target concept in machine unlearning, introducing three new scenarios beyond the conventional "all-matched" setting: *target mismatch* (forgetting data covers only a subset of the concept), *model mismatch* (model output granularity differs from the concept), and *data mismatch* (combination of both). The paper proposes TARF (TARget-aware Forgetting), a three-phase annealed gradient ascent and target-aware gradient descent framework that identifies false-retaining data via "representation gravity" and separates entangled concept representations. Empirically, TARF achieves order-of-magnitude reductions in the Gap metric over all baselines in the target mismatch and data mismatch settings on CIFAR-10/100 and ImageNet-1k.

---

## Strengths

- **Novel and extensible problem taxonomy**: The formal decoupling of $\mathcal{L}_D$, $\mathcal{L}_M$, and $\mathcal{L}_T$ into four combinatorial scenarios is clean and principled. CIFAR-100's class/superclass hierarchy is a well-motivated controllable testbed, and Figure 1 instantiates each scenario concretely rather than abstractly.

- **Convincing demonstration that prior methods fail**: Figure 2 and Table 3 show empirically that all existing methods—including SCRUB, $L_1$-sparse, and BS—fail substantially in the three new mismatch settings, with Gaps ranging from 8.86 to 48.99. This motivates the problem without hand-waving.

- **Substantial empirical improvements in classification**: In Table 3, TARF achieves Gap of 1.23/0.21 (CIFAR-10/100) in target mismatch versus the next-best GA at 20.80/8.86; and 0.96/1.17 versus GA at 5.89/2.43 in data mismatch. These are order-of-magnitude improvements with real practical significance.

- **Scalability to ImageNet-1k**: Table 4 confirms TARF achieves the best Gap scores across all four tasks at ImageNet-1k scale, demonstrating the method is not limited to small benchmarks.

- **Phase I identification validated empirically**: Figure 5(a) shows that target-concept classes experience significantly larger accuracy drops than non-target classes during Phase I, confirming that representation gravity provides a meaningful identification signal.

- **Annealing design choice justified by ablation**: Figure 7 (middle-left) compares constant vs. increasing vs. annealed (decreasing) gradient ascent schedules, showing the annealed variant best approximates the Retrained reference for model mismatch forgetting—supporting the design rationale.

---

## Weaknesses

### Fatal
None.

### Major

- **The UA metric in mismatch settings does not directly measure target-concept forgetting.** The paper's stated goal is to forget $\mathcal{D}_t = \mathcal{D}_f \cup \mathcal{D}_{fr}$. The Retrained reference is trained on $\mathcal{D} \setminus \mathcal{D}_f$ (which *includes* $\mathcal{D}_{fr}$), so its UA = 0.00 in target mismatch is consistent only with UA being measured on $\mathcal{D}_f$, not $\mathcal{D}_t$. The Retrained model still has full knowledge of $\mathcal{D}_{fr}$; if UA were measured on $\mathcal{D}_t$, the Retrained model's UA could not be 0.00. Consequently, the Gap metric in Table 3 for target/data mismatch measures proximity to a reference that is not actually the forgetting goal: a model that has forgotten $\mathcal{D}_f$ but *still knows* $\mathcal{D}_{fr}$. TARF's design does address $\mathcal{D}_{fr}$ through Phase I identification and Phase II forgetting, but the main evaluation tables do not directly demonstrate that $\mathcal{D}_{fr}$ is actually forgotten. The paper should add a "UA-T" metric (accuracy on all of $\mathcal{D}_t$) alongside the current UA. Figure 2's right panel gestures toward this for baselines, but there is no analogous post-TARF measurement in the main results.

- **The LLM experiments (Table 5) are unconvincing and unexplained in the main text.** Across all four task settings for LLaMA3.2-1B-Instruct, TARF(GA) and TARF(NPO) yield numerically identical results (e.g., QA Prob on F = 0.0095, QA Prob on R = 0.0094 for target mismatch in both variants). This invariance to the choice of base optimizer is not explained and suggests either an implementation issue or a degenerate outcome where TARF's phase structure overrides the optimizer entirely. Additionally, in the representation mismatch setting, all methods including the baseline GA collapse to QA Prob on R = 0.0000—meaning the model loses all retained knowledge. TARF provides no protection against this collapse. The paper dismisses these results to "Appendix F.8" without any discussion in the main text. Since the LLM extension is presented as evidence of generality, its failure modes belong in the main text. As presented, the LLM generalization claim is unsupported.

### Minor

- **Oracle assumption on the number of target classes.** Section 2 explicitly states: "we assume that the number of classes in $\mathcal{D}_{un}$ belonging to the target concept is known in target mismatch forgetting." In practical unlearning scenarios (privacy, copyright), the cardinality of implicit concept classes is not known in advance. The paper acknowledges a weakly-supervised robustness study in Appendix F, but the main Table 3 results are achieved under this oracle condition. A sensitivity analysis (what happens when the estimate is ±1–2 classes off?) belongs in the main text alongside the main results.

- **Two unexplained TARF rows in Table 2.** For CIFAR-100 model mismatch, Table 2 contains two rows both labeled "TARF (ours)" with Gap values of 2.65 and 1.36. The difference between these configurations—presumably different hyperparameter settings or different numbers of identified false-retaining classes—is not described in the main text.

- **Theorem 3.2 is a one-step, first-order bound, while the algorithm runs for multiple epochs.** The theorem provides an upper bound on per-step loss change as a function of representation distance $d_h(x_1, x_2)$. Multi-epoch dynamics are not covered. More critically, Definition 3.3's operational proxy $I_{\text{con}}(x, y, \theta) = |\ell(f_\theta(x), y) - \ell(f_{\theta^t}(x), y)|$ is asserted to reflect $d_h$ but this connection is not derived. The theorem motivates the intuition but the gap between formal bound and algorithmic proxy is not bridged.

### Trivial

- **Framing in Section 4.2 is slightly misleading.** The text claims TARF "generally performs better (or comparable with the best method)" but in the conventional all-matched setting TARF is strictly worse than SCRUB on CIFAR-100 (Gap 1.11 vs. 0.71). This is not a problem—all-matched is not the paper's focus—but the framing should be more precise.

---

## Nice-to-Haves

- **Add a two-stage ablation baseline**: identify $\mathcal{D}_{fr}$ using Phase I, then apply SCRUB on $\mathcal{D} \setminus \mathcal{D}_t$. This would isolate whether TARF's advantage comes from the Phase I identification signal (which any method could exploit) or from its specific joint forgetting/retaining objective. The current Figure 7 (right) compares gradient ascent vs. gradient cleaning on selected data, but does not include applying a strong baseline (e.g., SCRUB) after identification.

- **Discussion of cases where concept boundaries are unknown** (e.g., where the hierarchical class structure is unavailable, or where concepts are defined by attributes rather than class labels). The conclusion's "open challenge" paragraph acknowledges this briefly; even a preliminary experiment with embedding-based concept identification would strengthen practical relevance.

- **Sensitivity analysis for Phase I threshold β** as a function of estimated number of target classes, presented in the main text rather than the appendix only.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "GA achieves UA=40.82% despite applying gradient ascent directly to $\mathcal{D}_f$"** — The critic uses this as evidence that UA cannot be measured on $\mathcal{D}_f$. In fact, GA's imperfect forgetting of $\mathcal{D}_f$ (40.82% is not 0) is entirely expected because: (a) GA does not include a retaining term and degrades the model globally, and (b) in the coarse-to-fine taxonomy, some representation overlap persists. This is not contradictory.

- **Harsh Critic: "SCRUB being better on all-matched benchmark invalidates the paper's claims"** — TARF's novelty lies in the mismatch settings, not the conventional all-matched setting. SCRUB being marginally better there (Gap 0.71 vs. 1.11 on CIFAR-100) does not undermine the paper. Already addressed under Trivial.

- **Harsh Critic: "Phase ordering of $t_0$ and $t_1$ unclear"** — Figure 4 makes the ordering visually clear (Phase I → II → III corresponds to $0 \to t_1 \to t_0 \to T$). The text states "$t_0$ and $t_1$ respectively control the end time of active forgetting and the begin time of retaining part." This is adequately described.

- **Harsh Critic: "Missing two-stage baseline using SCRUB after identification"** — Included as a Nice-to-Have rather than a Major weakness, since the ablations do test the identification step's contribution and the three-phase design has clear theoretical motivation.

- **Formatting/parsing artifacts in Table 2** — The apparently swapped Gap values in the Retrained/FT rows for CIFAR-100 are PDF parsing artifacts; the original submission does not have this issue.

---

## Novel Insights

The paper's most interesting insight is what it calls "representation gravity": the observation that gradient ascent on forgetting data propagates its effect to other data proportionally to their proximity in representation space. This is both a *challenge* (entangled representations in model mismatch cause spillover) and an *opportunity* (under-entangled representations in target mismatch produce a signal that identifies false-retaining classes). Using a single phenomenon both as a diagnostic and as the basis for the identification step is elegant. The formalization in Theorem 3.2, though limited to a one-step bound, converts an empirical observation into a principled design rationale for all three phases of TARF. The insight that the appropriate response to these different gravity regimes (strong gravity → separation, weak gravity → identification) explains the unified but multi-phase structure of the algorithm.

---

## Suggestions

1. Add a "UA-T" column in Table 3 (accuracy on $\mathcal{D}_t = \mathcal{D}_f \cup \mathcal{D}_{fr}$ after unlearning) to directly demonstrate target-concept forgetting for both TARF and baselines. This is the most direct evidence for the paper's core claim and is currently absent from the quantitative results.

2. Either fix the LLM experiments or restrict the claim: explain in the main text why TARF(GA) ≡ TARF(NPO) numerically and why the representation mismatch task collapses for all methods, or frame Table 5 explicitly as a preliminary/exploratory result rather than evidence of generality.

3. Include a brief sensitivity analysis for the "number of target classes" oracle assumption in the main text (e.g., a 3-row table showing Gap when the estimate is exact, +1, and -1).

4. Clarify the two TARF rows in Table 2 for CIFAR-100 with explicit labels (e.g., TARF with $k=0.05$ vs. TARF with $k=0.02$).

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison to this paper |
|---|---|---|
| SalUn (gn0mIhQGNM) | 7.5 (Spotlight) | Stronger: cleaner evaluation, broader applicability, no oracle assumption, no LLM failures |
| In-/OOD Unlearning (HVFMooKrHX) | 6.6 (Poster) | Comparable novelty (novel framing of unlearning regimes), but that paper has theoretical guarantees rather than empirical improvements |
| Score Forgetting Distillation (gjwhDHeAsz) | 6.5 (Poster) | Comparable: strong empirical results across modalities |
| Rethinking LLM Unlearning (huo8MqVH6t) | 6.0 (Poster) | Comparable scope, similarly strong results in primary domain |
| UGradSL (hwXUmwJAq5) | 3.0 (Reject) | Much weaker: no novel problem, limited contribution |
| SUN Training-free MU (p7mgNvOD9Q) | 4.0 (Withdrawn) | Weaker: incremental, limited evaluation |
| Who's Harry Potter (PDct7vrcvT) | 5.25 (Withdrawn) | Comparable novelty but narrower scope, weaker experiments |
| Concept Forgetting via Label Annealing (2L7KQ4qbHi) | 3.0 (Withdrawn) | Much weaker: narrow contribution, weak results |

**Positioning**: The paper's core classification results are convincing and its problem formulation is genuinely novel—placing it above the medium anchors (5–5.25). The two major weaknesses (evaluation gap for core claim and degenerate LLM results) prevent it from reaching SalUn/spotlight territory (7.5). The paper is closest to the 6.0–6.5 posters in terms of contribution quality, but the identified evaluation gap is more concerning than the weaknesses in those papers. I settle on **6.0**, treating the paper as a marginally above-threshold contribution that requires the evaluation fix before it fully validates its central claim.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>