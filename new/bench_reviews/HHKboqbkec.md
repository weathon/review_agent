Now let me do calibration searches to compare this paper against similar work in the corpus.Now I have enough information to write the final review. Let me compile my findings.---

## Summary

This paper proposes a **weak-to-strong Bayesian ToM reasoning** approach that enables test-time scaling of multimodal Theory-of-Mind inference to 70B and 405B parameter LMs without retraining them. The key idea is to transfer the "ToM behavior" learned by a small post-trained LM (captured as the ratio π^E / π^N) to guide likelihood estimation in a larger LM within the BIPALM Bayesian inverse planning framework. The method is evaluated on MMToM-QA across multiple Llama family sizes and five unseen thematic environments, demonstrating improvements over directly post-training large models and over the baseline BIPALM system.

---

## Strengths

- **Weak-to-strong control outperforms direct large-LM post-training (Table 2)**: Across all three Llama families, the `70B-ours` configuration consistently exceeds `70B-post-trained` (e.g., Llama-3.1: 79.38% vs 71.86% overall; Llama-3: 78.76% vs 73.05%), demonstrating that transferring the behavioral delta from a small post-trained model is more effective and stable than directly fine-tuning the large model.

- **Generalization to five unseen thematic environments (Table 4)**: The `8B↔405B` configuration achieves 80.2–80.5% across all five out-of-distribution scenarios (Andersen tales, ancient Egyptian, outer space, wild west, medieval castle), substantially outperforming `70B-post-trained` (71.8–75.3%), confirming that the approach generalizes beyond the apartment post-training domain.

- **Width vs. depth ablation of the weak controller (Table 3)**: The finding that 4B-width models outperform 4B-depth models as controllers (78.52% vs 78.38%) while underperforming in zero-shot provides a non-obvious and informative result about what properties of the weak model matter, grounded in learning theory on width-depth trade-offs.

- **Concept-level mechanistic insight (Figure 3 / Section 3.7)**: The analysis showing that post-trained small LMs shift probability mass from room- and furniture-level concepts to item-level concepts (e.g., wine, wine glass) while large LMs spread mass broadly is a concrete and informative explanation of the complementary roles, rather than just reporting accuracy numbers.

---

## Weaknesses

### Fatal
None in the sense of fabricated data or provably incorrect experimental results.

### Major

- **Equation 6 contains a critical symbol error that makes the core formula formally incorrect.** The equation as written reads:
$$\bar{\pi} = \frac{1}{Z}\,\pi^\mathcal{E}\cdot\frac{\pi^\mathcal{E}}{\pi^\mathcal{N}}$$
which reduces to $(\pi^\mathcal{E})^2 / \pi^\mathcal{N}$ — the large LM $\pi^\mathcal{L}$ plays no role, which contradicts every surrounding sentence. The inline description immediately following Eq. 6 says "$\pi^\mathcal{E}$ represents the policy distribution from the **naive large LM**," but $\pi^\mathcal{E}$ is defined throughout Section 2.2 as the **post-trained small LM** (see also Eq. 5). The normalization factor $Z$ repeats the same error. The intended formula — clearly $\frac{1}{Z}\pi^\mathcal{L}\cdot(\pi^\mathcal{E}/\pi^\mathcal{N})$ — is recoverable from context, but having the paper's primary method equation be formally wrong as submitted is a significant deficiency. Theorem 1 (referenced for theoretical support) is in the stripped appendix, making it impossible to verify whether the proof applies to the written formula or the intended one.

- **The core mechanism is proxy tuning, and this identity is not disclosed.** Once the symbol error is corrected, the formula $\bar\pi \propto \pi^\mathcal{L}\cdot(\pi^\mathcal{E}/\pi^\mathcal{N})$ is precisely the proxy tuning formula (Liu et al., 2024), which applies a fine-tuned small model's ratio to redirect a large base model at inference time. The paper cites Liu et al. (2024) in the related work only as a generic "reweighting technique" without disclosing that its primary operational formula is identical. The actual contribution is the **application context** — using proxy tuning as the likelihood estimation step within a Bayesian ToM inference loop — which is a meaningful engineering contribution, but is presented as a novel mechanism. This misrepresentation of novelty should be corrected by explicitly situating the contribution within proxy tuning.

- **The headline "~4.6% improvement over state-of-the-art" is not clearly derivable from any single table.** The abstract and main text (line 261) state 4.6% improvement over "the existing best baseline," but: (a) BIPALM w/ Llama-2-7B in the multimodal section of Table 1 achieves 80.0%, and Ours achieves 81.3% — a gap of only 1.3%; (b) the closest Table 2 calculation (405B-ours at 81.29% vs Llama-2 70B-post-trained at 76.43%) gives ~4.86%, not 4.6%. The comparison point is never specified. The 4.6% figure conflates results from different tables, model families, and configurations, making it unverifiable from the presented results.

### Minor

- **The 70B post-training baseline appears suboptimally tuned, which inflates the advantage of weak-to-strong control.** Section 3.1 states the 70B LoRA is configured with rank 8, alpha 16, and Section 3.3 attributes underperformance to "difficulty finding optimal hyperparameters for larger models" — but no hyperparameter search for the 70B condition is reported. For Llama-3.1, the 70B zero-shot baseline outperforms 70B-post-trained on several subtasks (e.g., 2.3: 16.0% post-trained vs. the paper's own acknowledgment of limited tuning), suggesting the baseline is undertrained. The claim that weak-to-strong control is superior to direct post-training at scale would be better supported with a properly searched 70B post-training baseline.

- **Evaluation is limited to a single benchmark (MMToM-QA) designed alongside the primary baseline (BIPALM).** Both the benchmark and the strongest baseline come from the same research group (Jin et al., 2024). No independent ToM benchmark is included to verify that gains are not specific to this benchmark's construction. This is a scope limitation, though the unseen-scenario transfer results in Table 4 partially mitigate it.

- **No variance or statistical significance reporting.** On a 600-question benchmark, a 1–2% difference corresponds to 6–12 questions. With comparisons across 7 task subtypes and multiple configurations, some apparent gains may not be statistically reliable. This is particularly relevant for claims about task-specific improvements in subtasks like 2.3.

### Trivial

- The main text framing that the method "avoids increasing post-training resource requirements" is technically true only for the large LM; the small LM still requires post-training, and running 405B at every Bayesian inference step is substantially more expensive at test time than BIPALM's 7B model. This framing should be clarified.

---

## Nice-to-Haves

- A compute-normalized comparison (e.g., equal FLOPs or wall-clock time budget) would make the "scalable" claim more rigorous, since running 405B per Bayesian step is expensive despite avoiding post-training.
- Including at least one independent ToM benchmark (e.g., FanToM, BigToM, OpenToM) would significantly strengthen the generalizability claim.
- Providing a proper hyperparameter sweep for the 70B post-training baseline would allow a fairer comparison and validate (or refute) the claim that weak-to-strong control is superior to direct fine-tuning.
- An explicit failure-mode analysis for weak-to-strong control (where does the small LM's bias miscorrect the large LM?) would balance the mostly positive Figure 3 analysis.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

1. **Harsh Critic: "Section 2.2 reads as if the POMDP formulation is original to this paper."** — The paper clearly states "Kaelbling et al., 1998; Jin et al., 2024" in line 118 and explicitly builds on BIPALM throughout. This is not misrepresentation; it is standard practice to reproduce the framework for completeness.

2. **Harsh Critic: "Inference cost of 405B makes the 'scalable' framing misleading."** — Weakened (moved to Trivial/Nice-to-have). The scalability claim refers to *training* cost, not inference cost. The paper never claims inference is cheaper; it claims post-training resources are saved. The framing could be clarified, but it is not fundamentally dishonest.

3. **Harsh Critic: "The 'unseen scenarios' are thematically relabeled VirtualHome, making the 'unseen' label misleading."** — Partially valid but overstated. The paper explicitly says the small LMs are "post-trained on the apartment" and tests transfer to thematically different domains. The underlying action-state structure from VirtualHome is the same, but the paper does not claim otherwise. Weakened to a minor clarification need, already mostly addressed in the paper text.

4. **Strength Finder: "The ratio-based formulation in Eq. 6 is interpretable and modular."** — Removed because Eq. 6 as written is incorrect (see Major weakness #1). The intended formula is interpretable, but we cannot cite the written equation as a strength.

5. **Strength Finder: "Avoids expensive post-training of large LMs [as a presentation strength]."** — Removed as a standalone strength because it is the paper's central claim, not an observation independent of the results, and the inference cost caveat complicates the framing.

---

## Novel Insights

The most genuinely novel observation is the width-versus-depth ablation of the weak controller (Table 3): 4B-width models post-train better and transfer better than 4B-depth models despite lower zero-shot baselines. This aligns with learning theory on width-depth trade-offs and provides an actionable principle for designing the "weak" component — namely, prefer width reduction over depth reduction when the goal is capturing transferable behavioral patterns for likelihood redirection. The concept-level analysis in Figure 3 (item-level focus shift in post-trained small LMs vs. broad coverage in large LMs) is a second non-obvious finding that provides mechanistic grounding for why the proxy-tuning-style ratio $\pi^\mathcal{E}/\pi^\mathcal{N}$ carries useful ToM-specific information.

---

## Suggestions

1. **Fix Equation 6** and its normalization: replace $\pi^\mathcal{E}$ (first factor in numerator and in $Z$) with $\pi^\mathcal{L}$. Also correct the inline description to say "$\pi^\mathcal{L}$ represents the naive large LM."
2. **Explicitly acknowledge the proxy tuning equivalence** in Section 2.2 and reframe the contribution as "applying proxy tuning as the likelihood estimation step within Bayesian ToM inference," situating the novelty honestly.
3. **State the exact comparison underlying "4.6%"** in the abstract, specifying the table row and baseline.
4. **Report a proper hyperparameter search for the 70B post-training baseline** (or cite evidence that 70B LoRA fine-tuning consistently underperforms), to strengthen the scalability conclusion.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Topic | Avg. Score | Decision |
|---|---|---|---|
| BIRD (fAAaT826Vv) | Bayesian+LLM probabilistic inference | 8.0 | Accept (Oral) |
| ExploreToM (246rHKUnnf) | ToM data generation/evaluation | 6.0 | Accept (Poster) |
| VNckp7JEHn | Test-time inference scaling | 5.75 | Accept (Poster) |
| MMToM-QA/BIPALM (sMFqEror1b) | Multimodal ToM benchmark + BIP-ALM | 4.75 | Reject |
| HyRe (8HQS1X2AK4) | Test-time alignment via reweighting | 5.33 | Reject |
| RuY1r1PDdQ | Single-task LLM evaluation benchmark | 3.0 | Reject |

**Positioning:**

The paper sits above MMToM-QA (4.75, Rejected) — the BIPALM precursor — because it provides a clearer methodological contribution (the proxy-tuning-based scaling mechanism) and richer analysis (Tables 3, 4, Figures 2–3). However, it sits below ExploreToM (6.0, Accepted) and VNckp7JEHn (5.75, Accepted) due to three compounding issues: the core Equation 6 is formally wrong as submitted, the relationship to proxy tuning is undisclosed (misrepresenting novelty), and the headline 4.6% claim is not transparently derivable. Papers at the 6-level typically have correct primary formulas and properly situated novelty claims. The single-benchmark evaluation and the undertrained large-model baseline further weaken the claim that weak-to-strong control is superior to direct fine-tuning. The paper is positioned at **4.5** — below the acceptance threshold, requiring major revisions on all three primary issues before it would be suitable for acceptance.

**Evaluation on key axes:**
- **Originality**: Moderate-Low. The core formula is proxy tuning applied in a new context; the application is meaningful but the novelty is not well-situated.
- **Importance of research question**: Good. Scalable multimodal ToM is a well-motivated problem.
- **Claims supported by experiments**: Partially. Main scaling results are positive, but the 4.6% headline is unclear and the key comparison baseline appears undertrained.
- **Soundness of experiments**: Moderate. Broad coverage but single benchmark, no variance reporting, questionable 70B baseline.
- **Clarity of writing**: Poor in the critical section (Eq. 6 and surrounding description are internally contradictory).
- **Value to community**: Moderate. The width-vs-depth and concept-level analyses are useful; the core contribution would be valuable if properly situated.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>