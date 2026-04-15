Now let me search for calibration papers to anchor the score.Now I have enough information to write the consolidated review. Let me synthesize everything.

---

## Summary

This paper introduces a taxonomy of "delusions" in target-directed RL agents, distinguishing generator-side problems (G.1: nonexistent/impossible targets; G.2: temporarily unreachable targets) from estimator-side misevaluations (E.0–E.2). Two new hindsight relabeling strategies, "generate" and "pertask," are proposed alongside a 2-slotted hybrid architecture that separates generator and estimator training data. Experiments in a custom grid-world environment (SSM) with two methods (Skipper in the main text, LEAP in the appendix) demonstrate that hybrid strategies reduce estimated delusion errors and delusional behavior frequencies, yielding improved OOD generalization performance.

---

## Strengths

- **Clear and operational taxonomy.** The G.1/G.2/E.0/E.1/E.2 classification is specific enough to drive experimental design and distinguish meaningfully different failure modes. The separation of generator hallucination from estimator delusion is conceptually crisp and avoids conflating phenomena that prior work had lumped together.

- **Mechanism-level evaluation.** Rather than relying solely on final returns, the paper directly measures E.1 and E.2 estimation errors against ground-truth distances and monitors delusional behavior frequencies throughout training. This supports attribution of outcomes to specific failure modes rather than confounds.

- **Well-motivated diagnostic environment.** SSM's irreversible state partitions (sword/shield acquisition) make G.2 (temporarily unreachable) targets concrete, measurable, and analyzable. The environment cleanly separates equivalence classes so that ground-truth reachability can be computed.

- **Multi-metric, multi-seed evaluation.** Fig. 3 reports eight sub-metrics across 20 seeds with 95% confidence intervals, covering generator behavior, estimation errors, delusional behavior frequencies, and OOD performance. The systematic decomposition is more rigorous than typical RL benchmark-only papers.

- **Principled 2-slotted design insight.** The observation that generator and estimator have conflicting training data needs (generator benefits from avoiding problematic targets; estimator must see them to reject them) is a meaningful and actionable design principle.

- **Practical guidelines.** Section 7 gives concrete, step-by-step guidance linking specific delusion types to specific mitigations, increasing the practical value of the framework.

---

## Weaknesses

### Fatal
*(None — no single issue invalidates the paper's core contribution.)*

### Major

- **Limited experimental scope in the main text.** The paper explicitly states "3 out of 4 sets of experiments are presented in the Appendix" (Sec. 5), leaving only Skipper on SSM visible to reviewers in the main body. Both SSM and the undisclosed second environment are custom-built grid worlds from the MiniGrid-BabyAI framework. The headline claim is that strategies "improve OOD generalization" of target-directed agents broadly, but the visible evidence supports only "these strategies help on SSM-like environments." Section 5.6 summarizes consistent conclusions across all 4 sets, but reviewers cannot verify this. The absence of a single continuous-control or image-based domain means the scope of the claim materially exceeds what is demonstrated.

- **Incremental algorithmic contribution.** The "generate" strategy is explicitly acknowledged to have been introduced in Zhao et al. (2024) (Sec. 4.1.1: "Zhao et al. (2024) ... proposed to train the estimator additionally with candidate targets proposed by the generator"). The paper reframes it as a HER relabeling strategy and adds "pertask" (cross-episode relabeling from the replay buffer). Neither strategy is technically surprising given the paper's own analysis; the primary contribution is conceptual/taxonomic rather than algorithmic. This is not a fatal flaw, but the paper's framing as a methods contribution should be tempered.

- **Correlational rather than causal mechanism evidence.** The paper asserts that gains come from correcting specific delusion types (E.1, E.2), and the multi-metric evaluation supports this narratively. However, "generate" and "pertask" simultaneously alter training data quantity, diversity, and the proportion of long-range vs. short-range pairs, without matched controls that vary only the delusion-relevant component. The causal chain "lower E.2 error → fewer delusional behaviors → better OOD performance" is plausible and partially evidenced by the sequential analysis in Sec. 5.5, but the paper cannot rule out generic effects of broader or more adversarial supervision.

### Minor

- **Update rules deferred but mechanistically important.** Section 4 explicitly "skips discussing update rules" and restricts empirical validation to Skipper and LEAP, which happen to have update rules that can punish unachieved targets. Sections 3.2 and 7 present proper update rules as a necessary condition for the strategies to work. The breadth of the prescriptive guidelines in Sections 4 and 7 therefore depends on an untested prerequisite. At minimum, the guidelines should more prominently note this assumption, and even an informal characterization of which update rule families satisfy it would strengthen the contribution.

- **Scalability of "pertask" unaddressed.** Relabeling with observations sampled across the entire memory ("all targets experienced before," Sec. 4.1.2) raises nontrivial questions about memory footprint and computational overhead as the replay buffer grows. The paper discusses the computational cost of "generate" (Sec. 4.1.1) but is entirely silent on "pertask" overhead. This information is needed for practitioners to assess feasibility in more complex domains.

- **Mixture proportions not justified.** The specific mixture ratios tested (50/50 for F-(E+P), 2/3–1/4–1/4 for F-(E+P+G)) are stated without discussion of how they were selected or whether performance is sensitive to them. Without at least a brief sensitivity note, practitioners cannot know how robust the prescription is.

- **G.2 target identification relies on exploitable state structure.** The paper notes (Sec. 5.3) that "G.1 & G.2 propositions are clearly identified" in experiments due to SSM's four equivalence classes. In environments without cleanly segmented state classes or access to ground-truth reachability, operationalizing "pertask"'s cross-episode relabeling in a way that actually exposes the estimator to G.2 pairs (vs. arbitrary distant pairs) may require non-trivial additional engineering. This limits the out-of-the-box applicability of the strategy, particularly for the second "another environment" whose details are in the appendix.

### Trivial

- The claim that agents without estimators (e.g., Director) are "at significant risk" (Sec. 3.2) is stated without a supporting experiment on such agents. This should be phrased more cautiously as a theoretical consequence of the framework.

---

## Nice-to-Haves

- Including at least one non-grid-world domain (e.g., continuous control with irreversible state transitions) in the main text would substantiate the general applicability claim without requiring the reader to accept appendix results on faith.
- An ablation explicitly comparing 2-slotted separation vs. single-slot mixture (same relabeling proportions, different architectural treatment) would validate the 2-slotted design as a contribution beyond the mix ratios alone.
- A brief mediation analysis or regression decomposing how much of OOD improvement is explained by E.2 error reduction vs. other changes would sharpen the causal narrative.
- Quantitative wall-clock/memory comparisons across strategies (particularly "pertask" vs. baseline) would enable practitioners to assess feasibility.
- A decision flowchart for practitioners to diagnose which delusion type dominates (G.2 risk vs. G.1 risk) before selecting strategies would increase practical value of Section 7.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic – Statistical reliability (20 seeds insufficient):** The paper reports 95% CI over 20 seeds in Fig. 3, which is solid practice for RL. The generic concern about seed count is not substantiated given the actual reporting. **Removed as unsupported.**

- **Harsh Critic – Hyperparameter tuning fairness across strategies:** The concern that hyperparameters might be tuned per strategy is raised without any evidence that this occurred. It is a generic reproducibility nitpick not grounded in anything paper-specific. **Removed per hard rule on reproducibility nitpicks.**

- **Spark – No comparison with non-target-directed baselines (PPO, SAC):** The paper's explicit scope is failure modes *within* target-directed agents, not whether the target-directed paradigm outperforms flat methods. Demanding a comparison outside the paper's stated scope is scope creep. **Removed as outside stated scope.**

- **Harsh Critic – Experiments not available for assessment (appendix):** The paper clearly states 3/4 experiments are in the Appendix due to page limits, and summarizes consistent conclusions in Sec. 5.6. Claiming the appendix experiments "cannot be assessed" is not a valid criticism of the paper's existence or correctness of those results. **Removed; the limited-scope concern is preserved under Major weaknesses.**

- **Neutral Reviewer – Psychiatric analogy adds limited depth:** While the psychiatric motivation is light, it serves as an organizing metaphor and the paper does not claim to import formal properties from psychiatry. This is purely a framing choice, not a methodological flaw. **Removed as stylistic.**

---

## Novel Insights

The paper's most genuinely novel insight is the identification and formalization of G.2 (temporarily unreachable) targets as a distinct failure mode category, separate from both G.1 (impossible) targets and general estimator miscalibration. The key observation is that trajectory-level relabeling strategies like "episode" systematically expose the estimator only to in-trajectory reachable targets, leaving a structural blind spot for cross-class reachability that cannot be closed by training harder or longer on the same type of data. This motivates the cross-episode "pertask" strategy as a necessary complement rather than an optional enhancement. The insight that generator and estimator components have *conflicting* training data needs—with the generator harmed by, and the estimator requiring exposure to, problematic targets—is also a useful design principle that applies broadly beyond HER.

---

## Suggestions

1. **Move at least one appendix experiment set into the main text.** The LEAP-on-SSM results or the second environment results should be presented (even briefly, with one representative figure) to provide visible support for the generality claims.
2. **Add a brief characterization of which update rule families satisfy the necessary conditions** (Sec. 3.2). Even an informal proposition ("update rules that continuously penalize unachieved targets in hindsight will satisfy the requirement") would make the scope of the contribution clearer.
3. **Report computational and memory overhead for "pertask"** alongside the existing discussion of "generate"'s overhead in Sec. 4.1.1.
4. **Soften generality claims in the abstract and introduction** to match the actual experimental scope (grid-world environments with clear state-class structure, two specific HER-based methods), and position the paper primarily as a conceptual and diagnostic contribution with initial empirical validation.

---

## Score and Decision

**Calibration:**

- **2uPZ4aX1VV** (Null Counterfactual HER for GCRL) — Accept, scores 8,8,5,6. Stronger algorithmic novelty (NCII), robotics environments, but with similar concerns about appendix-dependent experimental claims and domain knowledge in practice. This paper is meaningfully below that bar.
- **OjCWG58ZyY** (Virtual Experiences for GCRL) — Reject, scores 6,5,6,5. Similar area (HER extension), similar scope (limited domain variety, no clearly dominant baseline advantage), rejected for lacking causal clarity and over-reliance on unanalyzed design choices.
- **BH8Nrt2dPf** (Horizon Generalization in RL) — Accept (poster), scores 6,5,3,3. Primarily conceptual, limited experiments, accepted borderline.
- **OvrmA3GMiX** (Transferable Sub-goals) — Reject, scores 6,3,3,3. Conceptual framing, limited validation, narrow domain, rejected.

The paper under review sits above OvrmA3GMiX (the taxonomy is better operationalized and the multi-metric experimental setup is more rigorous) and is comparable to OjCWG58ZyY. It has a cleaner conceptual contribution than OjCWG58ZyY but weaker empirical breadth in the main text. The 2uPZ4aX1VV bar is too high given the incremental algorithmic novelty and narrow experimental scope here. BH8Nrt2dPf suggests borderline conceptual papers can be accepted, but that paper had a full theoretical analysis as its primary contribution; this paper's theory is lighter.

The paper makes a genuine, useful contribution—the taxonomy is operationally sound, the experiments are thoughtfully designed, and the insights are actionable. However, the main-text evidence is limited to a single custom environment/method combination, the primary algorithmic contribution ("generate") is acknowledged prior work, and the broader claims outstrip the demonstrated scope. This places the paper at a borderline, leaning toward weak accept/reject, consistent with the 5–5.5 range of similar papers.

**Final Score: 5.0**

**Rationale for borderline:** The conceptual framework is valuable and the paper is better than average in mechanism-level analysis, but the experimental scope in the main text is too narrow for the headline claims, the algorithmic novelty is modest, and the causal evidence for the delusion-correction mechanism is correlational. A revision moving one additional experiment set to the main text and softening generality claims would substantially strengthen the case.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>