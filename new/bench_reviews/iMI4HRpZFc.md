## Summary
This paper studies failure modes of target-directed RL agents through the lens of “delusions”: cases where agents hold false beliefs about self-generated targets and therefore pursue unreachable or unsafe subgoals. It proposes a taxonomy separating problematic targets (G.1 nonexistent, G.2 temporarily unreachable) from estimator failures (E.0/E.1/E.2), and introduces two relabeling strategies, **generate** and **pertask**, plus hybrid two-slot training schemes to better train generator and estimator components. Empirically, the paper shows on controlled environments that these strategies reduce estimator errors and delusional behaviors and improve OOD performance.

## Strengths
- **Clear and useful failure-mode taxonomy.** The G.1/G.2 and E.0/E.1/E.2 decomposition is the paper’s strongest contribution. In particular, the emphasis on **temporarily unreachable targets** (G.2) is insightful and well supported by the SSM construction with irreversible semantic classes such as \(\langle 0,0\rangle \to \langle 1,0\rangle\).
- **Diagnosis and mitigation are tightly connected.** The proposed strategies are not arbitrary tricks: “generate” is explicitly designed to expose the estimator to generator-proposed candidates, while “pertask” exposes it to cross-episode targets that reveal temporary unreachability. The paper also correctly identifies that generators and estimators can have conflicting training-data needs and motivates the two-slot design in Sec. 4.3.
- **Good mechanistic evaluation.** Rather than only reporting success rates, the paper measures candidate pathology, estimator error by delusion type, delusional behavior frequency, and downstream OOD performance. This is the right style of evidence for a failure-analysis paper.
- **Controlled environment enables unusually direct inspection.** Because SSM is fully observable and its reachability structure is known, the paper can compute ground-truth distances and explicitly separate E.1 and E.2 cases. That lends credibility to the diagnosis within the studied setting.
- **Empirical gains appear real in the studied setting.** In the main Skipper-on-SSM experiment, the hybrid strategies outperform the atomic baselines in aggregated OOD performance and reduce the corresponding delusion metrics, supporting the paper’s central practical message at least in this regime.

## Weaknesses

###: Fatal
- None.

### Major:
- **The paper overclaims generality relative to the evidence.** The framing repeatedly targets “target-directed decision-making” broadly, and Sec. 4 says the ideas are “applicable generally” and extend “beyond HER.” However, the concrete methods and validations are overwhelmingly HER-centric: Sec. 4 explicitly assumes an agent “learning both components exclusively from hindsight-relabeled transitions,” and the proposed implementations are relabeling strategies. The main paper presents only one detailed experiment set (Skipper on SSM), with the other three deferred to the appendix. This supports a meaningful claim about HER-trained target-directed agents in controlled settings, but not the broader scope suggested by the title/abstract/conclusion.
- **The evidence for broader OOD-generalization claims is narrower than the rhetoric suggests.** The experimental setup is intentionally constructed to surface exactly the identified pathology: SSM has irreversible/segregated state classes; Sec. 5.1 states that the initial-state setup and short MEL “increases risks of E.2”; Sec. 5.2 fixes evaluation initial states to a particularly hard semantic class and position. This is a valid stress test, and I do not consider it unfair, but it means the results demonstrate that delusion-aware relabeling helps on benchmarks designed to expose delusions. That is weaker than showing delusion mitigation is a broadly established driver of OOD generalization across diverse target-directed settings.
- **Mixture performance is not well isolated from generic relabeling-diversity and tuning effects.** The hybrids use hand-chosen proportions (e.g., 50/50, 50/25/25), but the paper does not provide sensitivity analysis or a principled selection rule. As written, the gains are consistent with the paper’s intended explanation—better exposure to problematic targets reduces delusions—but also with a weaker interpretation that a richer relabeling mix simply improves learning on this benchmark. Given that the central scientific claim is causal and mechanism-specific, this missing analysis matters.

### Minor
- **Limited main-text empirical diversity.** Only one environment/method pair is presented in detail in the body, with the rest moved to the appendix. Even if the appendix experiments are supportive, the main paper alone does not make the cross-method/cross-environment case particularly strongly.
- **The scope of applicability is narrower than some of the exposition suggests.** Sec. 2 describes the estimator as optional, but the mitigation recipe in Secs. 3.2, 4, and 7 effectively requires an estimator capable of rejecting problematic targets. The paper’s insights remain useful, but they apply most directly to a subclass of target-directed methods with explicit evaluators.
- **No quantitative accounting of computational overhead for “generate.”** Sec. 4.1.1 acknowledges additional computation, but the paper does not report wall-clock or training-time overhead. For a practically motivated relabeling method, this would be useful.
- **Lack of tuning guidance for practitioners.** Sec. 4.2 correctly notes tradeoffs among short-distance, long-distance, and problematic pairs, but the paper does not turn that into actionable guidance for choosing the mixture in a new domain.
- **The conceptual framing is stronger than the formal grounding.** The taxonomy is intuitive and useful, but the paper does not formalize conditions under which specific delusions arise or are corrected. This is not required for an empirical RL paper, but it limits how far the conclusions can be generalized analytically.

### Trivial
- None.

## Nice-to-Haves
- Add a sensitivity study over the mixture ratios for F-(E+G), F-(E+P), and F-(E+P+G).
- Quantify the compute overhead of “generate.”
- Include at least one higher-dimensional or continuous-control domain in the main paper to test whether the taxonomy and mitigation remain effective beyond MiniGrid-style structure.
- Provide a more explicit mapping from the abstract framework to classes of methods that do and do not have usable estimators.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Request for comparisons to non-target-directed RL baselines.** This is largely outside the stated scope: the paper is about diagnosing and mitigating failure modes in **target-directed** agents, not proving superiority over standard RL more generally.
- **Complaint that the paper should compare to unspecified additional HRL/model-based/uncertainty methods.** Without concrete evidence that such comparisons are necessary to validate the core claim, this is too open-ended and risks becoming a generic baseline request.
- **Demand for formal sample-complexity or convergence theory.** For this type of empirical failure-analysis paper, lack of theory is a limitation but not a core flaw. It is better framed as a nice-to-have than a requirement.
- **Criticism of confidence intervals/statistical testing alone.** The paper already reports CIs in Fig. 3, and absence of additional pairwise testing is not, by itself, a substantive weakness here.
- **Any suggestion that the comparisons are unfair because the authors fix the generator to “future” and thereby strengthen baselines.** The asymmetry here does not disadvantage the baselines; if anything it removes poor generator choices to better isolate estimator effects, which is acceptable for the narrower claim being tested.

## Novel Insights
The paper’s most valuable insight is not merely that unreachable subgoals are bad, but that **temporally structured unreachability** creates a distinct estimator-learning problem that standard trajectory-level HER systematically under-covers. This helps explain why a method can look competent on ordinary hindsight relabeling yet still fail badly at decision time when proposed targets lie outside the support of training source-target pairs. The two-slot perspective is also genuinely useful: it clarifies that generator training should avoid problematic targets while estimator training may need deliberate exposure to them, which is a sharper design principle than simply “add more relabeling diversity.”

## Suggestions
- Narrow the paper’s main claim to what is convincingly demonstrated: delusion-aware relabeling improves HER-trained target-directed agents in environments with meaningful G.1/G.2 structure.
- Add a mixture-ratio sensitivity study and, if possible, a simple adaptive mixing heuristic.
- Move at least one additional experiment from the appendix into the main paper, preferably one that differs materially in environment structure.
- Quantify the training-time overhead of “generate.”
- Clarify in the introduction/conclusion that the empirical evidence is strongest for methods with explicit estimators and HER-like training.
- If space permits, include a more direct causal ablation: match non-delusional estimation quality while varying exposure to G.1/G.2 targets to isolate the effect of delusion reduction itself.

## Score and Decision
**Assessment by axis:**  
- **Originality:** Good. The taxonomy, especially G.2/E.2, is a real conceptual contribution.  
- **Importance:** Moderately high within goal-conditioned / target-directed RL; less clearly broad beyond that niche.  
- **Support for claims:** Mixed. The paper supports a narrower claim well, but overstates generality.  
- **Experimental soundness:** Solid for a controlled diagnostic study, but limited in breadth and lacking sensitivity analysis.  
- **Clarity:** Generally clear and structured.  
- **Community value:** Meaningful for researchers working on target-conditioned planning and HER failure modes.

**Calibration against human-reviewed anchors:**  
- Compared to **Skipper** (`eo9dHwtTFt.md`, scores 6/6/5/6, accepted poster), this paper is somewhat narrower and more diagnostic, with less broad experimental appeal and more overclaiming, though it does offer a useful failure taxonomy. I place it slightly below that acceptance-level anchor.  
- Compared to **Goal-Conditioned RL with Virtual Experiences** (`OjCWG58ZyY.md`, scores 6/5/6/5, reject), this paper has a clearer diagnosis and stronger mechanistic evaluation, but still shares concerns about breadth and generality. I place it around the upper part of that borderline-reject range.  
- Compared to stronger accepted relabeling work such as **Null Counterfactual Factor Interactions** (`2uPZ4aX1VV.md`, scores 8/8/5/6, accepted poster), this submission is clearly less mature experimentally and more limited in scope.  
- Compared to lower-end MiniGrid/HRL rejects like **Option Boosting** (`3RfGSbXUt8.md`, scores 3/3/5), this paper is substantially stronger conceptually and empirically.

Overall, this looks like a **borderline paper with a real idea and credible controlled evidence, but insufficient breadth and too much rhetorical overreach for acceptance in its current form**.

**Score: 5.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>