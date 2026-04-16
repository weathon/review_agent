## Summary

The paper studies failure modes of target-directed RL agents that use self-generated intermediate targets, focusing on cases where agents pursue unreachable or unsafe targets. It introduces a taxonomy of “problematic targets” (G.1 nonexistent, G.2 temporarily unreachable) and “delusional” estimator errors (E.0/E.1/E.2), then proposes two hindsight relabeling strategies (“generate” and “pertask”) and hybrid generator/estimator relabeling to expose estimators to such targets. Experiments in a custom MiniGrid-based environment (SSM) with the Skipper method (and additional results in the appendix) suggest that these hybrids reduce certain error metrics and improve OOD generalization.

## Strengths

- **Clear decomposition of generator vs. estimator roles.**  
  The paper cleanly abstracts target-directed agents into a generator that proposes targets and an estimator that evaluates them (Sec. 2, Fig. 1). It makes a useful conceptual distinction between hallucinations (bad targets proposed) and delusions (failure to downweight bad targets), which aligns with how many recent planning-based goal-conditioned methods are architected.

- **Intuitive taxonomy of problematic targets and estimator errors.**  
  The G.1/G.2 categories (invalid/impossible vs. temporarily unreachable targets, Sec. 3.1) and E.0/E.1/E.2 estimator delusions (Sec. 3.2) are well-motivated and concretely exemplified in SSM. The notion of temporary unreachability via irreversible transitions (e.g., sword/shield possession classes) is especially clear. This gives readers a vocabulary to talk about distinct reachability-related errors.

- **Environment tailored for diagnosis.**  
  The SSM environment is designed so that delusions are visible and analyzable: discrete gridworld, clearly defined semantic classes ⟨sword, shield⟩, lava traps that induce terminal states, and guaranteed viable paths. This allows computing exact shortest-path distances and cleanly labeling G.1/G.2/E.1/E.2, which is rare in OOD generalization work and makes the analysis in Fig. 3 interpretable.

- **Concrete, simple mitigation strategies.**  
  The “generate” strategy (training the estimator using generator-proposed candidate targets) and “pertask” (sampling targets from anywhere in the replay buffer for the same task) are straightforward to implement on top of HER (Sec. 4.1). Their intended roles—“generate” for G.1/E.1 and “pertask” for G.2/E.2—are clearly argued, and Table 1 gives a concise summary of tradeoffs.

- **Hybrid 2-slot relabeling design.**  
  Separating generator and estimator relabeling distributions (Sec. 4.3) is a good design insight: generators benefit from focusing on useful, reachable targets while estimators benefit from seeing problematic examples. The paper systematically evaluates several hybrids (“F-(E+G)”, “F-(E+P)”, “F-(E+P+G)”) and shows they outperform pure atomic strategies on its metrics (Fig. 3d–h).

- **Empirical evidence of nontrivial improvements.**  
  On Skipper in SSM, the hybrids reduce certain E.2-related estimation errors and behavior frequencies relative to standard “future”/“episode” HER variants, and achieve noticeably higher aggregated OOD success rates (Fig. 3h). The narrative that “F-(E+P)” and “F-(E+P+G)” better handle temporarily unreachable G.2 targets is supported by the plotted metrics (especially Fig. 3f,g,h).

## Weaknesses

### Fatal

None. The paper is a real, coherent piece of work with nontrivial contributions; there is no single flaw that completely invalidates the empirical results or makes it “not even a paper.” However, there are structural issues where the framing and strength of claims are not fully supported.

### Major

- **Conceptual overreach of the “delusion” framing.**  
  The core construct of “delusion” is introduced with psychiatric language (“obviously wrong beliefs”, “inability to reject false beliefs”, “belief formation vs belief evaluation systems”), but the operationalization in RL reduces to partitioning estimator approximation error across different subsets of source–target pairs. For example:
  - E.0 is “misevaluating non-delusional targets”; E.1/E.2 are misevaluations on G.1/G.2 targets (Sec. 3.2). These are simply subsets of the error surface defined by a choice of labeling; there is no formal criterion that distinguishes “ordinary approximation error” from “delusional belief.”
  - The supposed incoordination between generator and estimator is described qualitatively but is never formalized beyond the fact that some source–target pairs are not sampled for the estimator.
  - The necessary conditions for addressing delusions (Sec. 3.2: appropriate update rules, training data coverage) are exactly the standard conditions for learning to approximate a value/distance function under distribution shift.
  
  The paper’s more modest technical contribution—analyzing how HER relabeling choices bias the estimator’s training distribution over reachable/unreachable pairs and proposing coverage-boosting strategies—stands on its own. But the stronger framing that “delusions” form a qualitatively new class of failure modes with a distinct causal story is not convincingly supported.

- **Loose causal link between strategies and “delusion” mitigation vs generic approximation gains.**  
  The metrics used to argue delusion reduction (Sec. 5.2, Fig. 3) confound the specific notion of delusion with general estimator quality:
  - “E.1/E.2 Estimation Errors” are L1 errors on distances for G.1/G.2 targets, with unreachable targets clipped to a maximum value. There is no threshold or decision-theoretic criterion connecting an error level to behavior that is “delusional” rather than just suboptimal.  
  - “Non-delusional estimation errors” (Fig. 3d) “include the case of E.0,” blurring the line between delusional and non-delusional error.
  - “Delusional behavior ratios” (Fig. 3c,g) track frequencies of G.1/G.2 *chosen* targets, but the text only briefly distinguishes candidate ratios vs behavior ratios. The exact conditioning (e.g., chosen given being proposed vs raw frequency) is not clearly spelled out in the main text.
  
  As a result, while Fig. 3 shows that hybrids reduce certain error metrics and change behavior frequencies, it is hard to separate:
  - Improvements truly due to better handling of unreachable/unsafe targets (as per the G.1/G.2/E.1/E.2 taxonomy), versus
  - Improvements due to broader data diversity and better global distance estimation.
  
  The conclusion’s claim that the methods “grant the agents the ability to address delusions autonomously and preemptively avoid delusional behaviors” (Sec. 8) is significantly stronger than what the current evidence justifies; the paper shows useful error reductions in a toy domain, not a general solution to delusional behavior.

- **Generality and scalability claims go beyond what is demonstrated.**  
  The paper repeatedly emphasizes that the strategies “should be expected to be applicable generally” and that separating training data for generator and estimator is “straightforward” beyond HER (Sec. 4.1, 4.3, 7). However:
  - All concrete implementations and experiments are on HER-based, dual-component methods (Skipper, LEAP) in small, fully observable gridworlds with manually designed structure and computable shortest-path distances.
  - “pertask” relies on sampling arbitrary targets “across the entire memory” (Sec. 4.1.2), which is straightforward in a 12×12 gridworld but could be problematic in high-dimensional state spaces where most random target pairs are essentially irrelevant or unreachable.
  - “generate” assumes a generator that can be cheaply queried during training and outputs targets in the same representation used by the estimator.
  
  Without at least one non-gridworld or higher-dimensional experiment, or a more careful theoretical discussion of when these strategies help vs hurt, the strong generality claims are not substantiated.

- **Missing baseline perspectives and ablations.**  
  The experimental comparison is almost entirely intra-family: different HER relabeling strategies for Skipper (and LEAP in the appendix). Missing are:
  - Comparisons to *non*-target-directed goal-conditioned methods (e.g., standard HER-trained UVFA) in the same SSM setting. This would illuminate whether the identified failure modes are genuinely specific to target-directed architectures or are generic.
  - Simple alternative ways to expose estimators to unreachable/unsafe pairs, especially in SSM where ground-truth reachability and distances are cheap to compute (e.g., synthetic negatives or explicit reachability labels). That could show whether HER-based relabeling is the most natural or efficient solution.
  - Ablations that hold generator quality fixed across variants more tightly. The paper partially addresses this by focusing on “F-**” variants (Sec. 5.4–5.5), but generator and estimator are still trained jointly under different relabeling distributions, so interactions remain entangled.

### Minor

- **Lack of environment diversity in the main text.**  
  Only Skipper on SSM is shown in detail; three of four experiment sets (including LEAP and a second environment) are in the appendix, summarized briefly in Sec. 5.6. This limits the reader’s ability to assess whether the findings are robust across methods and task structures.

- **Mixture proportion choices are ad hoc and under-analyzed.**  
  The hybrids use specific ratios (e.g., 50% episode, 25% pertask, 25% generate for F-(E+P+G)), but the paper provides neither principled selection criteria nor sensitivity analysis. Given that the main gains come from choosing mixtures, robustness to these choices matters for practical utility.

- **Terminological density and some confusion.**  
  The introduction of many labels (G.x, E.x, types of HER, behavior ratios) makes the exposition harder to follow. In particular, the overlap between “non-delusional” errors and E.0 is not crisply delineated in Sec. 5.2 and Fig. 3d.

- **Limited discussion of computational overhead.**  
  The paper notes that “generate” carries computational cost (Sec. 4.1.1) but does not quantify it (e.g., generator calls per update, wall-clock slowdown). For real-world applications with tight time budgets, this could be important.

### Trivial

- Some figure captions are duplicated and verbose (e.g., Fig. 2 and Fig. 3 have multiple, near-identical captions in the text). This is likely a formatting artifact but contributes to clutter.

## Nice-to-Haves

- Additional visualizations of the learned distance estimators over the SSM grid (e.g., heatmaps showing ground-truth vs estimated distances for representative targets) to make “false beliefs” more tangible.
- Case-study trajectories contrasting a baseline and a hybrid agent on the same evaluation task, highlighting exactly where delusional target choices occur and how the hybrid avoids them.
- A short, explicit discussion of how to approximate G.1/G.2/E.1/E.2 in environments where exact reachability is not computable (e.g., using learned reachability classifiers or uncertainty thresholds).

## Removed Points

These points are flagged to be removed, treat them with caution. They generally either misunderstand the paper, overreach beyond what’s reasonable to ask, or are rooted in generic expectations rather than this work’s actual content.

- **Claim that “delusions” are purely circular redefinitions with no added value beyond standard approximation error.**  
  While the construct is not fully formalized, the taxonomy of G.1/G.2 and E.0/E.1/E.2 and the explicit focus on target reachability and temporal structure do add some conceptual clarity beyond generic “approximation error.” The weakness has been kept but softened: the paper overclaims conceptual novelty rather than being completely circular.
- **Implied requirement to test on large-scale benchmarks (e.g., D4RL, robotics) as a condition for any contribution.**  
  The paper explicitly positions itself as a diagnostic study in controlled environments with full ground-truth access. While broader benchmarks would strengthen the case, their absence does not invalidate the contributions given the stated scope. This concern has been reframed as a limitation on generality rather than a fatal flaw.
- **Suggestion that the psychiatric analogy is inherently inappropriate.**  
  The paper uses the analogy for intuition and does not base any technical claims on psychiatric models. Criticizing the analogy as such would be stylistic; the substantive issue is the lack of precise formalization, which is already covered under major weaknesses.

## Novel Insights

None beyond the paper’s own contributions. The reviewers’ concerns largely align with common patterns in HER/goal-conditioned RL work (limited environment diversity, scalability questions, need for more ablations). The most distinctive insight is already present in the paper: that generators and estimators in target-directed agents have conflicting data needs and benefit from decoupled training distributions.

## Suggestions

- **Reframe and tighten the conceptual claims.**  
  Present the main contribution more modestly as: (i) a reachability-focused analysis of target-directed failure modes under HER relabeling, and (ii) practical relabeling strategies and hybrids that improve estimator coverage of problematic targets. Reduce reliance on the psychiatric “delusion” narrative or clearly mark it as an informal analogy.

- **Clarify metrics and behavioral definitions.**  
  In Sec. 5.2 and the main text, explicitly define:
  - How “G.1/G.2 candidate ratio” and “behavior ratio” are computed (conditioned on what).
  - How “non-delusional errors” relate to E.0.
  - What error ranges actually lead to different target choices in Skipper/LEAP.  
  This would make the connection between estimator errors and decision-time behavior much clearer.

- **Add at least one more experiment to the main text.**  
  Promote one of the appendix sets (e.g., LEAP on SSM or Skipper on the second environment) to the main body with a figure analogous to Fig. 3. This would better support claims that the strategies generalize across architectures and environments.

- **Include sensitivity analyses for mixture proportions.**  
  For one environment/method, vary the mixing ratios for “episode/future/pertask/generate” in a systematic small grid and report OOD performance and key delusion metrics. If performance is robust across a range, this reassures practitioners; if not, it highlights the need for tuning heuristics.

- **Discuss scalability and limitations explicitly.**  
  Add a subsection in the discussion or conclusion that:
  - Acknowledges reliance on small, fully observable MDPs with computable distances.
  - Outlines challenges and potential adaptations for continuous/high-dimensional settings (e.g., approximate reachability via learned models, subsampling in “pertask”).
  - Clarifies that the current results are diagnostic and do not yet demonstrate gains on large-scale benchmarks.

- **Compare against at least one alternative estimator-augmentation baseline.**  
  In SSM, where ground-truth distances are known, implement a simple baseline that augments the estimator’s loss with penalties on unreachable or long-distance pairs sampled synthetically, without using HER-style relabeling. Even if it underperforms, this will clarify what is specific about the proposed strategies.

### Evaluation on key axes

- **Originality:** Moderate. The relabeling strategies and hybrid 2-slot training are natural extensions of existing HER ideas, but the systematic reachability-centric analysis and taxonomy are a meaningful conceptual addition within the target-directed RL niche.
- **Importance of research question:** Moderate to high within goal-conditioned / target-directed RL: unreachable/unsafe intermediate targets are a real concern and interact with OOD generalization and safety.
- **Support for claims:** Mixed. Empirical support for “these mixtures can improve OOD performance and reduce certain estimator errors” is solid in SSM; support for broader conceptual and generality claims is weaker.
- **Soundness of experiments:** Reasonable within the chosen setting (20 seeds, well-instrumented metrics), but missing some key baselines and ablations.
- **Clarity of writing:** Mixed. The high-level story and environment descriptions are clear, but terminology density and overlapping categories can be confusing.
- **Value to the community:** Moderate. As a diagnostic study of HER and target-directed agents in carefully controlled gridworlds, the paper can help clarify failure modes and inspire better data selection schemes, but its current framing may oversell the concept rather than emphasizing the practical lessons.

## Score and Decision

To calibrate, I compared to several human-reviewed papers:

- **Skipper (eo9dHwtTFt.md; scores 6,6,5,6)** – Similar domain (MiniGrid-based, target-directed RL with planning), with concerns about scalability and limited domains but judged overall as a solid, above-threshold contribution. The present paper is conceptually related but somewhat less mature empirically (no more complex domains, heavy framing around “delusions” that is not fully nailed down).
- **Goal-Conditioned RL with Virtual Experiences (OjCWG58ZyY.md; scores 6,5,6,5)** – Another HER-based method with multiple components and limited environment diversity; reviewers saw it as reasonable but not groundbreaking. This paper feels somewhat weaker in empirical breadth and baseline coverage.
- **Null Counterfactual Factor Interactions (2uPZ4aX1VV.md; scores 8,8,5,6)** – A stronger HER-related paper with both solid theory and more varied experiments; clearly above the current submission’s level.
- **Bias-Resilient Multi-Step GCRL (llXCyLhOY4.md; scores 3,3,3,3)** – A HER-improvement paper criticized for too many moving parts and insufficient ablations; weaker overall than this submission, which has more coherent diagnostics.

Relative to these anchors, this paper sits below Skipper and the better HER-latent works, but clearly above very weak submissions. I therefore place it around the borderline but slightly negative region.

MY FINAL SCORE: <pineapple>4.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>