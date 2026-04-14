## Summary

This paper presents mechanistic evidence that a Deep Repeated ConvLSTM (DRC) agent, despite being model-free, learns to internally plan in Sokoban. The authors define two planning-relevant concepts — Agent Approach Direction (C_A) and Box Push Direction (C_B) — and show via linear probing that the agent's hidden states linearly encode these concepts far better than raw observations. They further demonstrate through causal interventions that these representations steer multi-step behavior, and that concept representation quality co-emerges with the ability to benefit from additional test-time compute. A qualitative analysis suggests the learned algorithm resembles parallelized bidirectional search.

---

## Strengths

- **Strong causal evidence via targeted interventions.** Unlike most interpretability papers that stop at correlation, Section 6.1 intervenes on specific concept vectors to steer the agent to follow suboptimal multi-step plans, achieving success rates of 94.6–98.8% (Agent-Shortcut) and 56.2–80.6% (Box-Shortcut) vs. random-probe baselines near 4–34%. Crucially, Figures 7–8 show the agent's *decoded internal plan* changes before its behavior does, suggesting the representations are upstream of action selection rather than merely epiphenomenal.

- **Test-time compute correlation as mechanistic fingerprint (Figure 9).** The tight co-emergence of probe F1 scores and extra-compute benefit across 50 training checkpoints provides an independent, behaviorally grounded validation that goes beyond the probing results alone. This is not merely correlation-as-evidence: it grounds the abstract planning claim in a concrete capability that practitioners already associate with planning.

- **Novel three-step methodology for investigating model-free planning.** The probe → plan formation → causal dependence framework is concrete, reproducible, and transferable to other architectures and domains — a genuine methodological contribution that fills a gap distinct from purely behavioral planning assessments.

- **Out-of-distribution generalization of decoded plans.** The appendices show the agent's C_A/C_B representations decode sensibly in OOD conditions: levels without the agent present, levels with extra boxes, and levels with dynamically appearing walls. This suggests the planning representations generalize in a way consistent with model-like behavior and is not merely a memorized policy over training-distribution states.

---

## Weaknesses

### Fatal
None.

### Major

- **Definitional circularity in C_A and C_B — the core epistemological gap.** Both concepts are defined by the agent's own future behavior over the remainder of the episode. The probe therefore measures whether the hidden state predicts what the agent *will in fact do*, not whether the agent is evaluating alternatives or predicting consequences of counterfactual actions. Representing a future trajectory is not equivalent to planning under the paper's own three-part characterization (formulate, evaluate, act). The intervention results establish causal *sufficiency* — that perturbing these representations changes behavior — but do not establish that in normal (unperturbed) operation the representations are *upstream* of decision-making rather than reflections of a decision already computed elsewhere. This distinction is central to the paper's strongest claim ("the agent is internally planning"), and the paper does not provide a direct information-flow analysis or path-tracing experiment to resolve it. The paper acknowledges the behavior-dependence of concept labels in Section 3.2, but does not subsequently argue why this does not undermine the planning interpretation.

- **Absence of a non-planning agent baseline.** The paper does not probe a feedforward ConvNet, a DRC with frozen recurrence, or a DRC trained for significantly fewer steps at matched task performance for C_A/C_B. Without this, it is impossible to know whether high probe F1 is specific to the planning behavior attributed to DRC agents, or whether any competent Sokoban policy — even a reactive one — encodes its own future trajectory and would yield similar probe scores. This is the single most straightforward experiment to conduct, and its absence means the paper cannot rule out the interpretation that these representations are an inevitable correlate of high competence on a deterministic task rather than a signature of planning specifically.

- **Bidirectional search claim is primarily qualitative.** The paper's most specific mechanistic hypothesis — that the agent uses "parallelized bidirectional search" — rests on visual inspection of selected examples in Figure 1 and appendix figures. There is no quantitative test: no measurement of where in the board plan arrows first appear across ticks, no comparison of expansion order against forward-only or backward-only heuristics, no aggregate statistics over many episodes. Given the specificity of the claim, this gap is noticeable. The paper honestly describes Section 5 as "qualitative evidence," but the abstract and introduction treat the algorithmic identification as a contribution, warranting stronger support.

### Minor

- **Box-Shortcut intervention gap unexplained.** Box-Shortcut success rates (56.2% at Layer 1, up to 80.6% at Layer 3) lag substantially behind Agent-Shortcut (94.6–98.8%). The paper notes the results are above the random baseline but does not analyze *why* C_B interventions are less effective — whether C_B is less linearly separable, whether action selection relies more directly on C_A, or whether the intervention method is less aligned with how box planning is encoded. Understanding this gap would sharpen the mechanistic picture.

- **Training co-emergence correlation is suggestive but not controlled.** Figure 9 shows correlation between probe F1 and extra-compute benefit across training checkpoints. However, both quantities increase with training competence, so the correlation may partly reflect an underlying confounder (overall policy quality). The paper does not control for baseline episode success rate. A partial correlation or ablation controlling for overall performance would make this evidence more compelling.

- **Intervention on handcrafted levels limits external validity.** The causal intervention experiments use bespoke Agent-Shortcut and Box-Shortcut levels designed to expose binary routing choices. It is not shown whether similar interventions steer behavior on natural Boxoban levels. Appendix B.3 reports intervention results for levels the agent cannot normally solve, which partially addresses this, but the community would benefit from seeing intervention effects on randomly sampled natural levels.

### Tiny

- **The "first mechanistic evidence" priority claim in the abstract needs more careful scoping.** This is a strong assertion that ICLR reviewers will scrutinize closely. Given the related work discussion already references Jenner et al. (2024) on look-ahead in chess agents, the claim should either be more precisely delimited (e.g., "first mechanistic evidence of full decision-time planning, including plan evaluation and action guidance, in a model-free RL agent") or hedged appropriately.

- **Intervention mechanics (Equation 1) are under-justified.** Adding the probe weight vector w_k to the cell state to encourage class k is a practice from representation engineering, but the paper does not explain why this should cleanly "set" the representation rather than create a perturbation that incidentally pushes the logit above threshold. A brief justification or citation to representation engineering literature would clarify the assumptions.

---

## Nice-to-Haves

- **Quantify the bidirectional search hypothesis.** Track when and where plan arrows (as decoded by probes) first appear across internal ticks on a large sample of episodes. Compute histograms showing whether new arrows appear preferentially near boxes (forward search) or near targets (backward search) at early ticks. Even a simple spatial heatmap split by tick number would substantially strengthen this claim.

- **Lesion experiment on planning-demanding vs. planning-light levels.** Corrupting or suppressing C_A/C_B representations should disproportionately harm performance on levels requiring long-horizon planning (where box moves are irreversible and route selection matters) compared to levels solvable with minimal lookahead. This would establish causal *necessity* as a complement to the causal *sufficiency* already demonstrated by interventions.

- **Controlled plan-revision experiment.** Measure whether the agent revises its decoded plan more frequently when the initial plan is infeasible (blocked route, deadlock configuration) versus when the initial plan is already valid. This would provide systematic evidence for the "plan evaluation" criterion beyond the qualitative examples in Figure 1(A)–(B).

- **Expand discussion of stochastic-environment generalization limits.** Since C_A and C_B rely on deterministic future trajectories, a brief discussion of how or whether the methodology would transfer to stochastic or partially observable environments would help contextualize the scope.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Harsh Critic: "The analogy to RL-trained LLMs and DeepSeek-R1 is speculative."** The connection is framed as motivation for why understanding emergent planning is important, not as a claim about mechanistic equivalence. This is a reasonable framing choice, not a substantive error. *Removed.*

- **Harsh Critic: "DRC model-free label is confused with non-modeling."** The paper's central question is precisely whether a model-free agent implicitly learns model-like planning, and the paper engages this directly in Section 2.1. *Removed.*

- **Harsh Critic: "Standard deviations insufficient — significance tests or confidence intervals required."** Reporting ±1 SD over five seeds is standard practice for this type of paper. For the probing results, the gap between agent hidden states and the observation baseline is large enough to be clearly significant without formal testing. *Removed.*

- **Harsh Critic: "Limited scope to one architecture/domain is a weakness."** The paper is explicit that it is a focused case study, and the additional results in appendices (PacMan, ResNet, DRC variants) extend generality. Demanding multi-architecture coverage in the main text would be scope creep for an interpretability study. *Weakened to nice-to-have context.*

- **Harsh Critic / Positive Reviewer: "Inference from 1x1 vs. 3x3 probe comparison to spatial localization is overstated."** The paper's reasoning is sound: if representations were distributed across neighboring cells, adding the 3x3 neighborhood would improve the probe meaningfully; the fact that it does not strongly (relative to the baseline improvement) supports localization. This is a reasonable inference from a sensible experimental design. *Removed.*

- **Multiple reviewers: demands for related work citations that reviewers allege are missing.** Per policy, missing related work is not flagged, as the reviewers may lack current knowledge. *Removed.*

- **Harsh Critic: "Broader impact section is thin."** This is a formatting/completeness nitpick that does not affect the scientific contribution. *Removed.*

---

## Novel Insights

The most genuinely novel observation emerging from the synthesis of these reviews is the tension between *causal sufficiency* and *causal necessity* in the paper's intervention experiments. The paper demonstrates that concept representations can be perturbed to steer long-horizon behavior (causal sufficiency) and that high-quality representations co-emerge with planning-like capacity during training. However, the reviews collectively highlight that these results do not establish that the representations are *upstream* drivers of action in normal operation, as opposed to downstream encodings of a plan computed elsewhere in the network. The specific architectural property that makes DRC amenable to this analysis — spatial correspondence between the ConvLSTM cell state and the Sokoban grid — is also what creates the circularity concern: a spatially structured recurrent network trained on a deterministic task will, by design, accumulate information about its future trajectory in spatially aligned representations, whether or not it is "planning" in the evaluative sense. Resolving this circularity via information-flow or path-tracing analysis would not just improve this paper; it would clarify what mechanistic evidence for planning can and cannot establish in future work.

---

## Suggestions

1. **Add a non-planning baseline probe comparison.** Train the same probing pipeline on a matched-performance feedforward Sokoban agent or a DRC with recurrence disabled (frozen hidden state). If C_A/C_B probe F1 scores drop substantially, this directly supports the claim that planning-specific computation is responsible for the representations. If they don't drop, the paper must grapple with why.

2. **Explicitly address the upstream/downstream question.** Even without full mechanistic circuit analysis, the paper could test whether the plan decoded at tick *t* predicts the agent's action better than the plan decoded at tick *t-1* after controlling for external observation content — a simple measure of whether the representation "leads" the behavior or merely reflects it.

3. **Quantify the bidirectional search claim.** Aggregate across many episodes the spatial distribution of newly appearing arrows at each internal tick, split by proximity to boxes vs. targets. This converts a qualitative hypothesis into a falsifiable quantitative test.

4. **Analyze the C_B intervention gap.** Investigate whether Box-Shortcut lower success is due to probe linear separability, representation geometry, or reliance on C_A for action selection. Even a brief ablation comparing intervention magnitude vs. success rate for C_A and C_B would shed light on the asymmetry.

5. **Calibrate the abstract's priority claim** to match the actual contribution: the paper provides strong convergent evidence (probing + causal intervention + training co-emergence) that a model-free RL agent encodes and causally uses representations consistent with a planning vocabulary — a more precise statement than "the first mechanistic evidence that model-free RL agents can plan."