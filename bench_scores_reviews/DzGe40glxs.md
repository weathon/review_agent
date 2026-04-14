---

## Summary

This paper investigates whether a model-free Deep Repeated ConvLSTM (DRC) agent, trained on Sokoban, internally performs decision-time planning. The authors apply a three-step concept-based interpretability methodology: (1) linear probing for planning-relevant concepts (Agent Approach Direction C_A and Box Push Direction C_B), (2) qualitative analysis of iterative plan formation over internal compute ticks, and (3) causal interventions on internal representations to steer agent behavior. They further show that the emergence of these concept representations during training correlates with the onset of planning-like behavior (benefiting from extra test-time compute), and perform a qualitative analysis suggesting the learned algorithm resembles parallelized bidirectional search.

---

## Strengths

- **Convergent three-line evidence methodology.** The paper triangulates its central claim using three distinct approaches — linear probing, qualitative plan visualization over internal ticks, and causal intervention — rather than relying on any single method. Each pillar provides meaningfully different types of evidence. The methodology is articulated clearly enough to be reusable on other agents and environments, which is a standalone contribution.

- **Striking probe performance gap over baselines.** The 1×1 hidden-state probes achieve Macro F1 of ~0.85 (C_A) and ~0.95 (C_B) versus observation baselines of ~0.25, across all three layers. The size of this gap, and the fact that the minimal improvement from 1×1 to 3×3 probes applies to hidden states but not the baseline, strongly supports spatially localized linear representations of future-trajectory concepts in the agent's cell state.

- **Causal intervention results are substantive.** Table 1 reports Agent-Shortcut success rates of 94.6–98.8% and Box-Shortcut rates of 56.2–80.6%, against random probe baselines of 27.8–33.7% and 4.1–31.5% respectively. The fact that the interventions modify not just behavior but the decoded internal plan (Figures 7 and 8) goes meaningfully beyond pure behavioral steering and provides the strongest causal evidence in the paper.

- **Training dynamics linking concept emergence to behavioral capability.** Figure 9 shows a strong correlation between probe F1 and the percentage of extra levels solved via extra test-time compute, measured across 50 training checkpoints. This links the mechanistic findings to a functional behavioral signature, providing co-emergence evidence that neither probing alone nor behavioral benchmarks alone could provide.

- **Breadth of supplementary results.** Appendices extend the analysis to out-of-distribution Sokoban levels (missing agents, extra boxes), Mini PacMan, and ResNet agents — a scope substantially broader than the main text suggests and that lends credibility to the generality of the phenomenon.

---

## Weaknesses

### Fatal
None. The core claim — that a DRC agent linearly encodes future-trajectory concepts, that these are causally involved in behavior, and that they co-emerge with planning-like capability — is substantially supported. The paper's overclaiming does not invalidate the evidence it provides.

### Major

- **Intervention evidence is limited to handcrafted shortcut levels, not natural Sokoban puzzles.** The causal claims in Section 6.1 rest entirely on two families of carefully engineered levels (Agent-Shortcut and Box-Shortcut) designed so that exactly two routes exist differing in length. The paper does not demonstrate that concept-vector interventions alter behavior on standard, unmodified Boxoban levels — for instance, redirecting which target a box is planned toward in a naturally occurring level. Without this, it is unclear whether the representations have planning-level causal influence in the general case, or whether the interventions exploit the engineered simplicity of the test levels. This substantially limits the strength of the causal claim.

- **The "bidirectional search" claim is supported only by cherry-picked visual examples without quantification.** Section 5 and Figure 1 claim the agent uses "parallelized bidirectional search" — a specific algorithmic characterization. This is supported by approximately five hand-selected examples. No automated metric is provided over a large episode sample (e.g., temporal ordering of arrow emergence near targets vs. boxes across ticks, statistics on backward-vs-forward frontier growth). Without such analysis, this claim functions as an illustrative hypothesis rather than a demonstrated finding. The paper should substantially downgrade this framing or provide quantitative backing.

- **The core interpretive ambiguity between "encoding future behavior" and "planning" is underresolved.** C_A and C_B are defined from the agent's *actual future behavior* in the episode. This means probes predicting them are, at least in part, decoding the agent's committed future trajectory or implicit policy summary — which a competent reactive policy would also produce. The critical question — whether the agent's representations *determine* upcoming action selection via an internal search process, or merely *reflect* a compressed policy state — is raised but not resolved. A key missing control is probing a matched feedforward (non-recurrent) agent for C_A and C_B: if a feedforward agent also achieves high F1, the iterative-search interpretation is significantly weakened; if it cannot, the recurrent-computation hypothesis is strongly supported. Section 5 acknowledges the qualitative nature of the evidence and defers to Section 6, but the intervention experiments (see Major weakness above) are themselves limited to engineered levels.

- **Single trained agent; generalization across seeds and checkpoints is unaddressed.** The paper states "the agent we study" with a single training run of 250M transitions. The mechanistic claims are presented as properties of "DRC agents" generally, but the analysis is conducted on one checkpoint of one seed. Whether the identified concepts, probe F1 levels, and intervention success rates are reproducible across training seeds or sensitive to training idiosyncrasies is unknown. In mechanistic interpretability work, this is a meaningful limitation.

### Minor

- **Observation baseline is structurally weaker than hidden-state probes.** The baseline probes receive static observations $x_t$ without any recurrent history. Because the hidden state captures temporal context accumulated over an episode, the comparison is asymmetric in a way that could overstate the implied representational specificity. A recurrent baseline — e.g., probing a randomly-initialized ConvLSTM run on the same observations — would provide a fairer comparison.

- **The "forced stationary" protocol may introduce distribution shift.** The 5-step "thinking" protocol (Section 5, Figure 6) forces the agent to remain stationary before acting, a situation absent from training. The observed improvement in probe F1 over these extra ticks may partly arise because the unrolled dynamics land in a more stereotyped hidden-state regime rather than purely reflecting deeper internal search. The paper does not discuss this confound.

- **"Iterative computation" inference from cross-layer probe uniformity is underdetermined.** Section 4.2 infers that similar probe performance across layers implies "iterative computation" that refines plans across layers. But similar cross-layer decodability is also consistent with simple feature copying, residual skip connections, or parallel encoding at all layers. This interpretation is plausible but not entailed by the data.

- **Training dynamics analysis (Figure 9) is limited to the first 50M of 250M training transitions without justification.** The paper does not explain why the emergence analysis covers only the first 20% of training. If the phenomenon saturates early, this should be stated explicitly.

### Tiny

- **"Unlikely to overfit" based on parameter count alone is not a sound argument.** The claim that 160/1440-parameter probes are "unlikely to overfit" does not follow from parameter count; overfitting depends on dataset structure, class correlations, and imbalance. The argument is unnecessary given the test-set evaluation design.

- **"Confirms spatial bijection" overstates the 1×1 to 3×3 comparison.** Section 4.2 says the minimal 1×1 to 3×3 improvement "confirms" a spatial bijection. This is consistent with localization but does not exclude distributed redundancy across positions. The language should be softened to "supports" or "is consistent with."

- **The AS vs. BS intervention success gap is noted but unexplained.** Table 1 shows systematically lower Box-Shortcut success rates, yet the paper attributes this to no specific cause. As the paper explicitly discusses both concepts as planning representations, this asymmetry — which might indicate that box-push dynamics are encoded more robustly or in a more distributed way — deserves at least brief analysis.

---

## Nice-to-Haves

- **Probe a feedforward (non-recurrent) agent on C_A and C_B.** This is the single most impactful experiment missing from the paper. A competent feedforward policy would also produce future trajectory regularities, but cannot engage in iterative internal search. If it achieves comparable probe F1, the iterative planning interpretation is threatened; if its F1 is significantly lower, the argument becomes substantially stronger.

- **Demonstrate concept-vector interventions on standard Boxoban levels.** Showing that interventions can redirect planned box routes on naturalistic levels (e.g., changing which target a box is steered to) would substantially strengthen the causal claim beyond the engineered shortcut setup.

- **Ablation: zero out or add noise to C_A/C_B directions and measure solve-rate degradation.** This would test whether the representations are causally necessary (not just influenceable), complementing the existing additive-steering interventions.

- **Provide aggregate statistics on plan formation patterns.** Report over hundreds of episodes: what fraction show backward extension from targets, what fraction show forward extension from boxes, what fraction show plan revision, and at which ticks. This would move the bidirectional search hypothesis from anecdote to empirical finding.

- **Include an explicit code/model release statement.** Standard ICLR reproducibility expectation; not currently mentioned.

- **Partial correlation analysis in Figure 9.** Does probe F1 predict extra-compute benefit above and beyond total solve rate? Controlling for training progress would make the co-emergence argument stronger.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Priority claim of first mechanistic evidence is irresponsible"** (Harsh Critic): The paper does state "first mechanistic evidence" and "first non-behavioral evidence," and these claims appear defensible within the paper's stated scope (concept-level mechanistic evidence, model-free RL, non-behavioral). While the paper could use slightly more cautious language, removing this as an overclaim is appropriate given the paper does qualify its methods and scope.

- **"Label leakage through policy determinism"** (Harsh Critic): In complex Sokoban environments with PSPACE-complete difficulty and four boxes, future trajectories are not trivially predictable from current state alone. The concern that near-deterministic policies trivially predict future labels does not apply strongly here. The observation-baseline comparison already addresses a substantial part of this.

- **"DeepSeek-R1 comparison feels trend-chasing"** (Harsh Critic): This is rhetorical/stylistic criticism about a motivational aside. Not a scientific weakness.

- **"Broader impact not discussed"** (Harsh Critic): Not a standard requirement for an interpretability/RL paper at ICLR and is too generic to constitute a scientific weakness.

- **"Much of the qualitative evidence is cherry-pickable examples"** (Spark Finder, re: Figure 5 static plans): Figure 5 is presented explicitly as illustrative examples, and the paper does not make statistical claims based solely on them. The quantitative probe results are the evidentiary foundation.

- **"Probe has an unfair comparison because it lacks temporal context"** duplicates the minor weakness above and is partially addressed by the existing observation baseline (which does capture the static state). The residual concern is kept as a Minor weakness above.

---

## Novel Insights

The most genuinely novel observation across all three reviews — and underemphasized even by the authors — is the **asymmetry between Agent-Shortcut (up to 98.8%) and Box-Shortcut (up to 80.6%) intervention success rates**, which the paper does not explain. This asymmetry suggests that the planning representations for self-movement (C_A) and for object dynamics (C_B) may be qualitatively different in terms of how causally potent, robustly encoded, or distributed they are — a distinction that could illuminate hierarchical structure in the learned planning algorithm. The second notable insight is that probe F1 improvement during "forced stationary" extra compute ticks plateaus after ~12 ticks out of 15 (Figure 6), suggesting the agent's search process has a natural depth limit. If this depth limit scales with architecture depth or tick count, it would be a mechanistically informative finding about what DRC agents can and cannot plan ahead.

---

## Suggestions

1. **Add feedforward-agent probing as a control experiment** — this single addition would most directly address the "amortized policy vs. online planning" ambiguity and substantially strengthen the paper's core claim.

2. **Demonstrate concept-vector interventions on 50–100 standard Boxoban levels**, even informally, to show the causal results are not an artifact of the shortcut-level design.

3. **Quantify the bidirectional search observation** with at minimum one automated metric over a large episode sample (e.g., the temporal ordering of arrow emergence near boxes vs. targets across ticks), and reframe this as a "consistent with bidirectional search" hypothesis rather than a demonstrated algorithmic characterization.

4. **Analyze and report on the C_A vs. C_B intervention asymmetry** — even a short paragraph on why box-push planning is harder to steer than agent-movement planning would be mechanistically informative and turn a confusing gap in Table 1 into a finding.

5. **Add a partial-correlation analysis to Figure 9**, controlling for total solve rate, to show that probe F1 independently predicts extra-compute benefit.

6. **Justify the 50M-transition window** in Section 6.2 or extend the emergence analysis to the full 250M training run.

7. **Clarify the single-seed limitation** in the limitations section, and ideally report that key quantitative results (probe F1, intervention success rates) hold across at least 2–3 independent training seeds.

---

**Overall evaluation:**

- *Novelty*: High. Providing mechanistic (concept-level) evidence of planning in a model-free RL agent, using the combination of probing + causal intervention + training-dynamics analysis, is a genuinely new approach to a longstanding question.
- *Technical soundness*: Moderate. The probing and intervention methodology is well-executed, but the central interpretive ambiguity (planning vs. amortized policy) is not fully resolved, and the causal evidence is limited to engineered environments.
- *Empirical support*: Moderate-to-strong for the probing and intervention claims; weak for the bidirectional-search algorithmic characterization.
- *Significance*: High in principle — mechanistic understanding of emergent planning has broad implications for both interpretability and RL — but the significance is somewhat constrained by the single-agent, single-environment setting and the unresolved core ambiguity.
- *Clarity*: Good overall, but the paper would benefit from more careful epistemic calibration between "we provide evidence consistent with" and "we show/confirm/demonstrate."