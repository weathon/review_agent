=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary

This paper provides the first mechanistic (non-behavioral) evidence that a model-free reinforcement learning agent — a Deep Repeated ConvLSTM (DRC) agent in Sokoban — can learn to internally plan. The authors introduce a three-stage interpretability methodology: (1) linear probing for planning-relevant concepts (Agent Approach Direction C_A and Box Push Direction C_B), (2) qualitative analysis of how plans form and evolve across internal ticks, and (3) causal intervention by injecting concept vectors into the cell state to steer multi-step behavior. The paper further shows that the emergence of these concept representations during training correlates with the emergence of planning-like behavior (benefiting from extra test-time compute).

---

## Strengths

- **Convergent causal evidence via principled intervention design.** The paper goes beyond correlation by showing that injecting probe-derived concept vectors into the cell state can redirect the agent over entire episodes. Agent-Shortcut interventions achieve 94.6–98.8% success vs. random-probe baselines of ~30%, and even Box-Shortcut interventions (56–80%) are far above their 4–31% baselines. Crucially, the authors show the interventions not only change behavior but also change the decoded internal plan (Figures 7–8), suggesting the mechanism runs through the representation rather than around it.

- **Spatial locality as a concrete empirical insight.** The minimal performance difference between 1×1 and 3×3 probes (compared to a much larger gap for observation baselines) is a specific, falsifiable empirical finding: the agent encodes planning-relevant concepts at spatially localized positions in its ConvLSTM cell states, consistent with the grid-aligned structure of Sokoban. This is not a generic claim about network representations.

- **Out-of-distribution generalization of plan representations.** The agents' representations of C_A and C_B extend meaningfully to levels outside the training distribution — levels without the agent present, levels with extra boxes/targets, and levels with appearing/disappearing walls (Appendices A.2.6–A.2.9). This suggests the representations are not memorized patterns but genuinely compositional, which strengthens the planning interpretation considerably.

- **Training emergence analysis tied to a specific behavioral signature.** Figure 9 shows that probe F1 correlates strongly with the ability to benefit from extra test-time compute — a well-established behavioral indicator of planning in DRC agents (Taufeeque et al., 2024). Using this specific behavioral proxy, rather than raw performance, is a targeted choice that links mechanistic and behavioral evidence.

- **Triangulation across three independent evidence types.** The paper uses probing, qualitative plan decoding, and causal interventions, each with different failure modes, to converge on the same conclusion. The convergence itself is evidence that the finding is not an artifact of any single methodology.

---

## Weaknesses

### Fatal
None.

### Major

- **Potential circularity in concept definition and the "policy encoding" alternative.** C_A and C_B are defined using the agent's *actual future behavior* in each episode. High probe F1 therefore partly proves only that the agent's recurrent cell state encodes information about its own future outputs — a property that any sufficiently expressive recurrent policy will exhibit to some degree, regardless of whether it is "planning." The paper invokes causal interventions to rebut this, and the interventions are its strongest evidence. However, the main text does not directly and explicitly engage with the "policy encoding" alternative: an agent could commit to a policy early in each time step, and the recurrent dynamics simply settle to a fixed point that trivially predicts future outputs. A more direct test — e.g., showing that the concept representations change meaningfully across internal ticks in ways that reflect search rather than policy convergence, or showing that these representations are specifically diagnostic on novel levels where the policy has not been trained — would substantially sharpen the argument. Appendix A.3.2 reportedly addresses this, but it is relegated to an appendix despite being central to the core claim.

- **Plan formation evidence (Section 5) is entirely qualitative and based on a small number of hand-selected examples.** The most ambitious claims in the paper — that the agent performs "parallelized bidirectional search," evaluates plans, and adapts routes — rest on Figure 1 (5 examples, A–E) and Figure 5 (6 examples), selected from many episodes. There is no quantification of: how often forward vs. backward vs. bidirectional search patterns appear; what fraction of episodes show clean plan evaluation (as in Figure 1A–B); or whether the bidirectional search pattern meets any formal definition of bidirectional BFS (e.g., whether the two frontiers provably meet at a point). The paper presents Section 5 as qualitative, which is appropriate, but the label "parallelized bidirectional search" carries algorithmic precision that the evidence does not support. Without statistics over episodes, readers cannot assess representativeness.

- **Training emergence correlation does not control for overall learning progress.** Figure 9 plots probe F1 against planning-like behavior across the first 50M of 250M training transitions. Both quantities increase monotonically with training. The natural alternative hypothesis — that both are jointly driven by general learning progress (reward, solve rate) — is not tested. If overall solve rate explains nearly all the variance in both quantities, the correlation would be largely uninformative about a special relationship between concept representations and planning. A partial correlation or regression controlling for solve rate would resolve this. Additionally, restricting the analysis to the first 50M of 250M transitions raises the question of whether the correlation holds or breaks down later in training.

### Minor

- **Box-Shortcut intervention asymmetry is unexplained.** Layer 1 Box-Shortcut success (56.2%) is only ~25 points above its random baseline (31.5%), while the corresponding Agent-Shortcut success (94.6%) is ~61 points above its baseline (33.7%). This large and systematic difference between C_A and C_B interventions is not explained. Possible explanations include: C_B representations being more distributed, the causal chain from C_B to action being more indirect, or box-pushing dynamics being harder to manipulate with additive steering. The absence of any analysis of this asymmetry is a gap, since C_B is the more planning-critical concept (it encodes box trajectories, which determine puzzle solutions).

- **The bidirectional search characterization lacks formal grounding.** The claim that the agent's planning "resembles parallelized bidirectional search" is stated as a discovery but never validated against any formal algorithmic criteria. Classical bidirectional BFS has specific properties (two frontiers expanding from start and goal, meeting in the middle). The paper shows arrows growing from boxes and from targets, but does not demonstrate these are algorithmically connected or that they meet in a way consistent with bidirectional search. This should be presented as a suggestive metaphor with supporting quantification or tempered in scope.

### Tiny

- The distinction between this paper's contribution and Jenner et al. (2024) (look-ahead in chess) is explained in Related Work but not in the Introduction, where the "first non-behavioral evidence" claim appears. Readers unfamiliar with Jenner et al. may not appreciate the distinction.

---

## Nice-to-Haves

- **Comparison to a DRC agent with N=1 ticks in the main text.** Appendix F reportedly includes evidence from DRC without internal ticks; including a brief summary in the main body would directly support the claim that internal ticks are mechanistically enabling planning, rather than requiring readers to search the appendix.

- **Episode-level statistics over the plan visualization examples.** Even a simple breakdown of what fraction of episodes exhibit forward-only, backward-only, bidirectional, or parallel search patterns (from a random sample of 100–200 episodes) would transform the qualitative Section 5 into a much stronger empirical claim.

- **Negative control probing.** Testing whether planning-*irrelevant* concepts (e.g., the color of a square, or proximity to a wall) are *not* linearly represented with comparable F1 would strengthen the claim that C_A and C_B are specifically selected for, not just an artifact of probe capacity.

- **Temporal precedence test in training emergence.** Rather than showing correlation across checkpoints, showing that concept F1 *leads* planning-like behavior in time (e.g., via a lag analysis) would establish temporal precedence, making the causal claim in Section 6.2 considerably stronger.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Missing ablation: DRC without internal ticks"** (Harsh Critic): This IS addressed. Section 7 explicitly states "Appendices F provides evidence of DRC agents planning both without internal ticks, and with additional internal ticks." The critic missed this.

- **"NEVER class dominates; macro F1 may be driven by NEVER classification"** (Harsh Critic): The paper explicitly acknowledges the class imbalance and uses macro F1 specifically to compensate (Section 4.2). This is a standard and appropriate statistical choice; the criticism is adequately pre-empted.

- **"25 intervention levels is a limited set"** (Harsh Critic): The 200 effective levels (25 × 8 transformations) is standard methodology for causal intervention studies in mechanistic interpretability. Demanding more is not standard in the field.

- **"Layer variation interpreted as iterative computation is speculative; all layers may simply converge to the same representation"** (Harsh Critic): This is a reasonable alternative interpretation but does not undermine the paper's core claim. The paper notes it as a hypothesis (Section 4.2), not a definitive finding. Appendix A.3.1 provides supporting evidence.

- **"The paper should provide circuit-level analysis identifying specific neurons/attention patterns"** (Spark Finder): Circuit-level mechanistic analysis is not a standard or expected methodological requirement for papers in this area, especially for recurrent convolutional architectures where circuit identification is substantially harder than in transformers.

---

## Novel Insights

The most genuinely novel insight, which goes beyond what any single sub-reviewer highlighted, is the *spatial locality* finding combined with the intervention evidence: the agent has apparently learned to use a bijective spatial mapping between its ConvLSTM cell states and the Sokoban grid, encoding planning-relevant concepts at spatially localized positions. This is not simply a statement about what concepts are represented, but about *how* they are architecturally organized — the cell state at position (x, y) specifically encodes predictions about what will happen at grid square (x, y), and additive interventions at those positions have localized, semantically meaningful causal effects on behavior. This spatial organization likely explains *why* the planning representations are tractable to extract and manipulate with simple linear probes and additive steering, and may generalize as a design insight: recurrent architectures with spatial correspondence to their environment may be naturally suited to learning localizable, intervenable planning representations.

---

## Suggestions

1. **Directly address the policy-encoding alternative in the main text.** Dedicate a paragraph or short subsection to explicitly distinguishing "the agent committed early to a policy that happens to predict future behavior" from "the agent is actively searching." The most convincing existing evidence for this distinction (Appendix A.3.2, showing plan quality improves in a manner consistent with deeper search) should be elevated to the main body.

2. **Add episode-level statistics to Section 5.** From a random sample of 1000 episodes, report the fraction that exhibit each qualitative planning pattern (forward, backward, bidirectional, plan evaluation, plan adaptation). This converts a qualitative finding into a quantitative one and allows readers to judge representativeness.

3. **Test or rule out the overall-learning-progress confounder in Figure 9.** Add a scatter plot of solve rate vs. planning-like behavior at the same checkpoints, and a partial correlation analysis. If probe F1 predicts planning-like behavior *above and beyond* solve rate, the main claim of Section 6.2 is substantially strengthened.

4. **Temper or operationalize the "bidirectional search" claim.** Either measure a specific, formal property of bidirectional search (e.g., that the two frontiers meet in the middle of routes) across decoded plans, or explicitly frame the bidirectional search characterization as a qualitative analogy rather than an algorithmic identification.

5. **Explain or investigate the Box-Shortcut intervention asymmetry.** Even a brief hypothesis — e.g., that C_B representations are more distributed due to the indirect causal link between box trajectories and immediate action selection — would address a genuine gap in the paper's self-consistency.

---

**Overall assessment:** This is a well-motivated, methodologically creative, and significant paper that makes a genuine contribution to mechanistic interpretability of RL agents. The novelty is high, the causal intervention design is technically sound and carefully controlled, and the significance for the broader question of emergent planning is clear. The main limitations are the qualitative nature of the plan formation evidence in Section 5 (which makes the "parallelized bidirectional search" characterization anecdotal), an underaddressed circularity concern in the main text, and an uncontrolled confound in the training emergence analysis. These are real issues that weaken the strength of the paper's most ambitious claims, but they do not invalidate the core contribution, which rests most solidly on the intervention results. The paper should be accepted contingent on more direct engagement with the circularity concern and the addition of even basic quantitative statistics over plan visualization examples.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
