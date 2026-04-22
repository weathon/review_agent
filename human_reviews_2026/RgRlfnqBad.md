# Center of Gravity-Guided Focusing Influence Mechanism for Multi-Agent Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Cooperative multi-agent reinforcement learning (MARL) under sparse rewards presents a fundamental challenge due to limited exploration and insufficiently coordinated attention among agents. To address this, we introduce the Focusing Influence Mechanism (FIM), a framework that drives agents to concentrate their influence to solve challenging sparse-reward tasks. FIM first identifies Center of Gravity (CoG) state dimensions, inspired by Clausewitz’s military strategy, which are prioritized because when they include task-relevant variables, their low variability can block learning unless agents sustain influence. To encourage persistent and synchronized influence, FIM then focuses agents’ attention on these CoG dimensions using eligibility traces that accumulate credit over time. These mechanisms enable agents to induce more targeted and effective state transitions, facilitating robust cooperation even under extremely sparse rewards. Empirical evaluations across diverse MARL benchmarks demonstrate that FIM significantly improves cooperative performance over strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Focusing Influence Mechanism (FIM), a novel framework for cooperative multi-agent reinforcement learning (MARL) in sparse reward environments. Drawing inspiration from Clausewitz's military theory of "Center of Gravity" (CoG), the authors propose a principled approach to identify state dimensions that exhibit low variability under typical agent behaviors but are critical for task completion. The framework consists of two key components: (1) **State Focusing Influence (SFI)**, which uses entropy-based criteria to automatically select CoG dimensions and designs counterfactual intrinsic rewards to guide agents to influence these dimensions, and (2) **Agent Focusing Influence (AFI)**, which employs eligibility traces to maintain persistent and synchronized attention across agents on shared CoG dimensions.

The method is evaluated on three benchmarks: a toy Push-2-Box environment, SMAC (StarCraft Multi-Agent Challenge), and Google Research Football (GRF). Results demonstrate that FIM consistently outperforms strong baselines including QMIX, LAIES, MASER, CDS, FoX, RODE, and QPLEX across all scenarios. The authors provide extensive ablation studies showing the necessity of both SFI and AFI components, as well as analysis of hyperparameter sensitivity.

**Key Contributions:**
1. A novel conceptual framework inspired by military strategy for MARL coordination
2. An entropy-based method for automatically identifying hard-to-change but task-critical state dimensions
3. A eligibility trace mechanism for sustaining coordinated multi-agent attention
4. Comprehensive empirical validation across diverse MARL benchmarks with strong performance gains

### Strengths
### 1. **Novel and Well-Motivated Conceptual Framework**
The application of Clausewitz's "Center of Gravity" concept to MARL is creative and well-motivated. The paper clearly articulates why identifying state dimensions that are "stable under typical behaviors but require coordinated effort to change" is important for sparse-reward cooperation. The Push-2-Box example (Figure 1) effectively illustrates the problem and solution.

### 2. **Principled and Automatic State Selection Method**
Unlike prior work (e.g., LAIES) that manually selects task-relevant features, FIM uses an entropy-based criterion to automatically identify CoG dimensions. The normalized temporal change metric (Eq. 3-5) provides a theoretically grounded approach that:
- Accounts for magnitude differences across dimensions through normalization
- Distinguishes between "frequently changing but trivial" vs "rarely but meaningfully changing" dimensions
- Requires no domain knowledge or manual feature engineering

The ablation study (Appendix I) convincingly demonstrates that entropy-based selection outperforms naive alternatives (no-selection, manually-selected EFI, and least-change LFI).

### 3. **Elegant Integration of Eligibility Traces for Multi-Agent Coordination**
The use of eligibility traces to promote persistent and synchronized influence (AFI) is intuitive and effective. The mechanism naturally handles target switching (when a focused object becomes unreachable) and enables sequential commitment to different CoG dimensions. The visualization in Figure 3 and Figure 7(b) clearly shows how this mechanism induces coordinated behaviors like "focus fire" in SMAC.

### 4. **Comprehensive Experimental Validation**
The experimental evaluation is thorough:
- **Diverse benchmarks**: Push-2-Box (toy), SMAC (8 challenging maps), GRF (8 scenarios including full-field variants)
- **Strong baselines**: Compares against 7 recent methods (LAIES, MASER, CDS, FoX, RODE, QPLEX, QMIX-DR)
- **Consistent improvements**: FIM achieves the highest success rates across all environments
- **Detailed analysis**: Includes trajectory visualizations (Fig 7, 13), entropy analysis (Fig 11, 12), and ablation studies
- **Generalization**: Extended evaluation on SMACv2 and MPE (Appendix H) demonstrates broader applicability

### Weaknesses
### 1. **Strong Dependence on Initial Policy Quality (Critical Issue)**
The most significant limitation is that CoG dimensions are identified using only 100K episodes from an **initial random policy** and then **remain fixed** throughout training. This design choice has several concerning implications:

**a) Exploration Bias:** If the initial policy fails to explore certain regions or trigger specific state transitions, dimensions that are actually critical may be missed entirely or assigned zero entropy (and thus excluded from CoGδ). For example:
- In a complex environment with conditional mechanics (e.g., "enemy shields only activate when health drops below 50%"), if the initial policy never reduces enemy health sufficiently, shield values would appear static and be excluded.
- This creates a chicken-and-egg problem: you need good exploration to identify important dimensions, but you need to identify important dimensions to guide exploration effectively.

**b) Distribution Shift:** As training progresses and agents learn better policies, the state visitation distribution changes dramatically. Dimensions that were hard-to-change initially may become easy later, and vice versa. The fixed CoG cannot adapt to this shift. While the authors acknowledge this limitation (Section 4.2, line 256-258), they do not address it in their experiments.

**c) Limited Evidence for Dynamic Settings:** The authors claim the framework can be extended to dynamic CoG updates but provide no experimental validation. Given that all tested environments (SMAC, GRF) have relatively static task objectives, it remains unclear whether FIM would work in domains where critical dimensions genuinely evolve over time (e.g., multi-stage tasks, curriculum learning scenarios).

**Suggested Improvement:** The paper would be significantly strengthened by:
- Ablation study showing impact of using policies at different training stages for CoG estimation
- At least one experiment with periodic CoG re-estimation (e.g., every 500K timesteps)
- Analysis of how CoG dimensions change (or remain stable) when estimated at different points

### Questions
1. **Initial Policy Bias and Dynamic CoG:**
   - What happens if the initial random policy never discovers certain critical state transitions? How would you detect and recover from such failures?
   - Have you tried estimating CoG at different training stages (e.g., after 500K, 1M, 2M timesteps)? How much do the selected dimensions change?
   - Can you provide experimental validation of the claimed capability to "periodically update β" (line 256-258)? Even a single experiment demonstrating this would strengthen the paper significantly.

2. **Hyperparameter Sensitivity and Transfer:**
   - Can you provide principled guidelines for setting δ, α, and η in new environments? For example, should δ be normalized by the average entropy across all dimensions?
   - How many hyperparameter configurations did you try per environment? How does this compare to the tuning cost of baselines?
   - Have you tested FIM on a completely new environment (not in the paper) to assess transfer difficulty?

3. **Marginal vs Conditional Entropy:**
   - Can you provide analysis or empirical evidence showing when marginal distribution p(Δd) is a good approximation of conditional p(Δd | st)?
   - In GRF, goalkeeper position is selected as CoG, but presumably it only matters when agents are near the goal. Does the marginal approximation adequately capture this context-dependence?

4. **Comparison with LAIES:**
   - In your experiments, did you use the original external state features from LAIES's paper, or did you select features yourself? Please clarify in the revision.
   - Why does LAIES fail on 27m_vs_30m and corridor specifically? Can you provide empirical analysis (e.g., which features did LAIES influence, and why was that suboptimal)?
   - What is the result of using FIM's CoG selection with LAIES's influence mechanism (without AFI)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
reinforcement learning in sparse reward settings. FIM selects low-entropy Center of Gravity (CoG) state dimensions for agents to collectively influence via counterfactual intrinsic rewards and eligibility traces. It is evaluated on Push-2-Box, SMAC, and GRF tasks, demonstrating superior performance over standard value decomposition methods (QMIX, QPLEX) and recent methods designed for sparse reward scenarios (LAIES, MASER, CDS, FoX, RODE).

### Strengths
1. Principled approach: The paper provides a clear motivation with entropy-based selection that requires no domain knowledge. Push-2-Box serves as an insightful example, and trajectory visualizations reveal interpretable behaviors such as focus-fire and goalkeeper disruption that align with human strategies.
2. Comprehensive experiments: FIM is robustly compared against both general MARL methods and multiple sparse-reward-specific algorithms across diverse benchmarks. The results indicate strong improvements, particularly in extremely sparse settings where existing techniques struggle.
3. Thorough ablations: Experiments carefully demonstrate the necessity of both selective state targeting (SFI) and persistent coordination (AFI), with sensitivity analyses informing the effects of key hyperparameters.

### Weaknesses
1. Sparse reward-specific baselines in Push-2-Box: While QMIX is included as the main baseline in Push-2-Box, it is not designed for sparse reward environments. For stronger evidence, the inclusion of additional baselines explicitly tailored for sparse rewards—such as intrinsic motivation or curiosity-driven exploration approaches—would be valuable. This would clarify whether FIM's superiority is general or primarily relative to methods with limited exploration capacity.
2. Formal justification for CoG selection: The assumption that low-entropy dimensions are always task-relevant needs further theoretical grounding. It is intuitive for box position in Push-2-Box, but less so in SMAC scenarios. A formal analysis of when H(d)<δ aligns with critical state features, or oracle-based ablations, would strengthen this claim.
3. Counterfactual baseline comparison: Despite referencing COMA as a key counterfactual approach, direct experimental comparison is lacking. Including results for COMA, and disentangling the effect of state-dimension-level versus action-level rewards, would better quantify FIM's contributions.

### Questions
1. Disentangling SFI and AFI contributions: Could you isolate the benefits from selective state targeting and temporal persistence? Ablate the components by combining LAIES features with FIM's eligibility traces, and report performance on representative tasks.
2. The empirical results show that FIM selects low-entropy CoG dimensions (e.g., health and shield in SMAC) as key coordination targets, and significantly boosts team performance. However, could you provide more insights or visualizations into situations where the entropy-based selection might fail to identify truly task-critical dimensions, especially in environments where the most relevant features do not correspond to lowest entropy? Are there cases where FIM focuses on misleading or sub-optimal state dimensions, and how might this affect coordination? Can the CoG selection mechanism be adapted or combined with domain knowledge to improve robustness across highly diverse scenarios?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a intrinsic-based reward shaping method to solve MARL problems with sparse reward. In particular, the intrinsic reward is calculated according to a normalized state difference in each state dimension. States with low entropy will be taken extra care and to be influenced by the agents. Extra pieces are included, e.g., counterfactual rewards, and the method is benchmarked on various simulation environments. Simulations result show that the propose method outperform its baselines.

### Strengths
The idea of CoG State Dimension Selection is interesting. The motivation of using intrinsic reward is clear. The paper is easy to follow and derivations and method development are sound.

### Weaknesses
I am not fully convinced by the choice/design of CoG State Dimension Selection. In the experiments, the proposed method incurs much larger variance than baselines. The delta definition, i..e, eq(3), is behavior policy dependent. I am not clear how the behavior policy can be systematically selected for various tasks. As the delta value is the key of the proposed method, the authors need to conduct more ablation study of the choice of beta.

### Questions
In practice, what are the scales of the H(d) for different dimension? Will the set CoG change a lot during training? How does the choice of beta affect the set CoG, e.g.., more deterministic behavior policy vs random policy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses scenarios in which agents must influence specific state dimensions to accomplish tasks. When these critical dimensions exhibit little variation under normal behavior, agents may fail to discover important state transitions and become trapped in local optima. To tackle this problem, the authors propose the Focusing Influence Mechanism (FIM), which explicitly selects Centers of Gravity (CoG) state dimensions and leverages eligibility traces to enable agents to actively change these otherwise stagnant dimensions, thereby improving exploration efficiency. Specifically, FIM introduces three modules: a state-level focusing mechanism which detects and identifies CoG state dimensions, counterfactual intrinsic rewards which measure each agent's marginal contribution to influencing CoG dimensions, and agent-level focusing mechanism which maintains coordinated and sustained influence via eligibility traces. Experimental results across several multi-agent reinforcement learning benchmarks show that FIM can achieve more efficient cooperative performance compared to existing methods.

### Strengths
The paper conducts experiments across multiple scenarios and demonstrates improved empirical performance, which supports the practical relevance of the approach.

### Weaknesses
1. **Motivation example is not fully convincing**

    The Push-2-Box example used to motivate the work is a highly contrived setting. In this environment, the design of the intrinsic rewards indirectly informs agents about the task goal, leading to strong performance. As a result, the observed success does not convincingly validate the general usefulness of CoG-based design (see Questions 3–6 below).

2. **Intrinsic rewards are task-specific and may encode task goals implicitly**

    In Push-2-Box example, the x and y positions of the boxes are chosen as CoG state dimensions, and the intrinsic rewards are directly designed to encourage agents to modify these values. Naturally, this leads to successful task completion. However, this is essentially an implicit encoding of the task objective, which could even be considered “cheating.” One could directly define intrinsic rewards proportional to the distance between the box and its original location, achieving similar results. Consequently, this example does not convincingly demonstrate that CoG-based design inherently improves task performance. I recommend using examples in the Multi-Agent Particle Environment to illustrate the motivation.

3. **Notation is confusing**

    The preliminary section should clearly define state dimensions and associated symbols (e.g., $D$), since the expression $s_t = (s^0_t, s^1_t, \dots, s^{D-1}_t)$ is ambiguous, and $s^i_t$ is not defined. Similarly, the usage of $\beta$ lacks clear explanation (see Question 2  below).

### Questions
1. In Figure 1, if a single agent can push the box (although slower), why does vanilla QMIX fail to move the box? Theoretically, one agent pushing one step at a time versus two agents pushing simultaneously should require similar effort. Why does vanilla QMIX not achieve the performance shown in Figures 1(c) and 1(d)? The motivation and the core reason why focusing on specific state dimensions improves performance remain unclear.
2. In Equation 3, is $\beta$ an individual agent’s policy or the joint policy of all agents? My understanding is that it represents the joint policy, which seems inconsistent with the definitions in the preliminaries.
3. Do the state dimensions have concrete meaning after encoding? Why does keeping ${CoG}_\delta$ fixed during training still yield strong performance? This makes it difficult to argue that the experimental results are due to CoG discovery. 
4. On line 266, the authors state that agents frequently switch between two boxes and fail to push either to the wall due to multiple CoG state dimensions, motivating the AFI mechanism. Can this mechanism be considered general, or is it only effective for a small class of scenarios? For instance, if there are three agents and three boxes, and pushing the boxes fastest only requires two agents, the method may waste time and even perform worse than each agent pushing one box.
5. In the box pushing example, what is the difference between two agents jointly pushing a box versus each agent pushing a separate box? Why must two agents push together? The main challenge is for agents to understand the task goal; once understood, the two setups montioned above are effectively equivalent. The apparent effectiveness of your method in the box-pushing example arises because the intrinsic reward directly encodes the task goal.
6. The method focuses on encouraging agents to change dimensions that are difficult to influence under the current policy. What is the fundamental rationale for doing so? Why should this lead to better task performance? (Please avoid using the box pushing example, since in that case changing those two dimensions directly corresponds to the task.)
7. In SMAC, why are the rewards for winning and losing battles the same (+1 and -1), whereas in GRF they differ by a factor of 100 (+100 and -1)?

### Soundness
2

### Presentation
1

### Contribution
2
