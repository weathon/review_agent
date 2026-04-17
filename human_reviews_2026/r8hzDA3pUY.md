# Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents

- Decision: Accept (Oral)
- Scores: 8, 8, 6, 4

## Abstract
Active reasoning requires large language model (LLM) agents to interact with external sources and strategically gather information to solve problems in multiple turns. Central to this process is belief tracking: maintaining an accurate representation of the underlying state and uncertainty in understanding and solving the problem. However, due to limited reasoning capabilities, LLM-based agents often suffer belief deviation: their internal beliefs drift from the true problem state, leading to loss of state awareness and uninformative or repetitive actions. Once this happens, errors compound in the trajectories used for reinforcement learning (RL), leading to misattributed credits and limited exploration. To address this issue, we propose to track belief deviation and develop $\mathbf{T^3}$, a simple yet principled method that detects excessive deviation and truncates training trajectories to suppress uninformative tail effects. Hence, $\mathbf{T^3}$ preserves credits for informative prefixes and systematically improves policy optimization. Across 5 challenging tasks, $\mathbf{T^3}$ consistently
enhances training stability and yields performance gains of up to 30 points while cutting token cost by up to 34%. These results highlight belief control as a key principle for building robust LLM agents capable of active reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel method, T3 (Truncating Belief-Trapped Trajectories), to mitigate belief deviation in LLM-based agents during multi-turn active reasoning under reinforcement learning (RL). The core idea is to detect when an agent enters a “Belief Trap Region” (BTR) where further reasoning becomes uninformative and truncate the trajectory early to preserve credit assignment for the earlier informative steps. Theoretical analysis establishes the adverse effect of BTR on gradient estimation, and experiments on five tasks (from AR-Bench and Multi-Turn Puzzles) demonstrate significant gains in stability, efficiency, and performance across multiple RL methods (PPO, GRPO, GSPO).

### Strengths
Novel Theoretical Insight: Identifies belief deviation and BTR as key bottlenecks in multi-turn RL for reasoning.

Simple & Effective Method: T3 integrates seamlessly into existing RL algorithms with minimal changes.

Rigorous Theory: Theorems formally show how uninformative trajectory tails can invert gradients and harm exploration.

Impressive Results: Up to 30% performance gain, and 34% reduction in token usage across various benchmarks.

Robustness: T3 generalizes well to OOD settings, LLM scales (3B–14B), and architectures (Qwen, LLaMA, DeepSeek).

### Weaknesses
(W1) Truncation proxy design remains task-specific and heuristic :While the theoretical formulation of T3 is grounded in the notion of epistemic stalling (i.e., halted belief progress), the practical implementation relies on task-specific heuristic proxies such as:
repeated “unknown” feedbacks in Situation Puzzles,
invalid guesses outside the hypothesis set in Guess Numbers,
similarity drops in Movie Recommendation.
These conditions work well for the selected benchmarks, but lack generality. In tasks with ill-defined or open-ended hypothesis spaces, such proxies may fail to accurately detect BTR (Belief Trap Region) entry, leading to either missed truncation or premature stopping.
my suggestion: The authors could explore more general-purpose or learnable truncation detectors, such as classifiers over internal LLM hidden states or CoT consistency metrics, to broaden T3’s applicability across domains
 (W2) Limited to sparse outcome-based reward setting:
T3 is explicitly designed for environments where only the final step yields a non-zero reward. While this aligns with many current RLHF-style settings, it raises concerns regarding:tasks with dense or shaped intermediate rewards, environments where step-wise feedback is available or desirable..
In such scenarios, truncating trajectories could remove valid informative signals from later steps, inadvertently harming credit propagation rather than helping it. T3’s compatibility with these more general reward formulations remains unexplored. Suggestion: Discuss how T3 would behave in dense-reward or hybrid reward settings, or propose adaptations for preserving mid-trajectory feedback signals.
(W4) Lack of analysis on false positive truncation and its impact.While the authors provide some ablations on hyperparameters (e.g., window size k), they do not explicitly model the cost of mis-triggered truncation or propose mitigation strategies.
What if the truncation proxy fires prematurely (false positive)?
Could this discard useful exploratory steps that would have led to reward?
Does it increase variance in gradient estimation during early-stage training?

### Questions
How does T3 behave under dense reward or shaped intermediate feedback?
Have you considered extending T3 beyond outcome-only reward settings?

Can the proxy condition for BTR be learned, rather than hand-crafted?
For instance, via classification over hidden states or CoT traces?

What is the impact of false-positive truncation in early training stages?
Does it increase gradient variance or harm exploration?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a method to provide early stop during the RL if the hypothesis space did not shrink by a meaningful amount. The paper proposes $T^3$ (Truncating Belief-Trapped Trajectories), a mechanism that uses task-specific proxy signals to detect and halt trajectories entering the "Belief Trap Region (BTR)". Experiments across five diverse reasoning tasks demonstrate that $T^3$ consistently improves varied RL algorithms (PPO, GRPO, GSPO) in both final performance and token efficiency.

### Strengths
1. The paper provides a strong theoretical grounding for why RL fails in long-horizon active reasoning, formally characterizing the BTR and proving how it inverts expected advantage in GAE, thereby corrupting gradient estimates.
2. $T^3$ is a straightforward, drop-in enhancement for standard RL algorithms (PPO, GRPO, GSPO) that yields significant gains with small change.
3. The authors include thorough analyses, including out-of-distribution generalization tests (where $T^3$ maintains robustness) and ablations of different proxy truncation conditions, confirming the importance of detecting BTR entry rather than indiscriminate truncation.

### Weaknesses
1. Dependency on Task-Specific Proxies: While the theoretical $T^3$ condition is general, its practical implementation relies on manually designing task-specific proxies (e.g., repetitive queries for Situation Puzzles, failure to reduce hypothesis space for Guess Numbers)999999999. Finding these proxies for novel, less-structured tasks might be challenging.

### Questions
1. In Figure 3, the first and fourth figures have different y-axis label than the second and third figures, why?
2. If we want to extend this to more real world cases, how would $T^3$ fit in these use cases?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper formalizes belief deviation in multi‑turn, active reasoning and defines a Belief‑Trap Region (BTR) where epistemic progress stalls. It proves that long, uninformative “tails” can flip the sign of early‑step advantages (advantage inversion), then proposes T³, an early‑truncation rule triggered by simple proxy tests of stalled progress in the reasoning trace. T³ is a drop‑in wrapper for PPO/GRPO/GSPO and shows consistent gains across five interactive tasks, smoother learning, and shorter responses.

### Strengths
Crisp failure‑mode lens. Clear POMDP framing with a truth‑anchored potential Ψ and a precise BTR definition; theory links tail length to advantage inversion (Theorem 2) 

Simple, general mechanism. T³ uses observable stalling proxies (e.g., non‑shrinking hypothesis sets, repeated Unknown judgments) so it can integrate with standard RL without changing optimizers 

The proposed method shows broad, consistent empirical gains. Improvements on CircuitDetection (CD), SituationPuzzles (SP), GuessNumbers (GN), PreferenceEstimation (PE), MovieRecommendation (MR) with 14/18 metric wins; OOD robustness; works across model sizes/types.

Stability & efficiency: Learning curves are smoother and responses shorter, implying fewer wasted tokens.

### Weaknesses
Theory–practice gap. Assumptions (e.g., value calibration to truth probability; linear update‑error growth) are strong and not empirically validated against measured beliefs/values. 

Privileged proxies. PE/MR use oracle preference similarity to trigger truncation, limiting deployability without ground truth. 
“Shorter is better” confound. Random truncation sometimes helps (Table 3), so more direct evidence of advantage inversion would strengthen the causal story. 

Tuning guidance. Heuristics for window size/thresholds are task‑specific; no adaptive rule is provided. 
Comparisons & scope. Limited real‑world tasks and reliance on an LLM judge in SP; baselines could be more apples‑to‑apples vs. open, strong models trained under similar budgets.

### Questions
No‑oracle proxies: How would you instantiate T³ on PE/MR‑like tasks without access to the true preference vector (e.g., entropy/disagreement/self‑consistency signals)? 
Direct test of Theorem 2: Can you measure early‑token advantages with/without truncation and show the predicted sign flips as tail length grows? 

Adaptive T³: Any online procedure to target a truncation‑ratio band (or validation return) and auto‑tune window/thresholds? Sensitivity to the SP judge identity/temperature? 

Actionable feedback

Add a one‑page algorithm box for T³ (inputs, trigger, where truncation enters GAE/PPO).
Provide equal‑length ablations to disentangle truncation vs. better credit assignment; include γ/λ/KL sensitivity.
Add a small calibration check: value vs. correctness reliability plots. Report token & wall‑clock to fixed reward thresholds.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets multi-turn active reasoning with LLM agents, where belief tracking often drifts, which produces redundant actions that poison RL credit assignment. It proposes T3 - early truncation of rollouts once belief-trap are detected via simple proxies (e.g., non-shrinking hypothesis sets, repeated “unknown” feedback, off-manifold guesses, or declining preference similarity). T3 is claimed to be a drop-in rule for PPO/GRPO/GSPO that preserves credit for informative prefixes and removes noisy tails. Across five tasks, it improves stability and OOD robustness, increasing performance gains with fewer rollout tokens.

### Strengths
the $T^3$ framework provides a useful lens to connect information gain with progress in active reasoning (though its usefulness is a bit questionable, see more below), and the proxy-based T3 is simple to implement and, empirically, shows consistent gains across tasks/algorithms

### Weaknesses
- while the BTR lens motivates truncation, Def. 2’s T3 condition does not give actionable guidance on setting $ k, \alpha, \beta $ beyond "detect stalling". The task-specific proxies in Sec. 3.1 are essentially heuristic and could be understood without Section 2. In other words, the T3 condition does not concretely guide how to pick the metric $d(\cdot,\cdot)$, the window size $k$, or the threshold $\Delta_{\min}$, nor does it yield principled settings for the concrete proxies ("unknown" counts in SP, hypothesis-set shrinkage in CD/GN, similarity trends in PE/MR).There is no quantitative link between $ \Delta \Psi$ and $ d (H_\tau, H_{\tau +1} ) $; can the authors articulate decision rules (e.g., choose $k$ to control an estimated upper bound on false truncations via concentration of an estimator of $\Delta \Psi$)?

- as to the two theorems established
  - assumption 1 appears a bit strong: it posits systematic error growth with uncertainty across all actions/observations, yet the experiments do not validate (or estimate) $m_\theta, c_0, U_0$.
   - the extra constants/conditions ($\eta$, $L_\pi$-Lipschitz policy, “non-degenerate observations”) appear task/model-dependent and unverified; the bound $U$ seems not computable from observable quantities.
  - theorem 2 motivates truncation but does not establish when actual policy gradients are meaningfully biased on the reported tasks.
  - the sufficient condition shows the possibility of a sign flip, but does not quantify when it occurs for real critics/discounts. 

- as to training dynamics/stability: figures show improved stability in several cases, but GRPO+T3 still collapses later in SP. The rationale for pairing PPO (CD/PE), GRPO (SP), GSPO (GN) is not explained; differences in optimizer/entropy/clip may confound stability claims. If T3 is “algorithm-agnostic,” a matched comparison across all tasks/algorithms would strengthen its stabilization role.

- as to OOD analysis consistency, CD OOD uses Qwen-14B (Table 2) while the main CD uses 7B; can the authors justify the switch? The narrative “too many references induce redundancy $\Rightarrow$ more BTR” in PE is plausible but unsubstantiated; no proxy trajectory statistics (e.g., stall frequency, hypothesis-shrinkage rates) are shown to support the causal link.

- authors need to be explicit about what default $\alpha$ and $\beta$, $k$ values are used for all 5 reasoning tasks. 

- regarding ratio of early-truncation vs. performance, the trends are confusing: authors report SP with $\alpha=0.9$, where the truncation ratio converges to 1 and performance increases, whereas PE with $\beta=0.8$ also $\rightarrow 1$ but performance decreases. Similarly, for CD, higher ratios ($k=1,2$) appear to hurt exploration, unlike SP. Is there actually a consistent correlation between performance and the truncation ratio? Readers need the rationale and a principled conclusion, not case-by-case explanations. A controlled sweep reporting correlation between ratio and performance, conditioned on proxy type, would clarify when "more truncation" helps vs. harms.

- in addition, regarding the summary claim in lines 418-420, can authors elaborate on this alignment: what theoretical quantities map to the chosen proxy thresholds and settings, and under what observable conditions (e.g., provable relation between $|H_t|$-shrinkage and $\Psi$-decrease in CD/GN)?

- some claims are casual: "This suggests that distillation can effectively boost the belief-tracking capability under finite state spaces"; that size/type differences “can be attributed to $m_\theta$" (lines 450-452) is speculative with current evidence. The first statement is based on one distilled LLaMA baseline on one task. It should be labelled as a hypothesis; broader study across tasks/models is needed.

- $m_\theta$ is introduced in Assumption 1 as a slope in the belief-update error lower bound, but later used to explain size/type effects qualitatively. can authors elaborate on it? How can we connect $m_\theta$ to measurable quantities (e.g., proxy stall rates) and estimate it empirically?

- minor presentation issues

  - authors need to list default ($k, \alpha, \beta$) per task and how you tuned them

  - define the “ratio of early truncation” in text (denominator, per-episode vs. per-step?)

  - in Fig. 3a,d the y-axis should read “Response length”

  - Table 1 should be placed near Sec. 3.3.1 analysis

### Questions
- is the ratio of early truncation ever defined/formalized in the text?
- what are variables $b, a, o$ in fig 1?
- How would section 2 choose ($k, \alpha, \beta$)? For instance, can we map $d(H_\tau, H_{\tau + 1})$ to an empirical estimator of $\Delta \Psi$ and pick $k$ so that the false-truncate probability is < δ (with a concentration bound)?

- Why does GRPO+T3 still collapse later in SP? Is this due to proxy false-truncation starving exploration, or value-function drift? Any mitigation (pro

- for binary similarity threshold 0.88 (PE): why 0.88?

### Soundness
2

### Presentation
2

### Contribution
1
