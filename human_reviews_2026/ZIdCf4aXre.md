# Energy Shields for Fairness

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4

## Abstract
Runtime fairness is not a one-time constraint but a dynamic property evaluated over a sequence of decisions.
  To ensure fairness at runtime it is necessary to account for past decisions, information neglected by conventional, static classifiers.
  Traditional fairness shields enforce runtime fairness abruptly,
  by intervening *deterministically* whenever a sequence of decisions violates the target for a running fairness measure. This motivates our *main conceptual contribution*: **energy shields**.
  An energy shield is a novel, lightweight, adaptive controller that monitors a sequence of decisions and intervenes *probabilistically* to ensure runtime fairness smoothly, by utilizing physics-inspired energy functions to nudge the sequence towards fairness:
  the more unfair the decisions, the stronger the nudging force becomes. This makes energy shields the **first** fairness shields to provide both *short-term safety and long-term liveness guarantees*.
  Safety ensures that the running fairness measure stays within a running target interval with high probability,
  and liveness ensures that the limit of the fairness measure lies within the limit target interval. 
  Intuitively, the short-term specifies the tolerated fairness values and the long-term specifies the desired fairness values. 
  We also provide a synthesis procedure for constructing the least intrusive energy shield for a given target specification, and demonstrate its efficiency experimentally. As a sanity check for the theoretical contributions, we evaluate our energy shields against existing fairness shields through the lens of short- and long-term fairness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes energy shields, a probabilistic, physics-inspired framework that enforces fairness at runtime by assigning higher “energy” to unfair decision trajectories and intervening with a probability proportional to that energy. Unlike prior deterministic fairness shields, the proposed approach offers both short-term (running) and long-term (limit) fairness guarantees. The authors provide theoretical results establishing exponentially decaying tail bounds for running fairness, convergence conditions for limit fairness, and a monotonicity property showing that steeper energy functions yield fewer violations. Leveraging this, they design a synthesis algorithm combining binary search and dynamic programming to derive the least intrusive shield satisfying both fairness requirements. Experimental results validate the practicality of the approach and show its advantages over previous fairness shields

### Strengths
1- The energy shield is the first probabilistic fairness controller capable of providing both running and limit fairness guarantees within one framework.

2- Offers provable probabilistic guarantees via exponentially decaying bounds on fairness violations and convergence proofs for long-term fairness stability.

3- Establishes that steeper energy functions reduce violations, forming the basis for an elegant synthesis algorithm using binary search and dynamic programming.

4- Balances fairness control and intervention cost, presenting a clear optimization objective for runtime fairness enforcement.

5- Connects algorithmic fairness with energy minimization principles and control-theoretic intuition, opening a new interdisciplinary perspective.

### Weaknesses
1- The decision process assumes a binary Bernoulli model with stationary probabilities, which limits realism for multi-class, contextual, or adaptive systems.

2- The experiments confirm basic properties but fail to explore scalability, robustness under non-stationarity, or performance on real-world datasets.

3- Theoretical analysis assumes specific functional forms (e.g., polynomial or exponential), yet no discussion is provided on how to choose or tune these for practical tasks.

4- The framework defines fairness in abstract mathematical terms; mapping it to standard group metrics (e.g., equal opportunity, demographic parity) is not demonstrated.

5- While rooted in formal verification, the paper does not clearly show how these results could integrate with modern fairness-aware learning or causal inference models.

### Questions
1- How can the proposed shielding mechanism be extended to non-binary or continuous decisions common in ML systems?

2- Can the energy function ζ be learned adaptively or parameterized using observed violations rather than predefined analytic forms?

3- Does the monotonicity property hold under model misspecification or non-stationary distributions?

4- How would the fairness targets (τ, S, L) map to conventional fairness metrics such as Equalized Odds or Demographic Parity in applied settings?

5- Can the proposed synthesis procedure scale to large decision horizons or online settings where dynamic fairness targets evolve over time?

6- Could the shield be integrated into reinforcement learning or policy optimization frameworks for sequential fairness?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a mechanism for synthesizing probabilistic energy shields for fairness in sequential decision-making. The setting is that there is a decision maker deciding true/false outcomes in a sequential fashion and one wants to enforce that after some burn-in time, some target of true/false decision ratios is maintained within some interval, and that in the limit the ratio converges to some given value. A shield is a monitor that occasionally nudges decisions to help the decision maker satisfy the fairness goal. The key gap to address is that deterministic shields can be "invasive/abrupt" and costly to execute as they require solving a dynamic program. The proposed approach is instead to have a probabilistic shield that only nudges with some probability, thus ensuring more "smooth" shielding. The key insight is to decide with what probability to nudge based on an energy function that captures how much the decisions are deviating from the target fairness goals. The results on two examples illustrate that the probabilistic controller achieves the desired goals compared to a deterministic one

### Strengths
- Lifting deterministic shields to probabilistic ones is a very natural direction 
- The paper is technically well-executed and the theoretical contributions seem non-trivial (note, I'm not an expert here, so I can't really check how hard/novel this part is). It provides a comprehensive theoretical analysis of the properties of the algorithm.

### Weaknesses
- The evaluation is a bit underwhelming. The paper evaluates on 3 very similar artificial benchmarks and mostly considers a very naive decision makers that just samples from p. I'm not sure what the benchmarks are for this setting.
- The evaluation discusses that that other shields over-intervene or are invasive. I couldn't find a definition of these concepts in a way that is measurable. While I see that the plots illustrate some phenomenon of this sort, is there a way to actually compute a metric?

### Questions
The notion of fairness in this paper seems more similar to calibration than fairness itself. Can the authors discuss how their definitions related to calibration? (e.g., this paper https://proceedings.mlr.press/v247/qiao24a/qiao24a.pdf)

Have you considered applying this approach to the predictions of auto-regressive language models (aka LLMs)? I can imagine a setting in which some set of words (maybe negative sentiments or known-GPT words like delve) should not be "overly used" and so they have a target probability. The shield would then monitor usage and occasionally "re-weigh" the LLM logits to ensure usage stays within target? In such a setting I would expect that your approach is more beneficial than deterministic or periodic monitors as those would make the output fluency (something measurable) worse.


MINOR/OTHER:
- No need to cite this, or do anything about it, but this line of work reminds me of older results in synthesizing controllers (and the idea of algorithmic improvisation). I can't find the paper that comes to mind on making sure a thermostat keeps temperatures within some bounds while in the limit reaching a desired temperature, but I found this maybe related paper (https://susmitjha.github.io/papers/iccps10.pdf). Perhaps the authors know these works, or if not, they might find this pointer useful

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces "energy shields," a new probabilistic approach to ensure fairness in sequential decision-making tasks. Unlike traditional deterministic methods, energy shields gently intervene based on physics-inspired energy functions, correcting unfair decisions with higher probability as unfairness increases. The authors define two fairness objectives, short-term running fairness and long-term limit fairness, and provide theoretical guarantees for both. They also propose an efficient synthesis method to identify the minimally intrusive shield and demonstrate experimentally that energy shields outperform existing deterministic fairness interventions.

### Strengths
1.	The method wraps any binary decision stream with a probabilistic controller that flips the current action with a probability derived from a bowl-shaped energy function. This yields smooth, minimally invasive corrections that gently steer outcomes toward a target without retraining models or exposing internals, making it easy to bolt onto existing systems.
2.	The paper derives explicit finite-time tail bounds for the probability and expectation of short-term fairness violations, providing mathematically rigorous safety guarantees. This substantially improves the theoretical credibility and reliability of the approach.
3.	Broader linkage and scope. This paper discusses how single-stream shielding relates to group fairness checks (e.g., demographic parity can reduce to controlling one Bernoulli stream) and connects short- vs long-term fairness to standard safety–liveness notions.

### Weaknesses
1. The paper’s empirical evidence comes entirely from simulated binary decision sequences rather than real-world datasets. This work is primarily theoretical and does not utilize datasets containing sensitive information, thereby confirming that no external real-world datasets were employed.
2. A limitation is the narrow choice of baselines. The paper evaluates energy shields only against two methods, which are both runtime-shielding approaches rather than broader fairness techniques. This restricts external comparability and makes it harder to assess how the method would compare to standard pre-processing, in-processing, post-processing, or other online controllers.
3. The authors may benefit from discussing how their runtime fairness-shielding method relates to existing online fairness-aware learning frameworks that explicitly optimize accuracy-fairness trade-offs during model training, such as [1] "FAHT: an adaptive fairness-aware decision tree classifier", and [2] "Preventing Discriminatory Decision-making in Evolving Data Streams".

### Questions
Please refer to the weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
