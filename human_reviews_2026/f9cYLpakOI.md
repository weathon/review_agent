# Endogenous Communication in Repeated Games with Learning Agents

- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Communication among learning agents often emerges without explicit supervision. We study endogenous protocol formation in infinitely repeated stage games with a costless pre-play channel. Each agent has a representation map that compresses private signals into messages subject to an information budget. Agents update strategies by no-regret learning with stochastic approximation and choose representation maps by a myopic objective that trades off predictive value and encoding cost. We provide three main results. First, if the stage game admits a folk-theorem set and the information budget exceeds a task-specific threshold, there exists a stable communication equilibrium in which messages are sufficient statistics for continuation payoffs. Second, when the budget is below the threshold, any stable equilibrium must be pooling on a finite partition that we characterize with a minimax information bound. Third, we give polynomial sample-complexity guarantees for convergence to an approximately efficient communicating equilibrium under mild regularity. Our analysis connects cheap talk, representation learning with information constraints, and multi-agent no-regret dynamics. The framework yields testable predictions for when emergent messages are interpretable, when they collapse, and how much data is needed for stable coordination.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors study a model in which no-regret learning agents are augmented with the ability to send costless messages to each other.

### Strengths
I think the intersection of agent communication and learning in games can produce interesting settings and research directions.

### Weaknesses
The paper is very poorly written. There are 10 references, some of which are only tangentially related, and none of which are even mentioned in the main body, unless I have missed something. The proofs are too informal and vague, and have several non-sequiturs. The setup is not specific enough. This is not a length issue either; the paper is only six pages long including appendix, and the extra length could easily have been used to provide much more relevant detail. These writing issues alone are enough to recommend rejection.

I implore the authors to add more detail. The setting certainly looks interesting enough that there could be some interesting results and analysis in this paper, but the writing issues meant that I gave up on attempting to parse the paper before being able to come to a complete understanding of what the claims and techniques are.

### Questions
None.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies pre-play communication in infinitely repeated games. Each agent observes a private signal and sends a discrete message via an encoder constrained by a mutual information budget. Policies are learned by mirror descent; encoders maximize expected continuation value minus λ\times MI. The authors define a “stable communicating equilibrium” where policies are best responses, encoders are budget-optimal, and learning converges. They show that: (1) if budgets exceed a problem-specific threshold κ*, value-sufficient messages enable efficient payoffs; (2) below threshold, any equilibrium pools signals into a finite partition bounded by exp κ, implying a welfare gap; and (3) standard no regret dynamics are sufficient to reach a near stable point with O(1/epsilon^2) data.

### Strengths
1. The paper poses a clear, meaningful problem and introduces a formulation that links repeated‐game incentives with information-constrained pre-play communication, with a notion of stable communicating equilibrium.

2. The thresholding results given by Theorems 1–2 is clean and interesting: when the information budget exceeds a problem-specific threshold value-sufficient messaging can implement efficient outcomes; when it does not, any equilibrium must pool signals, leading to an unavoidable welfare loss.

### Weaknesses
1. The writing is often unclear. Key terms such as the formal definition of V, the notion of Lipschitz continuity, and the exact meaning and role of the learning rate \eta are never properly defined. It’s also confusing to bundle assumptions about the game itself and the learning algorithm into one block. The reference to “standard folk theorem” should be made explicit rather than assumed.

2. The proofs are mostly brief sketches and difficult to follow. The theorems are not stated in a fully formal way, and several terms used in them are never clearly introduced.

3. The related-work discussion is thin. It mentions prior directions in broad terms but does not cite or compare against specific, closely related papers.

Overall, the paper is very hard to follow, especially for readers who are not already experts in all relevant literatures. Clearer structure and more careful exposition would make it far more readable.

### Questions
1. Could the authors clearly define all notation and formally state each theorem, giving precise definitions for verbal notions and complete proofs instead of sketches? The paper is quite hard to follow, and clearer formalization would make it easier to evaluate.

2. In Theorem 3, the learning-rate choice (\eta_t \propto t^{-1/2}) appears inconsistent with Assumption 1’s requirement that (\sum_t \eta_t^2 < \infty)?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes endogenous communication among learning agents in infinitely repeated stage games with a costless pre-play channel. Each agent compresses its private signal via an encoder subject to an information budget, then plays the stage game; policies are updated by no-regret learning, while encoders optimize a myopic value-minus-information objective.

### Strengths
1. The paper cleanly ties cheap talk and information bottlenecks: it formalizes value-sufficiency, defines a budget threshold, proves existence of efficient communication above the threshold, and a necessary pooling structure with an explicit welfare-gap lower bound below it. These results offer actionable predictions about when emergent messages become informative vs. collapse

2. The stability notion is coupled to no-regret policy updates and information-penalized encoder updates, with a convergence guarantee of samples under standard step sizes and ergodicity. The provided alternating scheme makes the framework concrete

### Weaknesses
1. The paper is incomplete, lack a great amount of details. The proof is only sketch.

2. Many assumptions are strong and unjustified.

### Questions
NA

### Soundness
2

### Presentation
1

### Contribution
2
