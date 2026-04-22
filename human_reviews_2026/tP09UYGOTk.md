# Emergent Alignment Via Competition

- Avg Score: 6.00
- Decision: Reject
- Scores: 4, 4, 8, 8

## Abstract
Aligning AI systems with human values remains a fundamental challenge, but does our inability to create perfectly aligned models preclude obtaining the benefits of alignment? We study a strategic setting where a human user interacts with multiple differently misaligned AI agents, none of which are individually well-aligned. Our key insight is that when the user’s utility lies approximately within the convex hull of the agents’ utilities, a condition that becomes easier to satisfy as model diversity increases, strategic competition can yield outcomes comparable to interacting with a perfectly aligned model. We model this as a multi-leader Stackelberg game, extending Bayesian persuasion to multi-round conversations between differently informed parties, and prove three results: (1) when perfect alignment would allow the user to learn her Bayes-optimal action, she can also do so in all equilibria under the convex hull condition; (2) under weaker assumptions requiring only approximate utility learning, a non-strategic user employing quantal response achieves near-optimal utility in all equilibria; and (3) when the user selects the best single AI after an evaluation period, equilibrium guarantees remain near-optimal without further distributional assumptions. We complement the theory with two forms of empirical evidence: First, we perform simulations of the best-AI selection game using best response dynamics, which show that competition among individually misaligned agents reliably improves user utility when the approximate convex hull assumption is satisfied, but does not always when it fails. Second, we show that synthetically generated AI utility functions (produced via perturbations of the same prompt to evaluate instances on a movie recommendation (MovieLens) and ethical judgement (ETHICS) dataset) quickly produce a convex hull that contains a good approximation of a given utility function even when none of the individual LLM utility functions is well aligned.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a new approach for AI alignment in the presence of multiple AI agents, treating the problem as a Stackelberg game and providing well-thought definitions of equilibria. The proposed approach could have a great positive impact in terms of preventing a human user from being exploited by any one particular AI agent.

### Strengths
- The modeling and definition using Stackelberg games and utility theory are constructed well. 
- I'm not super familiar with this field, but the definition of what we hope to see in the metrics and the goal of having an utility lower bound is expressed clearly. 
- The key idea of utility being contained within convex hull and its relation to bayes-optimal actions is very interesting.

### Weaknesses
- The experimental results are convincing but rather limited in scope.The proposed approaches and game-theoretic formalisms are very interesting; having stronger sets of experiments can make the ideas more convincing. I would be willing to raise my score if this concern is addressed. This is also not my home field, so I am not sure what additional sets of experiments I could suggest to improve the experiment section..
- The method of constructing "aligned AIs with noisy approximations" by using perturbed prompts seems like a rather weak way of producing such agents. Are there any reward-guided or reduced-rationality LLM sampling approaches that could be done in place of this?

### Questions
see weaknesses

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper asks whether competition among misaligned AIs can deliver outcomes close to a perfectly aligned assistant. It models a user interacting with multiple agents as a multi-leader Stackelberg game (multi-round Bayesian persuasion) and proves three guarantees under an approximate convex-hull condition on utilities. The authors ran Simulations and synthetic-utility experiments  based on MovieLens, ETHICS setting and the reuslts suggest that with sufficient model diversity, user utility approaches the aligned benchmark.

### Strengths
1. The multi-leader Stackelberg framing in the problem of alignment is novel and the author provided an equilibrium guarantees which is conceptually interesting.
2. The author did a good job in writing Clearly. And good formalization adds weight. The including of nice tie-in to quantal response and information-substitutes is very good.
3. The simulations in the paper are extensive and well organized.

### Weaknesses
I have some questions about the assumptions and the simulations the authors made in this paper:

1. Pluralistic alignment: the paper centers on a single representative user. In the real world, users have conflicting utilities. A hull that’s “good” for group A may be bad for group B. Without a multi-user treatment (or distributional guarantees), the policy relevance is limited.

2. Misalignment is modeled as noise, it's hard to justify that they are real divergence. Experiment 1 creates ~100 “AI personas” by rephrasing the prompt, then shows a convex-hull combination can approximate the “human” utility. This mostly injects random variation, not strategic or goal-level misalignment. It shows “averaging over small perturbations beats any single noisy agent,” which is unsurprising and not strong evidence for robustness under real incentive conflicts (e.g., sales-maximizing pharma models). I would reconsider my score if the author can make a reasonable argument on this point or making alternative simulation of goal-level misalignment LLMs.

3. I think core assumption of Approximate Weighted Alignment is very strong. The theory needs the user’s utility to lie (approximately) in the convex hull of provider utilities. That implies misalignments diversify and cancel. In practice, providers may share systematic biases (similar data, incentives, or redteaming), so the user’s utility can sit outside the hull. Strategic misalignment may not wash out by averaging in this case.

4. I think my biggest worry about the claims in this paper is that the follower rationality assumption is too demanding. The main results still rely on a user who can parse strategies and best-respond (or commit to stylized bounded-rational rules). Real users are limited, distractible, and manipulable. Even the quantal response model remains idealized and presumes well-formed posteriors. The guarantees may not work under, say realistic cognition limits or pure lazyness which in most of practical cases, little users will compare multiple LLMs.

### Questions
Can you include experiments with strategic misalignment (objective functions that explicitly pull users to provider goals), not paraphrase noise?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles the problem where a user might interact with misaligned AI systems. Through the framework of Bayesian persuasion in a multi-leader Stackelberg game, it shows one overarching finding in three different variants and settings of the game: A user can determine an (approximately) optimal output (with regards to its own preferences) from the outputs of multiple misaligned AIs _as long as_ the users utility function lies in the convex hull of the AIs utility functions. With more AI systems, they argue, this condition will be more likely to be satisfied, and they show some experiments along those lines in a simple stylized setup with two datasets.

### Strengths
- It studies the problem in a fitting formal framework, and is able to derive results with a clear interpretation
- It is well-written with a great overview of results in Section 1

### Weaknesses
- The authors admit this themselves, and this is a theory paper first and foremost, but I have to reiterate that the experiments are very simplistic. As of now, your experiments have misalignment occur from paraphrasing, which to me resembles more like Gaussian noise-like misalignment. Are there stylized experiments you could run that cover strategic misalignment, resembling one of the two motivational applications you discuss , for example, about companies that are misaligned to drive their profits?

### Questions
In Proposition 1, you discuss a central condition that allows Alice to recover approximately optimal value in your later results, namely, that a perfectly aligned can make Alice learn her best Bayes action. Can you discuss this a bit further, specifically on how realistic it is for this condition to be satisfied? (Remark 2 already shows that it holds in settings where the message space is very rich.) 
Because in the small sample of Bayesian Persuasion work I've seen, the Stackelberg leader usually forces the follower to randomize. But that is also precisely because the leader is not perfectly aligned with the follower.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Τhe authors examine a setting where a user (human) interacts with multiple AI agents (AI models) with possibly different utilities (misallignment). In this setting there exists an underlying state **y** which both the user and the agents do not fully know but rather posses some information $x_A$ and $x_B$ respectively. 
The user converses with the models to decide on an action **a**. All the utilities depend on the chosen action **a** of the user and the true underline state **y**. The goal of the user is to maximize their utility $u_A = u_A(y, a)$. 
The authors model this interaction as a Stalckberg game with **k** multiple leaders (the AI agents) who first commit to some "conversation rule $C_{B_i}$, $i \in [k]$ and the follower (the user) who chooses a conversation rule $C_A$ then best-responds by choosing some deterministic decision rule $D_A$ (an action that maximizes their utility). Because $C_A$ and $D_A$ depend on the k rules $C_{B_i}$ the user observed, the authors focus their analysis on the Nash equilibrium $C_B^*$.   
Under the assumption that the AI models are interchangeable in the sense that different AI models induce the same joint distribution and that the agent's utility can be written approximately as a weighted sum of the utilities of the AI agents, they show that the user can achieve approximately optimal utility in the Nash equilibrium.

### Strengths
- The problem is generally well-motivated 
- The paper includes a deep theoretical analysis with game-theoretic principles. 
- The presentation of the model and assumptions is clear.
- I really appreciated the experimental section especially since this is a non-trivial setup to do experiments and because the analysis had these two assumptions that at first sight do not seem to hold trivially.  It is interesting to see how indeed for real datasets the convex hull of agents utilities can incur an alignment with the user's utility, as the number of participating agents grows.

### Weaknesses
Minors
1. It is a little bit hard to read the Identical Induced Distribution Condition in the sense that $\cal{I}$ was defined with three inputs, and here it has one input which is denoted as a tuple (although I understand that it is just $C_B^*$ without the i-th Bob strategy and so on). 
2. Beyond empirical evaluation is there another intuition behind how the weighted alignment condition occurs in real-world settings? Maybe the authors have in mind some literature that they came up with this idea.

### Questions
In the communication protocol it is mentioned that Alice observes the conversation rules of the agents (and that she also chooses one). Can you provide examples of what is this quantity in practice?

### Soundness
3

### Presentation
3

### Contribution
3
