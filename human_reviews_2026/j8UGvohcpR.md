# Learning to Trust: Bayesian Adaptation to Varying Suggester Reliability in Sequential Decision Making

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Autonomous agents operating in sequential decision-making tasks under uncertainty can benefit from external action suggestions, which provide valuable guidance but inherently vary in reliability. Existing methods for incorporating such advice typically assume static and known suggester quality parameters, limiting practical deployment. We introduce a framework that dynamically learns and adapts to varying suggester reliability in partially observable environments. First, we integrate suggester quality directly into the agent's belief representation, enabling agents to infer and adjust their reliance on suggestions through Bayesian inference over suggester types. Second, we introduce an explicit "ask'" action allowing agents to strategically request suggestions at critical moments, balancing informational gains against acquisition costs. Experimental evaluation demonstrates robust performance across varying suggester qualities, adaptation to changing reliability, and strategic management of suggestion requests. This work provides a foundation for adaptive human-agent collaboration by addressing suggestion uncertainty in uncertain environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a POMDP-based framework that allows an autonomous agent to (1) maintain a Bayesian belief over discrete “suggester types” that encode unknown and possibly time-varying reliability, and (2) actively request suggestions via an explicit, cost-bearing “ask” action. Belief updates are performed with a factored MOMDP representation to keep computation tractable. Extensive experiments on Tag and RockSample show that the agent quickly adapts its trust when suggester quality drifts, and strategically limits costly queries. A final heuristic-suggester ablation demonstrates robustness to model mismatch.

### Strengths
Novel integration of latent suggester reliability and agent-initiated queries inside a single Bayesian decision-theoretic framework.
Sound modeling choice: MOMDP factorization keeps the hidden state small, enabling off-the-shelf solvers (SARSOP) to scale to the augmented state space.

### Weaknesses
Discrete-type assumption: real-world reliability is almost certainly continuous and context-dependent; the chosen five-point discretization may be too coarse and is not motivated by data.
Scalability concerns: experiments are limited to small toy domains; the hidden component Y×T is still |T| times larger, which will hurt solvers when |S| or the horizon grows.
Limited novelty in ask mechanism: “query=information-gathering action” is well-known in POMDP sensor management; the paper does not theoretically analyze value-of-information or provide new solver tricks.

### Questions
1. How sensitive are the policies to the granularity of T and to the exact numeric ask-cost used? Any theoretical analysis?
2. Experiments are limited to small toy domains, which is not convincing. Please outline how the same POMDP formulation would be instantiated when the action space is continuous (e.g., 2-D mouse drag, 6-DOF robot joint commands) and suggestions arrive as natural-language or GUI-event streams. Would you discretize the continuous space, or move to a continuous-state POMDP / POMDP-lite solver?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper considers integrating external suggestions (e.g. from humans) into autonomous decision-making. In this context, the paper proposes modelling (i) suggester’s suggestion to be distributed as a tempered action-value function with a (temperature) rationality parameter $\lambda \geq 0$, (ii) dynamic suggester type (that is characterized by discretized $\lambda$) as a lazy random walk, and (iii) incorporating “ask action” option to request suggestion.

### Strengths
The method section 3 is easy to follow and the proposed contributions/components are introduced clearly with motivations. The paper also experiments in the setting where the proposed suggester model is misspecified (Section 5.4).

### Weaknesses
The contributions/components (i)–(iii) listed in the summary box are somewhat orthogonal, especially (i)–(ii) relative to (iii). Without comprehensive empirical experiments demonstrating a significant performance improvement over justified baselines, the overall contribution looks like a sum of incremental components. Further, proper ablation studies are critical in this case to understand the strengths and weaknesses of the individual components (maybe Tables 1–2 may touch on this, but it is difficult to discern without a clear narrative thread in the main text).

Experimental Section 5 is insufficient: it lacks proper discussion of the hypothesis, baselines, and evaluation metrics. For this reason, it is difficult to judge (i) whether the proposed method preforms well overall, (ii) what are the components of the proposed method that contribute the most, i.e. ablation studies, and (iii) what is the trade-off between the increased computational complexity and the improvement in empirical performance.
 
“Results summarized in Table 5 indicate that incorporating heuristic-based suggestions within our noisy rational modeling framework significantly improved agent performance compared to scenarios lacking suggestions.” Without proper discussion of the experimental setup conclusions like that are difficult to judge. 

“Numerous simulations were conducted to ensure statistical robustness.” This is too vague, etc.

### Questions
As $\lambda$ is continuous, why not treat is as such rather than discretize? Is there some other reason to keep it discrete than convenience?

“…We address this through a two-stage approach: first solving the original POMDP without the ask action to derive state-action values, then using these values to parameterize the suggestion observation model…” Sounds computationally heavy, does it?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies autonomous agents operating in POMDPs who receive external action suggestions (e.g., from a human or another agent) whose reliability may vary over time. Prior work typically assumes fixed and known suggester reliability, which does not reflect real human behavior or real-world sensing systems. Main contributions are: 
- Model suggester reliability as a latent variable and infer it dynamically via Bayesian updates.
- Introduce an explicit ask action, enabling strategic querying of suggestions under cost.
- Demonstrate robust adaptation to varying suggestion quality and ability to avoid low-value queries.
- Show empirical results across Tag and RockSample, with both rational and heuristic suggesters.

### Strengths
- Novel Motivation: 
(1) Addresses a real and growing need in human-AI teaming: adapting trust to variable advice quality.
(2) Aligned with the trend of interactive assistance and trust calibration.

- Formulation:
POMDP and MOMDP are modeled in the scenarios: (1) Present a solid use of the MOMDP structure to efficiently manage the expanded state space introduced by modeling suggester reliability as a latent variable. (2) The Bayesian update mechanism for jointly inferring environment state and suggester quality is principled and well motivated.

- Experiments:
(1) The study covers a broad range of settings, including static, dynamic, and heuristic suggesters, as well as scenarios involving ask costs and limitations on querying. (2) It further provides informative ablations, such as fixed-λ models, discrete-type inference, and dynamic type transitions, helping isolate the contributions of each component. (3) Include relevant baseline comparisons: normal agents, naive fixed-λ agents, noisy-rational suggesters, and multi-type agents in both static and dynamic configurations

### Weaknesses
- Human study missing: For human-trust motivation, no human-in-the-loop experiments are conducted. Although this paper acknowledges this, it is still important for this paper.

- Scalability: Tag and RockSample are standard but small. What if (1) the larger POMDP domains, (2) higher-dimensional latent human models, (3) multiple suggesters or groups of helpers.

- Reliance on **known** Q-values: The ask suggestion model uses pre-solved Q values. What if (1) Q is inaccurate, (2) Q value needs to be learnt. Is it possible to apply to RL or online learning settings.


- "Does your AI agent get you? A personalizable framework for approximating human models from argumentation-based dialogue traces". This paper seems also estimating the belief.

### Questions
Your method discretizes suggester rationality (λ) into a small fixed set:
- Q1: How sensitive is performance to the choice of λ grid values (e.g., {0,1,2,5,10})?

- Q2: Would adaptive or continuous inference over λ (e.g., particle filtering or Bayesian regression) further improve performance?

- Q3: If λ lies between grid points, how does belief estimation degrade?


The ask-action model assumes access to accurate Q-values from the solved base POMDP.

- Q4: How robust is the ask mechanism when Q-values are approximate or learned online (e.g., under model mismatch or RL)?

- Q5: Could errors in estimated Q(s,a) lead to biased belief updates about suggester reliability?

Experiments: Tag and RockSample.

- Q6: How does the computational cost scale with the number of suggester types and belief complexity?

- Q7: Could the method scale to larger, continuous-state problems or multi-human settings?

Human-in-the-loop:
- Q8: Do you anticipate additional challenges when suggesters are real humans? How would you integrate explicit human feedback or confidence signals

### Soundness
3

### Presentation
3

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
The paper studies the human-AI interaction problem under a POMDP (or MOMDP) framework. Human, as the suggester, provides occasional suggestions to the AI (autonomous agent) in a sequential decision-making environment, and the AI can utilize the suggestions to refine its belief of the underlying state. The suggestion is also captured by a quality parameter to reflect different levels of confidence during the suggestion. The levels will also be used in belief updating and thus the decision-making of the AI.

### Strengths
The paper is well-written and easy to follow. The model is well explained and provided with nice intuitions.

### Weaknesses
My main concern is the contribution of the paper:

The paper should be viewed more as a "conceptual" work. As noted above, the model is newly proposed and well-explained, but I find it hard to apply it in a real-world scenario. For the following reasons:
- Solving such a model requires knowing a lot of parameters like the transition matrix, the noisy rational suggester model, etc. 
- Generally, the POMDP framework makes the model inapplicable to a real-world scenario with a moderate state space size.

The key model component is from (Asmar & Kochenderfer, 2022) and there is no algorithm specifically designed for the model. 
- Would there be algorithms that can utilize the model structure to solve the problem more efficiently?
- Any theoretical guarantee for the case if the model is misspecified or the parameter wrongly estimated?

### Questions
See above,

### Soundness
3

### Presentation
3

### Contribution
2
