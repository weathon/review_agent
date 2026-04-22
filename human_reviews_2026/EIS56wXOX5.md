# Strategically-Linked Decisions in Long-Term Planning and Reinforcement Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Long-term planning, as in reinforcement learning (RL), is often hard to interpret as it involves strategies: collections of actions that work toward a goal with potentially complex dependencies. In particular, some actions are taken at the expense of short-term benefit to enable future actions with even greater returns. In this paper, we quantify such dependencies between planned actions with *strategic link scores*: the drop in the likelihood of an earlier action under the constraint that a follow-up action is no longer available. We use strategic link scores to (i) explain black-box RL agents by identifying strategically-linked pairs among decisions they make, and (ii) improve the worst-case performance of decision support systems by distinguishing whether recommended actions can be adopted as standalone improvements, or whether they are strategically linked hence require a commitment to a broader strategy to be effective. We demonstrate these use cases with maze-solving and chess-playing examples as well as simulated healthcare and traffic environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
# Summary

The paper focuses on quantifying the causal dependencies between actions on a plan (policy).It formalizes the concept of strategically linked actions and defines strategic link score of two actions (producer and consumer).

It is worth noting that causal links were introduced in planning more than 30 years ago [1].
It seems like the authors have a very simplistic view of planning. While I understand that your setting does not assume state features, consider for a moment state features and actions having preconditions (in the simplest form, which of the state features are true) and effects (change in state features), as in e.g., classical planning. Then, an action would "produce" a feature that some other actions along the way would "consume" (require in the precondition). But it does not have to be pairs of actions, in the sense that a single action can achieve precondtions used by multiple actions along a plan (e.g., two doors opened by the same key), there can be alternative plans of the same cost that would use different actions that consume the same precondition (like two alternative doors). My point is, this property is not a property of action pairs. Looking at action preconditions and effects makes you see it more clearly. There was quite a body of research in planning on causal links, might worth a look.
Another closely related topic is action justification on a plan (e.g., [2,3,4,5,6]), talking about the reason for an action to be on a plan (providing some precondition for at least one following action or achieving a goal fact). The literature describes perfectly justified plans as a plan that cannot be reduced by removing actions and keeping it a plan. It seems like the authors are looking for similar concepts.


# Soundness

The main idea, strategic link score, as defined in Equation (4) does not make sense for planning in general, for the reasons described above. For instance, it is not hard to craft examples where the score will not drop when only one "pay-off" action is blocked. Of course, there are families of problems for which that is not possible, but I am not sure how to characterize such problems. 

# Novelty

Novelty is limited, for the reason expressed above.

# Scholarship

[1] Weld, D. S. (1994). An Introduction to Least Commitment Planning. AI Magazine, 15(4), 27.
[2] Fink, E.; and Yang, Q. 1992. Formalizing Plan Justifications. In Proc. CSCSI 1992.
[3] Fink, E.; and Yang, Q. 1993.  A spectrum of plan justifications.  In Proceedings of the AAAI 1993 Spring Symposium,23–33.
[4] Lindner, F.; and Olz, C. 2022.  Step-by-Step Task Plan Explanations Beyond Causal Links. In31st IEEE InternationalConference on Robot and Human Interactive Communication, RO-MAN 2022
[5] Sreedharan, S.; Muise, C.; and Kambhampati, S. 2023. Generalizing Action Justification and Causal Links to Policies. In Proceedings of the Thirty-Third International Conference on Automated Planning and Scheduling (ICAPS).
[6] Salerno, M.; Fuentetaja, R.; and Seipp, J. 2023. Eliminating Redundant Actions from Plans using Classical Planning. In Proc. KR 2023, 774–778.

# Clarity

The presentation of ideas is sufficiently clear.

# Evaluation and Reproducibility

The evaluation is quite extensive, but meaningfulness of the results is questionable.

### Strengths
1. The authors present the ideas clearly
2. The paper include formal definitions
3. The paper presents a wide empirical evaluation

### Weaknesses
1. The related work ignores the literature on the same and related concepts that come from the field of planning
2. The authors present a very simplistic and unrealistic view of planning, as actions that are there to enable other actions in pairs
3. The main concept, "strategic link score" is not well-justified, as I mention above, as it is not a function of two actions
4. As a result of the former, the empirical evaluation does not provide meaningful insights

### Questions
1. Could you comment on the connection between your concept and the concepts from the planning literature above?
2. Could you comment on the soundness concerns?1. The related work ignores the literature on the same and related concepts that come from the field of planning
2. The authors present a very simplistic and unrealistic view of planning, as actions that are there to enable other actions in pairs
3. The main concept, "strategic link score" is not well-justified, as I mention above, as it is not a function of two actions
4. As a result of the former, the empirical evaluation does not provide meaningful insights

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of explaining RL policies at the planning level, and proposes the strategic link score, which captures the temporal dependencies of the state-action pairs of a policy in terms of set-up and pay-off. The paper demonstrates how the strategic link score captures such dependencies in GridWorld and Chess environments. The paper also applies the proposed score for action recommendation, where the proposed score enables an appropriate level of granularity, forming a nice middle ground between recommending atomic actions and recommending full policies. The paper further demonstrates the utility of the proposed score for general non-reward-based planners, e.g., traffic simulators.

### Strengths
- The paper is visually appealing and very well-written, and key concepts are explained very clearly.
- Claims are well supported by evidence through well-explained demonstrations in a number of environments.
- The strategic link score could be of interest to researchers in explainable RL.

### Weaknesses
1. There seems to be a critical issue with the proposed method, which is that the strategic link score seems to fail whenever there exists more than one possible future pay-offs.
  - To give a concrete example, consider a GridWorld environment similar to that in Figure 3a. But now, instead of having a single door, there are two neighboring doors unlocked by the same key leading to the shortcut. Now, if we block one of the doors, the agent can still access the other door, so picking up the key is still part of the optimal strategy. Then, the probability of picking up the key is not decreased, and the strategic link score is 0, failing to connect the action of picking up keys to either of the future doors. 
  - Then if we were to perform strategy-aware recommendation in this setting, it is possible for the agent to adopt the recommendation of picking up the key without adopting the actions of opening doors, causing a failure similar to pick-and-choose.
  - Whenever the future pay-off of a set-up can be realized in more than one way, the strategic link score fails to link the set-up to future pay-offs. This is because the score can only handle one-to-one correspondence of set-up and pay-off, and is incapable of linking a set-up to a combination of parallel/independent pay-offs.
2. The strategic link score seems to be limited to small tabular environments.
  - Assuming access to a planner (e.g., an RL algorithm) that returns a policy given an environment, to compute the strategic link score, you first use the the planner to return a policy, which is then used to generate a trajectory that is most likely under the returned policy. Then, for every state-action pair $(s, a)$ in the trajectory, you re-solve the environment while forbidding $(s, a)$ from being chosen by the policy. Then, the complexity of the method is $O(TP)$, where $T$ is the length of the trajectory, and $P$ is the complexity for the planner to solve the environment. In RL, for example, you would need to run an RL algorithm to solve MDPs $T$ times.
  - Such computation seems intractable for large MDPs that's costly to solve, and MDPs that have a long/infinite horizon. 
  - Also, can the authors comment on the method's application on continuous state or action spaces? Is there a way to forbid an action in a continuous action space?
3. In relation to the above weakness, limitations is under-discussed in the paper. 
4. Metrics of statistical significane (e.g., confidence intervals) are not reported in Figure 7.


Some minor issues

5. In the final paragraph on page 5, "In the second layout (Figure 4a)". Should be 4c instead of 4a.
6. On line 288, there's an unfinished sentence: "We can make this interpretation because"
7. Typo in the caption of Fig. 10: "When it comes to the decision of saying"    saying $\longrightarrow$ staying
8. First letter of "markov" should be capitalized in the references.

### Questions
1. In Figure 7, how is the baseline performance (black horizontal solid line) determined when you randomly generate the MDP? Is it the average of the performance of the 100 suboptimal agents? 
2. For pick-and-choose and strategy-aware recommendation, how do you pick which of the recommended (group of) actions to incorporate when the number of recommendations implemented is smaller than the number of recommendations.
3. I do not follow how, in Figure 12, the error in strategic link score can further go down when the error in reward function increases as stochasticity becomes the highest.
4. The paper writes: "This is by definition of link scores; every decision is strategically linked to itself with a score one." However, by Eq. 4, isn't the link score of a state-action pair $(s, a)$ to itself $\pi(a | s) - 0 = \pi(a | s)$? When $\pi(a | s)$ was not 1 to begin with, how is the link score to itself 1?
5. What is the shaded region in Figure 12?

### Concluding Comments
Despite the paper being very well-written, due to the critical issue (see Weaknesses 1), I have provided an initial score of 4.

### Soundness
3

### Presentation
4

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
The paper proposes a method to expose multi-step strategies in sequential decision-making systems by defining a strategic link score between pairs of actions in different states. A link score $𝑆_{𝑠,𝑎→\tilde{s},\tilde{𝑎}}$ measures how much the policy’s probability of taking an earlier “setup” action $(𝑠, 𝑎)$ drops if a later “payoff” action $(\tilde{s},\tilde{𝑎})$ is made unavailable. Intuitively: “Would the agent still bother doing step 1 if it couldn’t later do step 2?” 

The paper argues this reveals strategic dependencies between actions, in a way that reward attribution / value explanations don’t. The paper (1) visualizes these links to interpret behavior in GridWorlds, chess, and traffic and (2) uses them to build “strategy-aware decision support,” where instead of recommending individual actions to a human operator, the system recommends bundles of linked actions so that partial compliance is less dangerous. This is tested in “Shortcuts” and a discretized hypotension treatment environment that mimics ICU decision support. 

Finally, the paper claims these links can be estimated even without direct access to the original planner, either by intervening in a simulator (traffic) or via IRL on demonstrations.

### Strengths
### 1. Clear, appealing problem statement.
The paper frames a real gap: existing explanations in RL mostly tell me why an action is “good long-term,” but not “this action is only worthwhile if we’ll later do that other action.” The authors focus on action-action contingency, which is intuitive to humans (“I grabbed the key only because I planned to open the door”). This is a crisp interpretability goal.

### 2. Actionable downstream use.
The decision-support angle is compelling. It’s not just interpretability for storytelling; the paper operationalizes it by grouping linked actions into bundles and shows that this “strategy-aware” interface can improve worst-case outcomes when a human only partially follows the agent’s advice, especially in environments with fragile multi-step maneuvers (Shortcuts), and at least does not hurt in a clinically flavored hypotension scenario.

### 3. Breadth of domains.
The method is demonstrated in (i) toy GridWorlds with keys and doors, (ii) a structured traffic network, (iii) a chess sequence with castling and follow-up play, (iv) a simplified ICU hypotension management setup. That breadth helps argue generality and shows the method isn't tied to a single planner or reward function.

### Weaknesses
### 1. Novelty vs existing ideas is under-defended.
I am not fully convinced (yet) that “strategic link score” is fundamentally distinct from prior notions like (i) causal influence of future constraints on earlier policy choices, (ii) credit assignment / Q-value sensitivity, (iii) options / skills / subgoal discovery. One might say: “You’re just measuring how much I stop wanting setup action A if payoff action B is impossible,” which sounds like standard counterfactual policy sensitivity. The paper needs a sharper, explicit contrast with causal RL explanations and hierarchical RL. Right now, that contrast is mostly implicit and could be challenged. 

### 2. Scalability / practicality.
Computing the link score requires re-planning under counterfactual interventions that forbid specific future actions at specific future states. That’s cheap in tabular / toy planners and in the softmax-over-Stockfish construction, but it’s unclear how this scales to high-dimensional continuous-control agents or giant neural policies without a fast differentiable planner. There is no complexity analysis, runtime table, or approximation method.

### 3. Validation is mostly qualitative.
Evidence that high link scores correspond to “true strategy” is presented as visual heatmaps (e.g. keys↔doors, shortcut pre-commitment, chess piece mobilization) and narrative interpretation. There’s no quantitative benchmark like precision/recall of discovered links against ground-truth enabling relations in GridWorld / Shortcuts, no human annotation study (“does this look like a setup-payoff pair?”), and no false-positive / false-negative analysis. That leaves room for the criticism that this is anecdotal. 

### 4. Chess experiment credibility.
The chess case study may look less impressive to experts, because the “policy” is generated by softmaxing Stockfish evals over legal moves rather than reading Stockfish’s true multi-ply principal variation. So it is not obvious that it is revealing latent plans; it might just be reconstructing obvious tactical follow-ups visible from engine eval deltas.

### Questions
1. Positioning / novelty. How is the strategic link score not just causal influence / Q-value sensitivity / option precondition discovery? Can you provide (a) a small formal example where traditional credit assignment says two actions are related but your link is zero, and vice versa, and (b) a concise statement of what is genuinely new?

2. Computational tractability. What is the computational complexity of computing all pairwise link scores along a trajectory of length $T$? How many re-plans do you need, and how does this behave for large neural policies where replanning is nontrivial? Do you have an approximation approach (e.g. masking logits in a learned policy network instead of full replan)?

3. Quantitative evaluation. Can you report precision/recall or AUROC for recovering “true” enabling relations (e.g. key→door, shortcut-setup→shortcut-payoff) vs baselines such as (i) temporal proximity, (ii) pure co-occurrence frequency, or (iii) reward-drop ablations? This would turn your qualitative figures into measurable impact.

4. Robustness / false positives. Does a high link score always imply that the setup action is useless without the payoff, or can it also just mean that disabling the payoff collapsed a whole profitable branch of future behaviors that share structure with that payoff? In other words: how specific is S? Are there systematic false positives?

5. Clinical external validity. In Hypotension, the “reward” is a shaped physiological stability score rather than actual patient-centered outcomes, and the human decision-maker is simulated by randomly dropping some actions from the agent’s bundle. How confident are you that this result will generalize to (a) real ICU objectives like mortality and (b) real clinician partial adoption patterns (e.g. “I’ll titrate fluids but I won’t start pressors yet”)? Can you discuss limitations here explicitly, so readers don’t overinterpret?

6. Black-box setting. Section 6 is very interesting. Under what assumptions does the IRL-based reconstruction of strategic links match the ground-truth planner’s links (e.g. identifiability or coverage assumptions)? Right now Fig. 12 suggests performance can improve with more stochastic behavior, but this is only briefly described. Can you elevate and formalize this claim?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel interpretability metric, "strategic link scores," designed to quantify dependencies between actions in policies, $\pi$, derived from long-horizon reinforcement learning. The score is formally defined as the decrease in the likelihood of selecting an "enabling" action, $\pi(a|s)$, when a related, future "follow-up" action, $\tilde{a}$, is unavailable. The authors motivate this by noting that optimal policies, $\pi^*$, often involve short-term sacrifices (i.e., selecting $a$ where $r(s, a)$ is low) to enable long-term gains. The utility of this score is demonstrated through policy analysis in several tabular planning environments and within the Maximum Entropy (MaxEnt) Inverse Reinforcement Learning (IRL) framework.

### Strengths
The paper addresses the critical and challenging problem of policy interpretability in long-horizon sequential decision-making. The idea of quantifying the link between a "setup" action ($a$) and a "payoff" action ($\tilde{a}$) is an intuitive and valuable contribution to the field of Explainable RL (XRL).

The core contribution, the "strategic link score," is clearly and formally defined. This provides a concrete, computable metric for what is often not a very clear, qualitative discussion of "strategy."

The authors connect their proposed metric to an established field by showing how strategic link scores can be used to analyze the results of Maximum Entropy (MaxEnt) IRL, providing a practical use case.

### Weaknesses
The paper is not very clear on the computational expense of the calculation of the strategic link score. Can the authors elaborate on the computation cost and the increase in the complexity analysis? Possibly in $\mathcal{O}(|\mathcal{S}|, |\mathcal{A}|)$
 
I believe that the definition of set-up action and pay-off is not very rigorous and requires to be more mathematically grounded in order to increase the applicability of the method to more applications. Additionally, this definition should connect the actions clearly to the defined MDP.
 
Moreover, the limitations of the paper are not clearly stated. I believe the inclusion of the limitations would further improve the readability of the manuscript.

### Questions
Authors should include at least two baselines (state-of-the-art) to verify their methods against them; if such methods do not exist, it should be clearly stated.

The manuscript mentions that inverse reinforcement learning (IRL) is used for strategic link scores using reward-based modeling, but it is not clear whether the strategic link score is used as an analysis tool applied to the results of IRL or whether it is integrated in the maximum entropy IRL objective function.

The paper's core objective is to analyze the cause-and-effect relationship between a setup action ($a$) and a future payoff action ($\tilde{a}$). This raises the causal relation among the actions and is fundamentally a question of causal influence. However, the manuscript fails to connect this work to the well-established and highly relevant field of Causal Reinforcement Learning [2], [3]. The authors should position their work within this literature, compare their proposed score to existing methods in Causal RL for identifying action influence, and justify why a new metric is needed.

[1] Bertoin, David, Adil Zouitine, Mehdi Zouitine, and Emmanuel Rachelson. "Look where you look! Saliency-guided Q-networks for generalization in visual Reinforcement Learning." Advances in neural information processing systems 35 (2022): 30693-30706.
[2] Seitzer, Maximilian, Bernhard Schölkopf, and Georg Martius. "Causal influence detection for improving efficiency in reinforcement learning." Advances in Neural Information Processing Systems 34 (2021): 22905-22918.
[3] Madumal, Prashan, Tim Miller, Liz Sonenberg, and Frank Vetere. "Explainable reinforcement learning through a causal lens." In Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 03, pp. 2493-2500. 2020.

### Soundness
2

### Presentation
3

### Contribution
2
