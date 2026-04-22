# Learning to Cooperate with Emergent Reputation via Multi-agent Reinforcement Learning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 6

## Abstract
Reputation, the aggregation of peer assessments diffused through social networks, is a pivotal mechanism for promoting cooperation in social dilemmas ubiquitous to distributed multi-agent systems comprising agents with limited perception and cognitive capabilities.
Exploring efficient reputation systems, comprising reputation assessment rules and reputation-based policies, is a long-standing challenge.
Previous work assumes predefined reputation assessment rules or models reputation as an intrinsic reward to learn policies, compromising the methods' ability for generalization and adaptation.
To address this, we propose a distributed multi-agent reinforcement learning method $\textbf{COOPER}$ ($\textbf{COOP}$eration with $\textbf{E}$mergent $\textbf{R}$eputation), which jointly learns reputation assessment rules and reputation-based policies entirely from environment rewards.
Notably, leveraging the underlying mechanisms of reputation, we deliberately design the constituent modules of $\textbf{COOPER}$ and the data flows among them, overcoming the latency and noise in the feedback signal, caused by the deep entanglement between reputation and policy.
Experiments on the donation game and the coin game in grid world environments demonstrate that $\textbf{COOPER}$ effectively adapts to various existing reputation systems and co-players.
Furthermore, we observe the co-emergence of reputation norms and cooperation in self-play settings. 
These results hold robustly across diverse social network topologies, underscoring the generalizability and efficacy of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper considers the problem of learning reputation mechanisms as a way to promote cooperation in multi-agent systems. The authors propose a technique that receives messages from neighbours (as defined by a graph topology), aggregates them to arrive at a reputation assessment, and uses the reputation assessment as input to the agent policy alongside the standard agent observations. The authors carry out evaluations showing that the technique promotes cooperation, can play alongside existing reputation norms, and also leads to cooperation when playing against itself.

### Strengths
S1. The work tackles an intelectually interesting problem, is well-grounded in the literature, and aims for a worthwhile goal (designing reputation mechanisms that are highly flexible).

### Weaknesses
W1. The approach assumes a single, static graph which is highly limiting.

W2. The evaluation leaves much to be desired and, in my view, does not support the overwhelmingly positive narrative in the paper.

### Questions
C1. Regarding the methodology, in my view, the single, static graph assumption is highly limiting. The work cannot account for the agent population growing (e.g., new nodes joining the network). It also cannot generalise across different topologies. Both these aspects may be addressed by using graph neural networks (GNNs) for performing function approximation.

C2. The evaluation is very small-scale and, in my opinion,  there is insufficient evidence to support the claims.
- 4 runs is much too little to draw reliable conclusions in RL; see, for example [1] and [2]
- The experiments are carried out over few very small (n=10) graph instances.  The work should demonstrate the findings hold over substantially more instances (consider pairing a run seed with a graph generation seed as an alternative experimental design) and over multiple numbers of agents.
- A sensitivity analysis should be performed over the network generation parameters; otherwise the values selected seem completely arbitrary

C3. The method is compared with vanilla PPO. But why? COOPER itself is a maximum entropy actor-critic architecture, with the reputation module added in. PPO is much simpler, and the performance differences may well be explained by the difference in architecture. COOPER should be compared with the same RL approach but without the reputation module and associated losses.

### Smaller comments

C4. For the experiment in Section 5.3, could you clarify the setup? Are all agents COOPER agents?

C5. Typos etc: "collectively" -> "collective" (L39), "Individuals help others" -> Add e.g. "The fact that" at the start (L86)

C6. It's probably cleaner to present all loss components from equation 1 and only introduce the one in equation 3 after the others (since it's the loss for the critic)

C7. You should clarify what Markov Decision Processes the baselines operate with (or specify only their differences, e.g., how do observations differ?)

### References

[1] Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018, April). Deep reinforcement learning that matters. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).

[2] Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. (2021). Deep reinforcement learning at the edge of the statistical precipice. Advances in neural information processing systems, 34, 29304-29320.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces COOPER, a reputation based reinforcement learning algorithm which is capable of finding cooperative solutions to social dilemmas by learning reputation assesment rules that are then used to condition agent's policies.  The method is composed of three primary modules: 1. a gossip-based reputation assesment module that aggregates the reputations of other agents (considered within a social network), 2. an interaction-based reputation assesment module that uses previous interactions to generate a reputation score, and 3. a policy network that conditions on these two assesments plus the history of interactions. Compared to previous works, the constituent modules of COOPER allow for the data to flow amongst them and overcome deep correlations between the policy and reputation modules, that often lead to instability issues like failure to converge. COOPER is shown to be able to learn sensible reputation norms in iterated donation games and the more challenging Coin Game.

### Strengths
I really appreciate the authors' efforts on the presentation of the manuscript. This work is extremely well written, and simple to understand. The method is also novel, compared to existing algorithms in the social dilemmas literature. Having established that these are the main strengths of the paper in more detail:

1. **Clarity**: The method is simple and presented such that each component is fully explained.  A good amount of the paper is spent on establishing the foundations necessary to understand the method, including other reputation-based methods.

2. **Relevance**: Social dilemmas are an often neglected subset of games that ironically show up in all sorts of real multi-agent interactions. I appreciate any efforts spent on designing algorithms that are capable of learning strategies with good performance on social dilemmas, especially those that do not rely on heuristics. In that regard, COOPER is a method that relaxes many of the heuristics of previous works.

3. **Novelty**: To the best of my knowledge there are no methods with a similar modular structure (for reputation-conditioned policies) as COOPER in the multi-agent reinforcement learning literature and more specifically, in the reputation-based literature.

### Weaknesses
The main concerns that I have regarding this paper is that the evaluation of of the algorithm leaves major holes and undermines the real usefulness of COOPER. I understand that previous works may have similar evaluation structures, but in my opinion showing that an algorithm is able to find cooperation by itself does not constitute a strong enough result. This is trivially achieved by summing the rewards of the players and running standard reinforcement learning algorithms. I am more interested in the reputation rules that COOPER learns, so here is my criticism: the learned reputation-based cooperation pattern (shown in Figures 7 and 8) do not correspond to well known solutions to social dilemmas, like tit-for-tat, which are well known Nash equilibria points of the game. In particular, these figures show that for fully connected social networks the agents converge to anti-tit-for-tat, a solution that is not a Nash and does not maximize social welfare. When the social network has limited connections the policy learned is close to always cooperate(which is exploitable), with the major caveat that the agent is less likely to cooperate if previously both of them cooperated. In other words, it finds exploitable solutions that are also Pareto sub-optimal.

With such underwhelming results it is hard for me to recommend acceptance unless the authors can address my concern, or explain if there is a flaw in my reasoning/interpretation.

On a separate note, there is an extensive line of Opponent Shaping literature that attempts to tackle social dilemma games [1, 2], including the very same Coin Game used as the hard evaluation benchmark of this paper. It is surprising to me that none of this literature is cited and that COOPER is not compared against, at least, some of these methods.  

References:

[1] Foerster, J. N., Chen, R. Y., Al-Shedivat, M., Whiteson, S., Abbeel, P., & Mordatch, I. (2018). Learning with Opponent-Learning Awareness. In AAMAS 2018, 122–130.

[2] Duque, J. A., Aghajohari, M., Cooijmans, T., Ciuca, R., Zhang, T., Gidel, G., & Courville, A. (2024). Advantage Alignment Algorithms. arXiv:2406.14662.

### Questions
I am interested in comparing the performance of reputation-assessment methods in social dilemmas when compared to alternative approaches, so having said that,

1. Why did the authors not compare (baseline) with other works in the Opponent Shaping literature, which have shown to be capable of learning robust strategies in many social dilemmas including the Coin Game?
2. What is the authors' intuition on why the different social network topologies lead to substantially different reputation norms? How does for instance a sparsely connected network lead to a reputation norm that is more cooperative than that of a fully connected network (this result seems counterintuitive to me)?
3. How do the authors justify the convergence of COOPER to strategies that do not constitute Nash equilibria of the game?

I am willing to reconsider my score if some/most of these concerns are appropiately addressed.

### Soundness
1

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a distributed multi-agent reinforcement learning framework that jointly learns both reputation assessment rules and reputation-based policies directly from environmental rewards. The evaluation is based on the coin and donation game. The model used for the reputation-based mechanism is not completely clear - see comments below.

### Strengths
+ The problem of learning through emergent cooperation is an interesting one.

### Weaknesses
- The design of the loss function does not appear grounded on clear foundations, especially in terms of summative loss.
- How do you select the values of the weights? $\lambda_{env}$ and $\lambda_{conf}$? 
- The mechanism of gossiping is not described in sufficient detail. Is it just one hop? Is it synchronous? It is also very unclear how the system is bootstrapped. How do you initialise the actual trust estimation? What is the impact of this initialisation on the results?
- The authors should rethink the notation that appears rather convoluted. For example, it is not necessary to write that a function $\psi_{\theta}$ is a function of a function of the node itself and its neighborhood, since the neighbourhood can be identified univocally from the node itself.
- The authors say that the loss in Equation (4) captures the human tendency to consider peers’ point of view, but it is very unclear why this might help for “emergent” reputation in the first place.
- The authors do not consider the impact of network size on the performance of their solution. In fact, it seems to me that that is an important dimension, especially considering different networks structures with the presence of hubs, such as Watts-Strogatz ones.

- The update presented in Section 5.2 is very difficult to understand and not clearly justified.

- In Section 5.2.2, the authors say it stimulates cooperation, but I think it is difficult to make that claim without comparing the results presented in the paper against a situation in which the mechanism is not present.

- The selection of the comparators is very difficult to justify. For example, the authors compare with a vanilla PPO, which appears not suitable/meaningful (it is designed for something completely different).

- The authors say that the negative reward is amplified. It seems to me that the positive is amplified as well given the design of the system. This part is not sufficiently justified.

- The discussion about the “leading eights” is unrelated to the rest of the paper (see Section 5.3.1).

- The ablation study presented in Section 5.3.2 is very difficult to understand. In fact, if you remove the evaluation mechanism, there is no information to be shared in the first place, I would say.

- Overall, given the information in the paper, it is rather difficult to reproduce the proposed solution. The authors do not report sufficient information to reproduce the proposed solution in my opinion, for example, in terms of parameters used for the simulations.

### Questions
- The authors should provide more information about the gossiping mechanism and its implementation.
- The author should clearly describe how the “trust” of another agent is estimated before being sent around. This is quite unclear. 
- How is the system initialised?
- Can you explain the design of the gossip-based update (see Section 5.2)? How do you select $\delta$?
- The concept of self-play is very unclear in this context. Can you please provide more details?
- Why do you say that the negative reward is amplified and not the positive one (Section 5.3)?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents COOPER (COOPeration with Emergent Reputation), a novel distributed Multi-Agent Reinforcement Learning (MARL) algorithm designed to promote cooperation in social dilemmas by integrating emergent reputation mechanisms. The core contribution is the framework's ability to jointly learn a reputation assignment rule (reputation norm) and a reputation-based policy entirely from extrinsic environmental rewards. This approach addresses a key limitation of previous MARL methods that rely on predefined assessment rules (e.g., RR, IR) or use reputation as a fixed intrinsic reward signal (e.g., LR2), which often compromises adaptability and generalization. COOPER achieves this through a dual-module architecture: a gossip-based assessment component (ψ) aggregates neighbors' opinions, while an interaction-based assessment component (ϕ) refines beliefs based on direct experiences. A critical alternating optimization scheme manages the deep entanglement between the policy and reputation modules. Experiments conducted on the donation game and the coin game across various social networks (small-world, scale-free, fully connected) demonstrate COOPER’s efficacy in adapting to existing reputation norms and stimulating sustained cooperation in decentralized self-play scenarios. The results also offer insights into the co-emergence of cooperation and flexible, network-dependent reputation norms.

### Strengths
- Originality

The originality of this work stems primarily from successfully tackling the decentralized, joint learning of reputation assessment and reputation-based policy entirely from environmental feedback. Prior state-of-the-art methods typically assume a pre-existing reputation norm, which simplifies the learning process but results in a lack of adaptability when facing novel environments or unfamiliar agents. For instance, methods like RR rely on specific, predefined rules such as Stern Judging, while IR uses existing rule-based agents to foster norm learning. Furthermore, approaches like LR2 model reputation solely as an intrinsic reward signal, which can be sensitive to balancing extrinsic and intrinsic rewards and may lead to misaligned learning signals, particularly in challenging scenarios where unilateral cooperation is costly. COOPER differentiates itself by allowing the reputation norm to emerge from scratch within a fully decentralized setting, which is a significant conceptual advance in MARL.
The architectural design further contributes to originality by integrating two sophisticated components within the reputation assignment module: the gossip-based assessment (ψ) and the interaction-based assessment (ϕ). This dual structure is deliberately crafted to reflect how human societies balance aggregated social opinions with personal experiences, thereby enhancing robustness. The technical innovation to enable this co-learning is the alternating optimization scheme. By reversing the data flow during optimization, COOPER overcomes the latency and noise inherent in the feedback loop between reputation and policy, ensuring stable convergence guided purely by the extrinsic rewards.


- Quality

Technically, COOPER is founded upon the PPO algorithm but is substantially augmented with dedicated, differentiable reputation modules (ψ and ϕ). The framework is formalized through a partially observable Markov game, clearly defining the components necessary for reputation dynamics and network influence. The optimization scheme incorporates a theoretically justified conformity loss ($L_{conf}$), which acts as a graph-based smoothness prior to regularize ψ towards neighborhood consensus, improving sample efficiency without allowing gossip to entirely overpower direct experience, since ϕ retains the ability to refine beliefs based on fresh interaction data.
The empirical section is comprehensive, validating COOPER’s capabilities across critical axes. First, the adaptation experiments convincingly demonstrate COOPER’s robustness in non-self-play settings, showing that it can recognize and respond optimally to varied, pre-defined reputation policies like ALLD-RA and TFT-RA, often sustaining higher cooperation and reward levels than baselines. Second, the self-play results consistently establish COOPER's superior performance in establishing cooperation across small-world, scale-free, and fully connected network topologies, often succeeding where PPO agents converge to defection. Third, the ablation study provides crucial evidence that both the gossip (ψ) and interaction (ϕ) modules are necessary for robust cooperation; the removal of either component leads to a noticeable decline or complete failure of norm emergence. Finally, the detailed analysis of emergent norms lends credibility to the framework's ability to learn flexible, context-dependent social strategies.


- Clarity

The motivation for leveraging reputation is clearly linked to human social dynamics (gossip, long-term benefits) and the challenge of social dilemmas in autonomous systems. The mathematical formulation of the learning problem is clear, using the Partially Observable Markov Game (MG) framework.
Crucially, the paper successfully clarifies the complexity of the co-learning challenge and the devised solution. The distinct roles of the rollout phase (ψ→π→ϕ) and the optimization phase (ψ→ϕ→π) are explicitly noted, ensuring the reader understands how the method handles the feedback loop delay. The loss function components ($L_{env}$, $L_{conf}$, $L_{ent}$) are well-defined, with $L_{conf}$ being clearly positioned as a regularizer towards neighborhood consensus. Furthermore, the experimental section is highly accessible to the MARL community. The visualization of the emergent reputation norms using experience-to-action heatmaps (Figure 7, 8) effectively translates complex decentralized behavioral patterns into interpretable structures, explicitly showing how the learned norms differ based on network position (hub vs. leaf) and comparing them to theoretical models like the "leading eight" norms. The detailed appendices regarding implementation parameters (e.g., GAE lambda=0.95, PPO clipping=0.3) and network generation facilitate reproducibility.


- Significance

By enabling the decentralized, end-to-end emergence of reputation norms, COOPER addresses a foundational challenge in multi-agent cooperation, offering a mechanism that promotes farsighted behavior in mixed-motive games where myopic strategies typically lead to collective failure. This is evidenced by COOPER's ability to stabilize mutual cooperation, where baseline PPO agents fail.
The paper's findings establish a strong link between MARL agent behavior and established principles from evolutionary game theory. Specifically, the empirical observation that network heterogeneity (scale-free networks) supports the highest cooperation ratios, with "hub agents" acting as key cooperation anchors, directly aligns with theoretical expectations regarding network structure and cooperation promotion (e.g., Santos & Pacheco, 2005). The demonstration that a single COOPER agent can identify the reputation-based strategies of defecting populations (ALLD-RA) and stimulate a "cooperation cascade" that raises the average group reward is a powerful result, demonstrating real-world potential for social intervention in distributed systems. Finally, the approach provides a flexible framework that can adapt to diverse reputation norms and co-players, unlike rigid rule-based systems.

### Weaknesses
- Originality

While the core objective of joint learning is novel, certain aspects of COOPER rely on existing components or theoretical insights, tempering the claim of complete originality in all sub-modules. The underlying learning foundation is based on the Proximal Policy Optimization (PPO) algorithm, a standard technique, rather than a novel learning mechanism tailored exclusively for this reputation objective. The architectural separation into ψ (gossip/social assessment) and ϕ (direct interaction assessment) is conceptually motivated by long-standing theoretical models of reputation and indirect reciprocity in human societies.


- Quality

The fundamental reliance on truthful communication is a significant technical limitation: the model assumes that agents truthfully share their reputation assessments, and this assumption is neither relaxed nor tested in adversarial environments where agents might disseminate misinformation to manipulate reputations. Such a fragility limits the practical quality of the system in real-world deployments.

A second quality concern relates to the constrained memory capacity used in the empirical setup. Agents rely on a simple "memory-two" setting for interaction history, meaning reputation assignment is based solely on the last two interactions with a co-player. This simple constraint restricts the complexity of the reputation norms ϕ can learn and may fail to generalize when more sophisticated, long-term memory norms (e.g., those requiring knowledge of an agent's history over ten past interactions) are required to foster cooperation.


- Significance

The reputation metric is modeled as a simple scalar value $ξ_{i→j} ∈[−1,1]$. The authors identify this as a limitation, noting that it may lack the expressiveness required to capture complex behavioral nuances in more sophisticated environments, potentially resulting in oversimplified social evaluations. If reputation needs to be multidimensional (e.g., reputation for reliability, speed, fairness), the current scalar approach fundamentally limits the framework's direct applicability to many real-world coordination problems.

### Questions
1. The efficacy of COOPER assumes agents truthfully share their reputation assessments during the gossip phase (ψ). Given that reputation systems can be manipulated, how robust is COOPER to untruthful or adversarial gossip, and have the authors considered mechanisms to mitigate misinformation dissemination in future work?

2. In the ablation study, removing the interaction module (ϕ) entirely caused cooperation to "fail to emerge entirely," as updates remained unanchored. Does this imply that the conformity regularization alone is insufficient to provide a stable norm, even in highly connected networks (like the fully connected network)?

### Soundness
3

### Presentation
2

### Contribution
3
