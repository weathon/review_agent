## Human Reviewer 1

### Summary
This paper considers the two-agent IRL problem for reward inference. As a solution, the authors modify the non-stationary IRL approach by Ashwood et al. (2022) and propose an approximation of the goal map consisting of two marginal maps and an interaction map. Additionally, they consider different levels of cognitive hierarchy, i.e., how deeply the agents' actions are related recursively. The proposed method is validated on a simulated cooperative foraging task and further applied to human data on the hallway task and monkey data on the chicken task.

### Strengths
First, the problem considered in this paper, to do IRL for the multi-agent setting, is interesting and relevant. Due to the complexity, there is not much work in this direction and I appreciate that the authors venture into this direction.

As mentioned, applying IRL to multiple agents can become difficult because the joint state space, to which the rewards need to be assigned, grows exponentially in the number of agents. The approximation that the authors propose, to have two marginal maps and an interaction map, which are learned and combined, seems an interesting and clever idea.

Also, the problem of modeling and inferring the cognitive hierarchy is non-trivial and relevant. A further strength of the paper is, therefore, that they propose a model and experimentally try to infer the level.

Finally, the method was applied on real human and monkey data to prove the applicability of the method.

### Weaknesses
While the problem the authors tackle is relevant and interesting, the proposed solution is very incremental to the work of Ashwood et al. (2022). As the paper discusses, the joint state space of the MAIRL problem poses a difficulty to existing IRL methods, as the number of states scales exponentially in the number of agents. While this is, to my understanding, the main selling point of the paper, the solution the paper proposes is limited to two agents, and in their experiments only two-player experiments are considered.
The approximation of the goal maps still seems interesting and probably could be extended to more than two players, but the paper lacks an analysis of the limitations of this approximation.

Further, I found the abstract and introduction imprecise and kind of overstated concerning the objectives of the proposed method. I would have wished for a more detailed description, of what kind of environments can be considered, what properties can be inferred… Also in the following sections, I had the impression that clarity could be improved (see also my questions on this).

The considered hierarchical models of behavior are highly simplified and I am unsure how realistic they are. The first model (Eq. 243) only considers random actions of the other agent. For the second model (Eq. 246), one assumes that the other agent acts according to the first model. I also think that the description of these models could be improved, as it took me a while to understand them (and I am still unsure if I understood them correctly and how optimal policies are finally computed). I am also unsure how meaningful the results are to infer the model underlying behavior in the experiments with AIC.

I am also unsure about the results of the real human and monkey experiments. The hallway task seems to indicate the limits of the Euclidian interaction map, showing that for each experiment there is the need to manually design and test different kinds of interactions. In the monkey experiment, there are not even states and, therefore, the main motivation for the applicability of the approach, a large joint state space, is not fulfilled and the method therefore not needed.

The limitations are only very superficially discussed (limited discrimination between the two models and first-level reasoning). There are many assumptions and limitations of the method that could be discussed but are not, e.g. agent’s beliefs, knowledge of the position of other agents, number of other agents, and different interaction maps.

### Questions
In the abstract, line 31, you write “MAIRL offers a new framework for uncovering human or animal beliefs in social behavior”. Where do you infer the agent's beliefs?

In the introduction, line 47, you write “our novel approach promises to reveal the latent model of the world and the computation of value and mental states of multiple interacting agents in social behavior”. Where do you infer world models and mental states?

Line 67: “While MARL with recursive reasoning models social behavior using known value functions, real-world interactions among humans or animals often lack explicitly defined value functions.” What do you mean by this? Do you have a reference for this?

Line 94: “While fruitful, this approach has several drawbacks. First, there is a lack of consistent mapping between the inferred value function and the specific mental state of the agent such as goal…” Could you elaborate on this? Why is the inferred value function for a mental state not consistent?

Line 101: “Alternatively, based on MDPs, …” Are the previously considered approaches not based on MDPs?

Eq. 1: This equation looks odd to me. What about the dependence of a state on the past state and action?

Figure 1: Why does agent 1 have two individual policies, which are combined?

Line 170: “In the case of cooperative open arena foraging, we showed that the joint reward could be perfectly reconstructed” Where?

Line 159: Do you also consider in your algorithm $K$ different goal maps? On the other hand, if you don’t reuse this equation from Ashwood et al. (2022), how do you use $u$ as defined in Eq.2?

Is there a marginalization over $a_2$ missing in the equation of $P(a_1 | s)$ (line 243)?

Could you explain Figure 2F in more detail, as in the caption I found it confusing what is meant with “...other who is placed at the origin for plotting purpose and marked with an asterisk”?

Why is column distance the best for the interaction map for the hallway task? Intuitively, I would assume that collisions happen if both agents remain in the same row and therefore a row distance of 1 would be more critical than a column distance of 0?
How can you create and test different interaction measures for a new application?

Fig. 4C: Why do you estimate maps independently for success and failure trajectories? In a setting where you don’t know the value map you usually cannot tell whether it was a success or failure, but this is what you want to learn.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces a novel approach to understand social behavior by inferring latent beliefs and intentions of agents. MAIRL leverages probabilistic recursive modeling and joint value decomposition to model complex social interactions. The method is validated on simulated tasks and real-world datasets involving humans and animals, demonstrating its ability to uncover goal-directed value functions, interaction terms, and cognitive hierarchy levels.

### Strengths
1. The combination of probabilistic recursive modeling and joint value decomposition for inferring latent beliefs in social interactions is interesting.
2. The method is applied to diverse scenarios, including simulated tasks and real-world experiments, demonstrating its versatility and effectiveness.

### Weaknesses
The presentation, especially Figure 1, could be clarified. Figure 1 is confusing, specifically regarding the following points:
	•	Could you clarify the meaning of “map” in the figure and in the section?
	•	Are all the labels at the top of the figure meant to be “Agent 1,” but in different colors? If so, why use different colored blocks to represent the same meaning?

Some expressions in the formulation appear incorrect or unclear: Is the statement in line 225 incorrect, as $\hat{a}_2$ should represent an action instead of a policy?

This work includes human datasets in the hallway task, yet there is no mention of ethics approval or dataset accessibility. It’s unclear if the dataset is publicly available or was collected by the authors. An ethics statement would be beneficial for transparency.  It is better to provide following documents: the ethics committee that approved the study, the consent process for participants, and details on how other researchers can access the dataset.

While the use of MAIRL to enhance Theory of Mind (ToM) capability is intriguing, the core idea feels relatively straightforward and lacks significant novelty. The decomposition approach is basic, and the experiments are limited in complexity. Exploring a more innovative and meaningful decomposition method could improve the work. Additionally, testing in more complex experimental environments, such as the two-player Overcooked video game or real ToM test tasks (inspired by study in psychology) [1], would provide a more robust evaluation.

[1] Strachan, J.W.A., Albergo, D., Borghini, G. et al. Testing theory of mind in large language models and humans. Nat Hum Behav 8, 1285–1295 (2024). https://doi.org/10.1038/s41562-024-01882-z

### Questions
"theory of mind (ToM)." in second line of Introduction should be "theory of mind (ToM)".

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents an approach to modeling social behavior in multi-agent settings, leveraging ToM principles through a probabilistic recursive model of cognitive hierarchy and joint value decomposition. By applying multi-agent IRL (MAIRL), the authors aim to uncover agents' latent beliefs and intentions in complex social environments. They validate this model in both cooperative and non-cooperative settings, demonstrating its potential to reveal meaningful behavioral insights without explicit task rules.

The integration of cognitive hierarchy and value decomposition in multi-agent inverse reinforcement learning is a compelling approach to understanding social behavior.

The approach seems to be generally sound and well evaluated, and so, I mainly have a few fundamental questions. Please note, some of my questions are merely exploratory, but I'd appreciate your thoughts:
- It wasn't immediately clear to me that how do you determine the appropriate level of cognitive hierarchy to model for each agent, especially in scenarios with varied agent capabilities? For example, different agents may interpret or respond to social cues at different levels.

- Given the known computational load in multi-agent recursive reasoning (increased k in k-level reasoning), how does the computational cost of your approach scale with an increasing number of agents or cognitive levels?

- Where do the trajectories needed for inverse multi-agent reinforcement learning are assumed to come from? Reliable and representative trajectory data are essential, and more insights would be very helpful, particularly given the challenges of collecting expert data from humans in multi-agent settings (see [1] for instance).

[1] Seraj et al “Mixed-initiative multiagent apprenticeship learning for human training of robot teams”, NeurIPS 2023

- How can we be confident that the "latent beliefs" uncovered by your MAIRL model truly represent the agents' underlying beliefs, rather than merely approximating observed behaviors? Can you shed light and point to specific empirical evidence and discussions, maybe?

- I think in a strategic and recursive reasoning scenario, it would be naive to assume fully cooperative agents or fully observable states. Human social behavior often includes non-verbal signals, contextual factors, and sometimes deceptive strategies. How does your model accommodate or account for these in multi-agent environments, particularly in scenarios with partial observability or intentional misinformation, such as a Byzantine Generals Problem (see [2] for instance)? This would address limitations of a strictly goal-oriented approach in highly dynamic social contexts.

[2] Konan et al. “Iterated reasoning with mutual information in cooperative and byzantine decentralized teaming”, ICLR 2022

- Related to the above question, your approach assumes that agents act in a goal-directed manner with underlying rational beliefs, is that correct? How robust is the model then when applied to agents with less predictable, impulsive, or irrational behaviors (again, see [2] for examples), which are common in social contexts? It would be valuable to understand the model's adaptability in scenarios where agent behavior deviates from optimal or rational planning.

### Strengths
-- See above

### Weaknesses
-- See above

### Questions
-- See above

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper is about multi-agent inverse reinforcement learning (MAIRL). It applies an existing method for IRL to a multi-agent context in order to infer the extent to which agents take each other's goals into account in social interactions. By decomposing the joint policy of two agents in different ways, different hypotheses about the extent to which each agent takes the other agent's behavior into account. The method is validated on a simulated foraging task in a grid world and then applied to two behavioral data sets: one cooperative task with human participants and one non-cooperative task with monkeys.

### Strengths
The application of MAIRL in the context of cognitive science as an inference method from a researcher's perspective to estimate the costs of two agents seems like a substantially novel contribution to me. To my knowledge, most of the other work in cognitive science on IRL in a multi-agent context seems to be on modeling theory of mind of the individual agents, but not on using MAIRL as a tool to analyze collaborative behavior. The central new methodological trick of the paper is the decomposition of the reward function into the rewards of the individual agents and a collaborative term, which makes inference in a multi-agent IRL setting more tractable. The conceptual framework is appealing and the applications to behavioral data promise to yield potentially interesting insights about the extent to which agents collaborate. I also appreciate that the code is available as supplementary material.

### Weaknesses
I have two main criticisms about this paper. First, the discussion of related work seems incomplete and at times confusing. Second, the evaluation of the method suffers from the fact that it was only evaluated on a single task, of which the authors say that it is a "simple task [that] does not require a sophisticated mentalization process", which raises doubts about the central claim that the method infers something about the "levels of cognitive hierarchy". I expand on these points in detail below.

1. The discussion of related work in the introduction and related work section seems quite superficial. It makes it seem like there is no prior work on inverse reinforcement learning in a multi-agent setting. While I see the novelty of the proposed application of MAIRL as an inference method from the perspective of a cognitive science researcher, there is relevant technical work on MAIRL and other applications of IRL in a multi-agent setting in cognitive science.

   a. There is no discussion of how the proposed approach compares to any other technical work on MAIRL or IRL in multi-agent settings. Some references that seem relevant from a cursory search:

    - Natarajan, S., Kunapuli, G., Judah, K., Tadepalli, P., Kersting, K., & Shavlik, J. (2010, December). Multi-agent inverse reinforcement learning. In *2010 ninth international conference on machine learning and applications* (pp. 395-400). IEEE.
    - Waugh, K., Ziebart, B. D., & Bagnell, J. A. (2011). Computational rationalization: the inverse equilibrium problem. In *Proceedings of the 28th International Conference on International Conference on Machine Learning* (pp. 1169-1176).
    - Reddy, T. S., Gopikrishna, V., Zaruba, G., & Huber, M. (2012, October). Inverse reinforcement learning for decentralized non-cooperative multiagent systems. In *2012 ieee international conference on systems, man, and cybernetics (smc)* (pp. 1930-1935). IEEE.
    - Rabinowitz, N., Perbet, F., Song, F., Zhang, C., Eslami, S. A., & Botvinick, M. (2018, July). Machine theory of mind. In *International conference on machine learning* (pp. 4218-4227). PMLR.

    b. Also, almost no reference is made to other work from cognitive science or behavioral economics about goal-inferences in multi-agent scenarios. Some references that seem relevant:
    
    - Wu, S. A., Wang, R. E., Evans, J. A., Tenenbaum, J. B., Parkes, D. C., & Kleiman‐Weiner, M. (2021). Too many cooks: Bayesian inference for coordinating multi‐agent collaboration. *Topics in Cognitive Science*, *13*(2), 414-432.
    - Carroll, M., Shah, R., Ho, M. K., Griffiths, T., Seshia, S., Abbeel, P., & Dragan, A. (2019). On the utility of learning about humans for human-ai coordination. *Advances in neural information processing systems*, *32*.
    - Kuleshov, V., & Schrijvers, O. (2015). Inverse game theory: Learning utilities in succinct games. In *Web and Internet Economics: 11th International Conference, WINE 2015, Amsterdam, The Netherlands, December 9-12, 2015, Proceedings 11* (pp. 413-427). Springer Berlin Heidelberg.
      
2. The point in the related work section, which discusses drawbacks of POMDPs and introduces MDPs as the favored model raised multiple questions for me.

    a. "While fruitful, this [POMDP] approach has several drawbacks. First, there is a lack of consistent mapping between the inferred value function and the specific mental state of the agent such as goal." Why is there a "lack of consistent mapping" in the POMDP approach, and why is this not the case in MDPs? Please explain.

    b. Confusingly, the papers cited as references for the POMDP approach (Baker et al., 2009; Velez-Ginorio et al., 2017) do not even use POMDPs, but seem to model inverse planning in an MDP setting.

    c. More generally, modeling a situation, which actually involves partial observability, with a model that does not properly take partial observability into account might lead to wrong inferences about their goals (see e.g. Straub, Schultheis, Koeppl & Rothkopf, NeurIPS 2023). If an actual task involves partially observability and therefore is described well by a POMDP, why would it be advantageous to use an MDP instead?

3. John Maynard Keynes' (1937) "general theory of employment" is cited as a reference for the claim that "even human reasoning has a depth of one or two levels". To me, the depth of recursive theory of mind in humans seems like an empirical question. While Keynes might be motivated by assumptions about human cognition, I am not aware that he performed behavioral studies investigating this question. Please clarify.

4. The only example on which the algorithm is evaluated using simulated data is the "cooperative foraging task" in Section 4.1. At the end of the section, the authors state that it was not possible to distinguish the "level-1 theory of mind agent" from a "chance prediction agent" given the data. The suspected reason for this is that "this simple task does not require a sophisticated mentalization process" (l. 347). This is in contrast to the claim in the introduction that the method estimates the "levels of cognitive hierarchy between two agents". In light of this evaluation, we cannot be sure that the inference method works for more complex tasks that might require "sophisticated mentalization processes".

Minor points:
- "Note that a temperature term in not needed [...]" should be "is" (l. 186)
- Both the distribution of reward strengths in the simulation (Supplementary A.1) and the prior over map weights are called $\sigma_0$. 
- "uniformed sampled" should be "uniformly sampled" (l. 680)

### Questions
1. Why is the "egocentric agents" model not included in the model comparisons (Fig. 3A, Fig. 4B)?
2. The parameter $\beta$ in Eq. (2) does not show up again anywhere in the paper. Does it have the same function as the map weights that are later called $\alpha$?
3. On hyperparameter selection: "We picked $K = 1, η_α = 0.01, η_{map} = 0.005, λ_1 = 5, λ_2 = 1$ based on training stability, model evidence and map interpretability." (l. 670) - What does it mean to pick the hyperparameters based on map interpretability? How was the model evidence traded off against this subjective criterion?
4. The inference of the column distance as the interaction map for the hallway task in Section 4.2.1 raises two of questions.
    a.  While the column difference indeed distinguishes the states in Fig. 5C and 5D, the row difference is also different between the two plots. Would a model that maximizes row distance instead of minimizing column distance not explain the behavior?
    b.  While Fig. 4 lists "expert", "success", and "failed" trials, Fig. 5 only shows "expert" and "failed" trials. Why?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4