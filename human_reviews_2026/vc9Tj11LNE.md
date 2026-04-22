# Stackelberg Learning from Human Feedback: Preference Optimization as a Sequential Game

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 10, 6, 6

## Abstract
We introduce Stackelberg Learning from Human Feedback (SLHF), a new framework for preference optimization. SLHF frames the alignment problem as a sequential-move game between two policies: a Leader, which commits to an action, and a Follower, which responds conditionally on the Leader's action. This approach decomposes preference optimization into a refinement problem for the Follower and an optimization problem against an adversary for the Leader. Unlike Reinforcement Learning from Human Feedback (RLHF), which assigns scalar rewards to actions, or Nash Learning from Human Feedback (NLHF), which seeks a simultaneous-move equilibrium, SLHF leverages the asymmetry of sequential play to capture richer preference structures. The sequential design of SLHF naturally enables inference-time refinement, as the Follower learns to improve the Leader’s actions, and these refinements can be leveraged through iterative sampling. We compare the solution concepts of SLHF, RLHF, and NLHF, and lay out key advantages in consistency, data sensitivity, and robustness to intransitive preferences. Experiments on large language models demonstrate that SLHF achieves strong alignment across diverse preference datasets, scales from 0.5B to 8B parameters, and yields inference-time refinements that transfer across model families without further fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Stackelberg Learning from Human Feedback (SLHF), a framework for preference optimization that models alignment as a sequential-move game between a Leader policy and a Follower policy. The authors propose StackelbergGDA, a two-timescale gradient descent-ascent algorithm to approximate the Stackelberg equilibrium. The framework naturally enables inference-time refinement through iterative sampling. Experiments on language models ranging from 0.5B to 8B parameters demonstrate that SLHF achieves strong alignment performance, with the Follower policy consistently outperforming RLHF and NLHF baselines.

### Strengths
- **Strong motivation and intuition:** The paper provides excellent motivating examples, particularly the Condorcet paradox analysis in Section 4.1, which clearly illustrates how SLHF, RLHF, and NLHF differ in handling intransitive preferences.

- **Practical value of inference-time refinement:** The Leader-Follower structure naturally supports inference-time improvement without additional training, which is valuable for LLM applications. The empirical results in Table 4 demonstrate impressive cross-model generalization.

- **Comprehensive experiments:** The paper includes thorough empirical evaluation across multiple model scales (0.5B to 8B parameters) and provides detailed ablations and additional results in the appendix.

### Weaknesses
### Major Weaknesses
1. **Mischaracterization and Missing Literature on NLHF** 

**Issue:** The claim in the introduction that "simultaneous play forces both players to optimize against a moving opponent which can hinder convergence" is not accurate. Recent NLHF works have established polynomial or even linear convergence rates to Nash equilibrium: https://arxiv.org/abs/2312.00886, https://arxiv.org/abs/2401.04056, https://arxiv.org/abs/2410.16714, https://arxiv.org/abs/2503.08942. The paper should acknowledge these theoretical convergence guarantees rather than suggesting NLHF inherently struggles with convergence.

**Critical omission:** A closely related work (https://arxiv.org/abs/2502.18099v2) that also studies Stackelberg games for LLM alignment was published over half a year ago and cannot be considered concurrent work. This significantly undermines the novelty claim. The authors must thoroughly discuss this work and clarify their contributions relative to it.

2. **Problematic Characterization of Mixed Strategies**

**Issue 1:** At the end of Section 3, the authors state: "when no action is majority-preferred the equilibrium necessarily involves mixed strategies. This inherent stochasticity can be undesirable in applications where consistency and reliability are critical."
This characterization is misleading for several reasons: (1) In RLHF/NLHF/SLHF with KL regularization ($\tau > 0$), the optimal policy is always a distribution over responses, not deterministic. (2) A "deterministic policy" in LLM is not well-defined, unless we sample with temperature 0, but the model still outputs a probability distribution.

**Issue 2:** The claim before Section 4.1 that "there exists a deterministic Stackelberg equilibrium" suffers from the same conceptual problem. With regularization (which the paper uses throughout), policies must be stochastic. The best response is essentially the RLHF solution when viewing win-rate as reward.

3. **Lack of Theoretical Analysis:** The paper provides no convergence guarantees for either Algorithm 1 or Algorithm 2. Key questions remain unanswered: Does StackelbergGDA converge to a Stackelberg equilibrium? What is the convergence rate? Under what conditions does convergence occur? Given that RLHF/NLHF methods now have established convergence theory, the lack of any theoretical analysis for SLHF is a significant weakness. The paper should at minimum discuss the challenges in proving convergence or provide experimental evidence of convergence behavior.

4. **Algorithm Presentation Issues:** Inconsistency: Algorithms 1 and 2 are essentially different algorithms (analogous to OGDA vs. OMWU, https://arxiv.org/abs/2006.09517). For consistency and clarity, I suggest replacing Algorithm 1 with a theoretical version of Algorithm 2.

### Minor Weaknesses and Questions

5. **Baseline choice:** The paper uses Nash-MD-PG as the primary NLHF baseline, which has been shown to converge extremely slowly (see Section 5 of https://arxiv.org/abs/2503.08942). How would SLHF compare against faster NLHF algorithms like those mentioned above?

6. The intransitivity analysis (57% of graphs contain cycles) is interesting but could be expanded—what is the typical cycle length? How does this compare to other datasets?

### Questions
Please address my concerns in the Weakness section. There is no other questions.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
The paper introduces Stackelberg Learning from Human Feedback (SLHF), a new framework that models alignment as a sequential two-player game between a Leader (the action proposer) and a Follower (the conditional refiner). Unlike RLHF (scalar rewards) or NLHF (simultaneous equilibria), SLHF exploits sequential asymmetry to capture richer preference structures and support inference-time refinement. The authors propose STACKELBERG-GDA to efficiently approximate equilibria and scale training to large LLMs (0.5B–8B). Empirical results show SLHF achieves strong alignment across diverse datasets, with Follower policies improving outputs even when transferred to unseen models.

### Strengths
This paper introduces an innovative game-theoretic Stackelberg structure for preference learning. 

The proposal is rooted in the existence of intransitivity in pairwise preferences. It proposes a rational computational solution that replicates the logic with additional transparency into the learning and inference process.

Experiments showed that it outperforms or matches RLHF/NLHF baselines across multiple datasets.

Some theoretical foundations are discussed, i.e., qualitative connections to RLHF and NLHF, constructive conditions for numerical approximation, standard regularity assumptions, an equilibrium analysis, and an optimization algorithm (Stackelberg-GDA).

Source code and assets are open.

### Weaknesses
The two-policy framework, i.e., Leader policy and Follower policy, increases computational and training costs.

### Questions
Despite the discussion of model non-transitivity in Appendix D1, can you elaborate further on the merits of intransitivity coverage, compared to real-valued reward models?

In the majority of RLHF literature, people rely on 'transitivity' assumptions for its simplicity, while in real-world datasets, binary reward models, e.g., the Bradley-Terry (BT) model, when explicitly used, are known to be subject to 'intransitivity' because they rely on scalar variables that assume all preferences are transitive.

For your information, the literature below studied representative preference datasets in the real world, where the 'transitive' relationship between preference annotations may not always hold. 
- https://arxiv.org/abs/2409.19325 (Duan et al., 2017)

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Stackelberg Learning from Human Feedback (SLHF), a new framework for aligning LLMs that models the problem as a sequential-move game between two policies: a Leader that commits to an initial response and a Follower that refines it. This sequential approach avoids the need for a single scalar reward model, unlike traditional RLHF, allowing it to better handle complex or intransitive preferences. The paper also proposes an algorithm, STACKELBERGGDA, to find the game's solution. A key advantage of this framework is its natural ability to perform inference-time refinement, where the Follower can be used to iteratively improve the Leader's output. Experiments demonstrate that the SLHF Follower policy not only improves upon its own Leader's outputs but also consistently refines and enhances the responses from other, independently trained models without any additional fine-tuning.

### Strengths
- Unlike standard RLHF, SLHF optimizes directly over pairwise preferences without collapsing them into a single scalar reward, allowing it to handle complex and intransitive preference cycles.
- The Leader-Follower structure naturally supports improving model outputs at inference time, as the Follower is explicitly trained to refine a given response, allowing for iterative improvement with more computation.
- By decomposing the problem, the Follower solves a simpler refinement task against a fixed action rather than a non-stationary opponent, leading to more stable learning.

### Weaknesses
- The method's success heavily relies on having a "well-specified and representative pairwise preference function, which can be unavailable. 
- The experiments suggest the method can be sensitive to biases in the preference judge (in this case, an "LLM-as-a-judge"). The authors attribute the gap between standard and length-controlled win rates to the judge model's "length bias," which the SLHF model may have learned to exploit.

### Questions
- In the practical implementation, the Leader and Follower share parameters. Could this limit the follower's ability?
- In Appendix D.3, this paper mentions that "increasing κ leads to a gradual decline in the Leader’s performance. While
the Follower benefits from increasing κ from 1 to 5". How to balance the performance of the leader and the follower? Which is more important in practice? 
- Regarding "refining outputs from other models", Does it imply the Follower learns a universal refinement rather than just a policy specific to its own Leader?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a stackleberg game formulation of the RLHF problem, in contrast to prior works which either use a bradley-terry model or search for a nash equilibrium.  The authors demonstrate that their stackleberg formulation is able to resolve common problems with BT-based models, i.e. cyclical preferences, while also allowing for further test-time adaptation using the follower model.

### Strengths
* The paper is generally easy to follow and can be understood. The overview of differetn approaches is nice too!
* The section demonstrating the different types of preference relationships that different models can address is very nice and useful! 
* The approach is well justified theoretically. 
* The experimental evaluation considers both preference dataset evaluation and general finetuning
* to my knowledge, using a stackleberg game for learning from feedback is novel.

### Weaknesses
* The experiment section lacks any ablations on the choices made. For example, how does the two-timescale schedule affect performance?
* The method seems like it will be computationally more expensive. 
* I am not sure why the stackleberg formulation makes sense. I can see how the nash formulation can resolve ambiguities in preferences vs BT, but realistically when would I want to have a leader and follower? Using the follower will double inference costs. 
* the gains of the leader vs the nash models seem marginal at best. This seems to indicate that a lot of the performance gains might be coming from just using more compute / tokens for a response i.e. adding context.
* The length bias seems really strong in the Alpaca results.


Nit:
* In eq 5, the order might be more intuitive if the leader is the inner optimization and the then follower moves after? the notation for eq is also not super well defined -- and it would be nicer if the symbols for the leader and follower were more clearly introduced.

### Questions
* How is the reference for the follower defined? Does this reference model even make sense if the prior model hasn't been trained as a follower?
* For a lot of LLM methods, compute matters. Could the authors comment on any difference in compute requirement vs nash vs BT model + PPO? What do results look like at compute parity?
* Baselines: could the authors comment on why a method like SPO was not relevant / considered as abaseline? It is also based on nash equilibrium?

### Soundness
3

### Presentation
3

### Contribution
3
