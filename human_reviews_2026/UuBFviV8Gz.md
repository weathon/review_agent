# Learning to Cooperate with Humans through Theory-Informed Trust Beliefs

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Real-world human--AI cooperation is challenging due to the wide range of interests and capabilities that each party brings. To maximize joint performance, cooperative AI must adapt its policies to the competence and incentives of its specific human partner. Prevailing approaches address this challenge by training on human data or simulated partners. In this paper, we pursue an orthogonal approach: grounded on theory from social science, we hypothesize that equipping agents with human-like trust beliefs enables them to adapt to human partners more effectively. We formulate the cooperative agent's problem as TrustPOMDP, a variant of POMDPs, and develop a trust model that captures three key factors known to shape human trust beliefs: ability, benevolence, and integrity (ABI). A key advantage of the approach is that it only requires minimal modifications to a POMDP agent. TrustPOMDPs can be trained with real or simulated partners, provided sufficient diversity in the three dimensions. Results from both simulated and human-subject experiments (N=106) show that TrustPOMDP-based agents adapt more rapidly and effectively to various partners, while baselines methods tend to over- or undertrust, reducing team performance. These findings highlight the promise of incorporating social science-informed trust models into RL agents to advance collaboration with humans.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the coordination problem in multi-agent reinforcement learning, by borrowing from literatures on trust in human-AI teaming, and constructing a novel trust-based POMDP formalism as well as a belief inference method. Experiments are conducted both in the popular simulation task of Overcooked, and with real humans.

### Strengths
The modeling of ABI presents a well-thought out application of the vast literature on trust to the RL setting. The methods for ABI inference and conditional-policy optimization are kept simple yet effective. The experiments, particularly the task and partner designs, are done well, and the statistical significance is convincing.

### Weaknesses
My main concern for this paper is that 1) the POMDP baseline and TrustPOMDP method perform very similarly, and 2) FCP and MEP severely underperform basic POMDP.
1) Given all the added mechanisms for ABI inference and conditioning, I would expect the performance gap to be way larger. 
2) It's hard to say how I would expect POMDP to compare to FCP and MEP, but I find it strange that just how severely FCP and MEP methods underperform basic POMDP. This leads me to think FCP and MEP have much left on the table in terms of hyperparameter tuning. 

I would be willing to raise my score if this main concern is addressed.

### Questions
- line 254: should the ground truth ABI labels be in {0,1} instead of [0,1], since they have been discretized to be binary? 
- Fig 5 and 7b: I recommend shifting  the X labels to the right a bit, so that they are centered with the bar plots.




On citation: 
It has been a while since I studied this field, but I believe there are a few key missing works that should be cited here. This includes the seminal work in [1], a recent work that studies trust in MARL for Hanabi agents [2], and the works of [3] that studied legibility and predictability. Although [3] is in the robotics domain, legibility and predictability are still notions adjacent to trust that I believe should be included in this work.


[1] John D Lee and Katrina A See. Trust in automation: Designing for appropriate reliance. Human factors, 46(1):50–80, 2004.

[2] H. C. Siu et al., “Evaluation of Human-AI Teams for Learned and Rule-Based Agents in Hanabi,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2021, pp. 16183–16195. 

[3] A. D. Dragan, K. C. T. Lee, and S. S. Srinivasa, “Legibility and predictability of robot motion,” in 2013 8th ACM/IEEE International Conference on Human-Robot Interaction (HRI), Mar. 2013, pp. 301–308. doi: 10.1109/HRI.2013.6483603.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper looks at the problem of incorporating trust considerations into the decision-making process of an AI agent. In particular, they take into account the ABI model and create a latent variable that corresponds to each dimension of this model in a POMDP representation of the agent decision-making problem. The values for these dimensions are inferred on the fly as the agent interacts with the human. The validity of their method is tested using both simulated experiments and with real user study and the initial results seems promising.

### Strengths
I believe the inclusion of concepts and insights from social science is a worthwhile pursuit. In regard to such efforts, starting from well-defined frameworks like ABI makes sense to me. I also appreciate the fact that the authors actually ran user studies to validate their proposed model.

### Weaknesses
Unfortunately, I have quite a few concerns about the current approach.

POMDP - First of all, POMDPs are not a good framework to capture human-AI interaction or multi-agent interactions in general. As the authors point out, while there are works that try to leverage POMDPs, these methods are far too limited and cannot capture many of the more important aspects of the human-AI interaction problem. For example, how humans would be actively reasoning about what the agent might be doing and trying to adapt to it. Also, the fact that the agent and human could have independent objectives, and in some cases, objectives that might conflict with each other. As such, I would recommend that the authors look at more general frameworks like I-POMDPs [1].

Assumptions about human inference - The method currently assumes that the method by which humans infer the ABI parameter is known upfront, and additionally that you can convert these dimensions into simple parameters of a model. While I am aware of some psychological evidence for the noisy rational model (though even that is known to fail at times), I didn't see any discussions about the other choices. Also, for the beta-distribution, the current reference points to a paper that shows that the distribution could conceptually be used to capture trust accumulation. However, I didn't see any user study data in that paper (maybe I missed it), which suggests that it is a descriptive model insofar as it can simulate how people's trust evolves.

Benevolence - Isn't the formulation of benevolence in individual time-steps a bit simplistic? As an extreme example, consider an agent that is following a policy that is optimal for both reward functions, versus one where the agent is going out of its way to help the human. In terms of perception, wouldn't the latter be perceived as being benevolent? However, in your current reward wheight scheme they might not be differentiated.
  
Minor Issue: Equation 7 seems to be incomplete

[1] Gmytrasiewicz, Piotr J., and Prashant Doshi. "Interactive POMDPs: Properties and preliminary results." International Conference on Autonomous Agents: Proceedings of the Third International Joint Conference on Autonomous Agents and Multiagent Systems-. Vol. 3. 2004.

### Questions
I would appreciate if the authors could respond to each of the points raised in the weakness section

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper deals with a very important research line and makes a valuable contribution towards cooperative AI through grounding in social theory and providing evidence of improved human-AI coordination. The integration of psychological constructs into a POMDP is novel and practically relevant. And their "TrustPOMDP" in general is shown to yield favourable results.

### Strengths
-This is a very important research line; we need more and more research and investigation in Human-AI cooperation. 

-The method draws on well-established social science theory, so-called the Ability, Benevolence, and Integrity (ABI) model of trust (Mayer et al., 1995). This connection bridges human behavioral theory and reinforcement learning (RL), offering conceptual interpretability uncommon in multi-agent learning.

-The paper also introduces  TRUSTPOMD, an extension to the standard POMDP by incorporating a latent trust-belief model, with only minimal additional variables per dimension,  yet still meaningfully alters the agent behavior.

### Weaknesses
-We need to see stronger evidence of generality beyond the commonly used "overcooked environment" as this is the only environment that the paper uses.  Also a clearer empirical comparison to baselines is required.  The authors did not include the ablated POMDP model in the evaluation with human participants but I would have liked to see, e.g.,  the effect of ABI inference on the subjective perceptions of human participants. As it stands now, it is difficult to argue that a plain POMDP model would not have achieved the same results in the human evaluation. 

Other suggestions to improve paper:

-Consider including one additional cooperative task (e.g., social dilemmas) to demonstrate domain generalization.

-Consider adding more complex examples of norm violations to test integrity, such as a partner who occasionally cheats or lies about completing a task. This would show how well the model handles deceptive behavior.

### Questions
Q1:Could you share any results or arguments that show that the ABI inference itself (rather than general POMDP adaptation) is what drives the higher trust and satisfaction reported by participants?

Q2:How does the model handle cases where a human partner changes behavior midway through a task, for example, becoming less cooperative or breaking norms?

Q3:Do you have any results as to what happens when you represent ABI as continuous values instead of binary ones?

### Soundness
3

### Presentation
2

### Contribution
2
