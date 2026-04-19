# Risk-Sensitive Variational Actor-Critic: A Model-Based Approach

- Decision: Accept (Poster)
- Scores: 8, 5, 5, 8

## Abstract
Risk-sensitive reinforcement learning (RL) with an entropic risk measure typically requires knowledge of the transition kernel or performs unstable updates w.r.t. exponential Bellman equations. As a consequence, algorithms that optimize this objective have been restricted to tabular or low-dimensional continuous environments. In this work we leverage the connection between the entropic risk measure and the RL-as-inference framework to develop a risk-sensitive variational actor-critic algorithm (rsVAC). Our work extends the variational framework to incorporate stochastic rewards and proposes a variational model-based actor-critic approach that modulates policy risk via a risk parameter.  We consider, both, the risk-seeking and risk-averse regimes and present rsVAC learning variants for each setting.  Our experiments demonstrate that this approach produces risk-sensitive policies and yields improvements in both tabular and risk-aware variants of complex continuous control tasks in MuJoCo.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a risk-sensitiv RL algorithm based on the idea of RL-as-inference. It presents a thorough derivation and a diverse set of experiments that highlights how the agent can be tuned to be risk seeking or avoiding with a simple parameter.

### Strengths
Overall, the paper is easy to follow, if somewhat dense in places. The idea is straightforward an well executed, building on a well-established framework.

The experimental section is thorough and contains both interesting small scale and reasonable large scale experiments. It would have been interesting to see if more complex or large dynamics systems such as image-based observations pose a challenge to the approach (I assume it would due to the difficulty of parameterizing a dynamics model?), but this is clearly something that can be relegated to future work.

Overall, my review is somewhat shorter than normal because I genuinely can't find too much to criticize. I think it's well done!

### Weaknesses
Overall, the paper does not have any major weaknesses that I would consider crucial for the decision to accept or reject.

As pointed out, the paper is somewhat dense in its writing. Some of the mathematical formalism could potentially moved to the appendix to declutter the main body, but given that the ideas presented are relatively standard and don't require exotic notation, this does not detract from the readability too much. The main thing contributing to this is that inline equations are used somewhat frequently.

One point that caused me a bit of confusion was that beta acts somewhat unintutively in that small absolute values of beta correspond to more extreme risk-averseness/seeking behavior, while large values correspond to the standard setting. This is briefly pointed out, but it could be elaborated on a bit more so that readers don't stumble.

### Questions
For standard RL as inference, it is known that variational algorithms produce optimistic policies if the transition dynamics are learned and optimized over. This is acknowledged in Section 3, yet I would encourage the authors to provide a more in-depth discussion and maybe some experiments if and when this poses a problem, especially for risk-averse agents? Also, is this phenomenon reversed for risk-averse agents, do they become more pessimistic? I expect some of this depends on the ratio of r_max vs beta?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes to solve risk-sensitive reinforcement learning (RL) problems with entropic risk measures based on control as a inference perspective. After deriving the evidence lower bound (ELBO) of the log-marginal likelihood of optimal trajectory, the introduces a EM-based algorithm to maximize the ELBO. For E-step, the author introduces a Bellman-like operator and arguses that it is a contraction map. Based on this theoretical result, the E-step can be implemented as a successive application of this Bellman-like operator. M-step is introduced as a standard RL problem with augmented reward function. The proposed method can be interpreted as a model-based actor critic method. Experimental studies shows the effectiveness of the proposed method.

### Strengths
- Risk-sensitive RL is an important research direction and entropic risk measure is a popular risk measure. Practical algorithm for this objective has a large group of potential audiences (significance).
- Experimental results are informative. In particular, the experimental studies in Section 6.1 and 6.2 are very good show cases of the concept of this paper (quality, significance).

### Weaknesses
Despite the strengths, the paper has clear weaknesses.

### Major concerns

- I am not convinced of the validity of the theoretical results. The proof is not concrete and self-complete, and the assumptions are not clearly stated. As a result, I am also not convinced of the validity of the proposed algorithm, despite its empirical performance. The followings are my concerns on the theoretical arguments.
  - L.732, the distribution $p$ is different for $s^{\prime}$ and $r$ as we can see from Eqs. (18) and (19). How the second equality holds?
  - How the contraction statement in L.739 - L.743 follows? I believe the inequality in L.741 - L.742 is not trivial for the current definition of $\mathcal{T}_{\pi}$ and needs a formal proof.
  - How the successive application of $\mathcal{T}_{\pi}$ results in the second equality in L.755? In general, maximum and expectation is not commute.
  - Even though the main text does not assume that the horizon of MDP, $T$, is a random variable, "the transient assumption of stopping MDPs" is used in the beginning of the page 15.

- In the abstract, the author states that "algorithms that optimize this objective have been restricted to tabular or low-dimensional continuous environments". However, the experiments in Section 6.3 is still limited to the smaller problems in Mujoco environments. To better demonstrate the scalability of the proposed method, experimental evaluation in larger domain, such as Humanoid, is necessary.


### Minor issues

- L.752, typo: T -> \mathcal{T}_{\pi}


## After reading author responses and revised manuscripts
- Thank you for revising the paper. The theoretical arguments are almost clear. I raise the score from 3 to 5 (as of 11/25).

### Questions
### Suggestion
Please address my concerns, especially on the theoretical arguments. If the theoretical results are validated, I am happy to re-evaluate the manuscript.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a risk sensitive model based actor critic algorithm based on variational inference principles; in a risk aware RL setting. The authors extend the setting to stochastic rewards, compared to past works on deterministic reward setting when studying risk sensitive RL, and proposes a EM style algorithm that alternates between learning the model and optimizing the policy. The algorithm is derived based on RL as probabilistic inference; studies the risk awareness in policy optimization setting; studies the challenges of model learning in stochastic reward setting - and experimentally demonstrates the efficacy of the proposed algorithm on a suite of benchmarks.

### Strengths
This paper has some of the following strengths : 

1. Unlike past works, the authors study both the risk-seeking and risk-averse settings and proposes a model based policy optimization algorithm based on first principles of RL as inference. 
2. The algorithm is derived based on standard formulation of entropic risk measure and the proposed objectives are derived based on the risk criterion. 
3. The paper is well written with the derivation of the key objectives for learning the model (rewards and transitions) and policy optimization well explained. 
4. Overall, for this setting, there are few aspects to consider when learning both the model and the policy. The authors extend the setting to the stochastic rewards setting and shows how a suitable model can be learnt even under stochastic rewards - a common setting to consider when studying risk awareness of learnt policies in RL. 
5. The EM Style algorithm derivation and the alternative optimization of the objectives are natural to consider when studying in this setting - and the authors do a good job in deriving a suitable algorithm that can work empirically. 
6. Experiment results demonstrates the usefulness of the rsVAC algorithm under few tasks - and takes some standard baselines on top of which the risk awareness criterion can be easily integrated to study effectiveness of the proposed algorithm.

### Weaknesses
Although the work seems good, there are few weaknesses that concern me about this work : 

1. The alternative EM style optimization of the objectives seems to bring in a lot of instability in the overall learning process. This has been a common issue in several past works on risk aware RL; for example, past works that even studied mean-variance RL, or when minimizing a variance measure in presence of inherent stochasticity in the environment. Similar to past works - I am worried about the effectiveness of the algorithm, beyond tasks that are demonstrated here, with the minimal baselines that are used for comparison. 

2. Experimental results do not seem convincing enough, and it is not clear what are the different aims of each experiment. It would be useful to better structure the experiments section to make it clear what different aspects the rsVAC algorithm seems promising in? 

3. There are several baselines for comparison in this line of work. I understand the work extends it to stochastic rewards setting -but to be more comprehensive, it would be useful to even show how effective past algorithms are in the stochastic rewards setting? 

4. I am not sure if the chosen baselines for experiments are suitable for the tasks? Can the authors compare to some more baselines?

5. The derivation of the algorithm, based on the EM style approach seems useful - but there has been past works which would do in a similar way? What’s the novelty here?

### Questions
1. Can the authors comment on the instability issues related to comment in weakness (1)? 

2. Have there been any ablation study related to instability when optimizing the different objectives, involved in model learning and then doing policy optimization? 

3. I would like to see more experiments comparing to past baselines studying risk aware RL; even if in deterministic settings. How well do those algorithms scale to stochastic rewards, and this can help demonstrate the effectiveness of rsVAC itself. 

4. Are there simple suite of tasks, not necessarily control based Mujoco ones - where rsVAC can be shown to be substantially useful? Or I do not see the novelty in the algorithm - and the way it is derived is quite standard too; so maybe I am missing something here? Section 6.2 seems useful in this context - but what happens to policy learning in this setting?

5. Objective-wise, can the authors comment on what makes the proposed objectives interesting to look into, which past works have ignored?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work presents an approach to optimize the entropic risk objective in RL by formulating it as a RL-as-probabilistic-inference objective, and optimizing a variational lower bound. This lower bound is solved using two iterative steps, where in one step the variational dynamics of the environment are learned, and in the other step, an actor-critic algorithm optimizes for a policy with a modified reward.
The risk-sensitivity of the approach arises from the sign of a parameter $\beta$, and the notion of risk here is related to the variance of the returns. Experiments in tabular and continuous environments demonstrate that the algorithm can learn risk-seeking and risk-averse behaviors based on $\beta$ and outperforms similar algorithms in the risk-averse setting.

### Strengths
The presentation of the paper was mostly clear, rigorous, and well-written.
The illustrative example of Figure 1 was clear and helpful in understanding. There were a good variety of experiments in different domains. The 2D stochastic environment experiments nicely show the effect of using $\beta$ to modulate risk behavior of the policy. 

The paper has originality and significance. From my understanding, these come from extending the RL-as-inference framework to stochastic rewards, and using this framework to also optimize for risk-averse policies.

### Weaknesses
- I think the presentation of the ELBO objective needs to be improved to clearly indicate that this is not an original objective of this paper. There are also many similarities to [Chow et al, 2021] in the overall formulation and the Bellman-like operators in the E-step optimization. The difference between these works needs to be highlighted more than what is given in L302-303. 
  - Since a major difference between the two works is given as VMBPO only optimizing for risk-seeking behavior, I am wondering: would you not be able to get a risk-averse policy by choosing the temperature $\eta < 0$ in VMBPO as well?  


- The algorithm presented, rsVAC, is quite complicated and potentially subject to a lot of instabilities in practice. My main concerns are:
  - (E-Step, section 4.1) Learning the variational dynamics and reward requires first to learn a model of the true dynamics and reward, $p_\theta$. How does the error in $p_\theta$ affect the variational distributions?
  - $V_\psi$ is used in Eq. (15) to learn the variational dynamics, and on the other hand it is optimized in the actor-critic step using the learned variational dynamics. I expect this would be quite difficult to stabilize in practice. Could you comment on that? 
  - $\beta$ parameter sensitivity: Is the convergence of the algorithm very dependent on the $\beta$ chosen? How was this sensitivity observed in the experiments?
- I don’t agree that Fig 3a shows rsVAC is more efficient at learning and obtains a higher return (L360). It looks like all the curves are within each other’s variability. I think more runs (~30) are needed before a conclusion can be made, which I feel is not unreasonable since this is a tabular environment.

Overall I think the originality and contribution of the paper are significant enough to be accepted, however, I have doubts as to the usability of the main algorithm in practice. 

Minor comments:
- I found the presentation of (12) confusing, it would be better if they are presented as equations.

### Questions
- You mention in L040 that optimizing the entropic risk requires knowledge of the transition kernel, citing Noorani et al, 2023. To my knowledge the algorithms proposed in that work (particularly R-AC -- Risk-sensitive online Actor-Critic) do not require knowledge of the transition kernel and propose an actor-critic algorithm to optimize the entropic risk much like this work. Could you expand on the comparison between the two works and what advantage this variational approach has to the R-AC of [Noorani et al., 2023]?
- Regarding objective (7), you mention that the controller may still learn risk-seeking or risk-averse behavior in L160 “if the change in return sufficiently compensates”. What does this mean? 
- In Section 4, it is mentioned that the model learning objectives are optimized using SGD. Since this is an online algorithm and these steps are done iteratively, does this mean that one step of SGD was taken for the optimization at each environment collection step or multiple steps, or optimized up to some convergence criteria? (e.g. L214, L222, L233) How was this decided in practice?


## Post-rebuttal
Increased my score as my questions have been answered

### Soundness
4

### Presentation
4

### Contribution
3
