# Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Offline reinforcement learning (RL) is a powerful approach for data-driven decision-making and control. Compared to model-free methods, offline model-based reinforcement learning (MBRL) explicitly learns world models from a static dataset and uses them as surrogate simulators, improving the data efficiency and enabling the learned policy to potentially generalize beyond the dataset support. However, there could be various MDPs that behave identically on the offline dataset and dealing with the uncertainty about the true MDP can be challenging. In this paper, we propose modeling offline MBRL as a Bayes Adaptive Markov Decision Process (BAMDP), which is a principled framework for addressing model uncertainty. We further propose a novel Bayes Adaptive Monte-Carlo planning algorithm capable of solving BAMDPs in continuous state and action spaces with stochastic transitions. This planning process is based on Monte Carlo Tree Search and can be integrated into offline MBRL as a policy improvement operator in policy iteration. Our "RL + Search" framework follows in the footsteps of superhuman AIs like AlphaZero, improving on current offline MBRL methods by incorporating more computation input. The proposed algorithm significantly outperforms state-of-the-art offline RL methods on twelve D4RL MuJoCo tasks and three challenging, stochastic tokamak control tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper frames offline MBRL as a BAMDP to address model uncertainty inherent in learning from offline datasets. The authors propose a novel Continuous BAMCP planning algorithm based on Monte Carlo Tree Search with double progressive widening for continuous domains. Integrating this with a policy iteration scheme, they present a framework termed BA-MCTS that distills the search/planning output into policy/value networks for efficient real-time decision making. Empirical evaluations are provided on D4RL benchmark, showing strong performance relative to several baselines.

### Strengths
1.Integrating BAMCP with offline MBRL and extending it to continuous spaces via the introduction of DPW is well-motivated and conceptually sound.
2.The paper includes well-executed ablation studies that systematically evaluate the individual contributions of the algorithm’s components, highlighting the advantages of the BAMDP formulation, MCTS planning, and the search-guided policy learning approach.
3.Theorem 4.1 establishes a consistency guarantee for the planning algorithm, proving its convergence to the optimal value in the pseudo-pessimistic BAMDP framework.

### Weaknesses
1.Unclear and inconsistent notation for augmented states: The notation used to represent augmented states—such as $(s, h)$ versus $(s, b)$—lacks clarity, and the distinction between history, belief, and memory storage in tree nodes remains vague. This ambiguity, evident across equations, algorithm steps (e.g., Algorithm 1, Page 5), and explanatory text, may cause confusion, particularly for readers less familiar with BAMCP/POMCP.
2.Insufficient contextualization within Bayesian and offline MBRL literature: The paper would benefit from a more thorough discussion connecting the proposed method to existing frameworks that model offline RL as a Bayes Adaptive MDP or epistemic POMDP, such as those presented in Ghosh et al. (2022) and Dorfman et al. (2021).
Ghosh et al. (2022): "Offline RL policies should be trained to be adaptive."
Dorfman et al. (2021): "Offline meta reinforcement learning – identifiability challenges and effective data collection strategies."

### Questions
1.In Table 1, should the mean prediction errors be calculated as weighted averages based on the dimensions of states and rewards? Why does the Walker2d environment exhibit such large prediction errors in Table 1, while showing very small performance variance in Table 2?
2.Why are the results reported in Table 2 inconsistent with those in Table 12?

### Soundness
2

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
3

### Summary
This paper views offline RL as a Bayes Adaptive Markov Decision Process (BAMDP) for effectively addressing uncertainty associated with a finite dataset, and an associated planning algorithm based on MCTS that is then folded back into learning a policy using a policy search procedure. Empirical simulations show that the proposed algorithm outperforms baselines in offline RL benchmark suites.

### Strengths
- Using search within offline RL based policy search is pretty interesting as a general approach, and this paper makes progress towards marrying these ideas in a clean manner. That said, my knowledge of this area isn't up to date particularly with the more recent efforts and so I do not know if there are similar efforts in offline RL that I am unaware of.

### Weaknesses
- The writing at times appears to come in thick and fast, and in a sense i understand why this happens given the different sub-routines but it does help to work a bit more to help clarity in exposition.

### Questions
- One thing that wasn't clear was is the output of the learner just the policy? 
- What do the tradeoffs look like if we tried to use the search procedure even at inference time, somewhat mirroring MPC style approaches? 
- For me, it wasnt totally clear what was the net positive from trying to use search in this setup, as it does appear as if one can effectively solve the problem using a mix of pessimism and dyna type sampling, and perhaps this isnt totally clear just from the current exposition.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper reframes Offline Model-Based Reinforcement Learning (MBRL) as a Bayes-Adaptive Markov Decision Process (BAMDP) to formally handle uncertainty over learned dynamics models.  
It introduces Continuous BAMCP, a Monte-Carlo Tree Search planner extended to continuous, stochastic domains using Double Progressive Widening (DPW), and integrates it into a policy-iteration framework named BA-MCTS.  Maintain a belief distribution over ensemble models and update it via Bayesian posterior rules.  Provide a *consistency theorem* guaranteeing exponential convergence of the planner’s value estimate. Empirically achieve convincing results on D4RL MuJoCo tasks and 3 Tokamak control tasks.

### Strengths
1. Novel idea of combining BAMDP formulation, MCTS planning, and offline MBRL into a unified algorithmic pipeline.
2. Provides a **consistency proof** for Bayes-adaptive planning in continuous spaces.
3. Bayesian belief updates give interpretable, adaptive weighting over model ensembles
4. Reward-variance penalty connects naturally to pessimistic (safe) RL literature
5. Strong empirical performance   
6. Generalization: Demonstrates transfer to real-world-style stochastic control (Tokamak tasks).

### Weaknesses
1. **Computation cost:** Continuous BAMCP remains expensive, many simulations per state, not scalable to large or long-horizon domains.
2. **Limited data efficiency analysis:** No study of how ensemble size or dataset quality affect performance.
3. **Offline-only limitation:** Once trained, policy does not update beliefs from online experience.

### Questions
- What is the computational cost (simulations per decision, wall-time) relative to other baselines?
- Could Continuous BAMCP be applied in learned latent spaces to improve scalability?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel framework for offline model-based reinforcement learning, which (1) trains an ensemble of models based on data from a behavior policy and builds a Bayes Adaptive MDP (BAMDP) of this ensemble (i.e., an MDP with uncertainty, where the belief about the transition and reward distributions is updated based on observed transitions); (2) introduces a shaped reward for the Bayes Adaptive MDP that penalizes actions with high uncertainty, called Pessimistic Bayes-Adaptive MDP; (3) extends an existing online-planning algorithm for BAMDP, called BAMCP, to handle continuous states and actions based on Double Progressive Widening; (4) proves that this extension, called Continuous BAMCP, approximates the optimal policy; and finally, (5) introduces an algorithm for efficiently learning a policy from the planning outcomes of Continuous BAMCP.

### Strengths
* The paper introduces a number of novel ideas for offline model-based RL: formulating an ensemble of learned models as a Bayes Adaptive MDP, extending BAMDP to handle continuous state and action spaces, and distilling the planning outcomes into a policy. According to the numerical results, most of these ideas are helpful in practice.
* The proposed extension of BAMDP, called Continuous BAMDP, is proven to converge.
* The proposed framework is compared to offline RL baselines on benchmark environments and data, and the numerical results are promising. There is also an ablation study.
* The paper is generally well organized and easy to follow.
* The source code is available, and it supported by some documentation.

### Weaknesses
* My main concern is that treating the ensemble of learned models as a Bayes Adaptive MDP for offline RL does not seem to make sense conceptually. Bayes Adaptive MDP is perfectly reasonable when the beliefs are updated based on transitions that are sampled from the real environment; each transition can provide new information, which we can use to update our beliefs (i.e., probabilities or probability densities associated to particular MDPs). However, in the proposed framework, the transitions are sampled from the learned models. What new information do these transitions provide? If we set our prior belief to be a uniform distribution, and then uniformly sample transitions according to these models, why is our expected updated belief different from uniform? For a particular sampled transition, the updated belief will of course be non-uniform; but why would it be non-uniform in expectation? And if it is, then are the belief updates just driven by randomness (e.g., probability of the model that was sampled uniformly at random would likely be increased)? How is this belief update useful?
* That being said, Table 1 in the main text does show that the Bayesian update significantly improves predictions for imaginary rollouts (compared to ground-truth rollouts in the real environments for the same actions sequences). So, for some reason, the Bayesian update is useful; but it is not obvious why. What is the explanation? Does the improvement hinge on the particular ensemble of neural-network models? Does it hinge on the characteristics of these neural networks? Does it give an advantage to neural networks whose output distributions are more concentrated? If yes, could the Bayesian approach be replaced with simply favoring such models? Or does the Bayesian update help by simply concentrating the beliefs to a very small support after a few steps (as shown by the numerical results), which is somehow better for the simulation? If yes, could the Bayesian approach be replaced by simply using one particular randomly chosen model for each episode?
* In the end, the numerical results do support the proposed approach. But it would be good to clarify why it works. (Appendix F.1 is interesting, but does not address the core question.)
* The paper includes a number of ideas, addressing various challenges. While the paper is overall well organized and easy to read, it might benefit from a clearer focus. For instance, the proposed framework would be novel and complete even without Continuous BAMCP; it would simply be restricted to discrete environments. The policy learning also seems somewhat optional since the framework could be presented as an online planning approach (based on model learned from offline experiences); according to the numerical results (Table 11 in the appendix), BA-MCTS and BA-MCTS-SL are pretty close, so the policy learning does not seem that important compared to the other parts.

### Questions
* Can you please explain why the Bayesian update works in the offline setting?

### Soundness
3

### Presentation
4

### Contribution
3
