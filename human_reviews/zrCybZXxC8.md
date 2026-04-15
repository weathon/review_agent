# E-MCTS: Deep Exploration by Planning with Epistemic Uncertainty

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 5

## Abstract
Deep model-based reinforcement learning (MBRL) is responsible for many of the greatest achievements of reinforcement learning. At the core of two of the approaches responsible for those successes, Alpha/MuZero, is a modified version of the Monte-Carlo Tree Search (MCTS) planning algorithm, replacing components of MCTS with learned models (of value and/or environment dynamics). Dedicated deep exploration, however, is a remaining challenge of Alpha/MuZero and by extension MCTS-based methods with learned models. To overcome this challenge, we develop Epistemic-MCTS. E-MCTS extends MCTS with estimation and propagation of epistemic uncertainty, and leverages the propagated uncertainty for a novel deep exploration algorithm by explicitly planning to explore. We incorporate E-MCTS into variations of MCTS-based MBRL approaches with learned (MuZero) and provided (AlphaZero) dynamics models. We compare E-MCTS to non-planning based deep-exploration baselines and demonstrate that E-MCTS significantly outperforms them in the investigated deep exploration benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Authors tackle the issue of deep exploration in model-based reinforcement learning agents, such as MuZero and AlphaZero. Deep exploration involves the capacity to evaluate not just the immediate rewards of an action but also its long term consequences. o enable deep exploration within MuZero, the authors introduce the "Epistemic MCTS" algorithm, which leverages epistemic uncertainty to facilitate more intelligent exploration. They test the effectiveness of their approach on deep-sea problem, a simple problem designed to test for deep exploration.

### Strengths
While MuZero and AlphaZero excel in various tasks, they often struggle to account for the long-term consequences of their actions, particularly in scenarios with sparse rewards. By enabling deep exploration in MuZero, authors can improve the performance of MuZero on many tasks. 

Authors  introduce the Epistemic-MCTS algorithm, which incorporates epistemic uncertainties into the decision-making process when selecting actions. This algorithm holds considerable promise as an independent research contribution.

Authors demonstrate the efficacy of their method on a simple and clear problem, which is greatly appreciated.

### Weaknesses
1. Although authors try to make their work more mathematically rigorous, I personally find it too hard to follow. Having a separate section on notation could be very helpful. 

2. The concept of reconstruction loss, which is new compared to the MuZero paper, is a noteworthy addition; however, it could benefit from more comprehensive explanation. This issue of less detailed explanations appears in multiple sections of the paper. For instance, in the concluding part of Section 2, the term "local uncertainties" is introduced for the first time, yet it lacks clarity regarding the specific context of "local" and the variables to which the authors are referring to.

3. The proposed algorithm, E-MCTS, necessitates the computation of Jacobians, a process that can be computationally intensive. This computational demand may limit the practical applicability of the algorithm to more complex problems.

4. I find the experimental setup to be somewhat constrained. While the deep-sea problem serves as a suitable testbed for assessing deep exploration, it would have been valuable to investigate whether their algorithm negatively impacts the existing capabilities of the MuZero algorithm.

### Questions
1. In Section 2.3 authors mention that they use RND and visit counts to approximate epistemic uncertainty. However the details about how and where these values are computed is not mentioned. It is not clear if the RND loss presented in Section 2.3 is used to estimate uncertainties in rewards or values or policies. Can authors clarify this? Another work on epistemic uncertainty estimation which could be relevant to authors is epistemic neural networks (https://arxiv.org/pdf/2107.08924.pdf)

2. Can authors clarify what "local uncertainties" mean at the end of Section 2.3, above the equation.

3.  U^\pi was introduced at the end of Section 2.3, but I wasn't able to find where it was being used. Can you clarify this?

4. In the second paragraph of Section 3, when talking about how to reduce epistemic uncertainty, authors say "2. planning in directions where the agent is more epistemically-certain". It is not clear which variable's epistemic uncertainty authors are referring to. Intuitively, taking a path with more epistemic uncertainty should reduce epistemic uncertainty. Can authors kindly clarify the statement?

5. The authors use Jacobians at various points within their algorithm, for instance, in Equation 6. Based on my understanding, Jacobian computations and the matrix multiplication, such as in the second term of Equation 6, can be computationally very demanding. It would be insightful to understand the strategies the authors employ to manage and mitigate these computational challenges.

6. In Algorithm 1, authors update terms like Sigma_k, in line 11 of the algorithm. Can authors clarify if the updates being performed locally or if the changes in lines 11-13 and lines 17-18 global?

7. Have the authors compared their algorithm on tasks other than deep sea and have they observed any regression when compared to baseline algorithms which use standard MCTS?

8. It looks like the default parameters of BootDQN were used in the experiments in Section 5. Since, the default parameters in bsuite were probably chosen to perform well on all bsuite tasks, it might be useful to tune hyperparamters such as ensemble size, num of sgd steps after each env steps, learning rate and prior scale.


Some suggestions:
1. it would be helpful to have a notation section, as many variables are based on the same letters (ex: v is used in various forms)
2. It might be helpful to include some diagrams on how different components fit together. This can be done for both training and inference.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents Epistemic-MCTS (E-MCTS), a method for incorporating epistemic uncertainty into AlphaZero-like model-based RL with MCTS. The goal is to encourage the selection of actions for decisions with more epistemic uncertainty caused by epistemic errors from the learned model. This approach facilitates exploration in states that require more interactions, thereby inducing deep exploration. Comparing to the baselines, E-MCTS outperforms in the investigated exploration benchmark (e.g., Deep Sea).

### Strengths
1. The idea of considering epistemic uncertainty in MCTS selection is interesting and reasonable for reinforcement learning (RL).
2. E-MCTS claims to provide a method to further improve the performance of existing model-based RL algorithms with MCTS, such as AlphaZero and MuZero.
3. This paper provides insights into estimating epistemic uncertainty using a recursive approximation of variance.

### Weaknesses
# Major
1. Although the paper mentions some literature reviews about uncertainty, it lacks a more comprehensive survey, particularly in the early deep reinforcement learning (DRL) research. Two foundational works in early DRL, VIME (Variational Information Maximizing Exploration) and IDS (Information-Directed Exploration for Deep Reinforcement Learning), should be included to strengthen the survey.
2. In Section 2.2, there is a concern regarding soundness. The original AlphaZero/MuZero models do not include a reconstruction loss. It would be more appropriate to refer to other methods, such as Dreamer or EfficientZero, that address this issue.
3. Regarding the experiments, it is noticeable that Deep Sea is a deterministic environment, whereas there is a stochastic variant available in bsuite. One may wonder why these experiments primarily focus on the deterministic version. This choice is particularly interesting given the presence of various sources of uncertainty in stochastic environments. It raises questions about the suitability of E-MCTS in stochastic environments and whether it can outperform AlphaZero/MuZero in complex scenarios for which the latter were specifically designed. Additionally, it's worth noting that we lack a straightforward MCTS baseline that does not suffer from epistemic uncertainty issues. It is possible that a simple MCTS approach may outperform AlphaZero/MuZero in this specific context, which could weaken the empirical evidence.
# Minor
1. Equation 3: It is unclear why there is a $max_\pi$ before $V^{\pi}(s_t)$ since there is no policy selection process among a set of policies. This inconsistency should be addressed.
2. Equation 11: The later part of the equation refers to $a_k$ without prior definition. It seems that all $a_k$ should be $a$. Additionally, the "$argmax$" function should be enclosed in parentheses to avoid confusion.
3. Regarding the references, there is an arXiv source with official publication.
    * Simple and scalable predictive uncertainty estimation using deep ensembles: NIPS 2017

### Questions
1. Why is Deep Sea (deterministic) considered a suitable environment for justifying E-MCTS? Does Deep Sea have any variation in state transitions or rewards that can induce epistemic uncertainty? In a deterministic environment, once we observe a sample, there is only one possible outcome, and further interactions do not reduce uncertainty by obtaining more samples of the same state-action pair.
2. Will the new exploration bonus $\beta\sqrt{\sigma^2_q(\hat{s}_k,a)}$ eventually converge to zero?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents Epistemic-MCTS (E-MCTS), an advancement of the Monte-Carlo Tree Search (MCTS) algorithm in deep model-based reinforcement learning, targeting improved exploration. By incorporating and disseminating epistemic uncertainty within MCTS, enhanced exploration strategies emerge. The approach employs an RND network as proxy-measure of novelty and calculate variance estimates of unobserved states, which is subsequently propagated through the search tree to guide exploration during planning. It is tested against three baselines on the Deep Sea benchmark and outperforms the baselines, though the baselines gradually approach its performance.

### Strengths
S1: Proposes a new exploration strategy for planning using MCTS based on a proxy measure of novelty, i.e., RND, and provides a practical algorithm that performs well in the Deep Sea domain with minimal computational overhead (though it varies based on the network architecture size).

S2: The method offers a mathematical approach to propagate uncertainty in predictions throughout the planning process.

### Weaknesses
The key idea appears to be allowing the search to recognize the uncertainty in the value predicted by the learned model for unobserved states and directing the search towards actions with greater uncertainty (higher variance). Additionally, a proxy-measure of novelty is employed to estimate this uncertainty.

W1: Some key concepts in the realm of exploration in MCTS haven't been touched upon. While the visitation counts themselves represent the uncertainty in the Q-value estimate at a node, other researchers have utilized the variance of predicted Q-value [1,2] and maintained it at each tree node with a Gaussian distribution to guide exploration during action selection. [3] adopts a more systematic approach to measure uncertainty in the Q-value of the unobserved state using a Gaussian process, promoting exploration based on the upper confidence derived from the variance of the GP. The advantage of using a proxy-measure of novelty over these methods isn't evident.

W2: The experiment section is somewhat limited in the diversity of the problems, making it challenging to deem the approach as robust and significant. While Deep Sea may be an illustrative example to showcase the strengths of E-MCTS, a broader experimental setting is essential to validate its edge over established methods.

W3: The writing could benefit from some refinement. For instance, the context in which "epistemic uncertainty" was introduced remained unclear until section 3. Moreover, by referencing the AlphaZero and MuZero models, it seems the authors might be differentiating between whether the transition model is learned or provided as a simulator. However, the current phrasing is somewhat confusing.

[1] Tesauro, Gerald, V. T. Rajan, and Richard Segal. 2012. “Bayesian Inference in Monte-Carlo Tree Search.” arXiv [Cs.LG]. arXiv. http://arxiv.org/abs/1203.3519.

[2] Bai, Aijun, Feng Wu, and Xiaoping Chen. 2013. “Bayesian Mixture Modelling and Inference Based Thompson Sampling in Monte-Carlo Tree Search.” Advances in Neural Information Processing Systems 26: 1646–54.

[3] Mern, John, Anil Yildiz, Zachary Sunberg, Tapan Mukerji, and Mykel J. Kochenderfer. 2020. “Bayesian Optimized Monte Carlo Planning.” arXiv [Cs.AI]. arXiv. http://arxiv.org/abs/2010.03597.

### Questions
Q1: How does the proposed method compare to the methods mentioned in W1? In what aspects is it better?

Q2: Can you present results from broader and more realistic experimental settings, such as Procgen?

Q3: In Figure 2 (right), the average regret continues to decrease even with high values of Beta. This trend seems counter-intuitive, implying that optimal regret is achieved mainly through high exploration and minimal exploitation. Could you elaborate on this observation?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Monte Carlo Tree Search (MCTS) as used in AlphaZero and MuZero implicitly reduces aleatoric uncertainty through its rollouts, but does not traditionally account for epistemic uncertainty, which can hinder exploration/

This paper introduces Epistemic MCTS (E-MCTS), that extends standard MCTS to utilize epistemic uncertainty (uncertainty in estimates that are reducible with more observations) to guide rollouts and decisions, which benefits exploration. E-MCTS allows the agent to pe

E-MCTS is tested on a variety of configurations of the DeepSea environment in the Behavior Suite, which is a hard exploration task. 
They find that their variant of E-MCTS is able to achieve a high return faster than baselines, and is able to discover states more quickly than the baselines. Additionally, they find that the benefits (in terms of regret) of E-MCTS scale with the environment size.

### Strengths
The motivation for E-MCTS is well-articulated, and the papers makes it clear what gap E-MCTS intends to fill.

The method itself appears to be sound, makes algorithmic sense, and seems to also be a good solution to the identified problem.

The presentation of the paper is good: it is well-written.

The paper convincingly shows the effectiveness of their method on a deep exploration task.

### Weaknesses
My primary issues with the paper have to do with experimentation. The paper does not test on a diverse set of environments and rather tests on different configurations (albeit with different difficulties) of the same DeepSea environment. While the DeepSea environment is certainly nontrivial and challenging, it has a very specific reward structure for which we would expect E-MCTS to perform well. While it serves as an excellent demonstration of the potential benefits of the proposed method, it does not demonstrate more generally the ability and tradeoffs of the proposed method. For example, does the introduction of the utilization of epistemic uncertainty estimates adversely impact results in environments where deep exploration is not required?  Moreover, to my knowledge, the proposed method is not compared against other domains for which MCTS is typically used. It would be nice to test E-MCTS on environments where MuZero is applied (albeit in a tractable way).


The paper has a lot of merits, and I believe with more comprehensicve experimentation it may warrant acceptance. Even results on a diverse set of standard environments, especially ones where MCTS is typically applied, would greatly improve the paper. As it stands, my interpretation of the experiments is that they "demonstrate" the potential of the method, but do not show the "effectiveness", which can be shown with other domains. Even showing that E-MCTS works well or does not harm MCTS in standard environments will show that its potential is not limited to environments necessitating deep exploration.


Suggestions:
- Given that DeepSea is the only environment tested, would recommend writing some description of the DeepSea environment.


Nits/typos:
- Section 2.1 "the the" -> "the"
- Figure 2 Caption: "perparameter" -> "hyperparameter"
- Table 3 in appendix has 'self-play networking updating inerval'. The "inerval" should be "interval"

### Questions
1. In figure 2, why 3 seeds for the 30x30 domain on the right, but 5 seeds on the left (where presumably the 30x30 domain is ran?)
2. How might exploration be balanced, tuned, or annealed over time in environments where deep exploration is not required?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
