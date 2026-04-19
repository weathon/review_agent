# Using Forwards-Backwards Models to Approximate MDP Homomorphisms

- Decision: Reject
- Scores: 3, 3, 3, 5

## Abstract
Animals are able to rapidly infer, from limited experience, when sets of state-action pairs have equivalent reward and transition dynamics. On the other hand, modern reinforcement learning systems must painstakingly learn through trial and error that sets of state-action pairs are value equivalent—requiring an often prohibitively large amount of samples from their environment. MDP homomorphisms have been proposed that reduce the observed MDP of an environment to an abstract MDP, which can enable more sample efficient policy learning. Consequently, impressive improvements in sample efficiency have been achieved when a suitable MDP homomorphism can be constructed a priori—usually by exploiting a practitioner's knowledge of environment symmetries. We propose a novel approach to constructing a homomorphism in discrete action spaces, which uses a partial model of environment dynamics to infer which state-action pairs lead to the same state—reducing the size of the state-action space by a factor equal to the cardinality of the action space. On MDP homomorphism benchmarks, we demonstrate improved sample efficiency over previous attempts to learn MDP homomorphisms, while achieving comparable sample efficiency to approaches that rely on prior knowledge of environment symmetries.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a forwards-backwards model to learn MDP homomorphism in discrete action and deterministic settings. Experiments are conducted on tabular gridworlds, cartpole, Predator Prey, Asterix (in MinAtar), to show its effectiveness over Homomorphic DQN and DQN.

### Strengths
The method is original and looks simple. The experiments on toy tasks are effective.

### Weaknesses
The assumptions are a bit limiting – deterministic dynamics and state-dependent rewards.

One major weakness is that I don’t know the connection between the forward and backward model (Eq 4 and 5), and MDP homomorphism (Eq 1 and 2). Why not directly learn Eq 1 and 2 (like my question below)? In addition, I don’t understand Eq 6. What is sigma(s,a), not sigma(s)? No definition. Also the hypothetical action part. 

The writing quality is fair. 
1. Notation writing can be improved. Be complete on $\mathcal R(s,a)$ and $\mathcal P(s’\mid s,a)$ in the background. $G$ should be in expectation. 
2. Please use \citet when you cite work used as a noun. 
3. Some sentences are hard to grasp. E.g. “which” and “homomorphism” are used twice at the beginning of page 3. Do not start a sentence with a math symbol like “\forall”. 
4. Define the symbols like \sigma and \alpha_s explicitly with inputs and outputs (you can check how van der Pol et al., 2020b define them). 
5. Typos like: Ln 7 in algorithm 1 the hat s, “Figure 3(a)” should be Figure 2(a). 

The experiments are only conducted on 4-5 tasks, which are not enough for evaluating an algorithm (as the experiments are the bulk of this work). Asterix results are not in favor of the algorithm, and more MinAtar environments are expected to run for a comprehensive comparison.

### Questions
1. Any references on the first sentence of the abstract: “Animals are able to rapidly infer, from limited experience, when sets of state-action pairs have equivalent reward and transition dynamics.”? Is it necessary to write here? 
2. For me, a straightforward approach to learn homomorphism in discrete action space, is to parameterize value function as $Q(\sigma(s), \phi_s(a))$, then train it end-to-end with some Q-learning loss, and also train $\sigma$, $\phi_s$ with MSE losses to satisfy (1)(2). It is similar to the approach in the continuous action version of homomorphism (Rezaei-Shoshtari et al., 2022). The action can be one-hot encoded. I am a bit surprised that this simple approach is not discussed in the paper, and what do you think of this approach?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an interesting method for learning MDP homomorphisms to reduce sample complexity. The proposed method uses a forward model and a backward model to abstract state action pairs, with hypothetical actions. Some experiments on tabular environments and coutinuous environments are done to verify the effectiveness of the learned homomorphisms.

### Strengths
The idea of learning homomorphisms with forward and backward models is interesting, although the concepts are not new.
The method has the potential to scale up in large visual RL problems, where state abstraction is important.

### Weaknesses
1. There are several important points not clarified or explained in the paper. For example, how can we learn good forward and backward models with function approximators? In theory, accurate forward and backward prediction requires complete coverage of the state action space, which can be even more expensive than learning a policy (without abstraction). But the algorithm directly assumes the availability of such models, which is unrealistic.
2. The success of the abstract largely depends on the correctness of the forward and backward models. If any of them give incorrect or biased prediction, the learned homomorphisms and values will fail. Combined with point 1 above, I am not convinced how the proposed algorithm can be widely applicable.
3. Although the paper tries to demonstrate empirical results on different benchmarks, the experiment still looks a bit limited. Only Q learning or DQN are considered as the base algorithm. Can it be combined with other RL algorithms such as PPO or SAC? 
4. The return on Asterix is much lower than reported by prior papers. The original nature DQN paper has return 6012 on Asterix, but why is the return reported by this paper below 20? Is there any rescaling of the reward?
5. From experimental results, the improvement made by the algorithm is mainly on the converging speed or sample complexity. However, I am not sure whether it is a fair comparison with baselines, since the proposed method trains the policy over a pretrained forward/backward model, which already consumes a lot of samples. 
6. The notations are not clear and sometimes confusing. For example, the hypothetical actions are sometimes referred to as $a_{hyp}$ and sometimes as $\hat{a}$. In Eq (7), $a^\prime$ is never defined. $\sigma$ is sometimes a function of states, sometimes a function of state-action pairs. In Eq (8), should the first $s^\prime$ be $\bar{s^\prime}$? Although the meanings can be roughly inferred from the context, it is still not reading-friendly.

### Questions
- Is there any experimental result with learned forward/backward model on large-scale environments?
- Can the forward/backward model be learned together with the policy?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper is concerned with sample efficiency in reinforcement learning (RL). The paper argues that RL systems can achieve greater sample efficiency by using a type of homomorphism they call Equivalent Effect Abstractions (EEAs). 

Presumably, EEAs permit one to map between representations of environment state while preserving the original reward structure. Provided that such mappings exist, can be efficiently implemented, and one can map multiple environment states to a single encoding, then a learning system can plausibly use the encodings to generalize policy evaluations / improvements over states.

The paper makes several claims.
1. It is possible to efficiently learn EEAs.
2. Learning EEAs requires no prior knowledge of reward structure.
3. EEAs reduce the size of the state-action space "by a factor equal to the cardinality of the action space."
4. EEAs can be used to improve the sample efficiency of policy learning.

I am recommending this paper is rejected, as it lacks support for all of its claims. Specifically, the paper contains little to no support for 1, 2, and 3. And the empirical support for claim 4 is not convincing. Moreover, I believe these issues are too substantial to be addressed with a simple revision.

### Strengths
* Sample efficiency is indeed an area that can benefit from more research.
* Exploiting symmetries and equivalences in the reward structure seems like a promising approach to designing new sample-efficient algorithms. 
* Methods applicable to deep reinforcement learning stand to have high significance addressing large-scale problems.

### Weaknesses
(Technically unsound) The paper does not contain enough information for a reader to fully understand what is being proposed.  
* Formalism lacks the precision needed to understand the main technical concepts. For instance, a "forwards model" and "backwards model" are never defined. These seem like parameterized functions. However, their input-output signatures are not described, and their parameters seem to change from Equation 6 to Algorithm 1.
* The paper fails to describe how EEAs, and its associated learning process, fit within the standard RL framework of action, observation, and reward. This is essential for others in the community to use the idea---to know how a step of policy evaluation or improvement is performed.
* The scope with which EEAs are presented is too broad, and this gives EEAs a somewhat nebulous identity. According to the paper, they apply to both tabular and deep settings, and settings where models are both available and learned. However, the paper provides no specific description of how the idea was instantiated in each respective setting. Ironically, the little information that is used to define EEAs limits their scope to a restrictive degree.
* The paper fails to establish EEAs as a distinct idea, separate from prior work in homomorphic MDPs (van der Pol et al., 2020) and using bisimulation methods.

(Technically unsound) The empirical results do not provide sufficient support for claim 4.
* Current methodology fails to account for the pre-training experience used to learn the EEA.
* Experiments are missing important baselines. For instance, a standard model-based learner that doesn't also learn a homomorphism.
* Experiments use an inconsistent methodology between baselines---for example, different numbers of seeds.
* Experiments show learning curves, but should go further---to translate them into metrics of sample efficiency, such as steps of experience (accounting for pre-training).

(Limited significance) The main contribution is potentially limited, as it only applies to fully-observable, deterministic MDPs where rewards are strict functions of state (i.e. no action dependence.)

(Limited significance) Related work does not make connections to paper's main ideas.

(Poor presentation) Mathematical notation is sometimes overloaded and often undefined.

### Questions
* Many references are missing parentheses. This makes some sentences difficult to read.
* The current manuscript points readers to external references to learn about homomorphisms. You could broaden your audience and add clarity to the paper by introducing the concept in the introduction.
* A better reference to use for introducing the MDP formalism is Sutton and Barto (The RL book), or Putterman's MDP book.
* Seem to be based on the assumption that different states that transition to the same state have the same values.
* This is not generally true:
  > Equivalent effect abstraction is based on a simple observation that is guaranteed to be true in the majority of RL problems if R and P are deterministic.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
MDP homomorphisms can be used to increase sample efficiency in reinforcement learning. Prior work has often used domain knowledge of symmetries to build these homomorphisms. The authors present a method for constructing them from experience, called equivalent effect abstraction, which uses the idea that actions that lead to the same next state often have the same value. The authors validate equivalent effect abstraction by testing its effect on the efficiency of value-based learning in a variety of RL environments: Tabular RL Maze, Cartpole, Stochastic Predator Prey, and MinAtar Asterix.

### Strengths
* The paper is written clearly, and the problem is well-motivated. Leveraging a backwards model to find actions with similar effects is clever and insightful.
* Equivalent effect abstraction is evaluated in several different types of environments, and the experiments conducted seem sound.
* In games well-suited to equivalent effect abstractions and where hypothetical actions are easy to define, empirical results suggest a positive effect on sample efficiency.
* Despite my comment in the *Weaknesses* section, the authors do a good job discussing the potential limitations of the approach related to stochastic state transitions and some of the complications that arise from edge cases when selecting a hypothetical action.

### Weaknesses
My main concern is the applicability of this approach beyond the domains that have been tested to showcase it. The approach depends on selecting a hypothetical action and the authors mention that one must be selected carefully in certain domains, but do not elaborate sufficiently. The domains selected for experiments seem well-suited to equivalent effect abstraction because they are generally spatial "navigation" tasks in 2D - I'm not sure how to describe the domains precisely, but where hypothetical actions are easy to select and equivalent effects are prominent. It would be interesting to see this idea applied in the other MinAtar games. The preliminary results in the supplementary material seem to suggest applying this approach may not be as straightforward in Breakout, for example.

### Questions
Why was asterix chosen as the only benchmark tested from MinAtar?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor
