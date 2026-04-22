# Consistent Zero-Shot Imitation with Contrastive Goal Inference

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
In the same way that generative models today conduct most of their training in a self-supervised fashion, how can agentic models conduct their training in a self-supervised fashion, interactively exploring, learning, and preparing to quickly adapt to new tasks? A prerequisite for embodied agents deployed in real world interactions ought to be training with interaction, yet today's most successful AI models (e.g., VLMs, LLMs) are trained without an explicit notion of action. The problem of reward-free exploration is well studied in the unsupervised reinforcement learning (URL) literature but fails to prepare agents for rapid adaptation to new demos. Today's language and vision models are trained on data provided by humans, which provides a strong inductive bias for the sorts of tasks that the model will have to solve. However, when prompted to imitate a new task, some methods perform distribution matching against the demonstration data without properly accounting for the difficulty of various tasks. The key contribution of our paper is a method for pre-training interactive agents in a self-supervised fashion, so that they can instantly mimic expert demonstrations. Our method treats goals (i.e., observations) as the atomic construct. During training, our method automatically proposes goals and practices reaching them, building off prior work in reinforcement learning exploration. During evaluation, our method solves an (amortized) inverse reinforcement learning problem to explain demonstrations as optimal goal-reaching behavior. Experiments on standard benchmarks (not designed for goal-reaching) show that our approach outperforms prior methods for zero-shot imitation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel algorithm for doing zero-shot goal-conditioned imitation learning by mixing ideas from contrastive RL and maximum entropy RL. The method learns a policy autonomously in an environment without supervision through automatic goal selection, goal inference, and max ent contrastive RL. At test time it receives a trajectory for which it must infer the intent and imitate. The authors' main theoretical result is that a state-of-the-art zero-shot RL (ZSRL) method, the forward backward representation, does not correctly infer the demonstrators reward function, whereas their method does. The authors report experiments on a series of common RL tasks found in the Jax GCRL benchmark and compare against some commonsense baselines.

### Strengths
The paper is well written and well motivated. It fills a gap in the goal-conditioned zero-shot imitation learning literature by doing goal inference instead of trajectory matching. The proof by counter example in Appendix A.2 is clear and demonstrates that FB is insufficient to recover the reward functions being optimized by a demonstrator in some settings. Across a variety of settings, the CIRL method out performs the FB representation; this includes goal conditioned environments where FB is clearly insufficient and more general reward environments where FB has typically performed well. 

These results show that CIRL is a promising algorithm and should be further investigated. Overall, I enjoyed reading this paper.

### Weaknesses
There are several methodological and experimental weaknesses that are important to address.
## Major concerns
1. **Is GoalKDE a reasonable goal sampling procedure?** In environments where many states are easily reachable from other states (i.e., reacher and pusher) it is clear KDE is a sufficient method for goal selection. However, in larger RL (and language) tasks, I believe that KDE will not work as well. Figure 6 does not give me a good indication of exactly what goalKDE can achieve given the varied performance when compared to the oracle. This leads me to the question: how sensitive is CIRL to the selection of the goal sampling? Are there environments (perhaps point_maze?) that might demonstrate the strengths and weaknesses of the goal selection.
2. **Mutual information between goals and trajectories**: There seems to be little motivation for the sentence, "by additionally noting that the g we infer should have high mutual information," on line 279. While I believe this is a *reasonable* statement, it does seem to be used as a precursor to simplifying the optimization objective $\mathcal{L}_\text{info}$. There are many settings where goal-conditioned policies look extremely similar for a majority of the trajectory, except for the final few steps (i.e., many pick and place tasks in robotics); I believe this would make  mean field approximation difficult. Do the authors have further justification for this statement?
3. **Possible missing baselines**: Would it be possible to demonstrate CIRL's performance against other ZSRL methods such as HILP [1] and PSM [2]? While FB has been a seminal work in this space, the aforementioned methods are strong alternatives to FB and might not suffer the same pitfalls.

## Minor concerns
These are nit-picks that do not influence my review of the paper, but are worth noting.
1. None of the code except for the README was accessible during my review at the [authors' link anonymous code link](https://anonymous.4open.science/r/cirl-3CD7/README.md) (all the training files threw a "The request file is not found" error)
2. Lines 9 and 10 in Algorithm 1 are missing ")"
3. There are two "the"s on line 286

[1] Foundation Policies with Hilbert Representations, Seohong Park, Tobias Kreiman, Sergey Levine

[2] Proto Successor Measure: Representing the Behavior Space of an RL Agent, Siddhant Agarwal and Harshit Sikchi and Peter Stone and Amy Zhang

### Questions
In addition to the questions asked in the Weaknesses section, please answer the following.
1. What was the architecture used for the variational goal inference? The authors claim it is inefficient to train models with varying trajectory length and so mean field approximation helps. Would a transformer architecture or RNN not solve this problem?
2. To give FB the benefit of the doubt, is it possible to train a goal inference model and use *only* the inferred goal to condition the FB policy? That is $z = B(s_\text{inferred goal})$

### Soundness
4

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
4

### Summary
Proposes and practices reaching goals during training time, then attempts to represent behavior to imitate as goal-reaching behavior. Compares this method against IRL and other zero shot RL methods in locomotion and goal reaching domains.

### Strengths
The method offers a modern reexploration of IRL methods.

The empirical results appear to suggest decent performance.

### Weaknesses
The method is introduced as being for use with human experts, but provides no human experiments.

The writing lacks polish and often overclaims or makes statements that are not supported by evidence.

The experiments are incomplete, lacking comparison with a wide range of URL methods.

### Questions
The sentence "The problem of pure exploration (which assumes no adata as input)", is not clear. Does the exploration assume no expert data initially? But data is generated through the exploration process and input? And through exploration the agents then might be equipped fro new tasks. Or if the problem is exploration, isn't this just a coverage issue, and there are no "new" tasks, since all tasks are just interpolation between existing tasks. 

It is not clear ther ea a "faulty tacit assumption that humans spend most of their time in the most rewarding states". Is this statement meant to be specific to all vision and language models? This seems like a statement relevant to a particular class of algorithms, or it needs to be better supported. 

Training a model purely from self-supervised practice is the same as unsupervised rl, and also has all of the issues of unsupervised RL---namely that it is exceedingly data intenstive, and the point of using a large model is to amortize that process with inductive biases. Arguably the problem setting is not zero shot imitation learning so much as it is classic unsupervised RL, where there is a substantial pretraining step followed by rapid reward function inference, except that the reward function is represented using a demonstration.
The idea that self-supervised learning should involve interaction is good in practice, but belies the reality that gathering interactions on a physical system is both expensive and often dangerous, especially when the agent has no experience with the rewal world and this is likely to damage itself or its environment.

The "goal-reaching observation" is a pure limitation, since  all goal reaching policies can be represented as reward functions, but not all reward functions are goal reaching. This is not an issue but it should be made clear this is a heuristic/limitation of the method.

The related  work should probably also include greater depth into unsupervised RL methods such as Proto-successor measure, etc., since these methods are probably the most equivalent. Methods like RLZero imitate videos from other embodiments by representing the states as occupancies, which is a generalized version of a goal.

The formalization appears to be somewhat incomplete. in particular there are missing definitions of terms: $\phi, \psi$ for the function $f^{*}_{\phi, \psi}(s,a,g)$, which makes it necessary to refer to the contrastive learning as GCRL paper (which is not cited in the formalism). Also, $\mathcal L_\text{Critic}(\phi, \psi)$  is missing from the paper, so that also need to be inferred. 

It will probably help to have a brief walk through of the mathematical transformations on lines 280-281 that make it possible to transform the minimum mutual information into a maximization of the variational distribution.

It is not clear in Lemma 1 what it means to "consistently infer rewards". Does this provide some bound on the likelihood that the correct reward will be inferred given some assumptions?

It seems like the representation of the "averaged backward representation of expert demonstration states" is not a good baseline considering that the current methods are using a goal-conditioned assumption. Shouldn't just the goal states be used for the backward representation for a more fair comparison?

The limitation mentioned for FB with exploration is well established, and it seems like this should not be left to future work at least to some extent, since it seems possible to get good-performing offline data to confirm if a well trained FB representation will achieve similar results since the offline data for these domains is readily available.

It is probably not accurate to say that CIRL works in non-goal based reward functions, since this is just a comparison against FB, not evidence really that the method is showing any significant departure from goal-based reward functions. As it is, with the way that the representation is trained there is no reason to believe that it can perform on arbitrary rewards, only that sometimes arbitrary rewards can be approximated using goals.

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
4

### Summary
The paper proposes a self-supervised learning method that explores in an environment to pretrain representations through interactions and use these representations to perform zero-shot imitation learning. The authors use the MaxEnt CRL objective to explore for goals sampled using a KDE. Along with training the Q function and policy using CRL, the authors train a variational goal inference model that predicts the trajectory conditioned goal distribution. At test time, given a demonstration, the variational model can be used to predict the goal and the GC-policy trained via RL be used to execute actions conditioned on the policy.

### Strengths
(1) While the field of zero-shot RL has been explored recently, most methods still assume access to offline data. This work falls in the right place in terms of the problem it is trying to solve. 

(2) The exploration which essentially is a combination of the goal sampler and CRL looks like a good addition that can be made to any zero-shot RL method and not specific to CIRL. 

(3) The goal inference module also on its own is an interesting way to infer intents from demonstrations and can be employed with other methods.

(4) CIRL produces policies consistent with the demonstrations, and the authors prove that FB can be inconsistent.

### Weaknesses
(1) The paper lacks a thorough literature review of self-supervised learning in RL. While not all of them possess zero-shot capabilities, there are a wide variety of self-supervised objectives in RL for pretraining. 

(2) Even for methods like [1, 2, 3] which assume offline data, this data is unlabelled non-expert data, generally collected by running an exploration algorithm in the environment. A combined algorithm can easily be created that first performs exploration and then learns a representation followed by zero-shot inference.

(3) The paper lacks this literature review through its introduction and motivation. It is true that self-supervised learning in agentic systems require interaction and exploration and also that a purely online zero-shot method is needed, but that does not mean that work has not been done in self-supervised learning of agentic systems (which is not from a supervised learning point of view).

(4) The work mentions that FB requires offline data (line 167). FB was initially proposed [1] as an online algorithm. It is true that later versions have focused more on the offline settings. Recent methods like [4] are online and possess zero-shot capabilities. 

(5) Definition 1 is very loose. During inference does the agent have access to a dataset or buffer. How efficient should the inference be? No updates to any parameters? Some updates allowed but no interactions with the real-world?

(6) The assumption that a goal can be inferred from a demonstration is very strong. There can be so many behaviors such as, “do a backflip” that will not have a well defined goal (nor an easily specifiable reward function). 

[1]: Touati, Ahmed, and Yann Ollivier. "Learning one representation to optimize all rewards." Advances in Neural Information Processing Systems 34 (2021): 13-23.

[2]: Agarwal, S., Sikchi, H., Stone, P., & Zhang, A. Proto Successor Measure: Representing the Behavior Space of an RL Agent. In Forty-second International Conference on Machine Learning.

[3]: Park, S., Kreiman, T., & Levine, S. (2024, July). Foundation Policies with Hilbert Representations. In International Conference on Machine Learning (pp. 39737-39761). PMLR.

[4]: Zheng, Chongyi, et al. "Can a MISL Fly? Analysis and Ingredients for Mutual Information Skill Learning." The Thirteenth International Conference on Learning Representations.

### Questions
(1) In the experiments, is FB trained online or offline? If offline, what dataset has been used?

(2) What is the 1-NN baseline?

(3) Goal Sampling says that the buffer is pre-filled at the start of training. If there is no access to any dataset, how is it pre-filled?

(4) KDE would be continuous distribution across the entire real-space? The lowest probability actions hence would in general be outside the action space of the agent. Is it assumed that the range of action space is known?

### Soundness
2

### Presentation
3

### Contribution
2
