# Architectural Inductive Biases Can Be Enough for State Abstraction in Deep Reinforcement Learning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
The ability to ignore irrelevant sensory information is central to intelligent behavior. In reinforcement learning (RL), existing methods typically rely on auxiliary objectives to achieve similar forms of abstraction. Such objectives tend to add significant complexity to the base RL algorithm. In this work, we take a step back and ask: can selective abstraction emerge naturally from reward optimization alone, without any additional objectives? Following prior work, we show that standard deep RL learns slowly or not at all in the presence of distracting, task-irrelevant state variables, failing to learn meaningful state abstractions. We then introduce a surprisingly simple neural network architecture change: a learnable, observation-independent attention mask applied to the inputs of the policy and value networks and trained end-to-end using only the RL objective. Despite its simplicity, this architectural modification consistently improves sample efficiency and learns to mask out distracting input variables across 12 continuous control tasks. We analyze the dynamics of gradient descent using this method on a linear regression task and demonstrate improved feature credit assignment. Finally, we conduct experiments on toy MDPs and show that the attention mask leads to accurate Q-value estimation and induces soft abstractions over a factored state space. Our findings challenge the need for complex auxiliary objectives to learn state abstractions in deep RL and suggest a simple baseline for future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an architectural modification for neural-network based policies that improves reinforcement learning performance in the presence of distractors. The problem setting is a factored state MDP, where parts of the state space do not contribute to the reward and transition dynamics. The authors propose learning an attention mask purely from the reward through either the actor or critic loss. Experiments on the modified DeepMind Control Suite show that the proposed masking often improves performance over vanilla algorithms (SAC, TD3, PPO). While the paper is well structured and provides ample empirical evidence, some concerns remain regarding the motivation and evaluation.

### Strengths
- The proposed mask is simple, easy to integrate with existing algorithms, and does not involve computational overhead or additional training steps. 
- The "Controlled Analysis" of the masking approach in supervised learning and toy MDPs provides some intuition on how distractors impede learning and why the masking may mitigate this.
- The evaluation is conducted using many different tasks from the DeepMind Control Suite as well as several state-of-the-art actor critic RL algorithms. Statistical relevance of the results is shown by using several random seeds for each task and algorithm. 
- Both structure and writing in the paper are very good.

### Weaknesses
- The proposed architecture seems to be mainly applicable to settings with factored state spaces, meaning the observation is a vector. Most existing work in the area of state abstraction focuses on visual domains, which seem to be more relevant. Given a state space with a time-invariant factorization, it seems more promising to identify the irrelevant states - e.g., using SHAP values - and then train agents using a reduced observation space. Not only is the proposed approach not tested on visual domains (e.g., Distracting Control Suite), it also does not seem applicable to such tasks. The motivation of the paper should be improved w.r.t. to these points. 
- The authors compare their approach to only one alternative - MaDi - which was devised for visual domains. In the original MaDi paper, the masking network consists of a small convolutional neural network to process RGB images. If a similar architecture was used in this work, this would be ill-suited for simple vector observations. Therefore, we suggest comparing to other approaches designed for factorized state spaces like the CBM proposed in (Wang et al. 2024). 
- The presentation could be improved by increasing the size of Fig. 1, 3, and 4.

### Questions
- Why is the factored MDP formulated for discrete states and actions? The tasks in the experiments seem to feature continuous state and action spaces. 
- Why is PPO not paired with MaDi in the evaluation?

### Soundness
4

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
This paper explores state abstraction in deep reinforcement learning. The focus of the work is to better understand what minimal ingredients might be needed to facilitate learning a state abstraction; many prior approaches rely on additional losses or auxiliary tasks, but it is not clear these are strictly necessary to learn a state abstraction. To understand this question, the work proposes a simple soft gating mechanism to the input features, captured by a per-feature gating mechanism, $\sigma$. In this way, if $\sigma$ maps a feature to zero, this encodes the fact that the learner should treat this feature as a distractor. Notably, $\sigma$ is modelled as a soft-gating mechanism rather than a strict one, which facilitates smooth updates to $\sigma$ based on the standard RL loss. The primary experiments come in two forms. First, variants of SAC are contrasted that make use of this learned gating mechanism on just the actor, just the critic, both, an and an oracle (among others). Normalized returns are compared across these methods. Then, in Figure 2, the weight values for the gating mechanism are visualized on a per-feature level and grouped according to whether the feature is a distractor or not. Then, two additional analyses are carried out: the first is in a linear regression setting with distractors (Props 1-3), and the second is a policy evaluation experiment with distractors (Fig. 3 and 4).

### Strengths
The paper has several strengths:
1. The work sets out to study a compelling question: are the standard RL loss and architectures sufficient on their own to learn a state abstraction? While there are some aspects of this question that could be framed in a slightly more precise way, I find the question to be valuable in its own right.
2. The approach is relatively simple: a soft gating mechanism as described by $\sigma$ is lightweight and can be incorporated into many kinds of deep RL architectures and learning settings.
3. The experimental design is clean and affords quick contact with intuition. I especially found Figure 2 to be a compelling experiment in light of the question at the heart of the paper.

### Weaknesses
There are two primary weaknesses (PW.x) that I believe should be discussed further:
- PW.1: The central question is slightly imprecise, and therefore leaves some ambiguity as to the aim and takeaways of the work
- PW.2: The gating mechanism relies on a feature space that is split between either relevant or irrelevant features.

In more detail:

W.1: I believe more could be done to clarify the central question of the work. My suggestion is to dig into what, specifically, it would mean for the question to have a negative answer, and why something like DQN is not already evidence of a positive answer.

In more detail: In some sense, the question being posed has an obvious answer, as evidenced by the advent of deep RL: DQN, Rainbow, and most early deep RL methods were among the first RL methods to achieve high performance in environments with rich input spaces such as images. The premise, in a way, of deep RL was that state abstraction and/or representation learning can be carried out implicitly in the service of value prediction (or control, or other broader goals of the learner). Of course as the present paper points out, much research has been dedicated to trying to learn good representations explicitly through the use of extraneous components like new losses, extra networks (auto-encoders), auxiliary tasks, and so on.

At present I do not believe it is clear what separates the core question ("Are architectural choices and the RL objective alone sufficient to learn abstract state representations?") from what DQN already achieved early on. I believe there is a reasonable answer to this, but I also believe that answer is not quite present in the current paper.

A related point, but one standard definition of state abstraction is in terms of state aggregation (as in Li, Walsh, Littman, 2006). Aggregation is a simple form of function approximation. Since neural networks are much more general forms of function approximators, it should be clear they are capable of state aggregation. So, I am again wondering: what is the precise hypothesis, and in what way does something like DQN not already provide evidence for that hypothesis?

W.2: The function $\sigma$ is relatively simple, and is perhaps only salient in settings when the input features are clearly strictly a distractor, or not, for all time. This seems less applicable in a domain like Atari, where input features are pixels, and pixel values can either take on relevance or not depending on the context. How do you see this framing extending beyond the case when explicit distractors are added, to something more like the Atari setup?


A short note: The proposed gating mechanism is closely related to soft state aggregation (Sing, Jaakkola, Jordan, 1994). I believe some discussion of this connection would be useful.


References:
- Singh, S., Jaakkola, T., & Jordan, M. (1994). Reinforcement learning with soft state aggregation. Advances in neural information processing systems, 7.

### Questions
My primary questions stem from the two major questions of the work.

Q1. What specifically does it mean for a deep RL method to have learned a state abstraction?

I can imagine a few responses:
- (i) It has learned to ignore explicit distractors that are included in the input space.
- (ii) It has learned any compression of the state space
- (iii) It has learned a useful compression of the state space, in line with some of the state abstraction families from Li, Walsh, Littman (2006).
- (iv) It has learned a meaningful state representation that facilitates effective decision making in the environment in question.

I believe this work is focusing on (i). But if so, it should make that clear, and illustrate that at least some past method does not already do this.

Q2: How reliant is the study on the existence of explicit distractors? In other words, how would the framing apply in settings like Atari, where input features are just pixels, and their relevance can evolve over time?

### Soundness
3

### Presentation
3

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
The authors propose methods that learn how to ignore irrelevant environmental variables without the use of auxiliary objectives. Instead, they propose an architectural change: “a learnable, observation-independent attention mask applied to the inputs of the policy and value networks and trained end-to-end using only the RL objective.” They implement and evaluate this approach in several simple environments.

### Strengths
The key idea (learning an attention mask) is simple and intuitive.

The authors provide a fairly extensive and compact survey of much of the important literature.

### Weaknesses
**Questionable assumptions and evaluation:** Practical and general methods for ignoring distractors would seem to necessarily be context-specific. That is, such methods would identify which features are (and are not) relevant to action selection in the current state, even when those features are different in different contexts. However, the authors appear to assume that relevant features are not context-specific (“…existing methods are typically context-dependent, suppressing features only locally when it negatively affects performance, whereas we use a simpler observation-independent mask.”). The fact that context-general suppression of features is successful in the authors’ experiments may have more to do with experimental methodology than with realism. That is, the authors evaluate their proposed methods by inserting features that are always irrelevant, rather than only irrelevant in some contexts.

**Artificial and limited experiments**: Section 6.1 describes a simple regression task and Section 6.2 describes experiments in which “The base environment is a custom MDP with continuous one-dimensional state space and two discrete actions.” These experimental setups seem artificial, even by the standards of traditional RL papers. The paper would be substantially improved by experiments with a greater range of more realistic problems.

### Questions
(none)

### Soundness
3

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
The authors show that architectural choices and RL objective alone are sufficient to learn abstract state representations. Specifically, they mask the inputs to the policy and value networks.

### Strengths
The proposed architectural improvements are simple and can be easily used together with existing RL algorithms. The authors provide theoretical insights on why an attention mask can lead to meaningful state abstractions.

### Weaknesses
The experiments can be strengthened by showing the proposed method scales with the the dimension of the controllable and uncontrollable distractors.

### Questions
1. the proposed method can be viewed as a form of regularization. how does it compare to having a dropout layer in the  actor and critic network? 
2. how does varying the dimension of controllable and uncontrollable distractors affect the performance of the proposed method? 
3. what is the choice of delta (in equation 4) in the experiments? could the proposed method handle larger values of delta?

### Soundness
2

### Presentation
2

### Contribution
2
