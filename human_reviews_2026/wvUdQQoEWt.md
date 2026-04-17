# Hierarchical W-Learning

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Inspired by a model of the brain called projective simulation, which has attracted interest among physicists in recent years, we develop a simple and generic new method for hierarchical reinforcement learning in this article. The proposed method generalizes the action-value Q function to  W function, enabling the agent to execute actions according to a hierarchical strategy. In the first part of the article, we present a rigorous construction of the hierarchical structure, along with the W-learning algorithm and the hierarchical policy gradient theorem. In the second part, as an example, we illustrate the W-learning procedure in the context of a navigation task. Experimental results show that the introduction of the hierarchical structure can lead to better performance than traditional Q-learning, provided the strategy is well designed and the update parameters are appropriately chosen. Various policy gradient methods are also investigated.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors claim to „develop a simple and generic new method for hierarchical reinforcement learning in this article“.

### Strengths
n/a

### Weaknesses
The experimental evaluation is insufficient. Just a very  simple (scalable) grid world problem is considered.
 
More importantly, the text does not study a new RL approach in isolation.
The experiments mainly compare two settings: RL without prior knowledge about the task and RL with prior knowledge. Any reasonable algorithm that incorporates meaningful prior knowledge should perform better, right?

It is not about hierarchical vs non-hierarchical, because all „hierarchical“ methods have more information about the task than the standard methods.

In the experiments, the „W-learning“ method is given extra information about the task.
See Figure 3, left branch, which gives hints for a target in the upper right corner (as it is the case in the first experiments) or, explicitly, the definition of F, the optimal policy w/o obstacle, in line 380. The function is provide in the „W-learning“ experiments, see Fog. 6.

There are many control experiments that are not considered, for example:
1. Simplest one: Start standard Q-learning with the Q function initialised to the optimal policy without obstacles (i.e. start with Q^F in a table).
2. Starting from an agnostic policy pi with parameter vector theta, define the policy pi’(s) = \theta’ F(s) + (1-theta’) pi(s) that incorporate prior information and use some actor-critic method to learn pi’ , i.e., learn (theta, theta’) .

Another comment regarding the presented experiments/results:
As the objective functions are different, the learning rates could be significantly different for Q- and „W“-learning.
While it is appreciated that several learning rates were tested, it would be interesting to see the results for an even higher learning rate for Q-learning (which is perhaps cooled down over time). Would there still be a significant difference?
What about a different initialisations of the Q function?



## Minor:

The factorisation in (3) seems to be only for a chain, not for a general hierarchical model.

„and many people believe that building a proper model of it may finally lead to artificial intelligence.“: This is a completely empty statement w/o giving a (working) definition of AI.

„As we can see, the clip structure is a model for the neural network in the brain.“: No, I do not see this. The figure shows some generic acyclic graph. This is totally empty statement. That there is no recurrence in the figure makes it extra sad.

What have the path integrals in quantum mechanics to do with the study?

The authors could check out the literature on policy/state-space factorisation.

Who is "et al." in the reference "Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT
press Cambridge, 1998."?

### Questions
See questions in "Weaknesses" above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a hierarchical reinforcement learning approach called W-learning, The method is based on abstractions called "Clips" which are intermediate structures or goals between the perception and the action. Rather than learning to map perceptions to actions, the execution engine goes through a sequence of clips until the final clit selects the action. Q-learning is generalized to W-learning, where W is a function of state, strategy and action. The approach is illustrated in two grid navigation domains where the intermediate "clips" correspond to  the task of reaching the final destination and the task of avoiding obstacles. The hierarchical approach is shown to perform better than the non-hierarchical approach.

### Strengths
Hierarchical reinforcement learning is a well-studied topic and yet not fully understood. The paper addresses an interesting problem and presents an apparently successful approach. The empirical results are reasonable.

### Weaknesses
The paper does not define the terms precisely enough to evaluate the soundness of the approach. Words like "Clip" and "strategy" need to be more precisely defined. It cites most relevant work, but does not explain how this work is different. For example the options or MAXQ framework addresses hierarchical RL in a much more precise notation and correctness proofs. The current framework is different, but it does not explain the embedded assumptions carefully and does not give the proof of the main theorem. 

To illustrate in a problem, it is not clear how one should interpret Figure1. The clips do not sequentially follow each other, but look like a directed acyclic graph. It is said that the clip structure is a "model of the brain". But what is its connection to the strategy hierarchy? Why is the brain relevant to model here? As a contrasting example, MAXQ is organized as a graph structure over tasks, where tasks have termination conditions, and are similar to subroutines. The procedural semantics of the MAXQ hierarchy allows a sound derivation of global value function in terms of local value functions. While the paper seems to be attempting a similar decomposition, it is much less clear because the strategies here are not tasks and they do not have termination conditions.

Appendices A and B are missing. 

It is also not clear what notion of optimality is applicable, since there are in general many notions (recursive optimality, hierarchical optimality, global optimality) might be in play.  It appears that V(s) should be a function of the strategy, but the paper does not acknowledge that. The meanings of different W functions should be clearly stated.  

Hierarchical policy gradient theorem: Can someone view the hierarchical policy controlled by a set of parameters as simply a policy that takes states and policy parameters and outputs a primitive action. If so, then can't someone just use the policy gradient theorem? 

The grid domains are too weak to illustrate the power of the framework. The domains seem to be setup such that each primitive action falls into one of the other goals. What happens if some actions fall under both goals, i.e, avoiding obstacles and also moving towards the goal. More ambitious domains have been attempted in the past literature on hierarchical RL. Multiple levels of hierarchies would be more interesting.

### Questions
Question: Clearly define the framework. What is a strategy? How does the hierarchical policy work given a set of parameters? Can hierarchical policy gradient can be viewed as an instance of simple policy gradient with a new (hierarchical) parameterization.

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
This paper proposes a method for hierarchically learning the Q-function in reinforcement learning (RL), inspired by projective simulation. A new W-function is introduced, which incorporates both the state sss and a hierarchical strategy $\omega$. The authors derive corresponding policy gradient theorems and present an empirical experiment on a navigation task demonstrating that the proposed method achieves faster convergence compared with standard Q-learning.

### Strengths
* The idea of incorporating a strategy component, in addition to actions, into the RL learning process is interesting and establishes a meaningful connection to hierarchical RL.
* The paper provides a detailed and illustrative navigation task that effectively bridges theoretical definitions (e.g.,  $\omega$ and $g$) with practical scenarios.

### Weaknesses
Motivation: The hierarchical modeling of the Q-function is not well motivated. Although the authors briefly mention “modeling intra-option behavior” and “handling dynamic tasks,” these ideas are neither theoretically developed nor empirically verified.

Experimental validation: The experiments are insufficient to convincingly demonstrate the advantages of W-learning over existing methods such as Q-learning.

- Evaluation is limited to a single navigation task, with comparison only against Q-learning.
- Even within this prototype task, W-learning achieves similar asymptotic performance to Q-learning, differing mainly in faster convergence speed.

Clarity of definitions: The relationship between the strategy $\omega$, action $a$, and W-function could be more clearly defined.

- Based on the paper’s description, ω\omegaω appears to act as an internal or intermediate action, and the W-function seems analogous to a Q-function extended with this additional variable. However, in the navigation task example, the strategy behaves more like a subgoal, which could be more explicitly articulated.

Minor:
The paper should use `\citep{}` instead of `\cite{}` for most references.

### Questions
How should one design and map the strategy variable $\omega$ in practice? Since this appears to be highly task-specific, does the initialization of  $\omega$ significantly affect performance, and how might this approach scale to more complex tasks with many possible actions and strategies?

### Soundness
2

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
This paper present a novel hierarchical reinforcement learning algorithm that generalize flat action-value Q to a new hierarchical W function, allowing the agent to execute actions according to an hierarchical strategy. The idea is inspired by the model of the brain called projective simulation, and each node in the hierarchy represent a separate task. The paper shows that this formulation side step the need to learn a termination function like we do in the option framework and is possible to learn all the tasks provided that the strategy could be represented in terms of a hierarchical clip structure. The learning algorithm for this new W function is extended to both the classic off-policy learning with an update rule derived from Q learning and to the actor-critic case by means of generalizing the policy gradient.

### Strengths
- The paper tackles the crucial and long-standing goal of developing effective hierarchical reinforcement learning (HRL) agents. This is a highly relevant research direction with the potential to significantly advance the field's ability to solve complex, long-horizon problems.

- While the theoretical development is dense, the simple and well-executed experiments provide an intuitive grounding for the W-learning concept. This section effectively clarifies the method's practical application and benefits.

- The W-learning formulation is novel and elegantly sidesteps significant limitations of prior HRL frameworks. Notably, it removes the need to explicitly learn termination functions and it seems to be able to converge to the optimal policy without the risk of hierarchical suboptimality.

### Weaknesses
- The paper is confusing the exact structure of a task/ clip structure is vague and hard to grasp. The paper spends much time on proposing an off-policy and actor-critic learning algorithms but fails to introduce and explain very well the new concepts introduced for a strategy and a clip unit making the paper hard to follow. I would suggest to first explain in detail the new idea and the hierarchical structure needed for this learning algorithm together with his properties and limitations before moving on to practical algorithms for learning that.

- The paper importantly is missing a broad discussion on the differences between the proposed new method and the methods already proposed in the literature like Option framework, Max Q, Feudal, Option Critic. Adding this discussion could help to clarify the differences of the proposed method compared to previous methods and highlight the advantages / disadvantages of the proposed methodology.

- The experiments are limited to a very easy discrete tabular domain, and the comparison is only against flat Q learning and only using the off-policy variant of W learning algorithm. 

- One of the main challenges modern HRL is facing is to move from hand-crafted hierarchies to learned hierarchies, this paper yet introduce a new framework that still relies on hand-crafted hierarchies and a broader discussion on how this framework could be extended to learned hierarchies is needed.

### Questions
See Weaknesses points.

### Soundness
3

### Presentation
1

### Contribution
2
