# Exploration Implies Data Augmentation: Generalisation in Contextual MDPs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
In the zero-shot policy transfer (ZSPT) setting for contextual Markov decision processes (MDP), agents train on a fixed set of contexts and must generalise to new ones. Recent work has argued and demonstrated that increased exploration can improve this generalisation, by training on more states in the training contexts. In this paper, we demonstrate that training on more states can indeed improve generalisation, but can come at a cost of reducing the accuracy of the learned value function which should not benefit generalisation. We hypothesise and demonstrate that using exploration to increase the agent's coverage while also increasing the accuracy improves generalisation even more. Inspired by this, we propose a method Explore-Go that implements an exploration phase at the beginning of each episode, which can be combined with existing on- and off-policy RL algorithms and significantly improves generalisation even in partially observable MDPs. We demonstrate the effectiveness of Explore-Go when combined with several popular algorithms and show an increase in generalisation performance across several environments. With this, we hope to provide practitioners with a simple modification that can improve the generalisation of their agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of zero-shot generalization in contextual Markov decision processes (CMDPs). In particular, the problem is investigated through the effects of exploration on aiding the generalization of RL agents. The authors demonstrate that training on more states that are reachable (i.e., those with non-zero probability of being encountered during training) can improve generalization to unreachable states during evaluation. Then, a method named Explore-Go is introduced, which consists of increasing the diversity of the starting state distribution by, at the start of every training episode, following a pure exploratory policy for a random number of steps, and then actually starting the episode from the state the agent ended in. Experiments were conducted in the FourRoom, ViZDoom, and DMC, and compared the performance and value error of the baseline method with that of the Temporally Equalised
Exploration (TEE) in the FourRoom domain.

### Strengths
- The paper is well-structured and well-written (easy to follow).
- The proposed method is simple and easily applicable, making it possible to combine with different methods that tackle orthogonal problems.

### Weaknesses
- Although the proposed technique improves test performance, it has been shown to sometimes decrease performance in the context of the training distribution.
- The setting only considers contextual MDPs that do not have context-conditioned transitions and rewards. Although the authors state that this is the case in many benchmarks, the assumption that the context is observable in $s$ can be very restricting.
- Although Section 3 discusses the problem intuitively, the paper could be strengthened with a more formal/mathematical analysis of the proposed method. For instance, what is the impact of the number of random steps $K$ on the generalization capabilities of the agent? How much does the method reduce the impact of inaccurate targets?
- Most of the experimental analysis focused on the simpler FourRoom domain. Moreover, the results in the DMC and ViZDoom benchmarks did not include the competing state-of-the-art baseline TEE.
- The gains in test performance are not that significant in some tasks (e.g., DMC, see Fig. 10).

### Questions
Below, I have a few questions/additional comments:

- Why can TEE not be used with on-policy approaches?

- “Any CMDP where the context is observable in s′ is in this subset, but multiple contexts can also share states s′.” This is not very clear to me. Do the authors mean that you are considering states in the form $s’=(s,c)$? In this case, T(s’, c’, a) = T(s’, a), but $c’ \neq c$? I suggest clarifying this, perhaps with a clear example.

- “Line 187: in which case they will never share the same underlying state”. If contexts never share the same underlying state, why consider a CMDP formulation? It seems to me that each context is actually a different MDP.

Minor:

- Line 92: The common convention is to use parentheses instead of brackets for defining tuples.

- Line 291: “To improve generalization by both training on additional states and providing accurate targets for these states, we propose them as the starting states of additional training contexts.” What is “them” in this sentence?

### Soundness
3

### Presentation
3

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
The authors introduce, identify, and address an issue with misled value targets in reinforcement learning policies when exploring. The proposed method holds out on training the policy in the first (uniformly sampled) $k$ steps of an episode. Meanwhile, a separate exploration policy guides the agent. Transitions for training only include data after the exploration phase ends at $k$.

### Strengths
The introduced method appears* simple, effectively introducing an alternative online policy which is dedicated to exploring the environment, and adding additional data to the experience replay.

*the exploration policy does not appear to be defined in detail. See questions.

### Weaknesses
My main questions seek a bit more clarity on the core assumptions and experimental design. I'd appreciate a clearer justification for the exploration problem as framed, and a bit more high-level context for the foundational concepts built upon, like Jiang et al. and $\pi_{PE}$. My primary concern is that the main experiment's training intervention (more start states) seems to directly correlate with the test evaluation (new start states), so I would appreciate clarity on what other axes of generalization are being measured. Please see my questions.

### Questions
“Recently Jiang et al. (2023) demonstrated that continuous exploration throughout training can improve
generalisation to test contexts. One of their core algorithmic contributions, temporally equalised
exploration (TEE), assigns different exploration coefficients β
… To separate the effects of continuous and better exploration, this paper uses a TEE variant where parallel workers follow count-based intrinsic reward with different β.”
It seems this work is referenced several times (lines 220, 263) throughout the work and appears to be a central foundation on which the submitted work relies. If this is the case, it is certainly deserving of more high-level explanation and summary than the technical brief that is given.
For example, elaboration for “intrinsic reward with different \beta” would be helpful (or a brief aside indicating that it is not crucial to understand).

What exactly is $\pi_{PE}$? The only explanation I can find for it is in lines 302-304, which gives a _very_ high-level definition. However, it is used in the pseudocode for Explore-Go in the appendix. Is it uniformly random? Is it entropy or information maximizing? Exploration is incredibly nuanced and its exact definition in $\pi_{PE}$ needs to be made clear in order to understand the implications of the experiments.

In the experiments, (a) the training environment is a grid-world-like environment in which the objective is to move from starting location to end location. (b) In the test environment, only environments with different starting and ending goals from the training environment are used. (c) Training with Explore-Go effectively only changes the distribution of starting states. Then, is it not obvious that a method which trains agents on more starting locations will certainly improve performance? Since, augmenting starting locations is directly correlated with the generalization evaluation. I feel that I must be missing something – I would appreciate clarification on what other axes of generalization are being evaluated in this experiment. Without this, I am afraid I do not see how the results move beyond the straightforward general machine learning hypotheses of “training on data closer to the test data will improve test scores.” Is there perhaps another experiment environment that might de-correlate the results with the coverage of starting states? For instance, a locomotion task where the contexts are within some friction range and the unreachable context lies outside this range?

In figure 4d, what does $\beta=0.5$ eventually come down to, provided enough frames?

“To get an accurate estimate of Qπ (s′, a′), we need to observe the consequences of choosing a′ in s′, but exploration often picks actions the policy would not. Therefore, bootstrapping from state-actions that are not trained on makes the targets of many exploratory transitions inaccurate.” It is certainly possible to model Q as a distribution such that the policy is picking actions from a distribution created by normalizing Q while it is still uncertain about its actions. In this case, sampling the policy and estimating Q is exactly the same. I believe what the authors mean to argue is perhaps a randomly exploring policy where its actions are completely unrelated to the task. If this is apparent from following Jiang et al., I would again encourage authors to dedicate more material to explaining (or reproducing and reframing) that reference before.

### Soundness
2

### Presentation
4

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
This paper proposes Explore-Go, a simple training procedure that improves generalization in contextual Markov decision processes (CMDPs) by expanding the coverage of initial states. Concretely, each episode begins with a short exploratory prefix to reach a diverse set of states; learning then proceeds from these reached states as if they were new “starts.” The method is evaluated on Four Rooms, DeepMind Control (DMC), and VizDoom, where it shows consistent gains over standard baselines.

### Strengths
1. Rich quantitative analysis. The experiments track multiple metrics, yielding conclusions that feel natural rather than cherry-picked.  
2. Simplicity and practicality. The idea is straightforward and easy to implement in common RL codebases.  
3. Thorough ablations. The paper includes many ablation studies that clarify which components matter.  
4. Strong empirical results. Within the tested settings, the method reaches or matches state-of-the-art performance.

### Weaknesses
1. Over-narrow CMDP setting (under-signaled).  
   The paper effectively restricts to a CMDP subclass where context only alters the initial-state distribution—a perfect match to Explore-Go’s design. This specialization is neither reflected in the title nor called out clearly in Section 2’s definition, which risks misinterpretation. As a result, the method can feel tailored to this specific subclass; its general significance is under-argued. The paper should explain why this CMDP subclass is important in practice.

2. Comparisons are limited beyond Four Rooms.  
   On DMC and VizDoom, the method is compared mainly against SAC, RAD, and APPO, which do not emphasize exploration. This leaves Four Rooms as the only environment where Explore-Go is contrasted with exploration-heavy baselines, making it harder to claim superiority in broader exploration settings. Given the paper’s already narrow problem scope, and that baselines like TEE are applicable more widely, the comparative value of these experiments is reduced.

3. Sample-efficiency cost is acknowledged but unquantified.  
   The method discards exploratory prefixes for training targets, which likely hurts sample efficiency. The paper notes this qualitatively but does not offer a direct, controlled measurement (e.g., equal wall-clock, equal updates, equal environment steps) to show the true cost-benefit trade-off.

4. Key insights remain largely intuitive.  
   Several claims—such as how exploration mitigates spurious correlations—are supported mainly by intuition and a toy example in Section 3. That example, however, relies on extremely low coverage relative to state dimensionality, which is highly constructed and may not reflect realistic regimes. Stronger validation in more complex settings is needed to substantiate these insights.

### Questions
1. Why modify the default reward in Four Rooms for the experiments? What behavior or evaluation property required this change?  
2. A substantial portion of the paper motivates exploration and its trade-offs, which is relatively familiar. The crucial unresolved point is why the chosen CMDP subclass (context alters only the start distribution) is broadly important. Could you add more environments to demonstrate that many practical tasks fit this subclass, thereby strengthening the case?  
3. Some legend placements are confusing—for example, in Figure 4, “Evaluation” appears in (a) while “Algorithm” appears in (d). Could you standardize or clarify legend placement?  
4. In Figure 4, the DQN baseline appears undertrained. With sufficient training time or a tuned schedule, does DQN approach Explore-Go’s performance? If not, what concrete barrier remains?

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
4

### Summary
The paper proposes to use exploration as a starting phase to increase generalization in contextual MDPs. The authors motivate this idea as a form of augmentation - training on more initial states artificially increases the number of contexts, which can potentially break spurious correlations that usually occur when training on a small number of contexts. Moreover, the paper argues that a naive application of exploration can reduce the accuracy of the learned value function, thereby hindering generalization. To address this, the proposed algorithm Explore-Go demonstrates improved generalization performance across several environments.

### Strengths
* The idea of using exploration for data augmentation to generate more contexts (tasks) in CMDP is new and well-motivated. I particularly like the illustrative CMDP task in Figure 2, which clearly explains the motivation for the proposed method. 

* The paper provides a thorough discussion on related work on exploration for generalization and their connection to this work.

* The experiments section contains 3 different domains, providing diverse evaluation scenarios.

### Weaknesses
* The writing and presentation could be improved. For example, Figure 2's illustration of how exploration improves generalization could be moved to the introduction to better motivate the method and engage readers earlier.

* The reasons why naive exploration reduces the accuracy of the learned value function are not clear to me.

### Questions
* The problem of wrong targets (inaccurate value function): I completely agree that incorporating state space exploration in the training process as an adaptive intrinsic reward to the extrinsic reward (as done in the SOTA TEE), leads to training on wrong targets- it depends on the weight coefficient between the intrinsic and extrinsic rewards if the target is to continue explor or to exploit (reach to the task’s goal). This mix of two contradicting rewards can lead the agent to learn the wrong target (continue to explore when it should exploit).  However, why this leads to an inaccurate value function is not clear to me. Could you clarify this point?

* When the value error in Figures 1 and 4 is calculated, is it the error between the extrinsic reward (what should be the true reward) and the intrinsic + extrinsic reward (for TEE and DQN with β = 0.5)? 

* In Figures 1 and 4 - Why is the value error of vanilla DQN higher than Explore-Go?

* Do you use memory in the initial exploration phase of Explore-Go?

* How would this method work for non-navigation tasks when the training environments have unrecoverable states? I assume the training performance decreases, but how will it affect the test and generalization performance?  

* For future work: an agent trained for pure exploration is known to exhibit less task-specific behavior that generalizes to unseen tasks (such as following the “right-hand rule” for solving mazes). Did you try combining pure exploration with Explore-Go at test time or as an initialization step? I wonder if pure exploration could improve generalization beyond simply providing data augmentation that breaks spurious correlations - perhaps by learning a more general behavior.

More writing suggestions: consider reducing the number of footnotes and integrating them into the main text to make the reading more fluent.

### Soundness
2

### Presentation
1

### Contribution
3
