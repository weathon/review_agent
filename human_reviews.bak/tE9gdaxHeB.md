# Inferring Time-Varying Internal Models of Agents Through Dynamic Structure Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
Reinforcement learning (RL) models usually assume a stationary internal model structure of agents, which consists of fixed learning rules and environment representations. However, this assumption does not allow accounting for real problem solving by individuals who can exhibit irrational behaviors or hold inaccurate beliefs about their environment. In this work, we present a novel framework called Dynamic Structure Learning (DSL), which allows agents to adapt their learning rules and internal representations dynamically. This structural flexibility enables a deeper understanding of how individuals learn and adapt in real-world scenarios. The DSL framework reconstructs the most likely sequence of agent structures—sourced from a pool of learning rules and environment models—based on observed behaviors. The method provides insights into how an agent's internal structure model evolves as it transitions between different structures throughout the learning process. We applied our framework to study rat behavior in a maze task. Our results demonstrate that rats progressively refine their mental map of the maze, evolving from a suboptimal representation associated with repetitive errors to an optimal one that guides efficient navigation. Concurrently, their learning rules transition from heuristic-based to more rational approaches.  These findings underscore the importance of both credit assignment and representation learning in complex behaviors. By going beyond simple reward-based associations, our research offers valuable insights into the cognitive mechanisms underlying decision-making in natural intelligence. DSL framework allows better understanding and modeling how individuals in real-world scenarios exhibit a level of adaptability that current AI systems have yet to achieve.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a novel Dynamic Structure Learning (DSL) framework that aims to model the time-varying internal models of agents, focusing on how agents (specifically rats) adapt their learning rules and environment representations in a dynamically evolving manner. Traditional reinforcement learning approaches assume static internal structures in agents, but the DSL framework allows for the agent's learning rules and environmental understanding to adapt over time. The authors apply this framework in a T-maze experiment with rats, demonstrating how the rats refine their mental models of the maze, evolving from suboptimal to optimal representations that minimize errors and maximize rewards.

### Strengths
1. The Dynamic Structure Learning (DSL) framework moves beyond traditional reinforcement learning's static model assumption, allowing agents to dynamically adapt their learning rules and environment representations.
2. By examining both heuristic-based (suboptimal) and rational (optimal) strategies, the paper offers valuable insights into different phases of decision-making and adaptation.

### Weaknesses
1. The experimental design of this framework primarily revolves around a simple T-maze task, making its performance in more complex environments uncertain.
2. The proposed DSL framework infers the agent’s internal structural transitions based on behavioral data; however, contemporary meta-reinforcement learning models (such as PEARL[1] and VariBAD[2]) already efficiently infer environment representations from observational data and can adapt rapidly to new tasks.
3. The four “agent structures” defined in the paper are specific, hand-designed combinations, lacking adaptability. This design might not scale effectively to more complex environments or tasks, as each combination is suited only to fixed situations.
4. The paper emphasizes the rats’ capacity for “imagination,” allowing them to infer the causal structure of the environment and update their internal models accordingly, suggesting that current AI systems cannot infer causal structures from observation. However, World Model-based algorithms (such as Dreamer[3]) effectively model environmental structures by representing environment states in latent space. Through repeated "imagination" or simulation, these models infer which actions lead to various outcomes, thereby forming an understanding of the environment.

[1] Rakelly et al., 2019, Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables

[2] Zintgraf et al., 2020, VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning

[3] Hafner et al., 2023, Mastering Diverse Domains through World Models

### Questions
Please see previous section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this paper, a framework called Dynamic Structure Learning is introduced and applied to modeling rat behavior during learning of a T-Maze task. The internal decision process of each rat is assumed to consist of one of two different learning rules in combination with a sub-optimal or an optimal graph representation of the environment. To infer the most probable sequence of internal agent structures, the parameters of the learning rules are computed first using maximum likelihood estimation, they are then used to compute a joint smoothing distribution, which in turn is used to compute the most likely agent structure for each trial. The paper spends a good portion on discussing why rats might choose to follow a wrong path during task learning.

### Strengths
The paper is original for addressing an interesting problem of inferring internal agent structure from data. The proposed method seems to be adequate for the given task and assumed agent structures.

### Weaknesses
Unfortunately, the paper seems to be very narrow in applicability. If the proposed method could be applied to a more general group of tasks, sadly it is not communicated clearly. The method itself is not well motivated and the T-Maze task is not explained adequately. Explanations of core concepts such as the learning rules are deferred to the Appendix. Instead a great portion of the paper is spent on analyzing individual rat behavior, which may be better deferred to the Appendix.

Most importantly, the assumed agent structures appear to be too simplistic. I.e. the internal maze representation can be flawed in many different ways. The results show that only two of the four possible agent structures are ever inferred from the real-world data, hinting at a flawed premise.

### Questions
- Please explain the hidden rule related to the starting feeder box in detail.
- Do you see a reason as to why strat 2 & 3 are never selected in Fig. 7?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper summarizes an adaptive cognitive framework called Dynamic Structure Learning (DSL). The purpose of DSL is to mitigate the strict assumptions of fixed environment representations and learning rules in Reinforcement Learning. The authors claim that simple reward-based association is not enough to explain adaptive behavior seen in intelligent agents. DSL allows an agent to adaptively change their learning rule and environment representation online. To show this in practice, the authors assume a fixed set of learning rule + environment representation combinations, guided by optimality, and test rats on a reward based maze navigation problem. The experiments are performed on five live male rats, that were first allowed to explore the environment. A subset of the rats exhibit sub-optimal behavior like looping initially, before converging on to the optimal sequence of actions. The authors explain this observation with DSL, that predicts an adaptation, guided by an effort minimizing heuristic. They are able to infer the switch from heuristic-based to reward-based representation and learning via DSL.

### Strengths
1. Real rat experiments: The authors conduct relevant behavior analysis experiments with real rats in reward collecting maze.
2. Interesting interplay between Activity based and Reward based credit assignment. This behavior is analogous to System 1 vs System 2 thinking in Psychology and AI literature.
3. Use of SMDP instead of MDP is more realistic for modeling optimal learning rule.
4. I liked the insight that biological agents like rats could potentially optimize for least action or least $\Delta$ action when frequent rewards are not received. It is complementary to the insights from active inference and free-energy principle. This could help researchers in RL take evidence from neuroscience more seriously

### Weaknesses
1. **A fundamental issue with the sub-optimal behavior assumption**: The authors observe a subset of rats looping around the Right Feeder (Figure 5) and claim that a heuristic-based approach is required to simulate this behavior, and that simple reward based association is not sufficient. I understand why this conclusion would have been made - the authors also propose alternative explanations and reject them. But I think that the claim regarding a heuristic might not hold true in the given rat maze behavior. Let me explain this step-by-step: 
    1. If we consider an agent that receives a reward for a particular sequence of state visits (similar to the authors’ experiment setup), Q-learning is sufficient to explain the looping behavior seen in Figure 5. 
    2. In Q-learning, rewards once obtained are propagated over the previously visited states in a dynamic-programming fashion to obtain a value function estimate for each state (and action). Values for all the states near the goal would be higher. The states farther in the trajectory would have lower values. 
    3. In case of Figure 5, the state value functions in the decreasing order of value would be: RF > B > A > LF based on the trajectory (good.LF). Add to that the fact that rats rarely move backward, according to the authors (Figure 4 caption), we start to see that the only option for the rat, is to move to state A.
    4. If we derive a policy at A from the above value function, we can immediately see the looping behavior (loop.RF in Figure 5).
    5. If the rats choose to explore more in the beginning, with an epsilon-greedy policy (higher noise in the policy), they might stumble across correct sequences more frequently, making the values more instructive of the correct sequence of actions. 
    6. Also note that the values are learnt in a continuous time setting, implying that if Q functions are made expressive enough (using RNNs or autoregressive models), the values over time would not be learnt just for states A, B, LF, RF, but rather for sequences (A→B→RF, LF→A→B→RF). Correct sequences would accrue more value eventually due to the dynamic programming nature of Q functions.
    7. Inverted loops are not seen as much because rats do not move backwards as much, which is essential to observe an inverted loop.
    8. Hence, we see that Q-learning with epsilon greedy exploration alone is sufficient to explain the observed behavior. In essence, if we apply Occam’s razor principle, Q-learning, a simple reward-based algorithm would be more preferable to DSL, a much more complex algorithm.
    9. **Authors,** I would like you hear your thoughts on this observation because this is regarding your central claim that other heuristic based approaches are required to explain the rat’s behavior. **I am open to changing my mind and increasing the score if I have made a mistake in my understanding of the experiment setup and in Q-learning algorithm**.
2. **Writing**: I was not happy with the overall quality of the writing. I found the choice of words and grammar poor, riddled with unnecessary acronyms, very random choice of uppercase and lowercase letters. Below, I mention some of them. I counted around 6-7 non-trivial acronyms (AS, IMR, LR, DRL, CRP, SAEM … etc) being used throughout the paper which made it really hard to parse and read. I eventually had to make a list and refer to it every few sentences. I think an abbreviation for Agent Structure (AS), for example, was unnecessary. I am not sure how popular these acronyms are, but please avoid them unless they are extremely common among researchers in the conference community. Examples of useful acronyms for this community are Reinforcement Learning (RL), Artificial Intelligence(AI), POMDPs, RNNs, … etc. The overall presentation of the content can be improved to cater to the readers with background in Machine Learning, Reinforcement Learning and Computational Neuroscience. 
3. **Connections to computational neuroscience**: Several recent advances in neuroscience like Active Inference [1] and Cognitive Maps [2][3][4] have developed much more sophisticated and mathematically precise models for learning, navigation and planning in the brain. Active inference already models actions in terms of minimizing free energy, which in my opinion is a more general algorithm for explaining behavior. Cognitive maps are environment representations motivated by place cells and grid cells, that quickly model the states for optimal navigation and planning. How do these compare against the authors’ proposed DSL model?

### Questions
My overall choice for reject is mostly because of the weakness (1) I detailed in previous section. I appreciate the efforts that went into the experiments and the design of the algorithm. But since I find that there might be flaws in the fundamental claim, I would request a thorough clarification to change my scores to an accept.

1. If it helps, the authors could perform a simulated maze environment with exact experiment conditions and reward a Q-learning agent with the “good” action sequences to see if loops are observed. Different exploration term “epsilon” might need to be tested. To make sure the behavior is simulated correctly, the authors can visualize the value functions evolving with experience, and also observe the number of “backward” movements the artificial agent makes. If the agent makes too many backward movements, I would not consider the simulation valid as it does not represent the actual rat actions.
2. Clarification on optimal and suboptimal internal maze representation (IMR) → What is the “with feeder box” and “without feeder box” representations? Can you please elaborate?
3. Please make a table for the 4 different Agent Structures/Strategies defined in Page 2 lines 70-80. It will be easier to visualize the different components of each strategy.

### Other Concerns

1. “Good path to LF, Good.LF, is shown in red, while good path to RF, good.RF, is shown in blue.”
    1. Is this reversed? The Figure 1.B does not match the writing. LF path is blue and RF path is red. Am I getting this wrong?
    2. Is the composition good.LF, good.RF really necessary? Maybe use brackets for (good.LF) and (good.RF)? 
    3. Also, maintain consistent casing → good.RF vs Good.LF
2. Figure 3, does the x-axis represent number of paths the rat took over time? Please make it clear in the figure and the caption.

### References

[1] Friston K, FitzGerald T, Rigoli F, Schwartenbeck P, O Doherty J, Pezzulo G. Active inference and learning. Neurosci Biobehav Rev. 2016

[2] Stöckl, C., Yang, Y. & Maass, W. Local prediction-learning in high-dimensional spaces enables neural networks to plan. *Nat Commun* **15**, 2344 (2024).

[3] Whittington, James CR, Joseph Warren, and Tim EJ Behrens. "Relating transformers to models and neural representations of the hippocampal formation." *International Conference on Learning Representations*

[4] Dileep George, Rajeev V. Rikhye, Nishad Gothoskar, J. Swaroop Guntupalli, Antoine Dedieu, and Miguel La ́zaro-Gredilla. Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps. *Nature Communications*

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper considers the problem of varying model and learning structures in sequential decision-making tasks. It specifically focuses on a rat maze task. It presents an approach called dynamic structure learning (DSL) to infer for each episode the underlying most likely model of a given set of candidates. As candidate models it considers combinations of two learning models (CoACA and DRL) and two maze representations. Finally, it presents an analysis of real-world data and simulations using the proposed approach.

### Strengths
The overall problem the paper considers, to infer varying models underlying behavior, is very relevant and interesting, and due to its complexity, it has been approached in only a few works before. In most RL works, a constant internal representation is assumed but it has been shown that humans and animals can adapt their internal representations through learning. With that, this paper represents an attempt to make progress in understanding behavior in the real world.

It is also to appreciate that the method was applied to real-world animal behavior, coming with the potential to demonstrate its applicability to real-world data.

### Weaknesses
While in the beginning, I was excited to read the paper due to its relevant and interesting topic, the process of reading was finally not a pleasant experience for me. In my opinion, the paper is not clearly written and it took me quite a while to understand it.

First, the "experiment" section, does not describe the task that rats have to solve. It is mentioned that rats can freely move around (apparently in both directions) and there is food given at two locations. However, the previously stated "hidden rule" for optimality is not explained. In the evaluation section, Fig. 4 presents valid paths and Fig 6 "suboptimal" and "optimal" maze representations, but no information is provided when the rat is rewarded. It is also still unclear to me whether these maze representations describe the transition or reward function. Unfortunately, there does not seem to be a formal description. I further would have found it more natural to move section 3.3.1 to the experiment section. Also, it is hard to discriminate the different arrows visually in Fig. 6.

For the modeling approach, the reasoning behind modeling choices does not become clear to me, see also my questions on this. Most importantly, I am surprised that the paper tries to make statements about the internal model of the agent for each episode. However, for learning only model-free (and no model-based) algorithms are considered. Possible internal task representations are discrete, with either the "optimal" or one "suboptimal" model. With that, transitions do not seem to depend on learning the task representation but are modeled using a Chinese restaurant process, preferring reusing a past model. With that, it is unclear whether the model actually can capture the representation learning process.

In general, I am unclear about the contribution and generality of the proposed model. With the CRP formulation and the two task representations, it seems very specific to the considered rat maze. The authors do not distinguish between a general approach (applicable to other domains) and the specific rat maze setting.

I also noticed that there is almost no related work discussed in this paper. Even if the topic is somehow niche, there is a lot of related work on learning task representations and learning rules, which should be discussed in a proper section. Some important related works that should be discussed are Bayesian inverse planning (Baker et al., 2009, 2017; Rabinowitz et al., 2018), inference of latent dynamics models (Reddy et al., 2018; Herman et al., 2016; Golub et al., 2013) and inferring reward functions of learning agents (Ramponi et al., 2020; Jacq et al., 2019). A proper comparison would help to contextualize the paper's contribution within the existing literature.

The evaluation of the method is also very limited. 3.1-3.3 does not seem to contribute to the evaluation of the proposed method. In 3.4, the method is applied but the results and implications are not explained in depth. Further, it remains unclear what the implications are of the inference that rats switch from both suboptimal LR and IMR to optimal ones.

Further minor issues are that the terminology of the agent structures (AS) was hard to understand: "LR suboptimal AS" stands for $LR_\text{subopt}$ and $IMR_\text{opt}$, "suboptimal AS" for $LR_\text{subopt}$ and $IMR_\text{subopt}$, etc. It would have helped me to choose terms including for each of $LR$ and $IMR$ the information whether it is optimal or not. Also, the caption of Figure 5 is hard to understand.

With all these issues of presentation, clarity, and my doubts about the validity of the model, I cannot see this draft accepted in its current form.

### Questions
How did you select the two specific learning algorithms CoACA and DRL? Why should a rat switch from a suboptimal learning algorithm to an optimal one?

How does the interplay between model-free learning and the task structure work? What was the rationale for using only model-free learning algorithms?

Line 220: "Rats are more likely to employ an AS based on IMRsubOpt early in the experiment, with a shift to IMRopt based AS once they learn the true structure of the task": Is that an observation or assumption?

What is the intuition to model IMR transitions with a CRP? According to the CRP model, the more the suboptimal model was chosen, the more likely it becomes to choose it again. But why does it make sense to stick to the same model if an animal is learning? What are your assumptions here and what observations/intuitions are they based on? 
How do you justify using discrete task representations rather than a continuous model evolution? I also think that Eq. 2 and 3 could additionally be explained in words, as it takes time to grasp the intuition of the formulas.

Line 94: “Our approach differs from existing methods to infer agents’ internal models in three key aspects. First, we do not assume that the agent always acts optimally” Which other existing methods do you refer to? Note that MaxEnt IRL does not assume that the agent always acts optimally.

Line 108: “DSL framework offers a more nuanced perspective on learning compared to...“ What do you mean here with more nuanced?

What do you need the eligibility trace for in DRL?

Why can a rat not try out new ASs, as soon as once optimal IMR was inferred? Suboptimal ASs still appear possible as long as they were taken in the past.

Line 269: "Since standard particle filters do not give good estimates of the smoothing distribution," I also find this sentence confusing. Particle filters do not solve the particle smoothing problem, but particle smoothers do. What about standard particle smoothing methods, such as FFBS or the reweighting-based particle smoother?

Can your model be easily applied to different scenarios? If yes, how and to which ones?

It seems that Algorithm 1 is not iterative. Do the parameters $\theta$ not depend on the estimates of $x$?

Line 362: "Alternate explanations for the high number of loop errors are possible but they are not in agreement with experiment data" Why are they not in agreement with the data?

Fig. 4 shows valid paths (under the representation?). Which of these paths are rewarded, only the one labeled "Good"? The caption reads "Turning back in reward boxes are not considered as valid paths as the rats rarely moved backward during the experiment": What is meant by this?

For the evaluation in section 3.4, it would be helpful to clarify: 1) What exactly it means for rats to switch from suboptimal to optimal LR and IMR, 2) How these switches relate to the learning process, and 3) What insights these results provide about the rats' cognitive processes.

Do you have mentioned the runtime of your algorithm?

### Soundness
3

### Presentation
1

### Contribution
2
