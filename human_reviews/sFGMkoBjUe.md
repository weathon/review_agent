# Lookahead Shielding for Regular Safety Properties in Reinforcement Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 3, 5

## Abstract
To deploy reinforcement learning (RL) systems in real-world scenarios we need to consider requirements such as safety and constraint compliance, rather than blindly maximizing for reward. In this paper we develop a lookahead shielding framework for RL with regular safety properties, which on the contrary to prior shielding methodologies requires minimal prior knowledge. At each environment step our framework aims to satisfy the regular safety property for a bounded horizon with high-probability, for the tabular setting we provide provable guarantees. We compare our setup to some common algorithms developed for the constrained Markov decision process (CMDP), and we demonstrate the effectiveness and scalability of our framework by extensively evaluating our framework in both tabular and deep RL environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the problem of safety in Reinforcement Learning by shielding. The core concept of shielding is to have an agent learn a policy while preventing it from taking harmful actions. In this paper, a new shielding method is proposed where two policies are learned separately: One to maximize the reward, and another to maximize the probability the agent stays in a set of "probabilistic safe states". The combined approach then follows the former policy while switching to the latter if the agent would leave the aforementioned probabilistic safe states.

### Strengths
While the core idea of shielding has been explored quite a lot recently, and many of the concepts used in the paper (such as proabilistic safe sets) are known in the literature, combining the methods in this way has (to the best of my knowledge) not been considered yet. The approach comes with a probabilistic (step-wise) safety guarantees that are computable and importantly only come with very mild assumptions (such as knowing the number of states) compared to previous safe appraches requiring additional information such as the entire transition structure. An unavoidable restriction is necessary is to assume the safe backup policy to be learned correctly (in particular that it identified the probabilistic safe states correctly), however in practice there is no way around this with this framework. This limitation is adressed in the appendix where the shielding approach is once provided with the correct backup policy and once has to be learned, with the results showing convergence to a near-optimal backup policy within relatively few samples.

The paper starts with an extensive section on the formal background which provides all necessary background knowledge to make the paper self-contained. The paper is mostly strucured and it is easy to follow the main concepts in the paper. Proofs for all statements are provided in the appendix. Related work is thoroughly discussed at the end of the paper.

I was especially impressed with the experimental evaluation in which the new proposed method outperforms state-of-the-art approaches in accumulated reward while simultaneously showing equally safe or even safer behaviour throughout the different examples. This holds for boith the tabular and deep RL setting. The implementation is also fully provided in the supplementary meterial. The appendix contains further details and insights of the experiment, including all hyperparameters used.

### Weaknesses
The main weaknesses of the paper are imprecisions in some formal sections.

Although the theoretical background is rigorously defined for the most part, it unfortunately lacks precision at some of the most important places: In Eq. 1 the shielding strategy -- the core contribution of the paper -- is defined. However, the definition is ambiguous: the term $Pr(⟨s, q⟩ \models \lozenge^{\leq N} accept)$ is not defined since $Pr$ was only defined on Markov Chains, not MDP. It is not entirely clear which policy is used to resolve the non-determinism in the MDP: $\pi_{safe}$ or $\pi_{safe}^N$ or the minimizing policy (from my understanding these are not necessarily the same if $\pi_{safe}$ is learned incorrectly)? Also the "(given $a$)" part is ambiguous -- I suppose from context that this means you actually resolve the non-determinism by performing one step according to $\pi_{task}$, and afterwards accoring to some safe policy. This should be clarified.

A similar question arises with the definition of the backup policy $\pi_{safe}$: In Remark 3.2 it is exaplained that the objective $\pi_{safe}$ is ought to minimize is non-markov, however only memoryless policies are defined and the paper. Also the time step is not an explicit input in the definition of $\pi_{shield}$ -- does this mean you restrict $\pi_{safe}$ to memoryless strategies?

For assumption 3.8 it is not clear how this is supposed to be parsed. It starts by "For all irrecoverable actions $a\in A$ ..." but then never uses $a$. Also I would expect this to be an implication, i.e. "if $Pr(...) > \varepsilon$ then $N\geq N^{*}$" from context, but I am not sure here.

---

While the experimental section is convincing and I have no reason to doubt the results, for reproduction of the results a proper explanation of how to setup the the experiments (e.g. required packages) and how to obtain the results in the paper and use the provided code (e.g. as a README) is instrumental. Ideally, a docker container or similar should be provided to avoid issues with other platforms and user setups.

For the presentation, I suggest emphasizing the separation from existing methods, in particular extending section 3.2. On the technical side this is quite clear since the CMDP approach is also formally defined here, however an intuition is missing here. For readers it would be insightful to know what the core difference behind the approaches are and why/for which kinds of models one would expect this shielding approach to outperform others such as CMDP.

While Theorem 3.10 provides theoretical guarantees on the safety it is not clear how practical these are, i.e. if the required number of samples is reasonable, or astronomically large. An application of Thm. 3.10 to the experimental section would be also be desirable discuss the tightness of the guarantees, i.e. whether there is a substantial gap between guaranteed safety and observed safety.

As a minor comment, Theorem 3.10 is highly related to [R3], where the problem of relating the number of samples to the probability that Eq. (5) holds is tackled directly. This should be cited. However, I also want to mention that their results are only an improvement if a bound on the branching factor is known (other than the trivial bound of number of states), see [R4].

### Questions
I have some questions for understanding Definition 3.1:

You say you train the backup policy with the goal of minimizing discounted expected cost. (l. 206) However this introduces a non-linear weighting of costs which may lead to the backup policy preferring some policy $\pi_1$ over $\pi_2$, even though the expected number of steps until reaching appect might be higher in $\pi_2$. **Why do you chose the discounted cost over the undiscounted cost which is easily learnable in the finite-horizon setting?** (Remark: In the infinite horizon case it would also not be necessary to introduce the $\omega$-automaton in Definition 3.1 since optimizing for your objective is equivalent to minimizing the (discounted) expected number of steps until the accept state in a single run, correct?)

In Remark 3.2 you write "It is important to note that for regular safety properties the corresponding cost function is defined over the product states and is thus non-Markov", implying the product Markov chain is responsible for the cost function being non-Markov. However, **isn't the finite horizon the culprint of making the cost function non-Markov**, i.e. would the cost function be Markov if you defined it as the infinite-horizon objective?

In my mind, using the infinite-horizon cost as objective (either discounted, or long-run average cost [aka mean payoff]) would be a more reasonable choice. In the finite horizon setting you specifically allow the backup policy to lead to unsafe behaviour, as long as it can guarantee it happens after the time horizon. This is highly undesirable in many practical settings. **How do you justify choosing the finite-horizon cost as the objective?**

---

Other questions:

* Can you clarify the ambiguities I pointed out in the "Weaknesses" section, i.e. (1) Which policy is used to resolve the non-determinism when determining whether to use $\pi_{task}$? (2) Can you formally states how "(given a)" is to be interpreted? Does the resolving policy always pick $a$ in $s$ or only for one step? (3) Is $\pi_{safe}$ memoryless? (4) Can you clarify Assumption 3.8?
* The shielding policy is presented as a binary choice between the preferred maximizing action and the safe backup action. Since usually an RL agent evaluates all action, it can usually also rank all actions. Have you considered extending this to a setting where the action with the highest ranking is chosen, restricted to those actions which satisfy the condition in Eq. (1)?
* Can you provide an example of the theoretical bounds of Thm. 3.10 applied to some examples in your experimental setup? How far are the bounds from the actual rate of unsafe behaviour observed?

---

minor comments

* the paper uses a mix of (mostly) AE and (some) BE, e.g. "minimize" vs. "maximise" and i.e. sometimes followed by a comma and sometimes not
* for intuitive understanding of safety properties, it might be beneficial to give the accepting state a negative name (e.g. "fail") instead of accept
* line 168: $\pi_{safe}$ is not necessarily low-reward
* line 237: "...is modified slightly to update give zero reward..." - I guess either "give" or "update" is too much here
* for the statistical model checking approach, I would suggest more recent citations that provide methods that are more sample efficient than Hoeffding's inequality, e.g. [R1,R2]
* for understanding, it would be helpful to define probabilistic safe sets (Def 3.6) earlier since you already mention them a lot in section 3
* l. 326: " Intuitively the ‘backup policy’ πsafe defines an (T -step) probabilistic safe set" -- the definion of the probabilistic safe set is independent of $\pi_{safe}$. Actually it is the other way around that $\pi_{safe}$ depends on the probabilistic safe set.

[R1] Jegourel, Cyrille & Sun, Jun & Dong, Jin. (2019). Sequential Schemes for Frequentist Estimation of Properties in Statistical Model Checking. ACM Transactions on Modeling and Computer Simulation. 29. 1-22. 10.1145/3310226. 

[R2] Meggendorfer, T., Weininger, M., & Wienhöft, P. (2024). What Are the Odds? Improving the foundations of Statistical Model Checking. ArXiv, abs/2404.05424.

[R3] Weissman, T., Ordentlich, E., Seroussi, G., Verdú, S., & Weinberger, M.J. (2003). Inequalities for the L1 Deviation of the Empirical Distribution.

[R4] Wienhoft, P., Suilen, M., Simão, T.D., Dubslaff, C., Baier, C., & Jansen, N. (2023). More for Less: Safe Policy Improvement With Stronger Performance Guarantees. International Joint Conference on Artificial Intelligence.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents "lookahead shielding," a framework for reinforcement learning with regular safety properties that requires minimal prior knowledge compared to previous shielding approaches. The framework operates by dynamically switching between a reward-maximizing "task policy" and a safety-focused "backup policy" based on probabilistic safety assessments using finite horizon model checking with learned environment dynamics. The paper provides theoretical analysis comparing the approach to constrained Markov decision process (CMDP) methods and proves safety guarantees for the tabular setting under reasonable assumptions. Through experiments in both tabular environments (media streaming, bridge crossing, and gridworld variants) and the Atari environment using deep RL, the paper demonstrates empirical effectiveness compared to baseline approaches like PPO-Lagrangian and constrained policy optimization (CPO). The key innovation lies in handling regular safety properties (which are non-Markovian) while requiring minimal prior knowledge of the environment, unlike previous approaches that demanded strong assumptions about environment dynamics or topology.

### Strengths
The paper reformulates the safety-constrained RL problem by separating concerns between a task policy and a backup policy, while introducing a dynamic switching mechanism based on probabilistic safety assessments. Also, the paper presents an integration of finite-horizon model checking with learned dynamics. It handles both exact and statistical model checking approaches, providing error bounds and computational complexity analysis for each case. The paper provides a treatment of regular safety properties, which are inherently non-Markovian. The approach of using deterministic finite automata (DFA) for specification and product construction for verification, while maintaining computational tractability, is interesting. Furthermore, the framework demonstrates sample efficiency through its use of counterfactual experiences in the backup policy training and integration of model-based planning with learned dynamics.

### Weaknesses
- The theoretical guarantees rely heavily on Assumptions 3.8 and 3.9, which are quite strong. Assumption 3.8 requires that any irrecoverable actions can be identified within a finite horizon N*, but there's no practical method provided to determine this horizon. 
- The product construction between the MDP and safety automata can lead to state space explosion. While the paper briefly mentions this issue, it doesn't provide a solution. For large DFAs or complex safety properties, the product state space could become intractable.
- The safety guarantees rely on union bounds across timesteps, which could be overly conservative in practice. For long episodes, this could lead to conservative behavior or require small per-time step violation probabilities.
- The need for repeated model-checking during execution could introduce significant computational overhead, especially in real-time applications.
- For complex safety properties, constructing appropriate DFAs could be challenging, and the paper does not address automated methods for DFA construction or optimization. The size of the DFA directly impacts computational complexity.

### Questions
- The union bound for episodic safety seems very conservative. Have you explored tighter bounds or alternative approaches? For instance, could martingale-based techniques provide tighter guarantees?
- How can we verify Assumption 3.8 (existence of finite N* for identifying irrecoverable actions) in practice? Could you provide a method or heuristic for estimating N*?
- For large DFAs representing complex safety properties, the product state space becomes intractable. Have you explored hierarchical or compositional approaches to manage this complexity?
- The backup policy seems crucial for performance. Could you elaborate on alternative designs for the backup policy beyond the current cost-based approach?
- Could you provide more details about the real-time performance requirements? What is the typical computation time for model-checking in your deep RL experiments?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents an approach to safe RL in which a 'shielding' style-approach is used, where absolute violations of constraints can be considered in contrast to CMDP-style approaches which consider violations of constraints in expectation over trajectories. The approach leverages a learned dynamics model (using prior work on world model learning e.g. DreamerV3) and learns two policies on-the-fly during training: one that is performant with respect to the reward function of the MDP, and one that will minimize (absolute) constraint violations in the form of a PCTL property. The "safe" policy is trained by seeking to minimize constraint violations that is cast in the form of a DFA that penalizes violating prefixes for the safety property. The approach outperforms baselines that consider CMDP approaches in both tabular and deep RL experiments.

### Strengths
The area of safe RL, and the consideration of a broad class of absolute constraints is an important and challenging problem. Bringing more modern learning-based techniques into the fold of safe RL approaches is valuable and I am glad the authors are trying to modernize more classical techniques like shielding by acknowledging the current limitations and applicability of the space. The experimental results show that CMDP approaches, as expected, are unable to satisfy the safety properties while achieving reward-performant policies.

### Weaknesses
First and foremost, the writing of the paper has a large number of typographical errors and clarity issues regarding the phrasing and wording of particular sentences and would heavily benefit from a thorough editing pass.

From the introduction of the paper, I was not fully clear on exactly how this work would build on prior efforts and be distinct from the (very large) body of work on addressing logical or absolute constraints. Ultimately, my understanding is that existing works do not usually try to learn the safe "fallback" policy, and some works make prior assumptions on dynamics (although this isn't really true, most works nowadays do not make such assumptions.) 

The theoretical guarantees made in the work are fairly conservative, which the authors acknowledge and I appreciate that. However, the guarantees made here also rely on the quality of the *learned* dynamics model and safe policy - in practice, we'll rarely know for sure that these learned components are 'good enough'. As a result, I would want to see a number of ablations in the experimental results to understand how and when the approach will work as expected, and if not, what those circumstances are. For example, how complicated do the safety properties need to be for the approach to start breaking down? As mentioned in remark 3.2, large DFAs that make convergence slow may hurt the approach's efficacy, even with counterfactual ER. When does this happen? Also, DFA or automaton-guided learning for temporal logic properties are *very* well-explored. Can we explore different choices of reward shaping or learning for the safe policy based on this?

A smaller point: using CER relies on Q-learning approaches, which are pretty generally worse than PPO/SAC for deep RL. This will limit empirical success. Prior works have developed logical CER approaches for PPO (see Eventual Discounting Temporal Logic Counterfactual Experience Replay from Voloshin et. al. 2023) that can be used.

Overall, from a contribution standpoint, I don't find that the step forward from existing works is significant, and the novel components of the work are not explored enough to highlight them. I think the interesting parts of using a DFA as a means to learn cost-minimizing policy and the relationship between the switching policy, the learned dynamics model, and the safe policy are not fully ablated in experimental results which makes it hard to fully appreciate the proposed approach. A more thorough distinction and description of related work would help the work a lot - the related works section in the main text does not highlight the technical differences enough.

Additional comments: Section 3.2 that relates the approach to CMDPs isn't totally necessary, it's pretty clear the two things are different. This section, in my opinion, can be struck and the extra space can be used to expand on the points I mentioned in previous paragraphs. There are some potential errors in the preliminary definitions (i.e. line 84: a state, action transition should map to just a single distribution over states instead of a distribution over distributions) that can be cleaned up.

### Questions
My questions are also mentioned in the previous section.

* How do different choices of reward shaping or learning for the safe policy based affect performance?
* How complicated do the safety properties need to be for the approach to start breaking down?
* Have you tried comparing your approach to previous approaches to temporal-logic constrained policy optimization e.g. [1] and [2], rather than just CMDP-based approaches?

[1] Policy Optimization with Linear Temporal Logic Constraints. Voloshin et. al. 2022.
[2] LTL-Constrained Policy Optimization with Cycle Experience Replay. Shah et. al. 2024.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents an approach to ensure safety in reinforcement learning through a framework called lookahead shielding. This method dynamically identifies and avoids unsafe actions during both training and deployment by using a backup policy that activates whenever potential safety violations are detected. The authors validate their framework across various environments, showing it can achieve a balance between safety and performance, outperforming conventional constrained Markov Decision Process methods in some cases.

### Strengths
The paper presents an approach to safety in RL through its lookahead shielding framework, which requires minimal prior knowledge of the environment's dynamics. The framework dynamically assesses the safety of actions, adjusting in response to evolving environments. Extensive empirical validation in both tabular and deep RL environments demonstrates the method’s scalability and effectiveness in balancing reward and safety. Additionally, by comparing with constrained Markov Decision Process (CMDP) methods, the authors offer insights into the framework's performance relative to traditional approaches.

### Weaknesses
The framework seems to rely on a lookahead mechanism that verifies the probability of reaching an unsafe state after an action is selected, rather than ensuring safety beforehand. This approach implies that actions may be applied before verification results fully confirm safety, potentially allowing unsafe actions to affect the system before they are flagged. In contrast, a real-time synthesis approach would formally generate safe actions upfront, guaranteeing that no unsafe actions are applied. The lack of explicit real-time formal synthesis in the paper suggests that the framework provides posteriori (post-hoc) safety assurances rather than proactively constraining actions in real time. This could limit its effectiveness in safety-critical applications where preemptive safety guarantees are essential. For applications that require complete prevention of unsafe actions, a real-time synthesis approach might be more appropriate, as it directly generates actions that comply with safety constraints before execution. Am I missing something here?

Additionally, there is already a body of work on online shielding that the paper does not cite or compare with. Notable works include:

"Online Shielding for Stochastic Systems" by Bettina Könighofer, Julian Rudolf, Alexander Palmisano, Martin Tappler, and Roderick Bloem.

"Online Shielding for Reinforcement Learning" by Bettina Könighofer, Julian Rudolf, Alexander Palmisano, Martin Tappler, and Roderick Bloem.

"Safe Reinforcement Learning with Nonlinear Dynamics via Model Predictive Shielding" by Osbert Bastani.

### Questions
Could you provide an explanation of the novelty of your results in comparison to the following papers:

"Online Shielding for Stochastic Systems" by Bettina Könighofer, Julian Rudolf, Alexander Palmisano, Martin Tappler, and Roderick Bloem.

"Online Shielding for Reinforcement Learning" by Bettina Könighofer, Julian Rudolf, Alexander Palmisano, Martin Tappler, and Roderick Bloem.

"Safe Reinforcement Learning with Nonlinear Dynamics via Model Predictive Shielding" by Osbert Bastani.

### Soundness
3

### Presentation
3

### Contribution
2
