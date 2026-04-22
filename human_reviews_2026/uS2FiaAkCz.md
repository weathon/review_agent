# Towards Monotonic Improvement in In-Context Reinforcement Learning

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 0, 4, 6

## Abstract
In-Context Reinforcement Learning (ICRL) has emerged as a promising paradigm for developing agents that can rapidly adapt to new tasks by leveraging past experiences as context, without updating their parameters. Recent approaches train large sequence models on monotonic policy improvement data from online RL, aiming to achieve continued improvement in testing time performance. However, our experimental analysis reveals a critical flaw: these models cannot demonstrate continued improvement like the training data during testing time. Theoretically, we identify this phenomenon as  *Contextual Ambiguity*, where the model's own stochastic actions can generate an interaction history that misleadingly resembles that of a sub-optimal policy from the training data, initiating a vicious cycle of poor action selection. To resolve the Contextual Ambiguity, we introduce *Context Value* into training phase and propose **Context Value Informed ICRL** (CV-ICRL). CV-ICRL uses Context Value as an explicit signal representing the ideal performance theoretically achievable by a policy given the current context. As the context expands, Context Value could include more task-relevant information, and therefore the ideal performance should be non-decreasing. We prove that the Context Value tightens the lower bound on the performance gap relative to an ideal, monotonically improving policy. We further propose two methods for estimating Context Value at both training and testing time. Experiments conducted on the Dark Room and Minigrid testbeds demonstrate that CV-ICRL effectively mitigates performance degradation and improves overall ICRL abilities across various tasks and environments. The source code and data of this paper are available at https://anonymous.4open.science/r/towards_monotonic_improvement-E72F.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors conjecture that the performance degradation encountered in algorithm-distillation-like in-context reinforcement learning algorithms is due to having suboptimal decisions in the context due to randomness. To combat it, they propose Context Value Informed ICRL (CV-ICRL) that augments the context with the latent value of the context-optimal policy. They theoretically show that the performance bound is improved when the policy has access to the context value, and empirically demonstrate the effectiveness of CV-ICRL on Dark Room and Minigrid testbeds.

### Strengths
- The authors propose a novel perspective on the cause of the degradation of performance when running algorithm distillation. They term it contextual ambiguity.
- The work presents a novel algorithm called CV-ICRL that augments AD, which the authors claim addresses the contextual ambiguity.
- The empirical study demonstrates improved returns of CV-ICRL and lower performance degradation frequencies.
- The statistical analysis in the empirical study is reasonably rigorous.

### Weaknesses
- One major concern is the lack of rigour in the theoretical claims. For instance, I found Definition 1 ambiguous. It is unclear how a context-optimal policy is defined. One interpretation could be that the context defines an empirical MDP, and the context-optimal policy is the optimal policy that solves that MDP. However, there could be more interpretations.
- Some notations are confusing and overloaded multiple times. As an example, I've seen at least $V$, $V_C$ and $V(s; C)$ used to define different kinds of values. Sometimes, it's hard to tell if it's denoting a function or a value being mapped to.
- The contextual ambiguity is merely a conjecture. The authors did not verify if it is the real culprit. If it is the real cause, one should expect the performance degradation to disappear once the contextual ambiguity is removed.
- I am not sure if the proof of Theorem 1 is correct. Particularly, it is unclear how the authors arrive at line 645 from line 644. As far as I understand, $V$ here is a mapping from the space of state and context to a scalar, while $J$ here is a mapping from the policy space to a scalar. Therefore, I don't see how $||J^* - J||{\infty}$ is an upper bound of $ ||V^* - V||{\infty}$.
- Using the data-generating policy as a proxy for the context-optimal policy seems unjustified. It also renders the method inapplicable in cases where the policy that generates the data is missing or inaccessible.
- CV-ICRL only exhibits a relatively minor performance improvement over the AD baselines. I think it's partly due to AD can already solve most of the tasks in the testbed reasonably well. It would make a stronger case if the authors could demonstrate the robustness of CV-ICRL in tasks where AD's performance degrades significantly.

Minor concerns:
- Though not drastically affecting comprehension, there are frequent grammar and spelling errors across the text. The paper would benefit from polishing the writing.
- The paper would benefit from referencing recent surveys on in-context reinforcement learning (e.g., Moeini et al., 2025) besides meta-reinforcement learning (Beck et al., 2023).

### Questions
The idea of CV-ICRL conditions on the premise that the model can perform accurate policy evaluation by predicting the optimal value based on the context. It is a nontrivial task because the optimal policy can be recovered from the optimal value function. Thus, I wonder why we still wish to use the model as a policy? Why don't we directly use the model to predict, say, the optimal action values, and extract the optimal policy from them?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The work proposes a modification to in context reinforcement learning via a so called context value. This context value is introduced to avoid ambiguities of contexts, by providing a clear label for individual context windows. The proposed context value based ICRL is evaluated for algorithm distillation style ICRL methods. In the empirical evaluation the work shows that their proposed method outperforms the algorithm distillation baseline. The idea of using a context value to avoid disambiguities of contexts is promising and could potentially inform further meta-RL research. However, I believe that the work is far from publication in it's current form. I am doubtful of some of the theoretical ideas as well as the empirical evaluation. Overall I vote for rejection.

### Strengths
The idea of using a context value to avoid disambiguities of contexts is promising and could potentially inform further meta-RL research.

### Weaknesses
I believe that a more thorough discussion of related work would be needed as meta-RL and ICRL methods are not just limited to the AD style approaches. Take for example the work by Melo (https://proceedings.mlr.press/v162/melo22a.html) which modifies the RL$^2$ paradigm to work with transformers. A follow-up on this work showed that cross-episode attention (through a hierarchical transformer architecture) essentially avoids disambiguities in context and enables better learning (https://openreview.net/forum?id=UENQuayzr1). Besides, contextual RL provides a different notion on what context means. In this setting, context is used to learn general policies/value functions to be able to adapt in a zero-shot setting. Works like "Contextualize Me" (https://openreview.net/forum?id=Y42xVBQusn) e.g. show how context-optimal policies behave and how "Context Values" can be used to learn general policies.

When discussing the contextual ambiguity, I fail to see why it is reasonable to assume monotonic improvement, especially when training across different tasks. Policies that are specialized to solve one task perfectly are much more likely to fail on another dissimilar task. Thus the monotonicity assumption seems to likely be false.

Further, Property 1 says that the context value is monotonic simply since adding another tuple to the context provides more information. This seems extremely wrong to me. The additional tuple could provide redundant data which would not increase the information about the task or irrelevant information or otherwise not add anything. In some cases adding more data can actually decrease information content. So I fail to see why the monotonicity should hold for the context value.

When discussing CV-ICRL with estimation of Vc through the context it is stated the reward-to-go is used as supervision signal. Since this value is then used as part the context to the agent. How exactly is that different to the reward-to-go used as context in the decision transformer?

The generalization experiments do not test for out-of-distribution generalization. Out of the 4 "novel" test tasks only the four rooms environment comes close to testing out of distribution capabilities as it is possible to observe states without any walls. All other environments are in the training distribution. Thus the experiments are testing interpolation capabilities but definitely not out-of-distribution capabilities. The survey by Kirk et al (https://www.jair.org/index.php/jair/article/view/14174/26890) provides a clear evaluation protocol for RL to assess out of distribution capabilities. I believe the claim (of the conclusion) that AD-like ICRL algorithms are capable of "learning-to-learn" does not hold.

I fail to see the utility of the ablation studies. While I appreciate that not including $\phi(C)$ provides insights about performance gains from auxiliary task training, not using any other target besides the reward to go would be more informative. Similarly, choosing a random function can not provide any meaningful insights at all. Since $\phi(t)$ was proposed to focus on the monoticity assumption, choosing *any other monotonic function* would be more informative than a random function.

### Questions
* How exactly is that different to the reward-to-go used as context in the decision transformer? 
* Why were only AD style methods considered?
* Why did you not consider a combination of $\phi(C)$ and $\phi(t)$ that tries to incorporate both task knowledge as well as monotonicity?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Context Value Informed ICRL (CV-ICRL), introducing a context value signal during training, which explicitly represents the ideal, theoretically achievable performance given the current context. It is argued that algorithm distillation-like ICRL algorithms work best with monotonically improving context, similar to their training data, while actual test-time contexts often contain noisy behaviors. The authors give theoretical results showing the addition of CV yields a tighter bound for performance, and empirically validates CV-ICRL in grid-world environments, especially regarding task generalization.

### Strengths
- Addresses an important problem in ICRL. Robustness to noise is critical under long-context and OOD scenarios.
  
- Provides theoretical guarantees about the benefits of the introduced context value.
  
- Empirical results show good overall performance with OOD task generalizations.

### Weaknesses
- The definition of context value is a bit vague. How to compute the optimal policy "only based on C, without any other information of $\tau$"?
  
- CV-ICRL seems to use the source policy for each episode of context in its training data as the context-optimal policy. There should be some explanations as to why this property holds.
  
- Experiments are conducted in very simple grid-world environments.

### Questions
- How to operationalize Def.1? Is there a way to practically compute the context-optimal policy?
  
- Arguments about the correctness of training-time context values are missing, and seem to require some non-trivial assumptions (e.g. dataset contains only optimal policies for each environment)

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper deals with in-context reinforcement learning (ICRL) and proposes Context Value Informed ICRL to reduce Contextual Ambiguity. They show that context value tightens the lower bound on the performance gap when considering idealistic monotonically improving policy. They perform experiments across two benchmarks - Dark Room and Minigrid - to demonstrate that their approach works better.

### Strengths
### Strengths:

1. This work identifies and introduces “Contextual Ambiguity” which denotes that even random action taken at early interaction can generate an interaction history that may mislead to assume different context.

2. They prove an improved performance bound with the guidance from the context value.

3. Ablations are conducted legitimately, and the experimental results are encouraging to show the efficacy of the proposed approach.

### Weaknesses
### Weaknesses:

1. Without enough history, the context value in test time still could be sub-optimal. Do you have any bound on how much history it would need to clearly identify the context?

2. While the paper validates the approach on two benchmarks, it is limited to very grid like setup. It would require more thorough experiments across other types of tasks to demonstrate wide applicability. Also, apart from AD-like baselines, comparisons with other strong baselines would strengthen the work.

3. Further, the memory or computational overhead due to these additional components has not been discussed. Also, I would like to see some discussion on the limitations.

### Questions
1. Is there any new hyperparameter that is introduced to learn the context-value?

### Soundness
3

### Presentation
2

### Contribution
3
