# SAINT: Attention-Based Policies for Discrete Combinatorial Action Spaces

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
The combinatorial structure of many real-world action spaces leads to exponential growth in the number of possible actions, limiting the effectiveness of conventional reinforcement learning algorithms. Recent approaches for combinatorial action spaces impose factorized or sequential structures over sub-actions, failing to capture complex joint behavior. We introduce the Sub-Action Interaction Network using Transformers (SAINT), a novel policy architecture that represents multi-component actions as unordered sets and models their dependencies via self-attention conditioned on the global state. SAINT is permutation-invariant, sample-efficient, and compatible with standard policy optimization algorithms. In 15 distinct combinatorial environments across three task domains, including environments with nearly 17 million joint actions, SAINT consistently outperforms strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes the Sub-Action Interaction Network (SAINT), a novel policy architecture for discrete combinatorial action spaces. The authors identify that existing methods, such as factorized and autoregressive policies, are limited. Factorized methods fail to model sub-action dependencies, while autoregressive methods impose an arbitrary, permutation-variant order. SAINT addresses this by treating the $A$ sub-actions as an unordered set. The architecture uses a learnable embedding for each sub-action index, conditions it on the global state $s$ via FiLM, and then processes the resulting set of vectors through a Transformer (self-attention) that omits positional encodings to achieve permutation-equivariance. The final context-aware vectors are decoded in parallel to produce per-sub-action distributions, and the entire policy is trained with a standard algorithm (PPO). The method is evaluated on CityFlow, a synthetic navigation environment (CoNE), and three discretized MuJoCo tasks.

### Strengths
1.  **Clarity:** The paper is written with outstanding clarity, making the problem, prior work, and the proposed method very easy to understand.
2.  **Problem Formulation:** The authors correctly identify a key limitation of existing approaches, namely the rigid and often incorrect inductive bias of a fixed autoregressive ordering.
3.  **Architectural Fit:** The idea of using a permutation-equivariant architecture is an elegant and principled solution for the *specific class of problems* where sub-actions are, in fact, an unordered set (e.g., selecting a set of traffic signals to turn green).

### Weaknesses
1.  **Incremental Novelty:** The technical contribution is thin. The method consists of a known neural architecture (self-attention on an unordered set) plugged into a standard, on-policy algorithm (PPO). This is an exercise in architectural engineering, not a new method, and its novelty is limited.
2.  **Fundamentally Questionable Inductive Bias:** The paper's entire motivation rests on the assumption that permutation-equivariance is a *universally desirable* property. This is a strong and, in many cases, incorrect inductive bias. For many (if not most) complex control tasks, sub-actions are *not* interchangeable. For example, in a Humanoid, the action dimensions for the left leg and right leg have distinct, non-exchangeable identities. In such tasks, SAINT's bias is as incorrect as a poorly chosen autoregressive order. The paper fails to discuss this critical limitation, instead presenting the bias as an unequivocal good.
3.  **Weak and Narrowly-Scoped Empirical Validation:** The experiments are not convincing.
    * The primary benchmarks, CityFlow and CoNE, are not standard, challenging testbeds for complex control. They appear to be simple or synthetic environments selected specifically to align with the method's permutation-invariant assumption.
    * The MuJoCo evaluation, which is the most critical, is cherry-picked (only 3 envs) and ultimately unconvincing. In Figure 3a (HalfCheetah), SAINT's performance is **indistinguishable** from the simpler factorized and autoregressive baselines. This result, from the authors' own experiments, directly contradicts the paper's claims and suggests the significant architectural and computational overhead of SAINT is often unwarranted.
    * The **omission** of harder, standard, high-dimensional tasks (e.g., discretized Humanoid) is a major flaw. These tasks have strong, state-dependent, and, crucially, **non-equivariant** action dependencies. Demonstrating performance on such tasks is essential for a paper claiming a general solution for combinatorial action spaces. And many prevailing VLA tasks / datasets are perfect test bed for this paper, I encourage authors test their algo on Open X-Embodiment like dataset. I do notice the in need of simulation env, their are already many of Isaac Sim / Unity based industrial level envs and leave the research for authors.

### Questions
1.  The premise of the paper is that permutation-equivariance is superior to a fixed autoregressive order. How do you defend this assumption for the large class of problems (e.g., Humanoid) where sub-actions are *not* interchangeable and have fixed, distinct identities?
2.  Why were more complex, high-dimensional MuJoCo tasks like Humanoid omitted from the evaluation? These are standard benchmarks and would provide a far more convincing test of the method's capabilities and limitations. (It's common practice to discrete continuous action space)
3.  In Figure 3a (HalfCheetah), SAINT provides no measurable benefit over the baselines. How do you reconcile this with the paper's central claim? Does this not suggest that the method's added complexity is often unnecessary?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Problem:

The paper addresses discrete combinatorial action spaces (each joint action contains multiple coordinated subactions) in RL. Taking the Cartesian product of subactions and applying standard RL is intractable as the action space grows exponentially, while existing simplifications (factorization or autoregressive sequencing) fail to reliably capture complex interaction effects between subactions.


Approach:

SAINT (Sub-Action Interaction Network using Transformers) models actions as unordered sets of subactions and uses self-attention (transformers) conditioned on the global state to capture their dependencies. The design is permutation-invariant, scalable, and compatible with common RL policy optimization algorithms (e.g., PPO, A2C). 


Overall assessment (I am not too familiar with the relevant literature solving the same problem):

The proposed approach, SAINT, takes an important step to address problems where lots of decisions must be made together and coordinated, especially in big systems. It’s most useful where those decisions really influence each other. It’s less useful when each decision is simple and doesn’t depend on the others. Overall, this makes RL smarter for complicated real-world tasks. The evaluation is thorough and convincing.

### Strengths
- The proposed approach can model complex, context-sensitive dependencies in large action spaces. It is permutation invariant, i.e., naturally fits unordered action compositions. 

- The evaluation conducted is extensive and compelling. I appreciate the ablations. Experiments show that the proposed approach consistently outperforms baselines on diverse tasks: state-independent (traffic control), state-dependent (navigation), and weakly dependent (discretized MuJoCo). The scalability of the approach is impressive.

### Weaknesses
- The approach may be less justified for low-dimensional or weakly structured domains.

Suggestions:

- Since combinatorial action spaces are common in offline RL (e.g., healthcare), systematic analysis in off-policy contexts could further establish SAINT's utility.

### Questions
- Did you consider environments where sub-action sets themselves change dynamically (e.g., road closures or reconfiguration in traffic networks)? How extensible is the permutation-invariant approach in such cases?

- Can SAINT be adapted effectively for offline RL, especially in settings with sparse and partial combinatorial action coverage?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a policy architecture that learns representations of combinatorial action space, by treating actions as unordered sets of sub-actions. They show empirically that the proposed method has strong performance.

### Strengths
The authors provide ablations showing the robustness of the  proposed method  on varying dimensionality and varying sub-action dependence.

### Weaknesses
The proposed method can have high computational costs. when action space is large, the learnable embedding vector e_i has high dimension. Adding state conditioning further increase the dimensionality.

### Questions
Why do the authors augment action embedding e_i with the global state, instead of using actions as queries and states as keys? Although the former could capture interactions among sub-actions, it may fail to capture interactions between actions and the state.

Could the authors explain why a categorical distribution is used (first equation in Sec 4.3, instead of the standard softmax?

The baselines in experiments are weak. Could the authors include at least one attention-based policy as a baseline? For existing methods that designed for continuous action space, we  may discretize the continuous actions to accommodate the combinatorial action space. 

How many seeds are used in Table 1-2 and in Figure 2?

### Soundness
2

### Presentation
2

### Contribution
2
