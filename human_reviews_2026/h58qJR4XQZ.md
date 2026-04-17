# CBFlownet: Generating Higher-Quality Candidates via Combinatorial Bandits

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 2

## Abstract
As a probabilistic sampling framework, Generative Flow Networks (GFNs) show strong potential for constructing complex combinatorial objects through the sequential composition of elementary components. However, existing {\GFNs} often suffer from excessive exploration over vast state spaces, leading to over-sampling of low-reward regions and convergence to suboptimal distributions. Effectively biasing {\GFNs} toward high-reward solutions remains a non-trivial challenge. In this paper, we propose {\modelname}, which integrates a combinatorial multi-armed bandit (CMAB) framework with GFN policies. The CMAB component prunes low-quality actions, yielding compact subspaces for exploration. Restricting GFNs to these compact subspaces accelerates the discovery of high-value candidates, while the reduced complexity enables faster convergence. Experimental results on multiple tasks demonstrate that {\modelname} generates higher-reward candidates than existing approaches, without sacrificing diversity. All implementations are publicly available at \url{https://anonymous.4open.science/r/CBFlowNet-E0BA/}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces CBFlowNet, a method to overcome the problem of oversampling the low-reward regions and undersampling the high reward regions in GflowNets. 

Methodology:
To do so, they prune the actions that yield to terminal states with a low expected reward. 
They root their method in the framework of combinatorial multi-arms bandit: form a state , given N possible actions (i.e. arms ), they choose the top-K arms (K < N ) according to a score (defined below). The other actions are pruned.
The score for arm i at training step t is simply the combination of: 
    - the mean reward of terminal states that are sampled at time t from arm i  (this is not specified in the paper, but I guess the mean reward is obtained with on-policy sampling), which encourages exploitation,
    - and a term rooted in UCB literature and that increases the score for rarely seen arms (i.e. encourages exploration).  

Experiments:
They run experiments on the bit-sequence task (with a modified and more complex DAG, because the original DAG is a binary tree and pruning an action there means we never visit all the subtree which comes after that action), the molecule design task and the RNA task.
They compare to multiple other baselines () . 
They compare the different methods in terms of : cumulative number of discovered modes, reward of the top-X samples, average reward.  They show that CBFLowNet outperforms other methods, but do not provide a theoretical justification of why it is the case.


Contributions: 
- Adding the UCB term to the score of each arm 
- Converting the GFN-binary trees, which do not yield themselves to action pruning (because this directly removes a full subtree), to non-binary trees

### Strengths
I like how the experiments section is presented:
- Extensive set of experiments on 3 different environments (same as those in the original GflowNet paper)
- Nice visualisation of the evolution of the action values
- Comparing to multiple baselines to train GflowNets with the goal of doing better credit assignment (LS , QGFN, student-teacher)
- Adding the baseline of RandGFN

### Weaknesses
- Except for the added UCB term ( sqrt ( ln(t) / T_i ) )  that encourages exploration, it is not clear what is the added contribution of this work compared to the previous work of Lau et al. It seems to the reviewer that here, the "mean on-policy reward" of each arm is the same as the Q value function of each arm (because Q(s,a) =  Expectation_{ \tau = (s, .. s_f) \sim P_F(.| s, a )}  (  Rew(s_f)). The only difference is that in this work, the value function is not a learned neural net, whereas in Lau et al,  Q was parametrized with a neural network, allowing them to generalise to unseen states. Also, it is claimed in the paper that they introduce pruning and that is was not used in Lau et al, however it has been used there, wee eqs 4 and 5 in https://arxiv.org/pdf/2402.05234 
- The maths and notations is not very rigorous sometimes or not defined 
   - e.g. Formally, if arm = arm′ , then Xt =d X′t must hold   . what is d ? Why is there a " ' "? Could you elaborate what you mean by this sentence?
   - Why are the properties of monononicity and smoothness are introduced?
   - monononicity:  the last sentence is not necessarily true: imagine a flow network that, for each base arm, puts all the flow in the  maximal-reward terminal state reachable from that arm, and zero flow for all other terminal states. However such flow loses the gflownet property of sampling proportionnally to the reward

### Questions
Primary questions:
- Could you explain the difference between your work and the work of Lau et al, except for the UCB added term? It seems to me that the core is the same (except that in your case the score is not learned with a neural net). 
If I were to compare your work with using the Q value function as Lau et al. and adding to it the UCB term to it, what would be the difference? What is the contribution of your work compared to Lau et al?

- Why did you choose to root your work in combinatorial multi armed bandits? I don't see something that is truly combinatorial here, as there is no interaction between the different arms, and the score of each arm is independent of other arms. In the end, the score amounts to the top-K arms, so there is no need to do a search of the best subset of size K in a set of size N.

- Have you done an ablation study using only the mean empirical reward as a score, i.e. without the exploration term in the UCB? 

Secondary questions (not as important):

- Could you elaborate why is independence among base-arm rewards violated under naive pruning? 
- In eq.3 , why is it necessary to normalise the reward for effective exploration? how are the reward normalised if we don't have a normalisation constant?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes CBFlowNet, which integrates the CMAB framework to prune the DAG that leads to proper exploration. Authors try to systematically integrate CMAB into the GFlowNet sampling process. Experiment results validate that the proposed method outperforms prior GFlowNet baselines.

### Strengths
- Over-exploration is one of the prevalent challenges when utilizing GFlowNets in real-world settings [1]. The paper seeks to address this crucial challenge using CMAB framework, a powerful tool for extracting a useful subset of actions.

- Experiment results validate the effectiveness of the proposed method. Especially, it shows a large improvement in the molecule design task, where the action space is much larger than in other benchmarks, making appropriate pruning effective.]


[1] Kim, Minsu, et al. "Local Search GFlowNets." The Twelfth International Conference on Learning Representations.

### Weaknesses
- In bit sequence generation, we can modify the $k$ to study the effect of the trade-off between trajectory lengths and the action space sizes.  With a single experiment, I'm not sure that the proposed method works well with different environment configurations.

- Similar to above, there are biological sequence tasks like GFP and AAV, which contain many more state-independent actions (20 amino acids). It would be nice to add some analysis of whether the proposed method works in those settings.

### Questions
Here are some comments I want to suggest:
- In the introduction, there is a section where training GFlowNets with tempered distribution. It would be nice to add Temp-GFN [1], which tries to learn from the diverse tempered distribution for improving sampling process.

- In Figure 3, some symbols are crashed.

- In the experiment section, it is better to write like "Trajectory Balance (TB, Malkin et al., 2022a)" rather than "Trajectory Balance(TB)(Malkin et al., 2022a)". 

- I cannot find which training objective is used (e.g., FM, DB, TB) for CBFlowNet. Maybe I missed it but it would be nice to clearly indicate that part. I'm also curious about the proposed method can be seamlessly applied to different training strategies.

[1] Kim, Minsu, et al. "Learning to Scale Logits for Temperature-Conditional GFlowNets." Forty-first International Conference on Machine Learning.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes **CBFlowNet**, a hybrid framework that augments **Generative Flow Networks (GFNs)** with a **Combinatorial Multi-Armed Bandit (CMAB)** component. The goal is to **reduce over-exploration** in GFNs by **pruning low-quality actions** via CMAB, thereby **biasing the flow network toward high-reward regions** without sacrificing diversity. The authors instantiate this idea using the **CUCB (Combinatorial Upper Confidence Bound)** algorithm to dynamically select a **super arm** of promising actions, restrict GFN sampling to this subspace, and iteratively refine the selection based on observed rewards. Experiments are conducted on **bit sequence generation**, **molecule design**, and **RNA sequence design**, where CBFlowNet reportedly **discovers more high-reward modes** and **converges faster** than standard GFN baselines.

### Strengths
- **Originality**:  
  The **combination of CMAB and GFNs** is **novel**. While GFNs have been enhanced with Q-values, local search, or adaptive teachers, **using CMAB to prune the action space** is **a new angle** that addresses the **over-exploration** issue in a **principled, bandit-driven** way.

- **Quality**:  
  The paper is **technically sound**, with **careful algorithm design**, **theoretical justification** (monotonicity & bounded-smoothness assumptions), and **extensive ablations** (K, ε, β, alternative arm-selection rules). The **empirical gains** in **mode discovery** and **top-1000 reward** are **consistent** across tasks.

- **Clarity**:  
  The **framework is well-structured**, with **clear notation**, **illustrative figures**, and **detailed pseudocode**. The **decomposition of actions** into **state-dependent** and **state-independent** components is **intuitive** and **generalizable**.

- **Significance**:  
  If **scalable**, CBFlowNet could **broadly improve** **combinatorial generative modeling** in **drug discovery**, **sequence design**, and **program synthesis**, where **high-reward diversity** is **critical**.

### Weaknesses
### W1. **Motivation for CUCB is Weak**
- The paper **assumes** that **CUCB is the right bandit algorithm** without **justifying why**.
- **No comparison** with **other CMAB algorithms** (e.g., **CTS**, **C-KL-UCB**, **ESCB**, **Thompson Sampling**) or **non-bandit pruning** (e.g., **reinforcement learning with curiosity**, **evolutionary search**).
- **Missing ablation**: **Replace CUCB with random pruning** → **same gain?** If so, the **contribution is incremental**.

### W2. **Generality Beyond GFNs is Unexplored**
- The **core idea**—**prune actions via CUCB**—is **not validated** on **other generative frameworks**:
  - **Flow Matching** (no MDP)
  - **Soft Q-Learning** (no flow constraints)
  - **Diffusion Models** (no sequential decisions)
- **No theoretical argument** that **CUCB pruning** is **uniquely compatible** with **GFNs**.  
  → **Limits scope**; readers **cannot tell** if this is **a GFN trick** or **a general principle**.

### W3. **Scalability of Combinatorial Action Pruning is Unaddressed**
- **Super-arm selection** is **top-K** from **N base arms**; **complexity is O(N log K)** per round.
- **N = 10⁵** in **molecule design**; **K = 30** → **~3×10⁶ arm-updates** per CUCB round.
- **No wall-clock breakdown** vs. **standard GFN**; **no memory footprint** as **N grows**.
- **Missing experiment**: **Increase N** (e.g., **10⁵ → 10⁶**) and **report**:
  - **Time per round**
  - **GPU memory**
  - **Regret degradation**



CBFlowNet offers a **creative bandit-based pruning mechanism**, and the **empirical gains are real**. However, the **contribution is incremental** because:

1. **No strong evidence** that **CUCB is critical**—**random pruning** might suffice.
2. **Generality** outside **GFNs** is **unexplored**; could be **a GFN-specific hack**.
3. **Scalability analysis** is **missing**; **10⁵ arms** with **K = 30** already feels **heavy**, and **no ceiling** is provided.

A **rebuttal** that **ablate CUCB vs. random pruning**, **tests Flow Matching**, and **scales N to 10⁶** could **flip** my decision.

### Questions
### Q1. Why CUCB?
> Can you **compare CBFlowNet** with **Thompson-Sampling CMAB** or **ESCB**?  
> If **CUCB is replaced by random top-K pruning**, how much **performance drops**?  
> This would **isolate** the **bandit contribution** from the **pruning contribution**.

### Q2. Does this work outside GFNs?
> Have you **tested CUCB pruning** on **Flow Matching** or **Soft Q-Learning** for **similar tasks**?  
> If **no**, what **theoretical property** of **GFNs** makes **CUCB pruning essential**?

### Q3. Scalability ceiling?
> What is the **largest N** (base arms) you have **successfully run**?  
> Provide **time & memory curves** for **N = 10³, 10⁴, 10⁵, 10⁶**.  
> Is **K = 0.1N** still **tractable** when **N = 10⁶**?

### Q4. Dynamic rewards?
> Section 6 mentions **limitation** under **non-stationary rewards**.  
> Could **non-stationary CMAB** (e.g., **Discounted UCB**, **Sliding-Window UCB**) **extend CBFlowNet** to **dynamic environments**?

### Q5. Baseline fairness?
> **TB-GFN** is **trained 5× longer** in **bit sequence** (50k → 10k steps).  
> Did you **match compute budget** or **wall-clock time**?  
> Please **report** **convergence curves vs. actual compute**.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a strategy for GFlowNet search space pruning to accelerate mode discovery.

### Strengths
- The authors propose a very interesting approach for GFlowNet search space pruning.
- Theoretical analysis is provided for selected components of the method.
- The authors report performance improvement (having said that, I have some questions about the experiments below).

### Weaknesses
- The authors claim that “Optimizing the inverse temperature parameter β presents non-trivial challenges, as its selection critically impacts both the exploration-exploitation balance and training stability in GFlowNets.” I feel like this statement needs a much more detailed discussion, as these challenges are not immediately obvious (past typical hyperparameter optimization).
- In my opinion, this lack of more detailed elaboration on that point is the biggest weakness of the paper. For instance, based on what I understood from the appendix, there was little attention given to beta tuning for baseline methods. As far as I understand, they all operate on the full search space, making the exact choice of beta not as critical, since the methods are compared in the exact same setting; whereas the proposed approach operates on subspaces, which would likely lead to a different choice for optimal beta. Would it be possible to demonstrate that improved performance is not due to a better choice of beta?
- The experimental component of the paper is fairly superficial. Besides perhaps Figure 4a, the remaining results essentially show the performance in terms of discovered modes across different tasks. There is little insight into why the method works (or why it doesn’t [that well], for that matter, as in e.g. RNA3 task).
- The paper has some minor formatting issues (e.g. in terms of whitespaces for citations), and some statements that frankly just don’t read much like a scientific paper (e.g. “What we found extremely strange is that (…)” - to be clear, it’s specifically about the phrasing used).

### Questions
- In the case of redefined base arms (Section 4.1.3), are these used exclusively for the sake of space pruning, or added as actions for the model (i.e. action space is modified)?
- What do “action values” refer to in Figure 4?
- Are the runs done across multiple random seeds? If so, why confidence intervals are not provided?
- When the term “reward” is used in e.g. Section 4.1.2, does this refer to the GFlowNet reward, or is it a terminology clash?

### Soundness
1

### Presentation
2

### Contribution
3
