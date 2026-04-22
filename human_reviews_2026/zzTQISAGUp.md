# Polychromic Objectives for Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 6, 8, 6

## Abstract
Reinforcement learning fine-tuning (RLFT) is a dominant paradigm for improving pretrained policies for downstream tasks. These pretrained policies, trained on large datasets, produce generations with a broad range of promising but unrefined behaviors. Often, a critical failure mode of RLFT arises when policies lose this diversity and collapse into a handful of easily exploitable outputs. This convergence hinders exploration, which is essential for expanding the capabilities of the pretrained policy and for amplifying the benefits of test-time compute scaling. To address this, we introduce an objective for policy gradient methods that explicitly enforces the exploration and refinement of diverse generations, which we call a polychromic objective. We then show how proximal policy optimization (PPO) can be adapted to optimize this objective. Our method (1) employs vine sampling to collect on-policy rollouts and (2) modifies the advantage function to reflect the advantage under our new objective. Experiments on BabyAI, Minigrid, and Algorithmic Creativity show that our method improves success rates by reliably solving a larger set of environment configurations and generalizes better under large perturbations. Moreover, when given multiple attempts in pass@$k$ experiments, the policy achieves substantially higher coverage, demonstrating its ability to maintain and exploit a diverse repertoire of strategies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The work introduces the notion of set RL, in which agents are maximizing rewards not with respect to individual trajectories, but with respect to a set of trajectories. This formulation is appealing as it naturally allows to directly optimize for reward maximization as well as diversity. Based of the notion of set RL, the authors propose polychromic objectives, which give a practical way of opimizing both for success as well as diversity of the set of trajectories. The work then further discusses how to adapt PPO to work with this particular polychromic objective before evaluating it on three different environments. The Poly-PPO implementation seems ot generally improve over pretrained policies and outperforming baselines. Finally, the work provides a more theoretical analysis on the effect of polychromic objectives on the entropy of a policy.

### Strengths
The notion of set RL seems appealing and could inspire novel learning approaches that are distinct from existing classical RL algorithms. Further, the polychromic objective seems like a good logical consequence of set RL.

### Weaknesses
The work discusses various aspects fairly shallowly. In the beginning, the work reads like it is written just to introduce the idea of set RL. The work raises the expectation in the reader that a more comprehensive discussion of set RL and how it might differentiate itself from multi-objective RL. However, before such a discussion starts, the work pivots to discuss a particular aspect of set RL in the form of polychromic objectives. Before going into depth on polychromic objectives, a particular instantiation of PPO is discussed that makes use of this objective. While the work states in line 166 that the "generality of set RL" has been discussed, the overall work only discusses a very particular instantiation that seems to require a lot of adaptation from classical RL to work.
Overall, a lot of design decisions seem to just fall out of the blue without being adequately discussed. I fail to see why, e.g., no other RL algorithm is being considered for the extension to set RL. Similarly, why is there no ablation on the choice of diversity function for the Poly-PPO implementation? The choice of REINFORCE as a baseline is never justified, nor is the choice of environments.

Taken altogether, the work does not seem fit for publication in its current state and there are too many loose ends that need to be taken care of before I would consider increasing my score. I am happy to adjust my score if the authors show that I have misunderstood crucial aspects but vote for rejection as is.

### Questions
* How is set RL distinct from multi-objective RL? Specifically, how is the notion of "sets" different from pareto-fronts?
* Why did you focus only on PPO and no other RL algorithm?
* Is the choice of diversity function a crucial one or is Poly-PPO fairly insensitive to this choice?
* How did you set the hyperparameters of your method? Did you tune the baselines and your new algorithm?

### Soundness
2

### Presentation
1

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
This paper studies how to induce diverse generations (trajectories) from a
policy trained with reinforcement learning (RL). The motivation is that current
policies trained with RL finetuning (RLFT) tend to collapse to a single mode of
behavior. One symptom of this is that increasing the number of trials afforded
to an RL policy does not increase its performance (i.e., pass@n coverage is
low), because similar trajectories are generated.

Thus, this paper proposes to create RL policies that generate diverse,
high-performing trajectories. This approach begins with defining set RL, an
extension of RL that considers objectives over a set of trajectories created by
a policy rather than over a single trajectory. Within set RL, the paper then
proposes a "polychromic objective" that motivates the trajectories generated by
a policy to be diverse and high-performing. Finally, the paper solves this
objective with an extension of PPO, referred to as "polychromic PPO."
Experiments are conducted on Minigrid, BabyAI, and Algorithmic Creativity,
showing that polychromic PPO can finetune policies to generate trajectories with
higher diversity, as evinced by higher pass@n scores.

### Strengths
1. I found the description of set RL and the polychromic objective intuitive and
   easy to follow. Since I am not as familiar with this field, I do not know how
   novel this objective is. However, I personally have not seen anything like it
   before. The closest I am aware of are works that seek to find diverse
   policies, like Diversity is All You Need and quality diversity algorithms.
   These works seek to find a set of diverse policies, while the current paper
   trains a single policy that can generate diverse trajectories.
1. I think the experimental evaluation is quite thorough, in that it seeks to
   answer a number of important questions about polychromic PPO, and it directly
   addresses the pass@n issue initially raised in the introduction.

### Weaknesses
1. The experiments are only run over 3 seeds, which is quite few. Furthermore,
   there do not seem to be error bars, and statistical testing was not performed
   to verify significant differences. Including these things would make the
   experiments more precise, e.g., statistical significance could be discussed
   instead of saying "achieves slightly lower validity" on lines 313-314. Below
   I list a couple of other minor issues with the presentation of the
   experiments:
   1. Table 1: It is unclear what values are bolded and why.
   1. Figure 2 could be cleaned up a bit. Namely:
      - The text is small and hard to read.
      - The y-axis bounds should bound the values in the graph, e.g., the graphs
        in the leftmost column should have bounds from 30% to 100% rather than
        30% to 90%.
      - The rows and columns should be labeled, instead of having the names only
        be in the caption.
   1. Figure 1 appears after Figure 2 in the paper.
   1. Line 307 is a comma splice.
   1. Figure 3 also has really small text.
1. The Entropy Analysis in Section 5 seems to provide valuable insights into how
   polychromic objectives work, but I found it difficult to follow, in part
   because I do not have background on what "entropy collapse" means. Perhaps
   some background could be provided on why we care to analyze the entropy of
   the policy?

### Questions
Below are minor comments and questions I found while reading the paper:

1. Please use the correct citation format for in-text citations. They should be
   author-year rather than numbers. Per the instructions in the ICLR template:

   > Citations within the text should be based on the natbib package and include
   > the authors' last names and year (with the et al. construct for more than
   > two authors).

1. In Eq. 2, is $s_0$ assumed to be fixed? $s_0$ is not really discussed when
   Eq. 2 is introduced.
1. Line 153-154: grammar mistake; maximize is used twice in the sentence?
1. I found it a bit confusing that the paper mentions Minigrid and BabyAI as two
   separate domains on which polychromic PPO is evaluated (line 64, 249). This
   is confusing because the results only seem to be shown for BabyAI (Table 1)
   and not Minigrid; furthermore, Section 4.2 only discusses BabyAI and not
   Minigrid. Are Minigrid and BabyAI the same benchmark?
1. Diversity in the experiments (Section 4.2) seems to be measured indirectly by looking at the pass@n ratings.
   Is it possible to measure diversity directly by sampling multiple rollouts of
   the policy and computing the diversity of those trajectories, e.g., using the
   diversity functions described on lines 263-264?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the problem in RLFT (Reinforcement Learning Fine-Tuning) where policies lose diversity and "collapse" onto a handful of behaviors. It proposes a new problem, set RL, and improves policy gradient algorithms like PPO. For instance, in the advantage function calculation, it redefines a shared advantage term based on a "polychromic objective" (which simultaneously evaluates reward and diversity); it uses a "vine sampling" strategy to collect the on-policy data needed to evaluate the performance of trajectory sets. Experiments demonstrate that this method can effectively improve the policy's success rate, generalization ability to perturbations, and pass@n coverage.

### Strengths
The paper defines the "Set Reinforcement Learning" (set RL) framework , which generalizes the optimization of a single trajectory in traditional RL to the optimization of a set of trajectories . This framework is not only theoretically novel (for example, the authors derive a performance difference lemma applicable to set RL ), but it also provides a clear mathematical foundation that allows algorithms to directly optimize more complex objectives beyond standard rewards (such as the "polychromic objective" in this paper, which combines reward and diversity ). This holds significant inspiration and meaning for future work.

### Weaknesses
1. The definition of trajectory diversity is not general but must be engineered for each specific environment. For instance, it is defined as visiting different "sets of rooms" in BabyAI/Minigrid but different "sets of nodes" in Algorithmic Creativity.

2. To avoid exponential sampling complexity in long-horizon tasks, the method instead relies on 'vine sampling' . However, this sampling strategy itself imposes a major constraint, as it requires the ability to reset the environment to arbitrary states. This is unrealistic in most real-world robotics tasks, thus severely limiting its practical applicability.

3. The algorithm is "computationally demanding". This high workload stems from the multiple layers of sampling required, including the complex "vine sampling" procedure to gather sets of trajectories and the Monte Carlo sampling used to estimate the value baseline for the polychromic advantage. This makes the algorithm's workload substantially larger than that of standard PPO.

### Questions
1. I am concerned about the requirement for excessive sampling, which appears to be a significant overhead. Could the authors provide a computational complexity analysis for the algorithm?


2. Standard practice for evaluating RL algorithms often involves 'deterministic evaluation', where the action with the highest probability (i.e., argmax) is selected. If the environment is static, collapsing to a single deterministic solution is actually acceptable and often desired. In this context, wouldn't forcing diversity inherently degrade the model's optimal performance? Did the paper include any experiments using this standard deterministic evaluation to compare with the stochastic results?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates a key limitation of reinforcement learning fine-tuning (RLFT) for pretrained policies namely, the collapse of behavioral diversity during fine-tuning, which leads to reduced exploration and exploitable outputs. The authors propose a new polychromic objective for policy gradient methods that explicitly encourages both exploration and refinement of diverse generations. They adapt proximal policy optimization (PPO) to this setting by introducing vine sampling for on-policy data collection and a modified advantage function consistent with the new objective. Experiments on BabyAI, Minigrid, and Algorithmic Creativity benchmarks show that the method improves success rates, generalization under perturbations, and coverage in pass@n evaluations. Overall, the work provides a principled approach to preserving and leveraging diversity in RL fine-tuning, addressing a major failure mode of current RLFT pipelines.

### Strengths
* I appreciate the time the authors spent to differentiate standard RL from set RL. The authors do a good job of establishing the setting before discussing the proposed polychromic algorithm. 
* The results suggest that the polychromic algorithm is that set scoring is indeed useful at maintaining diversity while keeping the main performance/validity equal to that of PPO. 
* This work carefully considers some of the flaws consistent in standard RL, entropy collapse and thoroughly discusses the possibilities of Polychromic PPO failure. 
* I do think the contribution is a interesting novel extension of PPO.

### Weaknesses
* Perhaps the largest issue that the authors did not discuss is the time complexity of the generation of vines for the use of the algorithm, which might be the main trade off readers would consider when determining if the added diversity is worth the complexity cost. It may be the case that the methods were not explicitly fairly compared with respect to this trade off. 
* Going a bit further on the initial point, would it be a fairer comparison to compare against other full Monte Carlo based algorithms that perform roll-out such as MCTS? Maybe I am mistaken, but is it reasonable not to include any runtime analysis against the proposed baselines?

### Questions
Is there anything that can be added besides the conclusion that can better inform the reader of the potential limitations above, or potential truncated sequences, as done by traditional monte carlo methods similar to n step bootstrapping?

### Soundness
3

### Presentation
4

### Contribution
2
