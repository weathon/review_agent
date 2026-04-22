# NEOL: REWARD-GATED ONLINE PLASTICITY FOR SCALABLE NEUROEVOLUTION

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
NeuroEvolution of Augmenting Topologies (NEAT) excels at discovering neural architectures and weights for control tasks (Stanley & Miikkulainen, 2002a).However, direct-encoding forces evolution to discover each connection strength individually; in high-dimensional weight spaces, this yields weak credit assignment and poor scaling on large continuous-control problems (Stanley et al., 2009; Peng et al., 2018). We propose NeuroEvolutionary Online Learning (NEOL), which decouples learning signals: the outer loop uses NEAT for topology search, while an inner, reward-modulated local plasticity rule (Hebbian, Oja, or BCM (Hebb, 1949; Oja, 1982; Bienenstock et al., 1982)) adapts synaptic weights online within episodes. Under fixed interaction budgets and multiple seeds across four standard control benchmarks spanning discrete and continuous action spaces, NEOL achieves higher final returns, tighter variability, and better sample efficiency than pure NEAT; gains are most pronounced in continuous control. These improvements are statistically significant (Wilcoxon rank-sum tests), and ablations indicate that benefits persist even when standard genetic weight mutation is reduced or disabled, evidencing a division of labour between structural evolution and online synaptic credit assignment. A simple, gradient-free separation of
topology search and reward-gated online plasticity reliably boosts performance and robustness, offering a practical template for linking neuroevolution with online learning and a scalable path toward more adaptive neuroevolutionary agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method for simultaneously evolving neural network topology and synaptic weights using an outer-loop/inner-loop framework, and evaluates it on four benchmark tasks.

### Strengths
- The idea of co-evolving topology and synaptic weights in a nested-loop framework is conceptually interesting and, to the best of my knowledge, relatively underexplored in the current literature.

### Weaknesses
1. **Misalignment of the Outer-Loop/Inner-Loop Framework:**  
While the separation of topology evolution (outer loop) and weight adaptation (inner loop) is structurally reasonable, the experimental design raises concerns about the authors’ understanding of the fundamental purpose of such a nested-loop architecture. Typically, the outer loop is expected to optimize meta-parameters (e.g., network topology, learning rules, or plasticity coefficients) across a *distribution* of tasks, so that the inner loop can generalize to *new, related tasks* using the optimized meta-knowledge. However, this paper applies the framework to a *single, stationary task*, which undermines the very rationale for using an outer-loop/inner-loop structure. This design choice appears conceptually flawed and limits the significance of the results.

2. **Over-Simplification of Synaptic Plasticity:**  
The paper adopts a *fixed, homogeneous* plasticity rule (i.e., a single, hand-tuned learning rate η) across all synapses. This is a significant oversimplification. A core objective in many prior works is to *evolve* plasticity rules or their hyperparameters (e.g., learning rates, modulatory signals) in the outer loop, enabling task-specific adaptation. The use of a uniform, human-specified plasticity coefficient not only reduces biological plausibility but also limits the adaptability and expressiveness of the model. The contribution thus risks appearing as an ad-hoc combination of topology evolution and plasticity (i.e., “A+B”) without a principled integration.

3. **Limited and Weak Baselines:**  
   The empirical evaluation lacks breadth and depth. Several relevant and recent works are omitted, particularly those that integrate plasticity with meta-learning, recurrent memory, or neuromodulation. For instance:
   - [1]: Growing with Experience: Growing Neural Networks in Deep Reinforcement Learning
   - [2]: Neuroplastic Expansion in Deep Reinforcement Learning
   - [3] Soltoggio, Andrea, et al. "Evolutionary advantages of neuromodulated plasticity in dynamic, reward-based scenarios." Proceedings of the 11th international conference on artificial life (Alife XI). MIT Press, 2008.
   - [4] Mishra, Nikhil, et al. "A Simple Neural Attentive Meta-Learner." International Conference on Learning Representations. 2018.
   - [5] Joachim Winther Pedersen and Sebastian Risi. Evolving and merging hebbian learning rules: increasing generalization by decreasing the number of rules. In Proceedings of the Genetic and Evolutionary Computation Conference, pp. 892–900, 2021.
   - [6] Wang, Fan, et al. "Evolving Decomposed Plasticity Rules for Information-Bottlenecked Meta-Learning." Transactions on Machine Learning Research.

4. The paper requires substantial revision for clarity and precision. For example:
   - The definition of “M” as “a total number of samples or a total number of interactions with an environment Env until time horizon T” is ambiguous. Please clarify whether M refers to episodes, steps, or transitions, and whether it is fixed or task-dependent.
   - The term “credit assignment” is used repeatedly (e.g., “reward-gated and behaviorally relevant credit assignment”), but its technical meaning is unclear in context. Is this referring to temporal credit assignment in RL, or to a biologically inspired learning signal? If the latter, please explicitly connect it to the plasticity rule and justify its relevance.

### Questions
See the weaknesses

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose NeuroEvolutionary Online Learning (NEOL), a hybrid framework that explicitly decouples structural evolution from weight adaptation. In NEOL, an outer loop employs standard NEAT for topological search, while a novel inner loop performs online, within-episode weight adaptation using reward-modulated local synaptic plasticity rules (specifically Hebbian, Oja's, and BCM). The method is evaluated on four classic control benchmarks (CartPole, LunarLander, BipedalWalker, Hopper), comparing the NEOL variants against a standard NEAT baseline. The results demonstrate that NEOL achieves statistically significant improvements in final return, sample efficiency (as measured by a custom SCORE metric), and solution robustness (lower variance), with gains being most pronounced in the continuous control tasks.

### Strengths
- The experimental comparison to NEAT is thorough. The use of 30 random seeds, multiple benchmarks spanning discrete and continuous action spaces, and appropriate statistical testing (Wilcoxon rank-sum) provides strong evidence for the central claim: that NEOL is a superior alternative to standard NEAT.

### Weaknesses
- The paper's motivation rests on NEAT's poor scaling, a problem that other methods (e.g., HyperNEAT, NEAT-PGS, modern Evolution Strategies) also purport to solve. More importantly, to position NEOL as a practical and relevant algorithm, it must be compared against standard gradient-based RL algorithms (e.g., PPO, SAC) or at least modern gradient-free methods (e.g., Salimans et al., 2017) on the same continuous control tasks. Without this context, it is impossible to know if NEOL is a competitive learning algorithm in 2026 or merely a better version of NEAT.

- While the specific implementation within NEAT may be novel, the high-level concept of a two-timescale system (outer-loop evolution, inner-loop plasticity/learning) is a foundational concept in the field (e.g., the Baldwin effect, and more directly, the extensive work on evolving plastic ANNs cited by the authors, such as Soltoggio et al., 2008, and Najarro & Risi, 2020). The plasticity rules (Hebb, Oja, BCM) and their reward-modulation (Frémaux & Gerstner, 2016) are also pre-existing. The paper's contribution is more an effective engineering integration and rigorous comparison rather than a fundamental mechanistic breakthrough

### Questions
1. The paper states it primarily uses a Lamarckian scheme (WRITE_BACK=True, Line 309), where adapted weights are inherited. An ablation (WRITE_BACK=False) is mentioned but no data is presented. How critical is this Lamarckian property? Does a purely Darwinian approach (where plasticity is only for evaluation fitness, but weights are not written back to the genome) also achieve significant gains over NEAT? This is a key mechanistic question.

2. How does NEOL's final performance and, critically, its sample efficiency (in wall-clock time or total environment steps) compare to well-tuned implementations of PPO or SAC on the Hopper-v3 and BipedalWalker-v3 tasks? This context is essential for positioning the work.

3. Could you clarify the "fixed total interaction budget B" (Line 317)? The SCORE metric (Eq 4) is an AUC, which is dependent on the total time horizon $T$ (or total samples $M$). How was $B$ used to normalize runs with different population sizes (e.g., $P=50$ vs. $P=300$)? Does a larger population run for fewer generations to maintain the same $B$? This is unclear.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces NEOL, a hybrid framework that integrates reward-modulated local plasticity into the NEAT algorithm. The central idea is to decouple topology evolution (handled by NEAT) from weight adaptation (handled by fixed online learning rules such as Hebbian, Oja, or BCM).
 
During each episode, synaptic weights adapt online according to a biologically inspired rule gated by a reward signal. After the episode, cumulative reward is used as the fitness signal for evolution. The authors show that NEOL improves convergence speed, fitness stability, and final performance over standard NEAT across several classic control benchmarks (CartPole, MountainCar, Acrobot, LunarLander).

### Strengths
1. The paper addresses an important research direction, combining evolution and life-time learning 

2.The experiments demonstrate faster and more stable convergence, even in environments that do not require lifetime adaptation. The authors provide reasonable mechanistic explanations (reward smoothing and intrinsic regularization from Oja/BCM rules).

3. The approach could easily be applied to other neuroevolution algorithms beyond NEAT.

### Weaknesses
1.
Evaluation is restricted to low-dimensional control tasks  that do not require within-lifetime adaptation. This makes it difficult to assess NEOL’s claimed advantage as an “online learning” or “adaptive” system. What about the T-maze or something GoalDirection HalfCheetah? 
2. No separate hyperparameter tuning between NEAT and NEOL.
Both methods use the same NEAT settings, except for additional plasticity parameters. 
3. NEOL is compared only to NEAT and an ablated NEAT (η=0). Missing are comparisons to e.g. Najarro & Risi (2020) and other meta-plasticity or evolutionary meta-learning methods. 
4. Limited novelty and missing early work. Other approaches have already combined NEAT with plasticity and are not mentioned, e.g. "Evolving adaptive neural networks with and without adaptive synapses" by Stanley et al.

### Questions
1. Have you attempted any environments requiring within-lifetime adaptation (e.g., GoalDirection Cheetah, T-Maze)?
2. How is the approach different to adaptive NEAT by Stanley et al?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose NEOL, a Neuroevolution approach that combines NEAT for evolving topologies of neural network architectures with online learning mechanims implemented via Hebbian learning (or similar). The authors show that this approach surpasses NEAT in performance across several locomotion task in an RL setting.

### Strengths
The approach is well motivated and the authors explain it clearly. The experimental setting is sensible with a collection of the popular RL tasks serving as way to compare both approaches.

### Weaknesses
First, the claim feels very weak, but more importantly, it has already been done (https://www.cs.utexas.edu/~nn/downloads/papers/stanley.cec03.pdf). The authors need to explain what is new regarding their approach and why this is interesting compared to previous work.

However, even if it was a completely novel approach, it is not clear to me that it is that surprising. Under a Lamarckian setting, if the changes in model weight transfer to offspring during evolution, then is it really that surprising that NEOL surpasses NEAT? That's the minimum I would expect, unless I am misunderstanding something. The authors claim that they include this control but don't show it.

### Questions
1. How does this compare to standard RL. Is there a particular setting where this is better? I am happy to also disregard this question if the authors give me some justification (e.g. they wish to explore more biologically plausible models), but they they need to give me a biologically interesting question they wish to answer. Currently there is a biological inspiration, but no particular question they wish to answer.
2. How much of the learning is driven by topological changes? Do models become deeper for example? There is currently no analysis of how the model is driving performance.

### Soundness
2

### Presentation
3

### Contribution
1
