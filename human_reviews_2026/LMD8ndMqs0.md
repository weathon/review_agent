# Is Temporal Difference Learning the Gold Standard for Stitching in RL?

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Reinforcement learning (RL) promises to solve long-horizon tasks even when training data contains only short fragments of the behaviors.
This *experience stitching* capability is often viewed as the purview of temporal difference (TD) methods. However, outside of small tabular settings, trajectories never intersect, calling into question this conventional wisdom.
Moreover, the common belief is that Monte Carlo (MC) methods should not be able to recombine experience, yet it remains unclear whether function approximation could result in a form of implicit stitching.
The goal of this paper is to empirically study whether the conventional wisdom about stitching actually holds in settings where function approximation is used.
We empirically demonstrate that Monte Carlo (MC) methods can also achieve experience stitching.
While TD methods do achieve slightly stronger capabilities than MC methods (in line with conventional wisdom), this gap narrows as we use larger neural networks.
Furthermore, we find that increasing critic capacity effectively reduces the generalization gap for both the MC and TD methods.
These results suggest that the traditional TD inductive bias for stitching may be less necessary in the era of large models for RL and, in some cases, may offer diminishing returns.
Additionally, our results suggest that stitching, a form of generalization unique to the RL setting, might be achieved not through specialized algorithms (temporal difference learning) but rather through the same recipe that has provided generalization in other machine learning settings (via scale).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper revisits a foundational belief in reinforcement learning that temporal difference (TD) learning uniquely enables “experience stitching,” the ability to recombine short trajectory fragments to solve long-horizon tasks. Using carefully controlled goal-conditioned grid environments, the authors design tasks that isolate stitching phenomena across three regimes: no stitching, exact stitching, and generalized stitching. Through extensive experiments comparing TD-based (DQN) and Monte Carlo (MC)-based (C-learning, CRL) methods under different model scales, they find that MC methods can also stitch effectively when model capacity is large, and that scaling the critic reduces generalization gaps more than the TD–MC distinction itself.

### Strengths
The study is conceptually insightful and challenges a long-standing assumption in RL with well-designed empirical evidence. The authors introduce a simple yet precise benchmark for studying compositional generalization and provide a clear taxonomy of stitching types. Experimental methodology is strong, consistent architectures, hyperparameters, and evaluation protocols across all baselines ensure fairness and reproducibility. Results are clear: while TD offers a small edge in exact stitching, model scale dominates as the key factor for achieving stitching and generalization. The paper is well-written, balanced, and highly reproducible.

### Weaknesses
The work’s main limitation lies in its restricted experimental scope: it focuses on discrete grid-worlds, leaving open whether the same conclusions hold in continuous or visual domains. The paper does not thoroughly analyze exploration efficiency, representation overlap, or the mechanism behind MC stitching. Some theoretical justification for why MC updates might yield implicit compositionality would make the findings more complete. Additionally, the effects of training scale versus network capacity are not fully disentangled, and the runtime or computational costs of scaling are not reported.

### Questions
The paper could be strengthened in several ways:
1.	It would be better if the work is extended to more complex domains, such as continuous-control or visual goal-conditioned environments, to test generality beyond discrete grids.
2.	It would be better if the authors include exploration analysis, examining how data diversity or sampling entropy affects stitching success.
3.	It would be better if the paper incorporates actor-critic baselines (e.g., SAC or PPO with goal conditioning) to verify whether the findings hold in standard continuous RL architectures.
4.	It would be better if discussions on theoretical links between representation learning and compositional generalization are given, offering intuition for why scaling may replace traditional TD bootstrapping.

### Soundness
2

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
This paper investigates the ability of RL algorithms to compose previously observed fragments of behavior to achieve longer-horizon goals, known as experience stitching. To explore this, the authors introduce a controllable benchmark using the Sokoban game, where agents are trained to perform simple object manipulation tasks and then tested on more complex manipulation tasks that require the composition (or stitching) of previously learned skills.

Building upon this benchmark, the paper presents a series of experiments comparing several Temporal Difference (TD) and Monte Carlo (MC) based RL algorithms. Several key findings are reported: first, MC methods can stitch experiences as well as TD-based methods when the evaluation trajectories lie within the support of the training data. Second, increasing the capacity of the value function allows both TD and MC methods to better generalize when evaluating their ability to compose skills.

The paper has some good ideas such as the taxonomy of Stitching and the testbed to measure Stitching generalization. It has a clear presentation and cohesive writing. But it suffers from poor empirical practices and mischaracterizing and missing prior work. For these reasons I recommend rejecting the paper and I believe with some rewriting and rerunning experiments more carefully, it can be a good paper.

### Strengths
* This paper explains and discusses Stitching as a specific way RL algorithms can generalize to new problems. This investigation is novel and interesting to the goal-conditioned RL and Skill and Options communities.
* The writing is coherent and easy to follow, with good use of diagrams to discuss the details of experiments.
*  The introduced testbed is clearly defined, and its design choices and details are explained and justified.
* Discussions around different variants of Stitching and how each experiment is set up to study each variant are coherent and clear.
* The paper mostly manages to place its findings in the context of prior work and justify its existence (with a couple of exceptions, see Weaknesses section)

### Weaknesses
* Experiments suffer from several poor empirical practices, which do not allow the reader to fully evaluate the findings.
  * Untuned hyperparameters: all algorithms share the same hyperparameters, and there is no description of how they are selected. This leads me to believe that they are not tuned for this new problem. The performance of untuned algorithms can vary greatly on new problems (Patterson et al., 2023). This fact alone is a major obstacle to trusting the outcome of experiments.
  * The authors report the Inter-Quartile Mean (IQM) of 5 seeds as the performance measure. IQM is not an appropriate choice for this experiment setup because it further obscures the data and shows an average over ~3 seeds, excluding the best and worst performing trials. IQM was designed to aggregate over many problems where the range of returns can vary greatly (e.g., Atari games), not a single problem with a consistent success measure. A better choice would have been to show individual trials if the compute budget only permits 5 seeds.
* Some experiment details are missing or not justified.
  * It is unclear if the evaluation is periodic or only at the end of training. If more than one evaluation is taking place, how was the final number being reported calculated?
  * The experiment is run for 500 million steps, which is a very large number. Why was this number chosen? If it is possible to significantly reduce this number, the rest of the computing budget could have been used to run more trials, which would have made the results more trustworthy.
  *  The appendix reveals the authors' use of a parallel environment setup, but this is not mentioned in the main body. This is an important decision and should be clearly pointed out.
* The authors, on several occasions, fail to define terms that should have been (e.g., waypoint, critic) or make claims about the RL community’s collective beliefs that I do not agree with (e.g., MC methods can not stitch together experiences).
* There is some missing literature that should be included for a complete treatment of the field. There is no mention of the Options framework (e.g., Sutton et al., 1999) or Skill chaining (e.g., Konidaris and Barto, 2009), both are quite related to this paper’s topic.

minor issues,
* Line 106, s and s^\prime should be swapped based on Figure 2
* Lines 108 - 117, labels do not match Figure 2
* Line 247-248, mention by construction twice
* Figure 8: Icons at the (1, 1) coordinates
* different kinds of stitching introduced in the related works section



references
- Patterson, A., Neumann, S., White, M., & White, A. (2023). Empirical Design in Reinforcement Learning. ArXiv, abs/2304.01315.
- Konidaris, G.D., & Barto, A.G. (2009). Skill Discovery in Continuous Reinforcement Learning Domains using Skill Chaining. Neural Information Processing Systems.
- Sutton, R.S., Precup, D., & Singh, S. (1999). Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. Artif. Intell., 112, 181-211.

### Questions
1. Why was IQM chosen as the reported metric?
2. How are the hyper parameters chosen? and why this choice is justified given that experiments are in a new problem where the algorithms were not tuned for?
3. Line 107, why Sutton Barto cited here? The book does not really cover such an experiment setup
4. Line 319, why was the TD paper cited here? Is there a part of this paper that suggests TD methods can compose sub-behaviours whereas MC methods can not?
5. What is the source of the common wisdom that MC methods can not Stitch experiences together while MC methods can?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper compares MC-method and TD-learning in terms of experience sticthing. The authors first characterize a three different stitching scheme : no-stitching (where no end-to end pairs are present in the training set), and exact-stitching  where train set only contains way points, generalized stitching where there are two different way points. Through experiments in a simplified environment, the authors show that MC-based methods can also achieve similar capabilities to that of TD-methods when the critic network capacity is sufficiently large

### Strengths
1. The authors tackle an importanat probelm of stitching performance of MC methods and TD methods. The argument that when the critic size is large, there is no significant gap in stitching performance of TD and MC methods is new and will be helpful to the community.

2. The authors try to formalize the stitching concept and the experiments are constructed in solid sense. The designed examples clearly represent the three difference scenarios of stitching that the authors consider.

### Weaknesses
1. It is not clear why three different scenarios ( exact stitching, no stitching, generalized stitching) could be representative scenarios to evaluate stitching performance. 
 
2. Structure of presentation : The related works section and preliminaries section seems to be somewhat not balanced : preliminaries overlap with related works making the preliminaries part too short. The authors could provide more detail on their setting. For example, the replay buffer $\mathcal{D}$ is loosely defined, and it is unclear whether it consists of sequence of trajectory or random i.i.d. pairs.

3. Definition of open and closed evaluation presented on page 5: The authors could have adopted a more formal mathematical framework to define these concepts, rather than relying solely on a literal or descriptive explanation.

### Questions
1. In the generalized stitching case, do we have $w\to w^{\prime}$ in the training set?


2. What is $\mathrm{supp}(\mathcal{D})$? What do the authors mean by the state support?


3. In Figure 4, when there aer four boxes, the success rate of DQN is than 0.5. Can we consider this as a successful DQN model which we try to do evaluation?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper considers the ability for algorithms to stitch segments of experience together, something typically seen as a key property of temporal difference methods as they explicitly encourage this through bootstrapping. The authors note that with complex feature spaces, the odds of paths intersecting is vanishingly small that the resulting stitching is enabled through generalization between nearby states. They use this to motivate a systematic study of stitching, and empirically demonstrate that Monte Carlo returns—an extreme which does no explicit stitching—can solve problems that require stitching. They further show that scaling the size of the network generally improves stitching ability, suggesting that techniques which promote generalization within neural networks are more important for the ability to stitch than temporal difference updates.

### Strengths
* The paper is well-motivated. While TD is often contrasted with MC by this ability to stitch, such intuitions are presented in tabular settings. The observation that generalization is necessary for this to happen with function approximation is lesser discussed.

* The paper provides a useful classification of stitching regimes: no stitching, exact stitching, and generalized stitching.

* They further detail an environment setup and how start/goal states can be configured to systematically test for each stitching regime with function approximation. Through this controlled experimental setup, they highlight two key observations: MC can sometimes stitch, and that scale is a key enabler of stitching.

### Weaknesses
* The evidence of MC stitching was notable in the generalized stitching regime, but surprisingly significantly less prevalent in the exact stitching case. This seems weird, if the setup which cleanly sets up trajectories for stitching led to worse stitching ability?

* Assuming GCDQN (MC) is a sound algorithm, there's no comparison between GCDQN (TD) and GCDQN (MC) in the comparisons of Section 5.2, despite it being used in 5.3. It feels like it would be a fairer and more convincing comparison between TD and MC if the base algorithm can be controlled. It's hard to draw conclusions between TD vs. MC if the results overall show substantial variability by underlying method (e.g., CRL and C-LEARN are both MC yet perform dramatically differently).

* Only 5 seeds were used which has been repeatedly shown to be questionable with regard to making proper statistical comparisons for the claims being made (e.g., Henderson et al., 2017; Colas et al., 2018; Patterson et al, 2023; Patterson et al, 2024). It would strengthen the paper to justify the number of seeds (e.g., by observing the trends in success rate over task difficulty, making it a larger number of seeds over a distribution of tasks, etc.).

### Questions
* The low performance of DQN MC was noted to potentially be an exploration issue—can the authors elaborate on why they believe this to be the case?

* On another note, how exactly was DQN MC implemented? It seems to be at odds with many aspects of DQN (e.g., replay buffers, off-policy bootstrapping without importance sampling, etc.).

* The title and tone of the paper gives the impression that in addition to challenging an idea that TD is uniquely capable of stitching, the other factors might be more important to the ability to stitch than the use of TD updates. e.g., by questioning whether it's the "gold standard", highlighting that the gap between small to large networks is larger than the gap between TD and MC, suggesting that TD is "less necessary", etc. However, the results still seem highly favorable for TD, where in all cases that MC exhibited stitching, TD stitched significantly better. The extent of the benefit from scale seemed highly variable depending on the algorithm choice, with the largest improvement through scale still being a TD method. It's unclear whether the title and tone are warranted, in light of this—can the authors comment on the choice to frame the exposition in this way?

### Soundness
2

### Presentation
3

### Contribution
3
