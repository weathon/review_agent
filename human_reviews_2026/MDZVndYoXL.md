# Sliced Distributional Reinforcement Learning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 2

## Abstract
Distributional reinforcement learning (DRL) models full return distributions rather than expectations, but extending to multivariate settings can be challenging. Univariate tractability is lost, and multivariate approaches are either computationally expensive or lack contraction guarantees. We propose Sliced Distributional Reinforcement Learning (SDRL) which lifts the tractable one-dimensional divergences to the multivariate case through random projections and aggregation. We prove Bellman contraction under uniform slicing for shared scalar discounts and under max slicing for general anisotropic matrix-discount updates, providing the first contraction result in this setting. SDRL accommodates a broad class of base divergences, instantiated here with Wasserstein, Cramér and Maximum Mean Discrepancy (MMD). In experiments, SDRL achieves competitive results on multivariate control tasks in MO-Gymnasium. As an application of matrix discounting, we extend multi-horizon RL with hyperbolic scalarization to the distributional regime. Taken together, these findings position slicing as a principled and scalable foundation for multivariate distributional reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a novel and principled framework for tackling the challenging problem of Multivariate Distributional Reinforcement Learning (MDRL). The core innovation is the application of Sliced Probability Divergences (SPD)  to compare deal with high-dimensional return distributions efficiently. The authors provide strong theoretical foundations, including the contraction guarantees for general matrix-discounted Bellman operators, and back their claims with empirical results on MuJoCo benchmarks.

### Strengths
The paper’s primary strength lies in its theoretical results. The authors demonstrate convergence properties and provide finite sample error bounds. In particular, the introduction of uniform slicing is shown to effectively mitigate the curse of dimensionality.

### Weaknesses
The paper lacks a formal definition of some notations, which could hinder readers' understanding. Specifically, the notation $\mathbb{S}$ is used without a clear definition , and the term $A_{s,a}$ in Theorem 4 is not defined. Additionally, in Theorem 4, if the authors consider the scenario where the range of $C$ (this term depends on the reward?) is unbounded. It would be beneficial to clarify these notations and explicitly discuss the unbounded reward case.

While the paper introduces a technically sound framework, the motivation behind using distributional reinforcement learning (RL) for multi-reward settings could be presented more convincingly. The practical significance of the approach is not fully articulated.

I would be willing to raise my score if the authors address my confusion regarding these points.

### Questions
1. Does the proposed framework account for the influence of correlated reward functions? If so, how is this incorporated into the framework?

2. The authors mention that the implementation of Maximum Mean Discrepancy (MMD) focuses on a single kernel. However, as far as I know, in distributional RL, as in [1], multiple bandwidths are often used, as a single bandwidth can degrade performance. Additionally, kernel methods are known to be sensitive to hyperparameter choices. Could the authors comment on these?

3. If the authors consider sliced Wasserstein distance or Cramer distance. I know there are rarely related work deal with multi-objective using these two metrics. if the authors have any comments?

4. The architecture of $Z_{\phi}$ is introduced without much detail. Considering that it is a generative model, did the authors consider using particular generative models, such as diffusion models or GANs? Further elaboration on this would be useful.

5. The authors employ a scalarization rule that computes a weighted average of the multiple rewards. Is this rule specific to the MO-MuJoCo environments, or could it be generalized? I note that in the standard MuJoCo setup, the reward is a linear combination of multiple reward components. 

6. The paper introduces a distributional version of the multi-horizon Bellman equation, which seems to be a natural extension of the expected multi-horizon Bellman equation. However, I’m curious about how the framework handles the case where two rewards are highly correlated (either positively or negatively). Does the proposed equation still hold in such cases? Does it have convergence properties, particularly regarding fixed points?

   

   [1] Distributional reinforcement learning via moment matching.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the multivariate distributional RL setting. The authors proposed a new algorithm based on the sliced probability divergence, which can extend multiple common univariate distributional RL algorithms into a multivariate version within one framework. Theoretical results are provided with a focus on the Bellman contraction with a discussion of related properties. Experiments are conducted on several Mujoco environments with some simple baselines.

### Strengths
1. The proposed method seems novel and technically sound, which is established by multiple previous works on sliced probability divergence.

2. The theoretical framework seems to be unified, which includes plenty of common divergences as its instantiations. The analysis, from a general metric property to contraction, is sound.

### Weaknesses
There are multiple apparent weaknesses in this work.

1. **Limited and less clear motivation**. To the best of my knowledge, the multivariate distributional RL has already been explored by many prior works in the literature, such as MMD variant in Zhang et al. (2021); Sinkhorn divergence in Sun et al. (2024), and Wiltzer et al (2024), where the proposed algorithms have already achieved satisfactory performance in large-scale environments under relatively clear foundations. Since the main motivation of this paper is in this setting, it would be important to clearly articulate the advantages over existing algorithms and the differences from them, which thus helps the readers to posit the contribution of this paper. It seems that the current motivation is mainly explained in Line 124, which is relatively weak and less unified. This too specific direction may not gain broad interest in the community.

2. **Explanation of theoretical results is insufficient**. Although the proof seems comprehensive, which involves many aspects, I have a difficult time understanding the interpretation of the given theoretical results. For instance, the plain sliced method uses random projection, which is more likely to be inefficient. This has not been explained by any theoretical results. In particular, it lacks an explanation of Theorems 6 and 7. Can we find any advantages of the proposed method over other baselines from the perspective of sample complexity? What does $S_s$ in Line 296 mean? It lacks explanation. What is the insight to understand the non-dimension-dependent penalty? That seems counterintuitive, as the dependence must have some role to play. If not, why? Can the contraction plain sup-sliced in Theorem 3 be extended to the general $\Gamma_t$? What is $A_{s, a}(C)$ in Theorem 4, which may not be defined?


3. **Experiments are very limited**. Most of the published papers that develop distributional RL algorithms conduct experiments on Atari games, but this paper only provides limited results in several MuJoCo environments. It is well-known that the demonstration in the actor-critic framework tends to be less clear [1] compared with the pure value-based RL in environments with a discrete action space, such as Atari. Prior works such as Zhang et al. (2021) and Sun et al. (2024) have already provided results on several Atari games, making this paper’s experiment very limited. In addition, there lack of considered baselines. 5 random sees are insufficient. It is hard to interpret the entangled curves in Figure 1.


4. **Unfinished and unclear writing**.  (1) I found this submission is unfinished with an empty Appendix E.3, which largely affects my impression of this work. (2) Connections with other areas are discussed in a less sufficient way, which easily confuses. For instance, the authors try to connect the proposed method with GVF, but GVF is general in the sense that it generalizes the reward and allows a discounting to be a function of s_t and a_t. However, the proposed algorithms mainly consider a discounting function of time t without focusing on the prediction function beyond the rewards. Therefore, this connection is relatively weak and likely to confuse readers. (3) How do we understand control with scalarization in Line 237? Typically, we just use the Bellman optimal operator with a max operation in the control. (4) It is less clear what the computation cost would be when we use the max-slice mentioned in Line 203. Readers expect sufficient explanation in the main content instead of looking into details in the appendix. (5) What is the joint vector of multivariate returns in Line 210? Can it be explained mathematically? (6) What does the ‘true’ multi-objective control mean? 

[1] Addressing Function Approximation Error in Actor-Critic Methods (ICML 2018)

### Questions
1. Max-sliced algorithms enjoy better theoretical properties at the cost of heavier computation. Does the experiment reflect this conclusion? If the practical improvement is limited, practitioners, in particular, are likely to quickly lose interest in this variant.

2. What would be the practice choice for $\Gamma_t$ in the proposed multivariate distributional RL setting? I understand the generality of such a formulation, but I suspect whether the setting is practical. For example, are there many papers that are in this setting with some choices of $\Gamma_t$ beyond the papers provided in Line 105?

3. The single choice of kernel in MMD seems questionable. Does the contraction of the proposed method hold for general kernels in MMD? If not, is there a counter-example available? If so, why?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed Sliced Distributional Reinforcement Learning (SDRL), an algorithm framework for distributional RL which uses (max or L^p) sliced probability divergence as measure of error. They showed that the distributional Bellman operator is a contraction under L^p slicing for shared scalar discounts, and under max slicing for general matrix discounts. They also analyzed the sample complexity of estimating uniform and max slicing using Monte-Carlo methods. They provided numerical experiments to demonstrate the efficiency of the proposed SDRL.

### Strengths
The paper is well-written, clearly demonstrating the problem of multivariate distributional RL and how to effectively learn the joint distribution of return vectors by minimizing (max or L^p) sliced probability divergence. It provides clear contraction analysis of their algorithms and presents some experimental evidence.

### Weaknesses
1.Compared to Wiltzer et al. (2024a), this paper does not consider the finite-dimensional representation of probability distributions and only focuses on the simplified non-parametric setting, while this aligns with the experimental setup. Given that the work is not theoretically oriented, this weakness is acceptable .

2.As a paper focusing on distributional RL, the experiments in the main text do not demonstrate the effectiveness of the SDRL algorithm in fitting the joint distribution of returns. In my view, it is necessary to include experiments that showcase this aspect.

## Minor Points:

Use \eqref instead of \ref for references of equations.

In line 36 and 46, the notation n is used without explain. 

In line 77-78, the sentence ‘Given a policy π(a|s), the agent seeks to maximize the expected discounted return’ seems confusing because Q is a vector and the policy \pi is given here.

In Equation (3) and (15), T^\pi should depend on time t.

In line 159 and 186, the notation L is used without explain.

In line 219, L represents projection count, but in line 360, L is a CDF-dominance constant, please avoid using the same notation.

In line 294-296, do not use s, which also represents state in this paper. 

In line 342-343, the notation A_{s, a} is used without explain.

Theorem 6 uses \hat{\mu}_n and \mu, but Theorem 7 uses P_n and P, please use consistent notations.

### Questions
See weakness 2

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
3

### Summary
This paper introduces Sliced Distributional Reinforcement Learning (SDRL) — a new framework for distributional RL in multivariate settings based on sliced probability divergences (SPDs). Existing multivariate distributional RL methods show non-contractive and have high-dimensional optimal transport. To bridge this gap, SDRL uses sliced versions of 1D divergences, achieving both theoretical soundness and computational scalability. Furthermore, the paper extends this framework to a Maximum-Sliced variant (MSDRL), which optimizes over projection directions rather than averaging them, enabling provable contraction even under general matrix-discounted Bellman operators.

### Strengths
+ Defines Sliced Probability Divergences (SPDs) by projecting high-dimensional distributions into 1D directions and averaging 1D divergences, then introduce the SDRL. 
+ Extends SDRL to maximum slicing, optimizing over directions instead of sampling and introduce MSDRL.
+ For both the SDRL and MSDRL frameworks, the paper provides theoretical guarantees for the contraction of the Bellman operator, proves that the proposed divergences preserve metric properties, and presents a detailed sample complexity analysis.
+ Reduces high-dimensional divergence computation to multiple 1D projections and improve the sample efficiency for multivariate Distributional Reinforcement Learning.

### Weaknesses
+ While SDRL generalizes across different divergence measures, its performance depends heavily on the choice of base metric. In addition, using random projection directions introduces sampling variance into the loss, causing slight fluctuations in the estimated divergences across batches.

+ The optimization over projection directions in MSDRL (via gradient ascent) increases computational cost, which may reduce its practical appeal compared to simpler variants.

+ Although the slicing technique is statistically consistent, it remains an approximation in practice since it is infeasible to integrate over infinitely many directions. Consequently, there is no guarantee of fully preserving high-dimensional structure after slicing. Each one-dimensional projection captures only linear combinations of variables along specific directions, potentially missing nonlinear or higher-order dependencies between dimensions.

+ The paper provides extensive theoretical analysis, but the experimental results do not demonstrate equally strong improvements. The empirical gains are modest — SDRL and MSDRL often match but rarely surpass MMD-based baselines. Moreover, the max-sliced variants, while theoretically superior, do not exhibit clear empirical advantages.

+ All experiments are conducted on standard MuJoCo control tasks, lacking evaluation on large-scale or complex high-dimensional domains. The work also does not address true multi-objective control, where learning must generalize across multiple scalarization rules.

+ The notation in the paper could be clearer; several symbols and terms are referenced before being properly defined, which may hinder readability.

### Questions
1.	The contraction proofs rely on several strong regularity assumptions about the base divergence, such as translation invariance and scaling Lipschitz continuity. However, these assumptions may not hold in complex multivariate distributional reinforcement learning settings. It remains unclear how the theoretical guarantees would extend—or whether they would still hold—when these conditions are violated.
2.	Are the theoretical results holds for all divergences or kernels (especially in high-dimensional MMD)?

### Soundness
3

### Presentation
2

### Contribution
2
