# Sail into the Headwind: Alignment via Robust Rewards and Dynamic Labels against Reward Hacking

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 5

## Abstract
Aligning AI systems with human preferences typically suffers from the infamous *reward hacking* problem, where optimization of an imperfect reward model leads to undesired behaviors. In this paper, we investigate reward hacking in offline preference optimization, which aims to improve an initial model using a preference dataset. We identify two types of reward hacking stemming from statistical fluctuations in the dataset: Type I Reward Hacking due to subpar choices appearing more favorable, and Type II Reward Hacking due to decent choices appearing less desirable. We prove that many (mainstream or theoretical) preference optimization methods suffer from both types of reward hacking. To mitigate Type I Reward Hacking, we propose POWER, a new preference optimization method that combines Guiasu's weighted entropy with a robust reward maximization objective. POWER enjoys finite-sample guarantees under general function approximation, competing with the best covered policy in the data. To mitigate Type II Reward Hacking, we analyze the learning dynamics of preference optimization and develop a novel technique that dynamically updates preference labels toward certain "stationary labels", resulting in diminishing gradients for untrustworthy samples. Empirically, POWER with dynamic labels (DL) consistently outperforms state-of-the-art methods on alignment benchmarks, achieving improvements of up to **13.0** points on AlpacaEval 2 and **11.5** points on Arena-Hard over DPO, while also improving or maintaining performance on downstream tasks such as mathematical reasoning. Strong theoretical guarantees and empirical results demonstrate the promise of POWER-DL in mitigating reward hacking.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposed two types of reward hacking in offline preference optimization . The Type I is when subpar choices appear more favorable due to sparse data, and the Type II is when good choices appear worse. They propose POWER(-DL), a new offline preference optimization method that (1) integrates weighted entropy with robust reward maximization to address Type I hacking and (2) use dynamically updating preference labels to resolve Type II hacking. Empirical results show that it outperforms state-of-the-art methods on benchmarks.

### Strengths
The use of weighted entropy and dynamic label is interesting and new as far as I am aware.

The reasoning is sound and convincing. For instance, for the first type of reward hacking, they established a lower bound (proposition 1) showing that regardless of the number of samples, certain existing algorithms will suffer from a constant suboptimality with probability of at least some constant. Then they show that their proposed algorithm avoids this issue (Theorem 1).

The experimental results are impressive. The proposed method consistently outperforms baselines across multiple models.

### Weaknesses
Both types of reward hacking issues arise from insufficient data coverage, a well-explored topic in bandits and reinforcement learning. Therefore, to me, the two new definitions of reward hacking don’t feel entirely novel to me. Although I admit that most prior work hasn’t focused on learning-from-preference settings. This consideration actually leads to some questions that I show in the Question section.


It might be beneficial to merge Theorem 2 with Theorem 1 to provide a unified theoretical result for the proposed algorithm. Currently, they are presented separately, so I wonder how the use of dynamic labels will affect the suboptimality bound shown in Theorem 1.

### Questions
Compared to Zhan et al., 2023a, what is the advantage of your method? I think the settings are pretty similar and the concentrability coefficient is defined in the same way (please correct me if I missed any subtitles). Since you claim to address both reward hacking issues by proving a finite-sample bound, as in Theorem 1, do Zhan et al. (2023a) address them as well?

If I simply set $w(\cdot)=0$, what would go wrong in Theorem 1? I believe that you have hidden some $w$-dependence term in the bound, and I am curious to know what that is.

Can you provide some intuitive explanations on why choosing $w(y)=1/|y|$ leads to $\log|\mathcal V|$ in Theorem 1?

Minor problem: I don't see how "Sail into the Headwind" is informative in the title

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces a new approach to address two forms of reward hacking due to low-coverage pairs. First, they propose a robust reward maximization objective, which enables a regret guarantee that all(?) previous methods lack. Second, they utilize a gradient freeze technique to prevent overreliance on low-coverage pairs.

### Strengths
I believe it is a strong paper with both convincing theoretical analysis and great empirical results.

### Weaknesses
Overall I think it's a very nice paper. But I do have some conceptual problems awaiting to be clarified by the authors: 
1. I am not sure the two components of your method are defending against each hacking type separately ( based on your paper construction, you are designing the approach this way).
It seems that dynamic labelling simply gives low trust to low-coverage regions. Then it should help both stop the less-desired outcomes from being pushed to a highly-desired position; AND stop highly-desired outcomes from being pushed to a lowly-desired position where the pair has low coverage.  Is that correct? Then should it be a technique for both hacking types?
2. It is nice that you have theoretical guarantee for POWER itself. And empirically, POWER-DL seems to work even better. But I feel the two moves could also interfere with each other due to the fact that highly-covered pairs will still influence the poorly-covered pairs and ultimately many outcomes are competing against each other --- hence even though your set the gradient to zero for poorly covered pairs, the final distribution of the actions can still be changed by other pairs. What are your thoughts on this?
3. Ps. typo on line 092: Should be POWER-DL right?

### Questions
1. Could you pls comment on weakness 1?
2. Could you pls comment on weakness 2?
3. What is the main reason you introduced the $\textbf{weighted}$ entropy term? Did you have other weight in mind other than the inverse response length?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses reward hacking in offline preference optimization (PO) methods by identifying two types of statistical errors that impact reward reliability: 
Type 1: Poorly represented, suboptimal choices in the dataset appear more favorable than they are due to statistical inaccuracies. 
Type 2: Due to poor statistical coverage, good choices are misrepresented as less favorable than their actual value. 
The authors demonstrate that many current offline PO methods are vulnerable to both forms of reward hacking, leading to suboptimal policy learning. They propose a novel offline PO method named POWER-DL to counteract these effects. This method integrates robust reward maximization with weighted entropy and dynamic preference labeling. Experimental results show that POWER-DL outperforms state-of-the-art methods on benchmark alignment tasks, such as Arena-Hard and AlpacaEval 2.0.

### Strengths
The paper is overall well-written. The related work is covered comprehensively, and theoretical statements are clearly articulated (although I haven’t verified the proofs in the appendix).

The theoretical analysis, particularly Theorem 1, helps guide and justify the design of POWER-DL. 

The proposed method, POWER-DL, is both theoretically grounded and easy to implement. Its approach—combining robust reward maximization with dynamic label updating—has practical implications.

The authors present extensive empirical results, with broad baseline coverage, that effectively demonstrate POWER-DL's advantages in addressing reward hacking and achieving better alignment.

### Weaknesses
Wu et al. (2024) introduce a dynamic beta-based approach that could be a relevant baseline here, especially given its emphasis on preference data quality. 

It would be helpful to see experiments on the robustness of POWER-DL to hyperparameters, particularly \beta and \eta. 

(Wu et al., 2024) \beta-DPO: Direct Preference Optimization with Dynamic \beta

### Questions
In Proposition 4 (I), POWER-DL is shown to recover the best-in-class policy for the constructive example given in Proposition 1. Does this imply that POWER-DL fully mitigates type 1 reward hacking? Or could there still be instances where type 1 reward hacking persists?

In Table 2, POWER-DL seems to perform better in the base setup (where it uses an existing preference dataset) than in the instruct setup (where a preference dataset is constructed by sampling from the initial model). Does this suggest that POWER-DL is more effective under a distribution shift between the initial model and the model(s) used to collect preference data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the preference optimization problem for LLMs. This work points out the two types of reward hacking that could lead to inaccurate learned reward models. To solve the problem, the proposed algorithm makes use of the weighted entropy for robust estimation. Also, a dynamic labeling procedure has been illustrated in order to diminish gradients for untrustworthy samples. The theoretical and empirical studies are performed to validate the conjectures and benchmark the performance of the algorithm.

### Strengths
- The presentation of the paper is clear, which makes it easy to read. 
-  The paper investigates one of the most important problems in preference optimization, i.e., reward hacking.
-  Some theoretical studies are conducted for analyzing the popular preference optimization, under the proposed conjectures. 
-  The algorithms are evaluated in some benchmarks and downstream tasks.

### Weaknesses
- There are many important and highly related literature that have not been reviewed in the paper. For example, 
a. Chen, Zixiang, et al. "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models." ICML 2024
b. Rame, Alexandre, et al. "WARM: On the Benefits of Weight Averaged Reward Models." ICML 2024
c. Pan, Alexander, Kush Bhatia, and Jacob Steinhardt. "The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models."  ICLR 2022
The insufficient review of the related literature on reward-hacking and model alignment limits the credibility of the two types of reward-hacking in the paper. 

-  As claimed in the paper, the type 2 reward-hacking is due to the decent choices appearing less desirable. It is reasonable to think about that if the over-pessimism in solving the type 1 reward-hacking is the root cause of not well-leveraging the decent samples. However, in this work, there is not a sufficient discussion on this point. See, Chen, Lichang, et al. "ODIN: Disentangled Reward Mitigates Hacking in RLHF." ICML 2024

- The examples used in propositions 1 and 2 are very simple, which deviated from the real applications in the fine-tuning of the LLMs, e.g., the definition of the reward function and policy classes in the examples. Therefore, from the results of propositions 1 and 2, it is only sufficient to obtain that the discussed popular preference optimization algorithms do not work well in such restrictive settings. However, the conclusion cannot be extended to the general settings in practice. 

-  The principle of maximum entropy (principle of the weighted entropy in the paper) has been widely studied in the RL literature and this is not a new idea. In particular, the work by Zhu, Banghua, Michael Jordan, and Jiantao Jiao. "Principled reinforcement learning with human feedback from pairwise or k-wise comparisons."  ICML 2023 has theoretically studied the maximum entropy in RLHF/IRL settings. 

-  As shown in Huang, Audrey, et al. "Correcting the mythos of kl-regularization: Direct alignment without overparameterization via chi-squared preference optimization." Arxiv, the KL divergence is not enough to induce pessimism in order to mitigate the reward-hacking. Following this argument, the weighted entropy may also encounter similar problems. 

-  The choice of the weights in the weighted entropy needs more details to elaborate; otherwise, the length-normalized technique has been utilized in the existing RLHF algorithms. 

-  Intuitively, the weighted entropy tends to yield a more uniform distributed policy. Therefore, this is a natural question about how to balance the exploration and exploitation tradeoff. 

-  A more pronounced discussion on ``favoring unbiased policies'' is needed. 

-  The \beta controls the strength of the explicit pessimism and \eta controls the strength of implicit pessimism. The two factors make the control over the pessimism complicated. In practice, how to balance the two types of pessimism, and in theory, how to mitigate over-pessimism? 

-  In resolving the type 2 reward-hacking problem, is there any theoretical guarantee/regret analysis on the algorithm with the dynamic labeling procedure? 

- There is no algorithm table for describing the details of the two proposed algorithms. 

-  In experiments, it would be better to present some results of the iterative extension of the current method. 

-  As the current method is much more complicated than the existing methods, e.g.,  DPO or SimPO, the latency of the algorithms should be presented. An efficient algorithm should balance performance and computational efficiency. 

-  The evaluation over the MT-bench is very important. Also, it would be better to compare the models in the instruction-tuned model Mistral as in SimPO. This could better position the work in the existing literature.

### Questions
Please see the questions in Weakness.

### Soundness
2

### Presentation
3

### Contribution
2
