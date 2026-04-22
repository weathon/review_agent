# Posterior Behavioral Cloning: Pretraining BC Policies for Efficient RL Finetuning

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Standard practice across domains from robotics to language is to first pretrain a policy on a large-scale demonstration dataset, and then finetune this policy, typically with reinforcement learning (RL), in order to improve performance on deployment domains. This finetuning step has proved critical in achieving human or super-human performance, yet while much attention has been given to developing more effective finetuning algorithms, little attention has been given to ensuring the pretrained policy is an effective initialization for RL finetuning. In this work we seek to understand how the pretrained policy affects finetuning performance, and how to pretrain policies in order to ensure they are effective initializations for finetuning. We first show theoretically that, by training a policy to clone the demonstrator's \emph{posterior} distribution given the demonstration dataset---rather than simply the demonstrations themselves---we can obtain a policy that ensures coverage over the demonstrator's actions---a minimal condition for effective finetuning---without hurting the performance of the pretrained policy. Furthermore, we show that standard behavioral cloning (BC) pretraining fails to achieve this without significant tradeoffs in terms of sampling costs. Motivated by this, we then show that this approach is practically implementable with modern generative policies in robotic control domains, in particular diffusion policies, and leads to significantly improved finetuning performance on realistic robotic control benchmarks, as compared to standard behavioral cloning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Posterior Behavioral Cloning (POSTBC), a new pretraining framework designed to make behavioral cloning (BC) policies more effective initializations for reinforcement learning (RL) finetuning. The authors theoretically show that posterior BC, which learns to clone the posterior distribution of the demonstrator’s behavior rather than its empirical action distribution, yields better action coverage without increasing suboptimality. To make this practical in continuous control settings, the paper develops an implementation using generative diffusion policies. The posterior distribution is approximated by training an ensemble of perturbed BC models to estimate posterior variance, and then fitting a policy to a mixture of BC and posterior-generated samples.

### Strengths
1. The paper tackles a highly relevant problem — how to design better pretraining objectives so that behavioral cloning (BC) policies serve as more effective initializations for reinforcement learning (RL) finetuning. While most prior work focuses on improving the finetuning algorithms themselves, this work shifts attention to the quality of the initialization, a perspective that is both conceptually insightful and practically important for sample-efficient RL in robotics and other domains.
2. The theoretical analysis is rigorous, well-motivated, and clearly presented.

### Weaknesses
1. Limited related work discussion and insufficient comparative evaluation: The paper’s experimental comparison is somewhat incomplete. While the work focuses on improving BC pretraining for RL finetuning, it does not adequately compare against other imitation learning approaches that could also serve as initializations for RL finetuning, such as DWBC, ISWBC, the DICE-family. Further, the literature review is also insufficient. As far as I know, there exists offline-to-online imitation learning, which trains a good initialization for online imitation learning.
2. Unclear treatment of the value function during RL finetuning: Since POSTBC pretrains only the policy but not a corresponding value function, it remains unclear how the subsequent RL finetuning phase initializes or learns its critic. Directly reusing a random or uninitialized value function could cause policy–value misalignment, especially in actor–critic frameworks where stable updates rely on consistent policy–value coupling. This raises the question of whether the improved finetuning performance comes purely from the better policy initialization, or if it is affected by potential instability in the critic’s early-stage learning. Furthermore, one might ask whether it is possible to derive a value function estimate from the posterior BC process itself (like a byproduct from policy learning).

### Questions
1. How does the performance change with the number of trajectories?
2. How does it perform with diverse quality of data?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes posterior behavioral cloning, a pretraining approach for behavior cloning policies in order to make the finetuning processes efficient. The paper suggests a theoretical result that the standard behavior cloning often provably fails to cover the demonstrator’s distribution, while the proposed posterior behavior cloning effectively covers.

### Strengths
The paper has a theoretical contribution on how the proposed posterior behavior cloning could be provably better than the standard behavior cloning, in terms of the action coverage. Based on the findings, the paper proposes a simple instantiation of the posterior behavior cloning for continuous control settings.

### Weaknesses
While the paper explicitly states that “there do not exist any approaches which aim to pretrain policies with a BC-like objective on demonstration data, with the aim of obtaining an initialization that is an effective starting point of finetuning”, this is exactly what the typical meta-learning for supervised learning does, and behavior cloning is just an example of supervised learning (the outer loop of gradient-based meta-learning corresponds to the “posterior behavior cloning”). Even if we narrow our attention to “training policies on reward-free demonstration data”,  there still exists extensive literature about meta-imitation learning (e.g., [1, 2]) that meta-trains a policy using reward-free demonstrations by performing behavioral cloning. Therefore, I’m highly concerned about the reliability and novelty of the paper, and also about the lack of comparisons with these approaches both in theory and experiments.
    
[1] Finn et al., One-Shot Visual Imitation Learning via Meta-Learning, 2017.

[2] Gao et al., Transferring Hierarchical Structure with Dual Meta Imitation Learning, 2022.

### Questions
(continuing from the concerns in the weaknesses part) Behavior cloning is an example of supervised learning, and there exist several studies [3, 4] on the relationship between hierarchical Bayesian learning and meta-learning. Which aspects of the proposed setting (e.g., MDP, continuous control, etc.) make a difference from the typical meta-learning? Does the proposed theory directly address the difference?
    
[3] Grant et al., Recasting gradient-based meta-learning as hierarchical bayes, 2018.

[4] Yoon et al., Bayesian model-agnostic meta-learning, 2018.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors investigate the problem of suboptimal RL finetuning performance when using a policy initially trained via behaviour cloning. The authors provide several theoretical results showing the error bounds for achieving an optimal policy when insufficient coverage is available with the demonstration policy. An algorithm is proposed that uses stochastic labels to help prevent loss of policy coverage on actions while also maintaining initial behaviour-cloning policy performance. The authors perform experiments on Libero and Robomimic to validate their algorithmic suggestions that initial performance is maintained and that their posterior policy accounting approach enhances RL finetuning.

### Strengths
I like the emphasis on providing theoretical foundations to justify the model's algorithmic choices. Showing sample bounds and the effects on policy cumulative reward as a function of the number of actions, states, and timesteps provides better justification for the problem. It is more insightful than just proposing an algorithm. We like the author's solution because of its ease of implementation and potential applications in continuous control.

### Weaknesses
Technically, the paper has not violated the page limit, but it ends abruptly on page 9 without a clear conclusion. This ending indicates that more time is needed to condense the current draft to fit the page limit, without the additional page in the rebuttal.  Figures 2 - 5 are squished together as a byproduct of this. It isn't easy to see the content of Figure 2, and the legend in Figure 3 dominates one of the presented plots. Discussion across many sections can likely be compressed to address these issues and remove redundant discussions. Here are some ideas:
- If Algorithm 1 is not the main algorithm, it can probably be cut or moved to the appendix. 
- The Equation before Equation (3) also looks quite similar, or otherwise frankly could be inline with the text. This suggestion might also apply to the Gaussian policy on lines 314/315.
- Line 309 - 311 summarizing the previous section could be cut. 
- Proposition 1: Could perhaps be moved to the appendix, and the same for Mathematical Notations.

Furthermore, although we find the theoretical justifications compelling, the existing experiments at a minimum need more nuanced discussion. As one of the central claims is that BC can improve exploration during RL finetuning, it would be helpful to understand why the results in Figures 4 & 5 appear mixed in the benefits of the author's algorithm. One idea might be to modify the training data distributions to create conditions similar to those described in the theory section, or at least examine the action distributions directly to see how variance is affected across the learned policies. 

Likewise,  the claims from Table 1 should be more specific. The 20% over BC and 10% success rate improvements over \sigma-BC are specific to Libero. All results are from the Best-of-N (1000 Rollouts). In other settings, we see similar performance between BC and PostBC, with \sigma-BC performing better in one case (Liberto Scene 1). The results for Libero on the 2000 Rollouts setting seem to be missing, or if there is a reason they are excluded, that should be explained.  It would also be good to conduct hypothesis tests to determine whether these results are statistically significant, thereby strengthening the author's claim. 

Other Comments: 
> Consider including the work on "robust policy optimization," which proposes a similar stochastic labelling approach for the online RL setting [1]. The author's project better motivates this decision theoretically, but the point is to acknowledge works that study applying a similar mechanism. 
[1] Rahman, Md Masudur, and Yexiang Xue. "Robust policy optimization in deep reinforcement learning." arXiv preprint arXiv:2212.07536 (2022).

> The "Best-of-N" approach should be more clearly explained in the Background section. Several significant results rely on this, but how it is performed precisely is difficult to discern in the current paper. 

> Table 1 — if this shows success rates, make that clear. The bold font is confusing, so either explain it in the caption or remove it.

> Lemma 6 in Appendix — If this is a proof for "Theorem 2," then it shouldn't be called a lemma.

### Questions
Definition 4.1 — What is the range of \gamma? 

Proposition 2 - Is there some meaningful range implicitly accounted for in this bound? Is J(\pi^\beta) bounded between 0 and 1? 

Proposition 2 - If this is an informal argument, is a proof or related citation provided to support this proposition? The proposition appears central to the paper's arguments but is not treated rigorously. 

Table 1: Why are the results for Libero Best-of-N (2000 Rollouts) missing?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes **Posterior Behavioral Cloning (POSTBC)**, a pretraining method that aims to make behavioral cloning (BC) policies more suitable for reinforcement learning (RL) finetuning. Rather than directly imitating demonstrations as in standard BC, POSTBC models the **posterior distribution of the demonstrator’s actions**, enabling higher entropy (exploration) in uncertain regions and more deterministic behavior in data-rich states. The authors provide theoretical results showing that standard BC can fail to cover the demonstrator’s action space, while POSTBC ensures coverage without compromising imitation performance. Using diffusion policies, the method demonstrates improved **sample efficiency and finetuning effectiveness** on robotic control benchmarks (Robomimic, Libero), achieving consistent gains over standard BC without degrading pretrained policy performance.

### Strengths
- The idea of leveraging a posterior distribution over actions for better RL finetuning efficiency is conceptually novel and practically meaningful.
- Clearly identifies and addresses a core limitation of standard BC in the context of RL finetuning.
- Provides a well-developed theoretical framework with sound reasoning about action coverage and RL-finetuning potential.

### Weaknesses
- If I understand correctly, the authors perturb the training actions with Gaussian noise, train multiple policies on these perturbed datasets, and compute the covariance of their predicted actions to approximate a posterior distribution. This approach seems somewhat ad hoc. If the base models have sufficient capacity to memorize the data, the estimated covariance would simply reflect the injected noise, effectively collapsing back to the σ-BC baseline. In classical bootstrap ensemble methods, one typically resamples data subsets rather than adding noise—this would arguably provide a more principled estimate of epistemic uncertainty.
- The experimental evaluation, while supportive, feels limited in scope and lacks deeper analysis or ablation (e.g., number of demonstrations, impact of ensemble size, or posterior weight α).

### Questions
1. The Libero benchmark does not officially provide reward functions. How were rewards defined in your RL finetuning setup? Did you use binary success/failure rewards or custom rewards?
2. Line 400 mentions limiting the number of demonstrations. Please clarify the exact number used per task and include an ablation study on this parameter. Intuitively, the gain of PostBC should decrease as the number of demo increases.
3. Have you considered using *bootstrapped resampling* (i.e., different subsets of the demonstration data) instead of Gaussian perturbations for ensemble training? This might yield a more faithful posterior estimate.

Overall I like the problems studied by this paper, I'd love to adjust my rating if the authors can settle my concerns.

### Soundness
3

### Presentation
3

### Contribution
3
