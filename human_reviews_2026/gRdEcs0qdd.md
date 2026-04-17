# Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
Scaling reinforcement learning to tens of thousands of parallel environments requires overcoming the limited exploration capacity of a single policy. Ensemble-based policy gradient methods, which employ multiple policies to collect diverse samples, have recently been proposed to promote exploration. However, merely broadening the exploration space does not always enhance learning capability, since excessive exploration can reduce exploration quality or compromise training stability. In this work, we theoretically analyze the impact of inter-policy diversity on learning efficiency in policy ensembles, and propose Coupled Policy Optimization which regulates diversity through KL constraints between policies. The proposed method enables effective exploration and outperforms strong baselines such as SAPG, PBT, and PPO across multiple tasks, including challenging dexterous manipulation, in terms of both sample efficiency and final performance. Furthermore, analysis of policy diversity and effective sample size during training reveals that follower policies naturally distribute around the leader, demonstrating the emergence of structured and efficient exploratory behavior. Our results indicate that diverse exploration under appropriate regulation is key to achieving stable and sample-efficient learning in ensemble policy gradient methods. Project page at https://naoki04.github.io/paper-cpo/ .

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper shows that in large-scale reinforcement learning, blindly pursuing policy diversity across multiple policies reduces sample efficiency and training stability. To address this, it proposes Coupled Policy Optimization (CPO): within a leader–follower framework it regulates diversity via a KL constraint and an adversarial intrinsic reward, and it uses importance sampling to efficiently absorb data from followers. Experiments show that CPO achieves higher sample efficiency and more stable, superior final performance than existing baselines on a range of dexterous manipulation tasks.

### Strengths
1. This paper proposes CPO, which within a leader-follower framework regulates the exploration radius and coverage via a KL constraint to the leader policy coupled with an adversarial intrinsic reward, while retaining efficient use of off-policy data.
2. This paper uses upper-bound analysis to connect the leader-follower KL with IS ratio deviation, ESS, and gradient bias, and, drawing on Pinsker-type inequalities and PPO clipping analysis, offers a quantitative justification for why distance control is needed.
3. On dexterous manipulation tasks under ultra-large parallel settings, CPO shows higher sample efficiency and more stable, superior final performance, supported by ablations and visualizations.

### Weaknesses
1. CPO essentially adds a KL constraint to the leader and an adversarial intrinsic reward on top of a leader-follower framework. The novelty is incremental.
2. The discriminator and KL regularization introduce additional compute overhead and hyperparameters, yet the paper lacks wall-clock comparisons under equal compute and equal interaction budgets, as well as a systematic hyperparameter sensitivity analysis.
3. The experimental evidence focuses mainly on dexterous manipulation with a single platform/parallelism setting, and the breadth of tasks and difficulty axes is insufficient to support conclusions.

### Questions
See Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, the authors theoretically analyze the impact of inter-policy diversity on learning efficiency in policy ensembles, and propose Coupled Policy Optimization (CPO), a method that regulates diversity via KL constraints between policies. The proposed CPO enables effective exploration and outperforms strong baselines (including SAPG, PBT, and PPO) across multiple dexterous manipulation tasks, showing advantages in both sample efficiency and final performance. Additionally, the experimental results are robust, and the paper is fluently written.

### Strengths
1. The paper is well-structured and clearly written, ensuring good readability.

2. The authors rethink the existing SAPG method by providing theoretical analysis and key insights, which effectively motivate the design of the proposed CPO.

3. The experimental results are robust and supported by sufficient evidence, and the overall writing flow is smooth.

### Weaknesses
1. More concrete examples could be added to illustrate and validate the key theoretical insights, which would strengthen the persuasiveness of the work.

2. The design of the proposed CPO is relatively straightforward, as it only uses KL divergence to constrain the distance between follower and leader policies, lacking further optimization or innovative adjustments.

- 2.1 The selection of the lambda hyperparameter in practice is somewhat heuristic, with no clear justification provided for its choice.

- 2.2 KL divergence is a relatively trivial and commonly used distance metric. In the field of policy diversity research, numerous works have proposed alternative diversity metrics [1-3]. A more in-depth discussion of these existing metrics, along with the rationale for selecting KL divergence in this work, would improve the justification for the method’s design.

[1] Effective diversity in population based reinforcement learning. 

[2] Policy space diversity for non-transitive games.

[3] Quality-similar diversity via population based reinforcement learning.

### Questions
1. Are there any specific guidelines or recommendations for the selection of the lambda hyperparameter?

2. The paper adopts SAPG as its baseline method. Could the authors explain why SAPG outperforms Population-Based Training (at least in their experimental settings)?

3. Why was KL divergence chosen as the policy distance metric? Would using reverse KL divergence alter the conclusions of this work?

4. Why is the score of SAPG very low in Two-Arms Reorientation in Table 1, but the proposed method, which is based on SAPG, achieves the highest score?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of excessive policy diversity in ensemble policy gradient methods  by improving the policy optimization method for followers. Specifically, it introduces KL constraints during the follower updates to regulate the distance to the leader, and incorporates an adversarial reward to prevent policy overconcentration . The proposed method (CPO) achieves structured and efficient exploration. Experiments demonstrate that this method can effectively improve sample efficiency and final performance.

### Strengths
1. The paper's insight is presented very clearly, and the logic is rigorous; we can easily follow the author's train of thought and logic to understand the method. The paper's writing quality is high.

2. The theoretical derivations are thorough, and the experimental validation is comprehensive. The visualization of KL divergence changes in Figure 4 is very valuable.

3. The method achieves a significant breakthrough on a high-difficulty task (Two-Arms Reorientation).

### Weaknesses
1. The model's generalizability appears limited. It is only effective in specific environments, such as AllegroHand, where follower policies are prone to significant divergence (resulting in high variance). In contrast, on other tasks like Regrasping, the performance improvement is not as pronounced .

2. The training cost is somewhat high. The paper's CPO method requires more backpropagation components (roughly 12 vs. 7 for SAPG) and more wall-clock training time per iteration (approximately 25% more) .

3. Although the design of the Adversarial Reward is interesting—leveraging the idea from DIAYN to classify policies based on state-action pairs —the ablation study shows its impact on the results is not significant. The necessity of this module is therefore questionable .

### Questions
1. I wonder whether "misaligned significantly" phenomenon consistently occur in all "complex tasks". Could you provide more analysis on what types or characteristics of environments are prone to causing this "misaligned significantly" state? Can this be primarily attributed to the environment's exploration complexity? Could I regard the KL constraint (CPO's core mechanism) essentially act as a mechanism to reduce exploration and increase exploitation;

2. Are the marginal benefits of improving the IS and ESS metrics on the final performance limited? I observed that the ESS was optimized by over 40-fold (e.g., from 2.23% to 94.1% in ShadowHand ), yet the final sample efficiency (as shown in the reward curves ) only showed a limited improvement (approx. 2-2.5x).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles the challenge of scaling reinforcement learning (RL) to large numbers of parallel environments, where a single policy’s limited exploration capacity can hinder performance. The authors focus on ensemble-based policy gradient methods that use multiple policies to promote exploration but note that excessive diversity can harm learning stability and efficiency. They present Coupled Policy Optimization (CPO), a method that constrains inter-policy divergence via KL regularization to balance exploration and coordination among ensemble members. Theoretical analysis establishes how controlled diversity improves learning efficiency, and empirical evaluations on dexterous manipulation tasks show that CPO outperforms prior methods such as SAPG, PBT, and PPO in both sample efficiency and final performance. Additional analysis demonstrates that follower policies self-organize around a leader, yielding structured and effective exploration. Overall, the work highlights the importance of regulated diversity for stable and efficient learning in ensemble policy gradient frameworks.

### Strengths
* It appears to be a novel and difficult task to introduce exploration incentives to the follower agents at risk of destabilizing the already off-policy training, however this paper appears to utilize the KL divergence and the discriminator in a way that promotes some exploration for the follower agents without training collapsing.
 * The paper is well written, and utilizes the background and method section well. The results are well thought out.
* Although the main algorithmic contribution appears to be simple, the KL divergence, the addition is well justified and explored thoroughly in the paper.

### Weaknesses
* I would have liked to know why a discriminator was chosen specifically, in comparison to other exploration based algorithms, in particular there are other methods that do not require the additional external training or usage of a functional approximation [1]. 

* Going further on the second point, although the KL divergence effects were explained, section 5.2 appears to be the only discussion on the usage of an exploration algorithm, and it appears that there is little explanation on how the adversarial signal improves the performance of the algorithm. If this component of the paper was discussed or there was more time spent analyzing this more somewhere in the paper I would be more confident staying at an accept. 

[1] Susan Amin, Maziar Gomrokchi, Harsh Satĳa, Herke van Hoof, & Doina Precup. (2021). A Survey of Exploration Methods in Reinforcement Learning.

### Questions
* I would say that although DIAYN uses a discriminator, their work is primarily interested in gathering skills and maximizing entropy, would it be reasonable to introduce to the reader as (ICM)[2] which explicitly relies on prediction error? I don't consider this an issue, but there are other methods that leverage prediction error that might make a better fit. 

[2] Pathak, D., Agrawal, P., Efros, A.A., Darrell, T.: Curiosity-driven exploration by
self-supervised prediction. In: International Conference on Machine Learning, pp.
2778–2787 (2017). PMLR

### Soundness
3

### Presentation
3

### Contribution
3
