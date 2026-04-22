# Importance Sampling Optimization Improves Online Preference Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Training large language models (LLMs) with online sampled data can help off-policy preference optimization approaches like DPO learn better. Recent methods such as Statistical Rejection Sampling Optimization (RSO) have emerged as attractive alternatives to online Reinforcement Learning from Human Feedback (RLHF), offering improvements in stability and scalability. Although RSO has shown promising results by using rejection sampling to obtain preference data from the estimated optimal target policy, it faces computational inefficiencies due to the high rejection rates inherent in its sampling process. To address these limitations, we introduce **Importance Sampling Optimization** (ISO), a novel approach that achieves the benefits of sampling from the optimal policy distribution while significantly improving sample efficiency.  ISO employs importance sampling to correct the distribution mismatch between the supervised fine-tuned (SFT) policy and the target optimal policy, enabling efficient use of all generated samples without rejection. Through extensive experiments across diverse tasks and models, we demonstrate that ISO achieves comparable or superior performance to RSO while requiring substantially fewer samples from the SFT policy. Reduces sampling overhead by up to 75\% while maintaining or improving win rates against both DPO and RSO baselines. Additionally, we show that ISO naturally extends to other preference optimization methods, providing a general framework for improving sample efficiency in preference learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper develops the importance sampling optimization (ISO) approach, to correct the mismatch between the SFT policy and the target optimal policy. ISO finds a way to use all generated samples without rejection, thus differs from the existing RSO (rejection sampling optimization) and DPO approaches, and strikes a balance between sampling efficiency and improved performance in preference learning.

### Strengths
Importance sampling is a well-established technique in stochastic optimization, including RL. Thus, ISO is built upon a solid theoretical foundation. The paper is well motivated (starting from Fig 1) and clearly written. The key is the pairwise importance weight in (7), modulated by the signed margin score in (8), which then (after normalization) goes into the loss function in (9), and then integrated into the preference learning pipeline. All these are clearly and logically spelled out in \S 3.

### Weaknesses
Cannot help feeling the paper's contribution falls a bit thin on technical novelty, given the well established status of importance sampling.

### Questions
At the end of \S4, there’s some description of the effect of \gamma in ISO, via (7). Wonder what’s choice of \beta in (10) in this case?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets the sample inefficiency of online preference learning methods like Statistical Rejection Sampling Optimization (RSO), which uses costly rejection sampling to align the sampling distribution ($\pi_\theta$) with the target optimal policy ($\pi^*$). The authors propose Importance Sampling Optimization (ISO), replacing rejection sampling with importance sampling to correct the distribution mismatch. ISO computes reward-based importance weights, allowing a DPO-style loss to utilize all generated samples efficiently. A heuristic "signed margin score" is added to potentially upweight informative pairs.

### Strengths
1. This paper leverages a importance sampling to solve the distribution mismatch problem and tackles the significant sample inefficiency and computational cost associated with RSO.
2. This paper demonstrates substantial reductions in the required number of sampled responses compared to RSO while maintaining or improving alignment performance. This is a major practical advantage.
3. Experiments are conducted across multiple model families, sizes, and standard alignment datasets, lending credibility to the results. The use of both proxy and independent golden reward models for evaluation adds robustness.

### Weaknesses
1. There is a critical misalignment between the theoretical setup and the algorithm's implementation regarding the proposal distribution for importance sampling. The loss function (Eq. 10) takes an expectation over samples drawn from $\pi_{sft}$, suggesting the importance weight $w(x, y_w, y_l)$ should correct for the ratio $\pi^*(y|x)/\pi_{sft}(y|x)$ (as derived in Appendix A.1). However, Algorithm 1 samples responses from the current policy $\pi_{\theta_t}$ (line 4). Applying weights derived assuming $\pi_{sft}$ to samples drawn from $\pi_{\theta_t}$ is incorrect and lacks clear justification. Furthermore, the notation $\mathbb{E}_ {(x,y_w,y_l)\sim \pi_{sft}}$ is imprecise, as $\pi_{sft}$ is a conditional distribution over $y$.
2. While positioned as an improvement for DPO-style methods (which are attractive for avoiding explicit reward models), ISO critically relies on an external, pre-trained reward model $r_\phi$ (Algorithm 1, line 5, Eq. 8) to compute the importance weights. This seems counter to the DPO philosophy and introduces a dependency not present in standard DPO. The paper does not clarify if this requires additional reward model training specific to the online setting. Using the DPO implicit reward $r(x,y) \propto \log (\pi_{\theta_t}(y|x)/\pi_{ref}(y|x))$ instead would likely be a poor proxy for the optimal reward $r^\star$ needed to estimate $\pi^*$, undermining the theoretical basis of the importance weights.
3. Importance sampling can suffer from high variance, particularly if the sampling distribution ($\pi_\theta$ or $\pi_{sft}$) is very different from the target distribution ($\pi^*$). While potentially better than RSO's rejection rate, the paper could discuss potential variance issues and how they are managed.
4. The signed margin score $(\sigma(r_w - r_l) - 0.5)$ is introduced somewhat heuristically to upweight informative pairs. While intuitive, a more formal justification or an ablation study isolating its specific impact on performance and variance would strengthen this component.

### Questions
Please see the weaknesses part above.

### Soundness
2

### Presentation
3

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
The authors propose to weight the SLiC loss by the magnitude of the reward scores.

### Strengths
N/A

### Weaknesses
I reviewed an earlier version of this paper where I raised several concerns. Unfortunately, it does not seem like any of them have been addressed. Thus, I am repeating them verbatim below:

(-) I really did try my best here but I don't think there's any reasonable interpretation of what the authors are doing as importance sampling. There is literally no ratio of two distribution's probabilities -- they never divide out the SFT policy's probabilities. I spent some time trying to do mental gymnastics to justify the product of factors that are used as weights as legitimate importance weights in any sense and I couldn't get that math to work out either. At best, I can say they weighted a usually unweighted loss.

(-) Off the top of my head, I think the most natural baseline here is https://arxiv.org/abs/2404.16767, which also essentially uses a weighted DPO-like loss. I would suggest including it in future experiments.

### Questions
(1) Is there a way to prove your re-weighting scheme is an unbiased estimate of importance weights $w(x, y) = \frac{\pi^{\star}(y|x)}{\pi_{sft}(y|x)}$?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new variant of online DPO method where an importance ratio term is introduced such that the update is performed under the optimal policy's generation distribution. The paper performs experiments that the ISO outperforms online DPO or rejection sampling.

### Strengths
1. The paper makes an interesting observation that, even though one can not sample from the optimal policy, one can still evaluate the optimal policy's density, thus enabling the importance ratio correction. 

2. According to the presented experiment results, ISO outperforms online DPO and iterative rejection sampling.

### Weaknesses
1. The method requires the access to the ground truth reward. With a reliable reward, one can simply perform online RL instead of contrastive learning. 

2. It is unclear the benefit of the importance sampling as it increases the variance of the estimator. 

3. The experiments are only performed for 2 iterations. 

4. The presentation of the paper seems unpolished, for example, line 225 is unfinished, and eq 3 should not be $\mathcal{L}_{\mathrm{DPO}}$.

5. The paper is confusing the optimal policy and the optimal KL regularized policy. In the importance ratio correction, the optimal KL regularized policy is used.

### Questions
How important is the modulating factor? This ablation seems missing from the experiments.

### Soundness
2

### Presentation
2

### Contribution
1
