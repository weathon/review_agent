# BaNEL: Exploration Posteriors for Generative Modeling Using Only Negative Rewards

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Today's generative models thrive with large amounts of supervised data and informative reward functions characterizing the quality of the generation. They work under the assumptions that the supervised data provides knowledge to pre-train the model, and the reward function provides dense information about how to further improve the generation quality and correctness. However, in the hardest instances of important problems, two problems arise: (1) the base generative model attains a near-zero reward signal, and (2) calls to the reward oracle are expensive. This setting poses a fundamentally different learning challenge than standard reward-based post-training. To address this, we propose BaNEL (Bayesian Negative Evidence Learning), an algorithm that post-trains the model using failed attempts only, while minimizing the number of reward evaluations (NREs). Our method is based on the idea that the problem of learning regularities underlying failures can be cast as another, in-loop generative modeling problem. We then leverage this model to assess whether new data resembles previously seen failures and steer the generation away from them. We show that our method can improve model performance without observing a single successful sample on several sparse-reward tasks, outperforming existing novelty-bonus approaches by up to several orders of magnitude in success rate, while using fewer reward evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Bayesian Negative Evidence Learning (BaNEL), which is an algorithm for post-training generative models using only negative rewards (failed attempts). BaNEL minimizes the number of reward evaluations (NREs) and learns the structure underlying failures. The proposed approach learns efficiently from a small number of failures, by allowing for multiple parameter updates for each collected experience. BaNEL uses Bayesian updates of the prior, which does not decrease the model’s likelihood for failed attempts, and thus better preserves the model’s pre-trained knowledge. The authors provide an experimental evaluation that shows BaNEL can achieve a success rate that is several orders of magnitude higher than other competing approaches for the same NRE budget.

### Strengths
* The paper provides a good overview of existing algorithms for post-training generative models from reward functions, including policy gradient, sparse RL techniques, and GFlowNet.
* The contributions appear to be novel and technically sound.
* The experimental results are reasonably thorough and show that compared to competing approaches, BaNEL indeed achieves substantial improvements in success rates for a fixed NRE budget. The experiments also show that BaNEL can improve its success rate using additional compute.

### Weaknesses
* The primary evaluation metric in the experimental results is success rate, with only a brief qualitative example of the quality and diversity of generated samples, as shown in Figure 2. A more extensive evaluation in terms of the quality and diversity of generated samples should be provided.
* No results on the computational runtime complexity are provided. How much more computationally expensive is BaNEL compared to existing methods for post-training generative models from reward functions?
* BaNEL systematically narrows the search space. However, exclusively learning from negative samples could lead the model to avoid entire regions of the solution space that might contain the optimal solution, because initial explorations in that area failed. This could result in the model converging to a suboptimal solution. Is this an issue in practice? If so, an analysis of this issue should be included in the paper.

### Questions
The authors should provide answers to the questions/issues listed under “weaknesses” above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BaNEL (Bayesian Negative Evidence Learning), a new post-training algorithm for generative models that operates entirely from negative (zero-reward) samples under conditions of extreme reward sparsity and expensive reward evaluations.
Instead of relying on positive reward signals or dense feedback, BaNEL trains an auxiliary generative model on failed attempts to estimate a “failure distribution.” It then forms a Bayesian posterior steering generation away from samples that resemble prior failures.
The method supports multiple updates per reward evaluation and integrates a sequential filtering–distillation scheme to progressively refine the policy while keeping computation efficient.
Experiments on MNIST 0→6 generation and GSM8K-Hard reasoning tasks show that BaNEL improves success rates by several orders of magnitude compared to Random Network Distillation (RND) and count-based exploration, despite using the same limited number of reward evaluations.

### Strengths
- The research question is important because, as large language models continue to advance rapidly, many challenging tasks appear and remain unsolved due to sparse or extremely limited reward signals. Understanding how to effectively train models under such sparse-reward conditions is therefore a crucial and timely problem.

### Weaknesses
- Some part of experiments need further clarification. For example, in MNIST, how success rate of 8e-26 are calculated. From figure 2, there seems no large difference between prior and posterior, which tend to show 5e-21 is just a variance.
- The experiments are limited, especially on GSM8k hard, only a toy setting on 6 subquestions are tested.  A more realistic setting (i.e. a model trained a difficult training set, say several hundred queries and tested on commonly used benchmarks) are expected to verify the effectiveness

### Questions
1. This paper discuss a situation that models are unable to receive reward (i.e. very low percent of trajectories can get correct answer), and use negative samples to learn. I am wondering whether this method can be integrated with the setting that we can receive reward (i.e. make the learning more efficient as we can learn from both correct samples and incorrect samples with additional information like BAYESIAN NEGATIVE EVIDENCE). As realistic setting, people can warm up the model with SFT trajectories or use reward model to somehow bypass the zero reward issue even on very complex tasks.
2. achieving 0.53 mean@5 which means it has around 10 successful trajectories for 100 samples, why the success rate is 1e-4?

### Soundness
2

### Presentation
2

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
This paper presents Bayesian Negative Evidence Learning (BaNEL), an algorithm designed to enhance the query efficiency of post-training where positive reward signals are extremely sparse and reward query is costly. In essence, BaNEL trains an auxiliary density model to fit the distribution of past failed behaviors, and subsequently leverages the density predicted by this model as the criterion for rejection sampling. Specifically, BaNEL filters out samples whose density exceeds some threshold and does not query the reward oracle for these samples, thereby reducing the volume of unnecessary queries to the reward oracle. The performance of BaNEL was empirically validated on a task specifically constructed using data from MNIST and GSM8K. The proposed method demonstrated favorable results in comparison to several established baseline algorithms, including RND and count-based methods.

### Strengths
The development of the idea is clear and easy to follow. 

The problem addressed by this paper is important. In scenarios where varifiable rewards are not available or obtaining rewards are costly, we need a mechanism to filter samples with high potential for information gain.

### Weaknesses
1. The proposed methodology remains a **passive** strategy. While it effectively reduces the number of reward queries by filtering out samples with a high affinity to past failures, it offers no inherent benefit in terms of true sample efficiency. This is because the underlying generative model continues to perform uninformed exploration across the entire sample space. Ideally, an active learning approach should be developed that explicitly models the distribution of failed attempts and actively guides the model away from failure regions.

2. Several claims made within the paper appear **unfounded and conceptually ambiguous**. For example, the authors mentioned that an ideal method should be able to scale with compute (line 51). If this implies that the ability to identify similarities should improve with increased computational resources, it's unclear why methods like RND and count-based exploration are cited as failing to scale with compute (lines 172 and 182). One could easily argue that the RND network, for example, can be updated with multiple epochs to achieve similar scaling. (In fact, I would say RND and BaNEL are conceptually similar in the sense that they are all training some scoring function to identify seen failures, except that BaNEL is using the density from some generative model.)

3. The empirical evaluations appear **insufficiently justified**, making the advantage of the proposed method difficult to comprehend. For example, the success rates in Section 5.1 are truly low, such that I am skeptical whether the benefit exists or not. Figure 3 only presents the averaged results over five seeds without confidence intervals or error bars, which makes it impossible for readers to examine the statistical significance of the scaling trend. Furthermore, the scaling trend appears noisy, and the x-axis pacing is uneven. In Figure 10, it seems BaNEL’s success rate always peaks during early training and then decreases. This is counterintuitive to me, since as the training progresses, the fitted density model should have seen more failure cases and therefore the success rate should instead be higher. Collectively, the presented evidence is insufficient to demonstrate the efficacy of BaNEL.

### Questions
1. Is there a specific rationale behind using the ratio p_\theta(x) / p_\phi(x)^\kappa, instead of simply 1 / p_\phi(x)^\kappa? The factor p_\theta(x) seems to downweight the rare samples from the current model. 

2. line 370: How did you calculate the exact success rate of generating the digit 6 in Section 5.1? Also what does it mean by using torch.logsumexp to evaluate the exact success rate? 

3. Given that the primary objective is to reduce the number of reward queries, a crucial missing baseline is the comparison against methods that train a proxy model or utilize a judge model for reward evaluation. What specific advantages does BaNEL offer over these well-established proxy-model approaches?

### Soundness
1

### Presentation
3

### Contribution
2
