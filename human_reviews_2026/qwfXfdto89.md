# RAPID: An Efficient Reinforcement Learning Algorithm for Small Language Models

- Decision: Reject
- Scores: 2, 0, 2, 2

## Abstract
Reinforcement learning (RL) has emerged as a promising strategy for finetuning small language models (SLMs) to solve targeted tasks such as math and coding. However, RL algorithms tend to be resource-intensive, taking a significant amount of time to train. We propose RAPID, a novel RL algorithm that can substantially reduce the running time of RL. Our key insight is that RL tends to be costly due to the need to perform both inference and backpropagation during training. To maximize use of computational resources, our algorithm performs inference in large batches, and then performs off-policy policy gradient updates in mini-batches. For off-policy updates, we incorporate group advantage estimation into the policy gradient algorithm, and derive an importance weighted estimator to correct for the bias arising from off-policy learning. Our experiments demonstrate that our algorithm can reduce running time by 11\%--34\% on three benchmarks compared to state-of-the-art RL algorithms while maintaining similar or better accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an RL algorithm, RAPID, which during each iteration first generates a large amount of samples and conduct offline policy gradient with group-relative rewards. Importance weightings are adopted for gradient estimation. The algorithm can save runtime compared with existing baselines like GRPO and PG. The experiments are conducted on three benchmarks, comparing RAPID with various baselines. The authors also provide analysis on the role of training batch size.

### Strengths
- Clear illustration of the algorithm.
- Extensive experiments on various benchmarks.
- The runtime is reduced compared with existing baselines.

### Weaknesses
- The figure 1 is a bit misleading. As in real experiments (say table 1), RAPID cannot save as much as 89% runtime compared with GRPO. And the number 89% only appears once, which is a bit confusing for readers.
- The reviewer cannot see much novelty in the algorithm design. It's more like a variant of GRPG (a simplified GRPO) with offline and iteratively generated batches, since all implementations are common practice in GRPO [1] and off-policy policy gradient [2].
- RAPID cannot outperform the simplest RL algorithm, PG, on Math and MiniF2F (two of three benchmarks in table 1), and the runtime advantage is not prominent on MATH compared with PG. Table 2 reflects the same trend on Qwen and llama.
- Figure 2 is noisy and cannot deliver convincing information.
- Hyperparameter H requires to be carefully tuned for good performance. This drawback further decreases the training efficiency.

[1] Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. https://arxiv.org/pdf/2402.03300

[2] https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#off-policy-policy-gradient

### Questions
- Could you compare the runtime memory usage of these methods? RAPID would use a larger batch size for inference.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes RAPID (Reweighted Advantage for Preemptive Inference of Data), a novel reinforcement learning (RL) algorithm for small language models (SLMs) to reduce training time during fine-tuning (for tasks like math, coding, and formal theorem-proving) while maintaining or improving accuracy; its core design includes alternating between large-batch inference (leveraging the memory efficiency of inference servers like vLLM to maximize efficiency) and small-batch off-policy policy gradient updates (optimizing gradient descent efficiency), integrating group advantage estimation into the policy gradient framework, and deriving an importance weighted estimator to correct off-policy learning bias—avoiding the performance limitations of KL-divergence regularization used in methods like PPO when taking multiple off-policy steps; the paper also conducts extensive experiments on three benchmarks (MBPP+ for coding, MATH for mathematics, MiniF2F for formal theorem-proving) using SLMs such as Qwen2.5, Llama 3.2 1B, and Gemma 3 1B, comparing with baselines like SFT, GRPO, Policy Gradient, and DAPO, and the results show RAPID reduces training time by 11%–34% across datasets (34% on MBPP+, 32% on MiniF2F, 11% on MATH) while achieving similar or better accuracy,

### Strengths
- The work focuses on an essential problem in reinforcement learning optimization, which is highly relevant for scaling RL algorithms to large models and complex environments.

- The manuscript is clearly presented, and the experimental section is complete enough.

### Weaknesses
1. The paper lacks novelty. The proposed components appear to be borrowed from existing popular approaches.

- The idea of inference in large batches has already been implemented in modern inference engines such as vLLM, which can automatically adjust batch sizes to the maximum available capacity.

- The use of group advantage estimation with normalization has been discussed in prior works such as GRPO and Kimi K1.5, and has already been widely adopted in subsequent research.
Therefore, the methodological contribution of this paper seems limited.

2. There are concerns regarding the correctness and credibility of the implementation and reported results. In Figure 1, the performance drop shown is excessively large, while the reported rollout time appears unrealistically low. If the only modification is increasing batch size, the time consumption should decrease partially but not be eliminated entirely. This raises doubts about whether the implementation and experimental measurements are accurate.

### Questions
Please see problem in section Weaknesses.

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
4

### Summary
This paper introduces RAPID (Reweighted Advantage for Preemptive Inference of Data), a novel reinforcement learning algorithm designed to enhance the efficiency of fine-tuning SLMs. The method alternates between large-batch inference and off-policy policy-gradient updates on smaller mini-batches, optimizing both inference and backpropagation phases. To address off-policy bias, RAPID employs an importance-weighted group advantage estimator and performs updates efficiently using stored experience data.

### Strengths
1. RAPID reduces RL training time, while maintaining competitive performance, making it especially valuable for resource-constrained deployment scenarios.

2. The alternating design elegantly decouples inference and training, and the importance-weighted group advantage estimator effectively mitigates off-policy bias.

3. The paper is well structured, with clear algorithmic exposition and empirical validation.

### Weaknesses
1. While runtime efficiency improves, accuracy gains over GRPO and PG are relatively modest, especially on the MATH and MiniF2F datasets (Table 1).

2. Evaluation is limited to three datasets (MBPP+, MATH, MiniF2F), which primarily cover mathematical and programming domains. Broader validation on diverse reasoning and language tasks (e.g., commonsense QA, dialogue, summarization) would strengthen generalizability.

3. The paper lacks ablation studies isolating the contributions of the importance weighting and group advantage estimation components. Such analyses would clarify their respective impacts on overall performance.

### Questions
1–3. See weaknesses above.

4. How sensitive is RAPID to the choice of importance weight clipping threshold $\eta = 2.0$ ? Was this parameter tuned per dataset, or is it robust across different tasks? An ablation study would be informative.

5. The algorithm introduces bias through importance weight clipping—has this bias been quantified empirically?

6. How does RAPID scale with larger models (e.g., 7B–13B)? Does the runtime advantage increase linearly with model size, or are there diminishing returns?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes RAPID, which focuses on reducing the running time of RL. Specifically, RAPID performs inference in large batches, and then performs off-policy policy gradient updates in mini-batches. Authors conducted experiments on MBPP+, MATH and MiniF2F with Qwen-2.5, 0.5B and 1.5B, Llama-3.2 1B, and Gemma-3 1B to illustrate the effectiveness of RAPID.

### Strengths
1. This paper is clearly written and easy to follow.

### Weaknesses
1. Although the authors claim in the title, abstract, and many other places throughout the paper that their primary focus is on small language models, I did not see any special design in their proposed algorithm, RAPID, that specifically targets them. Furthermore, the baselines they compare against, such as DAPO and GRPO, are also not algorithms specifically designed for small language models.
2. Authors claim that "We refer to the algorithm combining group advantage estimation and the policy gradient algorithm as group relative policy gradient (GRPG); this algorithm is a traditional on-policy RL algorithm", which is an erroneous statement. Equation (3) of the GPRO paper apparently considers importance sampling for off-policy updates.
3. Limited technical novelty. Authors claim the contribution of RAPID is doing inference at a large batch and then updating the policy with mini-batches to save computation. However, this is indeed the way that GRPO does! Please read carefully the open-source code of GRPO in verl.

### Questions
1. Could you please elaborate more on the value term in Equation (4)? From the theory of baseline, the baseline term can be anything as long as it is independent of the current policy. So I wonder the effectiveness of adding an additional IS term in value estimation.

### Soundness
1

### Presentation
3

### Contribution
1
