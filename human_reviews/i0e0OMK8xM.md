# WARP: On the Benefits of Weight Averaged Rewarded Policies

- Decision: Reject
- Scores: 6, 6, 5, 5

## Abstract
Reinforcement learning from human feedback (RLHF) aligns large language models by encouraging their generations to have high rewards, using a reward model trained on human preferences. To prevent forgetting of pre-trained knowledge, RLHF usually incorporates a KL regularization; this forces the policy to remain close to its initialization, though it hinders the reward optimization. To address the trade-off between KL and reward, in this paper we introduce a novel alignment strategy named Weight Averaged Rewarded Policies (WARP), merging policies in the weight space at three distinct stages. First, it uses the exponential moving average of the policy as a dynamic anchor in the KL regularization. Second, it applies spherical interpolation to merge independently fine-tuned policies into a new enhanced one. Third, it linearly interpolates between this merged model and the initialization, to recover features from pre-training. This procedure is then applied iteratively, with each iteration's final model used as an advanced initialization for the next, progressively refining the KL-reward Pareto front, achieving superior rewards at fixed KL. Experiments with Gemma policies validate that WARP improves their quality and alignment, outperforming open-source models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel alignment technique based on weight average (WA)  for reinforcement learning from human feedback (RLHF) applied to large language models (LLMs). The proposed WARP approach is designed to enhance the balance between reward optimization and the retention of pre-trained knowledge, tackling issues such as catastrophic forgetting and reward hacking often faced in RLHF. The WARP technique integrates three model-merging stages: using the Exponential Moving Average (EMA) as a dynamic anchor in KL regularization, Spherical Linear Interpolation (SLERP) for merging fine-tuned policies, and Linear Interpolation Towards Initialization (LITI) to regain pre-trained features. These steps iteratively refine the KL-reward Pareto front, progressively enhancing model performance. Experimental results on the Gemma 7B model indicate that WARP attains a better Pareto front and achieves superior alignment and generalization across benchmarks, outperforming existing methods.

### Strengths
1. **Innovative WA-based Approach to Alignment**: WARP introduces a layered model-merging strategy that addresses KL-reward trade-offs innovatively, leveraging EMA and SLERP in a structured way to boost model alignment with human preferences. Furthermore, the iterative application of WARP stages can progressively refine the KL-Reward Pareto front. 
2. **Empirical Validation**: The authors conducted comprehensive experiments to verify the effectiveness of the proposed four algorithmic components: EMA as a dynamic anchor, SLERP, LITI and iterative WRAP. Besides, comparison experiments on the Gemma 7B model show that WARP consistently outperforms open-source models, particularly in alignment and benchmark performance, providing strong evidence for its efficacy.
3. **Scalability and Adaptability**: The WARP process is compatible with distributed learning setups, enabling scalability through parallelizable fine-tuning iterations and potentially making it viable for broader applications in open-source and collaborative environments.

### Weaknesses
1. **Increased Computational Cost**: The biggest weakness of WRAP is the computation aspect. The iterative nature of WARP requires multiple RL runs and merging stages, which could be computationally demanding, potentially limiting its practicality for resource-constrained applications.

### Questions
1. Can the authors provide insights and more detailed explanations on why interpolating weights towards the initialization can attain a better Pareto front than the one revealed during RLHF?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work gives a novel alignment strategy WARP. This method first uses EMA anchor to train multiple policies with RLHF, then use spherical averaging to average policies, and finally use Linear interpolation to combine with the init model. This process can be repeated. Many experiments have been done to show the effectiveness of this method.

### Strengths
1. This work has a good presentation and is easy to understand. 

2. The method is shown clearly. The intuition is given and the ablation study for each stage is given.

3. Many experimental results are given to support the method.

### Weaknesses
1. I think it is better to add some more introduction for weight averaging. Since I am not quite familiar with this, I may wonder if this is a method  conducting averaging by each parameter or some other techniques.

2. I hope to get some intuition for why WARP can reach a Pareto front, which seems non-trivial for me.

### Questions
See weakness part above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies fine-tuning language models to align with human preferences. It addresses the trade-off between KL divergence and reward, which may help mitigate the alignment tax issue. Specifically, it introduces Weighted Averaged Reward Policies (WARP), merging policies in the weight space using techniques such as exponential moving average and spherical linear interpolation. Experiments on Gemma models demonstrate the effectiveness of the proposed method.

### Strengths
- This paper is well-written and easy to follow. Extensive related work is provided to help readers understand the research problem.
- The proposed methods seem to be simple yet effective in balancing the trade-off between KL divergence and reward.
- Extensive numerical results are provided to justifiy the proposed method.

### Weaknesses
- The main drawback of this paper is that techniques like exponential moving average and SLERP are well-known, which renders the technical novelty insufficient.
- Despite extensive numerical evaluation, it is unclear why model averaging helps balance the trade-off between KL divergence and reward. According to the reviewer’s understanding, achieving this trade-off effectively requires the policy to explore intelligently and update gradually during initialization. However, it is unclear how this relates to the proposed technique or if there are other possible mechanisms and explanations.

### Questions
Some experimental details are unclear:

- In Table 1, the reviewer would like to know who is responsible for evaluating the side preference rate. Is it conducted by GPT-4 or by human labelers? Additionally, could you report the response length? If judged by GPT-4, the win rate may be more easily biased by the response length.

- The reviewer would like to know what kind of dataset is used for RLHF. It is quite surprising to see improved performance in the benchmarks shown in Table 2. The results in the table seem to suggest there is no alignment tax.
- The main numerical evaluation in this paper is the reward. Since only Table 2 appears to show the benefit of KL (I assume), could the authors provide evaluation results of benchmarks in Table 2 at different points along the KL-reward trade-off curve, or for configurations with different hyperparameters?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper improves the trade-off between KL and reward during RLHF by proposing Weight Averaged Rewarded Policies (WARP). WARP is designed to optimize the KL-reward Pareto front of solutions, by using three variants of WA at three different stages of the alignment procedure: Exponential Moving Average (EMA), Spherical Linear intERPolation of task vectors (SLERP) and Linear Interpolation Towards Initialization (LITI).

### Strengths
- The paper is clearly written and easy to understand, using exponential moving average of the base policy in RLHF should be less conservative than a fixing target.
- The empirical analysis is robust, featuring comparisons against state-of-the-art models and demonstrating WARP's effectiveness in balancing KL regularization and reward optimization.
- The paper considers a wide variety of tasks to demonstrate potential improvements.

### Weaknesses
- Application of existing techniques to RLHF: The novelty is limited, the techniques used in the paper are almost proposed from the deep learning or reinforcement learning literature. For example, EMA anchor used ideas from trust-region updated reinforcement learning algorithms like TRPO [1]; SLERP uses idea from model merging by weight averaging [2,3]. LITI uses idea from WiSE-FT [4]. Given previous work that already applies similar ideas to reward modeling RLHF [5], the novelty is further weakened. A rigorous theoretical framework to justify why applying these techniques to the policy is better than reward is missing.
- Ablation studies: The paper introduces three different procedures, it's better to provide a detailed ablation studies to each procedure to verify no one procedure is useless.
- Training WARP is costly, requiring multiple iterations and multiple RL runs at each iteration. Also, more hyperparameters are introduced, making the algorithm more complex.

[1] Trust region policy optimization.

[2] Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.

[3] Diverse weight averaging for out-of-distribution generalization.

[4] Robust fine-tuning of zero-shot models.

[5] WARM: On the Benefits of Weight Averaged Reward Models.

### Questions
1. Why should we have the Linear Interpolation Towards Initialization (LITI) step? This step takes an interpolation
from the merged model towards the initialization as the next iteration's initialization. Why do we need that? Can't that be implemented by adjusting the \beta in KL regularization with updated base policy and using only one iteration?
2. In Algorithm 1, is line 13 and the ouput using the same \eta?

### Soundness
3

### Presentation
3

### Contribution
2
