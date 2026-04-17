# Conditional Advantage Estimation For Reinforcement Learning In Large Rea- Soning Models

Guanxu Chen2,1∗ Yafu Li1† Yuxian Jiang3 Chen Qian4 **Qihan Ren**2 JingYi Yang3,1 Yu Cheng5 Dongrui Liu1† **Jing Shao**1†
1 Shanghai Artificial Intelligence Laboratory, 2 Shanghai Jiao Tong University, 3 Fudan University, 4 Renmin University of China, 5 The Chinese University of Hong Kong lm.cgx@sjtu.edu.cn yafuly@gmail.com {liudongrui, shaojing}@pjlab.org.cn

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) for large language models (LLMs) has achieved remarkable progress in enhancing LLMs' reasoning capabilities on tasks with clear correctness criteria, such as mathematical reasoning tasks. Several training metrics, such as entropy or response length, have been observed to correlate with different reasoning behaviors in reinforcement learning. Prior approaches incorporate such priors through reward or advantage shaping, which often relies on hand-crafted penalties and preferences (e.g., higher-is-better or lower-is-better). However, without careful hyper-parameter tuning, these directional priors can be overly biased and may lead to failure. To this end, we introduce Conditional advAN*tage estimati*ON (CANON), amplifying the impact of the target metric without presuming its direction. Specifically, CANON regroups the sampled responses into two groups based on the higher or lower value of a target metric, measures which metric trend contributes to better performance through inter-group comparison, and identifies the better response within the same group.

In summary, CANON based on entropy consistently outperforms prior methods across three LLMs on both math reasoning and high-complexity logic tasks. When applied to response length, CANON further improves token efficiency, yielding a more favorable Pareto frontier in the performance–cost trade-off.

## 1 Introduction

Recently, Large Reasoning Models (LRMs) such as Gemini 2.5 Pro (Comanici et al., 2025), DeepSeek-
R1 (Guo et al., 2025), and OpenAI-o1 (Jaech et al., 2024), continue to push the boundaries of performance on reasoning tasks. A key technique driving this success is Reinforcement Learning with Verifiable Rewards (RLVR), which enables models to refine answers through multi-step reflection. Algorithms designed for RLVR, most prominently GRPO (Shao et al., 2024) and its variants (e.g., DR.GRPO, Liu et al. (2025a)), have become central to achieving superior performance. In previous works, some training metrics are observed to be closely correlated with model behavior, which can guide the training process and improve LLMs' performance (Hassid et al., 2025; Gandhi et al., 2025; Wang et al., 2025). To incorporate such a human prior, some methods integrate these metrics through reward shaping (Arora & Zanette, 2025; Luo et al., 2025) and advantage shaping (Chen et al., 2025b; Cheng et al., 2025) to guide the model's reasoning behavior. For example, an over-length penalty is used to boost reasoning efficiency, and the entropy signal is leveraged to maintain exploration for better performance. However, these methods usually introduce human priors by adding penalty and reward terms, which hold handcrafted priors that specific metrics are either to be higher-is-better or to be lower-is-better. Without careful hyper-parameter selection, these priors can be overly biased and drive specific metrics up or down directly, thus failing to enhance performance robustly. Simple handcrafted priors towards one specific direction are hard to work in different scenarios. For instance, higher-entropy responses tend to be exploratory and may correctly answer complex questions, whereas lower-entropy responses
∗Our code is available at §CANON. †Corresponding Author.

exhibit higher certainty and achieve greater accuracy on most questions within their capability (Cheng et al., 2025; Prabhudesai et al., 2025; Wang et al., 2025). Therefore, we aim to amplify the impact of specific metric changes without presupposing preferences, naturally identifying inherent tendencies in model rollouts that can be leveraged to facilitate learning of beneficial behaviors, such as enhancing exploration or improving reasoning efficiency. To this end, we regroup the sampled responses into two groups based on the higher or lower values of a given metric during the process of RLVR training. Specifically, we can sort the sampled responses according to the value and split them into two groups. Based on this, we propose Conditional advAN*tage estimati*ON (CANON), which computes the inter-group advantage by comparing a response with the group that it does not belong to, and gets the intra-group advantage across its own group conversely. The inter-group advantage reveals which trend of metrics leads to higher accuracy. Meanwhile, the intra-group advantage identifies better responses within the same group. Taking the metric of entropy as an example, if groups with lower entropy (i.e., higher certainty) yield higher average rewards, the inter-group advantage tends to select correct responses with low entropy, efficiently exploiting existing features to boost performance. In contrast, correct rollouts with higher entropy receive more advantages in the intra-group comparison because the average reward of their group is lower, thereby encouraging truly effective exploration. We theoretically prove that when the two groups have equal size, the inter-group advantage amplifies the impact of the grouping metric on the advantage computation. In this setting, DR.GRPO can be formulated as a uniform weighting of these two advantages, which is a special case of CANON. We consider the metrics of generation entropy and response length, evaluating the effectiveness of CANON on three open-weight LLMs across six math reasoning benchmarks and three challenging logic reasoning tasks. Empirical results show that emphasizing the inter-group advantage based on entropy yields a 1.9-point accuracy gain on math tasks. In contrast, for high-complexity reasoning problems, the intra-group advantage proves crucial, achieving a 5.2-point improvement on the most challenging subset. Through scheduling of these advantages, CANON further achieves a superior and comprehensive performance across three models and two tasks. Furthermore, CANON based on response length substantially enhances reasoning efficiency, establishing a new Pareto frontier in the performance–efficiency trade-off. In low-token-budget scenarios for math tasks, it achieves **2.63×** higher performance and reduces token consumption by **45.5%** at the same performance level.

## 2 Related Work

Advantage Estimations in Reinforcement Learning. In PPO, the advantage estimation is provided by Generalized Advantage Estimation (GAE, Schulman et al. (2015)).To avoid the computational cost of the critic model, several methods, such as ReMax (Li et al., 2023), RLOO (Ahmadian et al., 2024), GRPO Shao et al. (2024), and REINFORCE++ (Hu, 2025), utilize alternative techniques like baseline reward and group-relative rewards for advantage estimation. ReMax compares the rewards with the baseline reward from the greedy decoding response. REINFORCE++ estimates the advantage by the normalization operation across the global batch for all queries. RLOO and GRPO estimate the advantage in a group relative manner. RLOO computes the average rewards of all other solutions in the group as the baseline reward, and GRPO utilizes the normalized rewards among the sampled solutions as the advantage estimation. Compared to GRPO, our method splits sampled responses into two groups based on specific conditions and selects the appropriate condition through inter- and intra-group comparisons, thereby efficiently optimizing key patterns that boost task performance. Reinforcement Learning with Verifiable Rewards. RLVR leverages the existing RLHF objective (Schulman et al., 2017) but replaces the reward model with a verification function, which is available in domains with verifiable answers, such as mathematics reasoning tasks (Guo et al., 2025; Lambert et al., 2024). Yu et al. (2025); Liu et al. (2025b); Chen et al. (2025a) consider the importance sampling techniques and contribute novel training paradigms and optimization objectives for better and more stable reasoning capabilities. Due to the sparse rewards during training, past methods utilize not only accuracy-based rewards but also explicitly integrate additional signals through reward shaping (Arora
& Zanette, 2025; Luo et al., 2025) and advantage shaping (Chen et al., 2025b; Cheng et al., 2025) to guide the model's reasoning and reflection. Arora & Zanette (2025) and Luo et al. (2025) utilize an over-length penalty to boost reasoning efficiency. Chen et al. (2025b) and (Cheng et al., 2025) consider the entropy as a measure of exploration and reshape the advantage computation. Gandhi et al. (2025) also observes four key cognitive behaviors of initial reasoning behaviors and strengthens

![2_image_0.png](2_image_0.png)

the capacity for self-improvement. However, these methods usually introduce human priors by adding penalty and reward terms, which hold handcrafted priors that can be overly biased and may fail to enhance performance without careful hyper-parameter selection. Our work amplifies the impact of specific metric changes without presupposing preferences, leveraging them to facilitate learning of beneficial behaviors.

## 3 Preliminaries

Proximal Policy Optimization (PPO, Schulman et al. (2017)) is a widely used method for policy optimization of LLMs. PPO utilizes the clip mechanism to update policy stably. PPO maximizes the following optimization objectives.

$$\mathcal{J}_{\text{PPO}}(\theta)=\mathbb{E}_{q\sim\mathcal{D},o\sim\pi_{\text{ad}}(\cdot|q)}\left[\frac{1}{|o|}\sum_{t=1}^{|o|}\min\left(r_{o,t}(\theta)\hat{A}_{t},\,\text{clip}_{1-\varepsilon}^{1+\varepsilon}(r_{o,t}(\theta))\hat{A}_{t}\right)\right]\,,\tag{1}$$
$${\mathrm{(2)}}$$

where πθold and πθ are used to denote the policy model before and after the update. q is a query sampled from the data distribution D, and the output o is generated by πθold . The clipping function with clip ratio ε is computed as clipba(x) = max(min(x, a), b) and the importance sampling ratio at time step t is defined as ro,t(θ) = πθ(ot|q,o<t)
πθold (ot|q,o<t)
.

To avoid the computational cost of the critic model, GRPO (Shao et al., 2024) estimates the advantage in a group relative manner. They sample G different solutions for the current query q as the group Gq := {o|o ∼ πθold (.|q)}, and calculate the normalized rewards as advantages within the group Gq.

$$\hat{A}_{q,o,t}^{\mathrm{GRPO}}=\frac{R_{o}-\operatorname{mean}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}\})}{\operatorname{std}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}\})}.$$

Due to the success of DeepSeek-R1, several studies have proposed improvements based on GRPO. DR.GRPO (Liu et al., 2025a) uses the GRPO advantages without standard deviation normalization and develops a token-level loss without length bias.

## 4 Conditional Advantage Estimation

Group-based advantage estimation methods, such as GRPO, typically use the average reward of all sampled responses within the group as a baseline reward. This may fail to provide a clear feedback signal for policy optimization due to the ambiguity of the comparison target. We propose CANON, which performs conditional regrouping by splitting all sampled responses into two groups based on the value of a specific metric. Leveraging these two groups, inter-group advantage identifies the metric trend that yields higher accuracy through cross-group comparison, while intra-group advantage selects superior responses within the same trend and prioritizes correct answers from the group with a lower average reward.

## 4.1 Conditional Regrouping

To explicitly introduce a comparison target, we regroup all the sampled responses based on specific conditions. Given any condition c, we denote the set of all outputs for the current query q that satisfy this condition in the sampled group Gq as C
+
q
:= {o|o satisfy *c, o* ∈ Gq}. The set of outputs that do not satisfy the condition can be denoted by C
−
q = Gq \ Cq. In this work, we focus on studying the relative conditions given by the training metrics, such as the entropy and length of the sampled responses. Specifically, we divide the responses into two non-overlapping groups based on the value of the metrics, as shown in Figure 1.

## 4.2 Advantage Estimation Based On Regrouping.

Given two groups, we can compute the inter-group advantage through comparison between different groups.

$$\hat{A}_{q,o,t}^{\mathrm{\footnotesize{inter}}}=\left\{\begin{array}{l}{{R_{o}-\operatorname{mean}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}^{+}\}),\mathrm{if}\;o\in G_{q}^{-}}}\\ {{R_{o}-\operatorname{mean}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}^{-}\}),\mathrm{if}\;o\in G_{q}^{+}}}\end{array}\right..$$
$$(3)$$

Meanwhile, we also compute the intra-group advantage by comparing each response with the mean reward of its own group.

$$\hat{A}_{q,o,t}^{\mathrm{intra}}=\left\{\begin{array}{l}{{R_{o}-\operatorname{mean}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}^{+}\}),\mathrm{if}\;o\in G_{q}^{+}}}\\ {{R_{o}-\operatorname{mean}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}^{-}\}),\mathrm{if}\;o\in G_{q}^{-}}}\end{array}\right..$$
$$\quad(4)$$

Although this may appear similar to the estimation of DR.GRPO within a smaller scope, due to the differing average advantages between groups, the intra-group advantage prioritizes correct responses from the group with a lower average reward (1 − mean({Ro
′ |o
′ ∈ G+
q } > 1 − mean({Ro
′ |o
′ ∈
G−
q } when mean({Ro
′ |o
′ ∈ G+
q } < mean({Ro
′ |o
′ ∈ G−
q } ). We can further combine the above two advantages into a unified formulation.

$$\hat{A}_{q,o,t}^{\mathrm{cannon}}=\mu\hat{A}_{q,o,t}^{\mathrm{inter}}+(1-\mu)\hat{A}_{q,o,t}^{\mathrm{intra}},$$
$$({\boldsymbol{S}})$$
q,o,t, (5)
where µ controls the balance between the inter-group and intra-group advantage. Figure 1 demonstrates a concise case of the computation of CANON.

To ensure that the advantages introduced by conditional regrouping provide a clearer contrastive signal, we theoretically analyze the situations under which inter-group advantage, compared to DR.GRPO, yields a stronger advantage signal in response to reward gaps under specific conditions. Theorem 1 (Situations with clearer advantage signal (proved in Appendix E)). Suppose that condition c is based on numerical comparisons and can be derived through sorting of metrics. Further assume that the sampled response o to query q satisfy condition c with probability p ∈ (0, 1)*, and* Eo satisfy c[Ro] ̸= Eo not satisfy c[Ro]*. Then, we have:*

$$\frac{\left|\hat{A}_{q,\sigma,t}^{\rm pure}\right|}{\left|\hat{A}_{q,\sigma,t}^{\rm pure}\right|}>1,\mbox{only when}|C_{q}^{+}|=|C_{q}^{-}|\mbox{if}|C_{q}^{+}|\mbox{is a constant.}\tag{6}$$

Based on Theorem 1, we divide the responses into two equally sized groups. In this way, DR.GRPO can be expressed as a special case of this unified form when µ = 0.5.

$$\hat{A}^{\rm DR,GRPO}_{q,o,t}=R_{o}-{\rm mean}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}\})=\frac{1}{2}\hat{A}^{\rm inner}_{q,o,t}+\frac{1}{2}\hat{A}^{\rm intra}_{q,o,t}.\tag{7}$$

Moreover, rather than a direct numerical amplification, CANON amplifies only the advantage attributable to the metric used for grouping, without amplifying the influence of other factors.

| Model                                                                                          | Math Reasoning   | High Complexity Reasoning   |      |      |      |      |                     |      |            |      |     |
|------------------------------------------------------------------------------------------------|------------------|-----------------------------|------|------|------|------|---------------------|------|------------|------|-----|
| AIME 24 AIME 25 Olympiad AMC MATH-500 GSM8k Tokens Acc Mid Large XLarge Tokens Acc             |                  |                             |      |      |      |      |                     |      |            |      |     |
| Base                                                                                           | 16.0             | 8.0                         | 26.4 | 41.6 | 61.2 | 61.6 | 2046 35.8 0.0       | 0.5  | 0.1        | 3303 | 0.2 |
| Instruct                                                                                       | 10.7             | 9.7                         | 39.7 | 49.3 | 82.2 | 94.8 | 1077 47.7 11.6      | 6.2  | 3.5        | 2647 | 7.1 |
| Previous Advantage Estimation                                                                  |                  |                             |      |      |      |      |                     |      |            |      |     |
| ReMax                                                                                          | 23.3             | 18.0                        | 48.1 | 62.8 | 83.4 | 90.3 | 2418 54.3 37.2 21.0 | 9.7  | 6246 22.6  |      |     |
| R++                                                                                            | 20.3             | 19.7                        | 45.8 | 58.3 | 82.6 | 90.0 | 4107 52.8 33.8 11.9 | 3.3  | 9923 16.3  |      |     |
| RLOO                                                                                           | 25.0             | 18.7                        | 51.3 | 64.3 | 84.0 | 91.0 | 2537 55.7 33.9 14.4 | 5.8  | 10610 18.0 |      |     |
| GRPO                                                                                           | 22.3             | 18.3                        | 47.3 | 60.6 | 83.8 | 90.8 | 3730 53.8 31.5 14.9 | 5.2  | 9406 17.2  |      |     |
| DR.GRPO (µ = 0.5)                                                                              | 27.7             | 20.3                        | 48.4 | 63.4 | 83.2 | 91.1 | 1522 55.7 39.2 24.4 | 15.1 | 4896 26.2  |      |     |
| Entropy-related Baselines                                                                      |                  |                             |      |      |      |      |                     |      |            |      |     |
| Entropy Adv                                                                                    | 26.7             | 16.7                        | 50.8 | 65.3 | 87.6 | 90.8 | 2389 56.3 30.8 17.1 | 7.5  | 8207 18.5  |      |     |
| Clip-Cov                                                                                       | 26.3             | 21.0                        | 49.0 | 63.5 | 84.8 | 92.1 | 1344 56.1 39.2 25.6 | 14.7 | 4045 26.5  |      |     |
| Our Methods (Conditional Groups based on Length)                                               |                  |                             |      |      |      |      |                     |      |            |      |     |
| CANON-Intra                                                                                    | 21.7             | 19.0                        | 49.9 | 63.0 | 86.2 | 92.2 | 2176 55.3 41.8 25.6 | 14.7 | 4364 27.4  |      |     |
| CANON-Inter                                                                                    | 27.3             | 19.3                        | 47.6 | 64.2 | 82.6 | 91.0 | 1008 55.3 42.7 28.6 | 17.1 | 3652 29.5  |      |     |
| Our Methods (Conditional Groups based on Entropy)                                              |                  |                             |      |      |      |      |                     |      |            |      |     |
| CANON-Intra                                                                                    | 25.0             | 16.0                        | 48.9 | 62.7 | 84.4 | 91.1 | 2959 54.7 39.1 27.8 | 20.3 | 3101 29.1  |      |     |
| CANON-Inter                                                                                    | 32.7             | 18.7                        | 51.7 | 64.2 | 87.0 | 91.1 | 1466 57.6 36.3 25.8 | 14.9 | 4415 25.7  |      |     |
| CANON-Dynamic                                                                                  | 30.0             | 17.7                        | 50.7 | 63.3 | 86.6 | 91.8 | 1452 56.7 40.4 30.5 | 16.6 | 3535 29.2  |      |     |
| Theorem 2 (Selective amplification based on specific metrics (proved in Appendix E)). Consider |                  |                             |      |      |      |      |                     |      |            |      |     |

Table 1: Overall performance based on **Qwen2.5-Math-7B**. We compare with the following baselines: (1) Qwen2.5-Math-7B-Instruct (Qwen-Instruct), (2) prior advantage estimation methods. All models are evaluated under a unified setting. **Bold** and underline indicate the best and second-best results, respectively.

Theorem 2 (Selective amplification based on specific metrics (proved in Appendix E)). Consider
independent conditions c1 and c2, and their corresponding sets C1 and C2 *(i.e.,* P(o ∈ C1 ∩
C2|*q, θ*) = P(o ∈ C1|q, θ)P(o ∈ C2|q, θ)). When we fix the condition c1*, then for any value of the*
performance gap a2+, a2− and the probability P(o ∈ C2|q, θ) that induced by whether c2 *is satisfied,*
we have
$$\frac{\left|\hat{A}_{q,o,t}^{i n t e r\,b a s e d\,o n\,c_{1}}\right|}{\left|\hat{A}_{q,o,t}^{D R.G R P O}\right|}\,\,\,i s\,\,a\,\,c o n s t a n t.$$
is a constant. (8)
which says CANON based on the condition c1 *will not amplify the influence of another independent* condition c2.

Therefore, CANON, when grouped by a specific metric, amplifies the influence of that metric during training, yet it does not predefine a preference for the magnitude of the metric. This design allows it to incorporate human priors while mitigating bias, which fully aligns with our original motivation.

## 4.3 Aligning With Training Target Through Weighted Advantage

According to Section 4.2, the selection between different trends of metrics only takes place in the inter-group advantage. By weighting different conditions within the inter-group advantage calculation, this enables fine-grained control over the trend of metrics with only tiny differences compared to DR.GRPO. For instance, by slightly reducing the weight of longer responses, CANON can accomplish reasoning of high token efficiency through the RL process. Specifically, the inter-group advantage in the Eq. 5 should be replaced with Aˆinter q,o,t,α where α is the weight of a specific group, and Aˆinter q,o,t,α is defined as:

$$\hat{A}^{\rm inter}_{q,o,t,\alpha}=\left\{\begin{array}{l}R_{o}-\alpha*{\rm mean}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}^{+}\}),{\rm if}\;o\in G_{q}^{-}\\ \alpha*R_{o}-{\rm mean}(\{R_{o^{\prime}}|o^{\prime}\in G_{q}^{-}\}),{\rm if}\;o\in G_{q}^{+}\end{array}\right..\tag{9}$$

$$({\mathfrak{s}}{\mathfrak{l}})$$

For example, setting α as 0.9 can achieve substantial length reduction with little performance drop, where C
+
qis considered the group with longer responses.

## 5 Experiments

The empirical evaluation of CANON consists of three parts. Firstly, we demonstrate the effect of intra-group and inter-group advantages, respectively, across six math reasoning benchmarks and one

![5_image_0.png](5_image_0.png)

high-complexity logic reasoning benchmark. In the second part, we perform several scheduling tricks to get the frontier in both tasks. At last, by weighting the longer responses with α < 1, we achieve efficient reasoning that reaches a better Pareto frontier.

## 5.1 Performance Of Intra-Group And Inter-Group Advantages.

Training Setup. We select the response length and the per-token generation entropy, respectively, to regroup the sampled solutions. We use a subset with 45k prompts from OpenR1-Math-220k (Hugging Face, 2025) that is filtered and constructed by Yan et al. (2025). Following DR.GRPO (Liu et al., 2025a) and DAPO (Yu et al., 2025), we correct the response-level length bias and utilize the clip-higher strategy (ϵ*high* = 0.28) for all experiments. We also remove both the KL loss and the entropy loss. We sample 16 responses per prompt and use temperature=1.0 for rollout generation. Our rollout batch size is 512, and the train batch size is 32. The responses to the same prompt are separated into two evenly sized groups by sorting ordinal variables. We conduct the main experiments on Qwen2.5-Math-7B (Yang et al., 2024) following Zeng et al. (2025); Liu et al. (2025a); Yan et al. (2025). We expand Qwen2.5-Math-7B's context limit from 4096 to 16384 by changing the rope theta from 10000 to 400001. We set the maximum answer length to 8192 and the learning rate is set to 1e-6. We use *Math-Verify* to give the 0-1 score for both training reward and evaluation accuracy.

Evaluation Setup. We evaluate the math reasoning capabilities on six commonly used benchmarks, such as MATH-500 (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021), AMC (Li et al., 2024), OlympiadBench (He et al., 2024), and AIME 24/25. Due to the tiny size of AIME 24/25 and AMC, we report *Avg@10* as the test accuracy. For the other benchmarks, we compute the *Pass@1* as the test performance. We calculate the average performance and token cost across all benchmarks. All models are evaluated under the same setting with a temperature of 0.6. The values in Table 1 are the percentage accuracy of the models evaluated. We also select three high-complexity subsets of ZebraLogic (Lin et al., 2025) with their solution space sizes greater than 103(Mid), 106(Large), and 109(XLarge), respectively. In this experiment, we record six metrics, including training reward, generation entropy, response length, the test performance of math tasks and logic reasoning task, and the marginal improvement gained from reflection.

Baselines. In this subsection, we fix α = 1.0 in Eq. 9 and present the results of µ = 0.0
(CANON-Intra) and µ = 1.0 (CANON-Inter) in Eq. 5. A more detailed scheduling on µ will be conducted in Section 5.2, and the adjustment of α will be covered in Section 5.3. We compare CANON with two types of baselines: (1) **Qwen2.5-Math-7B-Instruct** (Instruct, Yang et al. (2024)), (2) **previous advantage estimation methods**, such as ReMax, REINFORCE++ (R++), RLOO, GRPO, and DR.GRPO, and (3) **entropy-related baselines**, such as Entropy Adv (Cheng et al., 2025) and Clip-Cov (Cui et al., 2025).

| Model                                                                                              | Math Reasoning   | High Complexity Reasoning   |      |      |      |      |                     |      |           |            |
|----------------------------------------------------------------------------------------------------|------------------|-----------------------------|------|------|------|------|---------------------|------|-----------|------------|
| AIME 24 AIME 25 Olympiad AMC MATH-500 GSM8k Tokens Acc Mid Large XLarge Tokens Acc Qwen2.5-Math-7B |                  |                             |      |      |      |      |                     |      |           |            |
| DR.GRPO (µ = 0.5)                                                                                  | 27.7             | 20.3                        | 48.4 | 63.4 | 83.2 | 91.1 | 1522 55.7 39.2 24.4 | 15.1 | 4896 26.2 |            |
| Cosin-First-Inter-Later-Intra                                                                      | 30.0             | 17.7                        | 50.7 | 63.3 | 86.6 | 91.8 | 1452 56.7 40.4 30.5 | 16.6 | 3535 29.2 |            |
| First-Inter-Later-Intra                                                                            | 28.0             | 20.3                        | 52.4 | 64.6 | 84.2 | 92.6 | 1328 57.0 41.7 26.6 | 16.5 | 3862 28.3 |            |
| Qwen2.5-Math-1.5B                                                                                  |                  |                             |      |      |      |      |                     |      |           |            |
| DR.GRPO (µ = 0.5)                                                                                  | 13.3             | 11.0                        | 43.9 | 48.8 | 77.0 | 84.3 | 2381 46.4 23.7      | 9.7  | 5.0       | 9215 12.8  |
| Cosin-First-Inter-Later-Intra                                                                      | 17.3             | 13.7                        | 40.6 | 50.0 | 76.0 | 83.9 | 2357 46.9 19.2      | 8.9  | 4.2       | 10382 10.8 |
| First-Inter-Later-Intra                                                                            | 16.0             | 10.0                        | 42.4 | 50.2 | 78.6 | 83.3 | 2479 46.8 27.0 16.3 | 7.9  | 7070 17.0 |            |
| Llama3.1-8B                                                                                        |                  |                             |      |      |      |      |                     |      |           |            |
| DR.GRPO (µ = 0.5)                                                                                  | 1.3              | 0.3                         | 8.3  | 11.3 | 32.0 | 78.9 | 9476 22.0 21.1 13.8 | 9.7  | 5864 14.9 |            |
| Cosin-First-Inter-Later-Intra                                                                      | 0.7              | 0.0                         | 7.1  | 12.4 | 33.8 | 81.4 | 2354 22.6 26.0 18.4 | 12.3 | 1685 18.9 |            |
| First-Inter-Later-Intra                                                                            | 2.0              | 0.0                         | 8.7  | 9.9  | 31.8 | 80.1 | 3488 22.1 25.1 17.5 | 10.6 | 5892 17.7 |            |

Inter-group advantage achieves higher accuracy and lower length in math tasks. The experimental results are shown in Table 1. CANON-Inter based on *Entropy* achieves an average performance of 57.6 among six math benchmarks, which is 1.9 points higher than the DR.GRPO (55.7). Specifically, CANON-Inter based on *Entropy* has the best performance on four of the six benchmarks, and is highly competitive with the top-performing models on the rest. In AIME24, the model's performance is 5.0 points higher than the DR.GRPO's. Meanwhile, CANON-Inter based on *Length* reduces the token cost by 33.8% compared with DR.GRPO, while maintaining nearly unchanged performance (55.7 vs. 55.3). The benefit of intra-group advantage grows as the logic reasoning task's complexity increases. Table 1 demonstrates that CANON-Intra based on *Entropy* achieves higher performance of 2.9 points and 36.6% shorter length compared with DR.GRPO. Its performance edge over DR.GRPO increases (from -0.1 to 3.4 and then 5.2) when the complexity becomes higher. The results of CANON-Intra based on *Length* shows another trend, whose inter-group advantage makes the best performance in this task. Training dynamics reflect different roles of CANON-Intra and CANON-Inter. To be specific, we record training curves under the setting of CANON based on *Entropy*. The training dynamic shown in Figure 2 indicates that both the training reward and the test performance of the math tasks increase rapidly when only CANON-Inter is utilized (µ = 1.0). Its generation entropy stably decreases, and the response length changes smoothly. When using only CANON-Intra (µ = 0.0),
the responses show a greater tendency for exploration. We divide the responses into two groups by counting reflection patterns and calculate the gap in average reward between the group with more and fewer reflections (Figure 2f). Figure 2 demonstrates that the trend of high-complexity reasoning performance is highly consistent with the curve of reflection gains. In the later stages of training (after approximately 90 steps), the reflection gain curve of intra-group advantage increases and finally crosses the zero point. At the same time, its performance experiences rapid growth, significantly outperforming the other two advantages.

## 5.2 Balancing Performance Via Advantage Scheduling

As shown in Table 1 and Figure 2, CANON-Inter and CANON-Intra outperform DR.GRPO on the math reasoning task and the complex logic reasoning task, respectively, but neither can achieve the best performance on both simultaneously. To this end, we schedule the CANON-Inter and CANON-Intra by leveraging accuracy and the training steps to achieve a better balance between the two scenarios.

Setup. We conduct experiments across six math benchmarks and three complex logic reasoning tasks on Qwen2.5-Math-7B (Yang et al., 2024), Llama3.1-8B (Dubey et al., 2024), and Qwen2.5-Math1.5B (Yang et al., 2024). For the two Qwen series models, we use the dataset introduced in Section 5.1. Due to the weak capability of Llama3.1-8B, we collect a simpler dataset with 35k samples from four open-source datasets and follow the other training setups described in Section 5.1. Please see the details of this newly constructed dataset in Appendix C.5. We draw a radar chart with the average performance of the two scenarios for visualization, and the results for CANON with scheduling are denoted as CANON-Dynamic. Scheduling strategies. All of the strategies are based on the coefficient µ in the Eq. 5, which balances the CANON-Inter and CANON-Intra. We try four scheduling strategies utilizing the training accuracy and training steps, respectively: (1) *First-Inter-Later-Intra*. We set the value of µ to 1 − Λ, where Λ denotes the mean accuracy of current whole batch; (2) *First-Intra-Later-Inter*. We set the value of µ to Λ. (3) *Cosin-First-Inter-Later-Intra*. We schedule the value of µ from high to low using a cosine annealing function with restarts and warm-up. (4) *Cosin-First-Intra-Later-Inter*. We schedule the value of µ from low to high using a cosine annealing function with restarts and warm-up.

Please see Appendix C.6 for more details. The shown results of CANON-Dynamic are derived from one of the tried scheduling strategies that achieve strong performance in both scenarios.

First-Inter-Later-Intra **consistently performs better than DR.GRPO across three models and two**
tasks. As shown in Table 2, all three models demonstrate the same trend that performs better than the baseline by first applying Inter-group advantage and then using Intra-group advantage. Qwen2.5-1.5B performs particularly well under accuracy-based scheduling, possibly because its training accuracy range (0–0.6) aligns well with its learning progress. In contrast, the other two models may achieve higher final accuracies, which—under the same scheduling scheme—trigger excessive exploration and consequently lead to suboptimal final performance. We utilize fixed min/max values of µ by applying cosine annealing based on training steps, achieving higher performance. Moreover, different models may have different numbers of parameters and different levels of capability. A specifically designed strategy is acceptable for better performance in practice. In this way, we select strategy *Cosin-First-Inter-Later-Intra* for Qwen2.5- Math-7B and Llama3.1-8B, and strategy First-Inter- Later-Intra for Qwen2.5-Math-1.5B to draw Figure 3. As shown in Figure 3, CANON-Dynamic outperforms DR.GRPO across all models and tasks, achieving a superior and more comprehensive performance. Although its math performance on Qwen2.5-
Math-7B lags slightly behind CANON-Inter, it still makes a better performance than DR.GRPO. The radar chart illustrates the trade-off between two types of tasks faced by CANON-Inter and CANON-Intra between two types of tasks, as well as the balanced but mediocre performance of DR.GRPO.

![7_image_0.png](7_image_0.png)

 

g

## 5.3 Weighted Conditions For Efficient Reasoning.

Training Setup. In this subsection, we utilize CANON based on response length with µ = 0.5 in the Eq. 5 and tune the α in the Eq. 9, where C
+
qis considered the group with longer responses.

A larger α means less compression of length. We follow the training setups described in Section 5.1 and reduce the maximum response length to 3072 for better efficiency. To be specific, we use CANON-Eff to denote the results of CANON with weighted conditions of length.

Evaluation Setup under different token budgets and varying hyperparameter settings of different methods. To systematically assess LRMs' reasoning efficiency (Qu et al., 2025), we introduce two types of curves: **budget-performance curves for each LRM** and **cost-performance curves of** different coefficients for all compared baselines. Specifically, we set a maximum budget for each benchmark based on its difficulty and the average unconstrained output length of LRMs (Appendix C.2), then slice the same response at various budget ratios to draw the budget-performance curves.

Moreover, we tune the length-controlling coefficients of each baseline to draw the cost-performance curves, recording their average performance and token cost to enable a comprehensive and fair comparison. Please see the subsection on Pareto frontier for the specific hyperparameters. In every comparison, the closer to the upper-left corner, the better (which represents high accuracy and high efficiency at the same time).

| AIME 24                                                                      | AIME 25             | Olympiad   | AMC       | MATH-500   | GSM8k   | Overall   |      |      |      |      |      |      |           |     |
|------------------------------------------------------------------------------|---------------------|------------|-----------|------------|---------|-----------|------|------|------|------|------|------|-----------|-----|
| Acc Tokens Acc Tokens Acc Tokens Acc Tokens Acc Tokens Acc Tokens Acc Tokens |                     |            |           |            |         |           |      |      |      |      |      |      |           |     |
| DR.GRPO                                                                      | 29.0                | 1640       | 19.0      | 1586       | 49.0    | 1172      | 64.6 | 1214 | 85.8 | 728  | 91.9 | 349  | 56.6 1115 |     |
| Clip Length                                                                  | 28.0                | 1177       | 18.3 1177 | 47.3       | 915     | 63.1      | 956  | 84.8 | 612  | 92.9 | 291  | 55.7 | 855       |     |
| Length Reward+                                                               | 31.7                | 1190       | 18.0      | 1208       | 46.7    | 864       | 61.8 | 937  | 84.6 | 546  | 91.9 | 255  | 56.2      | 869 |
| Length Reward∗                                                               | 27.3                | 1087 13.7  | 1027 46.4 | 707        | 61.0    | 779       | 83.0 | 463  | 92.2 | 198  | 53.9 | 710  |           |     |
| CANON-Eff (α = 0.88) 27.3                                                    | 816                 | 15.3       | 862       | 43.9       | 582     | 59.3      | 649  | 84.4 | 386  | 91.4 | 166  | 53.6 | 577       |     |
| CANON-Eff (α = 0.96) 29.7                                                    | 1216 19.0 1136 48.4 | 881        | 62.3      | 936        | 85.8    | 533       | 92.0 | 233  | 56.2 | 822  |      |      |           |     |

Table 3: The comparison between different methods towards efficient reasoning. Bold and underline indicate the best and second-best results, respectively. The detailed performance is from the topperforming models for each method, specifically α=0.96 for CANON-Eff. We include CANON-Eff with α = 0.88, which has comparable performance with the baseline Length Reward (*).

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

Figure 4: Budget-Performance and Cost-Performance Curves for Efficient Reasoning. This figure compares the reasoning efficiency of CANON-Eff against baselines under various token budgets. Baselines. We select three types of baseline methods towards efficient reasoning: (1) Clip Length that directly clips the maximum output length (Hou et al., 2025), (2) Length Reward (+) that adds length penalties terms in the training reward (Luo et al. (2025), +coeff ∗ (
meanGq(L)
L − 1)), and (3)
Length Reward (∗) that multiplies a normalized length coefficient on the reward (Arora & Zanette (2025), ∗(1 − coeff ∗ sigmoid( L−meanGq(L)
stdGq
(L)))). All these baselines are conducted with DR.GRPO.

CANON **achieves better performance with shorter responses compared with baselines.** We present the detailed performance of the top-performing models for each method across various benchmarks in Table 3. CANON-Eff with α = 0.96 Pareto dominates the results of Clip Length and Length Reward (+), reducing the length by 26.3% compared to DR.GRPO while only decreasing performance by 0.4 points. Figure 4 shows that CANON-Eff with α = 0.96 consistently outperforms the baseline methods in both low-token-budget and high-token-budget scenarios. Since models trained with the Length Reward (∗) exhibit significantly lower length with low performance at the same time, it is difficult to fairly compare with other baselines. To this end, we include an additional model trained with CANON-Eff with α = 0.88 that has comparable performance. 4b indicates that CANON with α = 0.88 shows better token efficiency compared with Length Reward (∗), achieving 2.63 times the performance of DR.GRPO in low-token-budget scenarios, while reducing token consumption by 45.5% at the same performance level.

CANON **achieves a better Pareto frontier and stably explores the entire frontier.** To draw the cost-performance curves for each method, we draw the Pareto frontier of CANON-Eff with the results of α = 0.5, 0.7, 0.8, 0.88, 0.96. For Length Clipping, we respectively present the results with maximum lengths of 2048 and 1024 in the Pareto frontier. For Length Reward (+), penalty coefficients of 0.001, 0.004, 0.005, and 0.1 are used, respectively. For Length Reward (∗), we utilize the coefficients of 0.05, 0.2, and 0.4. 4c shows that all the frontier from baselines are dominated by the frontier of CANON-Eff's. It is noteworthy that after the coefficient of Length Reward (+) is adjusted from 0.004 to 0.005, its performance drops from 54.8 to 22.5. In contrast, CANON-Eff remains consistently stable, exploring the Pareto frontier efficiently.

## 6 Analysis

![9_image_0.png](9_image_0.png)

In this section, we analyze how CANON-Dynamic and CANON-Eff effectively improve the task performance and reasoning efficiency.

Different advantage combinations of CANON select different trends of the target metrics. Due to the different baseline rewards being compared, CANON-Inter tends to favor correct answers from the group with a higher average reward, while CANON-Intra selects correct answers from the group with a lower average reward. We compare the effects of CANON on their target metrics across seven different settings, with µ ranging from 0.0 to 1.0. When entropy is considered, figure 5 shows that a larger µ (favoring more CANON-Inter) leads to a reduction in entropy, whereas a smaller µ (favoring more CANON-Intra) promotes an increase in entropy. The results demonstrates a hierarchical trend in the metric changes, indicating the effectiveness of controlling and selecting different trends from CANON-Inter and CANON-Intra. In this way, CANON-Dynamic can boost the task performance by adjusting different combinations of the two components.

Figure 5: CANON shows hierarchical trends of target metrics through different combinations of CANON-Inter and CANON-Intra.

CANON **can achieve positive gains of more rethinking** and high training efficiency through scheduling of two advantages. As shown in Figure 6, we record the performance genuinely brought by reflections and the curve of training reward. Although CANON-Intra achieves positive gains from more reflections, its training reward experiences a significant decline. In contrast, CANON-Inter, which shows a similar trend of DR.GRPO, has not yet achieved positive returns even by step 360, but maintains a higher training reward. CANON-Dynamic, on the other hand, not only achieves positive gains of rethinking but also makes a training reward on a par with CANON-Inter's. This explains why CANON-Dynamic can achieve comprehensive leading performance in both math and complex logic reasoning tasks.

Table 4: The performance comparison between the direct numerical amplification of advantage and CANON.

Methods *Math Logic* DR.GRPO 55.7 26.2 Direct Numerical Amplification Numerical Scaling 56.1 25.1 Entropy Adv 56.3 18.5 CANON
CANON-Intra 54.7 **29.1** CANON-Inter **57.6** 25.7 CANON **amplifies only the advantage attributable to the** metric used for grouping, without amplifying the influence of other factors. As shown in Table 4, directly scaling the advantage (A = A ∗ 2) fails to improve performance the way CANON does. Any minor gains likely stem from faster learning progress due to an effectively larger learning rate, but this comes at the cost of degraded performance—particularly on out-of-domain logical reasoning tasks. This suggests that the key to CANON' success is not simply amplifying the advantage signal, but rather selectively amplifying specific signals, and that's why we introduce a regrouping operation.

## 7 Conclusion

![9_image_1.png](9_image_1.png)

In this paper, we introduce CANON, a novel reinforcement learning framework for large reasoning models that leverages human priors on training metrics (e.g., entropy, response length) without presuming their directional impact on performance. Extensive experiments across six math reasoning benchmarks and three high-complexity logic reasoning tasks demonstrate that CANON significantly outperforms prior advantage estimation methods like DR.GRPO. CANON also supports flexible weighting of different metric trends, where CANON based on response length achieves a superior Pareto frontier in the performance-efficiency trade-off. Our analysis further confirms that CANON promotes beneficial behaviors such as effective exploration and reflection, which are critical for solving complex reasoning problems.

## Acknowledgments

This work is supported by Shanghai Artificial Intelligence Laboratory.

## Ethics Statement

This work aims to introduce human priors about key metrics into reinforcement learning by proposing a novel advantage estimation framework named CANON, which amplifies the impact of target metrics without presuming preferences. The experiments in this paper are limited to reasoning tasks conducted on open-source models, datasets, and benchmarks, which will not raise ethical concerns. We hope to explore the potential of CANON to enhance the security of large language models in the future, thereby promoting their reliable and trustworthy development.

## Reproducibility Statement

We aim to include both the high-level and low-level details of our method in the setup paragraphs of Section 5 and Appendix C to reproduce our results. All experiments are conducted on open-source LLMs and benchmarks. We employ open-source datasets for the Qwen series LLMs, provide a detailed description of the prompts used for training and evaluation, and comprehensively present the construction process of the training dataset for the Llama series LLM. Our code implementation is based on VeRL (Sheng et al., 2024), which is applied with focused modifications in the advantage computation part, enhancing the reproducibility of our work. Please access our code base via the following anonymous link: CANON.

## Bibliography

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. *arXiv preprint arXiv:2402.14740*, 2024.

Daman Arora and Andrea Zanette. Training language models to reason efficiently. *arXiv preprint* arXiv:2502.04463, 2025.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. *arXiv preprint arXiv:2506.13585*, 2025a.

Minghan Chen, Guikun Chen, Wenguan Wang, and Yi Yang. Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization. *arXiv preprint arXiv:2505.12346*, 2025b.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. *arXiv preprint arXiv:2506.14758*, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. *arXiv preprint arXiv:2505.22617*, 2025.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Michael Hassid, Gabriel Synnaeve, Yossi Adi, and Roy Schwartz. Don't overthink it. preferring shorter thinking chains for improved llm reasoning. *arXiv preprint arXiv:2505.17813*, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.

3828–3850, 2024.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang.

Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.

Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum.

Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model, 2025. URL https://arxiv.org/abs/2503.24290.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https:
//github.com/huggingface/open-r1.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. *arXiv preprint arXiv:2411.15124*, 2024.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q. Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. https://huggingface.co/ datasets/Numinamath, 2024. Hugging Face repository, 13:9.

Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient method for aligning large language models. *arXiv preprint arXiv:2310.10505*,
2023.

Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, and Yejin Choi. Zebralogic: On the scaling limits of llms for logical reasoning. arXiv preprint arXiv:2502.01100, 2025.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. *arXiv preprint arXiv:2503.20783*, 2025a.