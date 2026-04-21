# Don't flatten, tokenize! Unlocking the key to SoftMoE's efficacy in deep RL

- Avg Score: 7.40
- Decision: Accept (Spotlight)
- Scores: 5, 8, 8, 8, 8

## Abstract
The use of deep neural networks in reinforcement learning (RL) often suffers from performance degradation as model size increases. While soft mixtures of experts (SoftMoEs) have recently shown promise in mitigating this issue for online RL, the reasons behind their effectiveness remain largely unknown. In this work we provide an in-depth analysis identifying the key factors driving this performance gain. We discover the surprising result that tokenizing the encoder output, rather than the use of multiple experts, is what is behind the efficacy of SoftMoEs. Indeed, we demonstrate that even with an appropriately scaled single expert, we are able to maintain the performance gains, largely thanks to tokenization.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
MoE (Mixture of Experts) has demonstrated significant potential. The latest research on MoE, exemplified by softMoE, attributes the performance advantages of these algorithms to structural sparsity. However, the authors of this paper propose a contrasting view. They observe that even a single-expert version of softMoE can perform quite well. Additionally, they find no evidence of expert specialization occurring in softMoE.

Through ablation studies, the authors discover that "tokenization" of the output from the convolutional encoder plays a crucial role. As a result, they suggest that tokenization is a highly important operation.

The authors investigate the importance of different components of softMoE in Atari game experiments, including: (i) the use of a learnable tensor Φ for obtaining dispatch (ΦD) and combine (ΦC) weights; (ii) processing p input slots per expert; (iii) architectural dimensions (network depth and width); (iv) the use of n experts; and (v) the tokenization of the encoder output.

A key finding is that tokenization significantly enhances performance. By applying only tokenization to the baseline, a notable performance improvement is observed, demonstrating that tokenization is a primary contributor to the effectiveness of softMoE. Furthermore, even a single-expert softMoE achieves relatively strong results.

Finally, the authors explore methods to improve the performance of multi-expert MoE models. They attempt reset and S&P methods, testing them with Rainbow and DER algorithms, but do not find any approach that consistently enhances multi-expert MoE performance.

### Strengths
The authors clearly present three different classical methods for processing encoder features: Flatten (baseline), SoftMoE architecture, and the routing mechanism.

### Weaknesses
1) The paper has a significant issue: the baseline scaled by *4 used throughout the experiments appears to perform worse, potentially due to suboptimal parameter. The use of the *4 baseline is problematic, as it may unfairly weaken the baseline, thus exaggerating the benefits of tokenization. 

The authors could run experiments comparing unscaled baselines to SoftMoE models with equivalent parameter counts. Specifically, It is crucial to compare the original-sized baseline with the SoftMoE where experts are scaled down by four times, and also compare it with a single expert SoftMoE scaled down by four times. 

When examining Figure 5, comparisons between unscaled results reveal that both the tokenized sum and tokenized average actually perform worse. This suggests that the effectiveness of tokenization depends on a balance between network structure, task complexity, and parameter settings, rather than indicating that tokenization is inherently beneficial. 

2) The paper also mentions that a single-expert SoftMoE performs well, which suggests that multiple experts may not be critical, and that tokenization is the key factor. However, Figure 10 indicates that combining a single expert with tokenization in SoftMoE does not improve performance in DQN. This contradict the claims that a single expert is effective. This indicates that the paper’s main assertions are not solid. Could the authors address this discrepancy and provide additional analysis or experiments to clarify the effectiveness of single-expert SoftMoE across different algorithms？

The truth might be that the authors' DQN setup is relatively underperforming, making the multiple-expert configuration beneficial to compensate the weakness of single expert. In contrast, stronger algorithms perform well enough with a single expert, making multiple experts redundant. Testing on more complex tasks might reveal a clearer advantage of using multiple experts over a single one. Therefore, a more reasonable interpretation would be that both tokenization and multiple experts contribute to SoftMoE’s performance, aligning with the fundamental concept of MoE rather than overemphasizing the importance of tokenization.

3) In Section 6, the authors aim to develop new techniques to improve expert utilization and maximize the benefits of MoE architectures in RL. However, only two tricks are listed, and the number of baselines compared is too limited. The paper fails to present methods that consistently improve multi-expert MoE performance. If the goal is to provide which trick may improve certain algorithm, a broader set of techniques should be explored.

### Questions
1) The authors mentioned that "they demonstrated that a performance drop is not observed when scaling down the dimensionality of each expert." Did you experiment with a setup where the dimensionality is reduced by 4 times in the SoftMoE model? Specifically, the experimental group would use SoftMoE with 4 experts, each scaled down by a factor of 4 to maintain the same overall model size as the baseline. The control group is baseline, which would not be scaled. This setup might highlight the importance of the number of experts in SoftMoE if the baseline performance remains unaffected.

2) Could the authors provided training curves for each individual Atari game? Comparing the performance of the Unscaled baseline and Unscaled tokenize models on each task.

3) Did the authors conduct experiments on environments like DeepMind Control (DMC) or Meta-World to thoroughly demonstrate the generalization capabilities of the tokenize?

4) Why is "Network Plasticity in RL" discussed in the related work section? Which part of the paper is relevant to plasticity? Does it correspond to Section 6? If so, the experimental results presented in Section 6 do not seem to lead to any valuable conclusions.

### Soundness
2

### Presentation
4

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
Obando-Ceron* et al. (2024) [1] demonstrated that SoftMoEs are effective architectures for scaling models in online RL. However, their paper did not explain the reasons behind this performance gain at scale. This submission analyzes factors that might contribute to the effectiveness of SoftMoEs at scale by ablating different components of the SoftMoE architecture within the Rainbow, DER, and DQN baselines on the Atari benchmark.

Overall, the authors show that tokenizing the encoder output, rather than using multiple experts, is the primary factor driving SoftMoE’s effectiveness. They also demonstrate that a single scaled expert with tokenization can match the performance of multiple experts.

[1]: Ceron, Johan Samir Obando, et al. "Mixtures of Experts Unlock Parameter Scaling for Deep RL." Forty-first International Conference on Machine Learning.

### Strengths
- The paper is well-written, easy to follow, and well-motivated.
- The authors conducted numerous empirical ablations on 3 different algorithms—primarily on Rainbow, but also on DER and DQN—using the popular Atari benchmark.
- I appreciate that the authors reported the IQM and Optimality Gap on 5 seeds for statistical significance.

### Weaknesses
While this paper provides an in-depth exploration of tokenization in SoftMoEs, there are areas where it could be improved to make it an accept.

1. The paper’s scope feels somewhat narrow, as it primarily focuses on *why* SoftMoEs work well when scaled up on the Atari benchmark using the Rainbow, DER, or DQN algorithms alone. As a result, it remains unclear whether SoftMoEs would be effective in more challenging OOD generalization benchmarks, such as Procgen, which present greater difficulty than the IID singleton environments like Atari.
2. Additionally, as shown in Figure 10, the SoftMoE-1 (scaled 4x) baseline performs significantly worse in the DQN setting. It would be beneficial for the authors to test their scaled-single-expert approach on another algorithm, potentially an actor-critic algorithm like PPO, in a pixel-based environment like Procgen.
3. Given the popularity of DQN, further investigation into the differences in DQN’s behavior compared to the other two algorithms would also add more value to this submission.

### Questions
I have some questions based on the current state of submission:

- For all trends observed in Section 4.2, do the authors anticipate that similar trends would hold for DER? Could the hypotheses be verified on an additional algorithm to confirm that the design choices assessed in Section 4.2 are generally applicable and not specific to one algorithm i.e. Rainbow?
- In Figure 5, did the authors observe a clear scaling trend from 1x to 2x, 4x, and 8x using the tokenized_sum baseline with either CNN or IMPALA? Also, in Figure 5, the tokenized scheme was [h\*w, d], which I believe corresponds to PerConv tokenization. Would similar trends hold if [d, h\*w] (i.e., PerFeat tokenization, similar to Figure 7) were used instead? This would help confirm whether tokenization generally improves performance, even if PerConv outperforms PerFeat, as long as both do better than simple flattening on the baseline.
- In Figure 8, when selecting 10% of the slots, are these slots chosen randomly, or is a specific heuristic used for pruning? Additionally, are these 10% slots fixed throughout training, or do they change randomly at each training iteration?
- Why was the analysis in Section 6 (Figure 11) not conducted for DQN?


**Typos:**

Page 3 Line 134: propose —> proposed

Page 8 Line 385: performance on all the 60 games —> performance on all **of** the 60 games

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper analyzes the effect of different components in the soft-of-mixture of experts (SoftMoEs) in online RL, the goal is to understand the key factors and design decisions that influence/derive the performance of (SoftMoEs), the analysis shows that tokenizing the output of the cnn encoder has the biggest effect on the performance, even when using a single expert or the baseline model, tokenizing the encoder output results.

### Strengths
- Results are significant with small variance over the 60 Atari games. 

- The effect of tokenization seems to transfer between architectures, which might suggest that using tokenization would always be helpful, at least in the algorithms used in the paper (Rainbow and DER).

- Showing that we are underutilizing the mixture of experts in (SoftMoEs) is an important implication of this paper, which encourages more research in this area.

### Weaknesses
- Authors should add a section that explains what a token is and what a slot it is, this is not defined in the paper and it will make the paper much more clearer. 

- The effect of tokenization in a single expert does not seem to transfer to DQN, which suggests there is something missing in the analysis, the authors suggested that it might be related to the categorical loss used in Rainbow and DER, but there is no further investigation.

### Questions
- In line 361, computational efficiency by combined tokens, I do not understand the point of the plot, is it to show that using fewer slots (which means better time complexity) still results in good performance? Can you add a plot that directly shows the relationship between the number of slots and time complexity?

- The authors argue that tokenizing the encoder output preserves the spatial information unlike flattening, can the authors run a baseline where the cnn encoder output is actually a vector? This can be done by adding a global average pooling layer after the last conv layer, which will reduce the spatial dimension to one.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the key factors that make SoFeMoE effective in visual reinforcement learning. Through comparisons, it demonstrates that tokenization and the combination of token weights, rather than simply using multiple experts, drive the performance improvements. The conclusions are based on experiments conducted in the Arcade Learning Environment, covering 60 games and using 5 seeds.

### Strengths
- The paper is well-written and easy to read.
 - The figures effectively illustrate the different analyses and help understand the main results.
 - The analysis is solid and well presented.

### Weaknesses
The main concern is the generality of the claim.
 - The experiments are conducted in one simulation platform with all discrete actions. It is unclear whether the same observation is universal applicable to other simulation platforms, especially involving agents with continuous states and actions.
 - All the results are conducted with 4 experts. Would the conclusions be applicable to different numbers of experts?
 - In Figure 6, it will be beneficial to visualize the performance with one unscaled expert to understand the results better.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper extensively studies the degree of contributions by each factor of SoftMoEs, a method mitigating the inverse proportionality between the performance and the architecture size of the online value-based deep reinforcement learning (DRL) methods. Despite the effectiveness of the approach, which component of SoftMoEs drives the improvement remains a mystery. The paper discovered that tokenization, the scheme of converting extracted features while maintaining its spatial structure, plays a significant role in SoftMoEs, bestowing the ability for the algorithm with a large network to perform well. From this insight, the paper further extends the argument to the redundancy of the experts, claiming that a single expert performs competitively against the multi-expert under certain conditions.

### Strengths
The paper tackles the important problem: *which factors of SoftMoEs benefit the most in mitigating the performance degradation of DRL algorithms with the increase of network size?* The problem is significantly important as it falls into the category of questions that ask the fundamental mechanism of the algorithms. Questions asking the fundamental mechanism of “why” the existing approach is effective are often overlooked in the community but also broadly impact the community in multiple ways (provides the common ground for the future methodologies, encourages the community to rethink the current approaches, etc.). Successfully answering the fundamental question thus turns this paper into an invaluable work with a potentially significant impact on the community.

The main finding of the paper is that tokenization of the extracted features of observations majorly contributes to performance improvements. This argument has been drawn and supported by the series of experiments along with the rigorous empirical analysis (adoption of IQM, 95% confidence interval over the stratified bootstrap samples), which reinforces the legitimacy of the claim.

### Weaknesses
- The significance of tokenization is limited to particular settings and does not apply robustly over the value-based online DRL. In fact, Figure 10 depicts that the scaled SoftMoE-1 resembles a similar performance/optimality gap from that of the scaled Baseline on DQN, indicating the minor, near-zero effect on the performance improvement by the tokenization. 
- The paper lacks some crucial details on the experimental settings of SoftMoEs. One of the prominent ones is the number of slots $p$. Unknown $p$ value reduces the confidence in the conclusion drawn in paragraph **Expert specialization** (line 231). Here, the paper claims that the specialization of experts is not the primary factor contributing to the high performance by increasing the number of $p$ to the number of tokens. However, the analysis might not be valid if the default value of $p$ is close to the number of tokens, and there is no way the readers can notice this unless specified in the paper. The paper would benefit from clarifying the actual values of the default configurations of SoftMoEs. 
- Some ablation studies omit important explanations, making it hard to follow the arguments. For instance, in the paragraph **Tokenization baseline**, the paper replaces the feature flattening operation of the Baseline with a tokenize-and-aggregate (either by average or sum) operation. The results suggest that tokenization with scaled representation significantly improves the performance of the Baseline, Rainbow-lite architecture. However, it is hard to connect this finding to the claim that “tokenization plays a major role in the efficacy of SoftMoE” (line 295). An additional explanation bridging similar logical gaps in the ablation studies (especially from the results to the final statement) would gradually improve the clarity of the paper. 
- The effort towards the hyperparameter sweeps is not explicitly mentioned in the paper. In empirical studies, hyperparameter sweeps are necessary for fair comparison, especially when the purpose of comparison is to determine the effectiveness of approaches. Omitting this step weakens the arguments made in section 6, mitigating experts’ redundancy by parameter reset and S&P. 
- Although mentioning how to improve expert utilization (section 6) marks an interesting future direction, it also feels unnecessary. This is mainly due to the open-ended analysis and the fact that the argument is slightly out of the main focus of the paper. The consistency and logical flow of the paper would be improved by moving section 6 to the appendix. 

In addition to these points, some minor formatting errors and ambiguous presentations caught my attention:

- Some references miss the year of publication. For instance, line 469 contains two works cited without the publication years. 
- Inconsistent citation formats. Capitalization of titles, format of venues, etc. 
- Some figures lack explanations. Specifically, the target quantity is unclear in the bar plots: Figure 1, 7, and 9. While it is mentioned that a human-normalized score is the target of measurements in the main text (section 4.1), it would be convenient to indicate within the figure. Also, since other figures indicate human-normalized scores, clarifying the target quantity in all the figures would further improve consistency.

### Questions
- How does the result depicted in Figure 5 reduce to the claim: “providing strong evidence that tokenization plays a major role in the efficacy of SoftMoE” (line 295)? 
- As mentioned in the weaknesses section, some empirically supported claims are skeptical due to the lack of information about the default configurations of SoftMoEs. Would the authors be able to list the precise configurations? 
- Is a hyperparameter search for random resets and S&P conducted in section 6?

### Soundness
3

### Presentation
3

### Contribution
4
