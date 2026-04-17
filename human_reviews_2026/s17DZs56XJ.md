# Training Matryoshka Mixture-of-Experts for Elastic Inference-Time Expert Utilization

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Mixture-of-Experts (MoE) has emerged as a promising paradigm for efficiently scaling large language models without a proportional increase in computational cost. However, the standard training strategy of Top-K router prevents MoE models from realizing their full potential for elastic inference. When the number of activated experts is altered at inference time, these models exhibit precipitous performance degradation. In this work, we introduce Matryoshka MoE (M-MoE), a training framework that instills a coarse-to-fine structure directly into the expert ensemble. By systematically varying the number of activated experts during training, M-MoE compels the model to learn a meaningful ranking: top-ranked experts collaborate to provide essential, coarse-grained capabilities, while subsequent experts add progressively finer-grained detail. We explore this principle at multiple granularities, identifying a layer-wise randomization strategy as the most effective. Our experiments demonstrate that a single M-MoE model achieves remarkable elasticity, with its performance at various expert counts closely matching that of an entire suite of specialist models, but at only a fraction of the total training cost. This flexibility not only unlocks elastic inference but also enables optimizing performance by allocating different computational budgets to different model layers. Our work paves the way for more practical and adaptable deployments of large-scale MoE models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Matryoshka MoE, that activates different number of experts per layer. This elastic activation can fit different computational budgets.

### Strengths
The exploration of M-MoE variants (such as batch-wise, layer-wise, etc) is comprehensive.

The writting is clear.

### Weaknesses
The biggest weakness of the work is its impracticality. Varying activation levels across layers introduce notable complexity for practical deployment, kernel implementation, and communication engineering. First, the model must be deployed on devices that can accommodate the maximum memory and computational requirements, right? However, when some layers activate fewer experts, this leads to more pipeline bubbles and results in resource waste. If the top-k activation is too large to fit on a device, your results indicate that continual training a specialist model with a smaller k performs well enough, and is more practical and cost-effective.  

Additionally, the technical contribution is somewhat weak, as the proposed method is relatively straightforward.

Below are some weaknesses related to the method and its results.

1. Figure 1 is misleading because its x-axis omits the range from 4 to 7. I conducted some experiments on OLMOE (pre-trained with topk=8) by adjusting k during inference. The 0-shot results are shown below:
|k|arc-c|arc-e|piqa|hellaswag|winogrande|
|-|-|-|-|-|-|
|8|47.1|78.16|80.14|58.01|68.67|
|7|45.9|78.37|79.87|58.08|68.11|
|5|44.97|76.94|79.16|57.14|66.22|

When using an inference k smaller than the pre-trained k=8, performance does drop—but the decline is far less drastic than Figure 1 suggests. This discrepancy arises because the figure omits the x-axis range where the slight drop occurs. I recommend the authors include the full x-axis and corresponding experimental results to avoid overstating the severity of the issue illustrated.

2. The experiment setup in Table 1 is unrealistic and unconvincing. The base model for continual training only activates 1 expert per layer (topk=1), but pre-training MoE models with topk=1 is rarely practiced in real-world scenarios. This setup makes increasing k during continual training easy: the model could simply add some useless but harmless experts, while the single expert trained during pre-training remains the primary contributor to performance. How to rule out this possibility?

3. For the specialist baseline (k*=6) in Table 2, the authors should report results for Inf.k = 2, 3, 4. As noted earlier, performance drops only slightly when Inf.k is marginally smaller than the pre-trained k. Currently, Table 2 only includes Inf.k=1 and the pre-trained k, omitting intermediate values. This incomplete comparison might overstates M-MoE’s advantage.

4. Table 3 has confusing bold text and lacks significance tests. Is the value 55.42 incorrectly bolded? Moreover, I do not think there are statistically significant differences in the results in Table 3. It is hard to support the claim that "earlier layers are more critical."

5. If you consider a low MODS to be a good metric and believe that earlier layers are more critical, in Figure 3, the Matroshka model performs worse than the Top-K model in shallow layers due to its higher MODS scores. How can this be explained?

### Questions
1. What is your base model? Is it an open or private model? I did not find detailed descriptions of this model. Additionally, why was this model pre-trained with only one expert activated per layer? This is not a commonly used setup.

2. What is the formulation of the Focused Spearman Correlation?

3. See Weaknesses.

### Soundness
1

### Presentation
2

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
This paper introduces Matryoshka-MoE (M-MoE), a training pipeline designed to enable elastic MoE inference. 
The authors argue that fixed-*k* training limits the flexibility of MoE inference. 
A reduction in activated experts will lead to significant performance degradation in a well-trained MoE. 
The design of M-MoE is straightforward, involving the activation of varing numbers of experts during training.
M-MoE achieves competitive results across the full suite of specialist models under different activation setups.
The authors propose different M-MoE strategies and demonstrate that their layer-wise M-MoE is particularly effective. 
Extensive experiments are presented to support the authors’ claims and highlight the effectiveness of M-MoE.

### Strengths
- The research question addressed in this paper is interesting and meaningful, providing researchers with insights into advancing MoE inference efficiency.

- The idea of introducing Matryoshka routing into MoE training is straightforward and well-motivated.

- The authors perform extensive experiments to demonstrate the effectiveness of their M-MoE, including continual pre-training with 1T tokens for a 20B language model and from-scratch pre-training, which strengthen their claims.

### Weaknesses
- I have concerns regarding some of the experimental results, as they do not adequately support the authors' claims or demonstrate the effectiveness of M-MoE (See Q1 & Q2).

- The authors lack empirical evidence to show that M-MoE can truly improve MoE inference efficiency, raising doubts about its practical applicability in real-world scenarios (See Q3 & Q4).

### Questions
---
Q1: What's the performance when number of Activated Experts is set between 3 and 8 in Figure 3.

I suggest the authors provide a finer-grained analysis of activated expert counts to better illustrate the trend and shape of the performance degradation.

---
Q2: Can the authors also inlude the top-k sepecialist baseline results beyond Inf. k = 1 and its native activation count in Table 1 and 2 ?
(e.g Top-k (k=6), *Inf.k* = 2,4)

I have the following concerns regarding Table 1 and 2.

(1) The authors state that the Top-k baseline suffers a severe performance drop when evaluated with a different number of activated experts.
How does an MoE trained with 6 experts perform when evaluated with 2 or 4 activated experts? Is its performance comparable to that of your M-MoE variants?

(2) The contitual pre-training setup in Table 1 seems unusual, where only a single expert is activated during per-training.
What is the performance of M-MoE when more experts are activated during the pre-training phase?

--- 
Q3: What's the end-to-end training throughput of your M-MoE variants and top-k specialist baselines?

What is the actual training efficiency of M-MoE relative to the baseline methods?

Does M-MoE introduce additional complexity to the pipeline, thereby reducing compatibility with existing MoE training and deployment frameworks?

Could the authors include a discussion of the following points:（1）training throughput and (2) inference speed comparisons, to provide further clarity.

---
Q4: What's the loading balance of M-MoE models?

I am curious whether Matryoshka routing introduces a shortcut in MoE routing — i.e., some experts are more easily activated and consistently assigned large routing weights, which could explain the observed performance improvement over Top-1.

A comparison of load balance across experts would help clarify any potential loading imbalances and shed light on the shortcut issue.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an interesting setting for MoE models: the number of activated experts (K) is altered at inference time. The authors identify that the fixed Top-K training paradigm leads to severe performance degradation if K is changed, preventing the elastic trade-off between computational cost and model quality. Then, the author proposes some strategies to handle this and conducted corresponding experiments.

### Strengths
1. The author proposes a training framework to train a moe flexible to different K while inference, while achieving performance similar to top-K pretrained ones.
2. Many experiments have been done.
3. The elastic inference setting is interesting.

### Weaknesses
1. I’m not sure whether this method should be compared with dynamically-activated MoEs (i.e., those that allow the model to choose the computing budget at inference time, rather than training with top‑p but inferring with top‑k), or whether the authors want to emphasize the necessity of allowing humans, rather than the model, to choose the computing budget.  

2. Could you also report the perplexity or the loss curve, since benchmark results may vary due to randomness (especially in the from-scratch setting)?  

3. Have you tried continuing training the Top‑K model for a while? I mean, using the same training cost as M-MoE to train a Top‑K model, then applying a short period of continued training to obtain versions with different K values. (This is somewhat similar to your current setting, but I would prefer that the base model not be an MoE with top‑k = 1.)

### Questions
1. It is quite interesting that your M-MoE model achieves better performance than Top-k = 4 model with an average k < 4. Can you provide some insights or explanations?
2. I suggest focus on from-scratch setting more since the continue-training setting is a little wired (the base model is top-1 yet you continue it to increase topk)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Matryoshka Mixture-of-Experts (M-MoE), a training framework designed to enable elastic inference in sparse MoE language models. Standard Top-K routing results in sharp performance degradation when the number of activated experts is reduced at inference time. M-MoE mitigates this by randomizing the number of active experts (K) during training. The authors compare three levels of stochasticity — global batch, micro-batch, and layer-wise — and find that layer-wise performs best in achieving stable, elastic performance.
Experiments include continual pre-training from a 1T-token single-expert model (additional 80B and 208B tokens), showing that M-MoE maintains high accuracy across K=1–6.

### Strengths
1. Identifies a critical limitation of Top-K routing: sharp accuracy drop when reducing K.
2. Proposes a simple yet effective training method that yields elastic inference.
3. Demonstrates strong empirical validation across multiple training regimes (continual and from-scratch).
4. Provides clear pseudo-code and detailed experimental settings, supporting reproducibility.

### Weaknesses
1. Missing quantitative comparison of training throughput (FLOP/s, GPU utilization) between Top-K and M-MoE.
2. Conceptually incremental relative to previous Matryoshka and dynamic MoE literature.

### Questions
1. Could the authors provide explicit FLOP/s or wall-clock throughput comparisons with fixed-K training?

### Soundness
3

### Presentation
3

### Contribution
3
