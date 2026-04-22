# NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Optimization

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 2, 8

## Abstract
Discrete diffusion language models (dLLMs) have recently emerged as a promising alternative to traditional autoregressive approaches, offering the flexibility to generate tokens in arbitrary orders and the potential of parallel decoding. However, existing heuristic sampling strategies remain inefficient: they choose only a small part of tokens to sample at each step, leaving substantial room for improvement. In this work, we study the problem of token sampling order optimization and demonstrate its significant potential for acceleration. Specifically, we find that fully leveraging correct predictions at each step can reduce the number of sampling iterations by an order of magnitude without compromising accuracy. Based on this, we propose Neural Indicator Sampling (NI Sampling), a general sampling order optimization framework that utilize a neural indicator to decide which tokens should be sampled at each step. We further propose a novel trajectory-preserving objective to train the indicator. Experiments on LLaDA and Dream models across multiple benchmarks show that our method achieves up to 14.3$\times$ acceleration over full-step sampling with negligible performance drop, and consistently outperforms confidence threshold sampling in the accuracy–step trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies acceleration methods for masked diffusion sampling. While masked diffusion models accept efficient prallel sampling/unmasking of multiple tokens without significant performance drop, how much speedup we can achieve has been unknown and perople used heuristic methods to determine the sampling order and parallelization magnitude. This paper introduces Trajectory-Preserving-Order, which measures to how much extent we can reduce the redundunt function evaluations of masked diffusion without changing the one-by-one sampling trajectory. In math and programming benchmarks with the top-1 probability argmax sampler, the authors confirm that up to 24x acceleration is theoretically possible. By learning the Trajectory-Preserving-Order by a neural network (neural indicator learning), NI sampling then achievs up to 14x acceleration, outperforming the threshold-based heuristic.

### Strengths
This paper is well written, and offers an evidence that the optimization of sampling positions in masked diffusion can greatly accelerate the sampling process, *even without changing the output/trajectory*, through the empirical studies with Trajectory-Preserving-Order. This offers a deeper understanding of masked diffusion sampling from a novel angle. The neural indicator learning also has a straightforward learning signal and succeeds in actually accelerating masked diffusions.

### Weaknesses
- [W1] A primary weakness is that this method works only for almost deterministic sampling strategies and tasks such as argmax sampling and math/programming tasks. Thus it does not align well with the direction of reproducing randomness, such as unconditional generation and creative tasks requiring diversity.
- [W2] As the authors admit in Secion 7, the performance of NI sampler in this paper is far from that of Trajectory-Prserving-Order and closer to the heuristic Threshold sampler without any training. I'm not sure how difficult learning the Trajectory-Preserving-Order is in reality.

Typos etc:
- L70: gsampling -> sampling
- L106: I find many `\cdots` for sequences (e.g., $1, \cdots, R$), but the standard notation aligning with amsmath (when the separator symbols are low) is `\ldots` (e.g., $1, \ldots, R$). It is not at all essential.
- L182: minize -> minimize
- L209: Preversing -> Preserving (same typo at **many** other positions as well)
- L272: Indicato -> Indicator

### Questions
While I lean to acceptance, I suspect that the NI sampler just conducts semi-AR sampling (that is not a major issue though). To better understand the method, I have the following question:
- [Q1] Trajectory-Preserving-Order may contain poisitions with a random coincidence such as comma and period. However, NI sampling in my intuition, when generalized well, follows mostly the natural order of words. That can result in a gap between the two, which might be very difficult to learn. I would like to see which word is sampled in which step, in examples like Tables 9 & 10, together with the result of Trajectory-Preserving-Order.

### Soundness
3

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
3

### Summary
This paper explores the potential for reducing sampling steps in discrete Large Language Models (dLLMs) by merging backward steps wherever possible. To approximate this sampling strategy, the authors propose Neural Indicator Sampling (NI Sampling), which learns to identify which positions can be unmasked at each step.

### Strengths
- The idea of merging multiple sampling steps into a single step is simple yet presents an interesting direction for accelerating generation.
- The proposed merging strategy significantly speeds up sampling while maintaining performance, at least in the scenario described in Section 3.

### Weaknesses
**Concerns about Diversity of Generated Samples**
- The approach relies heavily on a pre-computed, deterministic trajectory. I am concerned that this may lead to reduced diversity in the generated samples, especially if the sampling algorithm is entirely deterministic (see my question below).
- Whether generation distribution given by the merging algorithm is close to that given by the full sampling in a distribution sense is not discussed.
- While the experiments evaluate generation performance in terms of accuracy and speed, they do not assess diversity (e.g., entropy used in LLM field), which is a crucial aspect of generation quality.
- The generation tasks on mathematical and code datasets are insufficient for evaluating diversity.

**Performance Limitations of NI Sampling**
- It is not clear what are missing components to fill the significant performance gap between NI sampling and ideal sampling presented in Section 3. The ablation study on network size suggests that this gap is not due to the network's expressive power. 
- Section 4.3 lists the network inputs, but it remains unclear what other components might be missing to fill the performance gap. It would be appreciated if insights into it can be more elaborated.

**Experiments**
- The proposed method is only compared with full sampling and the simple threshold-based method. However, there are related works [1,2] that should be discussed at least, even if they were originally proposed for vision domains.
- It would be beneficial to present Figure 4 and related figures as time vs. accuracy plots.

[1] Jose Lezama et al., "Improved Masked Image Generation with Token-Critic," ECCV 2022.  
[2] Jose Lezama et al., "Discrete Predictor-Corrector Diffusion Models for Image Synthesis," ICLR 2023.

**Typos**
- "gsample" in Line 70

### Questions
- Is the proposed method limited to deterministic reference (trajectory)?
- Is the sampling procedure after which positions will be unmasked at each step is deterministic (based on top-1 sampling)?
- Could you clarify the statement "which may be due to the variance in evaluation" in Line 376. The meaning is unclear.

### Soundness
1

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes NI Sampling, a learnable framework to accelerate dLLMs by optimizing the token sampling order.

Instead of relying on heuristic confidence-based thresholds, the method introduces a neural indicator that predicts which masked tokens can be safely revealed at each step. To train this indicator, the authors design a trajectory-preserving objective, ensuring consistency with reference sampling trajectories while achieving significant speedups.

Empirical results show up to 14× acceleration compared to full-step sampling with negligible performance drop across benchmarks such as GSM8K, MATH, HumanEval, and MBPP.

### Strengths
* **Learnable Scheduler**
  Improving the sampling efficiency of discrete diffusion models is an important and timely research direction. The idea of approaching this problem through a *learnable* mechanism for sampling order optimization is particularly interesting and potentially impactful.

### Weaknesses
* **Overfitting to Reference Trajectory**
  (Please correct me if I have misunderstood this point.)
  A major limitation of NI Sampling is that it appears to produce generations that overfit to the *reference trajectory* used during training. In this sense, it differs fundamentally from efficient sampling studies in *continuous* diffusion models, where speedup is achieved by reducing *numerical errors* of ODE solvers rather than altering the learned data distribution.

While the paper claims acceleration in sampling speed, this efficiency primarily stems from treating tokens as *conditionally independent*—i.e., ignoring inter-token correlations. In contrast, methods that preserve token correlations (e.g., [1]) achieve efficiency without such independence assumptions. Consequently, NI Sampling resembles trajectory-conditioned approaches such as [2], which restrict the model’s output distribution to a small subset of trajectories, potentially degrading sample diversity.

[1]: *Jump Your Steps: Optimizing Sampling Schedule of Discrete Diffusion Models*, [https://openreview.net/forum?id=pD6TiCpyDR](https://openreview.net/forum?id=pD6TiCpyDR)
[2]: *Beyond Autoregression: Fast LLMs via Self-Distillation Through Time*, [https://arxiv.org/abs/2410.21035](https://arxiv.org/abs/2410.21035)

### Questions
* **Diversity and Trade-off Analysis**
  As mentioned above, the main concern is that NI Sampling might overfit to the reference trajectory, thus harming sample diversity. To quantify this, please consider reporting:

  1. **Token entropy**, to measure the diversity of generated tokens.
  2. **pass@K**, to evaluate the diversity–accuracy trade-off more explicitly.

Additionally, the paper would benefit from a **theoretical analysis** of when and why merging steps (as done in the trajectory-preserving principle) does not harm the underlying distribution. In other words, even if the merged distribution deviates from the base model’s, a formal understanding of this trade-off would allow users to make informed decisions about when NI Sampling is appropriate. Currently, the method lacks such theoretical grounding, making it unclear under what conditions the distribution shift is acceptable.

[1]: *Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design*, [https://arxiv.org/abs/2402.04997](https://arxiv.org/abs/2402.04997)

### Soundness
1

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
This paper proposes NI Sampling, an efficient decoding framework for pre-trained diffusion LLMs that optimizes token sampling order with a tiny neural indicator predicting which masked positions can be simultaneously and safely revealed at each step. Labels for the indicator are derived from a trajectory-preserving criterion that merges steps when the current predictions already match a reference trajectory, turning correct predictions into parallel. On LLaDA-8B/1.5 and Dream-7B across GSM8K, MATH, MBPP, and HumanEval, NI Sampling delivers significant speedups over full-step sampling and threshold-based sampling with negligible accuracy loss.

### Strengths
- I enjoyed reading this paper. It's well-motivated. The trajectory-preserving case is great and clear, demonstrating the great potential of merging sampling steps for acceleration
- The proposed solution is conceptually simple and practical. A lightweight MLP as the indicator brings marginal extra cost.
- The proposed method achieves consistent and significant speedups across multiple benchmarks compared to threshold-based sampling and full-step sampling, with negligible accuracy loss.
- Detailed description of the setup of NI, analysis, and ablation studies, in addition to the main results.

### Weaknesses
I think this paper and its contributions could be further enhanced with more investigation into the transferability of the neural indicator:
- I got the difference between LLaDA models and Dream, but for models from the same family (e.g., LLaDA-8B and LLaDA-1.5 used in this paper), would it be possible to use a shared pretrained neural indicator?
- Would it be possible to use the same pretrained neural indicator for different generation window lengths?

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
