# CompeteSMoE - Statistically Guaranteed Mixture of Experts Training via Competition

- Decision: Reject
- Scores: 8, 6, 4

## Abstract
Sparse mixture of experts (SMoE) offers an appealing solution to scale up the model complexity beyond the mean of increasing the network's depth or width. However, we argue that effective SMoE training remains challenging because of the suboptimal routing process where experts that perform computation do not directly contribute to the routing process. In this work, we propose competition, a novel mechanism to route tokens to experts with the highest neural response. Theoretically, we show that the competition mechanism enjoys a better sample efficiency than the traditional softmax routing. Furthermore, we develop CompeteSMoE, a simple yet effective algorithm to train large language models by deploying a router to learn the competition policy, thus enjoying strong performances at a low training overhead. Our extensive empirical evaluations on both the visual instruction tuning and language pre-training tasks demonstrate the efficacy, robustness, and scalability of CompeteSMoE compared to state-of-the-art SMoE strategies. We will publish the implementation upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose CompeteSMoE, a SMoE training algorithm based on a competition mechanism, aimed at addressing the suboptimal issue of separation between expert computation and routing decisions in the routing process of traditional SMoE. Inspired by the Winner-take-all principle, its core idea is to activate all experts and select the Top-K experts based on their neural responses. Theoretically, it is proven that this mechanism has better sample efficiency and convergence rate than traditional softmax routing. Experiments are conducted on a wide range of benchmarks, demonstrating the effectiveness of the scheme.

### Strengths
1 The benchmarks covered in the experiments are relatively comprehensive, including both vision and language models of different types, providing strong support for the generalizability of the scheme.

2 The authors provide detailed hyperparameter configurations, making the reproducibility of the method convincing.

3 It provides statistical guarantees for the competition mechanism in SMoE. Through the convergence analysis of Gaussian MoE models, the convergence rate of the Total Variation distance for density estimation is derived, and a lower bound is established based on Voronoi loss, filling the gap in the theoretical analysis of the competition mechanism.

### Weaknesses
1 Regarding the hyperparameters ω and A used in this manuscript, what is their hyperparameter sensitivity for LLMs of different sizes?

2 How does the change in the total number of experts and the number of activated experts affect the method proposed in this paper?

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CompeteSMoE, a new approach to training the routing module in Sparse Mixture of Experts (SMoE) models. The method builds on a competition mechanism inspired by the winner-take-all principle. Periodically during training, all experts are activated for each token, and routing decisions are made based on the magnitude of their neural responses. These competition-based assignments serve as supervision signals for the router, which is trained to approximate the winner-take-all routing behavior. The authors propose a practical algorithm with scheduled router training and demonstrate promising empirical performance on both visual instruction tuning (VIT) and language pretraining tasks, outperforming several existing SMoE variants.

The breadth of analysis is satisfactory, though the language pretraining section would benefit from additional experiments that show whether the efficacy of CompeteSMoE holds up to scale, as well as the most popular modern architectural choices (more in Weaknesses  and Questions)

### Strengths
- The paper has a clear motivation and a relatively straightforward implementation of the proposed idea.  
- The presentation is clear and easy to follow.  
- The formulation of periodical winner-takes-all for SMoE routing decision, alongside having the router emulate those deisions appears novel, to the best of my knowledge.  
- Experiments cover both vision and language domains.  
- The paper includes thorough ablations analyzing the individual contributions of the competition mechanism and the diversity loss.  
- The experiment that rotates routing choices to test whether the model benefits from routing decisions or behaves randomly is an interesting addition.  
- The results are generally positive, though the effect sizes on benchmarks are small.  
- The analysis of wall-clock time and peak memory during training shows minimal overhead for the proposed method.

### Weaknesses
- The language pretraining setup uses 13B tokens for a 1B-parameter model. According to the Chinchilla rule of thumb (around 20 tokens per parameter), this corresponds to a moderate undertraining regime, while many modern models are heavily overtrained. See e.g. [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- The chosen configuration of `topk = 8` out of 24 experts is not the most typical. Recent trends favor sparser settings where only a few experts are active out of a large total. An ablation over different sparsity levels would make the empirical evaluation more complete.  
- The observed effect sizes on benchmarks are small. The improvements are more consistent in the vision domain, but those experiments rely on upcycled models, which complicates the interpretation relative to the more straightforward from-scratch language pretraining setting.  
- Limited ablation of hyperparameters: while ad-hoc guidelines for α, β, γ, and Amax are provided, the robustness of results to these choices is not extensively analyzed in the main text.  
- Minor issues with typos and unclear formulations (e.g., “200” instead of “200K” around line 1545, based on the supplementary code).

### Questions
- To strengthen the paper on the language pretraining front:  
  1. It would be helpful to investigate how the efficiency of CompeteSMoE changes with training horizon, model size, and Mixture-of-Experts architecture configurations (total number of experts and number of selected experts).  
  2. The effect size on benchmarks is small, and given the scale of experiments it's unclear how much of the variation is noise. It would be very informative to see evaluation benchmarks, training loss and validation loss trajectories throughout the entire training run, ideally for different configurations from suggestion 1.  
- Why was the diversity loss applied during pretraining from scratch, since the model was not upcycled?  
- Could the authors clarify the meaning of the following passage (lines 129–134)? It is somewhat unclear how joint learning of the task loss and competition policy alleviates the sample inefficiency issue or why this makes training feasible on limited hardware.
- It would be very valuable to understand the impact of frequency of competition on model quality. If memory is an issue, the authors can consider training a smaller model and thus being able to vary the competition probability up.
- Teaching the router to pick magnitude-maximising experts biases the model toward bigger activations. It would be very interesting to see the evolution of mean magnitude of tokens in the residual stream after each MoE layer, and how it evolves over time, and how all that changes from baseline SMoE to CompeteSMoE. The reason to care about this is that large activations can be difficult to quantize, which comes up in low-precision training.
- The paper would benefit from an analysis of the evolution of routing scores through training

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel competition-based training mechanism for SMoE models that achieves faster convergence than traditional routing. Building on this idea, the authors develop CompeteSMoE, a method that leverages competition to enhance SMoE training. To support their claims, the paper provides a theoretical convergence analysis of the proposed approach. Finally, experiments are conducted on popular vision and text benchmarks, complemented by an additional analysis of routing behavior.

### Strengths
Despite addressing the well-researched subfield of routing in SMoE, the authors successfully identify an original and interesting perspective on this topic. The paper is well-written and easy to follow.  A significant strength is that the authors ground their methodological claims in theoretical analysis. Furthermore, the proposed method is evaluated across both images and text, which strongly supports the paper's conclusions.

### Weaknesses
The claim in the abstract regarding a scalable method is not substantiated in the main text, as the experiments were conducted on a very limited scale. To support the scalability claim, additional experiments involving larger datasets and more diverse conditions would be necessary.

### Questions
- I am not convinced by ECR metric (Section 5.2.2 b). Intuitively, I would like router network to have a possibility to adapt to updated experts till the end of the training process. Why the authors claim that it is actually good, that the rate decaying faster?

- Section 4.2 does not appear to be directly relevant to the main objectives of the study.

- Upcycling was used as a means to bypass the costly pre-training phase. However, I believe that conducting smaller-scale experiments involving pre-training would likely yield more reliable insights.

Minor comment:
- No units in Table 4 for Train / Infer

### Soundness
3

### Presentation
3

### Contribution
2
