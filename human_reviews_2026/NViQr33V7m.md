# Enhancing Generalization via Sharpness-Aware Trajectory Matching for Dataset Condensation

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Dataset condensation aims to synthesize datasets with a few representative samples that can effectively represent the original datasets. This enables efficient training and produces models with performance close to those trained on the original sets. Most existing dataset condensation methods conduct dataset learning under the bilevel (inner- and outer-loop) based optimization. However, the preceding methods perform with limited dataset generalization due to the notoriously complicated loss landscape and expensive time-space complexity of the inner-loop unrolling of bilevel optimization. These issues deteriorate when the datasets are learned via matching the trajectories of networks trained on the real and synthetic datasets with a long horizon inner-loop. To address these issues, we introduce Sharpness-Aware Trajectory Matching (SATM), which enhances the generalization capability of learned synthetic datasets by optimizing the sharpness of the loss landscape and objective simultaneously. Moreover, our approach is coupled with an efficient hypergradient approximation that is mathematically well-supported and straightforward to implement, along with controllable computational overhead. Empirical evaluations of SATM demonstrate its effectiveness across various applications, including in-domain benchmarks and out-of-domain settings. Moreover, its easy-to-implement properties afford flexibility, allowing it to integrate with other advanced sharpness-aware minimizers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduced the sharpness-aware minimizer into the bilevel-optimization-based dataset condensation to address the optimization issue from the complicated loss landscape. To reduce the computational overhead of the added optimization, two strategies are proposed to approximate the hypergradient calculation. Experiments across various settings show the effectiveness of the proposed solution to optimizes the bilevel objective function and sharpness simultaneously.

### Strengths
1. The optimization issue of the bilevel problem in dataset condensation is very important. This work studied this problem from the less-explored loss landscape view. 
2. The proposed strategies for mitigating the doubled computational cost of SAM are mathematically sound and practical.
3. The proposed solution achieves consistent improvements over state-of-the-art trajectory-matching competitors across standard benchmarks.

### Weaknesses
1. The novelty of the proposed solution is not large. The introduction of SAM into dataset condensation  in Section 4.1 seems a simple application of SAM. The truncated unrolling hypergradient is similar to k-step SGD in [1]
2. Although the proposed method is effective, the improvement compared with the SOTA methods is incremental.
3. Some latest SOTA methods for dfficient dataset distillation are missing [2,3,4].

[1] Meta Label Correction for Noisy Label Learning. AAAI 2021

[2] Provable and Efficient Dataset Distillation for Kernel Ridge Regression. NeurIPS 2024

[3] M3D: Dataset Condensation by Minimizing Maximum Mean Discrepancy. AAAI 2024

[4] Accelerating Dataset Distillation via Model Augmentation. CVPR 2023

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on dataset condensation (also known as dataset distillation), where the goal is to learn a small synthetic dataset such that models trained on it perform well on the real data distribution. The authors target methods that use trajectory matching (e.g. MTT) – i.e. they optimize synthetic data by matching the training trajectory of a network on synthetic data to the trajectory on real data. While these methods can produce high-fidelity condensed sets, they suffer from two major issues: the bilevel optimization with long unrolled training trajectories is extremely costly (time- and memory-intensive), and the outer-loop loss landscape becomes highly non-smooth and “sharp,” leading to poor generalization (e.g. the synthetic set overfits to the specific model or training path used in optimization).

To tackle these challenges, the paper introduces Sharpness-Aware Trajectory Matching (SATM). SATM integrates the concept of Sharpness-Aware Minimization into the trajectory matching paradigm, aiming to simultaneously minimize the trajectory-matching loss and the sharpness of the outer-loop loss.

A straightforward application of SAM in this setting would double the already heavy computation (since it requires computing gradients at a perturbed point), so the authors propose two efficiency strategies: (i) a truncated unrolling of the inner loop to approximate the hypergradient (reducing memory and computation by not backpropagating through the entire long trajectory at once), and (ii) trajectory reusing with gradual perturbations, which allows re-utilizing segments of the network’s training trajectory and the corresponding gradients for the sharpness calculation without starting from scratch each time.

They also derive a closed-form solution for adjusting the learning rate in the sharpness computation, which simplifies implementation and offers theoretical guarantees by bounding the error introduced by these approximations.

### Strengths
Experiments demonstrate that SATM yields significant improvements in generalization of condensed datasets. Models trained on SATM-produced synthetic sets not only perform well in-distribution but also transfer better to different architectures and out-of-domain tests, alleviating the overfitting to the training network that plagues other methods.

Notably, SATM outperforms prior trajectory matching approaches on standard condensation benchmarks and achieves noticeable gains on ImageNet-1K condensation, a challenging task where most existing methods fail to produce viable condensed sets.

It also runs faster than or on par with the most efficient recent method (TESLA) while maintaining similar memory usage, indicating that the sharpness-aware additions are cost-effective.

In summary, this work is novel in bringing flat-minima concepts to dataset condensation and demonstrates that doing so markedly boosts the versatility of the distilled data.

### Weaknesses
One concern is that the method introduces several hyperparameters (for the sharpness term, truncation length, perturbation scale, etc.) and added complexity, which may require careful tuning. Maybe i have missed, but i would appreciate if the authors provide complexity on the setup of hyper-parameters, and transferability of these values to other datasets and architectures.

One other concern is lack of extensive discussions with existing methods. 
For example, 
- https://arxiv.org/abs/2303.04449 discusses 1) sharpness and 2) generality of dataset distillation to other architectures. But, authors does not mention about this paper.
- https://arxiv.org/abs/2403.16028 discusses the relationship between dataset bias and dataset distillation. This is related to out-of-distribution generality of dataset distillation.

### Questions
Discussed in weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a novel dataset condensation technique that employs two hypergradient approximation strategies to address the significant computational overhead caused by sharpness minimization. SATM outperforms trajectory-matching-based competitors on various dataset condensation benchmarks under both in-domain and out-of-domain settings.

### Strengths
1. The paper is well written and easy to follow.
2. Dataset distillation/condensation is an important topic but still faces many challenges to be addressed.

### Weaknesses
1. The novelty of the method is limited. It primarily combines sharpness-aware optimization with MTT, without introducing a new data compression technique, only some modifications to the optimization steps.
2. It would be better if the authors compared their method with other approaches that also incorporate sharpness-aware minimization (SAM).
3. The paper lacks comparisons with several recent state-of-the-art methods, such as RDED [1], NRR-DD [2], and SRE2L [3], among others.

[1] On the diversity and realism of distilled dataset: An efficient dataset distillation paradigm. CVPR 2024

[2] Enhancing Dataset Distillation via Non-Critical Region Refinement. CVPR 2025.

[3] Squeeze, recover and relabel: Dataset condensation at imagenet scale from a new perspective. NeurIPS 2023.

### Questions
See weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper introduces “Sharpness-Aware Trajectory Matching (SATM)” , which focuses on optimization of the sharpness of the loss landscape and the objective simultaneously.
- This paper aims to address limitations of trajectory matching methods for dataset distillation.
- The paper has combined ideas from trajectory matching dataset distillation method such as MTT and minimization of sharpness of loss landscape.
- Two computational approximations to replace the costly calculations are introduced, namely “Truncated Unrolling Hypergradient (TUH)” and “Trajectory Reusing (TR)”.

### Strengths
1. Specific methods to deal with computational complexity are provided. Both methods are supported with theoretical bounds and practical implementation.
2. The paper has provided experimental results on various standard benchmark datasets such as CIFAR-10, CIFAR-100, Tiny ImageNet and ImageNet-1K (along with various subsets of ImageNet-1K).

### Weaknesses
**1. Limited baselines**

Recent advances in dataset distillation methods, including decoupled distillation, diffusion-based, and optimization-free dataset distillation methods, are not considered.
Below are some of the SOTA dataset distillation works, with which comparison should have been made to be more insightful.

- "Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective" (NeurIPS, 2023)
- Diversity-Driven Synthesis: Enhancing Dataset Distillation through Directed Weight Adjustment (NeurIPS, 2024)
- On the Diversity and Realism of Distilled Dataset: An Efficient Dataset Distillation Paradigm (CVPR, 2024) 
- Efficient dataset distillation via minimax diffusion (CVPR,2024)
- DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation (CVPR, 2025)

**2. The experiments are limited to (relatively simple) ConvNet architecture only**

The true impact of the proposed method cannot be analysed without considering more complex and standard architectures such as VGG, ResNet, ViT to name a few.

3. The improvements in generalization accuracy as shown in Table 2 are marginal (often < 1%).

4. Ablation studies are not provided regarding the impact of trajectory reusing and TUH. It would have been very useful to study how these approximations impact the generalization accuracy. 

5. The runtime cost as compared to the TESLA method is quite high despite the modest increment in accuracy, which weakens the argument of efficiency.

### Questions
- Please refer to the weaknesses  section of the review.

### Soundness
2

### Presentation
2

### Contribution
2
