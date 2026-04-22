# On-the-Fly Data Augmentation via Gradient-Guided and Sample-Aware Influence Estimation

- Avg Score: 3.33
- Decision: Reject
- Scores: 0, 4, 6

## Abstract
Data augmentation has been widely employed to improve the generalization of deep neural networks. Most existing methods apply fixed or random transformations.
However, we find that sample difficulty evolves along with the model's generalization capabilities in dynamic training environments.
As a result, applying uniform or stochastic augmentations, without accounting for such dynamics, can lead to a mismatch between augmented data and the model's evolving training needs, ultimately degrading training effectiveness.
To address this, we introduce SADA, a Sample-Aware Dynamic Augmentation that performs on-the-fly adjustment of augmentation strengths based on each sample's evolving influence on model optimization.
Specifically, we estimate each sample’s influence by projecting its gradient onto the accumulated model update direction and computing the temporal variance within a local training window.
Samples with low variance, indicating stable and consistent influence, are augmented more strongly to emphasize diversity, while unstable samples receive milder transformations to preserve semantic fidelity and stabilize learning.
Our method is lightweight, which does not require auxiliary models or policy tuning. It can be seamlessly integrated into existing training pipelines as a plug-and-play module.
Experiments across various benchmark datasets and model architectures show consistent improvements of SADA, including +7.3% on fine-grained tasks and +4.3% on long-tailed datasets, highlighting the method's effectiveness and practicality.
Code will be made publicly available soon.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes a dynamic automatic data augmentation technique that adjusts the strength of various image transformations---both geometric and color-based---based on the difficulty of individual samples. The sampling strategy evolves during training using a combination of exponential moving average (EMA) and a sliding window mechanism. The authors conduct a fairly thorough comparison with existing automatic data augmentation methods, which are typically static. The code is not provided at submission time.

Disclaimer: I previously reviewed this paper for NeurIPS, where it was rejected. Except for a few added references, this submission is essentially identical to the original. None of the reviewers’ concerns appear to have been addressed, so I have not made any changes to my original review.

### Strengths
The empirical results appear promising. The idea of adapting data augmentation dynamically during training, rather than relying on a fixed strategy, is both intuitive and worth exploring further.

The paper provides a broad empirical comparison with other augmentation strategies.

### Weaknesses
1) The most significant concern is the lack of proper discussion regarding EntAugment (Yang et al. 2024b), a closely related prior work that also proposes a dynamic per-sample augmentation strategy during training:
   - Despite its clear relevance, EntAugment is not described properly in the submission.
   - All baseline numbers are copy-pasted from EntAugment without proper credit, rather than obtained in the context of this submission.
   - Performance metrics from EntAugment, which are comparable to those of the proposed method, are not included in the result tables. This omission is problematic (unless there is a good reason not to include them, but I cannot find any).

2) There are inconsistencies in the reported results compared to the literature, likely due to the lack of a standardized evaluation protocol:
  - For example, TrivialAugment reports 98.2 on CIFAR-10/SS and 84.3 on CIFAR100-WRN in published results, higher than the numbers reported in this submission.
  - Many baseline methods (e.g., DATA, TA, AA) rely on architectures like ShakeShake-26 and WRN-28-10. It’s unclear why this paper uses different variants (e.g., SS-32, WRN-50-2), making direct comparison difficult.

3) Given these discrepancies, the absence of publicly available code at the time of submission is a notable drawback.

4) The complexity analysis is misleading. The method introduces an additional factor L in computational cost, yet the paper presents the overall complexity O(NKL) as effectively O(N), downplaying the actual overhead.

5) Minor clarity issues are present. For instance, Figure 1 refers to “inverse sampling difficulty,” a term that has not been defined at that point in the paper, making it hard to interpret

### Questions
I found the lack of comparison/discussion with EntAugment very problematic on many aspects described above. Unless a convincing explanation is given, I will vote for a reject.

### Soundness
2

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
3

### Summary
The paper introduces a sample-aware data augmentation strategy that dynamically adjusts the augmentation strength for each sample. This adjustment is based on estimating a sample's influence by projecting its gradient onto the model update direction. Specifically, the method applies stronger augmentation to samples exhibiting low variance and weaker augmentation to those considered unstable. This augmentation process is performed on-the-fly, and experimental results demonstrate its efficacy across various classification tasks.

### Strengths
- The core idea of dynamically adjusting augmentation strength based on gradient information is both logical and potentially powerful.
- The work includes a comprehensive set of experiments on classification tasks, covering various settings such as closed-set and open-set, and different k-shot scenarios.

### Weaknesses
- The paper does not sufficiently detail the computational overhead. Since the gradient must be calculated at every optimization step to determine the augmentation strength, a clear analysis of the computational complexity and the resulting wall-clock time overhead during training is necessary.
- The experimental results show varying degrees of advantage: a noticeable gap on Tiny-ImageNet, a moderate improvement on CIFAR-10, and almost similar test accuracy for ImageNet-1k (Table 7). Given the potential training time overhead, the marginal benefit on large-scale datasets like ImageNet-1k is not sufficiently advantageous to justify the added complexity.
- The method's performance appears to be highly sensitive to the choice of hyperparameters, specifically the window size and decay factor. Although the optimal window size shows a consistent decreasing trend, the optimal decay factor does not exhibit a clear tendency, which raises concerns regarding the robustness of the method. This suggests a user might need to perform an extensive grid search for the decay factor whenever the experimental setting (e.g., training data, classifier architecture) is changed.

### Questions
- Given the variety of augmentations (e.g., geometric, color), is the same decay factor applied universally across all categories of augmentation, or are different decay factors employed for different augmentation categories?
- Would the proposed sample-aware augmentation strategy offer benefits when applied to image generation tasks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SADA, a plug-and-play augmentation scheme that adapts per-sample augmentation strength on the fly using training-dynamics signals. At each step, the method first estimates a sample’s influence by projecting its gradient onto the accumulated model-update direction, and then measures stability as the temporal variance of this influence over a short window (EMA-smoothed). Stable (low-variance) samples receive stronger augmentations; unstable (high-variance) samples receive milder ones to preserve semantics. The influence is made efficient via a first-order loss-difference approximation. A bound is sketched that links SADA to reduced generalization complexity via a per-sample sensitivity term. Experiments on CIFAR-10/100, Tiny-ImageNet, ImageNet-1k, several fine-grained datasets, and long-tailed benchmarks report consistent gains and favorable accuracy-cost tradeoff.

### Strengths
1. The paper proposes an intuitive yet effective data augmentation approach that adapts per-sample augmentation strength on the fly. The methods takes sample variance into consideration via gradient projection, and avoids intense per-sample computation via a series of approximation, making it practical in a wide range of classification tasks.

2. The proposed method shows consistent performance improvement in extensive experimental settings. The method outperforms other methods on CIFAR-10/100, Tiny-ImageNet, and is competitive on ImageNet-1k. The method also achieves improvements on transfer-learning, fine-grained and long-tail datasets.

### Weaknesses
1. The proposed method introduces a principled approach for sample-aware augmentation. However, in each step a random augmentation operation is selected. Different image transformation process can impact the sample at different levels, which may interfere with the delicately designed augmentation strength.

2. The method introduced hyper parameters like window size and decay factor. Their values seem to be set based on experimental guidance, which probably need to be tuned individually for different tasks (standard, transfer or long-tail classification) and datasets.

### Questions
1. How does the method perform with controlled randomness for image transformation operations? Or is the random selection of operations an important factor in the approach?

2. How does the method affect the training dynamics? In particular, would the sample difficulty score in Figure 1 be distributed more evenly at the latter stage of training?

### Soundness
3

### Presentation
3

### Contribution
3
