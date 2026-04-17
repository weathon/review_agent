# Train on Validation (ToV): Fast data selection with applications to fine-tuning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
State-of-the-art machine learning often follows a two-stage process: $(i)$ pre-training on large, general-purpose datasets; $(ii)$ fine-tuning on task-specific data.  In fine-tuning, selecting training examples that closely reflect the target distribution is crucial. However, it is often the case that only a few samples are available from the target distribution. Existing data selection methods treat these target samples as a validation set and estimate the effect of adding or removing a single sample from the training pool by performing inference on the validation set.

We propose a simpler and faster alternative that inverts the usual role of train and validation: we perform inference on the training pool before and after fine-tuning on the validation set. We then select samples whose predictions change the most. Our key insight is that the training samples most affected by fine-tuning on a small validation set tend to be the most beneficial for reducing test loss on the target distribution. Experiments on instruction tuning and named entity recognition tasks show that, in most cases, our method achieves lower test log-loss than state-of-the-art approaches. We support our findings with theoretical analysis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a method for selecting task-specific fine-tuning data using a validation set representative of the target distribution. The approach finetunes a model on the validation set and performs inference on candidate training samples, selecting those whose losses change the most. Experiments demonstrate that the method outperforms uncertainty-based selection and a recent task-specific selection method. Finally, the authors further provide a theoretical justification for its effectiveness.

### Strengths
- Training on the validation set introduces a new perspective in task-specific data selection.
- The method is conceptually simple and readily implementable.
- A theoretical analysis is presented, providing provable error bounds.
- The manuscript is well structured and written with clarity.

### Weaknesses
- The paper claims that the proposed method avoids the computational burden of computing influence functions, but does not provide any empirical or theoretical evidence. It is unclear how expensive the proposed method is compared to training on the full set or influence-based selection.
- Only log loss is reported in the evaluation. For instruction-following and NER tasks, log loss does not adequately capture output quality. Metrics such as F1 (for NER) or semantic similarity and human preference scores (for instruction-following) would provide a more meaningful assessment.
- A gap exists between the empirically evaluated method and the theoretical formulation, and the paper does not make the connection between them explicit.

### Questions
- What is the purpose of fine-tuning on a small random subset before fine-tuning on the validation set? Additionally, why is it necessary to repeat this process for multiple iterations?
- In exp 3 and NER tasks, why does LESS perform so badly as a task-specific selection method?

### Soundness
2

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
5

### Summary
This paper studies how to efficiently select training samples so that the resulting model performs better on a small validation set drawn from a target distribution. The authors observe that standard training-sample scoring methods rely on per-sample gradient computations over both the training and validation sets. To reduce this cost, they propose an approximation based on the change in training samples' loss after performing a gradient update using the full validation set. Two algorithms are developed based on this insight—the second being theoretically guaranteed but empirically less effective. Experimental results demonstrate consistent performance improvements over two baseline methods.

### Strengths
1. The motivation of this paper is strong, especially for scenarios where some existing packages do not support per-sample gradient computation. 
2. The observation of train–validation symmetry is interesting. It is also a good choice to put this point in the introduction section, as it helps readers quickly grasp the core insight of the paper.
3. The proposed method could be applied to certain LLM-related tasks, such as instruction tuning and named entity recognition.
4. The theoretical analyses and experiments look good and convincing.

### Weaknesses
1. Although the train–validation symmetry is an interesting observation, it is established under the assumption of independence of x. In Proposition 1, the Hessian matrix is expected to be of order 1, which supports the utility of the proposed estimator. However, it would be beneficial to discuss this point in more detail, particularly in relation to influence functions and data Shapley. I understand that Strategy 2, which increases diversity via random sampling, can heuristically mitigate this issue to some extent.
2. The proposed method can also be interpreted as a task-dependent data pruning technique. From this perspective, it would be helpful to include a discussion comparing it with representative works in this area.
3. The original problem can be formulated as a bi-level optimization problem, typically with continuous sample weights. I did not look into whether Xia et al. (2024) is following this idea, but the current baseline choices seem somewhat weak in comparison.
4. It would be better to experimentally quantify the efficiency (how fast) of the proposed method, as this is claimed as the key contribution.

### Questions
1. Instead of selecting important samples, why not consider applying target-aware domain generalization techniques here, especially in the context of LLM tuning?
2. Since both training and validation sets are accessible, how about directly measuring sample similarity between them to guide selection?

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
Train on Validation (ToV) is a simple data-selection method for fine-tuning. Leveraging from a principle they drove in the paper, train–validation symmetry, they built a fast scoring rule that does not depend on the per-example gradients, or Hessian-vector product, as opposed to prior works. They simply compute each pool example’s loss before and after a brief train-on-validation step and rank by the loss drop. Through experiments on instruction tuning and named entity recognition (NER), they show that ToV typically beats random/uncertainty and often outperforms a strong baseline (LESS), despite requiring less machinery.

### Strengths
1 - The core idea is elegantly simple. By exploiting a symmetry between training and validation loss changes (Eq. 6), the method derives a tractable influence proxy that requires only two loss evaluations and a brief “train-on-val” phase, thereby avoiding N validation passes or per-example gradient computations.

2 - They propose two algorithms (Method A/B) and theoretically connect their scores to linearized influence in a gradient descent setting, framing ToV as an influence-style selector that avoids gradients and HVPs.

3 - Empirically, ToV consistently outperforms random and max-uncertainty baselines and, in most cases, exceeds LESS on instruction tuning.

### Weaknesses
1 - Because efficiency is central to the paper’s contribution, the claim of practicality should be backed by quantitative evidence, wall-clock time, memory usage, and forward/backward pass counts on identical hardware. LESS has warm-up and gradient-store overhead, whereas ToV involves repeated validation fine-tuning and two full loss sweeps per cycle. A detailed cost comparison would make the “fast” claim more credible.

2 - Since LESS explicitly warns that loss and accuracy may not align in LLM evaluation, relying solely on test log-loss in ToV is limiting. Including complementary metrics, such as accuracy, MMLU-style subsets, or task-specific measures like exact-match or F1, would strengthen the evidence that lower log-loss corresponds to better task quality.

3 - For completeness, ToV should be compared with at least one alignment-based selector (e.g., importance resampling) and a datamodel-style [1] to evaluate whether its advantage persists when competing methods also avoid per-example gradients.

[1] Park et al, "Trak: Attributing model behavior at scale" ICML 2023

### Questions
1 - Repeated training on the validation set during scoring raises concerns about overfitting the selection policy. Please clarify how ToV prevents overfitting to $Z^{val}$, and whether performance on the test set deteriorates as the number of ToV cycles increases.

2 - It would be helpful to discuss regimes where the train-on-val = train-on-x symmetry assumption does not hold. Reporting these failure modes or limitations would add practical value and clarify the boundaries of ToV’s effectiveness.

3 - The shift from token-level to sequence-level selection may introduce biases stemming from uneven sequence lengths, as discussed in LESS. Please clarify how ToV accounts for or corrects this bias.

4 - What is the effect of base subset size $m$?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper tackles data selection from a large training pool, with small datasets for validation and test. The proposed method efficiently scores each example from the training pool by how much one GD step on it contributes to lower validation loss, only with gradients on samples from the validation dataset, not from the training pool, by a clever trick based on first-order Taylor approximation. In section 2, experiments with tasks like instruction tuning and NER show that the proposed method (Method A) is more effective than baselines with respect to log-likelihood loss. Theoretical analysis is provided for a variant of the proposed method (Method B) in Section 3, justifying the proposed method from an asymptotic viewpoint.

### Strengths
- The problem addressed in this paper, data selection from a large training pool with a small validation dataset, is important for real-world fine-tuning.
- The derivation of the proposed method, which eliminates the need for computing gradients on the training pool, is straightforward and effective.
- Experimental results indeed show the effectiveness of the proposed method (Method A) compared to baselines in log-likelihood loss.
- Theoretical results provide further justification for their approach beyond just Taylor approximation, based on the asymptotic analysis.

### Weaknesses
- The main trick in this paper, i.e., swapping gradients between training and validation data in computing the difference in validation loss after one or multi-steps gradient descent, is not novel itself. Such a trick have already appeared in [Liu et al. 2019] and [Savani et al. 2025] for example. Nevertheless, as far as I know, its application to data selection may be novel.
- I have some concerns on the experimental design in Section 2. While the experiments report only log-likelihood loss, especially in instruction tuning, it does not imply actual improvements for downstream tasks and the instruction-following capability. Also, the choice of the NER task is not reasonable since entity classification for each token is less useful in real world and there are more suitable tasks than NER.
- There is a non-negligible mismatch between the experimental results (focusing on Method A) and the theoretical results (focusing on Method B). Even if Method A is empirically better than Method B, I think Method B should be mainly evaluated also in experiments (or Method A should be mainly analyzed theoretically) for consistency throughout the paper.
- The paper title `Train on Validation` is somewhat misleading. The paper just proposes data selection using gradients on validation data, and actual training is mainly performed on the selected data from the training pool.

[Liu et al. 2019] DARTS: Differentiable Architecture Search (ICLR'19)

[Savani et al. 2025] Antidistillation Sampling, https://arxiv.org/abs/2504.13146

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
