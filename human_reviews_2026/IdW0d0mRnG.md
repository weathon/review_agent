# Heads collapse, features stay: Why Replay needs big buffers

- Decision: Accept (Poster)
- Scores: 6, 8, 8

## Abstract
A persistent paradox in continual learning (CL) is that neural networks often retain linearly separable representations of past tasks even when their output predictions fail. We formalize this distinction as the gap between *deep* (feature-space) and *shallow* (classifier-level) forgetting. We reveal a critical asymmetry in Experience Replay: while minimal buffers successfully anchor feature geometry and prevent deep forgetting, mitigating shallow forgetting typically requires substantially larger buffer capacities.
To explain this, we extend the Neural Collapse framework to the sequential setting. We characterize deep forgetting as a geometric drift toward out-of-distribution subspaces and prove that any non-zero replay fraction asymptotically guarantees the retention of linear separability. Conversely, we identify that the ``strong collapse'' induced by small buffers leads to rank-deficient covariances and inflated class means, effectively blinding the classifier to true population boundaries. By unifying CL with out-of-distribution detection, our work challenges the prevailing reliance on large buffers, suggesting that explicitly correcting these statistical artifacts could unlock robust performance with minimal replay.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The author developed an asymptotic framework to analyze feature geometry with and without replay buffers. They demonstrate that replay can reliably alleviate deep forgetting (loss of feature separability), but does not alleviate shallow forgetting (misalignment between classifier weights and features). This work extends NC theory to multi head CL settings, characterizes the effects of buffer size and weight decay, and establishes a theoretical connection between CL and OOD. The empirical results of CIFAR100, Tiny ImageNet, and CUB-200 validate the theoretical findings.

### Strengths
Extends Neural Collapse analysis to continual learning and multi-head architectures—an unexplored direction. Uses asymptotic analysis and connects NC with OOD theory in a rigorous manner. Establishes a bridge between NC, CL, and OOD detection, enriching all three research domains.

### Weaknesses
While conceptually strong, it provides limited actionable guidance for improving CL performance.

### Questions
Why do we focus on discussing Multi Head Models? Is this model commonly used in modern continuous learning and multi task learning? Do you analyze whether the purpose of this model is to increase workload or has practical significance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an asymptotic analysis of replay-based continual learning under the neural collapse phenomenon. The authors study how replay buffers affect shallow and deep forgetting. Empirically, they find that replay-based continual learning effectively mitigates deep forgetting but still suffers from shallow forgetting even when the replay buffers are large. They then use Neural Collapse theory to analyze the limiting geometry of features and heads in three continual learning setups. They also identify a connection between continual learning and OOD detection, showing that under weight decay, the distribution of OOD inputs converges to a degenerate null distribution. They also show the effect of replay in their framework, demonstrating that deep forgetting is not mitigated by the model with small replay buffer because of the approximation error when using the buffer distribution to approximate the true class distribution.

### Strengths
1. This work proposes a novel framework for replay-based continual learning. The results and implications are meaningful and helpful to the community. 
2. The authors provide a sufficient and comprehensive theoretical analysis for replay-based continual learning, considering three different setups and showing the effect of replay.
3. The empirical study is consistent with the theoretical findings, across both real-world and simulated datasets.
4. The work is well structured and easy to follow for the readers.

### Weaknesses
1. In Theorems 1, 2, and 3, it seems that $\nu = 1 - \eta \lambda$ is required to be non-negative or greater than $-1$, but I do not find any explicit condition on $\nu$.

2. The explanation of why replay cannot strongly mitigate shallow forgetting is not convincing to me. The authors argue that the approximation error is the key reason, but there is no formal result to support this claim, which limits the contribution of this paper.
 
3. In the experimental results, the authors only vary the buffer size from $0\\%$ to $10\\%$. This seems insufficient to support their theoretical findings.

4. There are some typos and inconsistencies:
   1. Lines 139, 141. Two citations are missing.
   2. Line 315, "Theorem 6" should be "Theorem 2."
   3. Line 987, "class-il", "domain-il", and "task-il" should be written as "CIL," "DIL," and "TIL," consistent with other figures. Moreover, the task indices in Figure 12 should be positive integers.

5. The term “balanced replay” is unclear. I think “balanced replay” refers to the replay buffer being sampled in a balanced manner from the training set, rather than the buffer size being equal to the size of the training data.

### Questions
1. How to understand the theoretical results when $\eta \lambda \geq 2$?
2. Why are experiments conducted only for buffer sizes varying from $0\\%$ to $10\\%$?
3. What is the precise meaning of "balanced replay"?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper analyzes how the replay buffer in continual learning scenarios influences model forgetting, distinguishing between shallow and deep forgetting. It further investigates these phenomena within the Neural Collapse framework, examining the geometric structure of the feature space and supporting the analysis with empirical results.

### Strengths
The paper is well written and clearly organized, with a pleasant and coherent flow of discussion. The topic addressed is novel and engaging. Moreover, the theoretical analysis is insightful and clearly explained.

### Weaknesses
While the theoretical discussion is sound and convincing, I have some concerns regarding the empirical analysis. First, it is unclear why the authors chose ResNet and ViT as reference models. It seems that the selected architectures could significantly influence the observed behaviors and results. If this is the case, the authors should explicitly discuss this aspect. Otherwise, a justification of why the chosen architectures do not affect the outcomes should be provided.  Along the same lines, the rationale behind considering both pretrained and from-scratch models is not entirely clear. In the case of pretrained models, it would be important to explain how the initialization was adapted to the continual learning setting, as mentioned in Section 1.1. Additionally, the discussion in Section 3.3.2 highlights the effect of weight decay, but the influence of other hyperparameters and architectural choices remains unexplored. Given their potential impact, especially in the context of deep forgetting, this omission seems non-negligible. The authors should include a discussion addressing this point to provide a more comprehensive understanding of the empirical results. Another aspect that would benefit from clarification is the adoption of the Neural Collapse (NC) framework. The authors should briefly discuss possible alternative frameworks and justify the choice of NC in this context.
Finally, the discussion on the distinction between multi-head and single-head settings could be improved by adding a short introductory explanation earlier in the paper to help readers unfamiliar with these concepts. In addition, Section 3 contains some citation issues, where “?” symbols appear instead of proper references, and these should be corrected.

### Questions
- Could the authors clarify the rationale behind choosing ResNet and ViT as reference architectures, and discuss how this choice might influence the observed behaviors and results?

- How were pretrained models adapted to the continual learning setting, and what motivated the comparison between pretrained and from-scratch training approaches?

- Beyond weight decay, have the authors examined the influence of other hyperparameters or architectural choices on the empirical results, particularly in relation to deep forgetting?

- What motivated the adoption of the Neural Collapse framework, and could the authors discuss potential alternative frameworks or justify why NC is particularly suitable for this analysis?

### Soundness
3

### Presentation
4

### Contribution
3
