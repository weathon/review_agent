# Enhancing Dataset Distillation with Concurrent Learning: Addressing Negative Correlations and Catastrophic Forgetting in Trajectory Matching

- Avg Score: 5.33
- Decision: Reject
- Scores: 5, 6, 5

## Abstract
Dataset distillation generates a small synthetic dataset on which a model is trained to achieve performance comparable to that obtained on a complete dataset. Current state-of-the-art methods primarily focus on Trajectory Matching (TM), which optimizes the synthetic dataset by matching its training trajectory with that from the real dataset. Due to convergence issues and numerical stability, it is impractical to match the entire trajectory in one go; typically, a segment is sampled for matching at each iteration. However, previous TM-based methods overlook the potential interactions between matching different segments, particularly the presence of negative correlations. To study this problem, we conduct a quantitative analysis of the correlation between matching different segments and discover varying degrees of negative correlation depending on the image per class (IPC). Such negative correlation could lead to an increase in accumulated trajectory error and transform trajectory matching into a continual learning paradigm, potentially causing catastrophic forgetting. To tackle this issue, we propose a concurrent learning-based trajectory matching that simultaneously matches multiple segments. Extensive experiments demonstrate that our method consistently surpasses previous TM-based methods on CIFAR-10, CIFAR-100, Tiny ImageNet, and ImageNet-1K.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper tackles the problem that matching between different segments of the training trajectory may be negatively correlated. Existing methods use Trajectory Matching (TM), which optimizes the synthetic dataset by matching its training trajectory with the real dataset. However, they overlook the negative correlation between different trajectory segments, leading to performance degradation. The authors propose a concurrent learning-based TM method that matches multiple segments simultaneously, reducing errors. With exhaustive experiments, their approach outperforms previous methods across several benchmark datasets.

### Strengths
[1] The writing of paper is good.

[2] The analysis of the negative correlation on accumulated trajectory error is comprehensive. For example, the author clearly specifies the accumulated error to initialization error and matching error. Afterwards, the authors calculate the correlation to validate the phenomenon.

[3] The proposed cocurrent training methods is reasonable. The experimental results validate the effectiveness of the proposed methods.

### Weaknesses
[1] The description of the accumulated trajectory error is not intuitive. The notations defined in Sec. 4.1 are complex and there is no graphic illustration of these notations.

[2] The explanation of the negative correlation is not clear, especially when the IPC is low. In addition, the roles of training dynamics are not validated with experimental evidence.

[3] The novelty of the concurrent training to tackle the problem is rather limited. While reasonable, the author only simply leverages the multitask learning to tackle the problem. Since the improvement compared to previous state-of-the-art methods is not very significant, the contribution is not above the acceptance bar of ICLR.

### Questions
Please see the weakness.

### Soundness
2

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
The paper analyzes the challenges of dataset distillation, i.e., negative correlation between different segments of a trajectory to match and catastrophic forgetting problem. To address this, the authors formulate trajectory matching as a continual learning problem and propose a method called Concurrent Training-based Trajectory Matching (ConTra). It employs multi-task learning to simultaneously match multiple segments, rather than sequential learning used in previous works. Experimental results show that ConTra consistently outperforms existing trajectory matching methods on various datasets, thus demonstrating its ability to minimize accumulated matching errors and achieve lossless condensation.

### Strengths
1.	The analysis on negative correlation between different trajectory segments is detailed and solid, enriching the discourse on continual learning.
2.	The idea of utilizing concurrent learning to tackle negative correlation is simple but novel.
3.	Extensive experiments on multiple datasets and downstream tasks are quite convincing.

### Weaknesses
1.	The paper needs to be further polished. There are numorous typos, such as line 140 ‘a expert’->’an expert’, line 457 ‘s the range’->’so the range’.
2.	It would be more intuitive if there is a figure to show the differences/advantages of your proposed ConTra compared to the previous TM methods. 
3.	The paper does not test the proposed method using different distillation and evaluation model size.

### Questions
1.	Is the proposed method sensitive to the model size used for distillation and evaluation? 
2.	Does the distillation training time decease when using concurrent learning compared to sequential learning?

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
4

### Summary
This paper addresses dataset distillation, aiming to create small synthetic datasets that enable models to achieve comparable performance to training on complete datasets. Due to convergence and stability challenges, Trajectory Matching methods typically match only segments of training trajectories, but they overlook negative correlations between different segments. The authors quantitatively analyze these correlations, finding that negative correlations can increase trajectory error and lead to catastrophic forgetting. To address this, they propose a concurrent learning-based TM approach that matches multiple segments simultaneously. Experiments show that this method outperforms previous approaches across various datasets.

### Strengths
1. This paper clearly identifies the problem, providing both theoretical and experimental analysis to demonstrate the existence of negative correlations between trajectory segments.

2. All the discussions in the paper are clear and straightforward.

3. The experiments contains all the necessary components with enough discussion

### Weaknesses
1. This paper includes a theoretical analysis of negative correlation; however, the theory presented in Section 4.1 primarily illustrates that training errors can accumulate across segments. While this is an important observation, it is not directly related to the negative correlation itself. A more detailed exploration of how these relate to negative correlation would strengthen the theoretical foundation of the proposed method.

2. The novelty of the proposed approach appears to be somewhat limited. To mitigate negative correlation, the method simply trains multiple segments concurrently, which is a strategy commonly employed as a baseline in continual learning scenarios.

3. The performance gains observed in the experiments are somewhat disappointing, particularly when considering the increased computational requirements associated with the proposed method. Given the additional complexity introduced, one would expect a more substantial improvement in performance to justify the costs involved (especially when compared to DATM). This raises questions about the effectiveness of the approach in real-world applications.

### Questions
See strengths and weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
