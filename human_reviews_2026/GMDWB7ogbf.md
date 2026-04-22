# Achieving Noise Robustness by additive normalization of labels

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 6, 6, 2, 2

## Abstract
As machine learning models scale, the demand for large volumes of high-quality training data grows, but acquiring clean datasets is costly and time-consuming due to detailed human annotation and noisy data filtering challenges. To address this, symmetric loss functions were introduced in the context of label noise, enabling models trained on noisy data to perform comparably to those trained on clean data without explicit noise knowledge. Loss functions satisfying a specific symmetry condition exhibit robustness to label noise. Building on this, we propose a novel method to derive noise-robust loss functions using monotonic functions and label normalisation, which involves a simple normalisation of labels that leads to noise robustness when labels are corrupted. Unlike other approaches, this method allows creation of new loss functions by defining application-specific monotonic functions rather than relying on predefined losses. We formally prove their theoretical properties, propose two concrete noise-robust losses, and demonstrate through extensive empirical evaluations on computer vision and natural language processing tasks that our losses outperform standard and existing noise-robust losses. Our evaluations indicate better learning of decision boundaries, faster convergence, and improved robustness to noise using the proposed loss functions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes “additive normalization of labels” for noise-robust learning. Given a $k$-class problem with one-hot target $y\in{e_1,\dots,e_k}$ and model scores passed through any strictly increasing component-wise map $f(p)=[f_1(p_1),\dots,f_k(p_k)]$, the authors replace $y$ by $\bar y=\frac{1}{k-1}(k y - \mathbf{1})$ and optimize $\ell(y,p)=-\langle \bar y, f(p)\rangle$. They prove that under symmetric label noise with flip probability $q$, the noisy risk equals the clean risk up to a positive scalar when $q<\frac{k-1}{k}$ (binary case reduces to $q<\tfrac12$). They argue that the loss is “symmetric” because $\sum_{y}\ell(y,p)=0$, and claim this entails robustness beyond symmetric noise. Two instantiations are presented: a noise-robust focal loss (NRFL) obtained by plugging the focal form into $f$, and a class-imbalance variant “weighted robust log loss” (WRLL) using $f_i(p_i)=\log(\alpha_i+p_i)$ with $\alpha_i$ inversely proportional to class frequency. Experiments on MNIST, Fashion-MNIST, CIFAR-10, several NLP tasks (News20, a synthetic JSON IE task, MALLS, GSM8k, OpenBookQA), and small LLMs reportedly show improved robustness, faster convergence, and clearer decision boundaries.

### Strengths
1. This article is well-written and easy to read.
2. Flexible template: any strictly increasing componentwise $f$ can produce a loss; easy to implement (NRFL, WRLL).
3. Some empirical gains reported over CE/MAE/NFL/RLL in several noisy setups; qualitative feature visualizations align with the claimed decision-boundary effect.
4. This article draws on both an image dataset and an NLP dataset, which is commendable.

### Weaknesses
1. This article evaluates the method only on very small datasets and does not include experiments on benchmarks with more categories or larger scale, such as CIFAR-100 or WebVision. It is well known that symmetric losses [1, 2, 3] are challenging to optimize. Results on small datasets like CIFAR-10 and MNIST are insufficient to assess optimization capability: even symmetric losses such as MAE and NCE—despite their optimization difficulty—can perform well in these settings. In contrast, on datasets with more categories or larger scale, such as CIFAR-100 and WebVision, symmetric losses are often harder to optimize and typically underperform. Consequently, using a symmetric loss in isolation has limited practical value. As a result, the experiments presented by the authors do not establish the practical significance of the proposed method.
2. The author did not compare with advanced methods such as GCE [1], APL [2], and AGCE [3]. The experimental results show that the method proposed by the author does not significantly outperform the most basic MAE.
3. Lack of details regarding NLP experiments, there are multiple processes and methods for training large language models, such as supervised training, alignment or add an MLP layer for classification training. The author did not clearly explain how they conducted the training.
4. Why only train for 15 epochs or 30 epochs on the CV dataset? This is unreasonable. For instance, on the CIFAR-10 dataset, one usually needs to train for at least 100 epochs.
5. The theoretical contribution is limited. The author did not make any new theoretical contributions.
6. A minor issue: Table ?? in page 12.

[1] Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy labels.

[2] Normalized Loss Functions for Deep Learning with Noisy Labels.

[3] Asymmetric Loss Functions for Learning with Noisy Labels.

### Questions
1. Why not conduct the experiments on CIFAR-100 and WebVision? Evaluate on real-world noisy datasets and on non-uniform/instance-dependent noise to support the broader robustness claims.
2. Why not compare with the advanced robust loss function? Add strong, modern baselines under the same budgets and report multiple seeds with confidence intervals.
3. Why only conduct training for 15 or 30 epochs? Please re-run image classification with standard training pipelines (reasonable learning rates, epochs) and equalize hyperparameters across methods. Current CIFAR-10 baselines (e.g., 34% at 0% noise with ResNet-18) are not credible.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Additive Normalization of Labels, a framework for constructing noise-robust loss functions by normalizing labels instead of losses. The authors theoretically prove that this additive normalization preserves the same optimal solution under label noise, satisfying the symmetry condition for noise tolerance. Based on this principle, they propose two new losses—Noise-Robust Focal Loss (NRFL) and Weighted Robust Log Loss (WRLL)—which consistently outperform existing methods across computer vision and NLP tasks. The approach is simple, theoretically grounded, and broadly applicable to real-world noisy learning scenarios.

### Strengths
1. The paper provides an interesting and reasonable insight into maintaining collinearity between clean and noisy labels.
2. The theoretical analysis clearly and rigorously validates the feasibility of the proposed additive normalization of labels.
3. Extensive experiments empirically demonstrate the effectiveness of the proposed approach.

### Weaknesses
1.In line 222, there is a condition requiring the noisy label $\tilde{y}$ and the clean label $y$ to be collinear, i.e., $q<\frac{k-1}{k}$. This implies that in a kkk-class classification task (e.g., k=10), the model learns the correct label direction only when the noise ratio is below 0.9. It would be interesting to empirically explore this noise-ratio boundary to gain additional insights for theory verification—for example, does the performance drop sharply when the noise ratio increases from 0.89 to 0.91?
2. Figure 1 only presents a PCA analysis under a 50% noise ratio. It would be helpful to also include PCA results under 0% noise or use an alternative visualization to further demonstrate the collinearity property.
3. The decimal places in Table 2 are inconsistent and should be standardized for clarity.

### Questions
Please refer to weaknesses.

### Soundness
2

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
4

### Summary
This paper introduces a new way to handle noise in data by changing labels, not the loss function. The main idea is to adjust labels using an additive method that centers target vectors. This makes sure the loss can handle noise, as shown in earlier studies. The authors prove that with symmetric label noise, minimizing risk with noisy labels is the same as minimizing clean risk if the noise rate $q &lt; \frac{k-1}{k}$. From this idea, the paper creates two loss functions: the noise-robust focal loss and the weighted robust log loss. Both keep a steady relationship with prediction confidence and are strong against noise due to the label adjustment. Tests on image classification (MNIST, CIFAR-10, Fashion-MNIST) and large language model tasks (like text classification, information extraction, translation, reasoning, and short answer grading) show these losses work better than cross-entropy, mean absolute error, and other methods when labels are noisy. The method is simple, based on theory, and works well in practice. It combines previous methods into one general approach that can be used in other areas beyond supervised learning.

### Strengths
The theory is solid and nicely links additive label normalization with known symmetry rules for handling noise. 
The results are strong in both vision and language areas, showing steady improvements with different noise levels. 
The method is simple, fast, and can be used widely without needing changes to the model or loss function. 
The paper clearly explains its purpose and how it relates to past methods using normalization and symmetry.

### Weaknesses
All experiments use the same type of uniform label noise. Theorem 1 says it can handle different types of noise, but no tests prove this. 
In real life, noise is often uneven and depends on the instance. 
NRFL: It is unclear if improvements come from normalization or the focal loss part. The focal loss seems separate from the noise handling method. WRLL: It uses frequency-based weighting to deal with class imbalance, but its link to noise handling is not proven by theory or tests. This mixes two different issues. 
The claim of allowing "application-specific" loss design is not shown with a clear method. The losses seem random, not based on the framework.
insufficient baseline comparisons with recent noise-robust methods
Paper uses 2025 formatting instead of 2026

### Questions
How is additive normalization of labels different from Ma et al. (2020b) loss normalization?

(1) Why does your method avoid the underfitting reported in Ma et al. (2020b)?
(2) Is there proof or empirical evidence that additive label normalization outperforms loss normalization?
(3) How does the training speed compare to standard losses under noisy labels?
(4) Please report results for normalized cross-entropy (without focal) versus the proposed noise-robust focal loss, to isolate the effect of additive normalization from the focal term.
(5) For the weighted robust log loss, compare:
(a) standard weighted loss without normalization,
(b) normalized unweighted loss,
(c) the proposed weighted robust log loss (WRLL).

On the noise bound in Theorem 1:
(6) The bound (k-1)/k suggests degradation at very high noise rates. How does performance behave beyond this threshold?
(7) Please explain why the noise limit is (k-1)/k and discuss how accurate or conservative this bound is in practice.
(8) What happens on non-uniform dataset like CIFAR-N, Clothing-1M more IDN or Class-dependent

### Soundness
2

### Presentation
3

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
The paper proposes “additive normalization of labels” to construct noise robust losses. They instantiate the framework into two specific losses: Noise Robust Focal Loss (NRFL) and Weighted Robust Log Loss (WRLL). Experiments on image classification (MNIST, Fashion-MNIST, CIFAR-10) and several NLP tasks (News20 classification, synthetic JSON information extraction, MALLS, GSM8k reasoning, OpenBookQA) claim improved robustness versus CE, MAE, RLL, and NFL.

### Strengths
1. The noisy label learning problem studied here has practical significance, and developing robust loss functions is a promising direction for further exploration.
2. Simple, clear construction; directly yields the classical symmetric-noise robustness criterion.
3. The paper has a complete structure, with experiments spanning both computer vision and natural language processing.

### Weaknesses
1. No new contributions: This paper cites [1]. After my review, the method proposed in this paper is exactly the same as that in [1]. Therefore, this paper may not have any new contributions. The authors might not have read [1] carefully. I have serious concerns about this matter.
2. Unfair experiments:  The authors claim that their method converges more quickly. However, NRFL and WRLL specifically used higher learning rates compared to the baselines, without maintaining consistency. This might be an unfair experiment. In addition, for NRFL, the authors added an additional parameter $\delta$ to adjust the gradient, but did not keep it consistent with the baselines, which could lead to unfair experiments.
3. The performance was mediocre: Although the authors claim that their method is effective, the experiments show that, even if there might be unfair learning rates and gradient-scaling issues, their method is inferior to standard MAE in many cases. This significantly undermines the significance of their approach.

### Questions
Please refer to "Weaknesses"

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a new foundational study on robust loss design for the LNL problem.
While numerous LNL methods exist, the authors’ goal appears to be a more fundamental exploration that could inspire future research on noise-robust loss functions.
Specifically, they define the probability of label noise (assumed to occur randomly) and formulate a loss function that minimizes the expected risk under this noise distribution.

### Strengths
1. The proposed method contains almost no hyperparameters, which makes it elegant and easy to reproduce. This also implies that relaxed variants of robust loss (those requiring tuning parameters) are not the primary focus of this work.

2. The authors conduct experiments on a wide range of benchmarks. Notably, the inclusion of NLP datasets in their experiments is quite novel in the LNL literature and demonstrates the potential generality of their approach.

### Weaknesses
1. The theoretical derivation is rather straightforward.

Intuitively, in a k-class classification problem, it is not difficult to reason about the level of random label noise that can be tolerated before performance degrades.
Although the authors explain this process clearly, the derivation itself offers limited new insight.
Moreover, the analysis assumes purely random (uniform) noise and does not consider more realistic or ambiguous cases, such as class-dependent or instance-dependent label noise.

2. The empirical results show limited improvement.

While the proposed method achieves small gains, the baselines compared against are not among the strongest or most recent methods in the LNL field.
Although the restricted comparison setup is understandable given the paper’s theoretical orientation, additional information or analyses would be required for the paper to serve as a solid foundation for future work.
I suggest introducing a relaxed normalization variant (e.g., one controllable by hyperparameters) and demonstrating superiority over established robust losses such as GCE or APL.

### Questions
I understand the authors’ proposed method and their intended objective; however, the experimental results and theoretical justification do not seem sufficient for this paper to be considered a new milestone in the field.
As mentioned in the Weaknesses, the authors should at least demonstrate the potential to extend their proposed framework or provide stronger evidence of theoretical noise tolerance under more challenging conditions. Such additions would make the paper significantly more convincing and impactful.

### Soundness
2

### Presentation
2

### Contribution
3
