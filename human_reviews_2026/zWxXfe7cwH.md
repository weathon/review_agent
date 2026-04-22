# Does the Data Processing Inequality Reflect Practice? On the Utility of Low-Level Tasks

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
The data processing inequality is an information-theoretic principle stating that the information content of a signal cannot be increased by processing the observations. In particular, it suggests that there is no benefit in enhancing the signal or encoding it before addressing a classification problem. This assertion can be proven to be true for the case of the optimal Bayes classifier. However, in practice, it is common to perform "low-level" tasks before "high-level" downstream tasks despite the overwhelming capabilities of modern deep neural networks. In this paper, we aim to understand when and why low-level processing can be beneficial for classification. We present a comprehensive theoretical study of a binary classification setup, where we consider a classifier that is tightly connected to the optimal Bayes classifier and converges to it as the number of training samples increases. We prove that for any finite number of training samples, there exists a pre-classification processing that improves the classification accuracy. We also explore the effect of class separation, training set size, and class balance on the relative gain from this procedure. We support our theory with an empirical investigation of the theoretical setup. Finally, we conduct an empirical study where we investigate the effect of denoising and encoding on the performance of practical deep classifiers on benchmark datasets. Specifically, we vary the size and class distribution of the training set, and the noise level, and demonstrate trends that are consistent with our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates when and why low-level data preprocessing can be beneficial for a strong classifier (e.g., one that converges to the optimal Bayes classifier). The authors first prove the existence of a preprocessing (dimension reduction) procedure that can improve classification performance. They then establish theoretical results illustrating the influence of various factors, such as the number of training samples and the degree of class separation. The analysis is further supported by both simulation studies and real-data experiments.

### Strengths
1. The problem studied in this paper is interesting and important.
2. The theoretical analysis is solid.
3. The authors consider various scenarios and influencing factors in their theoretical derivations.

### Weaknesses
1. The writing can be improved. For example, Section 3.2 discusses multiple scenarios; it would be helpful to outline them at the beginning of the section. In addition, providing some intuition behind the theorems would improve readability and accessibility.
2. Some theorems could potentially be extended (see the question below).
3. The data processing analyzed in this paper can restrictive (see the question below).

### Questions
1. Is the data processing step restricted to the dimension reduction procedure defined in Equation (7)?
2. In Theorems 5 and 6, is it possible to characterize how $\hat{p}_x$ compares to $\hat{p}_z$? Specifically, does the difference between them vary with certain parameters?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Consider the binary classification problem.
It is a general belief that preprocessing data makes the performance of the Bayes classifier worse assuming that we know the ground truth distribution.
We would like to ask a question: If we use a practical classifier and we do not know the ground truth distribution, does the conclusion of the performance of the practical classifier getting worse still hold?
In this paper, the authors study this question and give a negative answer.
The authors consider the following setting.
Suppose the underlying distribution is the mixture of two equally weighted standard Gaussians with symmetric means.
Each sample from the training set is drawn from the distribution with a label indicating which Gaussian it is drawn from and the estimated mean for each Gaussian is the simple averaging of the corresponding samples.
Each sample from the test set is also drawn from the distribution without a label.
The classifier is to classify the sample to be the Gaussian whose estimated mean is closer to the sample.
The preprocessing is a linear transformation whose matrix is semi-orthogonal and norm preserving.
The result shows that the error for the preprocessing data can be smaller than the error for the original data.

### Strengths
- The problem is well-motivated.
The result may open a new line of work on this topic.

### Weaknesses
- Though the authors provide some experimental results, the theoretical explanation is based on a simplified setup.
In other words, the theory-practice gap is large that there may be a limited interpretation when extending to more realistic settings.

### Questions
.

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
This paper investigates theoretical perspectives on the data processing inequality in classification, suggesting that appropriate preprocessing can improve finite-sample performance.

### Strengths
1. The paper studies an interesting question: whether the data processing inequality reflects practical deep learning behavior and connects a foundational information-theoretic concept to empirical DNN practice.

2. The theoretical derivation is mathematically sound under its assumptions, offering a clean proof for the potential benefit of preprocessing under finite-sample conditions.

3. The paper is well-written and visually clear, with carefully structured derivations and supporting empirical examples.

### Weaknesses
1. The theoretical model y→x→z implies a generative process, but deep models for classification follow a discriminative direction 
x→z→ $\hat{y} $. This mismatch weakens the claimed alignment between the theoretical framework and modern deep learning models.

2. Eq. (3) assumes a Gaussian Mixture Model where data 𝑥 is generated conditional on label y. This assumption is conceptually inverted from real-world classification, where 𝑥 precedes 𝑦.

3. The theory defines 𝐴 as a linear transformation matrix, while experiments use nonlinear modules (DnCNN, Lu et al., 2025). The inconsistency makes it unclear whether the empirical findings genuinely validate the theory.

4. Although 𝐴 is constructively derived, the paper lacks intuitive explanation of how 𝐴 preserves or amplifies discriminative information. Visualizing its action in feature space or relating it to known transformations would strengthen the conceptual link between theory and practice.

5. The preprocessing step 𝐴𝑥 resembles data augmentation. If its primary function is to improve robustness via representation smoothing, the novelty may be overstated. Clarification is needed on whether the proposed framework represents a fundamentally new mechanism.

6. The experiments inject noise into data and then apply denoising, creating a circular setup that demonstrates improvement. This does not convincingly validate the theoretical claims or show robustness in realistic settings.

7. CIFAR-10 uses denoising (DnCNN), while miniImageNet uses encoding (Lu et al., 2025). Why not keep the processing method consistent, and how is the appropriate process determined for different datasets? The theory treats the processing step 
𝐴 as a general transformation matrix and is not dataset-specific.

Minor: 

1. The experiments on small datasets such as CIFAR-10 and miniImageNet, and ResNet only, are not enough to convincingly support the theoretical claims. Evaluating on larger-scale datasets and other architectures, such as ViT, is encouraged. Though the paper emphasizes theoretical contributions, demonstrating practical benefits is still necessary to substantiate its real-world relevance.

2. Can you provide the source code for Figure 1? This will help me better understand the connection between the theoretical derivation and its empirical illustration.

### Questions
Please see weakness.

### Soundness
3

### Presentation
4

### Contribution
2
