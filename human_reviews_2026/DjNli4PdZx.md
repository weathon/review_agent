# LiLAW: Lightweight Learnable Adaptive Weighting to Meta-Learn Sample Difficulty and Improve Noisy Training

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Training deep neural networks in the presence of noisy labels and data heterogeneity is a major challenge. We introduce Lightweight Learnable Adaptive Weighting (LiLAW), a novel method that dynamically adjusts the loss weight of each training sample based on its evolving difficulty level, categorized as easy, moderate, or hard. Using only three learnable parameters, LiLAW adaptively prioritizes informative samples throughout training by updating these weights using a single mini-batch gradient descent step on the validation set after each training mini-batch, without requiring excessive hyperparameter tuning or a clean validation set. Extensive experiments across multiple general and medical imaging datasets, noise levels and types, loss functions, and architectures with and without pretraining demonstrate that LiLAW consistently enhances performance, even in high-noise environments. It is effective without heavy reliance on data augmentation or advanced regularization, highlighting its practicality. It offers a computationally efficient solution to boost model generalization and robustness in any neural network training setup. Code in Supplementary Material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new method, LiLAW, that improves deep neural network training in challenging conditions like noisy labels and data heterogeneity. It dynamically adjusts loss weights based on sample difficulty using three learnable parameters. LiLAW prioritizes informative samples by updating weights with a mini-batch gradient descent step on the validation set after each epoch, requiring minimal hyperparameter tuning and no clean validation set. Extensive experiments show LiLAW enhances model performance. It’s computationally efficient, doesn’t heavily rely on data augmentation or advanced regularization, and is applicable to various datasets, noise levels, loss functions, and model architectures.

### Strengths
The method is computationally efficient, doesn’t heavily rely on data augmentation or advanced regularization, and is applicable to various datasets, noise levels, loss functions, and model architectures.

### Weaknesses
Indeed, I reviewed this paper for ICML 2025. In my previous review, I noted that the experimental evaluation was insufficient, as several commonly used datasets were missing from the comparison. However, these datasets are still not included in the current version, and the authors have not provided any explanation for their exclusion.

### Questions
None.

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
5

### Summary
The article introduces LiLAW, a lightweight meta-learning method designed to improve the training of deep neural networks under noisy and heterogeneous data conditions. The proposed method dynamically adjusts the loss weight of each training sample based on its evolving difficulty, categorized as easy, moderate, or hard, using three learnable parameters (\alpha, \beta, and \delta). Extensive experiments across multiple datasets, noise types, and architectures validate the effectiveness of the proposed method.

### Strengths
The proposed method demonstrates versatility and is applicable to multiple domains, such as, natural images, medical images, and time-series tasks. The proposed method is model-agnostic and can be applied to different benchmarks. The adaptive learning mechanism avoids the need for extensive hyperparameter tuning.

### Weaknesses
- The issue of noisy labels in classification is a widely studied problem. However, this paper lacks comparisons with classical noisy label learning methods in the experiments.
- Applying meta-learning techniques to learn the sample weights is a typical approach in noisy label learning. Authors are suggested to clarify the distinctive advantages of their proposed method compared to such meta-learning approaches and include comparative experiments.
- The approach of using noisy data as the validation set contradicts the theoretical analysis of prior meta-learning reweighting methods. This point is supported solely by empirical evidence and lacks theoretical interpretations.
- Using meta-learning method will obviously increase the calculation time. Authors are suggested to compare the proposed method with the vanilla method in time.

### Questions
- What are the physical meanings of the meta-learned \alpha, \beta and \delta, and what is their mechanism of impact with respect to the sample weights?
- In the method section, the paper proposes to initialize \alpha<\beta<\delta. However, in the experiments, why the paper chooses to set \alpha=10, \beta=2, \gamma=6?
- The proposed method is designed to address label noise. What is the underlying reason for its effectiveness in handling input noise, as demonstrated in Table 1?

### Soundness
3

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
4

### Summary
The paper propose a re-weighting strategy for training with noise label. It assign different weights to cases with different confidence and correctness levels using three additional hyperparameters, which are optimized using meta learning. The experiments are conducted on dataset with different noise levels and show better results on defending the noise. The method is also tested on different kind of dataset to proof the generalizability. However, some necessary ablations and comparisons are missing, which are important for understanding why this method performs better.

### Strengths
The proposed method is simple and easy to understand and implement. And it provides an effective strategy of defending noise in label at different levels. However, there are some necessary ablations and comparisons missing to better understand the superiority of this method.

### Weaknesses
See the questions part below for concerns

### Questions
1. The claim ""s_i[\hat{y_i}] < max(s_i) implies an incorrect prediction when \hat{y_i}=y_i" is correct. But how to know whether \hat{y_i}=y_i given the noisy target only? The motivation is to assign weights to samples with different confidence and correctness level, but the definition of correctness is unable to verify. 

2. Is the noise level of validation set changing according to the training set? Or it is fixed? Have you tried using a clean validation set to see how the noise level of validation set affect the results?

3. Please explain the claim" easy, moderate, hard examples activate W_alpha, W_beta, W_delta respectively" in chapter 3. The sentence after "Geometrically, ..." is hard to understand. like "Wβ when β is small and/or..." 
The three hyper parameters jointly define the final weights as the total weight is a sum of W_alpha, W_beta, W_delta, I don't see how they can separately correspond to different type of examples.

4. What is the definition of easy, moderate, hard examples

5. Are there any ablations showing the difference of using different combination of W_alpha, W_beta, W_delta? That will help understand each part

6. The design of different form of W_beta seems redundant.  Actually W_beta(beta) = 1- W_alpha(beta)

7. Using confidence is a common way of defending noise, like assigning more weights to cases with more confidence. How is the comparison with such methods.

8. About complexity analysis, the paper claims the LiLAW keeps the same complexity but from the algorithm, LiLAW requires two times of forward and two times of backward for each mini-batch, which sounds like a doubled time.  Even the second backward only changes three parameters, that doesn't mean it only cost O(3) time in the actual implementation. Is there any actual time cost comparison with/without LiLAW?

### Soundness
3

### Presentation
2

### Contribution
2
