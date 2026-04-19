# Trained Transformer Classifiers Generalize and Exhibit Benign Overfitting In-Context

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Transformers have the capacity to act as supervised learning algorithms: by properly encoding a set of labeled training (''in-context'') examples and an unlabeled test example into an input sequence of vectors of the same dimension, the forward pass of the transformer can produce predictions for that unlabeled test example.  A line of recent work has shown that when linear transformers are pre-trained on random instances for linear regression tasks, these trained transformers make predictions using an algorithm similar to that of ordinary least squares.  In this work, we investigate the behavior of linear transformers trained on random linear classification tasks.  Via an analysis of the implicit regularization of gradient descent, we characterize how many pre-training tasks and in-context examples are needed for the trained transformer to generalize well at test-time.  We further show that in some settings, these trained transformers can exhibit ''benign overfitting in-context'': when in-context examples are corrupted by label flipping noise, the transformer memorizes all of its in-context examples (including those with noisy labels) yet still generalizes near-optimally for clean test examples.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies in-context learning of linear classification tasks on linear transformers. The authors characterize how many pre-training tasks and in-context samples are needed for generalization, with an analysis of the implicit regularization of gradient descent. They show that after pre-training the transformer can tolerate smaller SNR than those tasks on which it was trained. Additionally, they show these trained transformers can memorize all in-context samples yet still generalize near-optimally when the in-context samples’ labels are flipped.

### Strengths
1. The study of in-context learning is important, and the investigation on linear classification setting is novel.
2. The results of benign overfitting of transformer, to the reviewer’s knowledge, is new.
3. The paper is well-written and easy to follow.

### Weaknesses
1. It would have been interesting to discuss the connection of task complexity results under linear classification to Wu et al. [1]. In addition, briefly mentioning the importance of investigating linear classification instead of regression in the paper would be beneficial.
2. Can you explain why it would be interesting to consider a different signal-to-noise ratio R\tilde at test time? Analyzing the difference between M and N seems to be more interesting. 
3. The reviewer personally would like to see consistent empirical observation with simple experiments (such as a simplified version of Raventós et al [2]) but understand this might be out of the scope of this work. Maybe the author can consider mentioning it as a limitation.

[1] How Many Pretraining Tasks Are Needed for In-Context Learning of Linear Regression? Wu et al. ICLR 2024   
[2] Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression. Raventós et al. NeurIPS 2023

### Questions
1. At row 72, 73 → ‘even when pre-training on simple and easy-to-learn datasets, the transformer can generalize on more complex tasks.’ What does easy and complex task mean in this context? Does it mean the signal-to-noise ratio?

Minor:
1. At line 331, a typo of ‘sufficiently’.

### Soundness
3

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
3

### Summary
The paper investigated the behavior of linear transformer models trained on random classification tasks, extending the existing work on linear regression tasks. It studied how linear transformers can generalize and exhibit _benign overfitting_, meaning that even if a model memorizes noisy training examples, it still performs well on test examples. This paper is completely theoretical, and no empirical evidence was given.

### Strengths
- This work theoretically analyzed how many pre-training tasks and in-context examples are needed for linear transformers to effectively generalize in linear classification tasks.
- This work studied the phenomenon of benign overfitting in this context and bounded the generalization error.
- The paper is solid and well presented. All the symbols are properly defined. The assumptions and theoretical results are clearly stated. (I did not carefully check the proofs.)
- The paper is well contextualized. The author mentioned related work (that I'm not familiar with) and discussed future work directions, which helped me position this work.

### Weaknesses
My biggest concern is the same for all theoretical work on simple models (e.g., single-layer linear transformer) with strong data assumptions (e.g., class-conditional Gaussian mixture). It is perfectly fine to start with a simple model with strong assumptions, as long as it is a good proxy for real-world problems or a stepping stone to understanding more complex problems. However, we need either theoretical or empirical evidence showing it is the case. In the current paper, there is a lack of discussion (or references) on whether or to what extent real-world data satisfies the data assumptions. It is unclear whether the theoretical results can be generalized to more complex models. Overall, I'd like to know how the theoretical results can guide our practice, which is not completely clear in the current version.

### Questions
In Figure 1, the author used a natural language example to show the phenomenon of benign overfitting. However, I don't think it satisfies the data assumption. Is there an example/illustration that precisely satisfies the data/model assumptions?

### Soundness
3

### Presentation
3

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
This work presents a sample complexity analysis for in-context learning of class-conditional Gaussian mixtures with a restricted linear attention transformer model (single layer). It uses the results established on how SGD on convex-linear transformer models has a bias toward maximum-margin solution (in direction). The work thus assumes the KKT conditions are satisfied at the SGD solution and uses them to quantify the sample requirement to achieve a small test error. It shows how --by choice of specific model and data parameters-- one can achieve benign overfitting for in-context training, a phenomenon in which noisy training examples are exactly memorized, yet the test accuracy remains near-optimal.

### Strengths
- The work references and uses prior work to great effect. It builds up on ideas established around the implicit regularization effect of the optimization algorithms used to train transformer models. SGD-based algorithms applied to convex-linear transformers have a directional bias towards maximum-margin solutions. Hence once can apply techniques used to study such solutions to derive generalization bounds such as the one presented in 4.1 and 4.2.

### Weaknesses
- Readability: The paper is very dense and in parts hard to follow, requiring multiple revisits to earlier sections for (non-standard) notation and technical definitions. This, unfortunately, severely cuts into the readability of the paper. Adding a table of notation either in the paper or the appendix can be helpful.

- Analysis: There is no empirical evaluation of the tightness of derived bounds. IMO the derived bounds are to be understood in the limit and to prove the theoretical possibility of benign over-fitting. They might be too lose in practice. Adding an study in which $M$ or $d$ can be varied can help establish the empirical tightness of the bound.  

- Scope: The work only focuses on a specific restricted linear model. It defers analysis on non-convex linear or softmax-based attention models to future work.

- Assumptions: Even with the SGD-based solutions being directionally biased towards max-margin solutions, it doesn’t necessarily follow that the KKT conditions are satisfied in real implementations with early stopping or partial convergence. It’s also a leap to assume this (line of) theoretical analysis can be extended to other settings beyond the restricted or convex linear case addressed in the paper. IMO these assumption should be studied/validated under realistic training settings. Adding such analysis to the paper can clear some of these points.

### Questions
- Could tighter bounds be established for stochastic (Gibbs) classifiers? This would be stochasticity in the loss that is not a result of label-flips on the test set, but is a result of stochastic classifier decision.


- Line 377 breaks a formula.

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
3

### Summary
This paper studies the ability of linear transformers to perform in-context learning for linear classification tasks. The authors consider a simplified, convex linear transformer architecture trained on a series of random, class-conditional Gaussian mixture models.

The paper investigates two key aspects:

1. the number of pre-training tasks required for the transformer to generalize well at test time, even when the test data has lower signal-to-noise ratios (SNR) and label-flipping noise not present in the pre-training data.

2. the phenomenon of “benign overfitting in-context,” where the transformer memorizes noisy in-context training examples while still achieving near-optimal generalization for clean test examples.

The authors claim this work to be the first theoretical analysis of in-context learning for linear classification using linear transformers, and also the first demonstration of benign overfitting capabilities.

This paper makes valuable theoretical contributions to the understanding of in-context learning in transformers, highlighting their generalization capabilities and the intriguing phenomenon of benign overfitting. While the simplified model and assumptions might limit direct practical implications, the paper opens up interesting avenues for future research on the theoretical foundations of in-context learning.

### Strengths
Rigorous theoretical analysis: The paper offers a detailed analysis of the implicit regularization of gradient descent during pre-training and leverages the max-margin framework to derive generalization guarantees.

Novel insights: The paper reveals the intriguing ability of the trained transformer to generalize under lower SNR and label-flipping noise at test time, even when such conditions are absent during pre-training.

Demonstration of benign overfitting: The paper provides theoretical and experimental evidence for the previously unobserved phenomenon of benign overfitting in transformers.

This paper presents novel theoretical insights into the in-context learning capabilities of transformers for linear classification tasks. The demonstration of benign overfitting is particularly noteworthy. While the simplified assumptions and focus on a convex architecture limit the direct applicability of the findings, the paper offers a valuable foundation for future research in understanding in-context learning in more realistic transformer settings.

### Weaknesses
Simplified architecture: The analysis relies on a 1-layer linear transformer model, which is a simplification of standard multi-layer transformer models with softmax-based attention. It is unclear whether about the applicability of the results to more complex transformer architectures.

Noise-free pre-training: The requirement of clean data during pre-training restricts the practicality of the approach in noisy environments.

Strong assumptions on data distribution: The paper focuses on class-conditional Gaussian mixture models, which might not be representative of the complex data distributions encountered in natural language processing tasks.

Absence of Numerical Validation: While the theoretical analysis is valuable, the paper lacks numerical experiments to validate the theoretical findings. Providing empirical evidence to support the theoretical claims, and demonstrating how the results generalize to more realistic settings, would enhance the paper's impact and persuasiveness.

### Questions
Practical Implications: Recognizing the limitations of the simplified setting, can the authors discuss any potential practical insights for realistic problems?

Exploration of Alternative Architectures: How would the results generalize to more complex transformer models? 

Pre-Training with Noisy Labels: The study assumes clean input data and labels during pre-training. How would the result change once noise is introduced in the pre-training data? 

Numerical Validation of Theoretical Results: Have the authors tried conducting numerical experiments to validate your theoretical findings? It would be highly insightful to see how well the theoretical predictions align with empirical observations in both the simplified setting used for analysis and in more realistic settings involving complex transformer architectures and real-world datasets. For instance, could the authors present experimental results demonstrating the relationship between the number of pre-training tasks, the signal-to-noise ratio, and the generalization performance, as suggested by your theoretical bounds? Furthermore, showcasing empirical examples of benign overfitting in-context would be compelling evidence to support this key finding.

### Soundness
3

### Presentation
3

### Contribution
3
