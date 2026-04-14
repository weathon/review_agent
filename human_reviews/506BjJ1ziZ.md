# COME: Test-time Adaption by Conservatively Minimizing Entropy

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 8

## Abstract
Machine learning models must continuously self-adjust themselves for novel data distribution in the open world. As the predominant principle, entropy minimization (EM) has been proven to be a simple yet effective cornerstone in existing test-time adaption (TTA) methods. While unfortunately its fatal limitation (i.e., overconfidence) tends to result in model collapse. For this issue, we propose to \textbf{\texttt{Co}}nservatively \textbf{\texttt{M}}inimize the \textbf{\texttt{E}}ntropy (\texttt{COME}), which is a simple drop-in replacement of traditional EM to elegantly address the limitation. In essence, \texttt{COME} explicitly models the uncertainty by characterizing a Dirichlet prior distribution over model predictions during TTA. By doing so, \texttt{COME} naturally regularizes the model to favor conservative confidence on unreliable samples. Theoretically, we provide a preliminary analysis to reveal the ability of \texttt{COME} in enhancing the optimization stability by introducing a data-adaptive lower bound on the entropy. Empirically, our method achieves state-of-the-art performance on commonly used benchmarks, showing significant improvements in terms of classification accuracy and uncertainty estimation under various settings including standard, life-long and open-world TTA, i.e., up to $34.5\%$ improvement on accuracy and $15.1\%$ on false positive rate. Our code is available at: \href{https://github.com/BlueWhaleLab/COME}{https://github.com/BlueWhaleLab/COME}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a Bayesian inference technique to address the overconfidence problem in test-time domain adaptation. Experiments demonstrate its effectiveness, achieving a SOTA performance.

### Strengths
- The motivation of the proposed method is clear. 

- The idea of transforming the optimization problem with a constraint (Eq. (7)) into a simpler form (Eq. (9)) is interesting and effective, which significantly simplifies the problem.

- Several theoretical results, as well as empirical results, are provided. Theorem 1 helps quantitative understanding of the proposed method.

- Code is available.

### Weaknesses
- Experiments on the dependence of the results on $p$ and $\tau$ are lacking. I would like to see the results and discussions when $p \neq 2$ and $\tau \neq 1$.

- (Major) The paper applies a Bayesian inference (Eq. (5)) with a regularization to the overconfidence problem inherent in TTA. The "EM" (entropy minimization) mentioned in the paper is, in a more general context, the unsupervised learning using soft pseudo-label, which has a wide variety of applications beyond TTA.
While the proposed method is technically sound, it would benefit from discussion in a broader context—such as unsupervised learning with a pretrained model, unsupervised domain adaptation, source-free domain adaptation, and semi-supervised learning—to emphasize its wide applicability as a key contribution.

- (Major) Error bars are missing. Could you provide error bars because several performance gains are marginal?

- The performance metric "Avg." in the tables are nonsense, while it is actually a bad convention in the field. Obviously, a "1% gain" is quite different in iNaturalist and SSB-Hard, for example.

- (Minor) A large part of the paper is dedicated to reviewing previously known works, such as the overconfidence problem, which hinders readability.

- (Minor) Much of the paper, particularly the description of the proposed method, is redundant. A more direct tone is generally preferable in academic writing.

- Overall, while the paper is well-written and the proposed method is interesting and effective, the paper would benefit from a more refined articulation of its contributions and focus. Additionally, the reproducibility issue should be addressed.

### Questions
- (Eq. (6)) What if a hyperparameter $\lambda \in \mathbb{R}$ is introduced as $-\sum_{k=1}^K b_k \log b_k - \lambda u \log u$? This seems to be a straightforward generalization of Eq. (6).

- Is there any quantitative correspondence between $\tau$ and $\delta$ (Eq. (9) & (7))?

- How can we set $p$ and $\tau$ (or $\delta$) in practice?

- To my understanding, the regularizer (Eq. (9)) effectively prevents spurious training that would drive the second term in Eq. (6) to zero by enforcing $e \rightarrow 0$. Is this correct?

- To me, the proposed method seems to be a simple combination of known techniques (to clarify, I do *not* claim that simplicity  alone is grounds for rejection at all). Could you clarify the differences of the proposed technique from other Bayesian approaches, confidence calibration algorithms, semi-supervised learning, and entropy regularization techniques? A quantitative and objective discussion would be preferable, which would significantly enhance the paper's contribution and clarity.

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
3

### Summary
This paper investigates the issue of model collapse in entropy minimization algorithms for test-time adaptation. The authors propose a novel entropy minimization approach that models prediction uncertainty by defining a Dirichlet prior distribution over model predictions. This method regularizes the model to favor conservative confidence for unreliable samples. Experiments on benchmark datasets demonstrate the effectiveness of the proposed algorithm.

### Strengths
The proposed algorithm introduces a rejection mechanism for unreliable samples in the TTA process, preventing the model from learning from potentially noisy labeled data. It is simple to integrate into existing TTA frameworks, and the experimental results indicate satisfactory performance.

### Weaknesses
The proposed method and its theoretical analysis rely heavily on existing techniques, which limits its technical novelty.

The core concept shares some similarity with research on learning with rejection. It is recommended to discuss how the proposed loss function compares with the loss functions used in learning with rejection, as outlined in [1].

There is a lack of experiments involving real-world applications with distribution shifts, as exemplified in [2]. Testing the proposed algorithm on real-world data streams in dynamic environments is suggested to validate its robustness.

References:

[1] Cortes, Corinna, Giulia DeSalvo, and Mehryar Mohri. "Learning with rejection." Algorithmic Learning Theory (2016).

[2] Yao, Huaxiu, et al. "Wild-time: A benchmark of in-the-wild distribution shift over time." Advances in Neural Information Processing Systems 35 (2022): 10309-10324.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The authors propose Conservatively Minimizing Entropy, a method for test-time adaptation (TTA) that improves test-time adaption by managing prediction uncertainty. Unlike traditional entropy minimization, which can lead to overconfidence, COME uses a Dirichlet distribution to model uncertainty, allowing the model to avoid forced classification on unreliable data.

### Strengths
- This paper is well-motivated, and the story makes sense. 
- Extensive experiments have been done to support the proposed method.

### Weaknesses
My major concerns include:
- For the proposed method: Why Dirichlet distribution is used? How is the Dirichlet distribution related to the final algorithm in Algorithm 1. In addition, what is the role of delta in Algorithm 1? It seems that the authors tell a long story about their algorithm, but the algorithm itself is rather simple.
- For the theoretical analysis: Could the authors provide a more detailed (theoretical) comparison between the proposed method and traditional EM? What is the benefit?
- For baselines: Some baselines are missing, for example,  [1] and [2].
- For the datasets: I'm curious why the authors follow literatures on outliers detection.
- For Theory 1: What is the exact benefit of the upper bound of model confidence? I think it will also hurt the performance on some "confident" samples. 

[1] Nado, Z., Padhy, S., Sculley, D., D'Amour, A., Lakshminarayanan, B., & Snoek, J. (2020). Evaluating prediction-time batch normalization for robustness under covariate shift. arXiv preprint arXiv:2006.10963.
[2]Zhou, A., & Levine, S. (2021). Bayesian adaptation for covariate shift. Advances in neural information processing systems, 34, 914-927.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the model collapse of the popular Entropy Minimization algorithm for Test-Time Adaptation. Motivated by the observation that the amplification of model over-confidence causes model collapse, this paper proposes to minimize entropy with respect to an augmented output distribution that includes the probability to reject a sample, which is an uncertainty estimation technique known as subjective logic. Moreover, a logit normalization is designed in order to avoid degenerated solutions. Theoretical analysis reveals that the resulting approach upper bounds model confidence during adaptation based on the sample-specific confidence of the initial model. The resulting algorithm, COME, can be easily embedded into general EM-based TTA methods with a few lines of code revision. Experiments across TTA, open-world and life long TTA settings demonstrate a significant and consistent improvement upon current EM baselines.

### Strengths
This paper accurately spots the paradox of EM's learning objective: minimization of entropy leads to over-confidence. And the paper proposes a simple yet effective solution to minimize entropy with respect to a probability distribution that faithfully estimates the uncertainty without over-confidence. It is a very reasonable idea to differentiate between the statistics used for prediction and for uncertainty estimation, which has long been considered the same in the TTA literature. Therefore, the algorithm enjoys the feature that the entropy minimization can be tailored to samples with different uncertainty, which is also supported by the monotonicity result in Theorem 1. The introduction of SL for uncertainty estimation is natural and perfectly compatible with softmax functions used for training models in most cases. As a result, the implementation is light-weight, model-agnostic, and extremely easy to embed into any TTA algorithms based on the EM objective. The experiments are convincing by covering both standard TTA tasks and more challenging settings of open-world TTA. A surprisingly significant 34.5% improvement on accuracy is reported on the model of SAR. And the algorithm has further addressed uncertainty estimation under continual distribution shift as a side product, which itself is also an important problem.

### Weaknesses
These are not necessarily weaknesses but rather some questions that I would like to confirm with the author.
1. How does the algorithm ensure that $b_k$ is non-negative for the computation of entropy, since $b_k$ is implemented as 
$(e^{f_k(x)}-1)/ \sum_{k'} e^{f_k'(x)}$ which could be negative?

2. Why does the algorithm keep $u$ close to $u_0$? Does it imply that the uncertainty estimation for the pretrained model is trusted? What if the pretrained model is over-confident? What about the alternative constraint $u \geq u_0$ which seems to be more conservative as is the objective of COME?

3. Average false positive rate is used in experiments to assess uncertainty estimation. However, FPR measures the correctness of uncertainty estimation with such a binary perspective: for the samples we predict 1, what is the actual proportion of 0. Uncertainty estimation considers a more sophisticated question: for the samples we predict with a probability 0.7, is the actual proportion of 1 exactly 0.7? Expected calibration error (ECE) is a better metric in this sense.

4. Are there standard errors of the reported results?

### Questions
1. The tightness for the upper and lower bounds in Lemma 1 is determined by the choice of p. By considering the simple model where f(x) outputs the same logit for all classes, the ratio between the upper and lower bound is minimized by $p=\infty$. Is is a better choice to consider $\| f(x) \|_\infty = \max_k | f_k(x) |$?

2. The usage of exp transformation of logits in the softmax function appears pivotal to the proof of Theorem 1. And if we take b(x) as a general non-negative function of f(x), the upper bound may reduce to 1. And minimizing the entropy of opinion in equation 6 can also lead to the Dirac distribution, which is over-confident.  Is there a characterization for the class of functions that could be used to form a reasonable belief b(x)?

### Soundness
3

### Presentation
4

### Contribution
3
