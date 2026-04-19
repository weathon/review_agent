# The importance of feature preprocessing for differentially private linear optimization

- Decision: Accept (poster)
- Scores: 6, 6, 8, 6

## Abstract
Training machine learning models with differential privacy (DP) has received increasing interest in recent years. One of the most popular algorithms for training differentially private models is differentially private stochastic gradient descent (DPSGD) and its variants, where at each step gradients are clipped and combined with some noise. Given the increasing usage of DPSGD, we ask the question: is DPSGD alone sufficient to find a good minimizer for every dataset under privacy constraints? 

As a first step towards answering this question, we show that even for the simple case of linear classification, unlike non-private optimization, (private) feature preprocessing is vital for differentially private optimization.  In detail, we first show theoretically that there exists an example where without feature preprocessing, DPSGD incurs a privacy error proportional to the maximum norm of features over all samples. We then propose an algorithm called *DPSGD-F*, which combines DPSGD with feature preprocessing and prove that for classification tasks, it incurs a privacy error proportional to the diameter of the features $\max_{x, x' \in D} \|x - x'\|_2$. We then demonstrate the practicality of our algorithm on image classification benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Differential privacy (DP) has been applied to many machine/deep learning tasks to keep privacy of dataset. DPSGD is one of the influential work to train private ML models. Although some theoretical works proves its convexity properties and leads the trend, DPSGD may not work well on raw features in image classification tasks. Beyond the DPSGD, this work demonstrates the feature preprocessing is necessary to optimize the performance of DPSGD by proposing a new algorithm - DPSGD-F as a combination of DPSGD and feature preprocessing.

### Strengths
1. Based on the proposed algorithm, this work does not only provide empirical results on specific image tasks, but it also provides theoretical guarantees on it step by step. It is really good to connect theory and practice in the research.
2. The counter-example of DPSGD in Section 4 is innovative and looks solid to investigates its limitation seldom considered by other works (to my knowledge).

### Weaknesses
1. The study subject - linear model - is relatively simple and less representative compared to SOTA ML models, especially for image classification task.
2. Except the Fashion-MNIST dataset (released in recent few years), original MNIST and CIFAR-100 datasets are relatively outdated compared to other image datasets.

### Questions
1. What is Omega tilde in Theorem 1?
2. Can you explain more about what the “uninformative” means in the Theorem 3?
3. For lemmas in Section 5, can you provide high level ideas of proofing for each one (probably one or two sentence per lemma) like you did in the Section 4 to make reader clearer about the correctness?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper investigated the significance of feature processing in differentially private linear models. The authors introduced an algorithm named DPSGD-F, which incorporates a preprocessing procedure for feature clipping within DP-SGD. The authors theoretically demonstrated the relationship between the excess risk and the feature diameter. Furthermore, their empirical results suggested that DPSGD-F outperforms DP-SGD on datasets such as MNIST, FMNIST, and CIFAR-100 (pretrained).

### Strengths
Strengths:

1. The authors introduced a novel variant of DPSGD known as DPSGD-F, which empirically outperforms DPSGD on particular datasets, and they supported the proposed algorithm with theoretical guarantees.

2. The authors presented a counterexample illustrating situations where DP-SGD does not converge.

### Weaknesses
Weaknesses:

1. I have several concerns regarding the motivation and theoretical results, which might undermine the paper's conclusion.  Below are some of my concerns about the counter example in Section 4 and Theorem 1 in Section 3.

    a) In the counterexample, the condition $\mu > n\epsilon$ is required. However, if I understand Theorem 1 correctly, the second term in Theorem 1 becomes a constant when $\mu > n\epsilon$ since $R = \sqrt{\mu^2 + 1}$. Consequently, this counterexample may not convincingly imply that DPSGD-F is superior to DPSGD. It is recommended to address this concern during the rebuttal session.

    b) The authors use this counterexample to emphasize the importance of feature diameter. Nevertheless, it appears that it's the angle between features for different classes, rather than the diameter itself, that is crucial. Particularly, when $\mu$ is sufficiently large, the features for different classes become nearly identical because, in comparison to $\mu e_1$, $e_2$ can be neglected. To make the counter example more convincing, please consider whether the example $x = e_1 + \frac{1}{\mu} e_2$ for $y=1$ and $x = e_1 - \frac{1}{\mu} e_2$ for $y=-1$ remains consistent when $\mu$ is sufficiently large.

    c) The upper bound in Theorem 1 is dimension-dependent since the authors required that $n\geq O(\sqrt{d}/\epsilon).

2. Given that the counterexample doesn't fully convince the reviewer, it is recommended that the authors conduct an empirical evaluation of DPSGD-F on a wider range of datasets. Specifically, it would be beneficial to compare the results with Table 1 in De et al.'s work (https://arxiv.org/pdf/2204.13650.pdf) using various choices of pretrained and finetuned datasets for further validation.

### Questions
Please reference the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper looks at the effect of feature pre-processing on the DP-SGD.
Specifically, they focus on the case of linear models and aim to show theoretically and empirically the impact of feature pre-processing.
First, they give a counter-example to show the error of DP-SGD must be proportional to the maximum norm of the features.
Then, they show that carefully pre-processing the features by subtracting the mean and shifting the bias by a private quantile estimation can reduce the error to be proportional to the dimension of the dataset.
The experiments show a clear improvement in accuracy while spending a minimal privacy budget on the feature pre-processing.

### Strengths
- The paper gives strong theoretical and empirical support for feature processing, a topic previously only reasoned about empirically. 
- The privacy utility trade-off is a significant roadblock in the adoption of DP-SGD, and this paper gives a simple solution to improve this trade-off and give a nice increase in accuracy.
- The paper is very well written.

### Weaknesses
## Theory does not match experiments
The algorithm is significantly modified for the experiment section. It was clear that the algorithm analyzed was designed to make the theory work, and components were dropped or simplified for the experiments.

The part I am concerned about is the normalization of the features before computing the mean. The reason is that the normalization must be carried out in a private manner to allow for bounded sensitivity. The current version seems to assume a normalized dataset without accounting for any privacy budget to carry out the normalization. 

### Note
I was unable to verify the proofs in this paper due to my lack of theoretical background. I apologize that I leave this verification to my fellow reviewers.

### Questions
Can the authors comment on the normalization issue?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a differentially private SGD algorithm which pre-processes the features before gradient descent steps. It is shown that, in linear classification problems and under certain regimes of privacy parameters, the new algorithm converges to a solution with lower optimality gap that the standard mini-batch DP-SGD algorithm without feature pre-processing. The theoretical improvement is supported by numerical experiments on image data sets.

### Strengths
* The paper's contribution is made very easy to understand by the side-by-side comparison between standard DP-SGD's optimality gap and the new algorithm's optimality gap. The paper is overall well-written.

* The example on why DP-SGD's excess loss must scale with $R$ is simple yet effective.

* There is an extensive set of numerical experiments supporting the theoretical results.

### Weaknesses
* On the improvement of Theorem 1 over Lemma 1, it appears that the comparison boils down to $\text{diam}(D)\sqrt{\text{rank}(M)\log(1/\delta)} + R\log n$ versus $R\sqrt{\text{rank}(M)\log(1/\delta)}$. While it is clear that, when the diameter is significantly smaller than $R$, the former's first term $\text{diam}(D)\sqrt{\text{rank}(M)\log(1/\delta)}$ is dominated by $R\sqrt{\text{rank}(M)\log(1/\delta)}$, there is a lack of discussion on whether/why $R\log n$ is also dominated by $R\sqrt{\text{rank}(M)\log(1/\delta)}$. The lower bound result, Theorem 2, did not address Theorem 1's second term scaling with $R\log n$ either.

* It is ambiguous whether the improvement afforded by feature pre-processing is specific to linear classification. The term "linear optimization" in the title and the sentence " *even* for the simple case of linear classification ..." in the abstract seem to suggest that there are reasons to believe the improvement can generalize to other settings. However, there is barely discussion of other settings in this paper. If there are reasons why feature pre-processing also improves over DP-SGD in other problems (or linear optimization in general), mentioning these reasons may significantly improve the paper's contribution.

* A few minor points on writing: 
    * the phrase "maximum norm of features" in the abstract could be understood as "$\ell_\infty$ norm of features". It may help to say "maximum Euclidean norm" instead.
    * The letter $G$ in Lemma 1 is quite far from the previous mention of $G$. It might help to define $G$ as the Lipschitz constant again in the theorem statement.

### Questions
Corresponding to the two points under "Weaknesses":

* Does the claim that Theorem 1 improves over Lemma 1 require any condition on $\text{rank}(M)$, $\delta$, $n$, or other parameters?
* What is the relationship, if any, between the benefit of feature pre-processing in linear classification and the "importance of feature processing for differentially private linear optimization" as the title says?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
