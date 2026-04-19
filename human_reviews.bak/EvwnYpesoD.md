# A Bias-Variance-Covariance Decomposition of Kernel Scores for Generative Models

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
Generative models, like large language models, are becoming increasingly relevant in our daily lives, yet a theoretical framework to assess their generalization behavior and uncertainty does not exist.
Particularly, the problem of uncertainty estimation is commonly solved in an ad-hoc manner and task-dependent.
For example, natural language approaches cannot be transferred to image generation.
In this paper, we introduce the first bias-variance-covariance decomposition for kernel scores and their associated entropy.
We propose unbiased and consistent estimators for each quantity which only require generated samples but not the underlying model itself.
As an application, we offer a generalization evaluation of diffusion models and discover how mode collapse of minority groups is a contrary phenomenon to overfitting.
Further, we demonstrate that variance and predictive kernel entropy are viable measures of uncertainty for image, audio, and language generation.
Specifically, our approach for uncertainty estimation is more predictive of performance on CoQA and TriviaQA question answering datasets than existing baselines and can also be applied to closed-source models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on the evaluation of generative models’ performance. Generalization and uncertainty are two primary consideration and the bias-variance-covariance decomposition are introduced for kernel scores and entropy. The performance is examined on vision, voice and text datasets.

### Strengths
This paper is clearly written and the studied topic is crucial in large language models.
The author develops the distributional correlation as a tool to insight into the generative model’s fitting.
The author performed extensive experiments to validate the effectiveness of the developed tool.

### Weaknesses
See the question part below.

### Questions
Can you explain the reason why this method works best on question answering datasets?
I’m wondering how to generalize the theory to others except the kernel scores.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a method to analyze the performance of a generative model in terms of bias (compared to the true distribution) vs. variance (with respect to replications of the model on independent training data), using kernel methods to estimate distributional distances.  It develops theory which defines RKHS-based variance scores and develops simple U-statistic-like estimators.  They demonstrate their approach in image generation (Infinite MNIST), Audio Generation (LJSpeech), and Natural Language Generation examples.  In image generation example, the method shows that the variance stays high and is reduced throughout training, while bias quickly converges.

### Strengths
This paper contributes a useful theoretical tool.  it is clearly written.  The image generation and audio generation examples demonstrate that the method can provide useful insights into the properties (e.g. distributional diversity, stability) of the generative models.  For example, the paper demonstrates some suggestive evidence for how mode collapse affects only the bias but not the variance.  We see that using predictive kernel entropy of LLM answers to predict answer quality can be a useful tool, and compares favorably to lexical similarity and semantic entropy.

### Weaknesses
Practitioners may find it challenging to apply this method to their case.  It is not clear what considerations should guide selection of a kernel.
Some details about the experiments are missing, which make it harder to interpret the results (see Questions).

### Questions
1. In figure 4, what are the points being plotted?  The 20 different models?
2. Why is only a single model used to compute kernel entropy? Are there any the advantages of using multiple models to compute the entropy?
3. How is the bias calculated in the mode-collapse experiment of section 5.1?  Is the original distribution used to compute the bias, or the modified distribution with reduced frequency of class 0 used?
4. Why use RBF kernel for images, but Laplace kernel for audio?
5. Why do we see an entropy increase as training progresses for audio, but not for images?  Could it be an artifact of the choice to use a different kernel for images vs audio?
6. In the language model experiment, does the superiority of using kernel entropy for predicting correct answers depend on the choice of kernel?  How are the AUC scores for Laplace kernel?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors propose the bias-variance-covariance decomposition for kernel scores, which is an unbiased and consistent estimator for uncertainty measurement in deep generative models such as large language models. Their approach only requires samples from the predictive distribution rather than the distributions themselves, which means that we can evaluate all terms in the composition for any deep generative model. 
Empirically they evaluate their approach on models with image and video data and large language models, and they showed that their kernel  entropy outperforms other baselines.

### Strengths
This paper is well-motivated; It is worth studying the generalization and uncertainty characterization in large language models. The paper is clearly written and well structured. To my knowledge, the math derivation in the methodology is sound. I found it useful that their estimator only would require samples from the predictive distributions rather than those distributions themselves, which has a large range of applications. Also, they provide comprehensive evaluations of both image data and text data.

### Weaknesses
The authors claim that one of the advantages of their uncertainty estimator is that it includes a covariance term. I would hope to see a more detailed discussion on this. For example, how that covariance term characterizes uncertainty that cannot be done by prior work.

### Questions
1. How do different sample sizes affect the empirical results? In Figure.2 there is a fixed sample size. I wonder how to tune that in general.
2. Is there any difference in terms of the outperformance of your estimator when you work on image data and text data?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
The authors propose a bias-variance-covariance decomposition for kernel scores, applying this decomposition to generative models. In particular they aim at assessing uncertainty of predictions by looking at the derived variance term (and the related predictive kernel entropy) as measures of uncertainty.

### Strengths
- The work is well-motivated, since evaluating predictive uncertainty for generative models is a relevant problem. 
- The writing of the paper is clear and easy to follow. 
- The results in Section 5.3 indicate that the proposed approach favourably compares to recent relevant work on assessing predictive uncertainty of LLMs in QA tasks.

### Weaknesses
Since the paper falls far from my area of expertise, and I am not sufficiently familiar with the mentioned related work, I'll refrain from commenting on the validity of the theoretical/methodological contribution.  However, specifically to the empirical results in the paper:
- A large part of the empirical results (aside from Section 5.3) do not seem to contain important and conclusive insights. The main take-away seem to be distributional variance showing correlation with MMD, while many other observations such as on the stability of training do not seem to be as valuable.
- "This includes the discovery that mode collapse of underrepresented minority groups is expressed purely in the bias." made as a claim in the Introduction should be probably revisited. As far as I understand, the authors conclude this from their results on a single model and a single experiment, which is not enough to make a general statement. 
- While the paper is largely motivated by the need for evaluating predictive uncertainty for generative models, the only assessment of the approach for this purpose is made for LLMs on QA task (Section 5.3). The authors claim that their proposed uncertainty measure is applicable to a large range of generative models, and that this is an advantage over previously proposed methods, so it would be important to test its effectiveness in assessing predictive uncertainty for other generative tasks (e.g. image generation).

### Questions
To fully understand the setup and results in Section 5.3 I had to open the work from which the setup is replicated [1]. I think the section would benefit from a more thorough introduction of the experiment, in order to be self-contained. 

[1] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations, 2023.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
