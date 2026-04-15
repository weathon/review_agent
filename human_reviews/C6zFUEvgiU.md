# Feedback-guided Data Synthesis for Imbalanced Classification

- Decision: Reject
- Scores: 3, 5, 6

## Abstract
Current status quo in machine learning is to use static datasets of real images for training, which often come from long-tailed distributions. With the recent advances in generative models, researchers have started augmenting these static datasets with synthetic data, reporting moderate performance improvements on classification tasks. We hypothesize that these performance gains are limited by the lack of feedback from the classifier to the generative model, which would promote the usefulness of the generated samples to improve the classifier’s performance. In this work, we introduce a framework for augmenting static datasets with useful synthetic samples, which leverages one-shot feedback from the classifier to drive the sampling of the generative model. In order for the framework to be effective, we find that the samples must be close to the support of the real data of the task at hand, and be sufficiently diverse. We validate three feedback criteria on a long-tailed dataset (ImageNet-LT) as well as a group-imbalanced dataset (NICO++). On ImageNet-LT, we achieve state-of-the-art results, with over 4% improvement on underrepresented classes while being twice efficient in terms of the number of generated synthetic samples. NICO++ also enjoys marked boosts of over 5% in worst group accuracy. With these results, our framework paves the path towards effectively leveraging state-of-the-art text-to-image models as data sources that can be queried to improve downstream applications.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This thesis utilises recent advances in generative modelling to address the shortcomings of synthetic data in representation learning and introduces feedback from downstream classifier models to guide the data generation process. To augment static datasets with useful synthetic samples, the research designs a framework that utilises pre-trained image generation models to provide useful and diverse synthetic samples that are close to the support of real data distributions to improve the representation learning task. This paper lays the groundwork for the effective use of state-of-the-art text-to-image models as data sources that can be queried to improve downstream applications.

### Strengths
- Originality. The paper designs a diffusion model sampling strategy that uses the feedback of the pre-trained classifier to generate samples that help improve its own performance, which improves the classification performance to a certain extent. Has a certain degree of innovation.
- Quality. The experimental design of the paper is reasonable, and the feasibility of the method is verified in ImageNet-LT and NICO++. 
- Clarity. The paper well-organized and clearly written. 
- Significance. The ideas proposed in this paper have certain contributions to this field.

### Weaknesses
1. The font format of the article is not uniform. Do the words in italics want to express any special meaning? Make it difficult for readers to read.
2. The charts are mixed up, for example, Figure 5. Is it a table or a graph? The sizes of some pictures also don’t match.
3. How about the time complexity of this method?
4. Are there more evaluation metrics to evaluate the performance of the proposed method versus the baseline method?

### Questions
Please refer to the weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
With the recent advances in generative models, researchers have started augmenting these static datasets with synthetic data, reporting moderate performance improvements on long-tailed classification tasks. 
The authors hypothesize that these performance gains are limited by the lack of feedback from the classifier to the generative model, which would promote the usefulness of the generated samples to improve the classifier’s performance.
In this work, the authors introduce a framework for augmenting static datasets with useful synthetic samples, which leverages one-shot feedback from the classifier to drive the sampling of the generative model. 
For the framework to be effective, they find that the samples must be close to the support of the real data of the task at hand and be sufficiently diverse. 
The authors validate three feedback criteria on a long-tailed dataset (ImageNet-LT) and a group-imbalanced dataset.

### Strengths
1. The problem definition to encourage the generated samples to be helpful to the classifier, inspired by active learning frameworks, is novel.
2. The proposed method performs better than the previous sample synthesis-based imbalance classification methods.

### Weaknesses
- The proposed solution for the problem definition is too naïve. For active learning methods, in addition to the confidence-based or entropy-based approach, margin margin-based approach is also possible. For the recent active learning criteria, such as BALD [1], VAAL [2], or MCDAL [3]. To claim the contribution of a complete research paper, the authors should devise an idea to leverage such recent active learning methods to find more novel solutions suitable for this problem.
[1] Deep Bayesian Active Learning with Image Data. ICML 2017.
[2] Variational Adversarial Active Learning. ICCV 2019.
[3] MCDAL: Maximum Classifier Discrepancy for Active Learning. TNNLS 2022.

- Also, instead of simply comparing among naïve active learning criteria, how about combining multiple losses (at least linear combination in the loss)? That would be more novel than the proposed solution.

- The experiment is also too weak. For the datasets, The authors only use ImageNet and NICO++. However, according to other recent Long-tailed recognition papers, they usually evaluate their methods on iNaturalist and Place-LT datasets to demonstrate the scalability. At least the authors should have evaluated their method on CIFAR datasets to show the effectiveness of their methods on other datasets.

- Also, a comparison with more recent state-of-the-art long-tailed recognition papers is missing. For example, CMO [4] is one of the recent long-tailed recognition methods based on sample synthesis. To claim the usefulness of the proposed method, the authors should compare the proposed method with recent long-tailed recognition papers, including [4].
[4] The Majority Can Help The Minority: Context-rich Minority Oversampling for Long-Tailed Classification. CVPR 2022.

- More analysis of the detailed design choices. For example, how are the hyper-parameters decided, such as w in Eqns (5), (6), (8)? As the authors proposed to add additional criteria, it would be necessary to analyze the effect of w on the performance.

### Questions
Please refer to the questions in the weakness.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The effectiveness of utilizing synthesized data is limited by the lack of feedback. This work proposes a framework to drive the sampling process of a generative model, thereby improving the usefulness of the generated samples.

### Strengths
● the experimental results were stunning, achieving state-of-the-art on ImageNet-LT
● the writing is clear and easy to follow
● the experiment is comprehensive, comparing three types of feedback criteria

### Weaknesses
ImageNet-LT is essentially a pseudo long-tail dataset, where the tail classes may not necessarily be the minority in the actual data distribution. Therefore, generative models can sample relatively well. However, for real-world long-tail distributions, is it also difficult for generative models to obtain sufficiently good samples?

### Questions
ImageNet-LT is essentially a pseudo long-tail dataset, where the tail classes may not necessarily be the minority in the actual data distribution. Therefore, generative models can sample relatively well. However, for real-world long-tail distributions, is it also difficult for generative models to obtain sufficiently good samples?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
