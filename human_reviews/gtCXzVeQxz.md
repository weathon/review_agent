# PHICO: Personalised Human-AI Cooperative Classification Using Augmented Noisy Labels and Model Prediction

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 8

## Abstract
The nuanced differences in human behavior and the complex dynamics of human-AI interactions pose significant challenges in optimizing human-AI cooperation. Existing approaches tend to oversimplify the problem and rely on a single global behavior model, which overlooks individual variability, leading to sub-optimal solutions. To bridge this gap, we introduce PHICO, a novel framework for human-AI cooperative classification that initially identifies a set of representative annotator profiles characterized by unique noisy label patterns. These patterns are then augmented to train personalised AI cooperative models, each tailored to an annotator profile. When these models are paired with human inputs that exhibit similar noise patterns from a corresponding profile, they consistently achieve a joint classification accuracy that exceeds those achieved by either AI or humans alone. We theoretically prove the convergence of PHICO, ensuring the reliability of the framework. To evaluate PHICO, we introduce novel measures for assessing human-AI cooperative classification and empirically demonstrate its generalisability and performance across diverse datasets including CIFAR-10N, CIFAR-10H, Fashion-MNIST-H, AgNews, and Chaoyang histopathology. PHICO is both a model-agnostic and effective solution for improving human-AI cooperation.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes the human-ai cooperation framework *Phico* for classification. It considers the interaction between AI and humans to make predictions, incorporating methods from noisy label and multi-annotator learning to personalize the decision process toward multiple annotator profiles. The authors provide theoretical proof of convergence (in the appendix) and a new metric, called alteration rate, that they use to quantify the impact of their framework.

### Strengths
- The problem of human-ai cooperation is relevant and interesting. 
- The framework itself seems novel.

### Weaknesses
- **Presentation of the framework:** I believe that the introduction needs to be improved. The framework in Figure 1 includes numerous elements that lack clarity, such as the green, blue, and orange matrices with entries that are occasionally underlined. Additionally, the individual components are insufficiently explained, while the final box (3) remains unexplained. The current design of this figure doesn’t fit well with the introduction, as its many components make it challenging to understand Phico's exact process and contributions.
- **Related Work:** The related work on LNL and MRL is brief and lacks depth. Expanding this section, e.g., by removing subsections and adding more details *and* content, could improve clarity. Additionally, some key references are incorrectly cited; for instance, the Crowdlab paper lacks its conference/journal details, and the VIT paper is cited on arXiv despite its acceptance at ICLR. These minor errors accumulate throughout the paper.
- **Convergence Proof:** One contribution of the paper, the proof of convergence, is presented very briefly in the main text. While placing a lengthy proof in the appendix is acceptable, highlighting this as a contribution requires a more thorough discussion of the proof and its implications within the paper itself.
- **More intuitions in the Methodology:** The methodology for the first three components (Annotator Profiles, Augmentations, and Personalized Model) is somewhat vague and would benefit from more intuitive explanations. Additionally, the many components make it challenging to follow the authors' arguments and underlying intuitions.
- **Applicability in Inference:** A crucial concern is the applicability and realism of the framework, especially for making predictions. Currently, the paper states:
    
    > To classify a testing user into one of the K profiles, we first ask the user to label each image in a validation set…
    > 
    
    If I understand this correctly, each new user must annotate a validation dataset before profiling and predicting is possible. This approach seems extremely expensive and unrealistic.
    
- **Minor problems:**
    - Misuse of `\cite` and  `\citep`. For example, in the introduction in the 3rd paragraph, the authors should use parenthesis around the citations.

### Questions
Can you provide feedback on the weaknesses I mentioned?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose **PHICO** as a new human-AI collaboration approach for classification in the area of learning-to-complement. The main idea is to combine noisy predictions of machine learning models and humans to obtain accurate estimates for instances' ground truth class labels. As a result, humans are not only tasked to annotate training instances but also to contribute (noisy) labels during inference, i.e., they annotate test instances.

The **training** of PHICO (theoretically shown to converge) consists of three steps.

- (1) At the start, annotator profiles are identified by clustering (via fuzzy $K$-means) the humans' individual noisy annotation patterns. Each of these noisy annotation patterns is a vector whose entries indicate potential disagreements between a single human's label and the consensus label of an instance. For example, the vector could indicate that a human tends to confuse images of trucks with cars. Each obtained cluster of humans is expected to represent one profile of a typical annotation pattern.
- (2) For each given profile, a data subset is constructed by collecting all annotations from humans belonging to the same profile. Then, a transition matrix is computed for each profile by comparing the humans' noisy labels with the consensus labels. Since a profile's data subset may contain only a few instances, a noisy label augmentation approach is proposed to increase the subsets' size. For this purpose, noisy labels are sampled according to the previously estimated transition matrix.
- (3) Finally, the profiles' data subsets serve as training data to jointly optimize the parameters of base models (corresponding to standard classification models), human label encoders (characterizing humans' annotation patterns), and combination models (estimating the joint distribution of the two previous model types). An extension of the cross-entropy is used as the loss function, where the consensus labels and noisy labels serve as targets.

For the **inference** of a test instance's class label, a human is queried to provide an annotation. Further, this human must be assigned to one of the profiles learned during training. This matching is obtained by requiring each human to label a validation set. Finally, the combination models belonging to the matched profile can combine the human's and the base model's predicted class label to estimate the test instance's ground truth class label.

In an extensive **empirical evaluation**, PHICO shows improved performance compared to related human-AI collaboration and noisy label learning approaches for datasets with real-world and simulated humans as annotators. Further, the authors explicitly demonstrate that PHICO majorly corrects the noisy labels of humans and base models.

### Strengths
**Major strengths:**

- **Human-AI collaboration** is a highly relevant topic and cannot only lead to performance improvements but also to more trust in AI since decisions take human judgments into account \[1\].
- The general **idea** of combining noisy class labels from humans and machine learning models at inference (test time) in a multi-rater noisy label setting seems novel.
- The **empirical evaluation** study shows performance gains across many real-world datasets with noisy class labels from multiple error-prone humans and includes ablations of PHICO's components, e.g., the number of annotator profiles, the regularization parameter of the loss function, and the noisy label augmentation.

**Minor strengths:**

- A new (simple) **evaluation score** is introduced, which enables the study of the benefit of human-AI collaboration. Concretely, it indicates how often class labels of humans are altered toward a correct or false class label.
- **Detailed analyses** show that PHICO can even predict correct class labels when both the base model and the human are wrong.

**References:**

- [1] Westphal, Monika, et al. "Decision control and explanations in human-AI collaboration: Improving user perceptions and compliance." Computers in Human Behavior 144 (2023): 107714.

### Weaknesses
**Major Weaknesses:**

- The matching between a test user and a profile (determined during training) seems to require a clean **validation set** (cf. definition of $\\mathcal{V}$ containing true labels $\\boldsymbol{y}\_i$ in Section 3.3). In practical applications, such a requirement can be highly limiting for two reasons. First, it is unclear how a validation set with clean labels is obtained. If an expert is available, giving such a task to this expert could lead to high annotation costs. Second, if each test user has to annotate this validation set, this also can lead to high annotation costs. For example, having 80 test users annotating a validation set of 200 instances (setup for CIFAR-10N) would correspond to 16,000 additional annotations.
- The **related work** covers the three domains of human-AI collaboration, noisy label learning, and multi-rater learning. Due to space limitations, I understand that discussing all different approaches in detail is infeasible. However, I think that particularly the multi-rater learning is not adequately represented. For example, CrowdLab, referred to as key development, seems to be a lesser-known approach (there is even no venue given in the references). Instead, there are many well-known [1, 2, 3] and recent [4, 5, 6, 7] approaches which are not mentioned.
- As a result of the sparse discussion on related work in multi-rater learning, the **evaluation** lacks an appropriate comparison to corresponding approaches. For example, more prominent multi-rater learning approaches could have been evaluated in analogy to the noisy label approaches in Table 4. Further, it is unclear why the multi-rater and noisy label learning approaches were only evaluated on CIFAR-10 with asymmetric label noise and not on the datasets CIFAR-10N, CIFAR10-H, FashionM-H, and Chaoyang. For example, the study [8] shows the evaluation of noisy label approaches on CIFAR10-N, while the study [4] presents results for multi-rater approaches on CIFAR-10N.
- One contribution is the **proof of the convergence** of PHICO's training. However, this contribution is not adequately presented in the main paper (there are only three lines referring to the proof in the Appendix). Moreover, after inspecting this appendix, the actual novelty of this contribution is questionable. According to my understanding, the first part of the proof uses the already proven fact that fuzzy $K$-means converges, while the second part generally applies to smooth loss functions with Lipschitz continuous gradients. The third part mainly states that if both previous steps converge, the overall training converges. Accordingly, I wonder whether the convergence proof for PHICO actually targets an open problem. As a result, I would only see this convergence proof as an important contribution, if it would provide unknown convergence rates or relaxed conditions, for example.

**Minor Weaknesses:**

- The computation of the transition matrix corresponds to a **class-dependent assumption**. Although such an assumption is made by several related approaches in noisy label [9] and multi-rater learning [1, 2], it does not reflect the actual instance-dependent annotation patterns of humans. Moreover, other works [6, 7] already propose solutions to estimate this instance-dependency. Accordingly, I would expect a brief reasoning why a class-dependent modeling has been selected.
- Since it is typical that a new approach cannot resolve all issues regarding a learning task, I would expect a detailed discussion of current **limitations**. Yet, Section 6 discusses very briefly potential extensions of PHICO (e.g., modeling of evolving annotation patterns). Accordingly, many aspects, e.g., the requirement for a validation set, modeling of only class-dependent annotation patterns, and requiring training users to provide a certain number of labels per class to compute label vectors for clustering, could have been discussed more transparently.
- It seems that **no code** has been provided for review, and there is no indication of whether the code will be made publicly available.

**References:**

- [1] Rodrigues, Filipe, and Francisco Pereira. "Deep learning from crowds." Proceedings of the AAAI conference on artificial intelligence. Vol. 32. No. 1. 2018.
- [2] Tanno, Ryutaro, et al. "Learning from noisy labels by regularized estimation of annotator confusion." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
- [3] Cao, Peng, et al. "Max-MIG: an Information Theoretic Approach for Joint Learning from Crowds." International Conference on Learning Representations. 2019.
- [4] Zhang, Hansong, et al. "Coupled confusion correction: Learning from crowds with sparse annotations." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
- [5] Ibrahim, Shahana, Tri Nguyen, and Xiao Fu. "Deep Learning From Crowdsourced Labels: Coupled Cross-Entropy Minimization, Identifiability, and Regularization." The Eleventh International Conference on Learning Representations. 2023.
- [6] Guo, Hui, Boyu Wang, and Grace Yi. "Label correction of crowdsourced noisy annotations with an instance-dependent noise transition model." Advances in Neural Information Processing Systems 36 (2023): 347-386.
- [7] Herde, Marek, Denis Huseljic, and Bernhard Sick. "Multi-annotator Deep Learning: A Probabilistic Framework for Classification." Transactions on Machine Learning Research. 2023.
- [8] Wei, Jiaheng, et al. "Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations." International Conference on Learning Representations. 2022.
- [9] Zhang, Yivan, Gang Niu, and Masashi Sugiyama. "Learning noise transition matrix from only noisy labels via total variation regularization." International Conference on Machine Learning. PMLR, 2021.

### Questions
- Eq. (2) computes the transition matrix for profile $k$ by comparing the noisy class labels of all annotators $\\mathcal{A}\_k$ in profile $k$ to the corresponding consensus labels. Accordingly, the sum iterates over all the class labels provided by the annotators $\\mathcal{A}\_k$. However, the sum is only normalized by $|\\mathcal{A}\_k|$ as the number of annotators in profile $k$. Is this correct, or should it be the number of annotations?
- Table 12 shows the performances for different values of the regularization hyperparameter $\\lambda$ and indicates that $\\lambda=0.1$ works best. Is this value also used for the main experiments in Section 4?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces PHICO, a human-AI cooperation framework for classification tasks. The method works by creating annotator profiles for the labelers, using these profiles to generate additional labels, and then training a model specific to each profile. The framework is validated through theoretical convergence proofs and experimentation across multiple datasets, showing better performance than several baselines. The authors also introduce the metric, “alteration rate”, to measure the impact of AI predictions on accuracy.

### Strengths
1. This paper is addressing an interesting and important topic of how to best combine the prediction powers of humans and AI in a cooperation framework. 

2. The idea to learn distinct user profiles based on how the users labels, and then training models specific to each profile is compelling. 

3. The authors included a proof of convergence for their method, which adds a nice layer of rigor to their method. 

4. The experiments seem robust, at least in terms of the number of previous methods compared to. 

5. They test their method on many different datasets.

### Weaknesses
1. The paper could benefit from clearer writing - both at a high level explaining the problem setting and method, and by providing details about their implementation. In particular how the annotator profiles are constructed. 

2. The inclusion of a proof is nice, but in my understanding it primarily relies on established results for gradient descent and fuzzy k-means - so I don’t think there is much theoretical contribution. 

3. The ‘alteration rate’ metric, while useful, seems fairly straightforward measure of how labels change given their method - I don’t believe this is particularly novel. 

4. The paper does include many comparisons to many previous methods/baselines - but it seems like this process could have a lot of variance so seeing the results run over several rounds would be much more convincing than a single reported experiment.

### Questions
1. Could you clarify how the alteration rate metric improves on or differs from existing metrics in human-AI cooperation?

2. Can you provide statistical measures, like means and standard deviations, for the experimental results? Particularly for Table 3. Or is there a reason this isn’t feasible? 

3. How would PHICO handle situations where annotator behavior changes over time?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Summary: This paper presents a methodology to profile annotators so as to learn more fine-grained models for human-AI collaboration in the context of multi-class classification data annotation. At its heart, the idea is to use K-Means clustering to divide the users (and the dataset) into K profiles (or categories/types). These profiles are learned at training time, and associated human-AI collaboration model is learned to assist with labeling. At test time each user is assigned to one profile, which sets which collaborative model will assist them in annotation. The main contribution here is an extensive analysis of this approach including---(1) Data augmentation to account for incomplete data problem caused by profiling, (2) designing appropriate metrics to measure the performance, (3) Extensive evaluation across multiple datasets.

### Strengths
Strengths:

1. Very useful problem: With the advent of general LLMs and AI models, incorporating AI into human workflows is as important as can be. So, building frameworks for fundamental improvements in humanAI collaborations are both important and timely.

2. Well motivated, and reasonable approach: The idea is very simple, which is a feature and not a bug. It's well thought out, and well implemented. The ideas discussed and the approach can be of use to the community to shape the field. 

3. Performance is good: It is mostly an incremental improvement above existing methods, but the experiments are quite exhaustive. All in all, while it may not evoke a paradigm shift in the field, it's solid enough to push the field forward. The experiments are shown across a wide variety of benchmarks, and largely the trend is consistent---PHICO works well.

4. Well written: The presentation of the content is such that a reader can easily understand it. This is quite helpful, especially given the diverse readership this paper is expected to have.

### Weaknesses
1. Figures need improvement: Figure 1 does not do justice to explaining the approach. The caption is incomplete, and the figure itself is not very high quality. I would advise improving the caption to explain the reader what is happening here. I could understand it only after re-reading the paper a couple times.

2. Statistical testing for PHICO: Largely, the performance gains of this approach over the baselines is marginal. Within a couple percentage points in most places. Thus, reporting error bars and statistical testing would go a long way to show that it actually improves performance over the baselines. Currently, the claim is not based in solid statistical reasoning.

3. Essentially, Annotator Profiles just mean dividing Annotators into types. It would be interesting to see what kind of types are discovered. Are there demographic patterns? Are there patterns in terms of their habits? How different are these truly?

### Questions
Please see above.

### Soundness
3

### Presentation
2

### Contribution
3
