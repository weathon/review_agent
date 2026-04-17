# Machine Unlearning in Low-dimensional Feature Subspace

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Data privacy in modern neural networks attracts growing interest in Machine Unlearning (MU), which aims at removing the knowledge of particular data from a pretrained model and meanwhile maintaining good performances on the remaining data.
In this work, a new perspective upon low-dimensional feature subspaces is presented to investigate MU.
We firstly demonstrate the potentials of separating the remaining and forgetting data in a low-dimensional feature subspace.
Then, such separability motivates us to seek a subspace ${\tt range}(\bf U)$ on the features of the pretrained model for unlearning, where the information of the remaining data is preserved and that of the forgetting data is therein diminished, leading to the proposed new method  named SUbspace UNlearning (SUN).
Compared to mainstream MU methods that require direct and massive access to the training data for model updating, SUN offers two key advantages well resolving these significant challenges in practice.
(i) SUN avoids frequent data visits and optimizes $\bf U$ involving two covariance matrices, which only requires one-shot feature fetching and thereby alleviates data privacy risks and computation.
(ii) SUN in implementation simply serves as a plug-in module to the pretrained model without modifications to its original parameters, reducing the parameter number and  computational overhead by orders of magnitude, which is of great practicality for handling multiple unlearning requests.
Extensive numerical experiments verify our superior unlearning accuracy with significantly less parameters and  computing time over variants of models, datasets, tasks, and applications. 
Code is available at the anonymous link https://anonymous.4open.science/r/4352/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a feature-level plug-in module to achieve machine unlearning. The core idea is based on the observation that features from different data sources exhibit stronger separability at the feature level. Leveraging this property, the method projects features into a subspace where the representations of the data to be forgotten are minimized as much as possible. Finally, the features are projected back into the original space to enable subsequent inference.

### Strengths
1. The paper exhibits a well-structured organization. Particularly in sections introducing the methodology, the presentation logically unfolds the implementation process of the proposed method.

2. The experimental part provides well-structured comparisons. Through these comparisons, the differences and enhancement effects of the proposed SUN method in critical indicators.

### Weaknesses
1. The design of such a plug-in seems to contradict the machine unlearning, and I still remain convinced about this approach. The target of unlearning is to resolve the impact of the forgetting data. However, plug-in only introduces a new modules and the vanilla weights are still saved in the model. If attackers continue to concatenate these to form a new model, will it not still be capable of recovering the ability to acquire the forgotten knowledge? It appears that this solution cannot be deemed effective when viewed from this perspective.

2. The author claims that the motivation comes from the idea that different data points are more separable in high-dimensional feature spaces. However, this claim does not seem to be theoretically proven or empirically validated in the paper. Moreover, this motivation is not necessarily intuitively correct. For classification tasks, where data are separated based on class labels, this reasoning might hold to some extent. However, for broader types of tasks, the motivation may not be valid. The discussion of this point in the paper is not sufficiently in-depth.

3. Even assuming that the motivation is partially correct, how should one decide which features to use and from which layer? The paper lacks sufficient ablation studies on the application of this module, as well as relevant comparative analyses. For example, how do features from different layers differ? How should the corresponding configurations be chosen under different model architectures?

4. The experimental performance does not appear to be sota. In some experiments, the results are inferior to those of existing methods.

5. The PCA projection method is overly simplistic. If the dataset is large in scale, PCA may not be feasible in terms of dimensionality. For example, when separating 10w data points in a 1k-dimensional subspace, substantial overlap is likely to occur.

### Questions
1. Could the authors provide some theoretical justification and empirical validation for this motivation? At present, this motivation does not seem self-evident.

2. Could the authors discuss the separation efficiency of the features after PCA? For instance, they could use cosine similarity or other metrics to illustrate the relationship among dimensionality, dataset size, and separation efficiency.

3. I noticed that the experiments in the paper mainly focus on classification baselines. This is related to the issues I mentioned earlier — the separability of the feature space is inherently tied to the category-based nature of classification tasks. However, for tasks in other domains, such as NLP, this approach may not be effective, since the feature spaces in those cases may not exhibit strong separability. Could the authors provide some relevant experiments to support the generality of their method?

4. Could the authors evaluate the performance of their method on some smaller networks, such as encoder–decoder or VAE architectures? The effectiveness of PCA becomes limited when dealing with a large number of categories or high-dimensional vectors. In current large-scale language models, the vocabulary size often reaches 100K or even 1M, and the vector dimensionality can exceed 10K. How does this method affect the transferability and performance of such large-scale models?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper uses feature learning to separate data that needs to be forgotten from the rest. This is a paper joining the growing line of literature for machine unlearning. The main idea is to use pretrained NN (except the last linear layer) architecture to learn the feature separation between remaining and leaving data. Then find a projection matrix using principle component analysis which can be added to the unlearned model to minimize effect of leaving data.

### Strengths
The topic of machine unlearning is very timely and important. If all the hypothesis and observations are true in the paper, then the proposed technique could serve as a good method for machine unlearning. It's great that the authors tried to apply SUN at different layers.

### Weaknesses
The paper is entirely built upon the observation and hypothesis that exists a low-dimensional feature subspace, where the features of forgetting and remaining datasets are easy to be separated. There is no justification for why this assumption makes sense. 

On a related note, why is it a good idea to fix g after pretraining? After pretraining, g(.) might be too restrictive and the separation may not be possible to do depending on the problem. 

The paper lacks any theoretical guarantees for the proposed methods. Not only it's not a certified removal, there is also no guarantee on how close to being certifiable it can be. If there is a PAC type of guarantee, this would be a lot more acceptable. 

The paper is poorly written. For instance, how exactly the computation for Z, $\Sigma$ are implemented? What is the relevance of $g$ and $h$?

### Questions
Could the authors comment on how their method compares against other certified unlearning methods such as the following? Both in terms of removal success and time complexity?

Zhang, Binchi, et al. "Towards certified unlearning for deep neural networks." arXiv preprint arXiv:2408.00920 (2024). 
Qiao, Xinbao, et al. "Hessian-Free Online Certified Unlearning." arXiv preprint arXiv:2404.01712 (2024).

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SUN (SUbspace UNlearning), a new approach to machine unlearning that operates in a low-dimensional feature subspace rather than directly updating model parameters. SUN learns a projection matrix on the pretrained model’s penultimate features to preserve information relevant to the remaining data while suppressing information related to the forgotten data. The resulting projection acts as a plug-in module without modifying the original model parameters. Experiments across multiple architectures and datasets show that SUN achieves comparable or superior unlearning performance to existing methods while reducing parameter updates and runtime by orders of magnitude.

### Strengths
- Paper is well-organized and easy to follow.
- Extending experiments to instance unlearning 
- Experiments extend to instance unlearning and face and emotion recognition, showing the effectiveness of the proposal under varying image classification tasks. 
- The proposed method is effective in continual unlearning scenarios.

### Weaknesses
- I think the analysis of the eigenvalues is incorrect. Having similar eigenvalues does not mean that subspaces collapse and are not separable. Similar eigenvalue information does not imply that principal directions are shared, too. We can not comment on this just by looking at the eigenvalues (corresponding directions matter). Moreover, we know that the penultimate hidden states of the model trained on the full data are separable: we apply a linear transform in the last layer, and the model has a high classification accuracy. Therefore, I do not agree with the indistinguishability argument. 
- The proposed method does not have theoretical grounding/guarantees. It is based on observations (also see the previous point) and supported with numeric results. 
- MIA used in the paper is a black-box attack. A white box attack would get a very high performance, which makes me reconsider the success of unlearning.

### Questions
See weaknesses.

### Soundness
2

### Presentation
4

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
This paper introduces SUN, a novel and efficient method for machine unlearning based on low-
dimensional subspace projection. Instead of retraining or distilling models, SUN identifies a feature
subspace that retains information from remaining data while removing information associated with
forgetting data.
The key idea is to learn a projection matrix that minimizes the covariance of forgetting features while
maximizing that of the remaining features, i.e., effectively finding an orthogonal subspace that forgets
undesired information. The optimization is performed on the Stiefel manifold, and the learned projection
is inserted as a plug-in layer between the pretrained model’s backbone and classifier.

### Strengths
Unlike parameter-space or data-space methods, SUN reinterprets unlearning as a geometric operation in
feature space. The subspace-based view of machine unlearning is both elegant and novel. SUN requires
only one forward pass to extract features, then computes a projection matrix offline — making it orders
of magnitude faster and more lightweight than retraining-based baselines and achieving one-shot
unlearning by requiring only feature-level access. The authors evaluate on multiple datasets and metrics
(accuracy, MIA gap, Avg.G), demonstrating consistent performance. The ablations on loss terms and
subspace dimensionality are convincing.

### Weaknesses
Although SUN eliminates the need for retraining, it still requires a complete forward pass through the
remaining dataset to estimate the feature covariance matrices. This may become computationally
expensive for very large-scale datasets or repeated unlearning requests. And because of the dependence
on the remaining data, I question whether it's reasonable to focus the comparison more on baselines that
only use the forget dataset.
Besides, the separability assumption (that forgetting and remaining features lie in distinct covariance
subspaces) is intuitively reasonable but lacks a formal bound or generalization guarantee, for example,
what if the forget data shares a large similarity with the remaining data?

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2
