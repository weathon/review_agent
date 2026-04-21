# Self-Supervised Detection of Perfect and Partial Input-Dependent Symmetries

- Avg Score: 4.75
- Decision: Reject
- Scores: 6, 5, 3, 5

## Abstract
Group equivariance ensures consistent responses to group transformations of the input, leading to more robust models and enhanced generalization capabilities. Nevertheless, this property can lead to overly constrained models when the symmetries considered in the group differ from those observed in data. While common methods address this by determining the appropriate level of symmetry at the dataset level, they are limited to supervised settings and ignore scenarios in which multiple levels of symmetry co-exist in the same dataset. For instance, pictures of cars and planes exhibit different levels of rotation, yet both are included in the CIFAR-10 dataset. In this paper, we propose a method able to detect the level of symmetry of each input without the need for labels. To this end, we derive a sufficient and necessary condition to learn the distribution of symmetries in the data. Using the learned distribution, we generate pseudolabels that allow us to learn the levels of symmetry of each input in a self-supervised manner. We validate the effectiveness of our approach on synthetic datasets with different per-class levels of symmetries e.g. MNISTMultiple, in which digits are uniformly rotated within a class-dependent interval. We demonstrate that our method can be used for practical applications such as the generation of standardized datasets in which the symmetries are not present, as well as the detection of out-of-distribution symmetries during inference. By doing so, both the generalization and robustness of non-equivariant models can be improved. Our code is publicly available at \texttt{url-removed-for-double-blind-review}.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work build upon the IE-AE architecture, enhancing it so that the distribution of observable group elements (assumed uniform) is learnt. This is effective when only partial symmetries exist in the data, even in the case where each class has a different group element distribution acting on it. This work focuses solely on the SO(2) group (continuous rotations). An additional network $\Theta$ that predicts the rotation is included on top of IE-AE, as well as a strategy based on Nearest-Neighbors to estimate the boundary of the uniform distribution.

The overall method is very well presented, and nicely backed with theory. The experiments on MNIST-like datasets also back the theory up, showing accurate learning of the symmetry boundaries.

### Strengths
* The theoretical derivation is sound and solid.

* The improvement over IE-AE seems reasonable from an architectural point of view. 

* The MNIST-like data experiments nicely show the properties of the method.

### Weaknesses
* Although the spirit of this work is fundamentally setting the grounds of the technique and the theory behind it, I believe experiments should explore data beyond MNIST. In that sense, I would strongly suggest the authors to explore (at least) CIFAR-like data, with 32x32 RGB images.

* There are some unanswered questions related to data that is inherently invariant to SO(2). For example, the Flowers dataset contains top-down photos of flowers, known to be almost invariant to rotation. Some experiment showing the limitations of the method (in this sense or another) would be extremely welcome.
  * In a similar vein, another way to show the limitations of the method (and strengths hopefully!) is to perform an experiment on data that does not strictly obey a uniform distribution of group elements.

* The work lacks an ablation study, given that several losses are combined, and there exist additional parameters such as $k$. Additionally, in the appendix there is mention to an additional consistency loss that is not mentioned in the main body, which I think should be amended for scientific rigour.

### Questions
* The estimation of the symmetry boundary using $E=2 \overline{|\psi(N_k(x))|}$ is tailored for distributions of the form $U(-\theta,\theta)$. Have the authors thought about generalizing the method to arbitrary distributions? Even if it is only the case of arbitrarily bounded uniform distributions. In such case, the concept of "center of symmetry" as explained in the paper would not be valid anymore. Although uniform symmetry distributions are sensible, multimodal ones could naturally arise in datasets too.
     * Related to this question, one experiment that would be interesting is to have non-overlaping rotation intervals per class (instead of all being subsets of the next class as in MNISTMultiple). 

* As far as I understand, the center of symmetry is an abstraction for "canonical rotation", which could be any absolute angle in practice. Therefore, the proposed method cannot estimate the absolute rotation of an image, but only the rotation wrt. its canonical one. Is this correct? 
  * What is the role of $L_2$. In general, it pushes the estimation given by $\psi$ to be concentrated around $e$, right? Is the underlying idea to use it as some sort of regularization? 

* How limiting is the assumption _"we shift from using equivalence classes $[x]$ to sets of objects semantically similar to $x$ for estimating $\theta_x$"_. Would it be possible to find the equivalence classes in practice? Or this is a form that helps the theoretical aspect of the method only?

* Substituting the equivalence class by the $k$-Nearest Neighbors might rely strongly on the NT-Xent loss being satisfied. How important is the NT-Xent loss? I think an ablation of the different losses that compose the final loss would be of interest for this work, so readers understand the contributions and how to practically use the proposed approach.
  * Additionally, NT-Xent usually requires quite large batch sizes. What is the training setting in this case? How are the positive and negative pairs formed?
  * Is there a way to bound the boundary estimation error as a function of $k$?

* In the Appendix I read: _"An additional consistency loss is trained to ensure that the representations created by the contrastive routine and the reconstruction routine are similar and do not conflict each other."_
  * Can the authors elaborate on this? If there is an additional loss that helps improve the results, I would suggest to include it in the main body of the paper. This adds another weighing parameter to the overall loss, which reinforces the need of an ablation study in that sense.

* What do the up-down arrows from $\theta$ to $\hat{\theta}$ represent in Figure 3? I suggest replacing them with "MSE" or $L_3$. Additionally, writing "supervised learning" might mislead some reader. It is supervised, but with pseudo-labels, which is a much less strong condition of the method (no need for annotations). 
  * Additionally, I suggest the authors to highlight in Figure 3 where the proposed losses are applied. For example $L_1$ takes the left and right $X$s. Loss $L_3$ takes $\theta$ and $\hat{\theta}$, etc. 

* About results in Table 2 (Improving non-equivariant models with symmetry standardization). I notice that the improvement becomes smaller as the range of rotation is smaller. What is the intuition behind that effect? In that Table, how does the proposed method compare with a method that infuses equivariance without knowledge of distribution like IE-AE? I think this comparison is required, to understand whether the improvements shown are significative wrt. SOTA.

* How does the method behave for images inherently invariant to rotation (eg. a circle). This would be the case of the Flowers dataset for example, where many flowers are photographed top-down and appear centered and "almost" invariant to rotation.

* I suggest adding [Suau et al. ICML 2023](https://arxiv.org/abs/2306.16058) given some similarities in the spirit of the work. In such paper, equivariances are learnt from data, and the distribution over $g$ elements is naturally learnt.

----
**Overall comment:**

The paper is very well written, almost typo-less, and with excellent language. I enjoyed the reading, I thank the authors for that. The theoretical formulation is solid, and the proposed method to learn symmetry boundaries is novel to the best of my knowledge. However, I still think this paper could strongly benefit from experiments with data beyond MNIST, as well as a thorough ablation study, as mentioned before. I also suggest to add some explicit discussion about the strengths and limitations of the method.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a method that predicts the distribution of symmetries of each input in a dataset. Building on the invariant-equivariant autoencoder (Winter et al., 2022) and partial group equivariant CNNs (Romero & Lohit, 2022), the proposed method is designed to detect the symmetry boundary of image rotations. Experiments on various rotated MNIST datasets show that the proposed method is effective in predicting symmetry levels, detecting samples with out-of-distribution symmetries, and improving non-equivariant models with symmetry standardization.

### Strengths
-	This paper implements the novel idea of capturing levels of symmetry at a sample-level, in contrast to previous partial equivariant models that captures levels of symmetry at a dataset level.
-	The proposed method can learn meaningful and consistent canonical representations defined by the center of symmetry. This is an improvement over (Winter et al., 2022), where the canonical representation depends on various factors in training.
-	The experiment demonstrates the effectiveness of the approach of detecting input-dependent symmetries. Specifically, the proposed method is able to predict symmetry levels, detect out-of-distribution symmetries, and standardize datasets to improve non-equivariant models.

### Weaknesses
-	While the proposed method seems general enough to process any symmetry, the only setting discussed is learning symmetry boundary of SO(2). Moreover, the assumption that the rotation symmetry is uniformly distributed in an interval seems restrictive.
-	It is not clear whether or to what extend different levels of symmetry exists in datasets. In the abstract, the authors motivated the task of detecting the level of symmetry by the example that pictures of cars and planes exhibit different levels of rotation in CIFAR-10 dataset. However, this difference is not quantified. 
-	While experiments in the paper demonstrate that the proposed method works as intended, it is not clear how these tasks are relevant in real-world applications. The practical value of detecting symmetry levels could be made more convincing by discussing possible scenarios where predicting levels of symmetry gives useful information or is expected to improve downstream model performance.
-	Since the proposed method builds on invariant-equivariant autoencoder, comparing against it in experiments would help demonstrate the advantage of predicting input-dependent levels of symmetry, particularly for the out-of-distribution symmetry detection and improving models with standardization experiments. Currently there is no baseline comparison for any of the experiments. 
-	Minor issue: the legends in figure 6 could be larger.

### Questions
-	Suppose a symmetry is outside the symmetry boundary, i.e. does not present in the training set. Do we want the neural network to be equivariant to it? If so, it seems detecting symmetry at a sample-level is less useful in avoiding over constraining the model. If not, doesn’t this limit the model’s generalization ability?
-	In section 3.2, why does minimizing $d(\psi(x),e)$ encourage $\psi(c_{[x]})=e$?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a method to solve for the classification task where the dataset consists of varying degrees of rotation for each class. The method consists of two steps: a) breaking down the input into an invariant and an equivariant component. b) Applying Partial G-CNN in the IE-AE architecture to capture the partial symmetries c) Applying self-supervised objective over it.

### Strengths
The problem statement of capturing perfect+partial symmetries is well motivated.

### Weaknesses
- Method section writing could be simplified, by taking rotation detection as a running example, and relating it with the notations intermittently.  
- Clustering results on the output of the pseudo-labels and the final result from the contrastive learning approach would be beneficial in understanding the method.  
- The novelty in the method is lacking as it builds up on the IE-AE with existing Partial G-CNN layers, coupled with contrastive learning approach. All of which are already existing in the literature.  
- Motivation for applying contrastive learning is unclear, why can’t pseudo labels be directly used to predict the result? What is the benefit of applying contrastive learning?

### Questions
- First and third rows in Fig. 4 are same. What is the reason behind it? Is it the input?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work suggests a novel method for detecting symmetries in a given dataset.
It is done in several stages, in an usupervised manner, and is able to estimate the distribution of symmetries that is shared by each subset of items that form an equivalence-class, under the specific group action. This allows a cannonical normalization of the data, which can be desirable for further processing.
The method is based on a proposition that shows the equivalence of different properties of the group action estimator for an equivalence-class of items. Theory, analysis and experiments are under the setting of image data with uniform distributed rotation symmetries over a continuous segment of angles. The method is demonstrated on several variants of the MNIST dataset, which contained controlled syntetic ranges of rotations. Experiments focus on prediction of the per-item distribution parameter (size of range), but also show improvements obtained by both supervised and unsupervised classifiers after symmetry-standardizing the data, using the proposed method.

### Strengths
1] The choice to tackle a rather challenging combination of several main difficulties (that were previously only tackled in separation): (i) Being unsupervised ; (ii) Dealing with partial (versus full group action) symmetries (iii) Allowing per item level of symmetry. Each of these components of the setting are important requirements towards the ability to tackle real data in a general way, and it is a very important goal, in my opinion, to make progress in this difficult setting.

2] The method is backed up by a very formal framework, of partial symmetry groups acting on items of the dataset, in a way that connects to previous work and gives a general perspective on the problem, under which very specific choices and assumptions have been made. This formulation gives hope that the method could be extended to other, more general, related settings. 

3] The experiments show that the method can pretty well predict the desired symmetry parameters, at least for the rotation variants of the MNIST dataset.

### Weaknesses
1] While the decision to generalize the setting of previous works (especiall w.r.t. Partial-G-CNNs and IE-AE) to deal with partial, per-item distributions, in an unsupervised manner, the result is much more limited and less applicable as a result:
* The assumption of pure rotation, that is uniformly distributed over a single segment (practically a 1-parameter estimation problem) is very limited and unnatural. This can be compared to IE-AE that handle discrete and continuous groups, including rotations, translations and permutations.
* This limited setting is very obvious in the testing, which includes only uniformly rotated versions of MNIST. The above mentioned methods include demonstrating the potential on some slightly more realistic datasets (such as CIFAR and Shapenet), where there is no control or no way to model the true underlying symmetries. Especially when considering the potential of unsupervised learning, it is a little disappointing to see it demonstrated on controlled data that could allow fully supervised training.
* There is not much of a discussion as to how you could drop some of the assumptions, if desired. For example, the uniformity is currently what enables rather easily estimating the center of the range and its boundaries. What would happen with other types and distributions of symmetries?

2] The need for  (and the quality of) the self-supervision and the boundary prediction network $\Theta$ itself is unclear. What I mean here is that it is not the case that the network $\Theta$ can generalize to a new dataset, so for the given dataset, you still have to run the pre-training and then, given a new sample, you could just predict the pseudo-label itself. I imagine that it would be more precise, but perhaps less efficient because of the nearest neighbor computation. This should further explained.

3] The way of calculating nearest neighbors is not specified (or perhaps I missed it). This is critical for identifying the equivalence classes.

### Questions
1] How are nearest-neighbors ("semantically similar inputs") found?
2] Generating the pseudo-labels: Why is the estimator that simply takes the mean angle considered to be robust? and in particular "more robust to outliers" than what?
3] Experiments: Have you tried to demonstrate the method on more realistic data? (which would perhaps require dropping the uniform or continuous assumptions, or perhaps trying other more general classes of symmetry).

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
