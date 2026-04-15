# Tailoring Mixup to Data using Kernel Warping functions

- Decision: Reject
- Scores: 6, 5, 8, 3

## Abstract
Data augmentation is an essential building block for learning efficient deep learning models. Among all augmentation techniques proposed so far, linear interpolation of training data points, also called *mixup*, has found to be effective for a large panel of applications. While the majority of works have focused on selecting the right points to mix, or applying complex non-linear interpolation, we are interested in mixing similar points more frequently and strongly than less similar ones. To this end, we propose to dynamically change the underlying distribution of interpolation coefficients through warping functions, depending on the similarity between data points to combine. We define an efficient and flexible framework to do so without losing in diversity. We provide extensive experiments for classification and regression tasks, showing that our proposed method improves both performance and calibration of models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a Kernel Warping Mixup technique toward a flexible distribution adjustment by considering the sample similarity. The proposed method changes the underlying distributions used for sampling interpolation coefficients by defining warping functions, allowing inputs and labels to be disentangled, and providing a framework that encompasses several variants of Mixup. To this end, the author introduces a similarity kernel that considers the distance between points when selecting a parameter for the warping function. This paper demonstrates the applicability of the Kernel Warping Mixup across classification and regression tasks, and cites improvements in performance, calibration, and efficiency while it requires less computation, additionally.

### Strengths
+ The paper provides a comprehensive analysis and comparison of related work in the field of data augmentation. The organization of the related work into different topics provides all-sided evaluations of the contributions.

+ The proposed idea is somehow novel since few existing works link the coefficients of interpolation with data geometry. 

+ The proposed method has been proven effective in improving the model calibration while the improvement of in-distribution generalization is not very significant.

### Weaknesses
- Compared to RegMixup, the proposed method brings marginal improvement in classification task accuracies on different models (Tables 2 and 3). Plus, the benchmark result of the Manifold Mixup in classification is missing. The concern is on the expressivity of the proposed method since general data augmentation aims to enrich the diversity of data and cover more uncertain data points as much as possible instead of encouraging similar data to be mixed more.

- The warped Gaussian Kernel into the original Beta distribution for the mixup coefficient $\lambda$ highly depends on batched data similarity. If the batch size is small (for large-scale tasks), the relative distances among the data will collapse, i.e. the same samples will be mixup with different $\lambda$ whether an outlier appears. In Eqn (6), if $n$ is small, the total distance is sensitive to the largest sample distance. It weakens the robustness of the proposed method.

- The evaluation and explanation of how the proposed method enhances calibration over vanilla Mixup are insufficient. For example, (Thulasidasan et al., 2019) have concluded that label smoothing in Mixup training significantly contributes to improved calibration, supported by comprehensive observations such as "score of winning class along the time" and "overconfidence". C-Mixup (Yao et al., 2022a) provides the theoretical justification for Theorem 3. To enhance the comparison, it would be better to establish a more meaningful connection between the proposed method and calibration, moving beyond only the presentation of ECE and Brier scores. 

=====================Typos==============

- Missing reference to the initial mention of "MIT" in Section 2.2.
- In Eqn(6), the LHS should indicate the dependence of $n$ if the measure on RHS depends on all $n$ samples.

### Questions
1. Since general data augmentation aims to enrich the diversity of data and cover more uncertain data points, is adjusting the distribution of coefficients based on data distribution reasonable to enhance the expressivity? Please see weakness 1.

2. Why choose different benchmarks between Tables 3 and 4, since Manifold Mixup can also applied to classification problems?

3. As suggested by (Wang et al., 2023), Mixup fails to improve the calibration in terms of other metrics such as Calibrated ECE and Optimal ECE. Can the author explain more or provide more evidence that the method in this paper does help the calibration? Why does altering the underlying distribution of $\lambda$ yield a better calibration over vanilla Mixup, even with the same similarity measure of input and output (Row 2 in Table 1)?

I am keenly awaiting the author's response to see how they address my concerns, and I am open to increasing my score based on the response.

### Soundness
3 good

### Presentation
4 excellent

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
This paper aims to improve the mixup algorithm by improving its data interpolation policy. Particularly, the paper proposes Kernel Warping Mixup that can dynamically change the sampling distribution of interpolation coefficient $\lambda$, so that when mixing data ponts that are "closer" under a certain metrics, the choice of $\lambda$ can have a higher degree of freedom; when mixing data points that are "farther", $\lambda$ should be chosen closer to 0 or 1. Experiments on both classification tasks and regression tasks are conducted, and the results show that the proposed Kernel Warping Mixup improves both the test accuracy and the calibration compared with conventional Mixup.

### Strengths
1. Proposed a variant mixup algorithm that has shown improvement on both generalization and calibration compared to conventional mixup.

2. Adequate experiments on common datasets, both of classifrication tasks and regression tasks.

3. The thought process of designing the proposed algorithm is explained clearly.

### Weaknesses
1. The idea of dynamically controlling the sampling of $\lambda$ in mixup is not novel, and the idea of controlling the pairing of the data examples based on distributional similarities (like k-mixup) is also well-investigated. As a result, combining these two types of ideas to formulate a new algorithm, like the one proposed in this paper, is also intuitively and empirically straightforward, and doesn't seem to be much surprising or interesting. In my opinion it the amount of contributions in this work is not sufficient to be a full paper in ICLR.

2. As a experiment-based work, the datasets used in the experiments are not adequate. For example, in classifications the authors only investigated their proposed algorithm on image datasets. In fact, some datasets, especially the ones with lower dimensionalities, tend to benefit less or even negatively from mixup, and also L2 distance between data points in these datasets may be more statistically meaningful. Such datasets are also worth of experiment verifications of the Kernel Warping Mixup. This also applies to the regression tasks.

3. The essential hyperparameters $\tau_{max}$ and $\tau_{std}$ used are chosen through cross-validation to find the optimal values. This may cost extra time before the real training is even started.

### Questions
1. In the last paragraph of Section 1, "... improve both performance and ...", what performance exactly? Is it refering to generalization performance?

2. Section 3.3. Why $\tau$ should be exponentially correlated with the distance?

3. Is there any principle or strategy to select the metrics of similarity? Like L2 or optimal transports or else?

4. How is the cross-validation used to find the hyperparameters conducted in details? Is it conducted before the real training? Or is it conducted simultaneously during the training? What objective is considered in finding the "optimal" $\tau$'s?

5. Algorithm 1. It seems that the values of $\tau$'s for the inputs and the targets are computed in the identical way. Then what is the point of defining them separately? And also, if the $\tau^o$ is computed separately, how would one define the similarity between two one-hot labels in classifications? Is it going to be like a simple equity indicator function?

6. Section 4.2. Why is the input distance or embedding distance not taken into consideration here? For some regression problem, I believe the similarity between two targets doesn't necessarily indicate a comparable similarity between their inputs.

### Soundness
2 fair

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper generally contributes a new way to data augmentation by change distributions of training data.

Here is a general summary:
1. Authors define warping functions to change the underlying distributions used for sampling
interpolation coefficients. This defines a general framework that allows to disentangle inputs
and labels when mixing, and spans several variants of mixup.
2. Authors proposed to then apply a similarity kernel that takes into account the distance between
points to select a parameter for the warping function tailored to each pair of points to mix,
governing its shape and strength. This tailored function warps the interpolation coefficients
to make them stronger for similar points and weaker otherwise.
3. Authors show that our Kernel Warping Mixup is general enough to be applied in classification as
well as regression tasks.

This major contribution of this paper is mixing the idea of regularization and kernel function and try to 
solve the problem from data perspective. If experiments result is convincible, this is a new way to consider 
data augmentation.

### Strengths
The ideology of this paper is quite plausible and is theoretically robust.

### Weaknesses
I don't think regression task can be convincing as downstream task for novelty. Regression task, in somehow, can already achieve very good result. I think a challenging downstream task such as object detection can make this paper more attractive.

### Questions
How do you define/calculate distance?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Mixup data augmentations are a widely used technique for deep learning, and most methods are focused on selecting the right points to mix or designing favorite mixing strategies. This paper tries to improve mixup by mixing similar data points more frequently than less similar ones and proposes a dynamically changing underlying distribution of mixing interpolation coefficients through warping functions, dubbed Kernel Warping Mixup. Extensive experiments for classification and regression tasks demonstrate the effectiveness and generalization abilities of the proposed mixup.

### Strengths
* (S1) It is an interesting and novel perspective of improving mixup augmentations by mixing more similar samples than less similar ones (but it also needs verification and empirical analysis to ensure the importance, as mentioned in W1). Experiment results show the effectiveness of the proposed method in comparison to classical mixup variants on both classification and regression tasks.

* (S2) Various analyzing metrics are used to verify the effectiveness of mixup classification and regression tasks, e.g., ECE, UCE, and NLL, which are not well studied in previous works.

* (S3) The overall writing is fruentcent and easy to follow. The implementation details and source code are available.

### Weaknesses
* (W1) Is the studied problem really important to mixup augmentations, i.e., it is practically useful to conduct mixup interpolation with similar samples? I cannot find any empirical analysis demonstrating that previous mixup methods will encounter serious drawbacks (e.g., performances, calibration abilities, generalization abilities to tasks) because of not mixing similar samples well. For example, the authors should provide some visualizations of effects or statistics to demonstrate the problem in addition to Figure 1.

* (W2) Weak experiments. In comparison to recently published works on mixup augmentation, this paper lacks comprehensive and solid comparison experiments to verify the effectiveness from three aspects. (a) The compared baselines are restricted to classical methods in classification tasks, and the performance gains are limited in all comparison results. These results make me doubt the importance of the studied problem in this paper, i.e., mixing more similar mixup samples than less similar ones. (b) The experiments are small-scale (e.g., CIFAR-10/100) with classical network architectures (ResNet variants) and old-fashioned baselines. There are many open-source mixup methods and benchmarks for classification and regression tasks [1, 2], and I suggest the authors consider more practical and modern experiment settings (e.g., large-scale experiments on ImageNet, modern Transformer backbones on CIFAR-100). (c) Since the proposed method is orthogonal to these mixup methods that improve mixup policies of samples (e.g., CutMix variants [3, 4], or randomly combining Mixup and CutMix [5]) or labels (e.g., TransMix [6], Decouple Mixup [7], and MixupE [8]), the authors should verify whether the proposed method can improve these existing mixup algorithms, rather than only test upon the vanilla Input Mixup.

* (W3) Hyper-parameter sensitivity. As shown in Appendix D and E (Table 6), the hyper-parameters of the proposed method vary significantly on different datasets and tasks. The ablation and sensitivity analysis of these hyper-parameters should be added. Meanwhile, I wonder how the authors determine the two hyper-parameters, which should be detailed in the appendix.

* (W4) The related work section is not well presented. I suggest the author combine Sec. 2.2 and Sec. 3.1 as a new section called Preliminary. Meanwhile, the authors may include more recently published mixup algorithms in different categories.

### Reference
[1] OpenMixup: A Comprehensive Mixup Benchmark for Visual Classification. arXiv, 2022.

[2] C-Mixup: Improving Generalization in Regression. NeurIPS, 2022.

[3] CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. ICCV, 2019.

[4] PuzzleMix: Exploiting Saliency and Local Statistics for Optimal Mixup. ICML, 2020.

[5] Training data-efficient image transformers & distillation through attention. ICML, 2021.

[6] TransMix: Attend to Mix for Vision Transformers. CVPR, 2022.

[7] Harnessing Hard Mixed Samples with Decoupled Regularizer. NeurIPS, 2023.

[8] MixupE: Understanding and Improving Mixup from Directional Derivative Perspective. UAI, 2023.

### Questions
Please refer to the weaknesses I mentioned.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
