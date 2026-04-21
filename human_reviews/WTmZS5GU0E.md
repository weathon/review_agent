# Leveraging Semantic and Positional Uncertainty for Trajectory Prediction

- Avg Score: 5.00
- Decision: Reject
- Scores: 3, 6, 5, 6

## Abstract
Given a time horizon with historical movement data and environmental context, trajectory prediction aims to forecast the future motion of dynamic entities, such as vehicles and pedestrians. A key challenge in this task arises from the dynamic and noisy nature of real-time maps. This noise primarily stems from two resources: (1) positional errors due to sensor inaccuracies or environmental occlusions, and (2) cognitive errors resulting from incorrect scene understanding. 
In an attempt to solve this problem, we propose a new framework that estimates two kinds of uncertainty, \ie, positional uncertainty and semantic uncertainty simultaneously, and explicitly incorporates both uncertainties into the trajectory prediction process. 
In particular, we introduce a dual-head structure to independently perform semantic prediction twice and positional prediction twice, and further extract the prediction variance as the uncertainty indicator in an end-to-end manner. The uncertainty is then directly concatenated with the semantic and positional predictions to enhance the trajectory estimation.
To validate the effectiveness of our uncertainty-aware approach, we evaluate it on the real-world driving dataset, \ie, nuScenes. 
Extensive experiments on 3 mapping estimation and 2 trajectory approaches show that the proposed method (1) effectively captures map noise through both positional and semantic uncertainties, and (2) seamlessly integrates and enhances existing trajectory prediction methods on multiple evaluation metrics, \ie, minADE, minFDE, and MR.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies an important problem of trajectory prediction, which considers both position and semantic uncertainties. The authors propose a dual-head structure to model the two uncertainties explicitly. Experiments show the effectiveness of the proposed method to some extent.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed method considers both positional and semantic uncertainties.
3. Experiments show the effectiveness of the proposed method.

### Weaknesses
1. Although the proposed method studies the positional and semantic uncertainties explicitly, the corresponding uncertainty modeling seems to be very simple and lacks enough technical contribution. It would be better to provide more details to show the novelty and technical contributions of this paper. 
2. It is not enough to show the effectiveness of the proposed method with only one dataset. It is encouraged to include more datasets, for example, NGSIM [1] and highD [2].

[1]. Convolutional social pooling for vehicle trajectory prediction, CVPR 2018.

[2]. The highD dataset: A drone dataset of naturalistic vehicle trajectories on german highways for validation of highly automated driving systems, ITSC 2018.

3. It is suggested to include more prediction methods [3, 4, 5] in Table 1, which could be more comprehensive to show the effectiveness of the proposed method.

[3]. Leapfrog diffusion model for stochastic trajectory prediction, CVPR 2023.

[4]. Wsip: Wave superposition inspired pooling for dynamic interactions-aware trajectory prediction, AAAI 2023.

[5]. Convolutional social pooling for vehicle trajectory prediction, CVPR 2018.

4. It would be better to provide a case study to intuitively show the model can learn the uncertainties.

### Questions
1. What is the technical contribution of the proposed method? Uncertainty modeling seems to be very simple.
2. Why does not include more datasets and baselines?

### Soundness
2

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
4

### Summary
The manuscript identifies two types of uncertainties in trajectory prediction tasks: positional uncertainty and semantic uncertainty. It then proposes a framework to estimate these uncertainties and incorporates them into the trajectory prediction process in an end-to-end manner. The authors validate the effectiveness of the proposed framework on the nuScenes dataset using multiple existing trajectory prediction methods.

### Strengths
1. The motivation of the manuscript is valid. HD maps are crucial for motion prediction in autonomous driving scenarios, but estimated HD maps often contain noise and errors, introducing uncertainty into motion prediction systems. The authors recognize this issue and attempt to mitigate its impact, which benefits the community.

2. The proposed framework easily generalizes to various HD map generation methods as well as alternative motion prediction methods.

3. The experimental results and ablation studies are promising.

4. The manuscript is well-written. This reviewer particularly appreciates the discussion section, which makes the methodology easier to follow.

### Weaknesses
1. Testing on only one dataset may raise concerns about the framework's generalizability to more complex scenes.

2. More theoretical support may be needed for using an auxiliary head for uncertainty estimation, and the uncertainty visualization could be made easier to interpret.

### Questions
1. What are res5c and res4f in line 94.
2. How did author generate Figure 1 (b).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The draft proposes and evaluate a new way of estimating online_HD-map uncertainties, and report a higher improvement on downstream trajectory prediction than current SoA on the topic (Gu et al. 2024).

### Strengths
The main interest of the paper is to pinpoint and evaluate the potential interest of using not only positional uncertainty but also "semantic" uncertainty (i.e. potential class error) of online-estimated HD-map, for improving robustness of trajectory prediction to HD-map errors.
Furthermore, according to the experiments on NuScene presented by authors, their approach can reduce the minADE and minFDE of downstream trajectory prediction by an extra ~5% compared to the recently published approach of (Gu et al. 2024).

### Weaknesses
There are some significant weaknesses in the draft:
  - first, contrary to what authors write, (Gu et al. 2024) do take into account also classification uncertainty (cf page 15 of that reference) ==> authors should be more rigourous on their comparison of the differences between the 2 methods
  - secondly, the way authors evaluate the positional uncertainty (lines 216-230 of the draft), i.e. by just considering 2 estimates \mu and \mu' (the second one after adding a dropout layer), seems unusual and less robust than an actual parametric estimation (such as the Laplace distribution hypothesis used by Gu et al. 2024)
 - third, authors evaluate improvements brought by their method only on variants of MapTR, but not on StreamMapNet 
 - finally, the way authors present their improvement in their table 1 and text, as % reduction compared to baselines *without* (Gu et al 2024) is somewhat misleading: compared to the latter their results are only approximately -5%, rather than the highlighted -10%

### Questions
- on line 94 of draft, what are "res5c" and "res4f" referring to ??
- why authors do not report the improvement brought by their approach if applied to StreamMapNet ?
- given the unusual way that authors use to estimate uncertainty, some analysis of its outcome (as in section 5.2 of Gu et al. 2024) would be more than welcome

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This article introduces a novel framework for trajectory prediction that addresses the challenges posed by real-time map noise resulting from sensor inaccuracies, misinterpretations, and other factors. The proposed method estimates both positional and semantic uncertainties, integrating them into the prediction process through a dual-head structure that performs independent predictions and extracts prediction variance as an indicator of uncertainty. Validated on the nuScenes dataset, this approach effectively captures map noise and enhances existing trajectory prediction methods.

### Strengths
1. The writing in this article is excellent, featuring clear logic and organization. Additionally, the figures and charts adhere to established standards, which is commendable.

2. This paper addresses two types of uncertainties: positional and semantic. It aims to mitigate the adverse effects of noise in High-Definition (HD) map estimation for trajectory prediction.

3. By integrating map elements that include both positional and semantic uncertainties into downstream models, this approach leverages the context of uncertainty to improve the accuracy of trajectory predictions.

4. The method can be integrated with existing mapping estimation and trajectory prediction approaches, consistently enhancing prediction accuracy. This is evidenced by significant improvements in minADE, minFDE, and MR on the nuScenes dataset when combined with specific backbone models.

### Weaknesses
I have the following concerns:

1. Although the authors have conducted numerous experiments and analyses, which is commendable, there is still an opportunity to expand the experiments further to enhance the credibility of the work. For instance, the authors could consider referencing some of the latest research in online mapping or trajectory prediction, such as:

    [1] MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping

    [2] MGMap: Mask-Guided Learning for Online Vectorized HD Map Construction

    [3] MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation

    [4] Producing and Leveraging Online Map Uncertainty in Trajectory Prediction

    [5] Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention

2. In future work, it is suggested that the authors attempt to explore research based on scene topology reasoning, such as:
    [1] TopoNet: A New Baseline for Scene Topology Reasoning

### Questions
As stated in the Weakness.

### Soundness
3

### Presentation
4

### Contribution
3
