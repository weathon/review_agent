# Differentiable Average Precision Loss in DETR

- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Average Precision (AP) and mean AP (mAP) remain the dominant metrics for evaluating object detectors, yet most training objectives optimize surrogates that are not well aligned with these ranking-based measures.
Most prior AP surrogates either rely on pairwise comparisons, incurring quadratic complexity, or only optimize classification, necessitating additional localization losses. We introduce differentiable average precision (DAP) loss, a smooth AP loss that directly optimizes a differentiable approximation to COCO-style mAP for one-to-one detectors.
Our key idea is to (i) replace non-differentiable sorting by modeling detection scores as continuous distributions and sweeping a series of thresholds to obtain a differentiable precision–recall curve, and (ii) use interpolation techniques to optimize localization task.
This yields a differentiable mAP approximation with linear time ($(O(N))$) in the number of predictions, enabling seamless integration with Hungarian matching.
We prove that, with respect to prediction scores, the gradient of DAP is sign-consistent—positive for positives and negative for negatives.
Empirically, fine-tuning pretrained DETR-family models with DAP for a small number of epochs delivers consistent COCO mAP gains without auxiliary losses or architectural changes.
DAP is simple to implement, computationally efficient, reduces hyperparameter tuning, and bridges the gap between training and evaluation for one-to-one detection. 
From-scratch training also delivers modest but positive improvements, albeit smaller than those obtained through fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors study the problem of training an object detector by using the performance measure (Average Precision -- AP) as the training objective. This is not the first time that this is attempted. Compared to what has been studied in the literature before, the authors consider mAP, the average of AP over different IoU thresholds. This sets the paper significantly apart from prior work since it includes both classification and localization aspects. mAP is not differentiable, similar to AP. Therefore, the authors introduce some approximations to obtain the proposed loss's derivatives. 

The loss function is used to train DETR-based detectors and evaluated on the COCO dataset. Some noticable improvements are reported over DETR, despite unpromising results on other detectors.

*Disclaimer*: I reviewed the same paper for ICLR 2025 and CVPR 2025. My review is updated based on the changes made by the authors.

### Strengths
+ It is worthwhile to use performance measures as training objectives for object detectors.
+ It is promising to adapt mAP as a loss function.
+ I acknowledge the novelty of the DAP loss formulation/approximation compared to existing ranking-based losses.
+ Compared to the previous versions, this time the authors bring forth the benefit of having linear time complexity. This is a big plus since existing methods suffer from quadratic complexity.

### Weaknesses
Weaknesses:

Although I am extremely fond of ranking-based training of object detectors, I strongly believe that the paper still has many issues, despite the revisions performed after ICLR 2025 and CVPR 2025 submissions:

1. A limitation of the proposed approach is that is can provide improvements only for finetuning and it performs subpar on training from scratch. The authors improved on this regard (both in terms of experiments and how they motivate their contributions). Although it is not ideal, I accept this as a limitation of a novel method/approach that can be addressed maybe in the future with another study.

2. "We prove that, with respect to prediction scores, the gradient of DAP is sign-consistent—positive for positives and negative for negatives." => Although this is highlighted as a significant concern, there is only a single paragraph in Section 3.4 on this.

3. Contrast with pairwise ranking surrogates: "Under SmoothAP Brown et al. (2020) with temperature τ = 0.01, we empirically observe ∂ SmoothAP/∂st < 0, i.e., the objective pushes down the score of a true positive (verified numerically)." => Since gradient descent multiplies the gradient with (-1), this should not have been an issue. The short and vague depiction in this paragraph is questionable.

4. Section 3.3: Please talk about alternative approaches to addressing the non-differentiability here.

5. There are too many typos or writing issues (please see Minor comments). I would have expected a more refined manuscript after so many revisions.


Minor comments:

- Abstract: "((O(N)))" => extra parantheses.
- "The Parameterized AP Loss Chenxin et al. (2021)" => If citations are not a part of the text, they should be enclosed within parantheses as "The Parameterized AP Loss (Chenxin et al., 2021)". There is a separate cite command in Latex for this.
- Please see the following guide for writing equations: https://wp.optics.arizona.edu/kupinski/wp-content/uploads/sites/91/2023/05/MerminEquations.pdf
- Fig 1 has subcaptions for the subfigures but not for the figure itself.
- "threshold; The value G" => "threshold. The value G".
- Eq 1: What is x?
- Eq 1: ∂Rα(x)/∂x => Given that Rα(x) is a univariate function, why don't you use dRα(x)/dx? That would also make the derivation in the equation easier to follow.
- Eq 1: I would add one more step/explanation on ∂Rα(x)/∂x relating it to ΔT. 
- Eq 4: I would replace H() with the indicator function to simplify the equations.
- "IoU,the optimization" => "IoU, the optimization".
- "According to Equation 5, L(·) is determined by 10 points" => Equation 4?
- "It is clear that within the range of IoU ∈ [0, 1], the gradient of function L(·) with respect to IoU either does not exist or is zero." => Please explicitly state that this is because of the H() function in Equation 5.
- "Figure 1b, We construct" => "Figure 1b, we construct".
- "continuous probabilistic distribution" => "continuous probability distribution".
- Eq 7: What is y? "fb(·) is the probability distribution for b" => But, what is the parameter here?
- "When fb(·) is represented as an impulse function δ(·), the equation above becomes equivalent to the original model." => Which equation? Please show how.
- "fb(·)are" => "fb(·) are".

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a differentiable average precision (DAP) loss designed for training object detectors to more directly maximise the mean Average Precision (mAP) metric used in evaluation. It avoids non-differentiable sorting (as used in the standard mAP metric calculation) by treating detection scores as continuous distributions, which enables the direct optimization of a smooth approximation of the precision-recall curve. Evaluated on COCO 2017, the proposed method shows consistent mAP improvements when fine-tuning DETR-style models (versus standard training losses), and also small gains even when training from scratch.

### Strengths
The overall goal of training object detectors specifically for the important evaluation metric of mean average precision is worthwhile, and the method proposed to do so is interesting, in particular the relaxation of sorting by replacing with a distributional interpretation of scores.

The proposed differentiable relaxation of mAP differs from existing relaxations in the literature, and (unlike existing methods) is specialised for DETR-style models with a one-to-one box matching stage. It separately considers both classification and localisation aspects of the mAP objective.

Results on COCO 2017 show a performance gain in terms of mAP compared with the same (DETR-style) models trained with standard losses. This performance gain is strongest in the case of fine-tuning for mAP following a standard pretraining phase; however there is still a gain even when training for mAP from scratch.

The method is somewhat flexible – results show it applied to several modern DETR-style transformer-based detection architectures including the original DETR, Deformable-DETR, RT-DETR, and Rank-DETR (which is perhaps the most similar in spirit in terms of how it is trained). The accuracy improvement due to the proposed method is fairly consistent across all architectures considered.

There are some experiments varying hyperparameters (e.g. batch size and matching cost function); these justify some design decisions.

Some limitations are explicitly discussed in Sec 5, notably the restriction to DETR-type architectures (with a matching stage), and the somewhat weaker performance of from-scratch training (though I do not consider this is a serious limitation in practice).

The paper is clear, well-structured, and pleasant to read.

### Weaknesses
The related work mis-represents some existing works in a way that makes the proposed method sound significantly more innovative than it is. In particular, Song 2016 does not in fact propose a differentiable approximation to mAP; instead it uses mAP itself in a loss-augmented setup similar to classic structured SVMs. Meanwhile, Henderson 2016 *does* in fact consider the localisation aspect of the mAP metric, not just classification, contrary to the statement at line 107. Moreover it also accounts for the full NMS procedure in the loss calculation.

There is no experimental comparison against well-established plug-and-play methods that are specifically designed for training for mAP, specifically Henderson 2016 and Song 2016. Those methods are architecture agnostic and as such can be applied to DETR. This is a very important baseline to include, since otherwise there is no evidence that the proposed approach to using mAP as loss is in fact better than these much older approaches. There is also no comparison to direct REINFORCE (or variance-reduced score-function gradient) methods that are applicable in this setting, and do not require significantly more technical machinery than the proposed approach.

Experimental results are only given on COCO 2017. It would be interesting to see how well the method works on other domains such as remote sensing, or on other establish natural image benchmarks such as OpenImages or even VOC 2012.

The training runs were apparently truncated at a fixed threshold of epochs; it is unclear what effect this has on the proposed method versus the baseline methods. Instead all models should be trained to validation-set convergence, to ensure there is no 'late learning' stage that can impact the ranking of different approaches.

There is no theoretical analysis / discussion of circumstances under which the proposed differentiable relaxation of mAP approximates the true global function, in particular in the minibatch setting. When used as a metric, mAP performs ranking over the entire dataset; this is significantly different to the training setting; it is unclear whether the proposed loss is an unbiased estimator of either non-differentiable minibatch mAP, non-differentiable full-dataset mAP, or differentiable full-dataset mAP.

### Questions
Most relevant issues are discussed under "Weaknesses" above. In particular…

Given that established, architecture-agnostic methods for optimizing mAP like Henderson 2016 and Song 2016 exist, as well as REINFORCE-type gradient estimators for the metric itself, please add these as baselines.

Consider adding at least one further dataset, preferably somewhat different in characteristics to COCO 2017. Also properly discuss the implications of stopping training before convergence, preferably showing the baselines do not improve after this point.

The proposed loss is a differentiable approximation of mAP calculated on minibatches, whereas the true mAP metric is a global ranking over the entire dataset. Can you provide some analysis showing the minibatch-based loss is a sound proxy (i.e. unbiased etc.) for the true, non-differentiable, full-dataset mAP?

The "ablation" experiments are not ablations in the proper sense (i.e. removing novel components of the model to show how important they are to performance); instead they are just varying certain hyperparameters. Perhaps this subsection should be renamed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a differentiable loss function to optimize  average precision (AP) as a training loss for transformer-based object detectors, DETRs in particular.  While prior work has explored using AP as the training objective, it typically focused on AP at a fixed IoU threshold (e.g., 0.5). In contrast, this paper targets COCO-style AP, which incorporates localization across multiple IoU thresholds. To this end, the authors propose differentiable approximations for both the localization and classification components of AP. For localization, they replace the step function in the IoU-vs-precision relationship (Figure 2) with a linear interpolation. For classification, they approximate the number of true positives within a score interval using the cumulative distribution function of a Gaussian. Experiments on COCO show that when the proposed loss is used to fine-tune the already trained model, performance slightly improves across several DETR variants.

### Strengths
Unifying training and evaluation objectives in object detection is an important problem. This paper addresses this problem by proposing a differentiable approximation of average precision (AP) to enable its direct use as a training loss for object detectors.

### Weaknesses
There are several issues with this paper:

First, an important baseline is missing. Since the paper approximates COCO-style AP (i.e., the average of AP across multiple IoU thresholds), it is natural to ask how existing AP-based losses such as AP Loss or Smooth-AP perform when optimized at different IoU thresholds. These comparisons are essential to contextualize the claimed improvements.

Second, the improvements shown in the main table (Table 1) are generally small (less than 1 AP points). How do we know whether the improvement is due to the DAP loss and not due to continued training of the base detector? Simply training the base detector further could improve the performance. 

Third, the proposed loss does not appear to perform well when training detectors from scratch. Prior works on AP-based losses (e.g., AP Loss, Smooth-AP) have demonstrated this capability. This remains a major empirical weakness.

Fourth, the paper lacks a direct comparison with Smooth-AP, which is conceptually very similar. The proposed probabilistic classification approximation closely resembles Smooth-AP. A theoretical and empirical discussion contrasting the method with prior differentiable AP formulations, such as AP Loss and aLRP Loss, is needed to clarify the contribution.

Finally, the approximation of the step-wise localization function L is not rigorously defined. Figure 2 suggests a linear interpolation, but the text does not specify the exact formulation. A precise mathematical definition should be provided.

Other minor issues: 
About the definition of AP given at line 150: If I am not mistaken, the PR curve is formed by applying a systematic thresholding to the confidence score of the object detector, not by changing IoU thresholds.  It is the COCO style AP (not the PR curve itself), which calculates the average of APs for different IoU thresholds. In fact, Equation 2 supports my view. 

AP is not a metric. Metric has a well-definition in mathematics. I think we can call AP a measure.

### Questions
- How do we know whether the improvement is due to the DAP loss and not due to continued training of the base detector with its original loss?
- Second row in table 1 has both Ori Loss and DAP loss checked. what does this mean? is this a typo?
- It is not clear how many epochs the fine-tuning is done. Does the epoch column in table 1 show that? If yes, what does it mean for the Ori Loss rows? Do you fine tune with Ori Loss with that many epochs?

### Soundness
2

### Presentation
2

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
This work proposes Differentiable Average Precision (DAP), a smooth, efficient loss that directly optimizes COCO-style mAP for one-to-one detectors like DETR and its variants, narrowing the gap between training objectives and evaluation metrics. DAP replaces non-differentiable sorting with continuous score distributions (Gaussian instantiation) and applies piecewise-linear interpolation for localization, achieving O(N) time without pairwise comparisons and integrating seamlessly with Hungarian matching. It also guarantees sign-consistent gradients (positive for positives, negative for negatives) under mild assumptions. Empirically, fine-tuning DETR-family models for a few epochs consistently improves COCO mAP without architectural changes or auxiliary losses.

### Strengths
1. The proposed DAP loss cleverly bridges the gap between the evaluation metric in detection task, and the original BCE/L1 loss. 
2. DAP achieves linear time complexity by eliminating pairwise comparisons and thus can integrate naturally with Hungarian matching in DETR.
3. The experiments are conducted in strong baselines, like Co-DETR, Rank-DETR.
4. Extensive results show its effectiveness.

### Weaknesses
1. The key difference between existing AP losses needs to be thoroughly discussed in the related work section.
2. The comparsion with related works is missing, e.g. Parameterized AP Loss (Tao et al, NeurIPS 2022).
3. As the author say in Sec 5, DAP loss is designed for one-to-one matching detector. However, H-DETR (DETRs with Hybrid Matching, CVPR 2023) can do one-to-many matching, still use bipartite matching. Adding the results on H-DETR would extend the scope of this manuscript.

Minor
1. Missing caption in Figure 1. A global caption is missing, only two sub-caption.
2. Each formula should have a comma or full stop at the end.

### Questions
1. Why the post-training epoch varies for different models in Tab. 1?
2. Can the proposed method extend to non-DETR detectors?

### Soundness
3

### Presentation
2

### Contribution
3
